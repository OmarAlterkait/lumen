"""The coverage harness: every sampled part's proposal at the reference covers its live density.

A draw is made at the reference values and weighted by live / proposal. That estimates the live expectation only where
the proposal is positive wherever the live density can be: a live density that is positive where the proposal is 0 is
light the estimator never draws (biased, silently). For a keyed parameter, any value in its range can be the
reference and any other the live value, so the check runs over grids that include the range's edges.

A ``Part`` is one sampled part: its keyed parameters with their range grids, the other parameters' values, its live
density over the part's test set (a categorical part's outcomes, a lobe's directions, a spectrum's energies) and the
density it is drawn from at the reference with one parameter keyed (only that one: a row whose other parameters are
keyed too may be covered by theirs). ``gaps(part)`` returns every (parameter, reference
value, live value, test point) where the live density is > 0 and the proposal is 0. The proposal is the sampler's own
(the code that draws), so a declaration that covers a part (the parameters that can zero a
probability or move a support, and their covering density: the defensive mixture, ``lumen.transport.estimator``)
shows up as the gap closing: ``surface_part`` reads the declaration (``SurfaceType.role``), ``glisur_part`` the facet
lobe the code draws from (``facets.lobe_name``), ``mie_part`` the lobe's cover (``Lobe.cover``), the scintillation
components and spectra their covered draws.

The parts are built from lumen's own definitions (``parts()``): the surface types' outcomes (``TYPES`` and the derived
types), the facet lobes, the bulk lobes, the scintillation components' choice and the spectra's draw (scintillation
and WLS). Readout draws (time spreads, charges) are made at the live values directly (no reference, no weight): no
coverage question, so no part.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from lumen.media import phase, spectra
from lumen.rules.choices import DEFENSIVE_SHARE
from lumen.rules.keys import PARAMS, SURFACE_KEYS
from lumen.sources import scintillation
from lumen.surfaces import facets
from lumen.surfaces import types as st
from lumen.surfaces.outcomes import NAMES
from lumen.transport import estimator

F32 = jnp.float32
CONT = np.ones(len(NAMES), bool)  # every outcome continues (detection and absorption are shares, not outcomes)


@dataclass(frozen=True)
class Part:
    """One sampled part (module docstring)."""

    name: str
    grids: dict  # keyed parameter -> values over its range (its edges included)
    base: dict  # every parameter's value where it is not the one varied
    live: Callable  # live(values: dict) -> (X,) density >= 0 over the part's test set
    proposal: Callable  # proposal(reference: dict, keyed: str) -> (X,) the density drawn from
    labels: Callable = field(default=lambda i: str(i))  # a test point's name in a report


@dataclass(frozen=True)
class Gap:
    """A test point the live density reaches and the proposal does not."""

    part: str
    keyed: str
    reference: float
    live: float
    point: str

    def __str__(self):
        """Show the gap on one line."""
        return f"{self.part}: {self.keyed} ref {self.reference:g} -> live {self.live:g}: {self.point} not drawn"


def gaps(part: Part) -> list[Gap]:
    """Return every gap of a part: for each keyed parameter, each reference and live value on its grid (the others at
    their base values), the test points where live > 0 and the proposal is 0."""
    out, seen = [], set()
    for k, grid in part.grids.items():
        for r in grid:
            q = np.asarray(part.proposal({**part.base, k: r}, k), np.float64)
            for v in grid:
                p = np.asarray(part.live({**part.base, k: v}), np.float64)
                for i in np.flatnonzero((p > 0.0) & ~(q > 0.0)):
                    key = (k, part.labels(int(i)))
                    if key not in seen:  # one report per parameter and point (the first values that show it)
                        seen.add(key)
                        out.append(Gap(part.name, k, float(r), float(v), part.labels(int(i))))
    return out


# ------------------------------------------------------------------------------------------------- the ranges
def grid(lo, hi, inner=(0.3,)):
    """Return a grid over [lo, hi]: both edges (a finite stand-in for an infinite one) and interior points."""
    hi_ = 1e3 if math.isinf(hi) else hi
    return tuple(sorted({float(lo), float(hi_), *(lo + f * (hi_ - lo) for f in inner)}))


def column_range(name):
    """Return the declared range of a surface type's column (the Geant4 key's, ``lumen.rules.keys.PARAMS``)."""
    key = {"nr": "REALRINDEX", "nk": "IMAGINARYRINDEX", "f_spec": "SPECULARSPIKECONSTANT"}.get(name)
    key = key or SURFACE_KEYS.get(name, name)
    return PARAMS[key].range


# ------------------------------------------------------------------------------------------------ surface outcomes
def _contexts():
    """Return crossing contexts: incidence cosines x index pairs (into a denser medium, out of one: TIR, matched)."""
    out = []
    for c in (1.0, 0.5, 0.05):
        s = math.sqrt(1.0 - c * c)
        for n1, n2 in ((1.33, 1.0), (1.0, 1.33), (1.5, 1.5)):
            out.append(dict(lam=F32(400.0), wi=jnp.asarray([s, 0.0, -c], F32), n=jnp.asarray([0.0, 0.0, 1.0], F32),
                            n1=F32(n1), n2=F32(n2)))  # fmt: skip
    return out


CTX = _contexts()


def surface_part(stype) -> Part:
    """Return the outcome choice of a surface type: test points (context, outcome); proposal the defensive mixture
    (``estimator.defensive``: where a keyed parameter has a role, ``SurfaceType.role``, the row's outcomes of reference
    probability 0 are proposed)."""
    grids = {k: grid(*column_range(k)) for k in stype.params}
    base = {k: float(np.clip(0.3, *column_range(k))) for k in stype.params}
    cont = [o for o in range(len(NAMES)) if CONT[o]]
    sup = np.zeros(len(NAMES), bool)
    sup[list(stype.outcomes)] = True

    def probs(values):
        p = {k: F32(v) for k, v in values.items()}
        return np.stack([np.asarray(stype.evaluate(p, c).probs, np.float64)[cont] for c in CTX])

    def live(values):
        return probs(values).ravel()

    def proposal(ref, keyed):
        p = probs(ref)
        if (
            stype.role(keyed) is not None
        ):  # the row's floor: every supported continuing outcome of probability 0 proposed
            p = np.where(sup[cont][None, :] & (p <= 0.0), 1.0, p)
        return p.ravel()

    def label(i):
        c, o = divmod(i, len(cont))
        return f"{NAMES[cont[o]]} (context {c})"

    return Part(f"surface:{stype.name}", grids, base, live, proposal, label)


def surface_types():
    """Return the surface types a detector's rows can have: ``TYPES`` and the derived types."""
    out = list(st.TYPES) + [st.PAINTED, st.REFLECTOR, st.SPECULAR_METAL]
    out += [st.transmitting(t) for t in (st.UNIFIED_METAL, st.UNIFIED_METAL_NORMAL, st.GLISUR_METAL, st.PHOTOCATHODE)]
    out += [st.into_opaque(st.PAINTED_T)]
    return out


# ------------------------------------------------------------------------------------------------------ facet lobes
def _directions(n=24):
    """Return outgoing directions over the upper hemisphere (a polar x azimuth grid)."""
    th = (np.arange(n) + 0.5) / n * (math.pi / 2)
    ph = np.arange(2 * n) / (2 * n) * 2 * math.pi
    t, p = np.meshgrid(th, ph, indexing="ij")
    return jnp.asarray(np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1).reshape(-1, 3), F32)


WO = _directions()
WI = [jnp.asarray([math.sqrt(1 - c * c), 0.0, -c], F32) for c in (1.0, 0.6, 0.2)]
UP = jnp.asarray([0.0, 0.0, 1.0], F32)


def _facet_density(logpdf, width):
    """Return 1 where one facet reflection's density is positive (its log density is not ``facets.LOG_TINY``, the
    value the code gives off the support), else 0: the support in exact arithmetic, not a float32 exp that
    underflows."""
    return np.concatenate([np.asarray(logpdf(F32(width), wi, UP, WO) != F32(facets.LOG_TINY), np.float64)
                           for wi in WI])  # fmt: skip


def glisur_part() -> Part:
    """Return GLISUR's facet reflection (one facet) in polish: drawn from the lobe at the reference radius 1 - polish,
    or, where polish is keyed (role "support"), from the covered lobe (``facets.lobe_name``)."""
    lo, hi = PARAMS["polish"].range
    prop = facets.FACET_LOBES[facets.lobe_name(st.GLISUR_METAL)]

    def live(v):
        return _facet_density(facets.glisur_facet_logpdf, 1.0 - v["polish"])

    def proposal(ref, keyed):
        lp = prop.facet_proposal_logpdf or prop.facet_logpdf
        return _facet_density(lambda r, wi, n, wo: lp({"sigma_alpha": r}, wi, n, wo), 1.0 - ref["polish"])

    return Part("facet:glisur", {"polish": grid(lo, hi, (0.5,))}, {"polish": 0.5}, live, proposal,
                lambda i: f"direction {i % WO.shape[0]} (incidence {i // WO.shape[0]})")  # fmt: skip


def unified_part() -> Part:
    """Return the unified facet reflection (one facet) in sigma_alpha, drawn at the reference width."""
    lo, hi = PARAMS["sigma_alpha"].range
    tab = facets.table_operands()

    def dens(s):
        return _facet_density(lambda r, wi, n, wo: facets.unified_facet_logpdf(r, wi, n, wo, tab), s)

    return Part("facet:unified", {"sigma_alpha": grid(lo, hi)}, {"sigma_alpha": 0.2}, lambda v: dens(v["sigma_alpha"]),
                lambda ref, k: dens(ref["sigma_alpha"]),
                lambda i: f"direction {i % WO.shape[0]} (incidence {i // WO.shape[0]})")  # fmt: skip


# ------------------------------------------------------------------------------------------------------- bulk lobes
def _sphere(n=24):
    th = (np.arange(n) + 0.5) / n * math.pi
    ph = np.arange(2 * n) / (2 * n) * 2 * math.pi
    t, p = np.meshgrid(th, ph, indexing="ij")
    return jnp.asarray(np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1).reshape(-1, 3), F32)


SPHERE = _sphere()


def mie_part() -> Part:
    """Return Mie's direction (both HG lobes, the lobe choice integrated out): drawn at the reference parameters, the
    lobe choice from the defensive mixture where a keyed parameter has a role (``Lobe.defensive``, ``Lobe.cover``)."""
    g_lo, g_hi = PARAMS["MIEHG_FORWARD"].range
    grids = {"forward_ratio": grid(0.0, 1.0), "g_forward": grid(g_lo, g_hi), "g_backward": grid(g_lo, g_hi)}
    base = {"forward_ratio": 0.7, "g_forward": 0.8, "g_backward": 0.4}
    wi = jnp.asarray([0.0, 0.0, 1.0], F32)

    def dens(v):
        p = {k: F32(x) for k, x in v.items()}
        return np.asarray(jnp.exp(phase.mie_hg_logpdf(p, wi, SPHERE)), np.float64)

    def proposal(ref, keyed):
        lobe = phase.MIE_HG
        return dens(lobe.cover(ref, DEFENSIVE_SHARE) if keyed in lobe.defensive else ref)

    return Part("lobe:mie_hg", grids, base, dens, proposal, lambda i: f"direction {i}")


def rayleigh_part() -> Part:
    """Return Rayleigh's direction: no parameter (its density is positive everywhere)."""
    wi = jnp.asarray([0.0, 0.0, 1.0], F32)
    d = lambda v: np.asarray(jnp.exp(phase.RAYLEIGH.logpdf({}, wi, None, SPHERE)), np.float64)  # noqa: E731
    return Part("lobe:rayleigh", {"none": (0.0,)}, {}, d, lambda ref, k: d(ref), lambda i: f"direction {i}")


# ----------------------------------------------------------------------------------- scintillation and spectra
def components_part(n=3) -> Part:
    """Return the scintillation component choice: drawn by the reference shares under the defensive mixture
    (``scintillation.flash``), its keyed parameters the yields SCINTILLATIONYIELDi (range [0, inf))."""
    keys = scintillation.KEYS["shares"][:n]
    grids = {k: grid(*PARAMS["SCINTILLATIONYIELD"].range) for k in keys}
    base = dict.fromkeys(keys, 1.0)

    def shares(v):
        return np.asarray(scintillation.shares(lambda k, d: F32(v.get(k, d)), tuple(range(n))), np.float64)

    def proposal(ref, keyed):
        s = jnp.asarray(shares(ref), F32)
        return np.asarray(estimator.defensive_q(s, jnp.ones(n, bool)), np.float64)

    return Part(f"scintillation:components{n}", grids, base, shares, proposal,
                lambda i: f"component {i + 1}")  # fmt: skip


def spectrum_part(name, energy) -> Part:
    """Return a spectrum's draw (a scintillation component's or a WLS emission's: Geant4's sampler of the table,
    ``spectra.g4_arrays``) in its knot values (range [0, inf), each one a keyed parameter); test points the knot
    intervals' midpoints."""
    e = np.asarray(energy, np.float64)
    mid = jnp.asarray(0.5 * (e[1:] + e[:-1]), F32)
    keys = [f"{name}[{i}]" for i in range(e.size)]
    grids = {k: grid(*PARAMS["WLSCOMPONENT"].range) for k in keys}
    base = {k: float(i > 1) for i, k in enumerate(keys)}  # its first interval without mass (a WLSCOMPONENT's ends)

    def dens(v):
        s = jnp.asarray([v[k] for k in keys], F32)
        return np.asarray(spectra.density(jnp.asarray(e, F32), s, mid), np.float64)

    def proposal(ref, keyed):
        s = jnp.asarray([ref[k] for k in keys], F32)
        return np.asarray(spectra.covered_density(jnp.asarray(e, F32), s, mid, DEFENSIVE_SHARE), np.float64)

    return Part(f"spectrum:{name}", grids, base, dens, proposal,
                lambda i: f"interval {i} [{e[i]:g}, {e[i + 1]:g}] eV")  # fmt: skip


def parts() -> list[Part]:
    """Return every sampled part lumen has (module docstring)."""
    out = [surface_part(t) for t in surface_types()]
    out += [glisur_part(), unified_part(), mie_part(), rayleigh_part()]
    out += [components_part(2), components_part(3), spectrum_part("spectrum", (2.0, 2.5, 3.0, 3.5))]
    return out
