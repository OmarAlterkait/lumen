"""Optics models of your own, registered by name: bulk media and surfaces, with declared parameters that become keys.

    @lumen.medium_model("mylab:water_hg", params={"g": Param("", (-0.99, 0.99), default=0.9)})
    def water_hg(material, base):                 # base: Geant4's reading of the material (lumen.media.MediumSpec)
        return base.with_channel("rayleigh", lobe=lumen.media.henyey_greenstein("g"))

    @lumen.surface_model("mylab:schlick", params={"R0": Param("", (0, 1), default=0.04)})
    def schlick(surface, base):                   # base: Geant4's reading of the surface (a SurfaceEntry, or None)
        def probs(p, ctx):
            c = lumen.surfaces.cos_incidence(ctx)
            r = p["R0"] + (1 - p["R0"]) * (1 - c) ** 5
            return lumen.surfaces.outcome_vector(specular=r, absorb=1 - r)
        return lumen.surfaces.SurfaceType("schlick", ("R0",), (lumen.surfaces.SPECULAR, lumen.surfaces.ABSORB), probs)

    det = lumen.load("tank.lumen", models={"materials/Water": "mylab:water_hg", "surfaces/Liner": "mylab:schlick"})
    det.params()["materials/Water/g"]             # 0.9: calibratable, differentiable (reweight)

Binding: a material's or surface's optics model is a name in the detector file (``OpticsModel.name``), or given at
load (``lumen.load(..., models=)``, without editing the file). lumen's own models are found first (``geant4``,
``lumen:opaque``, ...), so a detector without a registered model builds as it always did; any other name is looked up
in ``lumen.plugins.optics_models`` (this module's decorators register there; installed packages register through the
``lumen.optics_models`` entry points).

Parameters: each declared name is the key ``materials/<M>/<name>`` or ``surfaces/<S>/<name>``, its value the record's
property of that name when the file has one, else the declared default. They are scalars. The model reads them by name
(a lobe's ``params``, a surface type's ``params``); the build maps them as direct reads, so ``pack`` makes them traced
and differentiable (reweighting: the draws stay, the weights move).

What a model can return: a medium model a ``MediumSpec`` (Geant4's channels with lobes and laws swapped or added); a
surface model a ``SurfaceType`` over the ten outcomes (``lumen.surfaces``) whose ``params`` are its declared names or
its Geant4 reading's (R, T, ... keep their keys). A detecting model (``detects=True``) gives DETECT and reads ``eff``,
the efficiency on entry lumen sets (1 under lumen:pmt, whose QE applies after tracking); a border into a sensor reads
it when the surface's record names it (or the group's surface reader does). New direction lobes on surfaces are not
open (the facet lobe carries one width), nor optics that vary with position.

Checks at first build, per model and type: plain JAX (no host callbacks); outcome probabilities of shape (10,) in [0,
1] summing to 1, only on the declared outcomes, on a grid of incidences at the parameters' range corners; a medium
lobe's sampler against its density (a chi^2), at most 3 uniforms per lobe.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable

import numpy as np

BUILTIN_MEDIA = ("geant4", "lumen:opaque", "lumen:same_as", "lumen:larsoft_lar")
BUILTIN_SURFACES = ("geant4", "lumen:larsoft_reflector", "lumen:wls_film")
KINDS = ("medium", "surface")
MAX_RAND = 3  # uniforms per lobe (the step's layout)
SUM_TOL = 1e-5  # |sum of outcome probabilities - 1|
P_TOL = 1e-6  # a probability below -P_TOL or above 1 + P_TOL is refused
LOBE_SAMPLES = 1 << 14
LOBE_BINS = 20


class ModelError(ValueError):
    """A registered optics model whose output breaks the rules of model code (see the module docstring)."""


@dataclasses.dataclass(frozen=True, eq=False)
class OpticsModel:
    """A registered model: its name, sort ("medium" or "surface"), function, declared parameters and ``detects``."""

    name: str
    kind: str
    fn: Callable
    declared: tuple  # ((name, Param), ...)
    detects: bool = False

    def values(self, rec) -> dict:
        """Return {declared name: value} for a record: its property of that name when present, else the default."""
        out = {}
        for k, p in self.declared:
            out[k] = float(np.asarray(rec.properties[k].value)) if k in rec.properties else p.default
        return out

    def keys(self, rec) -> dict:
        """Return {declared name: (key, Source column)} of a record ("constant" when the file has it, else
        "default")."""  # noqa: D205, D209
        section = "materials" if self.kind == "medium" else "surfaces"
        return {k: (f"{section}/{rec.name}/{k}", "constant" if k in rec.properties else "default")
                for k, _ in self.declared}  # fmt: skip


def _check_declaration(name, params):
    from lumen.params import Param

    if ":" not in name or name in BUILTIN_MEDIA + BUILTIN_SURFACES:
        raise ValueError(f"optics model {name!r}: a 'distribution:name', not one of lumen's own")
    for k, p in (params or {}).items():
        if not isinstance(p, Param) or p.moves or p.shape != () or p.default is None:
            raise TypeError(f"optics model {name!r}: {k!r} must be a scalar lumen.Param with a default")
        if "/" in k:
            raise ValueError(f"optics model {name!r}: parameter name {k!r} (no '/')")


def _register(name, kind, params, fn, detects=False):
    from lumen import plugins

    m = OpticsModel(name, kind, fn, tuple(sorted((params or {}).items())), bool(detects))
    from importlib import metadata

    old = plugins.optics_models._entries.get(name)
    if isinstance(old, metadata.EntryPoint) or (isinstance(old, OpticsModel) and _same_code(old.fn, fn)):
        # its package's entry point being loaded (the module registers it), or the same definition run again
        plugins.optics_models._entries[name] = m
        _CHECKED.difference_update({k for k in _CHECKED if k[0] == name})
        return m
    plugins.optics_models.register(name, m)
    return m


def _same_code(a, b):
    return (getattr(a, "__module__", None), getattr(a, "__qualname__", None)) == \
        (getattr(b, "__module__", None), getattr(b, "__qualname__", None))  # fmt: skip


def medium_model(name: str, params: dict | None = None):
    """Return a decorator registering ``f(material, base) -> MediumSpec`` as the medium model ``name``."""
    _check_declaration(name, params)
    return lambda fn: _register(name, "medium", params, fn)


def surface_model(name: str, params: dict | None = None, detects: bool = False):
    """Return a decorator registering ``f(surface, base) -> SurfaceType`` as the surface model ``name``."""
    _check_declaration(name, params)
    return lambda fn: _register(name, "surface", params, fn, detects)


def lookup(name: str, kind: str, what: str) -> OpticsModel:
    """Return the registered model ``name`` of ``kind`` (``what`` names the record in errors)."""
    from lumen import plugins
    from lumen.build.library import LibraryError

    try:
        m = plugins.optics_models.get(name)
    except plugins.NotRegistered as e:
        raise LibraryError(f"{what}: optics model {name!r} is not lumen's own nor registered ({e})") from None
    if not isinstance(m, OpticsModel) or m.kind != kind:
        raise LibraryError(f"{what}: optics model {name!r} is not a {kind} model")
    return m


def registered(name: str, kind: str):
    """Return the registered model ``name`` of ``kind``, or None (lumen's own names, an unknown name)."""
    from lumen import plugins
    from lumen.rules.optics import SENSOR_SURFACES

    if name in (BUILTIN_MEDIA if kind == "medium" else BUILTIN_SURFACES + tuple(SENSOR_SURFACES) + ("lumen:counting",)):
        return None
    try:
        m = plugins.optics_models.get(name)
    except plugins.NotRegistered:
        return None
    return m if isinstance(m, OpticsModel) and m.kind == kind else None


# --------------------------------------------------------------------------------------------- building an entry
def medium_entry(rec, name):
    """Return (MediumEntry, sources {param path: Source}) of a material under its registered medium model."""
    from lumen.build import library as lib
    from lumen.build import surfaces as regions
    from lumen.media.spec import MediumSpec
    from lumen.rules import optics as rules

    m = lookup(rec.optics.name, "medium", f"material {rec.name!r}")
    base = rules.medium(rec, name)
    spec = m.fn(rec, MediumSpec(base.medium, dict(base.params), m.values(rec)))
    if not isinstance(spec, MediumSpec):
        raise ModelError(f"medium model {m.name!r}: return a lumen.media.MediumSpec, not {type(spec).__name__}")
    _check_medium(m, spec)
    sources = lib._medium_sources(rec)
    keys = m.keys(rec)
    for path, k in spec.used.items():
        sources[path] = lib.Source(keys[k][0], keys[k][1], f"declared by {m.name}")
    return regions.MediumEntry(spec.medium, spec.params), sources


def surface_entry(rec, base, *, sensor=False, eff=None):
    """Return the SurfaceEntry (or SensorEntry) of a surface under its registered model; ``base``: Geant4's reading."""
    from lumen.build import surfaces as regions
    from lumen.surfaces.outcomes import DETECT
    from lumen.surfaces.types import SurfaceType

    m = lookup(rec.optics.name, "surface", f"surface {rec.name!r}")
    stype = m.fn(rec, base)
    if not isinstance(stype, SurfaceType):
        raise ModelError(f"surface model {m.name!r}: return a lumen.surfaces.SurfaceType, not {type(stype).__name__}")
    if m.detects != (DETECT in stype.outcomes) or (sensor and not m.detects):
        raise ModelError(f"surface model {m.name!r}: detects={m.detects} but its type "
                         f"{'gives' if DETECT in stype.outcomes else 'never gives'} DETECT"
                         + (" (a sensor's border needs a detecting model)" if sensor else ""))  # fmt: skip
    vals, base_p = m.values(rec), dict(base.params) if base is not None else {}
    if m.detects:
        base_p["eff"] = float(eff) if eff is not None else _efficiency(rec)
        if "eff" not in stype.params:
            raise ModelError(f"surface model {m.name!r}: a detecting type reads 'eff', the efficiency on entry")
    missing = [k for k in stype.params if k not in vals and k not in base_p]
    if missing:
        raise ModelError(f"surface model {m.name!r}: its type reads {missing}, neither declared nor Geant4's reading")
    params = {k: vals[k] if k in vals else base_p[k] for k in stype.params}
    _check_surface(m, stype, params)
    return (regions.SensorEntry if sensor else regions.SurfaceEntry)(stype, params)


def _efficiency(rec):
    from lumen.rules import optics as rules

    return rules.flat_fraction(rec, "EFFICIENCY", 1.0)


# --------------------------------------------------------------------------------------------------- the checks
_CHECKED: set = set()


def _no_callbacks(what, fn, *args):
    import jax

    from lumen.sources.function import _callbacks

    bad = sorted(_callbacks(jax.make_jaxpr(fn)(*args).jaxpr))
    if bad:
        raise ModelError(f"{what}: host callbacks {bad} (model code is plain JAX, per photon)")


def _corners(m, params):
    """Return parameter sets: the values given, and each declared range's finite corners."""
    sets = [dict(params)]
    axes = []
    for k, p in m.declared:
        if k in params:
            lo, hi = p.range
            axes.append([(k, v) for v in (lo, hi) if np.isfinite(v)] or [(k, params[k])])
    for combo in itertools.product(*axes) if axes else ():
        sets.append({**params, **dict(combo)})
    return sets


def _check_surface(m, stype, params):
    import jax.numpy as jnp

    from lumen.surfaces.outcomes import N_OUTCOMES

    key = (m.name, stype.name)
    if key in _CHECKED:
        return
    what = f"surface model {m.name!r} ({stype.name})"
    c = jnp.linspace(0.02, 1.0, 12)
    s = jnp.sqrt(1.0 - c * c)
    wi = jnp.stack([s, jnp.zeros_like(c), -c], -1)
    n = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0]), wi.shape)

    def probs(p, wi_, n_):
        ctx = dict(lam=jnp.full(wi_.shape[:-1], 420.0), wi=wi_, n=n_, n1=jnp.full(wi_.shape[:-1], 1.33),
                   n2=jnp.full(wi_.shape[:-1], 1.0))  # fmt: skip
        return stype.probs(p, ctx)

    p32 = {k: jnp.float32(v) if np.ndim(v) == 0 else jnp.asarray(v, jnp.float32) for k, v in params.items()}
    _no_callbacks(what, probs, p32, wi, n)
    allowed = np.zeros(N_OUTCOMES, bool)
    allowed[list(stype.outcomes)] = True
    for p in _corners(m, params):
        v = np.asarray(probs({k: jnp.asarray(x, jnp.float32) for k, x in p.items()}, wi, n), np.float64)
        if v.shape != (len(c), N_OUTCOMES):
            raise ModelError(f"{what}: probabilities of shape {v.shape[1:]}, expected ({N_OUTCOMES},)")
        if np.any(v < -P_TOL) or np.any(v > 1 + P_TOL) or np.any(np.abs(v.sum(-1) - 1.0) > SUM_TOL):
            raise ModelError(f"{what}: outcome probabilities outside [0, 1] or not summing to 1 at {p}")
        if np.any(np.abs(v[:, ~allowed]) > P_TOL):
            raise ModelError(f"{what}: a probability on an outcome it does not declare (outcomes {stype.outcomes})")
    _CHECKED.add(key)


def _check_medium(m, spec):
    for ch in spec.lobes:
        lobe = next(c.lobe for c in spec.medium.channels if c.name == ch)
        if (m.name, lobe.name) in _CHECKED:
            continue
        check_lobe(lobe, spec.params[ch].get("lobe", {}), f"medium model {m.name!r} ({lobe.name})")
        _CHECKED.add((m.name, lobe.name))


def check_lobe(lobe, p, what="lobe"):
    """Check a bulk lobe: at most 3 uniforms, plain JAX, its sampler against its density (chi^2 in the cosine about
    the incoming direction, ``lumen.validate.stats.chi2_passes``) and the density's normalisation."""  # noqa: D205, D209
    import jax
    import jax.numpy as jnp

    from lumen.validate import stats

    if lobe.n_rand > MAX_RAND:
        raise ModelError(f"{what}: {lobe.n_rand} uniforms; a lobe draws at most {MAX_RAND}")
    if lobe.logpdf is None or lobe.surface:
        raise ModelError(f"{what}: a bulk lobe has a density (logpdf) and no surface normal")
    pj = {k: jnp.asarray(v, jnp.float32) for k, v in p.items()}
    wi = jnp.asarray([0.0, 0.0, 1.0])
    rand = jax.random.uniform(jax.random.PRNGKey(0), (LOBE_SAMPLES, lobe.n_rand), minval=1e-7, maxval=1.0 - 1e-7)
    _no_callbacks(what, lambda r: lobe.sample(pj, wi, wi, r), rand[0])
    wo = jax.vmap(lambda r: lobe.sample(pj, wi, wi, r))(rand)
    mu = np.clip(np.asarray(wo[:, 2], np.float64), -1.0, 1.0)
    edges = np.linspace(-1.0, 1.0, LOBE_BINS + 1)
    count = np.histogram(mu, edges)[0]
    # each bin's probability: Gauss-Legendre in mu (64 nodes a bin), the azimuth averaged on a ring of 16
    x, wq = np.polynomial.legendre.leggauss(64)
    lo, hi = edges[:-1, None], edges[1:, None]
    m_q = (0.5 * (hi - lo) * x + 0.5 * (hi + lo)).ravel()
    phi = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
    sq = np.sqrt(np.clip(1 - m_q**2, 0, 1))
    dirs = np.stack([np.outer(sq, np.cos(phi)), np.outer(sq, np.sin(phi)), np.outer(m_q, np.ones_like(phi))], -1)
    lp = jax.vmap(jax.vmap(lambda d: lobe.logpdf(pj, wi, wi, d)))(jnp.asarray(dirs, jnp.float32))
    dens = 2 * np.pi * np.exp(np.asarray(lp, np.float64)).mean(1)  # per unit mu
    prob = (dens.reshape(LOBE_BINS, -1) * wq).sum(1) * 0.5 * (edges[1:] - edges[:-1])
    if abs(prob.sum() - 1.0) > 0.02:
        raise ModelError(f"{what}: its density integrates to {prob.sum():.4f}, not 1")
    expect = prob / prob.sum() * LOBE_SAMPLES
    ok = expect > 5.0
    chi2 = float(np.sum((count[ok] - expect[ok]) ** 2 / expect[ok]) / max(ok.sum() - 1, 1))
    if not stats.chi2_passes(chi2, max(int(ok.sum()) - 1, 1)):
        raise ModelError(f"{what}: its sampler does not draw its density (chi2/ndf {chi2:.2f})")
