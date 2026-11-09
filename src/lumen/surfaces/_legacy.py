"""The legacy surface types, carried over unchanged until the factorised surface models replace them.

A legacy surface type is static (hashed by identity). ``probs(p, ctx)`` returns the probabilities of the ten outcomes
of ``lumen.surfaces.outcomes`` (shape (10,), summing to 1 by construction) and ``lobe_params(p, ctx)`` the parameters
of the lobes it uses. Nothing about sampling lives here: the outcomes that end the photon are deposited in
expectation and a continuing one is drawn by the step; directions come from the lobes and the outcome maps.

``ctx`` is a dict: ``lam`` (nm), ``wi``, ``n`` (the normal facing the photon), ``n1``, ``n2`` (the indices before
and after at lam), the photon's Jones vector ``er``, ``ei`` and, on a layered surface, ``layer``. Without the Jones
vector the Fresnel shares are the unpolarised ones and a type with no ``amps`` is a perfect mirror.

There are sixteen types plus ``coated`` (a wavelength-shifting film over any opaque type): PASS, ABSORBER,
DIELECTRIC, DIELECTRIC_RT, DIELECTRIC_RT_SENSOR, REFLECTOR, SENSOR, PHOTOCATHODE, COUNTING, DIELECTRIC_METAL,
GLISUR_METAL, GLISUR_METAL_EFF, SPECULAR_METAL, UNIFIED_METAL, PAINTED and PAINTED_T.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from lumen._core.safe import clip01, sanitize
from lumen.media import phase
from lumen.surfaces import fresnel, polar
from lumen.surfaces.outcomes import (
    ABSORB,
    BACKSCATTER,
    DETECT,
    DIFFUSE,
    DIFFUSE_T,
    GLOSSY,
    INDEX,
    N_OUTCOMES,
    NAMES,
    REEMIT,
    REFRACT,
    SPECULAR,
    STRAIGHT,
    TERMINAL,
)

COATED_FACE = 0.5  # the film is on the faces whose facing normal has n . axis above this
GRAZING_FILM = 1e-6  # |cos| floor of the path length through a film


def _no_lobes(p, ctx):
    return {}


@dataclass(frozen=True, eq=False)
class SurfaceType:
    """A legacy surface type (see the module docstring).

    ``probs(p, ctx)`` gives the (10,) outcome probabilities for the parameters named ``params``, and only the
    ``outcomes`` listed can be non-zero. ``lobe_params(p, ctx)`` gives the lobes' parameters (``sigma_alpha`` for the
    GLOSSY lobe). ``reemit`` is the spectrum of the REEMIT outcome (``lumen.media.spectra``) or None, and
    ``reemit_lobe`` its direction lobe (None: Lambertian about the normal). ``glossy`` names the chain lobe of the
    GLOSSY outcome: "unified" (Geant4's unified model) or "glisur". ``stack`` marks a layered surface (``coated``),
    whose ``lobe_params`` also gives ``film`` and ``coated``. ``amps(p, ctx)`` gives the complex amplitudes
    (r_s, r_p, t_s, t_p) of SPECULAR and REFRACT; None is a perfect mirror.
    """

    name: str
    params: tuple[str, ...]
    outcomes: tuple[int, ...]
    probs: Callable
    lobe_params: Callable = _no_lobes
    reemit: object = None
    glossy: str = "unified"
    reemit_lobe: object = None
    stack: bool = False
    amps: Callable | None = None

    def __repr__(self):
        """Show the type's name."""
        return f"SurfaceType({self.name})"


def outcome_vector(**shares):
    """Return the (10,) outcome probabilities with the named shares (outcome name = value) and zeros elsewhere."""
    v = [jnp.zeros(()) for _ in range(N_OUTCOMES)]
    for name, x in shares.items():
        v[INDEX[name]] = x
    return jnp.stack(jnp.broadcast_arrays(*v), -1)


def cos_incidence(ctx):
    """Return the cosine of the angle of incidence, clipped to [0, 1]."""
    return jnp.clip(-jnp.sum(ctx["wi"] * ctx["n"], -1), 0.0, 1.0)


def is_polarised(ctx):
    """Return whether the context carries the photon's Jones vector."""
    return "er" in ctx


def _state_fresnel(ctx):
    return fresnel.state_reflectance(ctx["wi"], ctx["n"], ctx["n1"], ctx["n2"], ctx["er"], ctx["ei"])


def _fresnel(ctx):
    if is_polarised(ctx):
        return _state_fresnel(ctx)
    return fresnel.unpolarised_reflectance(cos_incidence(ctx), ctx["n1"], ctx["n2"])


def chained_cuts(*p):
    """Return Geant4's chained tests ``rand < p1``, ``rand < p1 + p2``, ... as probabilities (one more than given).

    Cumulative sums clipped to [0, 1] and differenced: any values, even ones summing past 1, give a distribution,
    and the last share is the rest. The clip's derivative is ``clip01``'s, so on the closed simplex (values >= 0
    summing to <= 1) the derivative is that of the linear map, also on its faces (R = 1, a zero value).
    """
    c = clip01(jnp.cumsum(jnp.stack(jnp.broadcast_arrays(*p)), 0))
    edges = jnp.concatenate([jnp.zeros_like(c[:1]), c, jnp.ones_like(c[:1])], 0)
    return tuple(edges[1:] - edges[:-1])


def _sigma_alpha(p, ctx):
    return {"sigma_alpha": p["sigma_alpha"]}


# ---------------------------------------------------------------------------------------------- pass and absorb
PASS = SurfaceType("pass", (), (STRAIGHT,), lambda p, ctx: outcome_vector(straight=jnp.ones(())))
ABSORBER = SurfaceType("absorber", (), (ABSORB,), lambda p, ctx: outcome_vector(absorb=jnp.ones(())))


# ---------------------------------------------------------------------------------------------------- dielectric
def _dielectric(p, ctx):
    r = _fresnel(ctx)
    return outcome_vector(refract=1.0 - r, specular=r)


def _dielectric_amps(p, ctx):
    return fresnel.amplitudes(cos_incidence(ctx), ctx["n1"], ctx["n2"])


DIELECTRIC = SurfaceType("dielectric", (), (REFRACT, SPECULAR), _dielectric, amps=_dielectric_amps)


def _dielectric_rt(p, ctx):
    """Geant4's polished dielectric_dielectric surface with REFLECTIVITY and TRANSMITTANCE tables.

    Notation: R, T the tables, F the Fresnel reflectance.

    Notes
    -----
    G4OpBoundaryProcess::PostStepDoIt (the same in WCSimOpBoundaryProcess with its photocathode model 0): one draw
    against R, then R + T. Below R the photon meets the two media's Fresnel interface (refraction or reflection, with
    its polarisation); between R and R + T it passes straight on (Transmission); above it is absorbed (DoAbsorption).
    Geant4 tests the absorption BEFORE Fresnel, so R < 1 here is not a reflectance. As a sensor the absorbed share is
    detected with ``eff`` (Geant4's EFFICIENCY, or 1 where the sensitive detector counts every absorption).
    """
    r, t, a = chained_cuts(p["R"], p["T"])
    F = _fresnel(ctx)
    eff = p.get("eff", 0.0)
    return outcome_vector(refract=r * (1.0 - F), specular=r * F, straight=t, detect=a * eff, absorb=a * (1.0 - eff))


DIELECTRIC_RT = SurfaceType(
    "dielectric_rt", ("R", "T"), (REFRACT, SPECULAR, STRAIGHT, ABSORB), _dielectric_rt, amps=_dielectric_amps
)
DIELECTRIC_RT_SENSOR = SurfaceType(
    "dielectric_rt_sensor",
    ("R", "T", "eff"),
    (REFRACT, SPECULAR, STRAIGHT, DETECT, ABSORB),
    _dielectric_rt,
    amps=_dielectric_amps,
)


# ------------------------------------------------------------------------------------- reflectors and sensors
def _reflector(p, ctx):
    """Reflectance R (a law evaluated by the caller), split by f_spec, f_glossy and the diffuse rest; 1 - R absorbed.

    Notation: R the reflectance.
    """
    R = p["R"]
    fs, fg = p["f_spec"], p["f_glossy"]
    return outcome_vector(specular=R * fs, glossy=R * fg, diffuse=R * (1.0 - fs - fg), absorb=1.0 - R)


REFLECTOR = SurfaceType(
    "reflector",
    ("R", "f_spec", "f_glossy", "sigma_alpha"),
    (SPECULAR, GLOSSY, DIFFUSE, ABSORB),
    _reflector,
    _sigma_alpha,
)


def _photocathode_amps(p, ctx):
    return fresnel.amplitudes(
        cos_incidence(ctx), ctx["n1"], p["nr"] + 1j * sanitize(p["nk"] > 0, p["nk"], fresnel.NK_FLOOR)
    )


def _photocathode(p, ctx):
    """Return the shares of a sensor window seen from a dielectric, with its complex index (nr, nk).

    Notation: R the reflectance, A = 1 - R the absorbed share.

    The reflected light is split like REFLECTOR's; the absorbed rest is detected with the quantum yield qe (a law in
    the tables) and otherwise absorbed. The angular dependence of detection is the Fresnel transmission; a separate
    angular efficiency would count it twice.
    """
    if is_polarised(ctx):
        rs, rp, _, _ = _photocathode_amps(p, ctx)
        fs_, fp_ = polar.fractions(ctx["er"], ctx["ei"], ctx["wi"], ctx["n"])
        R = jnp.clip(fresnel.reflectance(fs_, fp_, rs, rp), 0.0, 1.0)
    else:
        R = fresnel.unpolarised_absorbing_reflectance(cos_incidence(ctx), ctx["n1"], p["nr"], p["nk"])
    A = 1.0 - R
    fs, fg = p["f_spec"], p["f_glossy"]
    return outcome_vector(
        specular=R * fs, glossy=R * fg, diffuse=R * (1.0 - fs - fg), detect=A * p["qe"], absorb=A * (1.0 - p["qe"])
    )


PHOTOCATHODE = SurfaceType(
    "photocathode",
    ("nr", "nk", "qe", "f_spec", "f_glossy", "sigma_alpha"),
    (SPECULAR, GLOSSY, DIFFUSE, DETECT, ABSORB),
    _photocathode,
    _sigma_alpha,
    amps=_photocathode_amps,
)


def _sensor(p, ctx):
    """Return the shares of an absorbing sensor: reflectance R split like REFLECTOR's, 1 - R detected with qe.

    Notation: R the reflectance, A = 1 - R the absorbed share.
    """
    R, fs, fg = p["R"], p["f_spec"], p["f_glossy"]
    A = 1.0 - R
    return outcome_vector(
        specular=R * fs, glossy=R * fg, diffuse=R * (1.0 - fs - fg), detect=A * p["qe"], absorb=A * (1.0 - p["qe"])
    )


SENSOR = SurfaceType(
    "sensor",
    ("R", "f_spec", "f_glossy", "sigma_alpha", "qe"),
    (SPECULAR, GLOSSY, DIFFUSE, DETECT, ABSORB),
    _sensor,
    _sigma_alpha,
)


def _counting(p, ctx):
    """Return the shares of a counting-only sensor (counted where the photon enters it), efficiency eff."""
    return outcome_vector(detect=p["eff"], absorb=1.0 - p["eff"])


COUNTING = SurfaceType("counting", ("eff",), (DETECT, ABSORB), _counting)


# ---------------------------------------------------------------------------------------------------------- metals
def _dielectric_metal(p, ctx):
    """Geant4's dielectric_metal surface, model unified, finish ground (G4OpBoundaryProcess::DielectricMetal).

    REFLECTIVITY R, the rest absorbed and detected with EFFICIENCY eff. A reflection is a spike (SPECULARSPIKECONSTANT
    p_spike), the facet lobe (SPECULARLOBECONSTANT p_lobe, width sigma_alpha), a backscatter (BACKSCATTERCONSTANT
    p_back) or Lambertian (the rest). Finish polished is p_spike = 1 (Geant4 ignores the constants there). Ground with
    sigma_alpha = 0 is not: GetFacetNormal returns the normal but ChooseReflection still applies the backscatter and
    Lambertian constants, so the importer refuses sigma_alpha = 0 on a ground finish. TRANSMITTANCE is not part of
    this type: it would put the photon into an opaque medium.
    """
    r, a = chained_cuts(p["R"])
    ss, sl, bs, lam = chained_cuts(p["p_spike"], p["p_lobe"], p["p_back"])
    return outcome_vector(
        specular=r * ss,
        glossy=r * sl,
        backscatter=r * bs,
        diffuse=r * lam,
        detect=a * p["eff"],
        absorb=a * (1.0 - p["eff"]),
    )


DIELECTRIC_METAL = SurfaceType(
    "dielectric_metal",
    ("R", "p_spike", "p_lobe", "p_back", "sigma_alpha", "eff"),
    (SPECULAR, BACKSCATTER, GLOSSY, DIFFUSE, DETECT, ABSORB),
    _dielectric_metal,
    _sigma_alpha,
)


def _glisur_metal(p, ctx):
    r, a = chained_cuts(p["R"])
    return outcome_vector(glossy=r, absorb=a)


def _glisur_metal_eff(p, ctx):
    r, a = chained_cuts(p["R"])
    return outcome_vector(glossy=r, detect=a * p["eff"], absorb=a * (1.0 - p["eff"]))


def _glisur_width(p, ctx):
    """Return the GLISUR facet radius 1 - polish, under the GLOSSY lobe's width name."""
    return {"sigma_alpha": 1.0 - p["polish"]}


def _specular_metal(p, ctx):
    return outcome_vector(**dict(zip(("specular", "absorb"), chained_cuts(p["R"]), strict=True)))


def _unified_metal(p, ctx):
    r, a = chained_cuts(p["R"])
    ss, sl, bs, lam = chained_cuts(p["p_spike"], p["p_lobe"], p["p_back"])
    return outcome_vector(specular=r * ss, glossy=r * sl, backscatter=r * bs, diffuse=r * lam, absorb=a)


# dielectric_metal, glisur, finish ground, polish < 1: R into the GLISUR facet chain (radius 1 - polish)
GLISUR_METAL = SurfaceType(
    "glisur_metal", ("R", "polish"), (GLOSSY, ABSORB), _glisur_metal, _glisur_width, glossy="glisur"
)
GLISUR_METAL_EFF = SurfaceType(
    "glisur_metal_eff",
    ("R", "polish", "eff"),
    (GLOSSY, DETECT, ABSORB),
    _glisur_metal_eff,
    _glisur_width,
    glossy="glisur",
)
# dielectric_metal, glisur with another finish or polish 1, or unified polished: R mirror
SPECULAR_METAL = SurfaceType("specular_metal", ("R",), (SPECULAR, ABSORB), _specular_metal)
# dielectric_metal, unified, any other (non-LUT) finish: spike, lobe, backscatter, Lambertian as Geant4's cuts
UNIFIED_METAL = SurfaceType(
    "unified_metal",
    ("R", "p_spike", "p_lobe", "p_back", "sigma_alpha"),
    (SPECULAR, BACKSCATTER, GLOSSY, DIFFUSE, ABSORB),
    _unified_metal,
    _sigma_alpha,
)


# ----------------------------------------------------------------------------------------------- painted surfaces
def _painted(p, ctx):
    r, a = chained_cuts(p["R"])
    return outcome_vector(specular=r * p["f_spec"], diffuse=r * (1.0 - p["f_spec"]), absorb=a)


def _painted_t(p, ctx):
    r, t, a = chained_cuts(p["R"], p["T"])
    return outcome_vector(specular=r * p["f_spec"], diffuse=r * (1.0 - p["f_spec"]), straight=t, absorb=a)


# dielectric_dielectric, polishedfrontpainted (f_spec 1: R mirror) or groundfrontpainted (f_spec 0: R Lambertian)
PAINTED = SurfaceType("painted", ("R", "f_spec"), (SPECULAR, DIFFUSE, ABSORB), _painted)
PAINTED_T = SurfaceType("painted_t", ("R", "T", "f_spec"), (STRAIGHT, SPECULAR, DIFFUSE, ABSORB), _painted_t)


# --------------------------------------------------------------------------------------------- the coated surface
def coated(name, backing, reemission):
    """Return ``backing`` with a wavelength-shifting film, thinner than any length the geometry resolves, on one face.

    Geant4 tracks the photon through a WLS volume of thickness d, index-matched to the medium in front, and onto the
    backing (TPB on a reflector foil or a sensor window). Nothing is folded or averaged: the step crosses the layers
    one vertex at a time, at the same point.

    - Film crossing (inward, or outward after the backing): absorbed with A = 1 - exp(-d / (L(lam) |cos|)), then
      re-emitted isotropically (G4OpWLS) with expected yield eta, wavelength from ``reemission`` and delay tau; else
      straight on. eta is the expected number of photons per absorption, carried as the outcome weight A eta; the
      absorbed share is A max(1 - eta, 0).
    - Backing: the backing's own outcomes at the photon's wavelength and direction.

    The film is on the faces whose facing normal has n . axis > 1/2; elsewhere the surface is the bare backing.
    Parameters: L (a law), d (m), eta, tau (ns), axis (3,), and the backing's as ``b:<name>``.
    Notation: A the film's absorbed share.
    """
    through = {REFRACT, STRAIGHT, DIFFUSE_T, REEMIT} & set(backing.outcomes)
    if through:
        # a transmitting backing (a coated window) would let light cross the film from behind, and a re-emitting
        # one would share the film's spectrum and delay: neither is modelled, so both are refused
        raise ValueError(
            f"coated {name!r}: backing {backing.name!r} has outcomes {sorted(NAMES[o] for o in through)}; "
            "only an opaque backing (reflect / absorb / detect) is supported"
        )
    backing_params = tuple(f"b:{q}" for q in backing.params)

    def split(p):
        return {q: p[f"b:{q}"] for q in backing.params}

    def is_coated(p, ctx):
        return jnp.sum(ctx["n"] * p["axis"], -1) > COATED_FACE

    def film_role(p, ctx):
        layer = ctx.get("layer", 0)
        return (layer == 1) | ((layer == 0) & is_coated(p, ctx))

    def probs(p, ctx):
        c = jnp.abs(jnp.sum(ctx["wi"] * ctx["n"], -1))
        A = -jnp.expm1(-p["d"] / (p["L"] * jnp.maximum(c, GRAZING_FILM)))
        film = outcome_vector(reemit=A * p["eta"], absorb=A * jnp.maximum(1.0 - p["eta"], 0.0), straight=1.0 - A)
        return jnp.where(film_role(p, ctx), film, backing.probs(split(p), ctx))

    def lobe_params(p, ctx):
        lp = dict(backing.lobe_params(split(p), ctx))
        lp.update(tau=p["tau"], film=film_role(p, ctx), coated=is_coated(p, ctx))
        return lp

    amps = None if backing.amps is None else (lambda p, ctx: backing.amps(split(p), ctx))
    return SurfaceType(
        name,
        ("L", "d", "eta", "tau", "axis") + backing_params,
        tuple(sorted(set(backing.outcomes) | {REEMIT, STRAIGHT, ABSORB})),
        probs,
        lobe_params,
        reemission,
        backing.glossy,
        phase.ISOTROPIC,
        True,
        amps,
    )


# ------------------------------------------------------------------------------------- values the model fixes (OUT)
_FIXED = {}  # (type, constants) -> the derived type: one object per pair, so rows with the same constants share it


def _const(v):
    return tuple(float(x) for x in np.ravel(v)) if np.ndim(v) else float(v)


def _probe_support(kind, consts):
    """Return the outcomes of ``kind`` with ``consts`` held that are not identically zero (terminal ones kept).

    Probed at every other parameter 0.5 and a generic incidence; a type whose probe needs more keeps its support.
    """
    try:
        p = {k: 0.5 for k in kind.params if k not in consts} | {k: jnp.asarray(v) for k, v in consts.items()}
        ctx = dict(lam=jnp.asarray(400.0), wi=jnp.asarray([0.6, 0.0, -0.8]), n=jnp.asarray([0.0, 0.0, 1.0]),
                   n1=jnp.asarray(1.33), n2=jnp.asarray(1.5), layer=0)  # fmt: skip
        pr = [float(x) for x in kind.probs(p, ctx)]
    except Exception:
        return kind.outcomes
    return tuple(o for o in kind.outcomes if TERMINAL[o] or pr[o] != 0.0)


def fixed(kind, consts):
    """Return ``kind`` with the parameters in ``consts`` held at those values: a type whose parameters are the rest.

    A value the model fixes is not a parameter (the edge rule's OUT): a lowering-implied constant (a frontpainted
    finish's f_spec, Geant4's default TRANSMITTANCE 0, a film's thickness and side). The derived type is named
    ``type{p=v,...}`` and its support is probed (``_probe_support``), so the edge floor never proposes an outcome the
    constants rule out.
    """
    consts = {k: _const(v) for k, v in consts.items()}
    if not consts:
        return kind
    key = (id(kind), tuple(sorted(consts.items())))
    if key not in _FIXED:

        def full(p):
            return {**{k: jnp.asarray(v, jnp.float32) for k, v in consts.items()}, **p}

        tag = ",".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}=({','.join(f'{x:g}' for x in v)})"
                       for k, v in sorted(consts.items()))  # fmt: skip
        amps = None if kind.amps is None else (lambda p, ctx: kind.amps(full(p), ctx))
        derived = SurfaceType(f"{kind.name}{{{tag}}}", tuple(q for q in kind.params if q not in consts),
                              _probe_support(kind, consts), lambda p, ctx: kind.probs(full(p), ctx),
                              lambda p, ctx: kind.lobe_params(full(p), ctx), kind.reemit, kind.glossy,
                              kind.reemit_lobe, kind.stack, amps)  # fmt: skip
        _FIXED[key] = (kind, derived)  # the base kept alive: its id stays unique
    return _FIXED[key][1]


TYPES = (
    PASS,
    ABSORBER,
    DIELECTRIC,
    DIELECTRIC_RT,
    DIELECTRIC_RT_SENSOR,
    REFLECTOR,
    SENSOR,
    PHOTOCATHODE,
    COUNTING,
    DIELECTRIC_METAL,
    GLISUR_METAL,
    GLISUR_METAL_EFF,
    SPECULAR_METAL,
    UNIFIED_METAL,
    PAINTED,
    PAINTED_T,
)
