"""The Geant4 surface types: one parameter table and one kernel (docs/dev/architecture.md, "Surfaces").

Every surface and sensor row of a detector is a row of one parameter table, and one kernel (``kernel``) gives its
continuing shares (the expected weight going on per continuing outcome of ``lumen.surfaces.outcomes``, shape (6,),
summing to at most 1: the rest is lost), its detected share (the expected weight deposited as detected, a scalar, not
part of the continuing shares), the parameters of its lobes, the complex amplitudes of its deterministic
outcomes and its deposit's scale. Nothing about sampling lives here: the deposits are made in expectation and a
continuing outcome is drawn by the step; directions come from the lobes and the outcome maps.

A ``SurfaceType`` is the structure of a row: the table columns it reads as parameters (keys, or the build's values),
constants for the rest (a value the model never reads, folded at load: ``fixed``), its interaction model
(``Interaction``) and flags. A model is a value: its parameters, its shares and the direction laws of its drawn
outcomes. Geant4's models (``MODELS``) share Geant4's chain of tests (``_geant4_chain``):

    r, t, a = chained_cuts(Rc, T)           Rc: R, or the complex index's reflectance (G4_COMPLEX)
    r -> the reflection split (spike / lobe / backscatter / Lambertian: p_spike, p_lobe, p_back), or the Fresnel
         interface (G4_DIELECTRIC: specular F, refraction 1 - F)
    t -> straight on;  a -> deposited as detected with eff, the rest lost
    far side into an opaque medium -> lost (``into_opaque``)

The kernel branches only where the cost differs: one ``lax.switch`` over the models present in the build. The columns
a row does not read hold neutral values (``COLUMNS``): x * 1, x + 0 and ``chained_cuts(R, 0)`` leave each Geant4
type's own probabilities, bit for bit.

``ctx`` is a dict: ``lam`` (nm), ``wi``, ``n`` (the normal facing the photon), ``n1``, ``n2`` (the indices before
and after at lam) and the photon's Jones vector ``er``, ``ei``. Without the Jones vector the Fresnel shares are the
unpolarised ones.

There are ten types plus ``transmitting`` (a dielectric_metal type with Geant4's TRANSMITTANCE): PASS, ABSORBER,
DIELECTRIC, DIELECTRIC_RT, PHOTOCATHODE, COUNTING, GLISUR_METAL, UNIFIED_METAL, UNIFIED_METAL_NORMAL and PAINTED_T;
PAINTED and REFLECTOR are PAINTED_T, and SPECULAR_METAL is UNIFIED_METAL_NORMAL, with constants fixed. Every type
Geant4's EFFICIENCY applies to reads it as ``eff`` (the absorbed share detected); an ordinary surface is that type with
``eff`` fixed at 0 (``no_detection``), a sensor the type itself.
"""

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from lumen._core import domains
from lumen._core.domains import FRACTION, POSITIVE
from lumen._core.safe import clip01
from lumen.params import Param
from lumen.surfaces import fresnel, polar
from lumen.surfaces.outcomes import (
    BACKSCATTER,
    DIFFUSE,
    FAR_SIDE,
    GLOSSY,
    INDEX,
    N_OUTCOMES,
    REFRACT,
    SPECULAR,
    STRAIGHT,
)

F32 = jnp.float32
DEFAULT_WIDTH = 0.1  # the GLOSSY lobe's width a row without one reports (never drawn)


# the table's columns and the neutral value of each (what a row that does not read the column holds): R, T and eff
# Geant4's cuts; the reflection split (p_spike, p_lobe, p_back: a painted finish's f_spec is its spike); the unified
# lobe's width and GLISUR's polish; a complex index
COLUMNS = {
    "R": 0.0,
    "T": 0.0,
    "eff": 0.0,
    "p_spike": 0.0,
    "p_lobe": 0.0,
    "p_back": 0.0,
    "sigma_alpha": DEFAULT_WIDTH,
    "polish": 0.0,
    "nr": 1.0,
    "nk": 0.0,
}


@dataclass(frozen=True)
class SurfaceType:
    """The structure of a surface row (see the module docstring); equal types are one type.

    ``params``: the names it reads from the row's parameters; ``outcomes``: the outcomes that can be non-zero.
    ``consts``: ((name, value), ...), the values fixed by the structure (the base type's own and those ``fixed``
    folds). ``alias``: ((name, column), ...) where a name reads another column (a painted finish's f_spec is the
    spike's). ``model``: the row's ``Interaction``
    (``MODELS``); ``normal``:
    the lobe reflects about the global normal, as the spike (sigma_alpha 0); ``glossy``: the facet-path lobe of the
    GLOSSY outcome, "unified" or "glisur" (whose width is 1 - polish); ``far_opaque``: the far-side outcomes are
    absorbed (``into_opaque``); ``per_normal``: read for a sensor model that applies its QE after tracking
    (``per_normal``). Which parameters the defensive mixture covers is the model's declaration
    (``Interaction.defensive``, read by ``role``). ``support(consts)`` gives the outcomes that remain possible
    when the parameters in ``consts`` are fixed at those values (structure, never a probe of values; None: every
    declared outcome).
    """

    name: str
    params: tuple[str, ...]
    outcomes: tuple[int, ...]
    consts: tuple = ()
    alias: tuple = ()
    model: "Interaction" = None  # type: ignore[assignment]
    normal: bool = False
    glossy: str = "unified"
    far_opaque: bool = False
    per_normal: bool = False
    support: Callable | None = field(default=None, compare=False, repr=False)

    def __repr__(self):
        """Show the type's name."""
        return f"SurfaceType({self.name})"

    @property
    def detects(self) -> bool:
        """Return whether the type can detect: it reads a detection efficiency ``eff`` (a parameter, or a constant other
        than 0; ``no_detection`` fixes it at 0)."""  # noqa: D205, D209
        return "eff" in self.params or bool(dict(self.consts).get("eff", 0.0))

    def role(self, name):
        """Return the defensive role of the parameter ``name`` of this type (``Interaction.defensive``: "zero" it can
        zero an outcome's probability, "support" it moves a lobe's support edge), or None."""  # noqa: D205, D209
        return self.model.defensive.get(self.column(name)) if name in self.params else None

    def range(self, name):
        """Return the (lo, hi) range of the parameter ``name`` of this type (its model's ``Param``)."""
        return self.model.params[self.column(name)].range

    def column(self, name):
        """Return the table column the parameter or constant ``name`` is read from."""
        return dict(self.alias).get(name, name)

    def row(self):
        """Return {column: value} of the row's constants: the neutral values, then the type's own (``COLUMNS``)."""
        out = dict(COLUMNS)
        out.update({self.column(k): v for k, v in self.consts})
        return out

    def evaluate(self, p, ctx):
        """Return the TypeEval of one row of this type at parameters ``p`` (by name, at ctx's wavelengths)."""
        cols = self.row()
        cols.update({self.column(k): v for k, v in p.items()})

        def get(c):
            v = cols[c]
            return jnp.stack([jnp.float32(x) for x in v]) if isinstance(v, tuple) else jnp.asarray(v, F32)

        return kernel(get, self.flags(), ctx, (self.model,), self.per_normal)

    def flags(self):
        """Return the row's Flags (the kernel's per-row structure) as Python constants."""
        return Flags(0, self.normal, self.glossy == "glisur", self.far_opaque, self.per_normal)


class TypeEval(NamedTuple):
    """A surface row evaluated at a crossing.

    The continuing shares (6,, ``Interaction``), the detected share (a scalar deposit), the GLOSSY width, the
    amplitudes (8,: (r_s, r_p, t_s, t_p) as (re, im) pairs; a perfect mirror's without a Fresnel interface) and
    ``inv_an``, the factor on the detect deposit: 1 / A_n for a row read for a sensor model's QE (``per_normal``),
    else 1.
    """

    probs: jax.Array
    detect: jax.Array
    sigma_alpha: jax.Array
    amps8: jax.Array
    inv_an: jax.Array


class Flags(NamedTuple):
    """A row's structure the kernel reads: ``model`` the index of its Interaction among the kernel's ``models``; the
    rest as ``SurfaceType``."""  # noqa: D205, D209

    model: object
    normal: object
    glisur: object
    far_opaque: object
    per_normal: object


def outcome_vector(**shares):
    """Return the (6,) outcome probabilities with the named shares (outcome name = value) and zeros elsewhere."""
    v = [jnp.zeros((), jnp.float32) for _ in range(N_OUTCOMES)]
    for name, x in shares.items():
        v[INDEX[name]] = x
    return jnp.stack(jnp.broadcast_arrays(*v), -1)


def cos_incidence(ctx):
    """Return the cosine of the angle of incidence, clipped to [0, 1]."""
    return jnp.clip(-jnp.sum(ctx["wi"] * ctx["n"], -1), 0.0, 1.0)


def is_polarised(ctx):
    """Return whether the context carries the photon's Jones vector."""
    return "er" in ctx


def chained_cuts(*p):
    """Return Geant4's chained tests ``rand < p1``, ``rand < p1 + p2``, ... as probabilities (one more than given).

    Cumulative sums clipped to [0, 1] and differenced: any values, even ones summing past 1, give a distribution,
    and the last share is the rest. The clip's derivative is ``clip01``'s, so on the closed simplex (values >= 0
    summing to <= 1) the derivative is that of the linear map, also on its faces (R = 1, a zero value).
    """
    c = clip01(jnp.cumsum(jnp.stack(jnp.broadcast_arrays(*p)), 0))
    edges = jnp.concatenate([jnp.zeros_like(c[:1]), c, jnp.ones_like(c[:1])], 0)
    return tuple(edges[1:] - edges[:-1])


# ------------------------------------------------------------------------------------------------ the Fresnel branches
def _amps8(a):
    """Return (r_s, r_p, t_s, t_p) as the (8,) float32 (re, im) pairs of ``TypeEval``."""
    a = [jnp.asarray(z, jnp.complex64) for z in a]
    return jnp.stack([f(z) for z in a for f in (jnp.real, jnp.imag)]).astype(F32)


MIRROR_AMPS = (*polar.MIRROR, 1.0 + 0.0j, 1.0 + 0.0j)  # a perfect mirror's (r_s, r_p, t_s, t_p): no interface
MIRROR_AMPS8 = tuple(x for z in MIRROR_AMPS for x in (z.real, z.imag))  # the same as ``TypeEval``'s (re, im) pairs


# each Fresnel branch returns (the cut's reflectance Rc, the shares of the reflected r: (specular, glossy, backscatter,
# diffuse, refraction), the amplitudes (8,))


def _none(get, ctx, f):
    """No Fresnel interface: R splits by the reflection constants (spike, lobe, backscatter, Lambertian: Geant4's
    cuts; at sigma_alpha 0 the lobe is the spike's mirror)."""  # noqa: D205, D209
    ss, sl, bs, lm = chained_cuts(get("p_spike"), get("p_lobe"), get("p_back"))
    zero = F32(0.0)
    split = (jnp.where(f.normal, ss + sl, ss), jnp.where(f.normal, zero, sl), bs, lm, zero)
    return get("R"), split, _amps8(MIRROR_AMPS)


def _real(get, ctx, f):
    """Return the two media's interface (G4OpBoundaryProcess::DielectricDielectric): R splits into specular F and
    refraction 1 - F, F the Fresnel reflectance of the amplitudes (``fresnel.amplitudes``: total reflection included),
    at the photon's own state when polarised."""  # noqa: D205, D209
    amps = fresnel.amplitudes(cos_incidence(ctx), ctx["n1"], ctx["n2"])
    fs, fp = polar.fractions(ctx["er"], ctx["ei"], ctx["wi"], ctx["n"]) if is_polarised(ctx) else (0.5, 0.5)
    F = jnp.clip(fresnel.reflectance(fs, fp, amps[0], amps[1]), 0.0, 1.0)
    zero = F32(0.0)
    return get("R"), (F, zero, zero, zero, 1.0 - F), _amps8(amps)


def _complex(get, ctx, f):
    """Return a complex index (nr, nk) seen from a dielectric (Geant4's dielectric_metal with RINDEX and KINDEX: a
    photocathode): its reflectance is the cut R, the reflection specular; the absorbed rest detected with eff. The
    angular dependence of detection is the Fresnel transmission; a separate angular efficiency would count it twice."""  # noqa: D205, D209
    amps = fresnel.amplitudes(cos_incidence(ctx), ctx["n1"], get("nr") + 1j * get("nk"))
    if is_polarised(ctx):
        fs_, fp_ = polar.fractions(ctx["er"], ctx["ei"], ctx["wi"], ctx["n"])
        R = jnp.clip(fresnel.reflectance(fs_, fp_, amps[0], amps[1]), 0.0, 1.0)
    else:
        R = fresnel.unpolarised_absorbing_reflectance(cos_incidence(ctx), ctx["n1"], get("nr"), get("nk"))
    zero = F32(0.0)
    return R, (F32(1.0), zero, zero, zero, zero), _amps8(amps)


def _geant4_chain(fresnel_branch):
    """Return the shares function of a Geant4 model: Geant4's chain of tests around ``fresnel_branch``.

    ``fresnel_branch(get, ctx, f)`` gives (the cut's reflectance Rc, the shares of the reflected r: (specular, glossy,
    backscatter, diffuse, refraction), the amplitudes (8,)); the chain cuts Rc and T (``chained_cuts``), deposits the
    absorbed share detected with eff (the "detect" channel; the rest of it is lost) and, on a row whose far side is
    opaque, loses the far-side shares.
    """

    def shares(get, f, ctx):
        Rc, split, amps = fresnel_branch(get, ctx, f)
        r, t, a = chained_cuts(Rc, get("T"))
        spec, glossy, back, diffuse, refract = split
        pr = outcome_vector(refract=r * refract, specular=r * spec, glossy=r * glossy, backscatter=r * back,
                            diffuse=r * diffuse, straight=t)  # fmt: skip
        pr = jnp.where(f.far_opaque, jnp.where(jnp.asarray(FAR_SIDE), 0.0, pr), pr)
        return pr, a * get("eff"), amps

    return shares


@dataclass(frozen=True, eq=False)
class Interaction:
    """A surface interaction model: a value, no registry.

    ``params``: {column: Param} the table columns it reads (unit and range; the columns a row does not read hold their
    neutral values, ``COLUMNS``). ``shares(get, f, ctx)``: the row's continuing shares (6,: the expected weight going
    on per outcome; the loss is 1 - their sum), its detected share (a scalar: the expected weight deposited as
    detected, independent of the continuing shares: a model may deposit a share and let the photon go on) and its
    amplitudes (8,): ``get(column)`` reads the row's values at ctx's wavelength, ``f`` its Flags (the
    cheap per-row variants: ``normal``, ``glisur``, ``far_opaque``). ``lobes``: {outcome: lobe name} the direction laws
    of the drawn outcomes it may give (GLOSSY's facet path is the row's: ``facets.lobe_name``). The kernel switches over
    the models present in the build (``kernel``), one branch each; rows of one model share a branch. ``defensive``:
    {column: role} the parameters a draw at the reference may fail to cover (``lumen.transport.estimator``): "zero" a
    share in [0, 1] that can zero an outcome's probability (covered by the uniform density over the outcomes of
    reference probability 0), "support" a parameter that moves the edge of a lobe's support (covered by the lobe's own
    cover, ``facets.covered``).
    """

    name: str
    params: Mapping[str, Param]
    shares: Callable
    lobes: Mapping[int, str] = field(default_factory=dict)
    defensive: Mapping[str, str] = field(default_factory=dict)

    def __repr__(self):
        """Show the model's name."""
        return f"Interaction({self.name})"


_FRAC, _IDX = Param("", FRACTION), Param("", domains.INDEX)
_CUTS = {"R": _FRAC, "T": _FRAC, "eff": _FRAC}
_ZERO = dict.fromkeys(("R", "T", "eff"), "zero")  # the cuts: each can zero an outcome's probability
# Geant4's surface models (G4OpBoundaryProcess): no interface (dielectric_metal and frontpainted reflections), the two
# media's Fresnel interface (dielectric_dielectric), a complex index (dielectric_metal with REALRINDEX, IMAGINARYRINDEX)
G4_REFLECT = Interaction("g4.reflect", {**_CUTS, "p_spike": _FRAC, "p_lobe": _FRAC, "p_back": _FRAC,
                                        "sigma_alpha": Param("rad", POSITIVE), "polish": _FRAC},
                         _geant4_chain(_none), {GLOSSY: "facets", DIFFUSE: "lambert"},
                         {**_ZERO, **dict.fromkeys(("p_spike", "p_lobe", "p_back"), "zero"),
                          "polish": "support"})  # fmt: skip
G4_DIELECTRIC = Interaction("g4.dielectric", _CUTS, _geant4_chain(_real), defensive=_ZERO)
G4_COMPLEX = Interaction("g4.complex", {**_CUTS, "nr": _IDX, "nk": Param("", POSITIVE)}, _geant4_chain(_complex),
                         defensive=_ZERO)  # fmt: skip
MODELS = (G4_REFLECT, G4_DIELECTRIC, G4_COMPLEX)  # the kernel's branch order


# ------------------------------------------------------------------------------------------------------- the kernel
def _shares(get, f, ctx, models):
    """Return (the continuing shares (6,), the detected share, the amplitudes (8,)) of a row: one
    ``lax.switch`` over the ``models`` present (static; one model compiles no switch)."""  # noqa: D205, D209
    if len(models) == 1:
        return models[0].shares(get, f, ctx)
    return jax.lax.switch(f.model, [lambda _, m=m: m.shares(get, f, ctx) for m in models], None)


def kernel(get, f, ctx, models, any_per_normal=True):
    """Return the TypeEval of one surface row at ctx (the module docstring).

    ``get(column)``: the row's value of a table column at ctx's wavelength (``COLUMNS``); ``f``: its Flags; ``models``
    the Interactions the rows may take (static: one branch each); ``any_per_normal`` (static): a row may be read for a
    sensor model's QE.

    Notes
    -----
    Geant4's chain (G4OpBoundaryProcess::PostStepDoIt, DielectricMetal, DielectricDielectric, DoReflection): one draw
    against R, then R + T (TRANSMITTANCE, straight on); below R the reflection (or the two media's interface), above R +
    T the absorption, detected with eff (EFFICIENCY). Geant4 tests the absorption BEFORE Fresnel, so R < 1 on a
    dielectric_dielectric surface is not a reflectance. A dielectric_metal reflection is a spike (SPECULARSPIKECONSTANT
    p_spike), the facet lobe (SPECULARLOBECONSTANT p_lobe, width sigma_alpha; GLISUR's facet path of radius 1 -
    polish), a backscatter (BACKSCATTERCONSTANT p_back) or Lambertian (the rest); a frontpainted finish's is specular
    (f_spec) or Lambertian. At sigma_alpha = 0 G4OpBoundaryProcess::GetFacetNormal returns the global normal, so a lobe
    reflection is the spike's mirror (``normal``).

    A row read for a sensor model's QE (``per_normal``) deposits its detected share over A_n, its absorbed-and-detected
    share at normal incidence with eff 1 at the photon's wavelength and the crossing's indices (unpolarised: at normal
    incidence s and p are one).
    """
    pr, det, amps = _shares(get, f, ctx, models)
    sigma = jnp.where(f.glisur, 1.0 - get("polish"), get("sigma_alpha"))
    inv_an = F32(1.0)
    if any_per_normal:

        def over_normal(_):
            c = _normal_incidence(ctx)
            a_n = _shares(lambda q: F32(1.0) if q == "eff" else get(q), f, c, models)[1]
            return jnp.where(a_n > 0.0, 1.0 / jnp.where(a_n > 0.0, a_n, 1.0), 0.0).astype(F32)

        if isinstance(f.per_normal, bool):
            inv_an = over_normal(None) if f.per_normal else inv_an
        else:
            inv_an = jax.lax.cond(f.per_normal, over_normal, lambda _: F32(1.0), None)
    return TypeEval(pr.astype(F32), jnp.asarray(det, F32), jnp.asarray(sigma, F32), amps, inv_an)


def _normal_incidence(ctx):
    """Return the crossing context at normal incidence, unpolarised (at normal incidence s and p are one)."""
    c = {k: v for k, v in ctx.items() if k not in ("er", "ei")}
    c["wi"] = -ctx["n"]
    return c


def normal_absorptance(stype, p, ctx):
    """Return A_n: the share of ``stype`` (params ``p``) absorbed and detected at normal incidence with eff 1, at the
    context's wavelength and indices (``_normal_incidence``)."""  # noqa: D205, D209
    return stype.evaluate({**p, "eff": jnp.float32(1.0)}, _normal_incidence(ctx)).detect


# --------------------------------------------------------------------------------------------- the base types
def _const(v):
    return tuple(float(x) for x in np.ravel(v)) if np.ndim(v) else float(v)


def _base(name, params, outcomes, consts=(), model=G4_REFLECT, **kw):
    return SurfaceType(name, params, outcomes, tuple(sorted(consts.items())) if consts else (), model=model, **kw)


def _split_support(outs):
    """Return the support rule of a type whose reflection is split by f_spec into specular and diffuse: a fixed split
    leaves out the share it makes 0 (f_spec 0: no SPECULAR; f_spec 1: no DIFFUSE)."""  # noqa: D205, D209

    def support(c):
        fs = c.get("f_spec")
        gone = {SPECULAR} if fs == 0.0 else {DIFFUSE} if fs == 1.0 else set()
        return tuple(o for o in outs if o not in gone)

    return support


def _painted_support(c):
    """Return the outcomes of ``PAINTED_T`` its fixed constants leave possible: the split's rule, and no straight share
    with T fixed at 0."""  # noqa: D205, D209
    kept = _split_support((STRAIGHT, SPECULAR, DIFFUSE))(c)
    return tuple(o for o in kept if not (o == STRAIGHT and c.get("T") == 0.0))


def _lobes_support(c):
    """Return the outcomes of ``UNIFIED_METAL_NORMAL`` its fixed constants leave possible: no spike without p_spike
    and p_lobe, no backscatter without p_back, no Lambertian share when the three fixed constants take it all."""  # noqa: D205, D209
    outs = {SPECULAR, BACKSCATTER, DIFFUSE}
    if c.get("p_spike") == 0.0 and c.get("p_lobe") == 0.0:
        outs.discard(SPECULAR)
    if c.get("p_back") == 0.0:
        outs.discard(BACKSCATTER)
    if all(k in c for k in ("p_spike", "p_lobe", "p_back")) and c["p_spike"] + c["p_lobe"] + c["p_back"] >= 1.0:
        outs.discard(DIFFUSE)
    return tuple(o for o in (SPECULAR, BACKSCATTER, DIFFUSE) if o in outs)


PASS = _base("pass", (), (STRAIGHT,), {"T": 1.0})
ABSORBER = _base("absorber", (), ())
# dielectric_dielectric, polished: the Fresnel interface
DIELECTRIC = _base("dielectric", (), (REFRACT, SPECULAR), {"R": 1.0}, model=G4_DIELECTRIC)
# Geant4's polished dielectric_dielectric surface with REFLECTIVITY R and TRANSMITTANCE T tables (the same in
# WCSimOpBoundaryProcess with its photocathode model 0): below R the interface, between R and R + T straight on,
# above it absorbed; as a sensor the absorbed share is detected with eff (EFFICIENCY, or 1 where the sensitive
# detector counts every absorption)
DIELECTRIC_RT = _base("dielectric_rt", ("R", "T", "eff"), (REFRACT, SPECULAR, STRAIGHT),
                      model=G4_DIELECTRIC)  # fmt: skip
# a sensor window seen from a dielectric, with its complex index (nr, nk): specular reflection, the rest detected
PHOTOCATHODE = _base("photocathode", ("nr", "nk", "eff"), (SPECULAR,), {"R": 1.0},
                     model=G4_COMPLEX)  # fmt: skip
# a counting-only sensor (counted where the photon enters it), efficiency eff
COUNTING = _base("counting", ("eff",), ())
# dielectric_metal, glisur, finish ground, polish < 1: R into the GLISUR facet path (radius 1 - polish)
GLISUR_METAL = _base("glisur_metal", ("R", "polish", "eff"), (GLOSSY,), {"p_lobe": 1.0},
                     glossy="glisur")  # fmt: skip
# dielectric_metal, unified, any other (non-LUT) finish: spike, lobe, backscatter, Lambertian as Geant4's cuts
UNIFIED_METAL = _base("unified_metal", ("R", "p_spike", "p_lobe", "p_back", "sigma_alpha", "eff"),
                      (SPECULAR, BACKSCATTER, GLOSSY, DIFFUSE))  # fmt: skip
# dielectric_metal, unified, ground, sigma_alpha 0: the lobe reflects about the global normal, as the spike
UNIFIED_METAL_NORMAL = _base("unified_metal_normal", ("R", "p_spike", "p_lobe", "p_back", "eff"),
                             (SPECULAR, BACKSCATTER, DIFFUSE), normal=True,
                             support=_lobes_support)  # fmt: skip
# dielectric_dielectric, polishedfrontpainted (f_spec 1: R mirror) or groundfrontpainted (f_spec 0: R Lambertian), with
# TRANSMITTANCE T (straight); the absorbed rest is DoAbsorption's, detected with EFFICIENCY eff
# (G4OpBoundaryProcess::PostStepDoIt). Without T it is ``PAINTED``; a reflector is ``REFLECTOR``.
PAINTED_T = _base("painted_t", ("R", "T", "f_spec", "eff"), (STRAIGHT, SPECULAR, DIFFUSE),
                  alias=(("f_spec", "p_spike"),), support=_painted_support)  # fmt: skip

TYPES = (
    PASS,
    ABSORBER,
    DIELECTRIC,
    DIELECTRIC_RT,
    PHOTOCATHODE,
    COUNTING,
    GLISUR_METAL,
    UNIFIED_METAL,
    UNIFIED_METAL_NORMAL,
    PAINTED_T,
)


# ---------------------------------------------------------------------------------------------- derived structure
def transmitting(stype):
    """Return a dielectric_metal type ``stype`` with Geant4's TRANSMITTANCE ``T``: a straight share.

    G4OpBoundaryProcess::DielectricMetal draws once: below the reflectance R (REFLECTIVITY, or the complex index's
    CalculateReflectivity) the photon reflects as ``stype`` reflects it, below R + T it goes straight on unchanged
    (Transmission), and the rest is absorbed (DoAbsorption: detected with ``eff``): the kernel's cuts with T read.
    Into a medium without RINDEX the straight share is absorbed where it lands (``lumen.rules.boundary``).
    """

    def support(c):  # the base's own rule, and the straight share unless T is fixed at 0
        kept = stype.outcomes if stype.support is None else stype.support(c)
        return (*kept, *(() if c.get("T") == 0.0 else (STRAIGHT,)))

    return dataclasses.replace(stype, name=f"{stype.name}|T", params=(*stype.params, "T"),
                               outcomes=(*stype.outcomes, STRAIGHT), consts=tuple(kv for kv in stype.consts
                                                                                   if kv[0] != "T"),
                               support=support)  # fmt: skip


def _fixed_support(stype, consts):
    """Return the outcomes of ``stype`` with ``consts`` fixed that remain possible: the type's own ``support`` rule
    (structure: which shares a fixed constant makes 0), else every declared outcome."""  # noqa: D205, D209
    kept = set(stype.support(consts)) if stype.support is not None else set(stype.outcomes)
    return tuple(o for o in stype.outcomes if o in kept)


def fixed(stype, consts):
    """Return ``stype`` with the parameters in ``consts`` fixed at those values: a type whose parameters are the rest.

    A value the model never reads is not a parameter (the edge rule's OUT, decided by structure, never by the value:
    docs/explanation/parameters.md): a frontpainted finish's f_spec. They are the row's constants. The derived type is
    named ``type{p=v,...}`` and its support is the type's own rule (``_fixed_support``), so the edge floor never
    proposes an outcome the constants rule out.
    """
    consts = {k: _const(v) for k, v in consts.items()}
    if not consts:
        return stype
    tag = ",".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}=({','.join(f'{x:g}' for x in v)})"
                   for k, v in sorted(consts.items()))  # fmt: skip
    rule = None if stype.support is None else (lambda c: stype.support({**consts, **c}))
    return dataclasses.replace(
        stype, name=f"{stype.name}{{{tag}}}", params=tuple(q for q in stype.params if q not in consts),
        outcomes=_fixed_support(stype, consts), consts=tuple(sorted({**dict(stype.consts), **consts}.items())),
        support=rule)  # fmt: skip


def into_opaque(stype):
    """Return ``stype`` with its far-side outcomes absorbed: a frontpainted transmission into a medium without RINDEX.

    Geant4's frontpainted finishes skip the RINDEX check, so the photon is transmitted and killed at its next step
    (NoRINDEX): for the light it is an absorption where it lands. The parameters stay (T is read; its derivative is 0).
    """
    outs = tuple(o for o in stype.outcomes if not FAR_SIDE[o])
    return dataclasses.replace(stype, name=f"{stype.name}|into_opaque", outcomes=outs, far_opaque=True, support=None)


def no_detection(stype):
    """Return ``stype`` with its detection efficiency ``eff`` fixed at 0: the type of a surface that is no sensor.

    Geant4 reads EFFICIENCY on every surface; lumen counts a Detection only at a sensor, so on any other surface the
    absorbed share is absorbed (a type without ``eff`` is returned as it is).
    """
    if "eff" not in stype.params:
        return stype
    return fixed(stype, {"eff": 0.0})


# the types that give the same probabilities as one of ``TYPES`` with constants fixed are that type:
# Geant4's frontpainted finish without TRANSMITTANCE, a reflector without detection (R split by f_spec, the rest
# absorbed) and a mirror metal (dielectric_metal, glisur with polish 1 or unified polished: R mirror)
PAINTED = fixed(PAINTED_T, {"T": 0.0})
REFLECTOR = no_detection(PAINTED)
SPECULAR_METAL = fixed(UNIFIED_METAL_NORMAL, {"p_spike": 1.0, "p_lobe": 0.0, "p_back": 0.0})


def per_normal(stype):
    """Return the detecting type ``stype`` read for a sensor model that applies its QE after tracking.

    The model's QE is the detection at normal incidence (the datasheet's, per photon). The deposit at a crossing is the
    type's physical detected share; the evaluation carries 1 / A_n (``TypeEval.inv_an``; ``normal_absorptance``: its
    absorbed-and-detected share at normal incidence with eff 1, at the photon's wavelength and the crossing's indices),
    by which the model's readout divides it, so QE times that is the detection whatever the cathode's absorptance;
    QE <= A_n is checked by every run and at load (``lumen.build.values``). The outcome
    probabilities are the type's own: the photon is absorbed, reflected or transmitted as the cathode does.
    """
    if "eff" not in stype.params:
        raise ValueError(f"per_normal: a detecting type with eff, got {stype.name}")
    return dataclasses.replace(stype, name=f"{stype.name}|per_normal", per_normal=True)
