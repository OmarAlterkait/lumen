"""Value checks: what a value must satisfy for the detector's built structure to hold, one rule eager and traced.

    params, source, invalid = check(det, params, source)

The checks, each a predicate on the values (NumPy on concrete values, JAX on traced ones: the same expression):

- every parameter is admitted by its declared range (``Params.info(key).range``, the one predicate
  ``lumen._core.domains.admits``: finite and inside it), a table's range where
  light evaluates it: its knots inside the light's domain and its values at the domain's ends
  (``lumen.rules.optics.light_domain``; Geant4 reads a table nowhere else), finite everywhere;
- every table's knot energies are strictly ascending, and those the build packs are the detector's as packed, in
  float32 (structure, as its knot count: ``lumen.build.pack`` keeps them as constants);
- a sensor model that applies its QE after tracking on a cathode read for it detects no more than the cathode absorbs
  at normal incidence, QE <= A_n at every wavelength (``lumen.build.lookup.detection_margins``: both at the run's
  values, the model's and the cathode's and media's, so a calibration, a ``Params.set`` or a fit step that breaks it
  is refused or counted, as at load);
- every source field declared a direction (``lumen.sources.DIRECTION``) is a unit vector to ``UNIT_TOL``, every
  numeric source field is admitted by its declaration, and a source's wavelengths (``lam``, its spectra's
  supports) lie inside the light's domain, where the tables are checked.

A concrete value that fails raises ``InvalidValue`` naming the check, as the build refuses such a value in a file. A
traced value (``jax.jit``, ``jax.vmap``, ``jax.grad``) cannot raise: ``invalid`` (int32, ``Light.health.invalid``)
counts its failing elements, and its derivative there is NaN (an identity whose tangent is multiplied by 1 where the
value passes, NaN where it fails), so a derivative at a value the engine does not model is never a silent number. A
value that passes is untouched, bit for bit, and a symbolic zero tangent stays symbolic.
"""

from __future__ import annotations

import copy
import hashlib
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from lumen._core.domains import admits
from lumen._core.safe import traced
from lumen.format.calibrations import Table
from lumen.params import Params
from lumen.rules.choices import UNIT_TOL
from lumen.sources._base import DIRECTION
from lumen.units import HC


class InvalidValue(ValueError):  # noqa: N818 (the documented error catalogue's name)
    """A value outside what the detector's built structure holds (``lumen.build.values``)."""


def _numeric(v) -> bool:
    return traced(v) or np.asarray(v).dtype.kind in "fiu"


class _Values:
    """The values under check, by slot: (key, "energy" | "value") of a parameter, ("source", field) of a source."""

    def __init__(self, params, source):
        self.params, self.source = params, source
        self.masks: dict = {}  # slot -> pass mask (traced slots only)
        self.count = jnp.int32(0)

    def get(self, slot):
        a, b = slot
        v = getattr(self.source, b) if a == "source" else self.params[a]
        if a == "source":
            return v.value if isinstance(v, Table) else v
        return (v.energy if b == "energy" else v.value) if isinstance(v, Table) else v

    def test(self, slots, ok, message, extra=()):
        """Apply one check: ``ok(xp, *values, *extra)`` returns a pass mask per slot (broadcastable to its value);
        ``extra``: values it also reads that carry no mask (a reference's)."""  # noqa: D205, D209
        vals = [self.get(s) for s in slots]
        if not any(traced(v) for v in (*vals, *extra)):
            masks = ok(np, *(np.asarray(v, np.float64) for v in (*vals, *extra)))
            if not all(np.all(m) for m in masks):
                raise InvalidValue(message(*vals))
            return
        masks = ok(jnp, *(jnp.asarray(v) for v in (*vals, *extra)))
        self.count = self.count + jnp.sum(~jnp.asarray(masks[0]), dtype=jnp.int32)
        for s, v, m in zip(slots, vals, masks, strict=True):
            if traced(v):
                m = jnp.broadcast_to(m, jnp.shape(v))
                self.masks[s] = m if s not in self.masks else self.masks[s] & m

    def require(self, ok, message):
        """Apply one check of the values as a whole: ``ok`` a boolean scalar computed from them; failing, a concrete
        one raises ``message()`` and a traced one counts once and makes every traced value's derivative NaN."""  # noqa: D205, D209
        if not traced((ok, self.params._values)):
            if not bool(ok):
                raise InvalidValue(message())
            return
        self.count = self.count + jnp.where(ok, 0, 1).astype(jnp.int32)
        for k in self.params:
            v = self.get((k, "value"))
            if traced(v):
                m = jnp.broadcast_to(ok, jnp.shape(v))
                self.masks[(k, "value")] = m if (k, "value") not in self.masks else self.masks[(k, "value")] & m

    def result(self):
        """Return (params, source) with each traced slot's derivative NaN where it fails."""
        vals = dict(self.params._values)
        out = self.source
        for (a, b), m in self.masks.items():
            if a == "source":
                out = copy.copy(out) if out is self.source else out
                v = getattr(out, b)
                new = Table(v.energy, _within(jnp.asarray(v.value), m)) if isinstance(v, Table) else \
                    _within(jnp.asarray(v), m)  # fmt: skip
                object.__setattr__(out, b, new)
                continue
            v = vals[a]
            if isinstance(v, Table):
                vals[a] = Table(_within(jnp.asarray(v.energy), m), v.value) if b == "energy" else \
                    Table(v.energy, _within(jnp.asarray(v.value), m))  # fmt: skip
            else:
                vals[a] = _within(jnp.asarray(v), m)
        return Params(vals, self.params._spec, self.params._layers), out


def _inside(lo, hi):
    return lambda xp, v: (admits(xp, v, lo, hi),)


def _table_inside(lo, hi, dom):
    """Pass a table (values v at knot energies e) inside [lo, hi] where light evaluates it: its knots inside the
    domain ``dom`` (eV intervals) and its values at the domain's ends (held at its end values beyond its knots); every
    value finite."""  # noqa: D205, D209

    def ok(xp, v, e):
        good = admits(xp, v, lo, hi)
        if dom is None:
            return good, True
        d = np.asarray(dom, np.float64).reshape(-1, 2)
        inside = xp.zeros(np.shape(e), bool)
        for a, b in d:
            inside = inside | ((e >= a) & (e <= b))
        ends = xp.interp(xp.asarray(d.ravel()), e, v)
        return (good | ~inside) & xp.isfinite(v) & xp.all(admits(xp, ends, lo, hi)), True

    return ok


def _in_domain(dom):
    """Pass wavelengths (nm) inside the domain ``dom`` (eV intervals)."""
    d = np.asarray(dom, np.float64).reshape(-1, 2)

    def ok(xp, lam):
        e = HC / lam
        inside = xp.zeros(np.shape(lam), bool)
        for a, b in d:
            inside = inside | ((e >= a) & (e <= b))
        return (inside,)

    return ok


def _support_in_domain(dom):
    """Pass a spectrum table whose support (knot intervals of positive mass) lies inside one domain interval."""
    d = np.asarray(dom, np.float64).reshape(-1, 2)

    def ok(xp, v, e):
        m = (v[1:] + v[:-1]) * (e[1:] - e[:-1]) > 0
        inside = xp.zeros(np.shape(m), bool)
        for a, b in d:
            inside = inside | ((e[:-1] >= a) & (e[1:] <= b))
        return (xp.all(~m | inside),)

    return ok


def _ascending(xp, e):
    return (xp.concatenate([xp.ones((1,), bool), e[1:] > e[:-1]]),)


def _f32(xp, a):
    return xp.asarray(a).astype(np.float32)  # values compare as the engine packs them


def _same_knots(e0):
    def ok(xp, e):
        same = xp.all(_f32(xp, e) == _f32(xp, e0)) if np.shape(e) == np.shape(e0) else xp.asarray(False)
        return (xp.broadcast_to(same, np.shape(e)),)

    return ok


def _unit(tol):
    return lambda xp, d: (xp.abs(xp.sqrt(xp.sum(d * d, axis=-1, keepdims=True)) - 1.0) <= tol,)


def _within_support(xp, e1, v1, e2, v2):
    """Pass where every knot interval of the live table (e1, v1) with mass lies where the reference's (e2, v2) has
    mass: Geant4's sampler of the reference proposes every energy the live spectrum emits."""  # noqa: D205, D209
    m1 = (v1[1:] + v1[:-1]) * (e1[1:] - e1[:-1]) > 0
    m2 = (v2[1:] + v2[:-1]) * (e2[1:] - e2[:-1]) > 0
    meet = xp.maximum(e1[:-1, None], e2[None, :-1]) < xp.minimum(e1[1:, None], e2[None, 1:])
    inside = (e1[:-1] >= e2[0]) & (e1[1:] <= e2[-1])
    ok = xp.all(~(m1[:, None] & meet & ~m2[None, :])) & xp.all(~m1 | inside)
    return ok, ok


def check(det, params, source=None, reference=None):
    """Return (params, source, invalid): every value checked (see the module docstring); ``det`` None: no tables.

    With a ``reference`` (``simulate(reference=)``), each spectrum the source draws (``Source.spectra``) must lie
    inside the reference's support: a wavelength is drawn from the reference's table and weighted to the live one,
    which cannot reach energies the reference never proposes.
    """
    w = _Values(params, source)
    dom = None if det is None else det.parts.library.domain
    _check_params(w, params, {} if det is None else det.parts.knots, dom)
    if det is not None:
        _check_detection(w, det, params)
    if source is not None:
        _check_source(w, params, source, dom)
    if source is not None and reference is not None:
        _check_support(w, params, source, reference)
    p, s = w.result()
    return p, s, w.count


def _check_support(w, params, source, reference):
    rv = source.values(reference)
    for sp in source.spectra():
        live = params[sp] if "/" in sp else getattr(source, sp)
        ref = reference[sp] if "/" in sp else rv[sp]
        slot = (sp, "value") if "/" in sp else ("source", sp)
        w.test([slot], lambda xp, v, e, er, vr: _within_support(xp, e, v, er, vr)[:1],
               lambda v, sp=sp: f"{sp}: the live spectrum emits where the reference's has no mass: drawn from the "
               "reference, it cannot be weighted to the live one (move the reference)",
               extra=(live.energy, ref.energy, ref.value))  # fmt: skip


def _check_params(w, params, packed, dom=None):
    for k in params:
        if k not in params._spec or not _numeric(w.get((k, "value"))):
            continue
        lo, hi = params.info(k).range
        if isinstance(params[k], Table) and k in packed:
            w.test([(k, "energy")], _same_knots(packed[k]), lambda e, k=k: f"{k}: the knot energies are the "
                   "detector's (structure, as the knot count: new knots are a new detector file)")  # fmt: skip
        elif isinstance(params[k], Table):
            w.test([(k, "energy")], _ascending, lambda e, k=k: f"{k}: the knot energies must be strictly ascending")
        if isinstance(params[k], Table):
            w.test([(k, "value"), (k, "energy")], _table_inside(lo, hi, dom), lambda v, e, k=k, lo=lo, hi=hi:
                   f"{k} has a value outside [{lo}, {hi}] {params.info(k).unit} where light evaluates it (its "
                   "knots inside the light's domain, its values at the domain's ends), or one not finite")  # fmt: skip
        else:
            w.test([(k, "value")], _inside(lo, hi), lambda v, k=k, lo=lo, hi=hi:
                   f"{k} has a value outside [{lo}, {hi}] {params.info(k).unit} or not finite")  # fmt: skip


_PASSED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # detector -> digests of concrete values that passed


def _check_detection(w, det, params):
    from lumen.build import lookup, pack
    from lumen.sensors import normal_detection

    digest = None
    if not traced(params._values):  # a concrete check that passed once passes again: not repeated per run
        h = hashlib.sha256(repr(sorted(params._values)).encode())
        for x in jax.tree_util.tree_leaves([params._values[k] for k in sorted(params._values)]):
            h.update(np.ascontiguousarray(np.asarray(x, np.float64)).tobytes())
        digest = h.digest()
        if digest in _PASSED.setdefault(det, set()):
            return
    readout = normal_detection(det.sensor_set, det.parts, params)
    if readout:
        margins = list(lookup.detection_margins(det.parts.library, det.parts.optics, readout, pack.named(det, params)))
        ok = jnp.all(jnp.asarray([jnp.all(d <= a * (1.0 + lookup.PER_NORMAL_RTOL)) for *_, d, a in margins]))
        w.require(ok, lambda: lookup.detection_refusal(det.parts.optics, margins))
    if digest is not None:
        _PASSED[det].add(digest)


def _check_source(w, params, source, dom=None):
    def unit(slot, name):
        w.test([slot], _unit(UNIT_TOL), lambda d: f"{name}: a direction is a unit vector (|d| = 1 to {UNIT_TOL})")

    for f, decl in source.FIELDS.items():
        name = f"{type(source).__name__}.{f}"
        if source.prefix + f in params:  # a value given by a key: its range is checked as the key's
            if decl is DIRECTION:
                unit((source.prefix + f, "value"), name)
            continue
        if getattr(source, f, None) is None or not _numeric(w.get(("source", f))):
            continue
        if decl is DIRECTION:
            unit(("source", f), name)
        w.test([("source", f)], _inside(*decl.range), lambda v, name=name, r=decl.range:
               f"{name} has a value outside [{r[0]}, {r[1]}] or not finite")  # fmt: skip
        if f == "lam" and dom is not None:
            w.test([("source", f)], _in_domain(dom), lambda v, name=name: f"{name}: a wavelength outside the "
                   f"light's domain ({_nm(dom)}: the detector's emitters' supports), where its tables are not "
                   "checked")  # fmt: skip
    if dom is not None:
        for sp in source.spectra():
            t = params[sp] if "/" in sp else getattr(source, sp, None)
            if "/" in sp or not isinstance(t, Table):
                continue  # a material's own spectrum is part of the domain
            w.test([("source", sp)], _support_in_domain(dom), lambda v, sp=sp: f"{sp}: the spectrum emits outside the "
                   f"light's domain ({_nm(dom)}: the detector's emitters' supports), where its tables are not checked",
                   extra=(t.energy,))  # fmt: skip


def _nm(dom):
    return ", ".join(f"{HC / b:.1f}-{HC / a:.1f} nm" for a, b in np.asarray(dom, np.float64).reshape(-1, 2))


def _within_rule(primals, tangents):
    a, ok = primals
    t = tangents[0]
    if isinstance(t, jax.custom_derivatives.SymbolicZero):  # no tangent: none made (the guards after it see none)
        return a, t
    return a, t * jnp.where(ok, jnp.ones_like(a), jnp.full_like(a, jnp.nan))


_within = jax.custom_jvp(lambda a, ok: a)  # ``a``; its tangent times 1 where ``ok``, NaN elsewhere
_within.defjvp(_within_rule, symbolic_zeros=True)
