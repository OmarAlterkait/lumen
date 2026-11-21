"""What every source shares: its parameters, the guard on its moving fields, its random numbers, its spectra.

A source is a frozen dataclass whose ``FIELDS`` are parameters ``source/<field>``, traced: ``values(params)``
takes each from ``params`` where it has the key, else from the source, plus the material values it ``reads``.
A source's ``label`` (int32: one for all its photons, or one per row of its rows, as its per-row fields) is each
photon's truth label (``Photons.label``): the truth rows carry it and an event axis reads it as the event. Labels are
not parameters and carry no derivative.
``photon_uniforms``: photon i's uniforms in (0, 1) (``lumen._core.rng.uniforms``) from its own key ``fold_in(key, i)``
at vertex 0, as the transport draws them: lumen's own sources draw every photon so (one random scheme); ``gaussian`` of
uniforms is ``jax.random.normal`` of the same bits, bit for bit. Every random choice (a wavelength from a spectrum, a
scintillation component, a Geant4 step) is made at the reference values (``emit(..., reference=)``, default the run's
own) and weighted to the live ones (``lumen._core.safe.ratio``), the transport's rule.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from lumen._core import rng
from lumen._core.errors import refusing
from lumen._core.safe import ratio, traced
from lumen.media import spectra
from lumen.params import NONE, PENDING, Param, Params
from lumen.rules.choices import DEFENSIVE_SHARE
from lumen.units import HC, INF

F32, PREFIX = jnp.float32, "source/"
POSITION = Param("m", (-INF, INF), method="moving_paths")
DIRECTION = Param("", (-1.0, 1.0), method="moving_paths")
TIME = Param("ns", (-INF, INF), method="reweight")
COUNT = Param("", (0.0, INF), method="reweight")
ROW_COUNT = "rows"  # the prepared values' entry holding a one-photon-per-row source's row count
NORMAL_LO = np.nextafter(np.float32(-1.0), np.float32(0.0))  # jax.random.normal's lower bound
sg = jax.lax.stop_gradient


@dataclasses.dataclass(frozen=True)
class Source:
    """The base of every source; ``FIELDS``: {field: lumen.Param}, ``MOVING``: those that move it (``Param.moves``).

    ``prefix``: its keys' prefix; ``check(params)``: checks run once before a simulation. A tangent on a ``MOVING``
    field is refused (``GradientNotAvailable``): a source-position derivative is not in this version.
    """

    FIELDS: ClassVar[dict] = {}
    prefix: ClassVar[str] = PREFIX
    label: Any = dataclasses.field(default=None, kw_only=True)

    @property
    def MOVING(self) -> tuple:  # noqa: N802 -- the Source protocol's name
        """Return the fields that move where the light starts (declared ``method="moving_paths"``): no derivative."""
        return tuple(f for f, p in self.FIELDS.items() if p.moves)

    def check(self, params=None) -> None:
        """Check the source before a simulation (a source function's output at first use); its values are checked
        with every value (``lumen.build.values``: a direction is a unit vector to ``UNIT_TOL``)."""  # noqa: D205, D209

    def spectra(self) -> tuple:
        """Return the spectra the source draws wavelengths from: its fields (a name) or material keys (with a "/")."""
        return ()

    def reads(self) -> tuple:
        """Return the keys of the material values the source reads from the parameters."""
        return ()

    def values(self, params=None) -> dict:
        """Return {field: value} and {material key: value}, float32 (see the module docstring)."""
        p = {} if params is None else params
        out = {f: p[PREFIX + f] if PREFIX + f in p else getattr(self, f) for f in self.FIELDS}
        for k in self.reads():
            if k not in p:
                raise KeyError(f"{type(self).__name__} reads {k!r}: pass the detector's parameters")
            out[k] = p[k]
        seq = lambda a: isinstance(a, (tuple, list))  # noqa: E731 -- a vector given as a tuple is one value
        # a host array is cast on the host: no device conversion compiled per array shape
        f32 = lambda a: jnp.asarray(a.astype(np.float32) if isinstance(a, np.ndarray) else a, F32)  # noqa: E731
        return jax.tree_util.tree_map(f32, {k: v for k, v in out.items() if v is not None}, is_leaf=seq)

    def refuse_overridden(self, params=None) -> None:
        """Refuse a ``source/<field>`` key of ``params`` that holds another value than the source's field where the
        key would replace it silently: a field traced on the source (its derivative lost), or a key ``parameters()``
        added (the copy of a field, so another source's). A key set (``Params.set``, a dict) is the value given; the
        key holding the field itself is one value. ``simulate`` and ``emit`` call it."""  # noqa: D205, D209
        p = {} if params is None else params

        def copied(k):  # a key parameters() added (its layer), not one set since
            return isinstance(p, Params) and p.explain(k).startswith("source ")

        stale = [f for f in self.FIELDS if PREFIX + f in p and (traced(getattr(self, f, None)) or copied(PREFIX + f))
                 and not _same(p[PREFIX + f], getattr(self, f))]  # fmt: skip
        if stale:
            raise ValueError(f"{type(self).__name__}: {stale} are given as {[PREFIX + f for f in stale]} in the "
                             "parameters with other values than the source's (a traced field, or the keys of "
                             "another source's parameters()): give each value once (set the key, or pass the source "
                             "its value)")  # fmt: skip

    def parameters(self, params=None):
        """Return ``params`` (a ``lumen.Params``, default empty) with this source's ``source/`` keys added."""
        fs = [f for f in self.FIELDS if getattr(self, f) is not None]
        spec = {PREFIX + f: self.FIELDS[f] if self._differentiable(f) or not self.FIELDS[f].available else
                dataclasses.replace(self.FIELDS[f], method=NONE) for f in fs}  # fmt: skip
        base = Params({}, {}, {}) if params is None else params
        return base.merged({PREFIX + f: getattr(self, f) for f in fs}, spec, f"source {type(self).__name__}")

    def _differentiable(self, f) -> bool:
        return self.FIELDS[f].available and f not in self.MOVING

    def emit(self, params, key, n: int, first_id=0, samples=None, *, reference=None):
        """Return the Photons of photons ``first_id .. first_id + n - 1`` of ``samples`` (default n).

        Their weights sum in expectation to the source's photons over the ``samples``. A tangent on a ``MOVING`` field
        or on a field declared without a derivative (method "none") raises ``GradientNotAvailable``. Every random choice
        (a wavelength from a spectrum, a component, a step) is drawn at ``reference`` (values of the same keys and
        fields; default: ``params``, the run's own) and weighted to the live values.
        """
        self.refuse_overridden(params)
        return self._emit(self._prepared(params, reference), key, n, first_id, samples)

    # {field: the rank of one row's value}: the fields that may carry one row per item (a photon, a step), padded to a
    # power of two; a field with one more axis than its rank has rows, and every such field (and a label array) has
    # the same number of them (refused otherwise, at construction and when the values come from the parameters)
    ROWS: ClassVar[dict] = {}
    LIGHTLESS: ClassVar[tuple] = ()  # of those, the fields zeroed in the padding rows so that they emit no light
    # a row is one photon: the source emits exactly its rows, each once (``photon_rows``), and has no sample count of
    # its own; its padding rows (past the last, up to the power of two) weigh 0. Else a row is an item photons are
    # drawn from (a step), ``samples`` of them
    PHOTON_ROWS: ClassVar[bool] = False
    REQUIRES_ROWS: ClassVar[bool] = False  # a source that is nothing but its rows (``FromArrays``)

    def __post_init__(self):
        name, label = type(self).__name__, None if self.label is None else _labels(self.label)
        if label is not None and label.ndim and not self.ROWS:
            raise ValueError(f"{name}: a label per row needs a source with rows (give one label for all its photons)")
        n = _row_count(name, {f: getattr(self, f) for f in self.ROWS if getattr(self, f, None) is not None}
                       | ({} if label is None else {"label": label}), self._ranks())  # fmt: skip
        if self.PHOTON_ROWS and self.REQUIRES_ROWS and n is None:
            raise ValueError(f"{name}: one photon per row, and no field has rows (give (N, ...) arrays)")

    def _ranks(self) -> dict:
        return {**self.ROWS, "label": 0} if self.ROWS else {}

    def photon_rows(self, params=None) -> int | None:
        """Return the photons a source of one photon per row emits (its rows), None for a source that draws them."""
        if not self.PHOTON_ROWS:
            return None
        v = self.values(params) | ({} if self.label is None else {"label": _labels(self.label)})
        return _row_count(type(self).__name__, v, self._ranks())

    def _prepared(self, params, reference=None):
        """Return (the static structure, the guarded values with their ``ROWS`` padded, the reference's values or
        None): what ``_EMIT`` takes."""  # noqa: D205, D209
        v = self.values(params)
        name = type(self).__name__
        held = {k: refusing(_refusal(f"{name}.{k}", d)) for k, d in self.FIELDS.items() if not self._differentiable(k)}
        v = {k: held[k](a) if k in held else a for k, a in v.items()}
        s = copy.copy(self)  # the structure: the static part of the jitted emission
        for f in (*self.FIELDS, "label"):
            object.__setattr__(s, f, None)
        if self.label is not None:  # an int32 per photon or per row, never a parameter
            v = v | {"label": _labels(self.label)}
        n = _row_count(name, v, self._ranks())
        if n is not None and self.PHOTON_ROWS:  # photon i >= n is padding: weight 0 (``_draw``)
            v = v | {ROW_COUNT: np.int32(n)}
        ref = None if reference is None else _pad_rows(self.values(reference), self._ranks(), self.LIGHTLESS)
        return s, _pad_rows(v, self._ranks(), self.LIGHTLESS), ref

    @staticmethod
    def _emit(prepared, key, n, first_id, samples):
        s, v, ref = prepared
        key = jax.random.key_data(key) if jnp.issubdtype(jnp.asarray(key).dtype, jax.dtypes.prng_key) else key
        return _EMIT(s, v, ref, key, jnp.asarray(first_id, jnp.int32), int(n), jnp.asarray(samples or n, F32))

    def draw(self, v, ref, key, n, first_id, samples):
        """Return the Photons from the values ``v``, its random choices made at the reference values ``ref``."""
        raise NotImplementedError

    def emitter(self, params, key, samples: int, *, reference=None):
        """Return ``rows(first_id, n)``: photons first_id .. first_id + n - 1 of one run of ``samples`` photons.

        ``simulate`` reads a run's photons through it, chunk by chunk; photon i is the same row whatever the chunks.
        A source that draws all its photons at once (``FunctionSource``) draws them here, once per run.
        """
        prepared = self._prepared(params, reference)  # once per run: the rows padded once
        rows = prepared[1].get(ROW_COUNT)
        if rows is not None and int(samples) != int(rows):
            raise ValueError(f"{type(self).__name__} emits its {int(rows)} rows, one photon each: simulate it with "
                             f"samples={int(rows)} (or None), not {int(samples)}")  # fmt: skip
        return lambda first_id, n: self._emit(prepared, key, n, first_id, samples)


def _same(a, b) -> bool:
    """Return whether a key's value is a field's: the same object, or equal concrete values in float32, as the source
    reads them (``values``; a table's knots too)."""  # noqa: D205, D209
    if a is b:
        return True
    if traced((a, b)) or (a is None) != (b is None):
        return False
    la, lb = (jax.tree_util.tree_leaves(v, is_leaf=lambda x: isinstance(x, (tuple, list))) for v in (a, b))
    return len(la) == len(lb) and all(
        np.array_equal(np.asarray(x, np.float32), np.asarray(y, np.float32)) for x, y in zip(la, lb, strict=True)
    )


def _draw(s, v, ref, key, first_id, n, samples):  # noqa: PLR0913
    ref = sg(v if ref is None else ref)  # every choice is drawn at the reference: the run's own values by default
    e = s.draw(v, ref, key, n, first_id, samples)
    if ROW_COUNT in v:  # one photon per row: the rows past the last (the padding) weigh nothing
        ids = first_id + jnp.arange(n, dtype=jnp.int32)
        e = e._replace(w=jnp.where(ids < v[ROW_COUNT], e.w, 0.0))
    if "label" in v and e.label is None:  # a label per photon row (or one for all), as the per-row fields
        lab = v["label"]
        ids = first_id + jnp.arange(n, dtype=jnp.int32)
        e = e._replace(label=jnp.broadcast_to(lab, (n,)) if lab.ndim == 0 else
                       jnp.take(lab, ids, mode="fill", fill_value=-1))  # fmt: skip
    return e


_EMIT = jax.jit(_draw, static_argnums=(0, 5))


def _labels(label):
    """Return a source's labels as int32 (a host array stays on the host until padded)."""
    a = np.asarray(label)
    if not np.issubdtype(a.dtype, np.integer) or a.ndim > 1:
        raise TypeError(f"a source's label is an integer or a 1-d integer array, got {a.dtype} {a.shape}")
    return a.astype(np.int32)


def bucket(n: int) -> int:
    """Return the power of two >= n (>= 1): the sizes compiled programs are built for, so few shapes compile."""
    return 1 << max(int(n) - 1, 0).bit_length()


def _row_count(name, v, ranks) -> int | None:
    """Return the rows of the fields ``ranks`` ({field: rank of one row}) that have them, None when none has; refuse
    fields of unequal row counts, or with more than one row axis."""  # noqa: D205, D209
    n = {}
    for f, r in ranks.items():
        a = v.get(f)
        shape = () if a is None else tuple(jnp.shape(a)) if hasattr(a, "shape") else np.shape(a)
        if len(shape) > r + 1:
            raise ValueError(f"{name}.{f}: one row axis at most, got shape {shape}")
        if len(shape) == r + 1:
            n[f] = int(shape[0])
    if len(set(n.values())) > 1:
        raise ValueError(f"{name}: its per-row fields have different numbers of rows {n}: give each the same rows "
                         "(or one value for all)")  # fmt: skip
    return next(iter(n.values()), None)


def _pad_rows(v, ranks, lightless):
    """Return ``v`` with the row fields (``ranks``: {field: rank of one row}) padded to ``bucket`` rows: the last row
    repeated, ``lightless`` fields zero. The row fields have one row count (``_row_count``)."""  # noqa: D205, D209
    have = [f for f, r in ranks.items() if f in v and jnp.ndim(v[f]) > r]
    if not have:
        return v
    n = int(jnp.shape(v[have[0]])[0])
    m = bucket(n)
    if m == n:
        return v
    out = dict(v)
    for f in have:
        a = v[f]
        if isinstance(a, jax.core.Tracer):  # traced (a derivative through the rows): pad in JAX
            tail = jnp.zeros_like(a[-1:]) if f in lightless else a[-1:]
            out[f] = jnp.concatenate([a, jnp.repeat(tail, m - n, 0)])
        else:  # on the host: no program compiled per row count
            h = np.asarray(a)
            tail = np.zeros_like(h[-1:]) if f in lightless else h[-1:]
            out[f] = jnp.asarray(np.concatenate([h, np.repeat(tail, m - n, 0)]))
    return out


# the one refusal of a source-position derivative (a tangent on a field that moves where the light starts)
MOVES = (
    "moves where the light starts: a source-position derivative is not available in this version "
    "(docs/explanation/known-limits.md)"
)


def _refusal(name, declared=None) -> str:
    """Return why the field ``name`` (its declared Param ``declared``) has no derivative: the guard's message."""
    method = None if declared is None else declared.method
    why = ("is a bound of the wavelength range the photons are drawn from: no derivative (calibrate it by value)"
           if method == "none" else
           f"has a derivative whose method ({method}) is not available in this version: calibrate it by value "
           "(docs/explanation/known-limits.md)" if method in PENDING else
           MOVES)  # fmt: skip
    return f"{name} {why}"


def lit(i, e):
    """Return the draw ``i`` (an index into the shares ``e``) moved to the first share above zero at or after it.

    A draw by searching a cumulative sum can land on a share of zero: the device's cumsum (a parallel scan) is not
    flat across zero entries (it can step by one float32 ulp there), and a normalised one can end below 1. Moving the
    draw to the next share with light moves at most that rounding's weight; a share of zero is never drawn.
    """
    ns = e.shape[0]
    on = e > 0.0
    nxt = jax.lax.cummin(jnp.where(on, jnp.arange(ns), ns)[::-1])[::-1]  # the first share above zero at or after
    last = ns - 1 - jnp.argmax(on[::-1])
    return jnp.minimum(nxt[jnp.clip(i, 0, ns - 1)], last)


def photon_uniforms(key, k: int, n: int, first_id=0):
    """Return (n, k): photon first_id + i's k uniforms at vertex 0 of its own key."""
    ids = jnp.asarray(first_id, jnp.int32) + jnp.arange(n, dtype=jnp.int32)
    return jax.vmap(lambda i: rng.uniforms(rng.vertex_key(rng.photon_key(key, i), 0), k))(ids)


def gaussian(rand):
    """Return standard normals from uniforms (``rng.uniforms``): ``jax.random.normal``'s map of the same bits."""
    f = rand - F32(2.0**-24)  # the bits' float in [0, 1), exact
    v = jnp.maximum(NORMAL_LO, f * (np.float32(1.0) - NORMAL_LO) + NORMAL_LO)
    return np.float32(np.sqrt(2.0)) * jax.lax.erf_inv(v)


def rows(a, n, first_id, rank):
    """Return a field's values for photons first_id .. first_id + n - 1: one per photon if it has a photon axis.

    Rows past the padded end read zeros, never a clamped (shifted) slice; a one-photon-per-row source gives them
    weight 0 (``_draw``), a source drawing from its rows never draws a lightless padding row.
    """
    a = jnp.asarray(a, F32)
    if a.ndim <= rank:
        return jnp.broadcast_to(a, (n,) + a.shape)
    return jnp.take(a, first_id + jnp.arange(n, dtype=jnp.int32), axis=0, mode="fill", fill_value=0)


def wavelength(live, ref, rand):
    """Return (lam (nm), weight p(lam; live) / q(lam)) of a spectrum (``lumen.media.spectra``): drawn by Geant4's
    sampler of the reference table under the one defensive mixture (``spectra.covered``: its knot intervals without
    mass covered), weighted to the live one (``lumen._core.safe.ratio``: exactly 1 at the reference where nothing is
    covered, its derivative the score; the live density is 0 outside its table)."""  # noqa: D205, D209
    e_r, v_r = sg(ref.energy), sg(ref.value)
    lam = sg(spectra.sample(spectra.covered(e_r, v_r, DEFENSIVE_SHARE), rand))
    energy = HC / lam
    q = spectra.covered_density(e_r, v_r, energy, DEFENSIVE_SHARE)
    return lam, ratio(spectra.density(live.energy, live.value, energy), q)
