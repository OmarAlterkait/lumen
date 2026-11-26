"""A source as one user function: ``@lumen.source(params=...)`` over ``f(v, key, n) -> lumen.Photons``.

    @lumen.source(params={"x": Param("m", shape=(3,), method="moving_paths", default=(0, 0, 0.2)),
                          "m": Param("", (0, 50), default=4.0)})
    def led(v, key, n):                 # v: the parameter values (traced), key: a JAX key, n: a static count
        ...
        return lumen.Photons(x=x, d=d, t=t, lam=lam, w=w * lumen.sources.reweight(logp))

    p = led.parameters(det.params())
    d = jax.jvp(lambda m: lumen.simulate(det, led, p.set("source/led/m", m), samples=1 << 20, seed=0).arrivals,
                (m0,), (dm,))[1]        # a tangent on a reweighted parameter (one on x is refused: it moves the light)

The function returns ``n`` photons whose weights sum, in expectation, to the photons the source stands for; ``simulate``
calls it once per run with n = samples and runs its photons in chunks (photon i is the same row whatever the chunk
size). Its parameters are the keys ``source/<name>/<field>`` (``name``: the function's, unless given), taken from the
``params`` given to ``simulate`` where they are set, else the declared defaults (``FunctionSource.parameters`` adds them
to a ``Params``).

A parameter that moves where the light starts (declared ``method="moving_paths"``: a position) has no derivative in
this version: a tangent on it is refused, and so is any tangent reaching the photons' x, d or lam.
A density that depends on a parameter (an axis, a width) enters by keeping the directions and weighting with
``reweight(logp)``; a wavelength or intensity parameter enters by ``reweight``. Checks at first use: the output's type,
shapes and dtypes, no host callbacks, and one draw at the defaults (finite, unit directions, weights not negative).
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from lumen._core import frames
from lumen._core.checks import callbacks
from lumen._core.errors import GradientNotAvailable, refusing
from lumen._core.safe import score, traced
from lumen._core.static import Static
from lumen.params import Param, Params, declared
from lumen.record import Photons
from lumen.rules.choices import SOURCE_CHECK_N, UNIT_TOL
from lumen.sources._base import MOVES, PREFIX, Source

F32 = jnp.float32
sg = jax.lax.stop_gradient


class SourceError(ValueError):
    """A source function whose output breaks its declaration (shapes, dtypes, a host callback)."""


def concat(*parts: Photons) -> Photons:
    """Return the photons of several pieces, in order (several emitters in one source).

    Each piece's weights stay its own (scale them to its share of the photons).
    """
    if not parts:
        raise ValueError("concat: no photons")
    pol = any(p.pol is not None for p in parts)
    lab = any(p.label is not None for p in parts)
    out = []
    for p in parts:
        n = jnp.shape(p.x)[0]
        q = Photons(**{f: jnp.asarray(getattr(p, f), F32) for f in ("x", "d", "t", "lam", "w")})
        q = q._replace(pol=None if not pol else jnp.zeros((n, 2, 3), F32) if p.pol is None else p.pol)
        q = q._replace(label=None if not lab else jnp.full(n, -1, jnp.int32) if p.label is None else
                       jnp.asarray(p.label, jnp.int32))  # fmt: skip
        out.append(q)
    return Photons(**{k: None if getattr(out[0], k) is None else jnp.concatenate([getattr(q, k) for q in out])
                      for k in Photons._fields})  # fmt: skip


reweight = score  # the score weight (lumen._core.safe.score), as a source function writes it


def cone_about(axis, c, phi):
    """Return (n, 3) unit directions at cosine ``c`` and azimuth ``phi`` (rad) about ``axis`` (normalised here)."""
    a = jnp.asarray(axis, F32)
    a = a / jnp.linalg.norm(a)
    e1, e2 = frames.basis(a)
    c = jnp.asarray(c, F32)
    s = jnp.sqrt(jnp.clip(1.0 - c * c, 0.0, 1.0))
    return (s * jnp.cos(phi))[:, None] * e1 + (s * jnp.sin(phi))[:, None] * e2 + c[:, None] * a


def gaussian_spot(key, n, *, centre, sigma, direction, t=0.0, lam, photons, pol=None):  # noqa: PLR0913
    """Return a pencil beam with a Gaussian transverse profile: ``n`` photons standing for ``photons`` in total.

    The positions are drawn once about the centre and width at their values; the live ``centre`` and ``sigma`` enter by
    ``reweight`` of the profile's density (their derivatives are score weights, the right estimator for a beam).
    """
    u = jnp.asarray(direction, F32)
    u = sg(u / jnp.linalg.norm(u))
    e1, e2 = frames.basis(u)
    g = jax.random.normal(key, (n, 2), F32)
    c, s = jnp.asarray(centre, F32), jnp.asarray(sigma, F32)
    x = sg(c) + sg(s) * (g[:, :1] * e1 + g[:, 1:] * e2)
    r = x - c
    r2 = jnp.sum(r * e1, -1) ** 2 + jnp.sum(r * e2, -1) ** 2
    logp = -r2 / (2.0 * s * s) - 2.0 * jnp.log(s)
    w = jnp.asarray(photons, F32) / n * reweight(logp)
    pol = None if pol is None else jnp.broadcast_to(jnp.asarray(pol, F32), (n, 2, 3))
    return Photons(x=x, d=jnp.broadcast_to(u, (n, 3)), t=jnp.broadcast_to(jnp.asarray(t, F32), (n,)),
                   lam=jnp.broadcast_to(jnp.asarray(lam, F32), (n,)), w=w, pol=pol)  # fmt: skip


@dataclasses.dataclass(frozen=True)
class FunctionSource(Source):
    """A source given as one function (see the module docstring); hashed by its function and declaration."""

    fn: Callable
    name: str
    declared: tuple  # ((field, Param), ...)

    @property
    def prefix(self) -> str:
        """Return the prefix of this source's keys: ``source/<name>/``."""
        return f"{PREFIX}{self.name}/"

    @property
    def FIELDS(self) -> dict:  # noqa: N802 -- the Source protocol's name
        """Return {field: Param} of the declared parameters."""
        return dict(self.declared)

    def __call__(self, v, key, n):
        """Call the function itself."""
        return self.fn(v, key, n)

    def values(self, params=None) -> dict:
        """Return {field: value} float32: from ``params`` where it has the key, else the declared default."""
        p, out = {} if params is None else params, {}
        for k, decl in self.declared:
            v = p[self.prefix + k] if self.prefix + k in p else decl.value()
            if v is None:
                raise KeyError(f"source {self.name!r}: {self.prefix + k} has no default; set it in the params")
            out[k] = v
        seq = lambda a: isinstance(a, (tuple, list))  # noqa: E731
        return jax.tree_util.tree_map(lambda a: jnp.asarray(a, F32), out, is_leaf=seq)

    def parameters(self, params=None):
        """Return ``params`` (a ``lumen.Params``, default empty) with this source's keys added at their defaults."""
        values, spec = declared(self.prefix, dict(self.declared))
        base = Params({}, {}, {}) if params is None else params
        return base.merged(values, spec, f"source {self.name}")

    def emit(self, params, key, n: int, first_id=0, samples=None, *, reference=None):
        """Return photons ``first_id .. first_id + n - 1`` of the function's ``samples`` (default n) photons under
        ``key``: one draw of them all, so photon i is the same whatever the chunks. A tangent on a parameter that
        moves the light (``MOVING``), or reaching the photons' x, d or lam, is refused. ``reference``: unused (the
        function draws its photons from its own density)."""  # noqa: D205, D209
        v = self.values(params)
        v = {k: refusing(f"source {self.name!r}: {k} {MOVES}")(a) if k in self.MOVING else a for k, a in v.items()}
        key = jax.random.key_data(key) if jnp.issubdtype(jnp.asarray(key).dtype, jax.dtypes.prng_key) else key
        rows = _FN_EMIT(Static(self), v, key, int(samples or n), jnp.asarray(first_id, jnp.int32), int(n))
        g = refusing(f"source {self.name!r}: a tangent reaching its photons' x, d or lam {MOVES}")
        return rows._replace(x=g(rows.x), d=g(rows.d), lam=g(rows.lam))

    def emitter(self, params, key, samples: int, *, reference=None):
        """Return ``rows(first_id, n)`` of one draw of all ``samples`` photons: the function is called once per run."""
        whole = self.emit(params, key, samples, 0, samples)
        return lambda first_id, n: jax.tree_util.tree_map(
            lambda a: jnp.take(a, first_id + jnp.arange(n, dtype=jnp.int32), axis=0, mode="fill", fill_value=0), whole
        )

    def check(self, params=None) -> None:
        """Check the function's output at first use (see the module docstring); raise SourceError."""
        if self in _CHECKED:
            return
        v = self.values(params)
        key = jax.random.PRNGKey(0)
        out = jax.eval_shape(lambda v_, k_: self.fn(v_, k_, SOURCE_CHECK_N), v, key)
        _check_shapes(self, out)
        jaxpr = jax.make_jaxpr(lambda v_, k_: self.fn(v_, k_, SOURCE_CHECK_N))(v, key)
        bad = sorted(callbacks(jaxpr.jaxpr))
        if bad:
            raise SourceError(f"source {self.name!r}: host callbacks {bad} (a source is plain JAX)")
        ph = self.fn(self.values(None) if traced(v) else v, key, SOURCE_CHECK_N)
        if traced(ph):
            raise GradientNotAvailable(
                f"source {self.name!r}: its function closes over a traced value it is not "
                "given: declare it as a parameter (lumen.Param) and give it in params"
            )
        _check_draw(self, ph)
        _CHECKED.add(self)


# the source functions whose checks have run: weak, so a source a caller drops is not held for the
# process's life
_CHECKED: weakref.WeakSet = weakref.WeakSet()


def _check_type(src, ph):
    """Refuse a source function's return that is not Photons."""
    if not isinstance(ph, Photons):
        raise SourceError(f"source {src.name!r}: return lumen.Photons, not {type(ph).__name__}")


def _draw(static, v, key, n, first_id, count):
    src = static()  # the source by its structure (``lumen._core.static``): its function's code, not the object
    ph = src.fn(v, key, n)
    _check_type(src, ph)
    pol = jnp.zeros((n, 2, 3), F32) if ph.pol is None else jnp.asarray(ph.pol, F32)
    lab = None if ph.label is None else jnp.asarray(ph.label, jnp.int32)
    e = Photons(x=jnp.asarray(ph.x, F32), d=jnp.asarray(ph.d, F32), t=jnp.asarray(ph.t, F32),
                lam=jnp.asarray(ph.lam, F32), w=jnp.asarray(ph.w, F32), pol=pol, label=lab)  # fmt: skip
    return jax.tree_util.tree_map(lambda a: jax.lax.dynamic_slice_in_dim(a, first_id, count, 0), e)


_FN_EMIT = jax.jit(_draw, static_argnums=(0, 3, 5))


def _check_shapes(src, out):
    _check_type(src, out)
    n = SOURCE_CHECK_N
    want = dict(x=(n, 3), d=(n, 3), t=(n,), lam=(n,), w=(n,), pol=(n, 2, 3), label=(n,))
    for k, shape in want.items():
        a = getattr(out, k)
        if a is None:
            continue
        kind = jnp.integer if k == "label" else jnp.floating
        if tuple(a.shape) != shape or not jnp.issubdtype(a.dtype, kind):
            raise SourceError(f"source {src.name!r}: Photons.{k} has shape {tuple(a.shape)} {a.dtype}, expected "
                              f"{shape} {'int' if k == 'label' else 'float'} (n = {n})")  # fmt: skip


def _check_draw(src, ph):
    vals = {k: np.asarray(getattr(ph, k)) for k in Photons._fields if getattr(ph, k) is not None}
    for k, a in vals.items():
        if not np.all(np.isfinite(a)):
            raise SourceError(f"source {src.name!r}: Photons.{k} is not finite at the defaults")
    norm = np.linalg.norm(vals["d"], axis=-1)
    if np.max(np.abs(norm - 1.0)) > UNIT_TOL:
        raise SourceError(f"source {src.name!r}: directions are not unit vectors (|d| up to {norm.max():.4g})")
    if np.any(vals["w"] < 0.0):
        raise SourceError(f"source {src.name!r}: negative weights")


def source(params: dict | None = None, name: str | None = None):
    """Return a decorator making ``f(v, key, n) -> Photons`` a source (see the module docstring)."""
    params = dict(params or {})
    for k, p in params.items():
        if not isinstance(p, Param):
            raise TypeError(f"lumen.source: parameter {k!r} must be a lumen.Param")

    def wrap(fn):
        n = name or fn.__name__
        if not n or "/" in n:
            raise ValueError(f"lumen.source: name {n!r} (no '/')")
        return FunctionSource(fn, n, tuple(sorted(params.items())))

    return wrap
