"""A source as one user function: ``@lumen.source(params=..., density=...)`` over ``f(v, key, n) -> lumen.Photons``.

    @lumen.source(params={"x": Param("m", shape=(3,), moves=True, default=(0, 0, 0.2)),
                          "m": Param("", (0, 50), default=4.0)}, density="direction")
    def led(v, key, n):                 # v: the parameter values (traced), key: a JAX key, n: a static count
        ...
        return lumen.Photons(x, d, t, lam, w, dlogp=lumen.sources.grad_logpdf(logp, d))

    light = lumen.simulate(det, led, samples=1 << 20, seed=0, wrt=["source/led/x"])

The function returns ``n`` photons whose weights sum, in expectation, to the photons the source stands for; ``simulate``
calls it once with n = samples and runs its photons in chunks (photon i is the same row whatever the chunk size). Its
parameters are the keys ``source/<name>/<field>`` (``name``: the function's, unless given), taken from the ``params``
given to ``simulate`` where they are set, else the declared defaults (``FunctionSource.parameters`` adds them to a
``Params``).

``density`` says how the directions were drawn, which decides the derivatives that move the light:

- "isotropic": uniform on the sphere. A position (a parameter declared ``moves=True``, in ``wrt``) is differentiated by
  the moving-source method, with the density's gradient zero (as ``lumen.sources.Point``).
- "direction": any smooth density p(d | x). The function must return ``dlogp`` = grad_d log p (``grad_logpdf``), the
  chain start's density gradient; without it ``simulate`` refuses. A density that depends on a parameter (an axis, a
  width) enters either by drawing the directions from the live parameter (``d`` carries the tangent) or by keeping
  them and weighting with ``reweight(logp)``. A density with a hard edge (a uniform cone) has no edge term: smooth the
  edge (a density whose gradient is zero on every photon is refused).
- "none": no density (a beam, a Cherenkov cone): a derivative that moves the photons is refused.

Without a moving parameter in ``wrt``, any tangent on the photons' x, d or lam is refused (it would miss the
solid-angle terms): declare the parameter ``moves=True`` and list it in ``wrt``. A wavelength or intensity parameter
enters by ``reweight``. Checks at first use: the output's type, shapes and dtypes, ``dlogp`` when required, no host
callbacks, and one draw at the defaults (finite, unit directions, a direction density with a gradient).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from lumen._core import frames
from lumen.sources._base import PREFIX, GradientNotAvailable, Source, _guard
from lumen.transport.state import Emission

F32 = jnp.float32
DENSITIES = ("isotropic", "direction", "none")
CHECK_N = 256  # photons of the first-use check's draw
UNIT_TOL = 1e-3  # |d| - 1 allowed in that draw
CALLBACKS = ("pure_callback", "io_callback", "callback")  # host callbacks (jax.debug.print is allowed)
sg = jax.lax.stop_gradient


class SourceError(ValueError):
    """A source function whose output breaks its declaration (shapes, a missing dlogp, a host callback)."""


class Photons(NamedTuple):
    """The photons a source function returns (n of them); ``pol`` and ``dlogp`` are optional.

    ``x`` (n, 3) m, ``d`` (n, 3) unit, ``t`` (n,) ns, ``lam`` (n,) nm, ``w`` (n,) the photons each stands for, ``pol``
    (n, 2, 3) the Jones vector's real and imaginary parts (None: unpolarised), ``dlogp`` (n, 3) the gradient in d of
    the direction density's log, grad_d log p(d | x) (``grad_logpdf``; required by ``density="direction"``).
    """

    x: jax.Array
    d: jax.Array
    t: jax.Array
    lam: jax.Array
    w: jax.Array
    pol: jax.Array | None = None
    dlogp: jax.Array | None = None


_HOLD = _guard("a lumen.concat piece without dlogp (its photons cannot move: give its dlogp)")


def concat(*parts: Photons) -> Photons:
    """Return the photons of several pieces, in order (several emitters in one source).

    Each piece's weights stay its own (scale them to its share of the photons). When some pieces give ``dlogp`` and
    others not, the others get zeros and their positions may not move (a beam beside an LED: its rows stay put).
    """
    if not parts:
        raise ValueError("concat: no photons")
    pol, grad = any(p.pol is not None for p in parts), any(p.dlogp is not None for p in parts)
    out = []
    for p in parts:
        n = jnp.shape(p.x)[0]
        q = Photons(*(jnp.asarray(a, F32) for a in p[:5]))
        q = q._replace(pol=None if not pol else jnp.zeros((n, 2, 3), F32) if p.pol is None else p.pol)
        if grad and p.dlogp is None:
            q = q._replace(x=_HOLD(q.x), dlogp=jnp.zeros((n, 3), F32))
        elif grad:
            q = q._replace(dlogp=p.dlogp)
        out.append(q)
    return Photons(*(None if f is None else jnp.concatenate([getattr(q, k) for q in out])
                     for k, f in zip(Photons._fields, out[0], strict=True)))  # fmt: skip


@jax.custom_jvp
def reweight(logp):
    """Return 1, exactly, with the derivative of ``logp``: the weight of draws kept fixed while their log density moves.

    ``w = w0 * reweight(logp_live(d))``: directions (or wavelengths, positions) drawn once and kept, their density's
    parameters live; the value is unchanged and the derivative is the score, d logp.
    """
    return jnp.ones_like(logp)


reweight.defjvp(lambda primals, tangents: (jnp.ones_like(primals[0]), tangents[0]))


def grad_logpdf(logpdf: Callable, d):
    """Return (n, 3) grad_d logpdf(d_i) of a log density of one direction (3,) -> scalar, at the fixed directions."""
    return sg(jax.vmap(jax.grad(logpdf))(sg(jnp.asarray(d, F32))))


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
    ``reweight`` of the profile's density (their derivatives are score weights, the right estimator for a beam). The
    beam has no direction density (``dlogp`` None): it cannot move.
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
    return Photons(x, jnp.broadcast_to(u, (n, 3)), jnp.broadcast_to(jnp.asarray(t, F32), (n,)),
                   jnp.broadcast_to(jnp.asarray(lam, F32), (n,)), w, pol)  # fmt: skip


@dataclasses.dataclass(frozen=True)
class FunctionSource(Source):
    """A source given as one function (see the module docstring); hashed by its function and declaration."""

    fn: Callable
    name: str
    declared: tuple  # ((field, Param), ...)
    density: str

    @property
    def prefix(self) -> str:
        """Return the prefix of this source's keys: ``source/<name>/``."""
        return f"{PREFIX}{self.name}/"

    @property
    def FIELDS(self) -> dict:  # noqa: N802 -- the Source protocol's name
        """Return {field: Param} of the declared parameters."""
        return dict(self.declared)

    @property
    def MOVING(self) -> tuple:  # noqa: N802 -- the Source protocol's name
        """Return the parameters declared ``moves=True``."""
        return tuple(k for k, p in self.declared if p.moves)

    @property
    def ORIGINS(self) -> tuple:  # noqa: N802 -- read by simulate for the moving-source clearance
        """Return the moving parameters that are points (last axis 3): the light's origins."""
        return tuple(k for k, p in self.declared if p.moves and p.shape[-1:] == (3,))

    @property
    def drawn_direction(self) -> bool:
        """Return whether the directions are drawn from a density (``density`` is not "none")."""
        return self.density != "none"

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
        from lumen.params import Params, declared

        values, spec = declared(self.prefix, dict(self.declared))
        base = Params({}, {}, {}) if params is None else params
        return base.merged(values, spec, f"source {self.name}")

    def emit(self, params, key, n: int, first_id=0, samples=None, *, moving=False):
        """Return photons ``first_id .. first_id + n - 1`` of the function's ``samples`` (default n) photons under
        ``key``: one draw of them all, so photon i is the same whatever the chunks. ``moving``: the moving-source
        method is on (else a tangent on x, d or lam is refused)."""  # noqa: D205, D209
        v = self.values(params)
        key = jax.random.key_data(key) if jnp.issubdtype(jnp.asarray(key).dtype, jax.dtypes.prng_key) else key
        rows = _FN_EMIT(self, v, key, int(samples or n), jnp.asarray(first_id, jnp.int32), int(n))
        if not moving:
            g = _guard(f"source {self.name!r}: a parameter that moves its photons (declare it moves=True and give it "
                       "in wrt)")  # fmt: skip
            rows = rows._replace(x=g(rows.x), d=g(rows.d), lam=g(rows.lam))
        return rows

    def check(self, params=None) -> None:
        """Check the function's output at first use (see the module docstring); raise SourceError."""
        if self in _CHECKED:
            return
        v = self.values(params)
        key = jax.random.PRNGKey(0)
        out = jax.eval_shape(lambda v_, k_: self.fn(v_, k_, CHECK_N), v, key)
        _check_shapes(self, out)
        jaxpr = jax.make_jaxpr(lambda v_, k_: self.fn(v_, k_, CHECK_N))(v, key)
        bad = sorted(_callbacks(jaxpr.jaxpr))
        if bad:
            raise SourceError(f"source {self.name!r}: host callbacks {bad} (a source is plain JAX)")
        concrete = not any(isinstance(a, jax.core.Tracer) for a in jax.tree_util.tree_leaves(v))
        _check_draw(self, self.fn(v if concrete else self.values(None), key, CHECK_N))
        _CHECKED.add(self)


_CHECKED: set = set()


def _draw(src, v, key, n, first_id, count):
    ph = src.fn(v, key, n)
    if not isinstance(ph, Photons):
        raise SourceError(f"source {src.name!r}: return lumen.Photons, not {type(ph).__name__}")
    if src.density == "direction" and ph.dlogp is None:
        raise SourceError(f"source {src.name!r}: density='direction' needs dlogp (lumen.sources.grad_logpdf)")
    if src.density == "isotropic" and ph.dlogp is not None:
        raise SourceError(f"source {src.name!r}: density='isotropic' has no dlogp (declare density='direction')")
    pol = jnp.zeros((n, 2, 3), F32) if ph.pol is None else jnp.asarray(ph.pol, F32)
    g = sg(jnp.asarray(ph.dlogp, F32)) if src.density == "direction" else None
    e = Emission(jnp.asarray(ph.x, F32), jnp.asarray(ph.d, F32), jnp.asarray(ph.t, F32), jnp.asarray(ph.lam, F32),
                 jnp.asarray(ph.w, F32), pol, jnp.full(n, src.drawn_direction), g)  # fmt: skip
    return jax.tree_util.tree_map(lambda a: jax.lax.dynamic_slice_in_dim(a, first_id, count, 0), e)


_FN_EMIT = jax.jit(_draw, static_argnums=(0, 3, 5))


def _check_shapes(src, out):
    if not isinstance(out, Photons):
        raise SourceError(f"source {src.name!r}: return lumen.Photons, not {type(out).__name__}")
    n = CHECK_N
    want = dict(x=(n, 3), d=(n, 3), t=(n,), lam=(n,), w=(n,), pol=(n, 2, 3), dlogp=(n, 3))
    for k, shape in want.items():
        a = getattr(out, k)
        if a is None:
            continue
        if tuple(a.shape) != shape or not jnp.issubdtype(a.dtype, jnp.floating):
            raise SourceError(f"source {src.name!r}: Photons.{k} has shape {tuple(a.shape)} {a.dtype}, expected "
                              f"{shape} float (n = {n})")  # fmt: skip
    if src.density == "direction" and out.dlogp is None:
        raise SourceError(f"source {src.name!r}: density='direction' needs dlogp (lumen.sources.grad_logpdf)")
    if src.density == "isotropic" and out.dlogp is not None:
        raise SourceError(f"source {src.name!r}: density='isotropic' has no dlogp (declare density='direction')")


def _callbacks(jaxpr):
    for eq in jaxpr.eqns:
        if eq.primitive.name in CALLBACKS:
            yield eq.primitive.name
        for p in eq.params.values():
            for sub in p if isinstance(p, (tuple, list)) else (p,):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    yield from _callbacks(inner)


def _check_draw(src, ph):
    import numpy as np

    vals = {k: np.asarray(getattr(ph, k)) for k in Photons._fields if getattr(ph, k) is not None}
    for k, a in vals.items():
        if not np.all(np.isfinite(a)):
            raise SourceError(f"source {src.name!r}: Photons.{k} is not finite at the defaults")
    norm = np.linalg.norm(vals["d"], axis=-1)
    if np.max(np.abs(norm - 1.0)) > UNIT_TOL:
        raise SourceError(f"source {src.name!r}: directions are not unit vectors (|d| up to {norm.max():.4g})")
    if np.any(vals["w"] < 0.0):
        raise SourceError(f"source {src.name!r}: negative weights")
    if src.density == "direction" and not np.any(vals["dlogp"] != 0.0):
        raise GradientNotAvailable(f"source {src.name!r}: dlogp is zero on every photon. An isotropic source is "
                                   "density='isotropic'; a uniform density inside a hard angular edge (a top-hat cone) "
                                   "has an edge term the method does not carry: smooth the edge "
                                   "(docs/howto/extending.md)")  # fmt: skip


def source(params: dict | None = None, density: str = "isotropic", name: str | None = None):
    """Return a decorator making ``f(v, key, n) -> Photons`` a source (see the module docstring)."""
    from lumen.params import Param

    if density not in DENSITIES:
        raise ValueError(f"lumen.source: density one of {DENSITIES}, not {density!r}")
    params = dict(params or {})
    for k, p in params.items():
        if not isinstance(p, Param):
            raise TypeError(f"lumen.source: parameter {k!r} must be a lumen.Param")
        if p.moves and density == "none":
            raise ValueError(f"lumen.source: {k!r} moves the photons of a source without a direction density")

    def wrap(fn):
        n = name or fn.__name__
        if not n or "/" in n:
            raise ValueError(f"lumen.source: name {n!r} (no '/')")
        return FunctionSource(fn, n, tuple(sorted(params.items())), density)

    return wrap
