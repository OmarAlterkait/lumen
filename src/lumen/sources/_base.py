"""What every source shares: its parameters, the guard on its moving fields, its random numbers, its spectra.

A source is a frozen dataclass whose ``FIELDS`` are parameters ``source/<field>``, traced: ``values(params)``
takes each from ``params`` where it has the key, else from the source, plus the material values it ``reads``.
``uniforms(key, k, n, first_id)``: row i is photon first_id + i's k uniforms in (0, 1), from row first_id + i of
``jax.random.bits(key, (N, k))`` (partitionable threefry: a row depends on its index only); ``gaussian`` of them is
``jax.random.normal`` bit for bit. ``photon_uniforms``: photon i's uniforms from its own key ``fold_in(key, i)`` at
vertex 0, as the transport draws them.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.random import threefry2x32_p

from lumen._core import rng
from lumen._core.safe import safe_div
from lumen.media import spectra
from lumen.units import HC

F32, INF, PREFIX = jnp.float32, float("inf"), "source/"
POSITION, DIRECTION = ("m", (-INF, INF), "moving_paths"), ("", (-1.0, 1.0), "moving_paths")
TIME, COUNT, WAVELENGTH = ("ns", (-INF, INF), "reweight"), ("", (0.0, INF), "reweight"), ("nm", (0.0, INF), "reweight")
NORMAL_LO = np.nextafter(np.float32(-1.0), np.float32(0.0))  # jax.random.normal's lower bound
sg = jax.lax.stop_gradient


class GradientNotAvailable(ValueError):  # noqa: N818 -- the documented error catalogue's name
    """A derivative the method cannot give (docs/explanation/known-limits.md)."""


@dataclasses.dataclass(frozen=True)
class Source:
    """The base of every source; ``FIELDS``: {field: (unit, range, gradient method)}, ``MOVING``: those that move it."""

    FIELDS: ClassVar[dict] = {}
    MOVING: ClassVar[tuple] = ()

    @property
    def drawn_direction(self) -> bool:
        """Return whether the directions are drawn from a density in solid angle (the moving-source method's case)."""
        return False

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
        return jax.tree_util.tree_map(lambda a: jnp.asarray(a, F32), {k: v for k, v in out.items() if v is not None},
                                      is_leaf=seq)  # fmt: skip

    def parameters(self, params=None):
        """Return ``params`` (a ``lumen.Params``, default empty) with this source's ``source/`` keys added."""
        from lumen.params import ParamInfo, Params

        fs = [f for f in self.FIELDS if getattr(self, f) is not None]
        spec = {PREFIX + f: ParamInfo(PREFIX + f, *self.FIELDS[f], self.drawn_direction or f not in self.MOVING)
                for f in fs}  # fmt: skip
        base = Params({}, {}, {}) if params is None else params
        return base.merged({PREFIX + f: getattr(self, f) for f in fs}, spec, f"source {type(self).__name__}")

    def emit(self, params, key, n: int, first_id=0, samples=None):
        """Return the Emission of photons ``first_id .. first_id + n - 1`` of ``samples`` (default n).

        Their weights sum in expectation to the source's photons over the ``samples``. A tangent on a ``MOVING`` field
        raises ``GradientNotAvailable`` unless the directions are drawn.
        """
        v = self.values(params)
        if not self.drawn_direction:
            v = {k: _guard(f"{type(self).__name__}.{k}")(a) if k in self.MOVING else a for k, a in v.items()}
        s = copy.copy(self)  # the structure: the static part of the jitted emission
        for f in self.FIELDS:
            object.__setattr__(s, f, None)
        key = jax.random.key_data(key) if jnp.issubdtype(jnp.asarray(key).dtype, jax.dtypes.prng_key) else key
        return _EMIT(s, v, key, jnp.asarray(first_id, jnp.int32), int(n), jnp.asarray(samples or n, F32))

    def draw(self, v, key, n, first_id, samples):
        """Return the Emission from the values ``v`` (each source's own photons)."""
        raise NotImplementedError


_EMIT = jax.jit(lambda s, v, key, first_id, n, samples: s.draw(v, key, n, first_id, samples), static_argnums=(0, 4))


def _guard(name):
    def rule(primals, tangents):
        if not isinstance(tangents[0], jax.custom_derivatives.SymbolicZero):
            raise GradientNotAvailable(f"{name} moves photons whose directions are not drawn from a density "
                                       "(drawn_direction False): no derivative "
                                       "(docs/explanation/known-limits.md)")  # fmt: skip
        return primals[0], jnp.zeros_like(primals[0])

    f = jax.custom_jvp(lambda a: a)
    f.defjvp(rule, symbolic_zeros=True)
    return lambda x: jax.tree_util.tree_map(f, x)


def uniforms(key, k: int, n: int | None = None, first_id=0):
    """Return (n, k) uniforms in (0, 1) of photons first_id + i (n None: the (k,) uniforms of the key itself)."""
    p = jnp.asarray(first_id).astype(jnp.uint32) + jnp.arange(1 if n is None else n, dtype=jnp.uint32)
    a = (p >> 16)[:, None] * jnp.uint32(k)  # the counter p k + j in 64 bits, as (hi, lo)
    b = (p & 0xFFFF)[:, None] * jnp.uint32(k) + jnp.arange(k, dtype=jnp.uint32)
    hi, lo = (a + (b >> 16)) >> 16, ((a + (b >> 16)) << 16) | (b & 0xFFFF)
    b1, b2 = threefry2x32_p.bind(jnp.broadcast_to(key[0], hi.shape), jnp.broadcast_to(key[1], hi.shape), hi, lo)
    rand = (((b1 ^ b2) >> (32 - rng.UNIFORM_BITS)).astype(F32) + 0.5) * F32(2.0**-rng.UNIFORM_BITS)
    return rand[0] if n is None else rand


def photon_uniforms(key, k: int, n: int, first_id=0):
    """Return (n, k): photon first_id + i's k uniforms at vertex 0 of its own key."""
    ids = jnp.asarray(first_id, jnp.int32) + jnp.arange(n, dtype=jnp.int32)
    return jax.vmap(lambda i: uniforms(rng.vertex_key(rng.photon_key(key, i), 0), k))(ids)


def gaussian(rand):
    """Return standard normals from ``uniforms``: ``jax.random.normal``'s float32 map of the same bits."""
    f = rand - F32(2.0**-24)  # the bits' float in [0, 1), exact
    v = jnp.maximum(NORMAL_LO, f * (np.float32(1.0) - NORMAL_LO) + NORMAL_LO)
    return np.float32(np.sqrt(2.0)) * jax.lax.erf_inv(v)


def rows(a, n, first_id, rank):
    """Return a field's values for photons first_id .. first_id + n - 1: one per photon if it has a photon axis."""
    a = jnp.asarray(a, F32)
    return jax.lax.dynamic_slice_in_dim(a, first_id, n, 0) if a.ndim > rank else jnp.broadcast_to(a, (n,) + a.shape)


def proposal(name, table):
    """Return a spectrum table's (energies eV, intensities) fixed wavelength proposal, Geant4's sampler (None: None)."""
    if table is None:
        return None
    e, v = np.asarray(table.energy, np.float64), np.asarray(table.value, np.float64)
    return spectra.geant4_spectrum(name, e, v), tuple(e.tolist()), tuple(v.tolist())  # static: hashable


def wavelength(prop, live, rand):
    """Return (lam (nm), weight p(lam; live) / p(lam; nominal)) at uniform rand, the weight exactly 1 at the nominal."""
    lam = sg(prop[0].sample(rand))
    p = _g4_density(live.energy, live.value, HC / lam)
    e0, v0 = jnp.asarray(prop[1], F32), jnp.asarray(prop[2], F32)
    same = jnp.all(sg(live.value) == v0) & jnp.all(sg(live.energy) == e0)
    return lam, jnp.where(same, F32(1.0), sg(p) / _g4_density(e0, v0, HC / lam)) * score(p)


@jax.custom_jvp
def score(p):
    """Return 1, exactly, with the derivative dp / p (the score of a density p; 0 where p = 0)."""
    return jnp.ones_like(p)


score.defjvp(lambda primals, tangents: (jnp.ones_like(primals[0]), safe_div(tangents[0], primals[0])))


def _g4_density(e, v, energy):
    i = jnp.clip(jnp.searchsorted(e, energy, side="right") - 1, 0, e.shape[0] - 2)
    return (v[i] + v[i + 1]) / jnp.sum((v[1:] + v[:-1]) * (e[1:] - e[:-1]))
