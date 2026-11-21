"""Named parameter keys as one vector: what the tools trace and fit.

    space = lumen.tools.Space(det.params(), ["materials/Water/ABSLENGTH", "surfaces/Liner/REFLECTIVITY"])
    x0 = space.vector(det.params())       # one float vector: each key's whole value, transformed
    p = space.params(x)                   # det.params() with those keys replaced (x may be traced)
    space.sigma(cov)                      # {key: standard error in the key's units}
    lumen.tools.Space(p0, [], factors=["materials/Water/ABSLENGTH"])   # the table's values times one number

Each key enters with its whole value: a number, a per-sensor array, or a table's values (its knot energies stay
fixed). The transform follows one rule from the key's declared range (``Params.info(key).range``): a lower bound of 0
(or a length's, just above it) means the logarithm (positive values, fractional steps), anything else the value
itself, divided by ``scales[key]`` when given (to bring identity keys to a natural scale). A key in ``factors``
enters as one number instead: the factor on its whole value (``Params.scale``; 1 at ``params``), in log. The tools
trace only ``x``; every other value of ``params`` is a constant, so changing ``x`` never recompiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from lumen._core import domains
from lumen.format.calibrations import Table


def _value(v):
    return v.value if isinstance(v, Table) else v


@dataclass(frozen=True, eq=False)
class Space:
    """Named keys of a ``Params`` as one vector (see the module docstring)."""

    base: object
    keys: tuple
    scales: dict = field(default_factory=dict)
    factors: frozenset = frozenset()

    def __init__(self, params, keys=(), *, scales=None, factors=()):
        """Declare the keys of ``params`` to vary; ``scales``: {identity key: its natural scale}; ``factors``: keys
        varied by one factor on their whole value (appended after ``keys``)."""  # noqa: D205, D209
        factors = tuple(factors)
        keys = tuple(keys) + tuple(k for k in factors if k not in keys)
        for k in keys:
            params.info(k)  # an unknown key: Params' own error, naming the nearest keys
        object.__setattr__(self, "base", params)
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "scales", dict(scales or {}))
        object.__setattr__(self, "factors", frozenset(factors))
        bad = [k for k in self.scales if k not in keys or self.log(k)]
        if bad:
            raise ValueError(f"Space: scales for {bad}: a scale is for a key of this space without a log transform")

    def log(self, key) -> bool:
        """Return whether a key is varied in log (a factor, or its declared range starts at 0)."""
        return key in self.factors or self.base.info(key).range[0] in (0.0, domains.LENGTH[0])  # 0, or just above

    @property
    def sizes(self) -> tuple:
        """Return each key's number of entries in the vector."""
        return tuple(1 if k in self.factors else int(np.size(_value(self.base[k]))) for k in self.keys)

    def _factor(self, params, k) -> float:
        a, b = (np.asarray(_value(p[k]), np.float64).reshape(-1) for p in (params, self.base))
        f = a[b != 0] / b[b != 0]
        if not f.size or not np.allclose(f, f[0], rtol=1e-5) or f[0] <= 0.0:
            raise ValueError(f"Space: {k} is not a positive factor times the space's own value")
        return float(f[0])

    def vector(self, params=None) -> np.ndarray:
        """Return the vector of ``params`` (default: the space's own) at the space's keys (float64, host)."""
        p = self.base if params is None else params
        out = []
        for k in self.keys:
            if k in self.factors:
                out.append(np.log([self._factor(p, k)]))
                continue
            v = np.asarray(_value(p[k]), np.float64).reshape(-1)
            if self.log(k):
                if np.any(v <= 0.0):
                    raise ValueError(f"Space: {k} has a value <= 0; a key varied in log needs positive values")
                out.append(np.log(v))
            else:
                out.append(v / self.scales.get(k, 1.0))
        return np.concatenate(out) if out else np.zeros(0)

    def params(self, x):
        """Return the space's ``Params`` with its keys set from the vector ``x`` (traceable)."""
        x = jnp.asarray(x)
        p, i = self.base, 0
        for k, n in zip(self.keys, self.sizes, strict=True):
            old = self.base[k]
            if k in self.factors:
                p, i = p.scale(k, jnp.exp(x[i])), i + 1
                continue
            shape = np.shape(_value(old))
            xi = x[i : i + n].reshape(shape)
            i += n
            new = jnp.exp(xi) if self.log(k) else xi * self.scales.get(k, 1.0)
            new = new.astype(jnp.asarray(_value(old)).dtype)
            p = p.set(k, Table(old.energy, new) if isinstance(old, Table) else new)
        return p

    def values(self, x) -> dict:
        """Return {key: its value at the vector ``x``} (a factor key: the factor; a table: its values), host numpy."""
        x, out, i = np.asarray(x, np.float64), {}, 0
        for k, n in zip(self.keys, self.sizes, strict=True):
            v = np.exp(x[i : i + n]) if self.log(k) else x[i : i + n] * self.scales.get(k, 1.0)
            out[k] = float(v[0]) if n == 1 else v.reshape(np.shape(_value(self.base[k])))
            i += n
        return out

    def sigma(self, cov, x=None) -> dict:
        """Return {key: standard errors in the key's units} of a covariance of the vector at ``x`` (default the space's
        own; first order)."""  # noqa: D205, D209
        sd, out, i = np.sqrt(np.diag(np.asarray(cov, np.float64))), {}, 0
        x = self.vector() if x is None else np.asarray(x, np.float64)
        for k, n in zip(self.keys, self.sizes, strict=True):
            s = sd[i : i + n] * (np.exp(x[i : i + n]) if self.log(k) else self.scales.get(k, 1.0))
            out[k] = float(s[0]) if n == 1 else s.reshape(np.shape(_value(self.base[k])))
            i += n
        return out
