"""A loss over a grid of one or two keys: its values, its derivatives along the axes, and their finite differences.

    s = lumen.tools.scan(loss, space, {"surfaces/Liner/REFLECTIVITY": np.linspace(0.05, 0.3, 21)})
    s.loss, s.grad, s.numerical_grad              # (n,) or (n1, n2); grad and numerical_grad (axes, *grid)
    pr = lumen.tools.profile(model, space, data, key, values)   # pr.nll: the likelihood, every other key refitted

``loss`` is plain JAX from a ``Params`` to a number; each scanned key is a single number of the ``Space``. The
derivative is taken with respect to the key's value (forward mode, one per axis); the numerical one is the central
difference of the scanned losses (one-sided at the grid's ends). A landscape and its zero crossings: not a validation
(``lumen.tools.check_gradient`` is). ``profile`` fixes one single-number key at each value and fits the others
(``lumen.tools.fit``): the profile likelihood, whose rise by 1/2 from its minimum marks the 1 sigma interval.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from lumen.tools.fit import fit


@dataclass(frozen=True)
class ScanResult:
    """A scan: the axes {key: values}, the loss on the grid, its derivatives and their finite differences."""

    axes: dict
    loss: np.ndarray
    grad: np.ndarray
    numerical_grad: np.ndarray


def scan(loss, space, axes: dict, x0=None) -> ScanResult:
    """Return the loss on the grid of ``axes`` ({key: values}, one or two keys), every other entry at ``x0``."""
    keys = list(axes)
    if not 1 <= len(keys) <= 2:
        raise ValueError("scan: one or two axes")
    offsets = dict(zip(space.keys, np.cumsum((0, *space.sizes))[:-1], strict=True))
    for k in keys:
        if k not in offsets or space.sizes[space.keys.index(k)] != 1:
            raise ValueError(f"scan: {k} is not a single-number key of the space")
    x0 = space.vector() if x0 is None else np.asarray(x0, np.float64)
    vals = [np.asarray(axes[k], np.float64) for k in keys]

    def at(values):
        x = jnp.asarray(x0, jnp.float32)
        for k, v in zip(keys, values, strict=True):
            x = x.at[offsets[k]].set(jnp.log(v) if space.log(k) else v)
        return x

    shape = tuple(len(v) for v in vals)
    out, grad = np.zeros(shape), np.zeros((len(keys), *shape))
    for idx in itertools.product(*(range(n) for n in shape)):
        point = [float(v[i]) for v, i in zip(vals, idx, strict=True)]

        def f(t, point=point):
            return loss(space.params(at([p + ti for p, ti in zip(point, t, strict=True)])))

        t0 = jnp.zeros(len(keys), jnp.float32)
        for a in range(len(keys)):
            v, d = jax.jvp(f, (t0,), (t0.at[a].set(1.0),))
            grad[(a, *idx)] = float(d)
        out[idx] = float(v)
    num = np.stack([np.gradient(out, vals[a], axis=a) for a in range(len(keys))])
    return ScanResult(dict(zip(keys, vals, strict=True)), out, grad, num)


@dataclass(frozen=True)
class ProfileResult:
    """A profile: the key, its values, the negative log-likelihood with the other keys refitted, their vectors."""

    key: str
    values: np.ndarray
    nll: np.ndarray
    x: np.ndarray


def profile(model, space, data, key: str, values, x0=None, **fit_kw) -> ProfileResult:
    """Return the profile likelihood of ``data`` in ``key`` (a single-number key of ``space``) at ``values``.

    At each value the other keys are fitted (``lumen.tools.fit`` with ``fit_kw``, ``key`` held fixed) from ``x0``
    (default the space's vector); ``x`` holds the whole vector at each value.
    """
    i = space.keys.index(key)
    if space.sizes[i] != 1:
        raise ValueError(f"profile: {key} is not a single-number key of the space")
    j = int(np.sum(space.sizes[:i]))
    x0 = space.vector() if x0 is None else np.asarray(x0, np.float64)
    nll, xs = [], []
    for v in np.asarray(values, np.float64):
        x = x0.copy()
        x[j] = np.log(v) if space.log(key) else v
        r = fit(model, space, data, x, fixed=(key,), **fit_kw)
        xs.append(r.x)
        nll.append(r.nll)
    return ProfileResult(key, np.asarray(values, np.float64), np.asarray(nll), np.stack(xs))
