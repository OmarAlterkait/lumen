"""A sensor response of your own: one JAX function of the histograms, its parameters declared as keys.

    @lumen.response(group="pmt", params={"sat": Param("pe", (1, 1e4), default=300.0)})
    def saturating_pe(h, v):                  # h: lumen.readout.Histograms, v: this response's and the group's values
        pe = lumen.readout.pe(h, v)           # lumen's own pieces take v as they take a Params
        return v["sat"] * (1.0 - jnp.exp(-pe / v["sat"]))

    det = det.with_responses(saturating_pe)   # adds sensors/pmt/sat to det.params()
    y = saturating_pe(light, det.params())    # differentiable in every key, no new simulation

Tracking fills the histograms (``lumen.readout.grid``: per sensor, wavelength cell x hit-angle cell, and time nodes);
a response is plain JAX from them to any observable, exact on the grid's knots. ``v`` holds the response's declared
values under their short names and every ``sensors/<group>/...`` key of the parameters, so ``lumen.readout.pe(h, v)``,
``time_nodes(h, v)`` and the other readout functions take it. Its parameters are after-tracking keys
``sensors/<group>/<name>``: ``Params.set``, a calibration and ``jax.grad`` take them, and changing one never
simulates again. The histograms keep the sensor, wavelength, hit angle (on a flat disc, the radius on the cathode) and
time; not the azimuth, the point on the cathode or the incidence direction.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from lumen.readout.detection import histograms, nodes, values


@dataclasses.dataclass(frozen=True)
class Response:
    """A sensor response (see the module docstring): ``response(light, params)`` -> its observable."""

    fn: Callable
    group: str
    name: str
    declared: tuple  # ((name, Param), ...)

    @property
    def prefix(self) -> str:
        """Return the prefix of this response's keys, ``sensors/<group>/``."""
        return f"sensors/{self.group}/"

    def parameters(self, params=None):
        """Return ``params`` (a ``lumen.Params``, default empty) with this response's keys added at their defaults."""
        from lumen.params import Params, declared

        values_, spec = declared(self.prefix, dict(self.declared), "after_tracking", f"response {self.name}")
        base = Params({}, {}, {}) if params is None else params
        return base.merged(values_, spec, f"response {self.name}")

    def values(self, params) -> dict:
        """Return ``v``: the group's keys of ``params`` and this response's values by short name (default if unset)."""
        p = {} if params is None else params
        v = {k: p[k] for k in p if k.startswith(self.prefix)}
        for k, decl in self.declared:
            x = p[self.prefix + k] if self.prefix + k in p else decl.value()
            if x is None:
                raise KeyError(f"response {self.name!r}: {self.prefix + k} has no default; set it in the params")
            v[k] = x
        return v

    def __call__(self, light, params):
        """Return the response of ``light`` (a Light or its Histograms) at ``params``."""
        h = histograms(light)
        if h.group != self.group:
            raise ValueError(f"response {self.name!r} reads group {self.group!r}, the histograms are {h.group!r}'s")
        return self.fn(h, self.values(params))


def response(group: str, params: dict | None = None, name: str | None = None):
    """Return a decorator making ``f(h, v)`` a sensor response of ``group`` (see the module docstring)."""
    from lumen.params import Param

    params = dict(params or {})
    for k, p in params.items():
        if not isinstance(p, Param) or p.moves:
            raise TypeError(f"lumen.response: parameter {k!r} must be a lumen.Param that does not move the light")
        if "/" in k:
            raise ValueError(f"lumen.response: parameter name {k!r} (no '/')")

    def wrap(fn):
        return Response(fn, group, name or fn.__name__, tuple(sorted(params.items())))

    return wrap


def time_nodes(light, params):
    """Return (nodes (S, nt), late (S, 2), early (S, 2)): the expected photoelectrons on the time nodes t0 + j dt.

    The time histogram scaled per sensor so its total equals ``pe`` (the group's QE, CE and efficiency), the delay
    applied; ``late`` / ``early`` keep (q, q t) beyond the last / before the first node. ``params`` as ``pe``'s.
    """
    h = histograms(light)
    return nodes(h, values(h, params))
