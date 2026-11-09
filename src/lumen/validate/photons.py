"""Photons through a loaded detector, for the checks: the model, the sources they emit and the sensor positions.

    s = photons.setup(det, lam_edges)                         # the model on the detector's navigator
    rows = photons.isotropic(jnp.array([0.0, 0.0, 0.0]), 400.0, 1 << 20, key)
    ops = photons.operands(s, det.pack(params))               # every value an operand: one compile per detector
    tally = lumen.run.run(spec, s.model, ops, rows, raw_key, replicas=8)

Sources follow the references' convention: isotropic, monochromatic, unpolarised photons at t = 0, weight 1 each.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from lumen.geometry import _legacy_tracked as tracked
from lumen.transport import item, state
from lumen.transport import step as steps

F32 = jnp.float32


@dataclass(frozen=True, eq=False)
class Setup:
    """A detector ready to run: the model (geometry, tables, step config) and its histogram shape."""

    det: object
    model: object
    tally: item.TallyShape

    @property
    def n_sensors(self) -> int:
        """Return the number of sensors."""
        return self.tally.n_sensors


def setup(det, lam_edges=None, step=None, clearance=None) -> Setup:
    """Return the Setup of a loaded detector.

    ``lam_edges`` (nm, ascending) default to its wavelength range; ``step`` the StepConfig (default: no moving
    source, no edge rows); ``clearance`` the moving-source method's field (``lumen.geometry.clearance.Clearance``).
    """
    geom = tracked.from_parts(det.parts, cache=True)
    edges = tuple(float(v) for v in (lam_edges or det.file.wavelength_range))
    model = steps.Model(geom, det.parts.optics, step or steps.StepConfig(), clearance)
    return Setup(det, model, item.TallyShape(len(det.parts.optics.tag_pvs), edges))


def packed(det, params=None) -> dict:
    """Return the engine's packed parameters of ``params`` (default: the detector's) as device arrays."""
    return jax.tree_util.tree_map(jnp.asarray, det.pack(params))


def operands(s: Setup, params, reference=None, controls=None):
    """Return the Operands of a run at packed ``params`` (sampled at ``reference``, default the same)."""
    return steps.operands(s.model, params, reference, controls, item.grid(s.tally))


@jax.jit
def isotropic(x, lam, n_key):
    """Return an Emission of isotropic photons from the points ``x`` (n, 3) m at ``lam`` (n,) nm, t = 0, weight 1."""
    n = x.shape[0]
    d = jax.random.normal(n_key, (n, 3), F32)
    d = d / jnp.linalg.norm(d, axis=1, keepdims=True)
    z = jnp.zeros(n, F32)
    return state.Emission(x.astype(F32), d, z, lam.astype(F32), z + 1.0, jnp.zeros((n, 2, 3), F32), jnp.ones(n, bool))


def pencil(x, d, lam, n) -> state.Emission:
    """Return an Emission of ``n`` photons from ``x`` along ``d`` at ``lam`` nm, t = 0, weight 1, unpolarised."""
    z = jnp.zeros(n, F32)
    return state.Emission(jnp.tile(jnp.asarray(x, F32), (n, 1)), jnp.tile(jnp.asarray(d, F32), (n, 1)), z, z + lam,
                          z + 1.0, jnp.zeros((n, 2, 3), F32), jnp.ones(n, bool))  # fmt: skip


def raw_key(seed, *folds):
    """Return the raw key data of ``PRNGKey(seed)`` folded in with each of ``folds`` (the loop's key operand)."""
    k = jax.random.PRNGKey(seed)
    for f in folds:
        k = jax.random.fold_in(k, f)
    return jax.random.key_data(k) if jnp.issubdtype(k.dtype, jax.dtypes.prng_key) else k


def sensor_centres(s: Setup) -> np.ndarray:
    """Return each sensor's placement centre (S, 3) in m: the translation of its navigator volume."""
    geom = s.model.geometry
    nav = geom.nav
    tov = np.asarray(geom.tag_of_vol)[: nav.n]
    centres = np.zeros((s.n_sensors, 3))
    t = np.asarray(nav.host["t"], np.float64)
    for v in np.flatnonzero(tov >= 0):
        centres[tov[v]] = t[v]
    if len(np.unique(tov[tov >= 0])) != s.n_sensors:
        raise ValueError(f"{s.n_sensors} sensors but {len(np.unique(tov[tov >= 0]))} sensor volumes in the navigator")
    return centres


def source_points(s: Setup, spec: dict) -> np.ndarray:
    """Return a check's source points (m): ``sources`` as given, or about the box centre of ``volume`` when named."""
    pts = np.asarray(spec["sources"], np.float64).reshape(-1, 3)
    if "volume" not in spec:
        return pts
    nav = s.model.geometry.nav
    i = next(i for i, d in enumerate(nav.defs) if d.startswith(spec["volume"]))
    return 0.5 * (np.asarray(nav.lo_w[i]) + np.asarray(nav.hi_w[i])) + pts
