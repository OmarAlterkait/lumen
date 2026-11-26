"""Photons through a loaded detector, for the checks: the model, the sources they emit and the sensor positions.

    s = photons.setup(det, lam_edges)                         # the model on the detector's navigator
    d = photons.Detected(s.n_sensors, s.lam_edges)
    light = lumen.simulate(det, source, samples=n, seed=seed, diagnostics=(d,))
    cells = photons.detected(light, d)                        # per sensor and band: q, q t, their errors (and times)

The checks run ``lumen.simulate`` with a diagnostic, as a user does, and read the
detected light per wavelength band (and its times)
through a diagnostic, ``Detected``. The derivative checks' toys run the production photon loop and readout
(``lumen.readout.fill.readout``, from its records) on a grid of wavelength bands, plus one output of their own
(``loop``).

Sources follow the references' convention: isotropic, monochromatic, unpolarised photons at t = 0, weight 1 each
(``lumen.sources.Point``; ``isotropic`` builds its rows).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from lumen import api, sources
from lumen.readout import fill
from lumen.readout.grid import Grid, charge_cells
from lumen.transport import step as steps

F32 = jnp.float32


@dataclass(frozen=True, eq=False)
class Setup:
    """A detector ready to run: the model (geometry, tables, step config), its sensors and wavelength bands (nm)."""

    det: object
    model: object
    n_sensors: int
    lam_edges: tuple

    def layout(self) -> fill.HistogramLayout:
        """Return the histograms of a run: per sensor, one charge cell per band."""
        return fill.HistogramLayout(self.n_sensors, Grid.bands(self.lam_edges))

    def loop(self) -> fill.Readout:
        """Return the checks' readout of a run (``loop``)."""
        return loop(self.layout())


def setup(det, lam_edges=None) -> Setup:
    """Return the Setup of a loaded detector: its Model is the one ``lumen.simulate`` runs (``lumen.api._model``).

    ``lam_edges`` (nm, ascending; the outer ones may be infinite) default to one band of every wavelength the source
    emits.
    """
    edges = tuple(float(v) for v in (lam_edges or (-np.inf, np.inf)))
    return Setup(det, api._model(det), len(det.parts.optics.sensor_pvs), edges)


@dataclass(frozen=True)
class Detected:
    """A records readout: the detected weight per sensor and band ("Q"), its q t ("Tw"), times, and moments.

    "Q" sums the deposits' ``q`` per sensor and wavelength band, "Tw" their q t; with ``nt`` > 0 also the arrival
    times on ``nt`` nodes of ``dt`` ns per sensor ``group`` and band ("H", split linearly between nodes) and beyond
    them (late, early: "L"). The moments (``moments``: a photon's adds to a cell summed, then squared) "Q2", "Tw2" and
    "QTw2" (of q + q t: with the other two, the photons' sum of q times q t) per sensor and band, and in a last row per
    band of the total over the sensors, each sensor's q times its ``weights`` (default 1).
    ``lumen.simulate(..., diagnostics=(Detected(...),))`` builds it from the run's deposit records
    (``lumen.diagnostics``); ``detected`` reads it.
    """

    n_sensors: int
    lam_edges: tuple
    group: tuple = ()
    nt: int = 0
    dt: float = 1.0
    weights: tuple = ()

    def _cells(self):
        return self.n_sensors * (len(self.lam_edges) - 1)

    def outputs(self):
        """Return {name: (cells, exact)}."""
        nb = len(self.lam_edges) - 1
        out = {"Q": (self._cells(), True), "Tw": (self._cells(), True)}
        out |= {k: (self._cells() + nb, True) for k in MOMENTS}
        if self.nt:
            ng = max(self.group) + 1
            out |= {"H": (ng * nb * self.nt, True), "L": (ng * nb * 2, True)}
        return out

    def _cell(self, dep):  # the deposit's (sensor, band) cell, its band, and whether it counts
        nb = len(self.lam_edges) - 1
        b = Grid.bands(self.lam_edges).axes[0].cells(dep)[0][0]  # the readout's band cell (-1 outside the edges)
        ok = (dep.sensor >= 0) & (b >= 0)
        return jnp.where(ok, jnp.maximum(dep.sensor, 0) * nb + b, -1), b, ok

    def emits(self, dep, weights=()):
        """Return the adds of one deposit record (each times its row weight by ``lumen.run.rebuild``)."""
        nb = len(self.lam_edges) - 1
        cell, b, ok = self._cell(dep)
        adds = [("Q", cell, dep.q), ("Tw", cell, dep.q * dep.time)]
        if self.nt:
            g = jnp.asarray(self.group, jnp.int32)[jnp.maximum(dep.sensor, 0)] * nb + b
            x = dep.time / self.dt
            j = jnp.floor(x).astype(jnp.int32)
            f = x - j
            for jj, w in ((j, 1.0 - f), (j + 1, f)):
                node = jnp.where(ok & (jj >= 0) & (jj < self.nt), g * self.nt + jj, -1)
                edge = jnp.where(ok & (jj >= self.nt), g * 2, jnp.where(ok & (jj < 0), g * 2 + 1, -1))
                adds += [("H", node, dep.q * w), ("L", edge, dep.q * w)]
        return adds

    def moments(self, dep, weights=()):
        """Return the moment adds of one deposit record: q, q t and q + q t at its cell and at its band's total."""
        cell, b, ok = self._cell(dep)
        total = jnp.where(ok, self._cells() + b, -1)
        w = jnp.asarray(self.weights, jnp.float32)[jnp.maximum(dep.sensor, 0)] if self.weights else 1.0
        q, tw = dep.q, dep.q * dep.time
        return [(k, c, v * f) for c, f in ((cell, 1.0), (total, w))
                for k, v in zip(MOMENTS, (q, tw, q + tw), strict=True)]  # fmt: skip


MOMENTS = ("Q2", "Tw2", "QTw2")  # ``Detected``'s moment outputs


def variance(q2, s, n: int):
    """Return the variance of a sum ``s`` of ``n`` i.i.d. photons from its per-photon second moment ``q2`` (float64,
    unbiased, not clipped: ``lumen.readout.grid.moment_error``'s estimate)."""  # noqa: D205, D209
    q2, s = np.asarray(q2, np.float64), np.asarray(s, np.float64)
    return n / max(n - 1.0, 1.0) * (q2 - s * s / n)


def detected(light, d: Detected) -> dict:
    """Return a ``Detected`` diagnostic's cells of a run (float64): "Q", "Tw" (S, bands), "Q2" (S, bands: Q's
    per-photon second moment), "Q_var" (the variance of each sensor's Q over the run's photons), per band the total's
    (each sensor's q times ``d.weights``) "total", "total_var", "Tw_total", "Tw_total_var", "cov_total" (its covariance
    with "Tw_total"), and "H" (groups, bands, nt), "L" when recorded."""  # noqa: D205, D209
    nb = len(d.lam_edges) - 1
    n = int(dict(light.provenance)["samples"])
    cell = lambda k: np.asarray(light.diagnostics[k], np.float64)  # noqa: E731
    q, tw = (cell(k).reshape(d.n_sensors, nb) for k in ("Q", "Tw"))
    q2, tw2, qtw2 = (cell(k).reshape(d.n_sensors + 1, nb) for k in MOMENTS)
    w = np.asarray(d.weights, np.float64)[:, None] if d.weights else np.ones((1, 1))
    tq, ttw = (w * q).sum(0), (w * tw).sum(0)
    cross = 0.5 * (qtw2[-1] - q2[-1] - tw2[-1])  # sum over photons of q times q t, of the totals
    out = dict(Q=q, Tw=tw, Q2=q2[:-1], Q_var=variance(q2[:-1], q, n), total=tq, total_var=variance(q2[-1], tq, n),
               Tw_total=ttw, Tw_total_var=variance(tw2[-1], ttw, n),
               cov_total=n / max(n - 1.0, 1.0) * (cross - tq * ttw / n))  # fmt: skip
    if d.nt:
        ng = max(d.group) + 1
        out["H"] = cell("H").reshape(ng, nb, d.nt)
        out["L"] = cell("L").reshape(ng, nb, 2)
    return out


@functools.lru_cache(maxsize=32)
def loop(layout: fill.HistogramLayout) -> fill.Readout:
    """Return the production readout of ``layout`` (``fill.readout``) plus the checks' time-weighted charge "Tw".

    Outputs (``lumen.run.run``: built from the photon loop's records, with its health): the readout's "charge:0"
    (S L), "arrival", "arrival_t", their moments, and "Tw" (S L): per charge cell, the sum of q t over the deposits the
    charge histogram counts (the mean arrival time, which the derivative checks compare).
    """
    if layout.index is not None:
        raise ValueError("photons.loop: a layout of every sensor in order (index None)")
    base, g = fill.readout((layout,)), layout.grid
    cells = layout.n_sensors * g.n_cells

    def emits(dep, weights=()):
        idx, w = charge_cells(dep, g)
        tw = [("Tw", idx[m], dep.q * w[m] * dep.time) for m in range(len(w))]
        return list(base.emits(dep, weights)) + tw

    def outputs():
        return base.outputs() | {"Tw": (cells, True)}

    return base._replace(emits=emits, outputs=outputs)


def packed(det, params=None) -> dict:
    """Return the engine's packed parameters of ``params`` (default: the detector's) as device arrays."""
    from lumen.build import pack

    return jax.tree_util.tree_map(jnp.asarray, pack.checked(det, params))


def operands(s: Setup, params, reference=None, controls=None):
    """Return the Operands of a run at packed ``params`` (sampled at ``reference``, default the same), under
    ``controls`` (default ``Controls()``: the default roulette, the one production runs use)."""  # noqa: D205, D209
    return steps.operands(s.model, params, reference, steps.Controls() if controls is None else controls)


def isotropic(x, lam, n_key):
    """Return the Photons of isotropic photons from the points ``x`` (n, 3) m at ``lam`` (nm, (n,) or one), t = 0,
    weight 1: ``lumen.sources.Point``'s draw under ``n_key``."""  # noqa: D205, D209
    n = x.shape[0]
    return sources.Point(x, n_photons=1.0, lam=lam).emit(None, n_key, n)  # one photon per row


def sensor_centres(s: Setup) -> np.ndarray:
    """Return each sensor's placement centre (S, 3) in m: the translation of its navigator volume."""
    geom = s.model.geometry
    nav = geom.nav
    tov = np.asarray(geom.sensor_of_vol)[: nav.n]
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
