"""Check a derivative: forward mode against finite differences on the same random numbers, over seeds.

    def f(theta, seed):                                    # observables at theta, every random number from seed
        p = params.set("source/led/m", theta)
        return lumen.simulate(det, led, p, samples=1 << 20, seed=seed).arrivals

    report = lumen.tools.check_gradient(f, 4.0)
    assert report.passes, report

Per seed s: AD = ``jax.jvp`` of f(., s) at theta0, and FD = the stencil of f(theta0 + k h, s) (the same seed on every
side, so the FD is the derivative of the same realisation's expectation). Over the seeds: the mean and its error of
AD and FD per observable and in total, and the paired z of AD - FD (its error from the seeds' spread of AD - FD, with
the float32 rounding of both added in quadrature: ``stats.FD_ROUNDING``, the checks' one resolution, relative to
AD and to each stencil side |f| (times the stencil's coefficients, ``STENCILS``)). The test is the comparisons' one
(``lumen.tools.stats.mean_test``, paired on the seeds, over the observables and their total): Student scores, the
largest against its null correlated as the observables are, at ``stats.ALPHA``. An observable with too few seeds
carrying it to reject at that level has no power and is left out (unresolved); with none left ``passes`` is None, or
True where AD equals FD on every seed (a derivative that is zero, or exact, on every realisation).

Choose ``h`` so the step moves the observables by much more than ``stats.FD_ROUNDING`` relative: the photon loop's
float32 shares (a surface's 1 - R, an outcome's p / q) put a fine structure of about that size into f, with no bias in
the derivative, and an FD across a smaller step measures that structure.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from lumen.tools import stats

STENCILS = {"richardson": 3.0, "central": 1.0}  # each stencil's sum of |coefficient| x h over its sides: its
# rounding is at most that times one side's, ``stats.FD_ROUNDING`` |f| / h


@dataclass(frozen=True)
class GradientReport:
    """The comparison of one derivative (see the module docstring); arrays per observable, the total last."""

    theta0: float
    h: float
    seeds: int
    ad: np.ndarray
    ad_sem: np.ndarray
    fd: np.ndarray
    fd_sem: np.ndarray
    z: np.ndarray
    resolved: np.ndarray
    stat: float
    p: float
    level: float
    passes: bool | None

    @property
    def z_total(self) -> float:
        """Return the paired z of the total."""
        return float(self.z[-1])

    def __repr__(self):
        """Show the summary (an assertion message reads it)."""
        return str(self)

    def __str__(self):
        """Summarise: the total's AD, FD and score, the largest score's p against the level, and the verdict."""
        verdict = "no power" if self.passes is None else "passes" if self.passes else "FAILS"
        rows = [f"check_gradient at {self.theta0:g} (h {self.h:g}, {self.seeds} seeds): {verdict}",
                f"  total   AD {self.ad[-1]:+.6g} +- {self.ad_sem[-1]:.2g}   FD {self.fd[-1]:+.6g} +- "
                f"{self.fd_sem[-1]:.2g}   z {self.z[-1]:+.2f}",
                f"  largest score {self.stat:.2f} over {int(self.resolved.sum())} observables and the total, p "
                f"{self.p:.3g} against {self.level:.2g}"]  # fmt: skip
        for i in np.argsort(-np.abs(self.z[:-1]))[:4]:
            if self.resolved[i]:
                rows.append(f"  [{i}]  AD {self.ad[i]:+.6g} +- {self.ad_sem[i]:.2g}   FD {self.fd[i]:+.6g} +- "
                            f"{self.fd_sem[i]:.2g}   z {self.z[i]:+.2f}")  # fmt: skip
        return "\n".join(rows)


def _observables(f, theta, seed):
    return jnp.ravel(jnp.asarray(f(theta, seed), jnp.float32))


def _stencil(f, h, stencil):
    if stencil == "central":
        return (f(h) - f(-h)) / (2.0 * h)
    sides = {x: f(x) for x in (-h, -h / 2, h / 2, h)}
    return stats.richardson(lambda x: sides[x], h)


def check_gradient(f, theta0, *, seeds: int = 8, h: float | None = None, stencil: str = "richardson",
                   tangent=None) -> GradientReport:  # fmt: skip
    """Return the GradientReport of d f(theta, seed) / d theta at ``theta0`` (see the module docstring).

    ``f(theta, seed)`` returns the observables (an array of any shape, flattened; or a scalar) with every random
    number drawn from ``seed``. ``theta0``: a number, or an array with ``tangent`` the direction (default ones).
    ``h``: the FD step along it (default 1e-2 max(1, |theta0|)); ``stencil``: "richardson" (O(h^4), four sides) or
    "central" (O(h^2), two sides). A derivative the method refuses raises as ``simulate`` does.
    """
    if stencil not in STENCILS:
        raise ValueError(f"check_gradient: stencil one of {tuple(STENCILS)}, not {stencil!r}")
    if int(seeds) <= stats.power_floor(stats.ALPHA):
        raise ValueError(f"check_gradient: more than {stats.power_floor(stats.ALPHA):.1f} seeds (fewer cannot reject "
                         "at the checks' level: stats.mean_test)")  # fmt: skip
    th0 = jnp.asarray(theta0, jnp.float32)
    t = jnp.ones_like(th0) if tangent is None else jnp.asarray(tangent, jnp.float32)
    h = float(1e-2 * max(1.0, float(jnp.max(jnp.abs(th0))))) if h is None else float(h)
    ad, fd, f0 = [], [], []
    for s in range(int(seeds)):
        obs = functools.partial(_observables, f, seed=s)
        v, d = jax.jvp(obs, (th0,), (t,))
        side = lambda e, obs=obs: np.asarray(obs(th0 + jnp.float32(e) * t), np.float64)  # noqa: E731
        g = _stencil(side, h, stencil)
        ad.append(np.asarray(d, np.float64))
        fd.append(np.asarray(g, np.float64))
        f0.append(np.asarray(v, np.float64))
    ad, fd, f0 = (np.stack(a) for a in (ad, fd, f0))
    ad, fd, f0 = (np.concatenate([a, a.sum(1, keepdims=True)], 1) for a in (ad, fd, f0))
    floor = stats.FD_ROUNDING * np.hypot(np.abs(ad.mean(0)), STENCILS[stencil] * np.abs(f0.mean(0)) / h)
    t = stats.mean_test(ad, fd, paired=True, floor=floor)
    if t["passed"] is None and np.array_equal(ad, fd):  # AD equals FD on every seed (a zero derivative): no test
        t = t | {"passed": True}
    return GradientReport(float(jnp.ravel(th0)[0]) if th0.ndim else float(th0), h, int(seeds), ad.mean(0),
                          stats.sem(ad), fd.mean(0), stats.sem(fd), t["z"], t["power"], t["stat"], t["p"],
                          t["level"], t["passed"])  # fmt: skip
