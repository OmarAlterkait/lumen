"""Check a derivative: forward mode against finite differences on the same random numbers, over seeds.

    def f(theta, seed):                                    # observables at theta, every random number from seed
        p = params.set("source/led/m", theta)
        return lumen.simulate(det, led, p, samples=1 << 20, seed=seed, wrt=["source/led/m"]).arrivals

    report = lumen.validate.check_gradient(f, 4.0)
    assert report.passes, report

Per seed s: AD = ``jax.jvp`` of f(., s) at theta0, and FD = the stencil of f(theta0 + k h, s) (the same seed on every
side, so the FD is the derivative of the same realisation's expectation). Over the seeds: the mean and its error of
AD and FD per observable and in total, and the paired z of AD - FD (its error from the seeds' spread of AD - FD, with
the sides' float32 floors of ``lumen.validate.gradients``). The rules are the gradient check's
(``lumen.validate.stats``): the total's |z| <= ``Z_MAX``, and the observables' mean z^2 within its band about the t null
(K - 1) / (K - 3) of K seeds. An observable whose AD and FD are both zero on every seed is left out (unresolved).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from lumen.validate import stats

STENCILS = ("richardson", "central")


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
    chi2: float
    ndf: int
    null: float
    passes: bool

    @property
    def z_total(self) -> float:
        """Return the paired z of the total."""
        return float(self.z[-1])

    def __repr__(self):
        """Show the summary (an assertion message reads it)."""
        return str(self)

    def __str__(self):
        """Summarise: the total's AD, FD and z, the observables' mean z^2 against its null, and the verdict."""
        rows = [f"check_gradient at {self.theta0:g} (h {self.h:g}, {self.seeds} seeds): "
                f"{'passes' if self.passes else 'FAILS'}",
                f"  total   AD {self.ad[-1]:+.6g} +- {self.ad_sem[-1]:.2g}   FD {self.fd[-1]:+.6g} +- "
                f"{self.fd_sem[-1]:.2g}   z {self.z[-1]:+.2f}",
                f"  per observable: mean z^2 {self.chi2:.2f} over {self.ndf} (null {self.null:.2f})"]  # fmt: skip
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
    from lumen.validate.gradients import FLOOR, FLOOR_V

    if stencil not in STENCILS:
        raise ValueError(f"check_gradient: stencil one of {STENCILS}, not {stencil!r}")
    if int(seeds) < 4:
        raise ValueError("check_gradient: at least 4 seeds (the t null needs K > 3)")
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
    dif = ad - fd
    floor_v = FLOOR_V * np.abs(f0.mean(0)) / h
    sd = np.sqrt(stats.sem(dif) ** 2 + (FLOOR * np.abs(fd.mean(0))) ** 2 + floor_v**2)
    resolved = np.any(ad != 0.0, 0) | np.any(fd != 0.0, 0)
    z = np.where(sd > 0.0, dif.mean(0) / np.where(sd > 0.0, sd, 1.0), 0.0)
    ok = resolved[:-1] & (sd[:-1] > 0.0)
    ndf = int(ok.sum())
    chi2 = float(np.mean(z[:-1][ok] ** 2)) if ndf else 0.0
    null = stats.t_null(int(seeds))
    spread = ndf <= 1 or stats.chi2_passes(chi2, ndf, null)
    passes = bool(abs(z[-1]) <= stats.Z_MAX and spread)
    return GradientReport(float(jnp.ravel(th0)[0]) if th0.ndim else float(th0), h, int(seeds), ad.mean(0),
                          stats.sem(ad), fd.mean(0), stats.sem(fd), z, resolved, chi2, ndf, null, passes)  # fmt: skip
