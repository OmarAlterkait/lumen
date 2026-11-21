"""Plots of a run, a comparison, a scan and fits, for notebooks (matplotlib, installed with lumen's ``plot`` extra).

    plot.display(det, values, label="PE")                     # one value per sensor on the sensor map
    plot.event(det, charge, time)                             # the charge and the mean arrival time, side by side
    plot.compare(lumen.compare.event.compare(det, ev, ...))   # lumen against Geant4: per sensor, pulls, times
    plot.scan(s, space=space, truth=x, paths={"Adam": r})     # a 1D or 2D lumen.tools.scan, fits' paths on it
    plot.profile(pr, fit=r)  plot.convergence(space, fits, x)  plot.errors(space, fits, x, crb)  plot.pulls(c, space)

The sensor map: each sensor at the azimuth about the longest axis of its sensors' box (through their centre) and the
cosine of its angle to that axis, so a barrel unrolls into a band with its end caps above and below, and planes of
sensors facing each other side by side. Units are the keys' own (a factor key: the factor).
"""

from __future__ import annotations

import numpy as np

SIGMA_1 = 0.6827  # the share of a Gaussian within 1 sigma


def _plt():
    import matplotlib.pyplot as plt

    return plt


def sensor_map(positions) -> tuple:
    """Return (azimuth rad, cos polar angle) of each sensor (positions (S, 3) m; see the module docstring)."""
    x = np.asarray(positions, np.float64)
    c = 0.5 * (x.min(0) + x.max(0))
    ax = int(np.argmax(x.max(0) - x.min(0)))
    d = x - c
    a, b = [(ax + 1) % 3, (ax + 2) % 3]
    r = np.linalg.norm(d, axis=1)
    return np.arctan2(d[:, b], d[:, a]), np.where(r > 0, d[:, ax] / np.where(r > 0, r, 1.0), 0.0)


def _positions(det):
    return det.sensor_table().position if hasattr(det, "sensor_table") else np.asarray(det)


def display(det, values, ax=None, *, title: str = "", label: str = "", log: bool = False,  # noqa: PLR0913 -- the plot's
            cmap: str = "viridis", vmin=None, vmax=None):  # fmt: skip
    """Draw one value per sensor (in ``det.sensor_table()`` order: ``Light.arrivals``') on the sensor map."""
    import matplotlib.colors as mc

    plt = _plt()
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    u, v = sensor_map(_positions(det))
    val = np.asarray(values, np.float64)
    top = vmax or (np.nanmax(val) if np.any(val > 0) else 1.0)
    norm = mc.LogNorm(vmin=vmin or max(val[val > 0].min(), 1e-3 * top), vmax=top) if log and np.any(val > 0) else None
    order = np.argsort(val)  # the brightest on top
    sc = ax.scatter(u[order], v[order], c=val[order], s=float(np.clip(4e4 / len(val), 1.0, 200.0)), cmap=cmap,
                    norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax, linewidths=0)  # fmt: skip
    ax.set(xlabel="azimuth (rad)", ylabel="cos(polar angle)", title=title, xlim=(-np.pi, np.pi), ylim=(-1.02, 1.02))
    ax.figure.colorbar(sc, ax=ax, label=label, shrink=0.85)
    return ax


def event(det, charge, time, *, title: str = "", charge_label: str = "expected photoelectrons"):
    """Draw an event's charge (log colour) and mean arrival time (ns) per sensor, side by side; return the figure."""
    plt = _plt()
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    q, t = np.asarray(charge, np.float64), np.asarray(time, np.float64)
    display(det, q, a, title=f"{title} charge".strip(), label=charge_label, log=True)
    lit = q > 3e-2 * q.max()  # the time of sensors with some light
    lo, hi = np.nanpercentile(t[lit], [2, 98]) if lit.any() else (None, None)
    display(det, np.where(lit, t, np.nan), b, title=f"{title} mean arrival time".strip(), label="ns", cmap="plasma",
            vmin=lo, vmax=hi)  # fmt: skip
    return fig


def compare(res: dict, *, title: str = "", names=("lumen", "Geant4")):
    """Draw ``lumen.compare.event.compare``'s result: per sensor lumen against Geant4, the pulls, the arrival times."""
    plt = _plt()
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    mu, n = res["lumen"], res["geant4"]
    s = res["summary"]
    top = max(mu.max(), n.max()) * 1.5
    a.loglog(n + 0.5, mu + 0.5, ".", ms=2, alpha=0.5)
    a.plot([0.5, top], [0.5, top], "k--", lw=0.8)
    a.set(
        xlabel=f"{names[1]} hits + 0.5",
        ylabel=f"{names[0]} expected + 0.5",
        title=f"{title} per sensor: total {names[0]}/{names[1]} {s['ratio']:.4f} ± {s['err']:.4f}".strip(),
    )
    m = 0.5 * (mu + n) >= 5.0  # the charge check's sensors (MU_MIN)
    pull = (mu[m] - n[m]) / np.sqrt(0.5 * (mu[m] + n[m]))
    g = np.linspace(-5, 5, 201)
    b.hist(np.clip(pull, -5, 5), bins=50, range=(-5, 5), density=True, alpha=0.6, label="sensors")
    b.plot(g, np.exp(-g * g / 2) / np.sqrt(2 * np.pi), "k", lw=1, label="normal (0, 1)")
    b.set(
        xlabel=f"({names[0]} - {names[1]}) / sqrt(mean)",
        title=f"pulls: chi2/ndf {s['chi2'] / max(s['ndf'], 1):.2f} over {s['ndf']} sensors",
    )
    b.legend()
    t = res["t"]
    for h, lab in ((res["times_geant4"], names[1]), (res["times_lumen"], names[0])):
        c.step(t, h / max(h.sum(), 1e-30), where="mid", label=lab)
    c.set(yscale="log", xlabel="arrival time (ns)", ylabel="share per bin", title="arrival times, every sensor")
    c.legend()
    return fig


def _vals(space, x, keys):
    v = space.values(x)
    return np.array([v[k] for k in keys])


def scan(s, *, space=None, truth=None, paths=None, ax=None):
    """Draw a ``lumen.tools.scan``: 1D the loss over its derivative (forward mode and finite differences); 2D the loss
    less its minimum with streamlines of -gradient, the fits' paths ({label: FitResult}, needs ``space``) and the
    truth (a vector of ``space``) as a star; return the figure."""  # noqa: D205, D209
    plt = _plt()
    keys = list(s.axes)
    tv = None if truth is None else _vals(space, truth, keys)
    if len(keys) == 1:
        fig, (a, b) = plt.subplots(2, 1, figsize=(6, 5), sharex=True, constrained_layout=True)
        x = s.axes[keys[0]]
        a.plot(x, s.loss, "o-", ms=3)
        a.set(ylabel="loss")
        b.plot(x, s.grad[0], "-", label="derivative (forward mode)")
        b.plot(x, s.numerical_grad[0], "o", ms=4, mfc="none", label="finite difference")
        b.axhline(0.0, color="k", lw=0.6)
        b.set(xlabel=keys[0], ylabel="d loss / d value")
        b.legend()
        for p in (a, b):
            if tv is not None:
                p.axvline(tv[0], color="k", ls=":", lw=1)
        return fig
    fig, a = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    x, y = (s.axes[k] for k in keys)
    z = (s.loss - s.loss.min()).T
    cs = a.contourf(x, y, z, levels=np.unique(np.quantile(z, np.linspace(0.0, 1.0, 21))), cmap="viridis_r")  # even
    # colour steps however steep the walls
    fig.colorbar(cs, ax=a, label="loss - its minimum")
    a.streamplot(x, y, -s.grad[0].T, -s.grad[1].T, color="w", density=0.8, linewidth=0.6, arrowsize=0.7)
    i, j = np.unravel_index(np.argmin(s.loss), s.loss.shape)
    a.plot(x[i], y[j], "wx", ms=8, label="grid minimum")
    for lab, r in (paths or {}).items():
        pv = np.array([_vals(space, p, keys) for p in r.path])
        a.plot(pv[:, 0], pv[:, 1], ".-", ms=4, lw=1.2, label=lab)
    if tv is not None:
        a.plot(*tv, "*", color="gold", mec="k", ms=14, label="truth")
    a.set(xlabel=keys[0], ylabel=keys[1], xlim=(x.min(), x.max()), ylim=(y.min(), y.max()))
    a.legend(fontsize=8, loc="upper right")
    return fig


def profile(pr, *, fit=None, space=None):
    """Draw a ``lumen.tools.profile`` (the likelihood less its minimum, the 1/2 line of 1 sigma) and a fit's value and
    standard error (a FitResult of ``space``); return the figure."""  # noqa: D205, D209
    plt = _plt()
    fig, a = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    a.plot(pr.values, pr.nll - pr.nll.min(), "o-", ms=4, label="profile likelihood")
    a.axhline(0.5, color="k", ls="--", lw=0.8, label="minimum + 1/2 (1 sigma)")
    if fit is not None:
        v, e = space.values(fit.x)[pr.key], fit.sigma[pr.key]
        a.axvspan(v - e, v + e, color="C1", alpha=0.2, label="fit ± 1 sigma (Fisher)")
        a.axvline(v, color="C1", lw=1)
    a.set(xlabel=pr.key, ylabel="- log likelihood, less its minimum", ylim=(-0.1, max(3.0, 1.1 * np.nanmax(
        np.minimum(pr.nll - pr.nll.min(), 10.0)))))  # fmt: skip
    a.legend(fontsize=8)
    return fig


def convergence(space, fits: dict, truth=None):
    """Draw each key's value along each fit's path ({label: FitResult}), the truth dotted; return the figure."""
    plt = _plt()
    keys = [k for k, n in zip(space.keys, space.sizes, strict=True) if n == 1]
    fig, axs = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 3.3), constrained_layout=True, squeeze=False)
    for a, k in zip(axs[0], keys, strict=True):
        for lab, r in fits.items():
            a.plot([space.values(p)[k] for p in r.path], ".-", ms=3, label=lab)
        if truth is not None:
            a.axhline(space.values(truth)[k], color="k", ls=":", label="truth")
        a.set(xlabel="step", title=k)
    axs[0][0].legend(fontsize=8)
    return fig


def errors(space, fits: dict, truth, crb_cov):
    """Draw each fit's error (fit - truth, with its standard error) against the Cramer-Rao bound (a band of ±1 sigma
    from ``crb_cov``, the inverse Fisher information at the truth), one panel per key; return the figure."""  # noqa: D205, D209
    plt = _plt()
    keys = [k for k, n in zip(space.keys, space.sizes, strict=True) if n == 1]
    bound = space.sigma(crb_cov, truth)
    tv = space.values(truth)
    fig, axs = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.3), constrained_layout=True, squeeze=False)
    for a, k in zip(axs[0], keys, strict=True):
        a.axhspan(-bound[k], bound[k], color="0.85", label="± Cramér-Rao bound")
        for i, (lab, r) in enumerate(fits.items()):
            a.errorbar(i, space.values(r.x)[k] - tv[k], r.sigma[k], fmt="o", capsize=4, label=lab)
        a.axhline(0.0, color="k", lw=0.6)
        a.set(xticks=range(len(fits)), xticklabels=list(fits), title=k, ylabel="fit - truth")
    axs[0][0].legend(fontsize=8)
    return fig


def pulls(cov, space):
    """Draw a ``lumen.tools.coverage``'s pulls per key with the normal density, and each key's 1 sigma coverage."""
    plt = _plt()
    keys = [k for k, n in zip(space.keys, space.sizes, strict=True) if n == 1]
    fig, axs = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.0), constrained_layout=True, squeeze=False)
    g = np.linspace(-4, 4, 161)
    t = len(cov.pulls)
    err = np.sqrt(SIGMA_1 * (1 - SIGMA_1) / t)
    for i, (a, k) in enumerate(zip(axs[0], keys, strict=True)):
        a.hist(np.clip(cov.pulls[:, i], -4, 4), bins=12, range=(-4, 4), density=True, alpha=0.6)
        a.plot(g, np.exp(-g * g / 2) / np.sqrt(2 * np.pi), "k", lw=1)
        a.set(xlabel="(fit - truth) / sigma", title=f"{k}\ncovered {cov.covered[i]:.2f} (expect {SIGMA_1:.2f} ± "
              f"{err:.2f})")  # fmt: skip
    return fig
