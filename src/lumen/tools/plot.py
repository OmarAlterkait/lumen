"""Plots of a run, a comparison, a scan and fits, for notebooks (matplotlib, installed with lumen's ``plot`` extra).

    plot.display(det, values, label="PE")                     # one value per sensor on the sensor map
    plot.event(det, charge, time)                             # the charge and the mean arrival time, side by side
    plot.compare(lumen.compare.event.compare(det, ev, ...))   # lumen against Geant4: per sensor, pulls, times
    plot.compare_events(lumen.compare.event.compare_events(det, evs, ...))   # many events: pulls, pooled
    plot.scan(s, space=space, truth=x, paths={"Adam": r})     # a 1D or 2D lumen.tools.scan, fits' paths on it
    plot.profile(pr, fit=r)  plot.convergence(space, fits, x)  plot.errors(space, fits, x, crb)  plot.pulls(c, space)

The displays draw each sensor where ``lumen.tools.layout`` puts it, from the sensors' positions: a TPC's light
detectors as planes in true scale, a tank's barrel unrolled with its caps above and below, anything else on an
equal-area projection. Units are the keys' own (a factor key: the factor).
"""

from __future__ import annotations

import numpy as np

SIGMA_1 = 0.6827  # the share of a Gaussian within 1 sigma


def _plt():
    import matplotlib.pyplot as plt

    return plt


UNLIT = (0.83, 0.84, 0.86, 1.0)  # a sensor without a value: light grey, under any colour map
EDGE = "0.55"  # the surfaces' outlines and their labels
FONT = 8  # the displays' text (pt)


def _layout(det, layout, select=None):
    from lumen.tools import layout as lay

    if isinstance(layout, lay.Layout):
        return layout
    return lay.of(det if hasattr(det, "sensor_table") else np.asarray(det, np.float64), layout or "auto",
                  select=select)  # fmt: skip


def _drawn(lay, values):
    """Return the values of the sensors the layout draws: every sensor's values, or the drawn ones' already."""
    val = np.asarray(values, np.float64)
    if lay.index is not None and val.shape != (len(lay.xy),) and val.ndim == 1 and len(val) > lay.index.max():
        return val[lay.index]
    return val


def _extent(lay, labels=True):
    """Return the drawn extent (x0, x1, y0, y1): the sensors, the surfaces' outlines and (``labels``) their labels."""
    pts = np.vstack(lay.shapes)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    for o in lay.outlines:
        if o[0] == "rect":
            x0, y0, x1, y1 = min(x0, o[1][0]), min(y0, o[1][1]), max(x1, o[1][0] + o[2]), max(y1, o[1][1] + o[3])
        elif o[0] == "circle":
            (cx, cy), r = o[1], o[2]
            x0, y0, x1, y1 = min(x0, cx - r), min(y0, cy - r), max(x1, cx + r), max(y1, cy + r)
    if labels and lay.labels:
        y1 = max(y1, max(y for _, y, _ in lay.labels) + 1.6 * lay.size)
    return x0, x1, y0, y1


def _frame(lay):
    """Return the display's limits: the drawn extent and a small margin."""
    x0, x1, y0, y1 = _extent(lay)
    pad = 0.01 * max(x1 - x0, y1 - y0)
    return x0 - pad, x1 + pad, y0 - pad, y1 + pad


def _scale(val, lit, *, log, diverging, vmin, vmax, cmap):  # noqa: PLR0913 -- the scale's settings
    """Return the colour scale (a matplotlib norm) and the colour map's name."""
    import matplotlib.colors as mc

    if diverging:
        top = vmax or (float(np.nanmax(np.abs(val[lit]))) if lit.any() else 1.0) or 1.0
        return mc.TwoSlopeNorm(0.0, -top, top), cmap or "RdBu_r"
    if log and lit.any():
        top = vmax or float(val[lit].max())
        return mc.LogNorm(vmin or max(float(val[lit].min()), 1e-3 * top), top), cmap or "viridis"
    lo = vmin if vmin is not None else (float(np.nanmin(val[lit])) if lit.any() else 0.0)
    hi = vmax if vmax is not None else (float(np.nanmax(val[lit])) if lit.any() else 1.0)
    return mc.Normalize(lo, hi if hi > lo else lo + 1.0), cmap or "viridis"


def _surfaces(ax, lay):
    """Draw the surfaces' outlines (the planes' frames, the barrel and its caps) and their labels."""
    from matplotlib.patches import Circle, Ellipse, Rectangle

    for o in lay.outlines:
        if o[0] == "rect":
            ax.add_patch(Rectangle(o[1], o[2], o[3], fill=False, lw=0.6, ec=EDGE, zorder=0))
        elif o[0] == "circle":
            ax.add_patch(Circle(o[1], o[2], fill=False, lw=0.6, ec=EDGE, zorder=0))
        else:
            ax.add_patch(Ellipse((0, 0), 2 * o[1], 2 * o[2], fill=False, lw=0.6, ec=EDGE, zorder=0))
    for x, y, t in lay.labels:
        ax.text(x, y, t, fontsize=FONT, color=EDGE, va="bottom", ha="left")


def _colorbar(ax, lay, norm, cm, label):
    """Add a colour bar beside the display, as tall as what is drawn (not the labels' room)."""
    import matplotlib.colors as mc
    from matplotlib.ticker import LogFormatterSciNotation

    plt = _plt()
    ax.apply_aspect()
    box = ax.get_position()
    fw = ax.figure.get_figwidth()
    _, _, lo, hi = _extent(lay, labels=False)
    to_fig = ax.transData + ax.figure.transFigure.inverted()
    y_lo, y_hi = to_fig.transform([[0.0, lo], [0.0, hi]])[:, 1]
    cax = ax.figure.add_axes((box.x1 + 0.06 / fw, y_lo, 0.12 / fw, y_hi - y_lo))
    cb = ax.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), cax=cax)
    cb.set_label(label, fontsize=FONT)
    if isinstance(norm, mc.LogNorm):
        cb.ax.yaxis.set_minor_formatter(LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(1, 0.4)))
    cb.ax.tick_params(which="both", labelsize=FONT - 1)
    cb.outline.set_linewidth(0.5)


def display(det, values, ax=None, *, title: str = "", label: str = "", log: bool = False,  # noqa: PLR0913 -- the plot's
            cmap: str | None = None, vmin=None, vmax=None, diverging: bool = False, layout=None,
            colorbar=True, select=None):  # fmt: skip
    """Draw one value per sensor (``det.sensor_table()`` order: ``Light.arrivals``') on the detector's flat layout.

    Each sensor is drawn with its own outline where ``lumen.tools.layout`` puts it (a TPC's planes in true scale, a
    tank's barrel unrolled with its caps). ``layout``: a ``Layout`` or a kind ("planes", "cylinder", "sphere"); by
    default the detector's own. ``select``: some sensors only, in detail (``lumen.tools.layout.of``'s; ``values``
    stays every sensor's). Sensors without a value (NaN; or 0 and below unless ``diverging``) are grey.
    ``diverging``: a colour scale centred on 0 (a difference, a pull). Returns the axes.
    """
    from matplotlib.collections import PolyCollection

    plt = _plt()
    lay = _layout(det, layout, select)
    val = _drawn(lay, values)
    if val.shape != (len(lay.xy),):
        raise ValueError(f"display: {val.shape} values for {len(lay.xy)} sensors")
    lit = np.isfinite(val) & (True if diverging else val > 0)
    x0, x1, y0, y1 = _frame(lay)
    if ax is None:
        ax = plt.subplots(figsize=(6.4, float(np.clip(5.8 * (y1 - y0) / (x1 - x0), 1.4, 8.0)) + 0.35))[1]
        ax.figure.subplots_adjust(left=0.01, right=0.9, bottom=0.01, top=0.93)
    norm, name = _scale(val, lit, log=log, diverging=diverging, vmin=vmin, vmax=vmax, cmap=cmap)
    cm = plt.get_cmap(name)
    colors = np.tile(np.array(UNLIT), (len(val), 1))
    colors[lit] = cm(norm(np.clip(val[lit], norm.vmin, norm.vmax)))
    _surfaces(ax, lay)
    order = np.argsort(np.where(lit, np.abs(val) if diverging else val, -np.inf), kind="stable")  # brightest on top
    ax.add_collection(PolyCollection([lay.shapes[i] for i in order], facecolors=colors[order], edgecolors="none"))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=FONT + 1, pad=3)
    if colorbar:
        _colorbar(ax, lay, norm, cm, label)
    return ax


def event(det, charge, time, *, title: str = "", charge_label: str = "expected photoelectrons",  # noqa: PLR0913
          layout=None, select=None):  # fmt: skip
    """Draw an event's charge (log colour) and mean arrival time (ns) per sensor on the detector's layout: side by
    side, or one above the other for a wide layout. A sensor expecting under one hit (photoelectron) shows no time.
    ``select``: some sensors only, in detail (``lumen.tools.layout.of``'s). Returns the figure."""  # noqa: D205, D209
    plt = _plt()
    lay = _layout(det, layout, select)
    x0, x1, y0, y1 = _frame(lay)
    aspect = (y1 - y0) / (x1 - x0)
    stacked = aspect < 0.45  # a wide layout: one above the other
    w = 10.0
    pw = w * (0.86 if stacked else 0.41)  # each display's width (in), its colour bar beside it
    ph = pw * aspect
    fig, (a, b) = plt.subplots(2 if stacked else 1, 1 if stacked else 2,
                               figsize=(w, (2 * ph + 0.75 if stacked else ph + 0.4)))  # fmt: skip
    fig.subplots_adjust(left=0.01, right=0.94 if stacked else 0.95, bottom=0.01, top=0.96 if stacked else 0.92,
                        wspace=0.18, hspace=0.12)  # fmt: skip
    q, t = _drawn(lay, charge), _drawn(lay, time)
    head = f"{title}  " if title else ""
    display(det, q, a, title=f"{head}charge", label=charge_label, log=True, layout=lay)
    lit = q >= 1.0  # a mean time from at least one expected hit or photoelectron
    lo, hi = np.nanpercentile(t[lit], [2, 98]) if lit.any() else (None, None)
    display(det, np.where(lit, t, np.nan), b, title=f"{head}mean arrival time", label="ns", cmap="plasma_r",
            vmin=lo, vmax=hi, layout=lay)  # fmt: skip
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


def compare_events(res: dict, *, title: str = "", names=("lumen", "Geant4")):
    """Draw ``lumen.compare.event.compare_events``' statistics: per event the totals and their pulls; per sensor and
    event the pulls against what an exact match gives (Poisson toys about lumen: skewed where counts are few), and
    Geant4 against lumen summed by expected hits; the arrival times of every event and their ratio. Returns the
    figure."""  # noqa: D205, D209
    plt = _plt()
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.2), constrained_layout=True)
    (a, b, c), (d, e, f) = ax
    mu, n, var = res["lumen"].sum(1), res["geant4"].sum(1), res["lumen_var"].sum(1)
    lo, hi = 0.8 * min(mu.min(), n.min()), 1.25 * max(mu.max(), n.max())
    a.errorbar(n, mu, xerr=np.sqrt(n), yerr=np.sqrt(var), fmt="o", ms=3, lw=0.8)
    a.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    a.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), xlabel=f"{names[1]} hits per event",
          ylabel=f"{names[0]} expected",
          title=f"{len(mu)} events: {names[1]} / {names[0]} {n.sum() / mu.sum():.4f}")  # fmt: skip
    g = np.linspace(-5, 5, 201)
    z = np.asarray(res["event_z"])
    b.hist(np.clip(z, -5, 5), bins=np.linspace(-5, 5, 21), density=True, alpha=0.6)
    b.plot(g, np.exp(-g * g / 2) / np.sqrt(2 * np.pi), "k", lw=1, label="N(0, 1)")
    b.set(xlabel=f"({names[1]} - {names[0]}) / sigma, per event", title=f"per event: mean {z.mean():+.2f}, sd "
          f"{z.std(ddof=1):.2f}")  # fmt: skip
    b.legend(fontsize=8)
    edges = np.linspace(-5, 7, 97)
    p, t = np.asarray(res["pull"]), np.asarray(res["pull_toy"])
    c.hist(np.clip(p, -5, 7), edges, density=True, histtype="step", lw=1.2, label=names[1])
    c.hist(np.clip(t, -5, 7), edges, density=True, histtype="step", lw=1.2, ls="--", color="k",
           label="exact match (Poisson toys)")  # fmt: skip
    c.set(yscale="log", ylim=(1e-5, 1), xlabel="pull per sensor and event",
          title=f"per sensor: mean {p.mean():+.3f} (exact match {t.mean():+.3f}), "
          f"sd {p.std():.3f} ({t.std():.3f})")  # fmt: skip
    c.legend(fontsize=8)
    bins = res["bins"]
    x = np.arange(len(bins))
    d.axhline(0, color="k", lw=0.8)
    d.errorbar(x, [100 * q["diff"] for q in bins], yerr=[100 * q["err"] for q in bins], fmt="o", ms=4, capsize=2)
    d.set_xticks(x, [f"{q['lo']:g}-{q['hi']:g}" if np.isfinite(q["hi"]) else f">{q['lo']:g}" for q in bins],
                 fontsize=8)  # fmt: skip
    d.set(xlabel=f"{names[0]} expected hits per sensor and event",
          ylabel=f"({names[1]} - {names[0]}) / {names[0]} (%)", title="by expected hits, summed")  # fmt: skip
    tt, hl, hg = np.asarray(res["t"]), np.asarray(res["times_lumen"]), np.asarray(res["times_geant4"])
    keep = (hl > 0) | (hg > 0)
    last = int(np.flatnonzero(keep).max()) + 1 if keep.any() else len(tt)
    e.step(tt[:last], hg[:last], where="mid", label=names[1], lw=1.0)
    e.step(tt[:last], hl[:last], where="mid", label=names[0], lw=1.0)
    e.set(yscale="log", xlabel="arrival time (ns)", ylabel="hits per bin, all events", title="arrival times")
    e.legend(fontsize=8)
    # the ratio in bins of equal expected content (about 1 % error each), so the late tail is not noise
    cum = np.cumsum(hl[:last])
    per = max(cum[-1] / 200.0, 1e4)
    groups = np.minimum((cum / per).astype(int), int(cum[-1] // per))
    k = np.unique(groups)
    sl = np.bincount(groups, hl[:last])[k]
    sg = np.bincount(groups, hg[:last])[k]
    tc = np.bincount(groups, tt[:last] * hl[:last])[k] / np.maximum(sl, 1e-30)
    f.axhline(1, color="k", lw=0.8)
    f.errorbar(tc, sg / sl, yerr=np.sqrt(sg) / sl, fmt="o", ms=2.5, lw=0.8)
    chi2 = float(np.sum((sg - sl) ** 2 / np.maximum(sl, 1e-30)))
    f.set(xlabel="arrival time (ns)", ylabel=f"{names[1]} / {names[0]}",
          title=f"arrival times, ratio: chi2/ndf {chi2 / max(len(k), 1):.2f} ({len(k)} bins)")  # fmt: skip
    if title:
        fig.suptitle(title)
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
