"""The gradient check: forward-mode derivatives against finite differences on the same random numbers, per family.

    rows = gradients.check("sbnd", reverse=True)       # the manifest's "sbnd" entry: its sources and wavelength

Families are discovered from the detector's packed parameters (only declared ones exist: docs/dev/design.md section
6.2): every medium's absorption and scattering lengths and re-emission constants (multiplied: v (1 + th)), every surface
type's parameters per row: a probability parameter (the surface type's ``domain``: a fraction in [0, 1], a film's yield
or thickness in [0, inf)) added at min(v - lo, hi - v) (a row at an edge of its domain: one-sided, inward), others
multiplied. Refractive indices are not differentiated (moving_paths+reweight, not available in 0.1).

Per family, on K replicas of ``photons`` photons (one run; the same photons and keys on every side):

- AD: ``jax.jvp`` of the per-sensor charge Q and time-weighted charge Tw, summed in ``groups`` groups of sensors;
- FD (each side sampled at its own values: the expectation's derivative), Richardson at h = 1/16, and the paired
  z of AD - FD in total, with the groups' mean z^2 against the t null (K - 1) / (K - 3); a group whose median replica
  sees no FD jump (|FD| < AD / 10) is "FD unresolved" and leaves the mean (the median rule);
- FDX (every side sampled at the nominal: the estimator's own realisation), Richardson at h = 1/1024: AD / FDX = 1.
SPREAD: the groups' mean z^2 passes below null + 3 sqrt((kappa - 1) / ndf), kappa the per-replica FD values' pooled
kurtosis (``kurtosis``, at least 3: the Gaussian band for a light-tailed family).
A family at an edge takes the inward one-sided stencil on an h-scan (h / 4 until F(h) agrees with both F(4h) and
F(h / 4)), the fixed-reference stencil when it moves no draw. A family whose AD is exactly 0 while its outcome is
drawn (the edge rows proposed it, or it moves no draw) is "zero" (``judge_zero``): gated on the FD and FDX totals
being 0, not on the groups' spread, which then measures only FD noise. Russian roulette is off in the comparison.
``reverse``: ``jax.grad`` of a fixed random weighting of Q and Tw against the same weighting of the forward
derivatives (lumen's own reverse check; tallymap's runtime replay is not used).
"""

from __future__ import annotations

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from lumen import run as runner
from lumen import sources
from lumen.transport import estimator, item
from lumen.transport import step as steps
from lumen.validate import photons, records, references, stats

DEFAULTS = dict(photons=16384, replicas=32, groups=50, h=1.0 / 16, hx=1.0 / 1024,
                edge_hs=(1 / 16, 1 / 64, 1 / 256, 1 / 1024, 1 / 4096, 1 / 16384))  # fmt: skip
FLOOR = 1e-5  # the sides' own float32 rounding, relative to the FD
FLOOR_V = 1e-7  # an FD's float32 resolution: FLOOR_V |f| / h
EXACT = 1e-4  # |AD / FDX - 1|, besides the FDX's float32 resolution
TRUNCATION_STEPS = 2  # h-scan steps of a family whose FD resolves its own truncation (1/16 -> 1/256)
# ZERO: a family whose AD is exactly 0 on every output while the edge rows proposed its outcome (or it moves no draw)
# is structurally zero (a dead end: ANNIE's LAPPD glass T); an exactly-0 AD with the outcome never proposed (the
# edge rows missing) is not, and stays gated as any family
INFINITE_VARIANCE = ("polish",)  # GLISUR's polish: unbiased, infinite variance (docs/dev/design.md section 9)
NO_DRAW = 1e-5  # shared- and fixed-reference FD closer than this (relative): the family moves no draw
REVERSE = 1e-4  # |reverse - forward| relative to the family's forward derivative
CONVERGED = 2.0  # |z| of F(4h) - F(h) below which the edge h-scan stops
POSITIVE_MEDIA = ("absorption", "length", "yield", "tau", "eta")
LENGTH_MAX = 1e6  # m: a medium whose lengths all exceed this transports no light of that kind here
F32 = jnp.float32


def families(optics, p0) -> list:
    """Return [(name, keys, row, mode, scale)] over the packed parameters (see the module docstring)."""
    out = []
    for path, v in jax.tree_util.tree_flatten_with_path(p0)[0]:
        keys = tuple(getattr(q, "key", getattr(q, "idx", None)) for q in path)
        a = np.asarray(v, np.float64)
        if keys[-1] == "energy" or not np.issubdtype(a.dtype, np.floating):
            continue
        if keys[0] == "media":
            fin = np.isfinite(a) & (a > 0)
            if (
                "index" not in keys
                and any(s in keys for s in POSITIVE_MEDIA)
                and fin.any()
                and not np.all(a[fin] >= LENGTH_MAX)
            ):
                out.append((keys[1].split("0x")[0] + ":" + "/".join(str(k) for k in keys[2:] if k != "value"), keys,
                            None, "mul", None))  # fmt: skip
        elif keys[0] == "kernel":
            rows = optics.rows.get(keys[1])
            dom = estimator.domain(optics, keys[1], keys[2])
            for r in range(a.shape[0]) if a.ndim else (None,):
                ar = a if r is None else a[r]
                nm = f"{keys[1]}.{keys[2]}[{(str(rows[r]) if rows is not None and r is not None else str(r))[:40]}]"
                if dom is not None:  # a probability parameter (the type's domain): added inside it
                    lo, hi = dom
                    c = np.minimum(ar - lo, hi - ar)
                    if np.all(c > 0) or not (np.all(ar <= lo) or np.all(ar >= hi)):
                        if np.any(c > 0):
                            out.append((nm, keys, r, "add", np.clip(c, 0.0, None)))
                    else:  # at an edge: one-sided, inward
                        out.append((nm, keys, r, "edge", np.ones_like(ar) if np.all(ar <= lo) else -np.ones_like(ar)))
                elif np.any(ar != 0):
                    out.append((nm, keys, r, "mul", None))
    return out


def _get(tree, keys):
    for k in keys:
        tree = tree[k]
    return tree


def _set(tree, keys, v):
    return v if not keys else {**tree, keys[0]: _set(tree[keys[0]], keys[1:], v)}


def apply(p0, fams, th):
    """Return the packed parameters at th (one value per family, traced)."""
    p = p0
    for j, (_, keys, row, mode, scale) in enumerate(fams):
        v = _get(p, keys)
        dv = v * th[j] if mode == "mul" else jnp.asarray(scale, v.dtype) * th[j]
        p = _set(p, keys, v + dv if row is None else v.at[row].add(dv if mode != "mul" else dv[row]))
    return p


def observable(s, p0, fams, rows, o):
    """Return (obs, ops): obs(th, thr, ops) -> (Q, Tw), each (K, sensors), at th, sampled at the reference thr.

    ``ops`` (the operands at the nominal values) is an argument of the jitted calls, so the navigator's arrays and
    the parameters are never constants of the program.
    """
    spec, k = item.photons(s.tally), o["replicas"]
    key = photons.raw_key(900)

    def obs(th, thr, ops):
        p = ops.params
        live = ops._replace(params=apply(p, fams, th), reference=apply(p, fams, jax.lax.stop_gradient(thr)))
        t = runner.run(spec, s.model, live, rows, key, replicas=k)
        return t.outputs["Q"].reshape(k, s.n_sensors, -1).sum(-1), t.outputs["Tw"].reshape(k, s.n_sensors, -1).sum(-1)

    return obs, photons.operands(s, p0, None, steps.Controls(roulette=0.0))


def groups(a, ng):
    """Return (..., ng + 1): ng groups of consecutive sensors, and the total last."""
    idx = np.array_split(np.arange(a.shape[-1]), min(ng, a.shape[-1]))
    return np.concatenate([np.stack([a[..., i].sum(-1) for i in idx], -1), a.sum(-1, keepdims=True)], -1)


def compare(ad, fd, f0, h, n):
    """Return AD against an FD, per replica (K, G + 1): totals, paired z, the groups' mean z^2 and its null."""
    k = ad.shape[0]
    dif = ad - fd
    fl = FLOOR_V * np.abs(f0.mean(0)) / h
    sd = np.sqrt(stats.sem(dif) ** 2 + (FLOOR * np.abs(fd.mean(0))) ** 2 + fl**2)
    z = np.where(sd > 0, dif.mean(0) / np.where(sd > 0, sd, 1.0), 0.0)
    unres = (np.median(np.abs(fd[:, :-1]), 0) < 0.1 * np.abs(ad[:, :-1].mean(0))) & np.any(ad[:, :-1] != 0.0, 0)
    ok = (sd[:-1] > 0) & ~unres
    return dict(ad=ad[:, -1].mean() / n, ad_e=stats.sem(ad)[-1] / n, fd=fd[:, -1].mean() / n,
                fd_e=stats.sem(fd)[-1] / n, z=float(z[-1]), chi2=float(np.mean(z[:-1][ok] ** 2)) if ok.any() else 0.0,
                ndf=int(ok.sum()), null=stats.t_null(k), unresolved=int(unres.sum()),
                kappa=kurtosis(fd[:, :-1][:, ok]), floor=float(fl[-1]) / n,
                zmax=float(np.max(np.abs(z[:-1][ok]))) if ok.any() else 0.0)  # fmt: skip


def kurtosis(x):
    """Return the replicas' standardised fourth moment, pooled over the groups (columns of x, K rows), at least 3.

    The spread band's noise model (``spread_passes``), from the per-replica FD values alone (never AD - FD or the
    groups' z: a real per-group bias must not widen its own band). A heavy-tailed FD (photons whose paths the FD's
    sides change) widens the band; 3 (Gaussian) is the floor, so a light-tailed family keeps the Gaussian band.
    """
    c = x - x.mean(0)
    m2, m4 = np.mean(c**2, 0), np.mean(c**4, 0)
    good = m2 > 0
    return float(max(np.mean(m4[good] / m2[good] ** 2), 3.0)) if good.any() else 3.0


def spread_passes(c) -> bool:
    """Return whether the groups' mean z^2 is within its band: null + 3 sqrt((kappa - 1) / ndf) (``SPREAD``).

    Var(z^2) = kappa - 1 for z of kurtosis kappa: 2 for Gaussian z (the Pearson band of ``stats.chi2_passes``);
    heavier tails widen it by the family's own estimate.
    """
    k = c.get("kappa", 3.0)
    return bool(c["ndf"] == 0 or c["chi2"] <= c["null"] + max(3.0 * np.sqrt((k - 1.0) / c["ndf"]),
                                                              stats.CHI2_FLOOR * c["null"]))  # fmt: skip


class _Sides:
    """The value calls of one family, each side computed once: FD (shared reference) and FDX (the nominal)."""

    def __init__(self, val, nf, base, ng):
        self.val, self.nf, self.base, self.ng, self.cache = val, nf, base, ng, {}  # val(th, thr) -> (Q, Tw)

    def at(self, j, x, fixed):
        if x == 0.0:
            return self.base
        if (j, x, fixed) not in self.cache:
            th = jnp.zeros(self.nf, F32).at[j].set(F32(x))
            q, tw = self.val(th, jnp.zeros(self.nf, F32) if fixed else th)
            self.cache[(j, x, fixed)] = (np.asarray(q, np.float64), np.asarray(tw, np.float64))
        return self.cache[(j, x, fixed)]

    def fd(self, j, h, one_sided, fixed):
        sides = {x: self.at(j, x, fixed) for x in ((0.0, h / 2, h) if one_sided else (-h, -h / 2, h / 2, h))}
        return np.stack([groups(stats.richardson(lambda x, c=c: sides[x][c], h, one_sided), self.ng) for c in (0, 1)])


def _edge_fd(sides, j, ad, v0, o, n):
    """Return (FD, h, stencil) of an edge family: the inward h-scan, fixed-reference when the family moves no draw.

    The scan goes down ``edge_hs`` (h / 4 each step) and stops at the first h whose FD agrees with both neighbours,
    F(4h) and F(h / 4) (``CONVERGED``): one agreement alone can be noise hiding a truncation (ANNIE's R7081
    dielectric_rt_sensor.T: F(1/16) and F(1/64) agreed, but F(1/64) was 2.2 sigma off the AD at 16x photons, and
    F(1/256), F(1/1024) fell on it; docs/explanation/validation.md).
    """
    fixed, fs = False, []

    def agree(a, b):
        diffs = (a[c][:, -1] - b[c][:, -1] for c in (0, 1))
        return max(abs(float(np.mean(d) / max(stats.sem(d), 1e-300))) for d in diffs) < CONVERGED

    for h in o["edge_hs"]:
        f = sides.fd(j, h, True, fixed)
        if not fs and float(np.max(np.abs(f - sides.fd(j, h, True, True))) / max(np.max(np.abs(f)), 1e-30)) < 1e-5:
            fixed, f = True, sides.fd(j, h, True, True)
        fs.append((f, h))
        if len(fs) >= 3 and agree(fs[-3][0], fs[-2][0]) and agree(fs[-2][0], fs[-1][0]):
            f, h = fs[-2]
            return f, h, "fixed-reference" if fixed else "shared-reference"
    f, h = fs[-1]
    return f, h, "fixed-reference" if fixed else "shared-reference"


def _fd_passes(ad, fd, v0, h, n):
    return all(abs(c["z"]) <= stats.Z_MAX and spread_passes(c)
               for c in (compare(ad[k], fd[k], v0[k], h, n) for k in (0, 1)))  # fmt: skip


def _truncation_scan(sides, j, ad, fd, h, v0, n):  # noqa: PLR0913 -- the family's pieces
    """Return (FD, h, note): h / 4 while AD - FD(h) fails and FD(h) - FD(h / 4) is resolved (the truncation is seen).

    A family whose FD is nearly free of noise (few or no flipped draws) can resolve the stencil's truncation at
    h = 1/16; a real AD error does not shrink with h, so it fails at every h. At most ``TRUNCATION_STEPS`` steps.
    """
    note = ""
    for _ in range(TRUNCATION_STEPS):
        if _fd_passes(ad, fd, v0, h, n):
            break
        fd2 = sides.fd(j, h / 4, False, False)
        if _fd_passes(fd, fd2, v0, h / 4, n):  # FD(h) and FD(h / 4) agree: the failure is not the truncation
            break
        fd, h, note = fd2, h / 4, f"; truncation resolved: h 1/{4 / h:.0f}"
    return fd, h, note


def _exact(c):
    return abs(c["ad"] - c["fd"]) <= EXACT * abs(c["fd"]) + 3 * max(c["floor"], c.get("rounding", 0.0))


def _with_rounding(c, c4):
    """Return c with FDX's own float32 error estimated from FDX at h / 4 (``c4``): |FDX(h / 4) - FDX(h)| / 3.

    The fixed-reference sides evaluate the parameter's terms (log sigma, alpha^2 / sigma^2, ...) at v (1 +- h) in
    float32: each such evaluation rounds the same way on every photon, so the error is a bias that adds up along a
    photon's reflections and grows as 1 / h (e.g. 2.5e-4 per unified-lobe reflection in d log w at h = 1/1024,
    none in float64). Rounding that scales as 1 / h makes FDX(h / 4) - FDX(h) three times FDX(h)'s error.
    """
    return dict(c, rounding=abs(c4["fd"] - c["fd"]) / 3.0)


def _fdx(sides, j, ad, v0, n, hx, one_sided):  # noqa: PLR0913 -- the family's pieces
    """Return (FDX, its h, FDX at h / 4): h / 4 while AD and FDX(h) disagree and FDX(h) - FDX(h / 4) is resolved.

    The realisation of a photon trapped between reflectors for thousands of bounces is steep in a parameter
    ((1 + c th)^n), so FDX at 1/1024 can still carry its stencil's truncation; an edge family starts at the h its
    FD's h-scan converged at. An AD that misses a term disagrees with a converged FDX at every h. FDX at h / 4 also
    gives FDX's own float32 error (``_with_rounding``).
    """
    fx, f4 = sides.fd(j, hx, one_sided, True), sides.fd(j, hx / 4, one_sided, True)
    for _ in range(TRUNCATION_STEPS):
        c = [compare(ad[k], fx[k], v0[k], hx, n) for k in (0, 1)]
        c4 = [compare(ad[k], f4[k], v0[k], hx / 4, n) for k in (0, 1)]
        if all(_exact(_with_rounding(a, b)) for a, b in zip(c, c4, strict=True)):
            break
        if all(_exact(compare(fx[k], f4[k], v0[k], hx / 4, n)) for k in (0, 1)):  # converged: AD differs
            break
        fx, f4, hx = f4, sides.fd(j, hx / 16, one_sided, True), hx / 4
    return fx, hx, f4


def judge(cq, ct, xq, xt, no_draw, exact_gated=True):
    """Return (pass, note) of one family: FDX exactness, then the FD's paired z and the groups' mean z^2.

    ``cq``, ``ct``: AD against FD (charge, time-weighted charge); ``xq``, ``xt``: AD against FDX. Exactness allows
    the FDX's own float32 error (three times the larger of ``FLOOR_V |f| / h`` and the estimate from FDX at h / 4,
    ``_with_rounding``) besides ``EXACT``. A family that moves
    no draw (``no_draw``: its FD with the shared reference is the fixed-reference function) has no statistics to
    test: its FD is the FDX's function at a coarser h, so only exactness is gated. ``exact_gated`` False: a
    parameter whose estimator has infinite variance (``INFINITE_VARIANCE``), whose realisation a finite difference
    does not resolve; its exactness is reported, its FD test gated.
    """
    if all(max(abs(c["ad"]), abs(c["fd"])) <= 3 * c["floor"] for c in (cq, ct)):
        return None, "below float32 resolution"
    exact = all(_exact(c) for c in (xq, xt))
    note = "" if exact_gated else "infinite variance (docs/explanation/known-limits.md): AD / FDX reported; "
    exact = exact or not exact_gated
    fd_ok = no_draw or all(abs(c["z"]) <= stats.Z_MAX and spread_passes(c)
                           for c in (cq, ct))  # fmt: skip
    note += ("" if exact else "AD / FDX off; ") + ("" if fd_ok else "AD - FD off; ")
    note += "moves no draw" if no_draw else ""
    return bool(exact and fd_ok), note


def is_zero(ad, proposed, no_draw):
    """Return whether a family is structurally zero (``ZERO``): AD exactly 0 on every output, its outcome drawn."""
    return all(np.all(np.asarray(v) == 0.0) for v in ad) and (no_draw or proposed > 0)


def _verdict(ad, proposed, comparisons, draws, exact_gated):
    """Return (zero, pass, note): a structurally zero family (``ZERO``) is judged on its totals, others by judge.

    ``draws``: (the family moves no draw, its edge stencil is fixed-reference).
    """
    no_draw, fixed = draws
    if is_zero(ad, proposed, no_draw or fixed):
        return (True, *judge_zero(*comparisons, proposed, no_draw or fixed))
    return (False, *judge(*comparisons, no_draw, exact_gated=exact_gated))


def judge_zero(cq, ct, xq, xt, proposed, no_draw):
    """Return (pass, note) of a structurally zero family (``ZERO``): AD exactly 0 on every output, its outcome drawn.

    Both FD and FDX then measure only noise and rounding: the FD total must be consistent with 0 (|FD| <= Z_MAX of
    its own error) and FDX 0 within its rounding (``_with_rounding``); the groups' spread is not applied. An AD that
    misses a term still fails on the FD total at enough photons.
    """
    fd_ok = all(abs(c["fd"]) <= stats.Z_MAX * max(c["fd_e"], c["floor"]) for c in (cq, ct))
    fdx_ok = all(_exact(c) for c in (xq, xt))
    why = "moves no draw" if no_draw else f"the edge rows proposed its outcome {proposed} times"
    note = f"zero: AD exactly 0, {why}; " + ("" if fd_ok else "FD total off 0; ") + ("" if fdx_ok else "FDX off 0; ")
    return bool(fd_ok and fdx_ok), note


def _moves_no_draw(sides, j, h, f):
    """Return whether a family's FD with the shared reference equals the fixed-reference one (no draw moves)."""
    fx = sides.fd(j, h, False, True)
    return float(np.max(np.abs(f - fx)) / max(np.max(np.abs(f)), 1e-30)) < NO_DRAW


class _Bench(NamedTuple):
    """One compiled observable: the setup with its edge rows, the value and forward calls, the FD sides."""

    s: object
    obs: object
    ops: object
    jv: object
    sides: _Sides
    v0: np.ndarray
    proposed: int = 0


def _bench(det, p0, fams, edge, spec, o):
    """Return the _Bench of a step config's edge rows (the photons are the same for every one)."""
    from lumen.validate import photons as ph

    s = ph.setup(det, step=steps.StepConfig(edge_rows=edge))
    obs, ops = observable(s, p0, fams, _rows(s, spec, o), o)
    val_ = jax.jit(obs)
    jv_ = jax.jit(lambda th, e, o_: jax.jvp(lambda t_: obs(t_, t_, o_), (th,), (e,))[1])
    th0 = jnp.zeros(len(fams), F32)
    base = tuple(np.asarray(v, np.float64) for v in val_(th0, th0, ops))
    sides = _Sides(lambda th, thr: val_(th, thr, ops), len(fams), base, o["groups"])
    return _Bench(s, obs, ops, lambda th, e: jv_(th, e, ops), sides, np.stack([groups(b, o["groups"]) for b in base]),
                  _proposed(s, _rows(s, spec, o), ops, o))  # fmt: skip


def _proposed(s, rows, ops, o):
    """Return how many photons drew an outcome the edge rows floored, at the nominal values (0 without edge rows).

    Such a photon carries z = p / eps, exactly 0 in value (``lumen.transport.estimator``): its final z is 0.
    """
    if not s.model.step.edge_rows:
        return 0
    spec = item.photons(s.tally)._replace(per_item=lambda ph: dict(z=ph.z))
    t = runner.run(spec, s.model, ops, rows, photons.raw_key(900), replicas=o["replicas"])
    return int((np.asarray(t.per_item["z"]) == 0.0).sum())


def check(name, *, detector=None, photons=None, replicas=None, reverse=False, only=(), extra=(), log=None,  # noqa: PLR0913 -- the check's settings
          shard=None) -> list:  # fmt: skip
    """Run the gradient check named ``name`` (a manifest entry); return its record rows (``log``: a line per family).

    Each family is differentiated alone (wrt = that family), so its edge rows are its own: the families with none
    share one compiled loop, each edge family has its own. ``shard`` = (i, n): only the families j with j % n == i,
    and no summary row (``merge`` makes it from the shards' rows); the reverse rows go with shard 0. The photons,
    keys and statistics are the same in every shard, so a family's row does not depend on the sharding.
    """
    import lumen
    from lumen.validate import photons as ph

    spec = references.spec("gradients", name, extra)
    o = DEFAULTS | {k: v for k, v in dict(photons=photons, replicas=replicas).items() if v is not None}
    t0 = time.time()
    det = lumen.load(detector or spec["detector"])
    p0 = ph.packed(det)
    fams = [f for f in families(det.parts.optics, p0) if not only or f[0].startswith(only)]
    nf, ng, n = len(fams), o["groups"], o["photons"]
    th0 = jnp.zeros(nf, F32)
    benches, out, fwd = {}, [], []
    mine = range(nf) if shard is None else range(shard[0], nf, shard[1])
    for j, (nm, keys, row, mode, _) in enumerate(fams):
        if j not in mine:
            continue
        t1 = time.time()
        edge = estimator.edge_rows(det.parts.optics, p0, [keys[:3] + (row,)] if keys[0] == "kernel" else [])
        if edge not in benches:
            benches[edge] = _bench(det, p0, fams, edge, spec, o)
        b = benches[edge]
        d = b.jv(th0, th0.at[j].set(1.0))
        fwd.append(tuple(np.asarray(v, np.float64) for v in d))
        ad = np.stack([groups(v, ng) for v in fwd[-1]])
        sides, v0 = b.sides, b.v0
        if mode == "edge":
            fd, h, stencil = _edge_fd(sides, j, ad, v0, o, n)
            no_draw = False
        else:
            fd, h = sides.fd(j, o["h"], False, False), o["h"]
            no_draw = _moves_no_draw(sides, j, h, fd)
            stencil = "fixed-reference (moves no draw)" if no_draw else "shared-reference"
            if not no_draw:
                fd, h, scan = _truncation_scan(sides, j, ad, fd, h, v0, n)
                stencil += scan
        fdx, hx, fdx4 = _fdx(sides, j, ad, v0, n, min(o["hx"], h) if mode == "edge" else o["hx"], mode == "edge")
        cq, ct = compare(ad[0], fd[0], v0[0], h, n), compare(ad[1], fd[1], v0[1], h, n)
        xq, xt = (_with_rounding(compare(ad[k], fdx[k], v0[k], hx, n), compare(ad[k], fdx4[k], v0[k], hx / 4, n))
                  for k in (0, 1))  # fmt: skip
        zero, passed, note = _verdict(fwd[-1], b.proposed, (cq, ct, xq, xt), (no_draw, stencil.startswith("fixed")),
                                      keys[2] not in INFINITE_VARIANCE)  # fmt: skip
        adfdx = [
            float(ad[c][:, -1].sum() / fdx[c][:, -1].sum()) if fdx[c][:, -1].sum() else float("nan") for c in (0, 1)
        ]
        out.append(records.row("gradients", name, f"dQ {nm}", cq["ad"], cq["ad_e"], z=cq["z"], passed=passed,
                               reference="finite differences on the same random numbers", mode=mode, h=h,
                               stencil=stencil, hx=hx, charge=cq, time_weighted=ct, ad_over_fdx=adfdx, fdx_charge=xq,
                               fdx_time_weighted=xt, note=note, zero=zero, proposed=b.proposed,
                               seconds=time.time() - t1, shard_of=shard[1] if shard else 1, families_total=nf,
                               family=j, settings=o))  # fmt: skip
        if log:
            log(
                f"{name} {nm}: AD {cq['ad']:+.4e} FD {cq['fd']:+.4e} z {cq['z']:+.2f}, AD / FDX {adfdx[0]:.6f}, "
                f"{'reported' if passed is None else 'pass' if passed else 'FAIL'} ({time.time() - t1:.0f} s)"
            )
    if reverse and (shard is None or shard[0] == 0):
        b = benches.get(()) or _bench(det, p0, fams, (), spec, o)
        fwd0 = [tuple(np.asarray(v, np.float64) for v in b.jv(th0, th0.at[j].set(1.0))) for j in range(nf)]
        out += _reverse(name, b.obs, b.ops, fwd0, b.s.n_sensors, o)
    if shard is not None:
        return out
    return out + [summary(name, out, nf, time.time() - t0, o)]


def summary(name, rows, nf, seconds, settings):
    """Return the "families passing" row of a check's family and reverse rows."""
    gated = [r for r in rows if r["pass"] is not None]
    fam = [r for r in gated if r["quantity"].startswith("dQ ")]
    return records.row("gradients", name, "families passing", sum(r["pass"] for r in fam), passed=all(
        r["pass"] for r in gated), reference="finite differences on the same random numbers", families=nf,
        gated=len(fam), reported=nf - len(fam), seconds=seconds, settings=settings)  # fmt: skip


def merge(name, shards) -> list:
    """Return a sharded check's rows (``shards``: the shards' row lists) with its summary row.

    Refuses shards that do not cover the families exactly once, or that ran with different settings.
    """
    rows = [r for s in shards for r in s if r["quantity"] != "families passing"]
    ex = [r["extra"] for r in rows if r["quantity"].startswith("dQ ")]
    fam = [e.get("family") for e in ex]  # the family's index (two families' labels can coincide when truncated)
    if len(fam) != len(set(fam)):
        raise ValueError(f"gradients {name}: a family appears in two shards")
    n = {e.get("shard_of") for e in ex}
    if len(n) != 1 or None in n or len(shards) != n.pop():
        raise ValueError(f"gradients {name}: {len(shards)} shard files, not one per shard of one sharding")
    nf = ex[0]["families_total"]
    if len(fam) != nf:
        raise ValueError(f"gradients {name}: the shards hold {len(fam)} of {nf} families")
    return rows + [summary(name, rows, nf, max(e.get("seconds", 0.0) for e in ex), ex[0].get("settings"))]


def _rows(s, spec, o):
    """Return the check's photons: isotropic from the sources (one drawn per photon), at the entry's wavelength."""
    n = o["photons"] * o["replicas"]
    x = photons.source_points(s, spec)
    src = jax.random.randint(jax.random.PRNGKey(51), (n,), 0, len(x))
    pts = jnp.asarray(x, F32)[src]
    return sources.Point(pts, n_photons=float(n), lam=spec["wavelength"]).emit(None, jax.random.PRNGKey(50), n)


def _reverse(name, obs, ops, fwd, n_sensors, o):
    """Return the reverse-mode rows: grad of sum(a Q + b Tw) against the forward derivatives' same weighting."""
    rng = np.random.default_rng(7)
    a = jnp.asarray(rng.normal(size=(o["replicas"], n_sensors)), F32)
    b = jnp.asarray(rng.normal(size=(o["replicas"], n_sensors)) / 100.0, F32)
    t0 = time.time()

    def loss(th, ops_, wq, wt):
        q, tw = obs(th, th, ops_)
        return jnp.sum(wq * q) + jnp.sum(wt * tw)

    nf = len(fwd)
    g = np.asarray(jax.jit(jax.grad(loss))(jnp.zeros(nf, F32), ops, a, b), np.float64)
    t_rev = time.time() - t0
    f = np.array([np.sum(np.asarray(a) * q) + np.sum(np.asarray(b) * tw) for q, tw in fwd])
    tol = REVERSE * np.abs(f) + 1e-6 * np.max(np.abs(f))
    ok = np.abs(g - f) <= tol
    return [records.row("gradients", name, "reverse mode against forward mode", float(np.max(np.abs(g - f) / np.maximum(
        np.abs(f), 1e-30))), passed=bool(ok.all()), reference="forward mode (jax.jvp)", reverse=g, forward=f,
        families_agreeing=int(ok.sum()), families=nf, seconds=t_rev)]  # fmt: skip
