"""The gradient check: ``simulate``'s derivatives, per parameter family, on the same seeds.

    rows = gradients.check("sbnd", reverse=True)       # the manifest's "sbnd" entry: its sources and wavelength

A family is a parameter key the photon loop reads by reweighting (``Params.info(key).method``, read into the packed
parameters): a probability with a domain (a fraction in [0, 1]) is added at min(v - lo, hi - v) (all at an edge:
one-sided, inward), any other multiplied, v (1 + th) (one-sided, inward, where H_VALUE would leave its declared range: a
group velocity at c). Each family is run alone on K seeds of ``photons`` photons from the entry's sources (isotropic, at
its wavelength), through ``lumen.simulate`` as production runs it. Per seed the outputs are the sensors' arrivals Q and
time-weighted arrivals Tw (each deposit's charge times its time, summed per sensor), summed in ``groups`` groups of
sensors, and the totals. Three tests gate a family; the two statistical ones are ``stats.mean_test`` (the comparisons'
one test: power from the effective number of contributing photons, Student scores, the largest over the outputs against
its correlated null), each at ALPHA / (2 x the check's families), the look-elsewhere of the check:

- **exact**: the forward derivative (AD) against the finite differences of the fixed-reference function
  F(th; reference th0) on the ladder h0 / 2^k, k = 0..3 (central, h0 = ``H``; one-sided inward at an edge, h0 =
  ``H_EDGE``), per seed and output (``stats.fd_order``): a cell whose differences contract at the stencil's order all
  along the ladder is in its asymptotic range and is tested at the fixed step (|AD - FD extrapolated| within ``EXACT``
  |AD|, the sides' float32 rounding ``HALF_ULP`` |F| / h and the extrapolation's own size); a cell that does not
  (pre-asymptotic, or a realisation not smooth at these steps: a moving support edge, square-root terms) is not, and
  its measured order is reported. A row with no cell tested is gated on the other two. With the reference held every
  draw and every roulette decision stays put, so this is the derivative of the estimate itself.
- **value**: F(th0 + H_VALUE; reference th0) against F(th0 + H_VALUE) run at its own values, paired on the same K
  seeds: E[F(th; reference th0)] = E[F(th)] exactly at a finite step (no Taylor term), the identity that makes the
  fixed-reference function's derivative, which the exact test checks, the expectation's. On the same seeds the two
  share their uniforms and F's own seed fluctuation cancels (on disjoint seeds the test measured that fluctuation,
  not the reweighting). Sharing them, the sides take the same decision wherever they can; where their probabilities
  differ (the roulette reads the weight at th0 on one side and at th on the other; an outcome drawn from p_ref
  against p) the side that agrees carries a ratio, an offset, and the compensation is a photon the other side
  resolves the other way. Where the family changes the weights little those flips are rare and K seeds show the
  offsets without them, so the test takes the flips' variance from the offsets (``disagreement``) as its floor.
  Both sides run at the ``TINY`` roulette, where the roulette's offsets are smallest, so the floor holds little
  but the draws' own disagreements and a reweighting bias stays visible. +H_VALUE shrinks a moving support edge
  (polish up), where the identity holds.
- **production**: AD at the default roulette against AD at ``TINY`` (a roulette no photon reaches in practice), paired
  by seed: the roulette moves no derivative. On ``production`` seeds (default 64, at least the K).

The row's value is AD at the default roulette, what production returns, the number the tests gate. The three
results combine by ``stats.verdict``: any failure fails the row (a failed exactness test too: it is deterministic and
needs no power); a statistical test whose outputs all lack power (too few contributing photons for any difference to
show at its level) and an exactness test with no cell tested give no result; a row with no test passing is reported,
as is a family with no effect on this light and no failure (AD at both roulettes below ``F32_EPS`` |F| on every seed:
it moves the totals by less than their float32 resolution). A failing row's note names the test and its statistic.
The summary row passes by ``stats.gate``: some row passes and none fails. The per-seed statistics (group arrays, the
ladder, the photon sums) are kept in the row (``seeds``), so the tests can be recomputed from the records.
``reverse``: ``jax.grad`` of a fixed random weighting of the sensors' arrivals, every family traced, against the same
finite-difference ladder under the exact rule; the row's value is the largest exactness ratio, a family with no cell
in its asymptotic range is not tested, and a row with no family tested is reported.
Why: docs/dev/validation.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

import lumen
from lumen import sources
from lumen._core.domains import FRACTION
from lumen.compare import photons, records, references
from lumen.format.calibrations import Table
from lumen.rules.choices import GRADIENT_EDGE_STEP
from lumen.tools import stats
from lumen.transport import estimator
from lumen.transport import step as steps

DEFAULTS = dict(photons=32768, seeds=16, groups=20, production=64)  # production: that test's seeds (>= seeds)
H = 1.0 / 64  # the finite difference's step (Richardson: O(H^4) central)
H_EDGE = H * GRADIENT_EDGE_STEP  # the one-sided difference's step (O(h^2)): a stated choice (lumen.rules.choices)
H_VALUE = 1.0 / 16  # the value test's step
EXACT = 1e-4  # |AD - FD| relative to AD, per seed and output
FLOOR_V = stats.FD_ROUNDING  # float32 rounding of a stencil's sides relative to their value (the checks' one)
HALF_ULP = 2.0**-24  # a side's rounding at best (a correctly rounded float32): the resolution at which the ladder's
# differences are taken as resolved (``stats.fd_order``'s resolve); FLOOR_V bounds it in the tolerance
TINY = 1e-6  # the roulette of the production test's comparison run and of the value test
F32_EPS = float(np.finfo(np.float32).eps)  # a derivative below F32_EPS |F| per unit th has no effect on this light
LENGTH_MAX = 1e6  # m: a medium length whose values all exceed this transports no light of that kind here
LENGTHS = ("LENGTH", "RAYLEIGH", "MIEHG")  # the medium keys that are lengths (ABSLENGTH, WLSABSLENGTH, ...)
F32 = jnp.float32


def families(det, params) -> list:
    """Return [(key, mode, scale, smooth)]: every reweighted key the photon loop reads (see the module docstring).

    ``mode`` "add" (``scale`` min(v - lo, hi - v), clipped at 0), "edge" (``scale`` +1 at lo, -1 at hi) or "mul";
    ``smooth`` False for a key that moves a lobe's support edge.
    """
    o, read, out = det.parts.optics, {}, []
    for (section, entry, *leaf), src in det.parts.key_rows.items():
        if src.key and src.column != "energy":
            read.setdefault(src.key, []).append((section, entry, tuple(leaf)))
    for key in sorted(read):
        if key not in params or key.startswith(sources.PREFIX):
            continue
        info = params.info(key)
        v = params[key]
        a = np.asarray(v.value if isinstance(v, Table) else v)
        if not info.available or info.method != "reweight" or not np.issubdtype(a.dtype, np.floating):
            continue
        dom, smooth = _domain(o, read[key])
        a = a.astype(np.float64)
        if dom is None:
            far = key.startswith("materials/") and any(s in key.rsplit("/", 1)[-1] for s in LENGTHS)
            if a.any() and not (far and np.all(a[np.isfinite(a) & (a > 0)] >= LENGTH_MAX)):  # all 0: v (1 + th) is 0
                lo, hi = info.range
                if np.any(a * (1 + H_VALUE) > hi) or np.any(a * (1 - H_VALUE) < lo):  # at its range's edge: inward
                    out.append((key, "edge", np.where(a * (1 + H_VALUE) > hi, -a, a), smooth))
                else:
                    out.append((key, "mul", None, smooth))
            continue
        lo, hi = dom
        c = np.minimum(a - lo, hi - a)
        if np.any(c > 0):
            out.append((key, "add", np.clip(c, 0.0, None), smooth))
        elif np.all(a <= lo) or np.all(a >= hi):
            out.append((key, "edge", np.where(a <= lo, 1.0, -1.0), smooth))
    return out


def _domain(o, uses):
    """Return (the (lo, hi) domain of a key read at ``uses`` (section, entry, leaf), or None; whether it is smooth).

    Both come from the one defensive declaration (``lumen.transport.estimator``): a parameter of role "zero" (a share
    that can zero a probability) is a fraction moved additively off its edge; one of role "support" moves a support's
    edge (not smooth at the ladder's steps).
    """
    dom, smooth = None, True
    for section, entry, leaf in uses:
        if section == "media":
            if len(leaf) >= 3 and leaf[1] == "lobe" and estimator.lobe_role(o, entry, leaf[0], leaf[2]) == "zero":
                dom = dom or FRACTION
            continue
        origin = f"{'surface' if section == 'surfaces' else 'sensor'}:{entry}"
        for i, r in enumerate(o.rows):
            if r.origin == origin and leaf:
                role = estimator.role(o, i, leaf[0])
                dom = dom or (r.surface_type.range(leaf[0]) if role == "zero" else None)
                smooth = smooth and role != "support"
    return dom, smooth


def at(params, fam, th):
    """Return ``params`` with family ``fam``'s key moved by th (traced, or a number)."""
    key, mode, scale, _ = fam
    v = params[key]
    a = jnp.asarray(v.value if isinstance(v, Table) else v, F32)
    new = a * (1.0 + th) if mode == "mul" else a + jnp.asarray(scale, F32) * th
    return params.set(key, Table(jnp.asarray(v.energy), new) if isinstance(v, Table) else new)


def groups(a, ng):
    """Return (..., ng + 1): ng groups of consecutive sensors, and the total last."""
    idx = np.array_split(np.arange(a.shape[-1]), min(ng, a.shape[-1]))
    return np.concatenate([np.stack([a[..., i].sum(-1) for i in idx], -1), a.sum(-1, keepdims=True)], -1)


@dataclass(frozen=True)
class _TimeSum:
    """A readout of the deposit records summing each deposit's charge times its arrival time per sensor: the
    time-weighted arrivals Tw (``lumen.diagnostics``, a records readout).

    Summed from the deposits, its derivative is the derivative of the sum also on a sensor with no light at the
    values (Q times the mean arrival time is 0 there, with derivative 0: light an edge row opens is missed).
    """  # noqa: D205

    sensors: int

    def outputs(self):
        return {"tw": (self.sensors, True)}

    def emits(self, dep, weights=()):
        return [("tw", jnp.where(dep.sensor >= 0, dep.sensor, -1), dep.q * dep.time)]


@dataclass(frozen=True)
class _PhotonGroups:
    """A diagnostic keeping each photon's Q and Tw per sensor group: the photon sums the tests' power reads.

    ``group_of``: the group of each sensor (``groups``' split). Cell (photon, group, Q or Tw).
    """

    photons: int
    group_of: tuple

    def outputs(self):
        return {"pg": (self.photons * len(set(self.group_of)) * 2, True)}

    def init(self, ph):
        return jnp.int32(0)

    def update(self, mem, ph, ph2, dep, pid):
        ng = len(set(self.group_of))
        g = jnp.asarray(self.group_of, jnp.int32)[jnp.maximum(dep.sensor, 0)]
        base = jnp.where((dep.sensor >= 0) & (pid < self.photons), (pid * ng + g) * 2, -1)
        return mem, [("pg", base, dep.q), ("pg", jnp.where(base >= 0, base + 1, -1), dep.q * dep.time)]


def photon_sums(c):
    """Return (2, 3, G + 1) of one run's per-photon outputs c (N, G, 2): sum |c|, sum c^2 and the photons that
    contribute, per output (Q and Tw, the groups and the total), the layout of ``groups``."""  # noqa: D205, D209
    c = np.asarray(c, np.float64)
    c = np.concatenate([c, c.sum(1, keepdims=True)], 1).transpose(2, 1, 0)  # (2, G + 1, N)
    return np.stack([np.abs(c).sum(-1), (c * c).sum(-1), (c != 0).sum(-1).astype(np.float64)], 1)


def disagreement(ca, cb):
    """Return (2, G + 1): per output, the variance the paired value test's seeds need not show, sum |d| (c - |d|)
    over the photons of one seed's two runs ca, cb (N, G, 2) on the same uniforms (the layout of ``groups``).

    A photon whose sides differ by d, of light c = max(|a|, |b|), took one decision that the sides resolve with
    different probabilities: on the side that agrees it carries their ratio (the offset d), and with probability
    about |d| / c the other side resolves it the other way, a difference of its light c (unbiased: the offset times
    the agreement equals the flip times its chance). That flip's Bernoulli variance, c^2 p (1 - p) with p = |d| / c,
    is |d| (c - |d|): known from the offsets whether or not a flip was drawn, 0 for a flip that was (already in the
    seeds' spread).
    """  # noqa: D205
    tot = lambda c: np.concatenate([c, c.sum(1, keepdims=True)], 1).transpose(2, 1, 0)  # noqa: E731
    a, b = tot(np.asarray(ca, np.float64)), tot(np.asarray(cb, np.float64))
    d, c = np.abs(a - b), np.maximum(np.abs(a), np.abs(b))
    return (d * np.maximum(c - d, 0.0)).sum(-1)


def _outputs(light):
    """Return (2, S): the arrivals Q and the time-weighted arrivals Tw (``_TimeSum``)."""
    return jnp.stack([light.arrivals, light.diagnostics["tw"]])


class _Runs:
    """The runs of one check: its detector, its source (the entry's points, one photon per row) and its seeds."""

    def __init__(self, det, spec, o):
        self.det, self.o, self.p0 = det, o, det.params()
        x = photons.source_points(photons.setup(det), spec)
        pick = jax.random.randint(jax.random.PRNGKey(51), (o["photons"],), 0, len(x))
        self.src = sources.Point(jnp.asarray(x, F32)[pick], n_photons=1.0, lam=float(spec["wavelength"]))
        ns = len(det.parts.optics.sensor_pvs)
        self.tw = _TimeSum(ns)
        idx = np.array_split(np.arange(ns), min(o["groups"], ns))
        self.pg = _PhotonGroups(
            o["photons"], tuple(int(g) for g in np.repeat(np.arange(len(idx)), [len(i) for i in idx]))
        )

    def _light(self, p, seed, reference, eps, per_photon):
        c = steps.Controls() if eps is None else steps.Controls(roulette=eps)
        diag = (self.tw, self.pg) if per_photon else (self.tw,)
        light = lumen.simulate(self.det, self.src, p, seed=seed, controls=c, reference=reference,
                               _jump=False, diagnostics=diag)  # fmt: skip
        out = _outputs(light)
        return (out, light.diagnostics["pg"]) if per_photon else out

    def value(self, p, seed, reference=None, eps=None, per_photon=False):
        """Return (2, S) at ``p``, drawn at ``reference`` (None: at ``p``), at the roulette ``eps`` (None: default);
        ``per_photon``: also each photon's Q and Tw per group (N, G, 2)."""  # noqa: D205, D209
        out = self._light(p, seed, reference, eps, per_photon)
        if not per_photon:
            return out
        return out[0], np.asarray(out[1], np.float64).reshape(self.o["photons"], -1, 2)

    def jvp(self, fam_p, seed, eps=None, per_photon=False):
        """Return ((2, S) value, (2, S) forward derivative) of th -> value(fam_p(th)) at 0; ``per_photon``: also each
        photon's derivative per group (N, G, 2)."""  # noqa: D205, D209
        if not per_photon:
            return jax.jvp(lambda t: self._light(fam_p(t), seed, None, eps, False), (F32(0.0),), (F32(1.0),))
        (v, _), (d, dpg) = jax.jvp(lambda t: self._light(fam_p(t), seed, None, eps, True), (F32(0.0),), (F32(1.0),))
        return v, d, np.asarray(dpg, np.float64).reshape(self.o["photons"], -1, 2)

    @staticmethod
    def ladder(f, one_sided):
        """Return (f(0) or None, {step: f(step)}) on the finite differences' ladder: +-h0 / 2^k, k = 0..3 (central, h0
        ``H``), or h0 / 2^k and 0 (one-sided, h0 ``H_EDGE``)."""  # noqa: D205, D209
        h0 = H_EDGE if one_sided else H
        hs = [h0 / 2**k for k in range(4)]
        xs = hs if one_sided else hs + [-h for h in hs]
        side = {x: np.asarray(f(F32(x)), np.float64) for x in ((0.0, *xs) if one_sided else xs)}
        return side.pop(0.0, None), side


def family(runs, fam, tests=1) -> dict:
    """Return one family's AD (at the default roulette) and its three tests (see the module docstring).

    ``tests``: the statistical tests of the whole check (the look-elsewhere: each at ALPHA / tests).
    """
    edge, k, n = fam[1] == "edge", runs.o["seeds"], runs.o["photons"]
    g, ladder, sums = _seeds(runs, fam)
    exact, value, prod, passed, note = _tests(g, ladder, sums, fam, k, tests)
    ad_t = g["ad"][:k, 0, -1]
    hm = (H_EDGE if edge else H) / 8  # the ladder's smallest step
    fd_tot = ((ladder[hm] - g["f0"]) / hm if edge else (ladder[hm] - ladder[-hm]) / (2 * hm))[:, 0, -1].mean()
    psums = {n: np.asarray(v).tolist() for n, v in sums.items()}
    seeds = {n: v.tolist() for n, v in g.items()} | {"photon_sums": psums,
             "ladder": {repr(h): v.tolist() for h, v in ladder.items()}}  # fmt: skip
    return dict(
        ad=float(ad_t.mean() / n),
        ad_e=float(stats.sem(ad_t) / n),
        fd=float(fd_tot / n),
        exact=exact,
        value=_brief(value),
        production=_brief(prod),
        passed=passed,
        note=note,
        mode=fam[1],
        seeds=seeds,
    )


def _seeds(runs, fam) -> tuple:
    """Return a family's runs over the seeds: the grouped outputs {name: (seeds, 2, G + 1)} (AD, its TINY run, the
    value, the values at H_VALUE against the reference and their own), the FD ladder {h: (K, 2, G + 1)} and the photon
    sums over the seeds."""  # noqa: D205, D209
    edge, p = fam[1] == "edge", runs.p0
    fp = lambda t: at(p, fam, t)  # noqa: E731
    k, ng = runs.o["seeds"], runs.o["groups"]
    grp = lambda v: groups(np.asarray(v, np.float64), ng)  # noqa: E731
    out: dict = {n: [] for n in ("ad", "tiny", "f0", "v_ref", "v_own")}
    sums: dict = {n: 0.0 for n in ("ad", "tiny", "v_ref", "v_own", "disagree")}  # photon sums over the seeds
    sides = []
    for seed in range(max(k, runs.o.get("production", k))):
        f0, ad, c = runs.jvp(fp, seed, per_photon=True)
        _, tiny, ct = runs.jvp(fp, seed, eps=TINY, per_photon=True)
        out["ad"].append(grp(ad))
        out["tiny"].append(grp(tiny))
        sums["ad"] = sums["ad"] + photon_sums(c)
        sums["tiny"] = sums["tiny"] + photon_sums(ct)
        if seed >= k:
            continue
        out["f0"].append(grp(f0))
        s0, ld = runs.ladder(lambda t, s=seed: runs.value(fp(t), s, reference=p), edge)
        sides.append({h: grp(v) for h, v in ld.items()})
        cs = []
        for n, ref in (("v_ref", p), ("v_own", None)):  # the same seeds: F's own seed fluctuation cancels
            v, c = runs.value(fp(F32(H_VALUE)), seed, reference=ref, eps=TINY, per_photon=True)
            out[n].append(grp(v))
            sums[n] = sums[n] + photon_sums(c)
            cs.append(c)
        sums["disagree"] = sums["disagree"] + disagreement(*cs)
    g = {n: np.stack(v) for n, v in out.items()}  # (seeds, 2, G + 1)
    ladder = {h: np.stack([sd[h] for sd in sides]) for h in sides[0]}
    return g, ladder, sums


def _tests(g, ladder, sums, fam, k, tests) -> tuple:
    """Return a family's (exactness, value test, production test, verdict, note) from its runs (``_seeds``)."""
    edge, smooth = fam[1] == "edge", fam[3]
    exact = stats.fd_order(ladder, g["f0"], g["ad"][:k], H_EDGE if edge else H, edge, FLOOR_V, EXACT, resolve=HALF_ULP)
    count = lambda *ns: np.stack([stats.effective_count(sums[n][:, 0], sums[n][:, 1]) for n in ns])  # noqa: E731
    # the value test sees the family only through the photons its derivative moves (the TINY run's: no roulette
    # thins them): an output where too few carry it cannot show a reweighting error, however bright it is
    floor = np.sqrt(sums["disagree"]).ravel() / k  # the flips the offsets imply, sampled or not (``disagreement``)
    value = stats.mean_test(g["v_ref"], g["v_own"], paired=True, tests=tests, floor=floor,
                            counts=np.minimum(count("v_ref", "v_own"), count("tiny")))  # fmt: skip
    prod = stats.mean_test(g["ad"], g["tiny"], paired=True, counts=count("ad", "tiny"), tests=tests)
    res = F32_EPS * np.abs(g["f0"][:, :, -1])  # moves the totals by less than their float32 resolution, per unit th
    none = all(np.all(np.abs(g[n][:k, :, -1]) <= res) for n in ("ad", "tiny"))
    # a family that moves a lobe's support edge (its role "support") has a realisation with square-root terms
    # sqrt(delta_i + c th) at every scale: one photon with delta_i below the ladder's last step gives an FD growing as
    # h^-1/2 next to its exact, finite AD, hidden in a cell by the other photons' smooth differences (measured on
    # SBND's polish), so no fixed ladder can test it: its exactness is reported, the other two tests gate it
    gates_exact = exact["applies"] and smooth
    exact_ok = exact["ratio"] <= 1.0 if gates_exact else None  # deterministic: needs no power
    stat = lambda n, t: f"{n} off (z {t['stat']:.2f}, p {t['p']:.2g} < {t['level']:.2g})"  # noqa: E731
    fails = ([f"not exact ({exact['ratio']:.3g} on {exact['tested']:.0%} of cells)"] if exact_ok is False else []) + [
        stat(n, t) for n, t in (("value", value), ("production", prod)) if t["passed"] is False
    ]
    notes = (
        []
        if gates_exact
        else [
            f"exactness reported ({exact['ratio']:.2f} on {exact['tested']:.0%} of cells): "
            + (f"FD order {exact['order']:.2f}" if smooth else "moves a support edge")
        ]
    )
    notes += [f"{n} test without power" for n, t in (("value", value), ("production", prod)) if t["passed"] is None]
    passed = stats.verdict([exact_ok, value["passed"], prod["passed"]])  # a failure is never dropped
    if none and passed is not False:
        note, passed = "below float32 resolution (no effect on this light)", None
    elif passed is None:
        note = "; ".join(notes + ["no test has power: reported"])
    else:
        note = "; ".join(notes + fails)
    return exact, value, prod, passed, note


def _brief(t) -> dict:
    """Return a test's record: its statistic, p-value, level, outputs with power, verdict and the per-output scores."""
    return dict(
        z=t["stat"],
        p=t["p"],
        level=t["level"],
        n=t["n"],
        passed=t["passed"],
        scores=np.round(t["z"], 3).tolist(),
        power=t["power"].astype(int).tolist(),
    )


def check(name, *, detector=None, photons=None, seeds=None, reverse=None, only=(), extra=(), log=None,  # noqa: PLR0913 -- the check's settings
          shard=None) -> list:  # fmt: skip
    """Run the gradient check named ``name`` (a manifest entry); return its record rows (``log``: a line per family).

    ``reverse``: also the reverse-mode row (None: the manifest's ``"reverse"``). ``seeds``: the seeds K. ``shard`` =
    (i, n): only the families j with j % n == i, and no summary row (``merge`` makes it from the shards' rows); the
    reverse row goes with shard 0. The photons and seeds are the same in every shard, so a family's row does not
    depend on the sharding.
    """
    spec = references.spec("gradients", name, extra)
    reverse = spec.get("reverse", False) if reverse is None else reverse  # the manifest's, unless overridden
    o = DEFAULTS | {k: v for k, v in dict(photons=photons, seeds=seeds).items() if v is not None}
    t0 = time.time()
    runs = _Runs(lumen.load(detector or spec["detector"]), spec, o)
    fams = [f for f in families(runs.det, runs.p0) if not only or f[0].startswith(only)]
    nf, out = len(fams), []
    for j in range(nf) if shard is None else range(shard[0], nf, shard[1]):
        t1 = time.time()
        r = family(runs, fams[j], tests=2 * nf)  # each family's value and production tests
        out.append(records.row("gradients", name, f"dQ {fams[j][0]}", r["ad"], r["ad_e"], z=r["production"]["z"],
                               passed=r["passed"], reference="fixed-reference finite differences on the same seeds",
                               mode=r["mode"], fd=r["fd"], exact=r["exact"], value_test=r["value"],
                               production=r["production"], note=r["note"], h=H_EDGE if r["mode"] == "edge" else H,
                               seeds=r["seeds"],
                               h_value=H_VALUE, tiny_roulette=TINY, roulette=float(steps.Controls().roulette),
                               seconds=time.time() - t1,
                               shard_of=shard[1] if shard else 1, families_total=nf, family=j, settings=o))  # fmt: skip
        if log:
            verdict = "reported" if r["passed"] is None else "pass" if r["passed"] else "FAIL"
            log(
                f"{name} {fams[j][0]}: AD {r['ad']:+.4e} FD {r['fd']:+.4e} exact {r['exact']['ratio']:.3g} "
                f"(cells {r['exact']['tested']:.2f}) value z {r['value']['z']:.2f} p {r['value']['p']:.2g} "
                f"production z {r['production']['z']:.2f} p {r['production']['p']:.2g}, {verdict} "
                f"({time.time() - t1:.0f} s)"
            )
    if reverse and (shard is None or shard[0] == 0):
        out.append(_reverse(name, runs, fams))
    if shard is not None:
        return out
    return out + [summary(name, out, nf, time.time() - t0, o)]


def summary(name, rows, nf, seconds, settings):
    """Return the "families passing" row of a check's family and reverse rows."""
    fam = [r for r in rows if r["pass"] is not None and r["quantity"].startswith("dQ ")]
    return records.row("gradients", name, "families passing", sum(r["pass"] for r in fam),
        passed=stats.gate(r["pass"] for r in rows), reference="fixed-reference finite differences on the same seeds",
        families=nf, gated=len(fam), reported=nf - len(fam), seconds=seconds, settings=settings)  # fmt: skip


def merge(name, shards) -> list:
    """Return a sharded check's rows (``shards``: the shards' row lists) with its summary row.

    Refuses shards that do not cover the families exactly once, or that ran with different settings.
    """
    rows = [r for s in shards for r in s if r["quantity"] != "families passing"]
    ex = [r["extra"] for r in rows if r["quantity"].startswith("dQ ")]
    fam = [e.get("family") for e in ex]  # the family's index
    if len(fam) != len(set(fam)):
        raise ValueError(f"gradients {name}: a family appears in two shards")
    n = {e.get("shard_of") for e in ex}
    if len(n) != 1 or None in n or len(shards) != n.pop():
        raise ValueError(f"gradients {name}: {len(shards)} shard files, not one per shard of one sharding")
    nf = ex[0]["families_total"]
    if len(fam) != nf:
        raise ValueError(f"gradients {name}: the shards hold {len(fam)} of {nf} families")
    return rows + [summary(name, rows, nf, max(e.get("seconds", 0.0) for e in ex), ex[0].get("settings"))]


def _reverse(name, runs, fams):
    """Return the reverse-mode row: grad of a random weighting of Q (seed 0) against its fixed-reference FD."""
    t0 = time.time()
    a = jnp.asarray(np.random.default_rng(7).normal(size=len(runs.det.parts.optics.sensor_pvs)), F32)

    def loss(th, reference=None):
        p = runs.p0
        for j, f in enumerate(fams):
            p = at(p, f, th[j])
        return jnp.sum(a * runs.value(p, 0, reference=reference)[0])

    th0 = jnp.zeros(len(fams), F32)
    g, l0 = np.asarray(jax.grad(loss)(th0), np.float64), float(loss(th0))
    ratio, ok, applies = [], [], []
    for j, f in enumerate(fams):
        e = f[1] == "edge"
        s0, ld = runs.ladder(lambda t, j=j: np.float64(loss(th0.at[j].set(t), runs.p0)), e)
        x = stats.fd_order(
            {h: np.reshape(v, (1, 1)) for h, v in ld.items()},
            np.reshape(l0, (1, 1)),
            np.reshape(g[j], (1, 1)),
            H_EDGE if e else H,
            e,
            FLOOR_V,
            EXACT,
            resolve=HALF_ULP,
        )
        ratio.append(x["ratio"])
        applies.append(x["applies"])
        ok.append(x["ratio"] <= 1.0 if x["applies"] else None)  # a cell outside its asymptotic range: not tested
    return records.row("gradients", name, "reverse mode against fixed-reference finite differences",
                       float(max(ratio, default=0.0)), passed=stats.verdict(ok), reference="fixed-reference finite "
                       "differences on the same seed", reverse=g, families_agreeing=sum(o is True for o in ok),
                       families_tested=sum(applies), families=len(fams), seconds=time.time() - t0)  # fmt: skip
