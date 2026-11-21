"""The checks' statistics: errors from seeds, the pass rules, finite differences and the two-sample time test.

Rules (docs/dev/architecture.md, "Validation and tests"; docs/dev/decisions.md, 12 and 13):

- a ratio passes when |z| <= ``Z_MAX`` (3);
- a chi2 / ndf passes when it is at most its null + 3 sqrt(Var / ndf), the null and Var the mean and variance of one
  term under the null hypothesis (``chi2_passes``): 1 and 2 for Pearson terms of known variance;
- a per-sensor term whose variance holds lumen's Monte Carlo variance from its photons' second moment (N photons:
  the estimate's own noise is negligible) is Pearson's, 1 and 2;
- K seeds give a t statistic with K - 1 degrees of freedom: E[t^2] = (K - 1) / (K - 3) (``t_null``) and
  Var[t^2] = ``t_var``;
- two Monte Carlo estimates of the same outputs agree when ``mean_test`` passes (the comparisons' one test, below);
- a test reports one of three results: True (passes), False (fails) or None (no power: it cannot reject at its level,
  or its error model has nothing to work with). Several results combine by one rule (``verdict``): a failure fails;
  with none passing (none given, or none with power) there is no verdict, never a pass; else they pass. A gate (a
  summary row, an exit status) passes only on a verdict of True (``gate``): it never passes on nothing.

**The comparisons' test** (``mean_test``). Two samples of per-seed output vectors (sensor groups, totals, Q and Tw),
paired on the same seeds or independent; H0: equal means in every output. One rule, at the one level ``ALPHA`` (the
two-sided tail beyond ``Z_MAX``) shared by the ``tests`` statistical tests of a check (the look-elsewhere: each at
ALPHA / tests):

1. *Power.* An output carried by an effective number n of contributions (Kish, ``effective_count``: per photon when the
   recorder gives photon sums, else per seed) cannot reject even the most extreme alternative, the light entirely
   missing from one sample, unless the chance e^-n that a sample shows none of its n carriers is below the level:
   outputs with n <= ln(tests / ALPHA) have no power and are left out (reported). A row with no output left has no
   power (reported, not gated). The cut is the level's, not a chosen number.
2. *Studentization.* Each output's mean difference over its standard error from the seeds (the paired differences'
   spread, or Welch's), referred to Student's t with min(seeds carrying it, n) - 1 degrees of freedom and no more
   than the per-seed kurtosis allows (``variance_df``: the sample variance's own scatter), as a normal score: the
   variance estimate is only as good as the carriers and the tails it sees.
3. *Look-elsewhere inside the row.* The statistic is the largest |score| over the outputs with power; its null is the
   largest |N(0, R)| with R the outputs' correlation estimated from the seeds (Q and Tw of a group are nearly the same
   light: they count once, by the data, not by a hand count of independent terms).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import kstwobign, norm
from scipy.stats import t as student

from lumen._core.checks import chi2_passes  # noqa: F401 -- the pass rule, shared with the plug-in checks
from lumen.rules.choices import FD_ROUNDING, NULL_DRAWS, SUITE_TESTS, Z_MAX  # noqa: F401 -- stated there

ALPHA = math.erfc(Z_MAX / math.sqrt(2.0))  # the two-sided tail beyond Z_MAX: the checks' one significance level
# FD_ROUNDING: one finite-difference side's float32 rounding relative to its value |F|, the checks' one
# resolution of a difference (``fd_order``, ``lumen.tools.check``): a stencil's 8 sides (3 + 4 + 1 one-sided) each
# correctly rounded at half an ulp, 2^-24, are 4.8e-7; a side summed over photons rounds above that, so twice it
# NULL_DRAWS: mean_test's simulated null, draws per expected exceedance at its level (p resolved to ~10 %);
# SUITE_TESTS: the test suites' statistical assertions sharing ALPHA (``z_limit``), a stated bound on their
# count, loops and parametrizations included; one normal score's threshold is then 4.50, the suites' before


def verdict(results) -> bool | None:
    """Return the combined result of tests' results (True, False, or None for no power) under the one rule: False if
    any fails, None if none passes (an empty set included: no power is never agreement), else True.
    """  # noqa: D205
    results = list(results)
    if any(r is False for r in results):
        return False
    return True if any(r is True for r in results) else None


def gate(results) -> bool:
    """Return whether a gate over tests' results passes: ``verdict`` True. Nothing passing (empty, or no power
    anywhere) fails the gate.
    """  # noqa: D205
    return verdict(results) is True


def ratio_test(mu, var_mu, n, level=ALPHA) -> dict:
    """Return the test that lumen's estimate ``mu`` (Monte Carlo variance ``var_mu``) agrees with a count ``n``.

    The difference mu - n over sqrt(lam + var_mu), lam = (mu + n) / 2 the Poisson variance from both samples' mean (as
    the per-sensor test's): a lumen estimate of 0 against a count has an error and a score, never z = 0. Power: the
    count's effective number of contributions (the count itself) above ``power_floor(level)``, as ``mean_test``'s; a
    count without power gives ``passed`` None (reported). Returns ``ratio`` mu / n (nan for n = 0), its error ``err``
    (the difference's standard error over n), ``z`` and ``passed`` (|z| within the level's two-sided score).
    """
    mu, var_mu, n = float(mu), float(var_mu), float(n)
    se = math.sqrt(0.5 * (mu + n) + var_mu)
    ratio, err = (mu / n, se / n) if n > 0 else (float("nan"), float("nan"))
    z = (mu - n) / se if se > 0 else 0.0
    if n <= power_floor(level):
        return dict(ratio=ratio, err=err, z=z, passed=None)
    return dict(ratio=ratio, err=err, z=z, passed=bool(abs(z) <= norm.isf(level / 2.0)))


def z_limit(outputs: int = 1, df: float | None = None, level: float | None = None) -> float:
    """Return the |score| the largest of ``outputs`` independent scores exceeds with probability ``level`` (Sidak).

    The scores are normal (an error from photon moments, or known), or Student's t with ``df`` degrees of freedom (a
    mean over K seeds against its standard error from them: df = K - 1). Default level: ALPHA / ``SUITE_TESTS``, the
    test suites' one threshold (each statistical assertion's share of the suites' ALPHA).
    """
    level = ALPHA / SUITE_TESTS if level is None else level
    tail = -math.expm1(math.log1p(-level) / int(outputs)) / 2.0
    return float(norm.isf(tail) if df is None else student.isf(tail, df))


def t_null(k: int) -> float:
    """Return E[t^2] of a t statistic over k seeds (k - 1 degrees of freedom)."""
    return (k - 1) / (k - 3) if k > 3 else float("nan")


def t_var(k: int) -> float:
    """Return Var[t^2] of a t statistic over k seeds (nu = k - 1 degrees of freedom): 2 nu^2 (nu - 1) /
    ((nu - 2)^2 (nu - 4)), infinite at nu <= 4 (no band then)."""  # noqa: D205, D209
    nu = k - 1
    return 2.0 * nu * nu * (nu - 1) / ((nu - 2) ** 2 * (nu - 4)) if nu > 4 else float("inf")


def sem(v, axis=0):
    """Return the standard error of the mean over ``axis`` (the seeds)."""
    v = np.asarray(v, np.float64)
    return v.std(axis, ddof=1) / np.sqrt(v.shape[axis])


def richardson(f, h, one_sided=False):
    """Return the derivative of f at 0: O(h^4) central (4 sides), or O(h^2) one-sided inward (f at 0, h/2, h)."""
    if one_sided:
        return (-3.0 * f(0.0) + 4.0 * f(h / 2) - f(h)) / h
    d1 = (f(h) - f(-h)) / (2 * h)
    d2 = (f(h / 2) - f(-h / 2)) / h
    return (4.0 * d2 - d1) / 3.0


def node_split(t, dt, nt):
    """Return (nodes (nt,), late, early): times split onto the nodes k dt linearly, as the checks' ``Detected``
    splits them."""  # noqa: D205, D209
    x = np.asarray(t, np.float64) / dt
    j = np.floor(x).astype(np.int64)
    g = x - j
    h, late, early = np.zeros(nt), 0.0, 0.0
    for jj, w in ((j, 1.0 - g), (j + 1, g)):
        ok = (jj >= 0) & (jj < nt)
        np.add.at(h, jj[ok], w[ok])
        late += float(w[jj >= nt].sum())
        early += float(w[jj < 0].sum())
    return h, late, early


def times(h_lumen, h_ref, dt, n_lumen):
    """Return the time comparison of two node histograms: mean and width each, the mean's difference and error, KS.

    The KS test uses the reference's hit count and lumen's detected weight ``n_lumen`` as the samples' sizes.
    """
    tn = dt * np.arange(len(h_ref))
    sl, sr = h_lumen.sum(), h_ref.sum()
    if sl <= 0 or sr <= 0:
        return dict(t_lumen=np.nan, t_ref=np.nan, dt=np.nan, dt_err=np.nan, ks_d=np.nan, ks_p=np.nan)
    ml, mr = (h_lumen * tn).sum() / sl, (h_ref * tn).sum() / sr
    wl = np.sqrt((h_lumen * (tn - ml) ** 2).sum() / sl)
    wr = np.sqrt((h_ref * (tn - mr) ** 2).sum() / sr)
    d = float(np.max(np.abs(np.cumsum(h_lumen) / sl - np.cumsum(h_ref) / sr)))
    en = np.sqrt(sr * n_lumen / (sr + n_lumen))
    return dict(t_lumen=ml, t_ref=mr, w_lumen=wl, w_ref=wr, dt=ml - mr, dt_err=wr / np.sqrt(sr), ks_d=d,
                ks_p=float(kstwobign.sf(d * en)))  # fmt: skip


def effective_count(sum_abs, sum_sq):
    """Return Kish's effective number of contributions, (sum |c|)^2 / sum c^2 (0 where nothing contributed)."""
    a, q = np.asarray(sum_abs, np.float64), np.asarray(sum_sq, np.float64)
    return np.where(q > 0, a * a / np.where(q > 0, q, 1.0), 0.0)


def seed_count(x):
    """Return the effective number of seeds carrying each output of a sample x (K, m): Kish over the seeds."""
    x = np.asarray(x, np.float64)
    return effective_count(np.abs(x).sum(0), (x * x).sum(0))


def power_floor(level):
    """Return the effective count below which an output cannot reject at ``level``: ln(1 / level) (e^-n = level)."""
    return math.log(1.0 / level)


def variance_df(x):
    """Return the degrees of freedom of each output's sample variance over the seeds of x (K, m), from its kurtosis.

    Var(s^2) / sigma^4 = 2 / (K - 1) + kappa / K (kappa the excess kurtosis), and a chi2 with df degrees of freedom has
    2 / df: df = 2 / (2 / (K - 1) + kappa / K), K - 1 for normal scatter, fewer for heavy tails (a per-seed tail check
    with no threshold: the tails set how far the t calibration can be trusted). At most K - 1.
    """
    x = np.asarray(x, np.float64)
    k = x.shape[0]
    c = x - x.mean(0)
    m2 = (c * c).mean(0)
    kappa = np.where(m2 > 0, (c**4).mean(0) / np.where(m2 > 0, m2 * m2, 1.0) - 3.0, 0.0)
    v = 2.0 / (k - 1) + np.maximum(kappa, 0.0) / k
    return np.minimum(k - 1.0, 2.0 / v)


def _scores(t, df):
    """Return the normal scores of t statistics with df degrees of freedom (same two-sided tail)."""
    p = 2.0 * student.sf(np.abs(t), df)
    return np.sign(t) * norm.isf(np.clip(p, 1e-300, 1.0) / 2.0)


def max_normal_p(z, corr, level, seed=0, m=None):
    """Return P(max |N(0, corr)| >= z): Bonferroni when that already rejects, else simulated.

    The simulation resolves the larger of ``level`` and one output's p-value (a lower bound of the largest's).

    ``corr`` None: m independent outputs (Sidak, exact).
    """
    m = corr.shape[0] if corr is not None else int(m)
    if z <= 0:
        return 1.0
    if corr is None:
        return float(-np.expm1(m * np.log1p(-2.0 * norm.sf(z))))
    one = 2.0 * norm.sf(z)  # the p-value of one output: the largest's is between it and m times it
    bonf = min(1.0, m * one)
    if bonf < level:
        return bonf
    w, v = np.linalg.eigh(corr)
    a = v * np.sqrt(np.clip(w, 0.0, None))
    rng = np.random.default_rng(seed)
    n, done, hit = int(math.ceil(NULL_DRAWS / max(level, one))), 0, 0
    while done < n:
        b = min(1 << 14, n - done)
        hit += int(np.sum(np.max(np.abs(rng.standard_normal((b, m)) @ a.T), 1) >= z))
        done += b
    return hit / n


def mean_test(a, b, *, paired, counts=None, level=None, tests=1, seed=0, floor=None) -> dict:  # noqa: PLR0913 -- the test's inputs
    """Return the comparisons' test that samples ``a`` (Ka, m) and ``b`` (Kb, m) have equal means (module docstring).

    ``paired``: the same seeds (Ka = Kb; the differences' spread), else independent (Welch). ``counts``: optional
    (2, m) effective numbers of contributions of each sample (``effective_count`` of its photon sums over all seeds);
    default, the seeds' (``seed_count``). ``level``: default ALPHA / tests. ``floor``: optional (m,) standard error
    known apart from the seeds (a resolution: a finite difference's float32 sides), added in quadrature to each
    output's and to the outputs' covariance as independent. Returns per output the score ``z``, the
    degrees of freedom ``df`` and ``power`` (bool); ``stat`` the largest |z| with power, ``p`` its p-value, ``n`` the
    outputs with power, ``level``, and ``passed`` (None: no output has power).
    """
    a, b = (np.asarray(x, np.float64).reshape(len(x), -1) for x in (a, b))
    ka, kb, m = a.shape[0], b.shape[0], a.shape[1]
    level = ALPHA / tests if level is None else level
    seeds = np.minimum(seed_count(a), seed_count(b))
    n_min = seeds if counts is None else np.asarray(counts, np.float64).reshape(2, m).min(0)
    if paired:
        d = a - b
        diff, se = d.mean(0), sem(d)
        df = np.minimum(np.minimum(seeds, n_min) - 1.0, variance_df(d))
        cov = np.cov(d, rowvar=False).reshape(m, m) / ka
    else:
        va, vb = a.var(0, ddof=1) / ka, b.var(0, ddof=1) / kb
        diff, se = a.mean(0) - b.mean(0), np.sqrt(va + vb)
        dfa, dfb = variance_df(a), variance_df(b)
        ws = np.where(va + vb > 0, (va + vb) ** 2 / np.where(va + vb > 0, va**2 / dfa + vb**2 / dfb, 1.0), 1.0)
        df = np.minimum(ws, np.minimum(seeds, n_min) - 1.0)
        cov = np.cov(a, rowvar=False).reshape(m, m) / ka + np.cov(b, rowvar=False).reshape(m, m) / kb
    if floor is not None:
        f2 = np.broadcast_to(np.asarray(floor, np.float64) ** 2, (m,))
        se, cov = np.sqrt(se * se + f2), cov + np.diag(f2)
    power = (n_min > power_floor(level)) & (se > 0) & (df >= 1.0)
    z = np.zeros(m)
    z[power] = _scores(diff[power] / se[power], df[power])
    if np.any((se == 0) & (diff != 0)):  # a difference with no spread on either side: certain
        return dict(z=np.where((se == 0) & (diff != 0), np.inf, z), df=df, power=power, stat=np.inf, p=0.0,
                    n=int(power.sum()), level=level, passed=False)  # fmt: skip
    if not power.any():
        return dict(z=z, df=df, power=power, stat=0.0, p=1.0, n=0, level=level, passed=None)
    s = np.sqrt(np.diag(cov)[power])
    corr = cov[np.ix_(power, power)] / np.outer(s, s)
    stat = float(np.max(np.abs(z[power])))
    p = max_normal_p(stat, corr, level, seed)
    return dict(
        z=z, df=df, power=power, stat=stat, p=float(p), n=int(power.sum()), level=level, passed=bool(p >= level)
    )


def reference_test(x, x_var, x_count, ref, ref_var, ref_count, *, level=None, tests=1) -> dict:  # noqa: PLR0913
    """Return the comparisons' test (``mean_test``'s rule) of lumen's values ``x`` (m,) against a reference's run.

    ``x_var`` lumen's variance of each value (its photons' second moment, ``lumen.readout.grid.moment_error``),
    ``x_count`` its effective number of photons (Kish); ``ref`` (m,) the reference's values, ``ref_var`` their variance
    (a count's Poisson variance), ``ref_count`` its effective number of contributions (a count: the count). The outputs
    are independent (a reference's sensors are; lumen's photons correlate them negligibly): the largest |score| is
    referred to m independent normals (Sidak). Each score uses Welch-Satterthwaite's degrees of freedom (the
    reference's variance known, lumen's from ``x_count`` - 1), its power the reference's carriers as in ``mean_test``.
    """
    x, v, xn = (np.asarray(a, np.float64).ravel() for a in (x, x_var, x_count))
    m = x.shape[0]
    level = ALPHA / tests if level is None else level
    var = np.asarray(ref_var, np.float64) + v
    n_min = np.asarray(ref_count, np.float64)  # the reference: fewer carriers than lumen, which runs more photons
    df = np.where((v > 0) & (xn > 1.0), var * var / np.where((v > 0) & (xn > 1.0), v * v / (xn - 1.0), 1.0), np.inf)
    power = (n_min > power_floor(level)) & (var > 0) & (df >= 1.0)
    z = np.zeros(m)
    z[power] = _scores((x - np.asarray(ref, np.float64))[power] / np.sqrt(var[power]), df[power])
    if not power.any():
        return dict(z=z, df=df, power=power, stat=0.0, p=1.0, n=0, level=level, passed=None)
    stat = float(np.max(np.abs(z[power])))
    p = max_normal_p(stat, None, level, m=int(power.sum()))
    return dict(
        z=z, df=df, power=power, stat=stat, p=float(p), n=int(power.sum()), level=level, passed=bool(p >= level)
    )


def fd_order(sides, f0, ad, h0, one_sided, floor_v, exact, resolve=None):  # noqa: PLR0913 -- the ladder, its tolerances
    """Return the exactness of AD against finite differences on the ladder h0 / 2^k, k = 0..3, their order measured.

    ``floor_v``: one side's rounding relative to |F|, the resolution of a difference following from the stencil (a
    float32 value correctly rounded: half an ulp, 2^-24). ``resolve`` (default ``floor_v``): the rounding at which the
    differences count as resolved when the contraction is judged; a looser value lets a kink's growth hide in the
    allowance, so the classification takes the tightest bound and the tolerance the safe one.
    ``sides``: {step: F} (F (K, m)): +-h_k (central, nominal order 2) or h_k and 0 (one-sided, order 1). Per cell (seed,
    output) the differences d_k = FD_k - FD_k+1 contract by 2^-p, p the order; a cell whose differences contract at
    least at the nominal order all along the ladder (|d_k+1| <= 2^-p0 |d_k| up to the sides' float32 resolution, both
    ratios: a kink between two steps breaks one of them) is in its asymptotic range and gets
    the fixed-step test: |AD - FD_inf| <= exact |AD| + resolution + |d_2| / (2^p0 - 1), FD_inf the extrapolation. Any
    other cell (pre-asymptotic, or a realisation not smooth at these steps) gets no fixed-step test: its measured order
    log2(|d_1| / |d_2|) is reported. Returns ``ratio`` (largest over the cells tested; 0 if none), ``tested`` (share of
    cells), ``order`` (median measured order of the others; nan if none), ``applies`` (any cell tested).
    """
    p0 = 1 if one_sided else 2
    hs = [h0 / 2**k for k in range(4)]
    f0 = np.asarray(f0, np.float64)
    fd, res = [], []
    for h in hs:
        if one_sided:
            fd.append((np.asarray(sides[h], np.float64) - f0) / h)
        else:
            fd.append((np.asarray(sides[h], np.float64) - np.asarray(sides[-h], np.float64)) / (2.0 * h))
        res.append((2.0 if one_sided else 1.0) * floor_v * np.abs(f0) / h)  # two sides' rounding over 2h, or over h
    d = [fd[k] - fd[k + 1] for k in range(3)]
    r = [res[k] + res[k + 1] for k in range(3)]
    c = 2.0**-p0
    q = r if resolve is None else [x * (resolve / floor_v) for x in r]  # the classification's resolution
    nominal = (np.abs(d[1]) <= c * np.abs(d[0]) + q[1] + c * q[0]) & (
        np.abs(d[2]) <= c * np.abs(d[1]) + q[2] + c * q[1]
    )
    ad = np.asarray(ad, np.float64)
    fd_inf = fd[3] - d[2] / (2.0**p0 - 1.0)
    tol = exact * np.abs(ad) + res[3] + r[2] + np.abs(d[2]) / (2.0**p0 - 1.0)
    ratio = np.where(tol > 0, np.abs(ad - fd_inf) / np.where(tol > 0, tol, 1.0), np.where(ad != fd_inf, np.inf, 0.0))
    big = (np.abs(d[1]) > r[1]) & (np.abs(d[2]) > r[2])
    orders = np.log2(np.abs(d[1][~nominal & big]) / np.abs(d[2][~nominal & big]))
    return dict(
        ratio=float(ratio[nominal].max()) if nominal.any() else 0.0,
        tested=float(nominal.mean()),
        order=float(np.median(orders)) if orders.size else float("nan"),
        applies=bool(nominal.any()),
    )
