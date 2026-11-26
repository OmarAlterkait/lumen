"""Derivatives of a model's expected counts, the Poisson Fisher information, and a fit, free of Monte Carlo bias.

    model = lambda p, seed: lumen.readout.pe(lumen.simulate(det, src, p, samples=n, seed=seed, reference=p0), p)
    J, mu = lumen.tools.jacobian(model, space, x)       # forward mode: one jvp per entry of the vector
    F = lumen.tools.fisher(model, space, x)             # J^T diag(1 / mu) J
    res = lumen.tools.fit(model, space, data)           # res.x, res.params, res.cov, res.sigma, res.history, res.path
    res = lumen.tools.descend(model, space, data)       # first order: Adam steps along the same score
    cov = lumen.tools.coverage(model, space, x, toys=20)    # refits of Poisson toys made at x: pulls, coverage
    data = lumen.tools.toy_counts(model, space, x, seed=7)  # one such toy: the mean of other seeds, drawn

``model`` is plain JAX from (a ``Params``, a seed) to expected counts (any shape: charge per sensor, time nodes). The
vector is the ``Space``'s (``lumen.tools.Space``).

**Two seeds.** Every quantity comes from two runs a, b of the model at independent ``seeds``, each kept for the whole
fit (so the score is a smooth function of the vector). One run's noise in mu and in J is correlated with itself: J^T
diag(1/mu) J counts the noise of J as information, and the likelihood falls where the noise is smoothed, so a fit on
one run stops at wrong values with errors that are too small. A product of the two runs' factors has no such term:

- the score is (Ta^T (n - mu_b) + Tb^T (n - mu_a)) / 2 with T = J / mu (a cell whose own mu is 0 adds nothing);
- the information is its derivative, (Ta^T Jb + Tb^T Ja) / 2;
- the covariance is the sandwich I^-1 V I^-1, V the score's variance: the data's Poisson noise and the Monte Carlo noise
  of the two runs (from their difference). It holds lumen's own error.

The information's eigenvalues have a Monte Carlo error, estimated from the two runs. A direction whose eigenvalue is
not ``DETERMINED`` of its errors above 0 (or above the float32 Jacobian's resolution) is undetermined: the fit does not
move along it, ``rank`` is below the vector's size, every entry it reaches has infinite variance (nan covariances), and
the fit is not converged. More samples or another source determine it, or ``fixed=`` removes it.

The fit takes Newton steps along the determined directions, each step's largest entry capped at ``max_step`` and halved
while the Newton decrement rises, and has converged when the decrement U^T I^-1 U / 2 is below ``tol``. ``descend``
takes first-order steps instead: Adam (Kingma and Ba, arXiv:1412.6980) along the score, ``lr`` its step in the vector's
units. ``nll`` is Poisson's negative log-likelihood of the two runs' mean, less its Monte Carlo bias. ``coverage`` fits
``toys`` Poisson draws of the model at ``x`` (the mean of ``TRUTH_SEEDS`` other seeds), each from ``x`` at its own two
seeds, and returns the pulls (fit - x) / sigma and the share of toys whose 1 sigma interval holds ``x`` (0.683 for a
Gaussian estimator).

A cell with no light in either run and n > 0 has no Poisson likelihood. The score, the information and the
covariance already leave such a cell out (its 1 / mu is 0); ``nll`` leaves it out of its sum too, and
``FitResult.dropped`` is how many cells were left out, so a fit whose data the model cannot produce there
says so by a count rather than by refusing to start.

``fisher(..., per_sensor=keys)`` eliminates those keys (a per-sensor efficiency, an offset: nuisances) by the Schur
complement and returns the information of the others. A per-sensor block degenerate with a global key (a global
efficiency with every per-sensor one free) leaves a singular information: fix one of them in the space.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from lumen.tools.losses import poisson_nll

HALVINGS = 12  # halvings of a step whose decrement rose before the fit stops
FOURTH = 3.0  # a Gaussian's fourth moment over its squared variance
DETERMINED = 3.0  # the eigenvalue of a determined direction, in its Monte Carlo standard errors
SEEDS = (0, 1)
TRUTH_SEEDS = 8  # the seeds a coverage's truth is the mean of
F32_EPS = float(np.finfo(np.float32).eps)


def _f(model, space, seed):
    return lambda x: jnp.ravel(model(space.params(x), seed))


def jacobian(model, space, x, seed: int = 0, entries=None):
    """Return (J (outputs, P), mu (outputs,)) of the model at the vector ``x`` and one ``seed``: a forward derivative
    per entry (only those of ``entries``, default all: the others' columns are 0)."""  # noqa: D205, D209
    f, x = _f(model, space, seed), jnp.asarray(x, jnp.float32)
    cols, mu = [], None
    todo = set(range(x.shape[0]) if entries is None else entries)
    for i in range(x.shape[0]):
        if i not in todo:
            cols.append(None)
            continue
        mu, d = jax.jvp(f, (x,), (jnp.zeros_like(x).at[i].set(1.0),))
        cols.append(np.asarray(d, np.float64))
    if mu is None:
        mu = f(x)
    mu = np.asarray(mu, np.float64)
    cols = [np.zeros(mu.size) if c is None else c for c in cols]
    return np.stack(cols, 1) if cols else np.zeros((mu.size, 0)), mu


@dataclass(frozen=True)
class _Pair:
    """The two runs at one vector: J (outputs, P) and mu (outputs,) of each."""

    Ja: np.ndarray
    ma: np.ndarray
    Jb: np.ndarray
    mb: np.ndarray

    @property
    def mean(self):
        return 0.5 * (self.ma + self.mb)

    def information(self):
        ta, tb = _over(self.Ja, self.ma), _over(self.Jb, self.mb)
        m = 0.5 * (ta.T @ self.Jb + tb.T @ self.Ja)
        return 0.5 * (m + m.T)

    def score(self, n):
        """Return the score of the counts ``n`` (the likelihood's gradient, uphill) and its variance."""
        ta, tb = _over(self.Ja, self.ma), _over(self.Jb, self.mb)
        u = 0.5 * (ta.T @ (n - self.mb) + tb.T @ (n - self.ma))
        t, var = 0.5 * (ta + tb), 0.5 * (self.ma - self.mb) ** 2  # var: of one run's mu
        v = t.T @ (self.mean[:, None] * t)
        return u, v + 0.5**2 * (ta.T @ (var[:, None] * ta) + tb.T @ (var[:, None] * tb))

    def directions(self, free):
        """Return (eigenvalues, eigenvectors, determined) of the information of the ``free`` entries."""
        w, v = np.linalg.eigh(self.information()[np.ix_(free, free)])
        ja, jb = self.Ja[:, free] @ v, self.Jb[:, free] @ v
        m = self.mean
        s2 = 0.5 * (ja - jb) ** 2  # the variance of one run's derivative; E[s2^2] is 3 var^2
        inv = np.where(m > 0, 1.0 / np.where(m > 0, m, 1.0), 0.0)[:, None] ** 2
        sd = np.sqrt(np.sum(inv * (2.0 * np.maximum(ja * jb, 0.0) * s2 + s2**2 / FOURTH), 0))
        floor = F32_EPS * len(free) * max(float(w.max(initial=0.0)), 0.0)
        return w, v, w > np.maximum(DETERMINED * sd, floor)

    def nll(self, n):
        """Return Poisson's negative log-likelihood of the mean of the two runs, less its Monte Carlo bias, over the
        cells the model can produce (the ``_dark`` cells left out)."""  # noqa: D205, D209
        m, var = self.mean, (0.5 * (self.ma - self.mb)) ** 2
        bias = 0.5 * np.sum(np.where(m > 0, n * var / np.where(m > 0, m, 1.0) ** 2, 0.0))
        keep = ~_dark(m, n)
        return float(poisson_nll(m[keep], n[keep])) - float(bias)


def _over(jac, mu):
    return jac * np.where(mu > 0, 1.0 / np.where(mu > 0, mu, 1.0), 0.0)[:, None]


def _pair(model, space, x, seeds, entries=None):
    a, b = seeds
    if a == b:
        raise ValueError(f"the two seeds are both {a}: a fit needs two independent runs")
    return _Pair(*jacobian(model, space, x, a, entries), *jacobian(model, space, x, b, entries))


def _dark(mean, n):
    """Return the cells the model cannot produce: no light in either run and n > 0 (no Poisson likelihood)."""
    return (mean <= 0) & (n > 0)


def _schur(info, drop):
    keep = np.setdiff1d(np.arange(info.shape[0]), drop)
    if not len(drop):
        return info, keep
    a, b, d = info[np.ix_(keep, keep)], info[np.ix_(keep, drop)], info[np.ix_(drop, drop)]
    return a - b @ np.linalg.solve(d, b.T), keep


def _indices(space, keys, what="fisher: per_sensor"):
    out, i = [], 0
    for k, n in zip(space.keys, space.sizes, strict=True):
        if k in keys:
            out += range(i, i + n)
        i += n
    unknown = set(keys) - set(space.keys)
    if unknown:
        raise ValueError(f"{what} keys {sorted(unknown)} are not keys of the space")
    return np.asarray(out, np.int64)


def fisher(model, space, x, per_sensor=(), *, seeds=SEEDS):
    """Return the Poisson Fisher information of the vector at ``x`` (the ``per_sensor`` keys eliminated), from the runs
    at the two ``seeds``: unbiased by their noise, so not positive definite where the data say less than the noise."""  # noqa: D205, D209
    F = _pair(model, space, x, seeds).information()
    return _schur(F, _indices(space, per_sensor))[0]


@dataclass(frozen=True)
class FitResult:
    """A fit: the vector, its Params, the covariance, the standard errors by key, each step's record, its path.

    The covariance is the estimate's, Monte Carlo error included (infinite variance on the entries an undetermined
    direction reaches: ``rank`` below the vector's size); the history holds per step the negative log-likelihood and the
    Newton decrement (``fit``) or the largest entry of the step (``descend``) where it starts; ``path`` the vector at
    the start and after each step; ``converged`` and ``reason`` the verdict and why it stopped; ``dropped`` the cells
    left out of ``nll`` because the model makes no light there and the data have counts (the module docstring).
    """

    x: np.ndarray
    params: object
    cov: np.ndarray
    sigma: dict
    history: list
    path: np.ndarray
    nll: float
    converged: bool
    reason: str
    rank: int
    dropped: int


def _newton(pair, n, free):
    """Return (the Newton step over the determined directions, its decrement)."""
    u = pair.score(n)[0][free]
    w, v, ok = pair.directions(free)
    c = v[:, ok].T @ u
    d = np.zeros(pair.Ja.shape[1])
    d[free] = v[:, ok] @ (c / w[ok])
    return d, 0.5 * float(np.sum(c**2 / w[ok]))


def fit(model, space, data, x0=None, *, steps: int = 20, max_step: float = 0.1, tol: float = 1e-6, seeds=SEEDS,  # noqa: PLR0913 -- the fit's settings
        fixed=()) -> FitResult:  # fmt: skip
    """Fit the space's keys to counts ``data`` from two runs at independent ``seeds`` (see the module docstring).

    Stops after ``steps`` steps or when the Newton decrement is below ``tol``. ``fixed``: keys held at ``x0`` (the
    covariance is still the whole vector's).
    """
    n = np.asarray(data, np.float64).ravel()
    x = space.vector() if x0 is None else np.asarray(x0, np.float64)
    free = np.setdiff1d(np.arange(len(x)), _indices(space, fixed, "fit: fixed"))
    pair = _pair(model, space, x, seeds, free)
    history, path, reason = [], [x], f"{steps} steps taken"
    d, dec = _newton(pair, n, free)
    for _ in range(steps):
        if dec <= tol:
            reason = "the decrement is below tol"
            break
        d = d * min(1.0, max_step / max(np.max(np.abs(d)), 1e-300))
        history.append(dict(nll=pair.nll(n), decrement=dec))
        for _ in range(HALVINGS):
            new = _pair(model, space, x + d, seeds, free)
            step, at = _newton(new, n, free)
            if at <= dec:
                break
            d = 0.5 * d
        else:
            history.pop()
            reason = f"no step lowered the decrement in {HALVINGS} halvings"
            break
        x, pair, d, dec = x + d, new, step, at
        path.append(x)
    return _result(_pair(model, space, x, seeds), space, x, history, path, n, free, tol, reason)


def _result(pair, space, x, history, path, n, free, tol, reason) -> FitResult:  # noqa: PLR0913
    _, dec = _newton(pair, n, free)
    cov, rank = _covariance(pair, n)
    lost = len(free) - int(pair.directions(free)[2].sum())
    converged = bool(dec <= tol) and lost == 0
    why = f"{reason}; predicted decrease {dec:.3g} {'within' if dec <= tol else 'above'} its bound {tol:.3g}"
    if lost:
        why += f"; {lost} of {len(free)} directions undetermined at this Monte Carlo noise"
    return FitResult(x, space.params(x), cov, space.sigma(cov, x), history, np.stack(path), pair.nll(n), converged, why,
                     rank, int(_dark(pair.mean, n).sum()))  # fmt: skip


def _covariance(pair, n):
    """Return (the estimate's covariance, the number of determined directions) of the whole vector."""
    p = pair.Ja.shape[1]
    if p == 0:
        return np.zeros((0, 0)), 0
    w, v, ok = pair.directions(np.arange(p))
    inv = (v[:, ok] / w[ok]) @ v[:, ok].T
    cov = inv @ pair.score(n)[1] @ inv
    reached = np.any(np.abs(v[:, ~ok]) > np.sqrt(F32_EPS), 1)
    cov[reached, :] = np.nan
    cov[:, reached] = np.nan
    cov[reached, reached] = np.inf
    return cov, int(ok.sum())


def descend(model, space, data, x0=None, *, steps: int = 50, lr: float = 0.02, tol: float = 1e-3,  # noqa: PLR0913 -- the solver's settings
            betas=(0.9, 0.999), seeds=SEEDS) -> FitResult:  # fmt: skip
    """Fit the space's keys to counts ``data`` by ``steps`` first-order steps (Adam; see the module docstring)."""
    n = np.asarray(data, np.float64).ravel()
    x = space.vector() if x0 is None else np.asarray(x0, np.float64)
    m, v, history, path = np.zeros_like(x), np.zeros_like(x), [], [x]
    b1, b2 = betas
    for k in range(1, steps + 1):
        pair = _pair(model, space, x, seeds)
        g = -pair.score(n)[0]
        m, v = b1 * m + (1 - b1) * g, b2 * v + (1 - b2) * g * g
        d = -lr * (m / (1 - b1**k)) / (np.sqrt(v / (1 - b2**k)) + 1e-12)
        history.append(dict(nll=pair.nll(n), step=float(np.max(np.abs(d)))))
        x = x + d
        path.append(x)
    pair = _pair(model, space, x, seeds)
    return _result(pair, space, x, history, path, n, np.arange(len(x)), tol, f"{steps} steps taken")


@dataclass(frozen=True)
class CoverageResult:
    """Toy fits at a known vector: the fits (T, P), their standard errors (T, P), the pulls, the 1 sigma coverage."""

    x: np.ndarray
    fits: np.ndarray
    sigma: np.ndarray
    pulls: np.ndarray
    covered: np.ndarray


def toy_counts(model, space, x, *, seed: int):
    """Return one Poisson draw of the model's counts at the vector ``x``, from other seeds than a fit's own.

    The mean of ``TRUTH_SEEDS`` runs at the seeds ``seed`` .. ``seed + TRUTH_SEEDS - 1``, drawn with ``seed``. Toy data
    a fit is judged by must come from other seeds than the fit's, or the fit reads its own noise as signal (the module
    docstring). Give a ``seed`` past the seeds the fit runs at; ``coverage`` does.
    """
    v = jnp.asarray(np.asarray(x, np.float64), jnp.float32)
    mu = np.mean([np.asarray(_f(model, space, seed + i)(v), np.float64) for i in range(TRUTH_SEEDS)], 0)
    return np.random.default_rng(seed).poisson(mu)


def coverage(model, space, x, *, toys: int = 20, seed: int = 0, **fit_kw) -> CoverageResult:
    """Return the CoverageResult of ``toys`` fits (``fit``, its settings ``fit_kw``) to Poisson draws of the model at
    ``x``, each fit started at ``x`` (see the module docstring). The truth's runs are at other seeds than the fits'."""  # noqa: D205, D209
    x = np.asarray(x, np.float64)
    first = 2 * int(toys) + int(seed)  # the toys fit at seeds 0 .. 2 toys - 1: the truth's are past them
    fits, sig = [], []
    for k in range(int(toys)):
        n = toy_counts(model, space, x, seed=first + k * TRUTH_SEEDS)  # its own data, from its own other seeds
        r = fit(model, space, n, x, seeds=(2 * k, 2 * k + 1), **fit_kw)
        fits.append(r.x)
        sig.append(np.sqrt(np.diag(r.cov)))
    fits, sig = np.stack(fits), np.stack(sig)
    pulls = (fits - x) / sig
    return CoverageResult(x, fits, sig, pulls, np.mean(np.abs(pulls) <= 1.0, 0))
