"""Derivatives of a model's expected counts, the Poisson Fisher information, and a Fisher-scoring fit.

    model = lambda p: lumen.readout.pe(lumen.simulate(det, src, p, samples=n, seed=0), p)   # params -> counts
    J, mu = lumen.tools.jacobian(model, space, x)       # forward mode: one jvp per entry of the vector
    F = lumen.tools.fisher(model, space, x)             # J^T diag(1 / mu) J
    res = lumen.tools.fit(model, space, data)           # res.x, res.params, res.cov, res.sigma, res.history, res.path
    res = lumen.tools.descend(model, space, data)       # first order: gradient steps (Adam) on the same likelihood
    cov = lumen.tools.coverage(model, space, x, toys=20)    # refits of Poisson toys made at x: pulls, coverage

``model`` is plain JAX from a ``Params`` to expected counts (any shape: charge per sensor, time nodes); its seed is
fixed inside, so the model is a smooth function of the values (one Monte Carlo dataset for every step). The vector is
the ``Space``'s (``lumen.tools.Space``). The fit is Fisher scoring on the Poisson likelihood of ``data`` (Gauss-Newton
with the metric J^T diag(1/mu) J), damped by a ridge of ``ridge`` times the median diagonal of the metric, raised
tenfold while a step does not lower the likelihood, each step's largest entry capped at ``max_step``. ``refresh``: steps
per Jacobian (it is reused in between). The covariance is the inverse Fisher information at the fit; at the truth the
same inverse is the Cramer-Rao bound. ``coverage`` fits ``toys`` Poisson draws of the model at ``x`` (each from ``x``)
and returns the pulls (fit - x) / sigma and the share of toys whose 1 sigma interval holds ``x`` (0.683 for a Gaussian
estimator). ``descend`` takes first-order steps instead: Adam (Kingma and Ba,
arXiv:1412.6980) on the gradient J^T (1 - data / mu) of the same likelihood, ``lr`` its step in the vector's units.

The likelihood is Poisson's as defined at mu = 0: a cell with mu = 0 and n = 0 adds 0 (and its derivative J to the
gradient); a cell with mu = 0 and n > 0 makes the data impossible (a likelihood of 0), so a fit or a descent from
such a vector is refused, and a step onto one is rejected (its likelihood rises to infinity). A cell's Fisher term
J^2 / mu needs J = 0 where mu = 0 (no light, no derivative: a Monte Carlo model's empty cell); otherwise the
information is infinite and refused.

Every result says whether it converged (``converged``, ``reason``): the predicted decrease of the likelihood at the
end, the Newton decrement g^T F^-1 g / 2 over the free entries, within the stopping tolerance or the likelihood's own
float32 resolution (eps32 (sum mu + sum n)): a fit whose ridge rose ``RIDGE_RAISES`` times without lowering the
likelihood stops, and is reported converged only by that criterion. A singular information (rank below the vector's,
at the float32 Jacobian's resolution) is reported: ``rank``, and an infinite variance (nan covariances) on every entry
the null space reaches.

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

RIDGE_RAISES = 12  # tenfold raises of the ridge per step (to 1e12 times its start: a step of float32 size) before
# the fit stops on a step that lowers nothing
F32_EPS = float(np.finfo(np.float32).eps)  # the model's float32 resolution: of its counts, Jacobian and likelihood


def _f(model, space):
    return lambda x: jnp.ravel(model(space.params(x)))


def jacobian(model, space, x, entries=None):
    """Return (J (outputs, P), mu (outputs,)) of the model at the vector ``x``: a forward derivative per entry
    (only those of ``entries``, default all: the others' columns are 0)."""  # noqa: D205, D209
    f, x = _f(model, space), jnp.asarray(x, jnp.float32)
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


def _information(jac, mu):
    empty = mu <= 0
    if np.any(jac[empty] != 0):
        raise ValueError(f"fisher: {int(np.any(jac[empty] != 0, 1).sum())} cells with mu = 0 have a nonzero "
                         "derivative: their information J^2 / mu is infinite")  # fmt: skip
    w = np.where(empty, 0.0, 1.0 / np.where(empty, 1.0, mu))
    return jac.T @ (w[:, None] * jac)


def _possible(mu, n, what):
    """Refuse data the model cannot produce: a cell with mu = 0 and n > 0 (the Poisson likelihood there is 0)."""
    bad = (np.asarray(mu) <= 0) & (n > 0)
    if bad.any():
        raise ValueError(f"{what}: {int(bad.sum())} cells have counts where the model expects none (mu = 0, n > 0): "
                         "the data are impossible at the start values")  # fmt: skip


def _gradient(jac, mu, n):
    """Return J^T (1 - n / mu), the Poisson likelihood's gradient; a cell with mu = 0 has n = 0 (``_possible``) and
    adds J."""  # noqa: D205, D209
    return jac.T @ (1.0 - np.where(mu > 0, n / np.where(mu > 0, mu, 1.0), 0.0))


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


def fisher(model, space, x, per_sensor=()):
    """Return the Poisson Fisher information of the vector at ``x`` (the ``per_sensor`` keys eliminated)."""
    F = _information(*jacobian(model, space, x))
    return _schur(F, _indices(space, per_sensor))[0]


@dataclass(frozen=True)
class FitResult:
    """A fit: the vector, its Params, the covariance, the standard errors by key, each step's record, its path.

    The covariance is the inverse Fisher information at the fit (infinite variance where it is singular: ``rank``
    below the vector's size); the history holds per accepted step the negative log-likelihood (``poisson_nll``) and
    the ridge (Fisher scoring), or the likelihood where the step starts and its largest entry (``descend``); ``path``
    the vector at the start and after each accepted step; ``converged`` and ``reason`` the stopping criterion's
    verdict and why it stopped (the module docstring).
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


def fit(model, space, data, x0=None, *, steps: int = 20, ridge: float = 0.02, max_step: float = 0.1,  # noqa: PLR0913 -- the fit's settings
        refresh: int = 1, tol: float = 1e-8, fixed=()) -> FitResult:  # fmt: skip
    """Fit the space's keys to counts ``data`` by Fisher scoring on the Poisson likelihood (see the module docstring).

    Stops after ``steps`` accepted steps or when a step lowers the likelihood by less than ``tol`` (relative).
    ``fixed``: keys held at ``x0`` (the covariance is still the whole vector's).
    """
    f = _f(model, space)
    n = np.asarray(data, np.float64).ravel()
    x = space.vector() if x0 is None else np.asarray(x0, np.float64)
    free = np.setdiff1d(np.arange(len(x)), _indices(space, fixed, "fit: fixed"))
    nll = lambda v: float(poisson_nll(f(jnp.asarray(v, jnp.float32)), n))  # noqa: E731
    cur, lam, history, J, mu, path = nll(x), ridge, [], None, None, [x]
    reason = f"{steps} steps taken"
    for k in range(steps):
        if J is None or k % max(refresh, 1) == 0:
            J, mu = jacobian(model, space, x, free)
            if k == 0:
                _possible(mu, n, "fit")
        F, g = _information(J[:, free], mu), _gradient(J[:, free], mu, n)
        damp = np.median(np.diag(F)) if F.size else 0.0
        for _ in range(RIDGE_RAISES):  # raise the ridge until the step lowers the likelihood
            d = np.zeros_like(x)
            d[free] = -np.linalg.solve(F + lam * damp * np.eye(len(free)), g)
            d *= min(1.0, max_step / max(np.max(np.abs(d)), 1e-300))
            new = nll(x + d)
            if new <= cur:
                break
            lam *= 10.0
        else:
            reason = f"no step lowered the likelihood in {RIDGE_RAISES} ridge raises"
            break
        gain = cur - new
        x, cur, J = x + d, new, None
        lam = max(lam / 10.0, ridge)
        history.append(dict(nll=cur, ridge=lam))
        path.append(x)
        if gain <= tol * max(abs(cur), 1.0):
            reason = "a step lowered the likelihood by less than tol"
            break
    return _result(model, space, x, history, path, n, free, tol, reason)


def _result(model, space, x, history, path, n, free=None, tol=0.0, reason="") -> FitResult:  # noqa: PLR0913
    J, mu = jacobian(model, space, x)
    info = _information(J, mu)
    free = np.arange(len(x)) if free is None else free
    g, ff = _gradient(J[:, free], mu, n), info[np.ix_(free, free)]
    dec = 0.5 * float(g @ np.linalg.lstsq(ff, g, rcond=None)[0]) if len(free) else 0.0
    nll = float(poisson_nll(mu, n))
    resolution = F32_EPS * float(mu.sum() + n.sum())  # the float32 likelihood's rounding
    bound = max(tol * max(abs(nll), 1.0), resolution)
    converged = bool(dec <= bound)
    cov, rank = _covariance(info)
    why = f"{reason}; predicted decrease {dec:.3g} {'within' if converged else 'above'} its bound {bound:.3g}"
    return FitResult(x, space.params(x), cov, space.sigma(cov, x), history, np.stack(path), nll, converged, why,
                     rank)  # fmt: skip


def _covariance(info):
    """Return (the inverse information, its rank): where it is singular at the float32 Jacobian's resolution
    (eigenvalues below eps32 x size x the largest), the entries the null space reaches get infinite variance and nan
    covariances, the others the inverse on the rest."""  # noqa: D205, D209
    p = info.shape[0]
    if p == 0:
        return np.zeros((0, 0)), 0
    w, v = np.linalg.eigh(info)
    null = w <= F32_EPS * p * max(float(w.max()), 0.0)
    rank = int(p - null.sum())
    if not null.any():
        return np.linalg.inv(info), rank
    keep = ~null
    cov = (v[:, keep] / w[keep]) @ v[:, keep].T
    reached = np.any(np.abs(v[:, null]) > np.sqrt(F32_EPS), 1)
    cov[reached, :] = np.nan
    cov[:, reached] = np.nan
    cov[reached, reached] = np.inf
    return cov, rank


def descend(model, space, data, x0=None, *, steps: int = 50, lr: float = 0.02,  # noqa: PLR0913 -- the solver's settings
            betas=(0.9, 0.999)) -> FitResult:  # fmt: skip
    """Fit the space's keys to counts ``data`` by ``steps`` first-order steps (Adam; see the module docstring).

    Every step takes one forward derivative per entry of the vector (the gradient, ``jacobian``); the result's
    covariance is the inverse Fisher information at the end, as ``fit``'s.
    """
    n = np.asarray(data, np.float64).ravel()
    x = space.vector() if x0 is None else np.asarray(x0, np.float64)
    m, v, history, path = np.zeros_like(x), np.zeros_like(x), [], [x]
    b1, b2 = betas
    for k in range(1, steps + 1):
        J, mu = jacobian(model, space, x)
        if k == 1:
            _possible(mu, n, "descend")
        g = _gradient(J, mu, n)
        m, v = b1 * m + (1 - b1) * g, b2 * v + (1 - b2) * g * g
        d = -lr * (m / (1 - b1**k)) / (np.sqrt(v / (1 - b2**k)) + 1e-12)
        history.append(dict(nll=float(poisson_nll(mu, n)), step=float(np.max(np.abs(d)))))
        x = x + d
        path.append(x)
    return _result(model, space, x, history, path, n, reason=f"{steps} steps taken")


@dataclass(frozen=True)
class CoverageResult:
    """Toy fits at a known vector: the fits (T, P), their standard errors (T, P), the pulls, the 1 sigma coverage."""

    x: np.ndarray
    fits: np.ndarray
    sigma: np.ndarray
    pulls: np.ndarray
    covered: np.ndarray


def coverage(model, space, x, *, toys: int = 20, seed: int = 0, **fit_kw) -> CoverageResult:
    """Return the CoverageResult of ``toys`` fits (``fit``, its settings ``fit_kw``) to Poisson draws of the model at
    ``x``, each fit started at ``x`` (see the module docstring)."""  # noqa: D205, D209
    x = np.asarray(x, np.float64)
    mu = np.asarray(_f(model, space)(jnp.asarray(x, jnp.float32)), np.float64)
    rng = np.random.default_rng(seed)
    fits, sig = [], []
    for _ in range(int(toys)):
        r = fit(model, space, rng.poisson(mu), x, **fit_kw)
        fits.append(r.x)
        sig.append(np.sqrt(np.diag(r.cov)))
    fits, sig = np.stack(fits), np.stack(sig)
    pulls = (fits - x) / sig
    return CoverageResult(x, fits, sig, pulls, np.mean(np.abs(pulls) <= 1.0, 0))
