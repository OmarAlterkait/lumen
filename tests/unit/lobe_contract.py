"""The lobe contract: what every drawn (non-deterministic) lobe must satisfy, checked against float64 quadrature
of its own logpdf.

  1  unit outgoing vectors
  2  the density integrates to 1 over the sphere (direction grid about the lobe axis, dense near it)
  3  the sampler follows the density: chi2 in (angle from axis, azimuth sector) bins whose edges are grid edges, so the
     expected counts are exact sums of grid cells
  4  score identity E[d log p / d theta] = 0 for every parameter (the premise of reweighting)
  5  finite gradients of logpdf and weight with respect to params, wi and n on sampled directions
  6  weight in [0, 1]
Densities must be BOUNDED in direction space for check 2-3 (true for every lobe here; a lobe whose density is
singular in direction space must be checked in its own latent space instead)."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from lumen._core import rng

F32 = jnp.float32
Z_MAX = 4.5


def z_of(x, truth=0.0):
    x = np.asarray(x, np.float64)
    return (x.mean() - truth) / max(x.std(ddof=1) / math.sqrt(len(x)), 1e-30), x.mean()


def _basis(r):
    a = np.array([1.0, 0.0, 0.0]) if abs(r[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(r, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(r, e1)


def direction_grid(axis, na=3000, nphi=720):
    """directions about `axis`: polar angle a = pi u^2 (dense near the axis), azimuth phi; solid angles dw."""
    u = (np.arange(na) + 0.5) / na
    A = np.pi * u**2
    dA = 2.0 * np.pi * u / na
    phi = (np.arange(nphi) + 0.5) / nphi * 2.0 * np.pi
    e1, e2 = _basis(axis)
    AA, PP = np.meshgrid(A, phi, indexing="ij")
    W = np.cos(AA)[..., None] * axis + np.sin(AA)[..., None] * (np.cos(PP)[..., None] * e1 + np.sin(PP)[..., None] * e2)
    dw = np.sin(AA) * dA[:, None] * (2.0 * np.pi / nphi)
    return dict(W=W, dw=dw, A=A, e1=e1, e2=e2, a_edges=np.pi * (np.arange(na + 1) / na) ** 2, na=na, nphi=nphi)


def eval_grid(fn, W, chunk=400_000):
    Wf = W.reshape(-1, 3)
    f = jax.jit(jax.vmap(lambda w: fn(jnp.asarray(w, F32))))
    return np.concatenate(
        [np.asarray(f(jnp.asarray(Wf[i : i + chunk], F32)), np.float64) for i in range(0, len(Wf), chunk)]
    ).reshape(W.shape[:-1])


def draw(lobe, p, wi, n, N, seed=0):
    keys = jax.vmap(lambda i: rng.photon_key(jax.random.PRNGKey(seed), i))(jnp.arange(N))
    u = jax.vmap(lambda k: rng.uniforms(k, max(lobe.n_rand, 1)))(keys)
    f = jax.jit(jax.vmap(lambda uu: lobe.sample(p, wi, n, uu)))
    return np.asarray(f(u), np.float64)


def check_lobe(lobe, p, wi, n, N=400_000, seed=0, label="", grid=(3000, 720)):
    """run the contract; returns a dict of measured numbers (printed by the caller)."""
    p = {k: jnp.asarray(v, F32) for k, v in p.items()}
    wi = jnp.asarray(wi, F32)
    n = jnp.asarray(n, F32)
    axis = np.asarray(lobe.axis(p, wi, n), np.float64)
    axis /= np.linalg.norm(axis)
    G = direction_grid(axis, *grid)
    logp = eval_grid(lambda w: lobe.logpdf(p, wi, n, w), G["W"])
    mass = np.exp(logp) * G["dw"]
    total = mass.sum()
    assert abs(total - 1.0) < 3e-3, (label, "density integral", total)
    wo = draw(lobe, p, wi, n, N, seed)
    assert np.max(np.abs(np.linalg.norm(wo, axis=1) - 1.0)) < 1e-5, (label, "unit vectors")
    # chi2 in grid-aligned bins
    na, nphi = G["na"], G["nphi"]
    cum = np.concatenate([[0.0], np.cumsum(mass.sum(1))]) / total
    idx = np.unique(np.searchsorted(cum, np.linspace(0, 1, 21)[1:-1]))
    idx = np.concatenate([[0], idx[(idx > 0) & (idx < na)], [na]])
    row_bin = np.searchsorted(idx, np.arange(na), side="right") - 1
    sec = np.arange(nphi) // (nphi // 8)
    exp = np.zeros((len(idx) - 1, 8))
    np.add.at(exp, (row_bin[:, None].repeat(nphi, 1), sec[None, :].repeat(na, 0)), mass / total * N)
    a_s = np.arctan2(np.linalg.norm(np.cross(wo, axis), axis=1), wo @ axis)
    p_s = np.mod(np.arctan2(wo @ G["e2"], wo @ G["e1"]), 2 * np.pi)
    obs = np.histogram2d(a_s, p_s, bins=[G["a_edges"][idx], np.linspace(0, 2 * np.pi, 9)])[0]
    ok = exp.ravel() >= 20
    c2 = float(np.sum((obs.ravel()[ok] - exp.ravel()[ok]) ** 2 / exp.ravel()[ok]))
    ndf = int(ok.sum()) - 1
    assert c2 / ndf < 1.6, (label, "sampler vs density chi2/ndf", c2 / ndf, ndf)
    # weights
    woj = jnp.asarray(wo[:50_000], F32)
    w = np.asarray(jax.vmap(lambda x: lobe.weight(p, wi, n, x))(woj), np.float64)
    assert np.all(w >= 0.0) and np.all(w <= 1.0 + 1e-6), (label, "weight range", w.min(), w.max())
    # score identity, per parameter
    zs = {}
    for name in lobe.params:
        g = jax.jit(jax.vmap(lambda x: jax.grad(lambda q: lobe.logpdf({**p, name: q}, wi, n, x))(p[name])))(
            jnp.asarray(wo, F32)
        )
        g = np.asarray(g, np.float64)
        zs[name], _ = z_of(g)
        zs[name + "_extreme"] = (float(np.min(g)), float(np.max(g)), int(np.argmin(g)), int(np.argmax(g)))
        assert abs(zs[name]) < Z_MAX, (label, "score identity", name, zs[name])

    # finite gradients (params, wi, n) of logpdf and weight on samples
    def tot(pp, a, b, x):
        return lobe.logpdf(pp, a, b, x) + lobe.weight(pp, a, b, x)

    gr = jax.jit(jax.vmap(lambda x: jax.grad(tot, argnums=(0, 1, 2))(p, wi, n, x)))(woj)
    for leaf in jax.tree_util.tree_leaves(gr):
        assert np.all(np.isfinite(np.asarray(leaf))), (label, "non-finite gradient")
    return dict(wo=wo, total=total, chi2=c2 / ndf, ndf=ndf, score=zs, mean_weight=float(w.mean()))
