"""Geant4's facet chains, the GLOSSY outcome's lobes: the unified model's and GLISUR's.

A chain lobe is Geant4's DielectricMetal reflection off a rough surface: a facet normal drawn for the current
direction, a mirror reflection about it, repeated while the direction points into the surface. Its marginal density has
no closed form; its path density (the product of one facet reflection's density along the chain) has, so the step
weights the whole path (``Lobe.path``, ``Lobe.path_logpdf``).

- **unified** (GetFacetNormal, unified): tilt alpha ~ N(0, sigma_alpha), kept if ``u f_max <= sin(alpha)`` and alpha
  < pi/2 (f_max = min(1, 4 sigma)), azimuth uniform, redrawn while the facet faces away. Per solid angle of the facet
  m the photon meets D(m) [V.m > 0] / F(theta, sigma), D(m) = exp(-alpha^2 / 2 sigma^2) min(1, f_max / sin alpha) /
  (2 pi Z(sigma)), F the facing share (tabulated: float64 quadrature, float32 bicubic Hermite, C1). One reflection
  wo = mirror(wi, m) has density D(h) / (4 F V.h), h = unit(V + wo). Valid sigma_alpha: ``rules.optics.SIGMA_MIN``
  to ``SIGMA_MAX``.
- **GLISUR** (GetFacetNormal, glisur): M = n + r s, s uniform in the unit ball, r = 1 - polish, redrawn while
  V . M <= 0. Per solid angle of m = M / |M|: p_m = 2 q (3 c^2 + q^2) / (3 Z), c = m . n, q^2 = c^2 - (1 - r^2), Z the
  cut ball's volume. Bounded, vanishing at the cone's edge: no boundary term. The width is carried as "sigma_alpha".

Sampling: rejection, at most FACET_TRIES draws per facet and CHAIN_K_MAX facets per chain (weight 0 after).

Notes
-----
G4OpBoundaryProcess v11.3.2 (identical in 10.7): GetFacetNormal, DoReflection, DielectricMetal.
"""

# ruff: noqa: N803 -- the module docstring's math names (Vl, M, W, Y, ...) as arguments
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from lumen._core import frames, rng
from lumen._core.safe import literal, safe_sqrt
from lumen.surfaces.lobes import Lobe

sg = jax.lax.stop_gradient
HALF_PI = 0.5 * math.pi
CHAIN_K_MAX = 10  # facets per chain: a chain still pointing into the surface after these has weight 0
FACET_TRIES = 64  # facet draws per facet: after these (probability < 1e-12) the last draw is kept
LOG_TINY = math.log(1e-30)  # a log density off the support
UP = (0.0, 0.0, 1.0)  # the local normal


def _up():
    return literal(UP)


# unified's tables: F on (x, log sigma), x uniform on [0, 1]; log sigma = log(1/4) + j log(2)/8, j in [J_LO, J_HI]
TABLE_NX = 257
TABLE_Y0, TABLE_DY, TABLE_J_LO, TABLE_J_HI = math.log(0.25), math.log(2.0) / 8.0, -52, 12
TABLE_NY = TABLE_J_HI - TABLE_J_LO + 1
_GX, _GW = np.polynomial.legendre.leggauss(16)


def _gauss_legendre(a, b, panels):
    """Return composite Gauss-Legendre nodes and weights on [a, b] (arrays broadcast; the last axis is the nodes)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    e = a[..., None] + (b - a)[..., None] * np.linspace(0.0, 1.0, panels + 1)
    lo, hi = e[..., :-1, None], e[..., 1:, None]
    x = 0.5 * (lo + hi) + 0.5 * (hi - lo) * _GX
    w = 0.5 * (hi - lo) * _GW
    return x.reshape(*x.shape[:-2], -1), w.reshape(*w.shape[:-2], -1)


def tilt_density(alpha, sigma):
    """Return Geant4's unified facet-tilt density, unnormalised: g(alpha; sigma) min(sin alpha, f_max) (float64)."""
    return np.exp(-alpha * alpha / (2.0 * sigma * sigma)) * np.minimum(np.sin(alpha), min(1.0, 4.0 * sigma))


def tilt_norm(sigma, panels=64):
    """Return Z(sigma), the integral of ``tilt_density`` over (0, pi/2), split where the min switches."""
    top = min(HALF_PI, 14.0 * sigma)  # the integrand is negligible past 14 sigma
    af = min(math.asin(min(1.0, 4.0 * sigma)), top)
    tot = 0.0
    for a, b in ((0.0, af), (af, top)):
        if b > a:
            x, w = _gauss_legendre(a, b, panels)
            tot += float(np.sum(w * tilt_density(x, sigma)))
    return tot


def front_fraction_x(x, sigma, panels=16):
    """Return the share of facets facing the photon, P(V . m > 0), at x = (2/pi) atan2(cos theta, sigma sin theta).

    The facets tilted less than atan(cot theta) all face V; past it the front arc is 2 acos(-cot theta cot alpha).
    With alpha = alpha_k + L tau^2 the arc's square-root edge is smooth in tau; the f_max kink is a panel edge.
    """
    x = np.asarray(x, np.float64)
    c = np.where(x >= 1.0, np.inf, sigma * np.tan(HALF_PI * np.minimum(x, 1.0)))
    ak = np.arctan(c)
    top = np.minimum(HALF_PI, ak + 14.0 * sigma)
    span = top - ak
    tf = np.sqrt(np.clip((math.asin(min(1.0, 4.0 * sigma)) - ak) / np.where(span > 0, span, 1.0), 0.0, 1.0))
    acc = 0.0
    for a, b in ((np.zeros_like(c), tf), (tf, np.ones_like(c))):
        t, w = _gauss_legendre(a, b, panels)
        al = ak[..., None] + span[..., None] * t * t
        arg = np.where(np.isinf(c)[..., None], 1.0, np.clip(c[..., None] * (np.cos(al) / np.sin(al)), -1.0, 1.0))
        acc = acc + np.sum(w * tilt_density(al, sigma) * np.arccos(arg) / math.pi * 2.0 * span[..., None] * t, -1)
    return 1.0 - acc / tilt_norm(sigma)


@functools.lru_cache(maxsize=None)
def unified_tables():
    """Return the float32 node values and grid derivatives of F (NX, NY) and log zeta = log(Z / sigma^2) (NY,).

    Derivatives by central differences of the float64 quadrature (one-sided at the x ends).
    """
    xs = np.linspace(0.0, 1.0, TABLE_NX)
    ys = TABLE_Y0 + TABLE_DY * np.arange(TABLE_J_LO, TABLE_J_HI + 1)
    d, de = 1e-5, 1e-6
    xp, xm = np.minimum(xs + d, 1.0), np.maximum(xs - d, 0.0)
    xp[0], xm[-1] = de, 1.0 - de
    F, Fx, Fy = (np.zeros((TABLE_NX, TABLE_NY)) for _ in range(3))
    lz, lzy = np.zeros(TABLE_NY), np.zeros(TABLE_NY)
    for j, y in enumerate(ys):
        s = math.exp(y)
        F[:, j] = front_fraction_x(xs, s)
        Fx[:, j] = (front_fraction_x(xp, s) - front_fraction_x(xm, s)) / (xp - xm)
        Fy[:, j] = (front_fraction_x(xs, math.exp(y + d)) - front_fraction_x(xs, math.exp(y - d))) / (2 * d)

        def z(yy):
            return math.log(tilt_norm(math.exp(yy)) / math.exp(2 * yy))

        lz[j], lzy[j] = z(y), (z(y + d) - z(y - d)) / (2 * d)
    Fxy = np.gradient(Fx, TABLE_DY, axis=1, edge_order=2)
    return {k: np.asarray(v, np.float32) for k, v in dict(F=F, Fx=Fx, Fy=Fy, Fxy=Fxy, lz=lz, lzy=lzy).items()}


def _hermite(t):
    t2 = t * t
    t3 = t2 * t
    return 2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t, -2 * t3 + 3 * t2, t3 - t2


def _cell(v, n):
    f = jnp.where(v < 0.0, 0.0, jnp.where(v > n - 1.0, n - 1.0, v))  # strict: a node keeps its gradient
    i = jnp.clip(jnp.floor(f), 0, n - 2).astype(jnp.int32)
    return i, f - i


def table_operands():
    """Return the unified tables as device arrays, to enter a loop as operands (the lobe parameter ``tables``)."""
    return {k: jnp.asarray(v) for k, v in unified_tables().items()}


def log_zeta(sigma, T=None):
    """Return log(Z(sigma) / sigma^2), C1 cubic Hermite in log sigma (``T``: the tables' operands, else constants)."""
    T = unified_tables() if T is None else T
    j, t = _cell((jnp.log(sigma) - TABLE_Y0) / TABLE_DY - TABLE_J_LO, TABLE_NY)
    a0, b0, a1, b1 = _hermite(t)
    v, dv = jnp.asarray(T["lz"]), jnp.asarray(T["lzy"])
    return a0 * v[j] + a1 * v[j + 1] + TABLE_DY * (b0 * dv[j] + b1 * dv[j + 1])


def front_fraction(sigma, cos_t, sin_t, T=None):
    """Return F(theta, sigma), the share of facets facing the photon: bicubic Hermite on (x, log sigma)."""
    T = unified_tables() if T is None else T
    x = jnp.arctan2(cos_t, sigma * sin_t) / HALF_PI
    i, tx = _cell(x * (TABLE_NX - 1), TABLE_NX)
    j, ty = _cell((jnp.log(sigma) - TABLE_Y0) / TABLE_DY - TABLE_J_LO, TABLE_NY)
    hx = 1.0 / (TABLE_NX - 1)
    ax0, bx0, ax1, bx1 = _hermite(tx)
    ay0, by0, ay1, by1 = _hermite(ty)
    A = {k: jnp.asarray(T[k]) for k in ("F", "Fx", "Fy", "Fxy")}
    out = 0.0
    for di, ax, bx in ((0, ax0, bx0), (1, ax1, bx1)):
        for dj, ay, by in ((0, ay0, by0), (1, ay1, by1)):
            ii, jj = i + di, j + dj
            out = out + (ax * ay * A["F"][ii, jj] + bx * ay * hx * A["Fx"][ii, jj]
                         + ax * by * TABLE_DY * A["Fy"][ii, jj]
                         + bx * by * hx * TABLE_DY * A["Fxy"][ii, jj])  # fmt: skip
    return out


def _unified_log_density(sigma, Vl, M, T=None):
    """Return log D(m) / F(theta, sigma) for the local facet direction M (any length); the caller masks the support.

    D(m) = exp(-alpha^2 / 2 sigma^2) min(1, f_max / sin alpha) / (2 pi Z(sigma)), f_max = min(1, 4 sigma).
    """
    inv = jax.lax.rsqrt(jnp.sum(M * M, -1))
    ch = M[..., 2] * inv
    s = safe_sqrt(M[..., 0] ** 2 + M[..., 1] ** 2) * inv  # sin(alpha), from the tangential part
    alpha = jnp.arctan2(s, ch)
    lf = jnp.minimum(0.0, jnp.log(4.0 * sigma))  # log f_max
    ls = jnp.log(jnp.where(s > jnp.exp(lf), s, jnp.exp(lf)))
    cos_t = Vl[..., 2]
    sin_t = safe_sqrt(Vl[..., 0] ** 2 + Vl[..., 1] ** 2)
    return (-alpha * alpha / (2.0 * sigma * sigma) + (lf - ls) - math.log(2.0 * math.pi) - 2.0 * jnp.log(sigma)
            - log_zeta(sigma, T) - jnp.log(front_fraction(sigma, cos_t, sin_t, T)))  # fmt: skip


def glisur_log_volume(r, h):
    """Return the log volume of the GLISUR facet ball B(n, r) on the photon's side of V . M = 0 (h = V . n)."""
    d = jnp.maximum(r - h, 0.0)
    return jnp.log(4.0 / 3.0 * math.pi * r**3 - math.pi * d * d * (2.0 * r + h) / 3.0)


def _glisur_log_density(r, Vl, M):
    """Return (log p_m, inside the support) for the local facet direction M (any length).

    p_m = 2 q (3 c^2 + q^2) / (3 Z), c = m . n, q^2 = r^2 - sin^2(alpha), Z the cut ball's volume.
    """
    m2 = jnp.sum(M * M, -1)
    c = M[..., 2] * jax.lax.rsqrt(m2)
    s2 = (M[..., 0] ** 2 + M[..., 1] ** 2) / m2  # sin^2(alpha), from the tangential part
    q2 = r * r - s2
    ok = (q2 > 0.0) & (c > 0.0) & (jnp.sum(Vl * M, -1) > 0.0)
    q2s = jnp.where(ok, q2, r * r)
    lp = (math.log(2.0 / 3.0) + 0.5 * jnp.log(q2s) + jnp.log(3.0 * jnp.where(ok, c, 1.0) ** 2 + q2s)
          - glisur_log_volume(r, Vl[..., 2]))  # fmt: skip
    return lp, ok


def unified_facet_logpdf(sigma, wi, n, wo, T=None):
    """Return log p1(wo | wi) = log D(h) / (4 F V.h), one unified facet reflection; LOG_TINY off the support.

    h = unit(V + wo), V = -wi: over the whole sphere, with an integrable 1 / angle singularity at wo = wi.
    """
    Vl = frames.to_local(-wi, n)
    H = Vl + frames.to_local(wo, n)
    ok = (H[..., 2] > 0.0) & (jnp.sum(Vl * H, -1) > 0.0)
    Hs = jnp.where(ok[..., None], H, _up())
    vh = jnp.sum(jnp.where(ok[..., None], Vl, Hs) * Hs, -1) * jax.lax.rsqrt(jnp.sum(Hs * Hs, -1))
    return jnp.where(ok, _unified_log_density(sigma, Vl, Hs, T) - jnp.log(4.0 * vh), LOG_TINY)


def glisur_facet_logpdf(r, wi, n, wo, T=None):
    """Return log p1(wo | wi) = log p_m(h) - log(4 V . h), one GLISUR facet reflection; LOG_TINY off the support."""
    Vl = frames.to_local(-wi, n)
    H = Vl + frames.to_local(wo, n)
    good = (H[..., 2] > 0.0) & (jnp.sum(Vl * H, -1) > 0.0)
    Hs = jnp.where(good[..., None], H, _up())
    lp, ok = _glisur_log_density(r, Vl, Hs)
    vh = jnp.sum(jnp.where(good[..., None], Vl, Hs) * Hs, -1) * jax.lax.rsqrt(jnp.sum(Hs * Hs, -1))
    ok = ok & good
    return jnp.where(ok, lp - jnp.log(4.0 * jnp.where(ok, vh, 1.0)), LOG_TINY)


def seed_key(rand):
    """Return a raw threefry key from 3 uniforms (rand = (k + 1/2) 2^-23: the 23-bit integers k are exact)."""
    k = (rand * jnp.float32(2.0**23)).astype(jnp.uint32)
    w0 = k[..., 0] | (k[..., 2] << 23)
    w1 = k[..., 1] | ((k[..., 2] >> 9) << 23)
    return jnp.stack([w0, w1], -1).astype(jnp.uint32)


def _front_arc(Vl, a):
    """Return the half-width of the azimuth arc of facets at tilt a that face V, acos(-cot theta cot a)."""
    sv = jnp.sqrt(Vl[0] ** 2 + Vl[1] ** 2)
    ratio = Vl[2] * jnp.cos(a) / jnp.maximum(sv * jnp.sin(a), 1e-30)
    return jnp.arccos(-jnp.minimum(ratio, 1.0))


def unified_facet(sigma, Vl, key):
    """Return a unified facet normal (local frame, Vl_z > 0) drawn from Geant4's distribution given V . m > 0.

    The tilt by rejection from a Rayleigh(sigma) proposal with acceptance min(sin a, f_max) / a * arc(a) / pi, the
    azimuth uniform in the front arc.
    """
    fmax = jnp.minimum(1.0, 4.0 * sigma)

    def body(st):
        t, a, done = st
        uu = rng.uniforms(jax.random.fold_in(key, t), 2)
        al = sigma * jnp.sqrt(-2.0 * jnp.log(uu[0]))
        ok = (al < HALF_PI) & (uu[1] * al * math.pi < jnp.minimum(jnp.sin(al), fmax) * _front_arc(Vl, al))
        return t + 1, jnp.where(done, a, jnp.minimum(al, HALF_PI - 1e-6)), done | ok

    init = (jnp.int32(0), jnp.float32(0.0), jnp.bool_(False))
    _, a, _ = jax.lax.while_loop(lambda st: (~st[2]) & (st[0] < FACET_TRIES), body, init)
    psi = (2.0 * rng.uniforms(jax.random.fold_in(key, FACET_TRIES), 1)[0] - 1.0) * _front_arc(Vl, a)
    ph = jnp.arctan2(Vl[1], Vl[0]) + psi
    sa = jnp.sin(a)
    return jnp.stack([sa * jnp.cos(ph), sa * jnp.sin(ph), jnp.cos(a)])


def glisur_facet(r, Vl, key):
    """Return a GLISUR facet normal (local frame): M = e_z + r s, s uniform in the unit ball, kept when V . M > 0."""

    def body(st):
        t, m, done = st
        uu = rng.uniforms(jax.random.fold_in(key, t), 3)
        z = 1.0 - 2.0 * uu[1]
        phi = 2.0 * math.pi * uu[2]
        sxy = safe_sqrt(1.0 - z * z)
        M = _up() + r * jnp.cbrt(uu[0]) * jnp.stack([sxy * jnp.cos(phi), sxy * jnp.sin(phi), z])
        ok = jnp.sum(Vl * M) > 0.0
        return t + 1, jnp.where((~done) & ok, M, m), done | ok

    init = (jnp.int32(0), _up().astype(Vl.dtype), jnp.bool_(False))
    _, m, _ = jax.lax.while_loop(lambda st: (~st[2]) & (st[0] < FACET_TRIES), body, init)
    return m * jax.lax.rsqrt(jnp.sum(m * m))


def reflect_local(Vl, m):
    """Return the mirror of the local direction -Vl about the facet m."""
    return 2.0 * jnp.sum(Vl * m) * m - Vl


def _chain_of(facet):
    def chain(p, wi, n, rand):
        """Return (W (CHAIN_K_MAX, 3) the directions after each facet, zero padded; k the facets used), detached."""
        width, wi, n = sg(p["sigma_alpha"]), sg(wi), sg(n)
        key = seed_key(rand)
        V0 = frames.to_local(-wi, n)

        def body(st):
            k, V, W, _ = st
            L = reflect_local(V, facet(width, V, jax.random.fold_in(key, k)))
            return k + 1, -L, W.at[k].set(L), L[2] >= 0.0

        init = (jnp.int32(0), V0, jnp.zeros((CHAIN_K_MAX, 3), V0.dtype), jnp.bool_(False))
        k, _, W, _ = jax.lax.while_loop(lambda st: (~st[3]) & (st[0] < CHAIN_K_MAX), body, init)
        return frames.to_world(W, n), k

    return chain


def _chain_logpdf_of(facet_lp):
    def path_logpdf(p, wi, n, W, k):
        """Return the log path density of a held chain: the sum of facet_lp along its k facets (live in p, wi, n).

        A scan over the rows, an unused one evaluated at (wi, its mirror) and masked: one facet's code under
        multi-tangent forward mode, unrolled for XLA.
        """
        width = p["sigma_alpha"]
        mirror = frames.reflect(wi, n)

        def body(c, w):
            acc, prev, j = c
            use = j < k
            lp = facet_lp(width, jnp.where(use, prev, wi), n, jnp.where(use, w, mirror), p.get("tables"))
            return (acc + jnp.where(use, lp, 0.0), w, j + 1), None

        (acc, _, _), _ = jax.lax.scan(body, (jnp.zeros((), W.dtype), wi, jnp.int32(0)), W, unroll=True)
        return acc

    return path_logpdf


def _chain_lobe(name, facet, facet_lp):
    chain = _chain_of(facet)

    def end(p, wi, n, rand):
        W, k = chain(p, wi, n, rand)
        return W[k - 1]

    return Lobe(name, delta=False, surface=True, n_rand=3, params=("sigma_alpha",), sample=end, logpdf=None,
                weight=lambda p, wi, n, wo: (jnp.sum(wo * n, -1) >= 0.0).astype(wo.dtype),
                axis=lambda p, wi, n: frames.reflect(wi, n), path=chain,
                path_logpdf=_chain_logpdf_of(facet_lp))  # fmt: skip


# Geant4's unified lobe (width sigma_alpha, valid in [0.005, 0.5]) and GLISUR's (width r = 1 - polish, carried under
# the name "sigma_alpha"); chosen by the surface type's ``glossy`` name
UNIFIED = _chain_lobe("unified", unified_facet, unified_facet_logpdf)
GLISUR = _chain_lobe("glisur", glisur_facet, glisur_facet_logpdf)
CHAINS = {"unified": UNIFIED, "glisur": GLISUR}
