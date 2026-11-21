"""Geant4's facet paths, the GLOSSY outcome's lobes: the unified model's and GLISUR's.

A facet-path lobe is Geant4's DielectricMetal reflection off a rough surface: a facet normal drawn for the current
direction, a mirror reflection about it, repeated while the direction points into the surface. Its marginal density has
no closed form; its path density (the product of one facet reflection's density along the path) has, so the step
weights the whole path (``Lobe.path`` folds the step's terms over it, ``Lobe.facet_logpdf`` is one facet's).

- **unified** (GetFacetNormal, unified): tilt alpha ~ N(0, sigma_alpha), kept if ``u f_max <= sin(alpha)`` and alpha
  < pi/2 (f_max = min(1, 4 sigma)), azimuth uniform, redrawn while the facet faces away. Per solid angle of the facet
  m the photon meets D(m) [V.m > 0] / F(theta, sigma), D(m) = exp(-alpha^2 / 2 sigma^2) min(1, f_max / sin alpha) /
  (2 pi Z(sigma)), F the facing share (tabulated: float64 quadrature, float32 bicubic Hermite, C1). One reflection
  wo = mirror(wi, m) has density D(h) / (4 F V.h), h = unit(V + wo). Valid sigma_alpha: ``rules.choices.SIGMA_MIN``
  to ``SIGMA_MAX`` (the tables' span; above it Geant4's own draw departs from this distribution: ``SIGMA_MAX``).
- **GLISUR** (GetFacetNormal, glisur): M = n + r s, s uniform in the unit ball, r = 1 - polish, redrawn while
  V . M <= 0. Per solid angle of m = M / |M|: p_m = 2 q (3 c^2 + q^2) / (3 Z), c = m . n, q^2 = c^2 - (1 - r^2), Z the
  cut ball's volume. Bounded, vanishing at the cone's edge: no boundary term. The width is carried as "sigma_alpha".
  Its derivative in r is not square-integrable against p_m (p_m ~ sqrt(delta) at a distance delta from the cone's edge,
  d p_m / dr ~ 1 / sqrt(delta)): drawn from p_m, the score has tail index 3/2 (infinite variance). d/dr of the ball's
  measure is the sphere's surface measure, so d p_m / dr = (A / V) (p_shell - p_m), p_shell = 2 r (c^2 + q^2) / (q A)
  the direction density of M uniform on the sphere |M - n| = r (A its area on the photon's side, V the cut ball's
  volume). Where r is a parameter (polish's defensive role "support", ``lumen.surfaces.types.Interaction``) the facet
  is drawn from the one defensive mixture with p_shell as GLISUR's covering density, (1 - e) p_m + e p_shell,
  e = ``rules.choices.DEFENSIVE_SHARE`` (s on the unit sphere instead of in the ball: radius 1 instead of
  u^(1/3)): one facet's weight p_m / q <= 1 / (1 - e) and its derivative |d p_m / dr| / q <= (A / V) / e, bounded
  (``covered``). The reference weight carries q / q (the density drawn from, the edge floor's rule): the
  roulette keeps a photon with probability wr / eps and divides its derivative by it, so a reference weight carrying
  p_m / q would hand a survivor eps (d p_m / dr) / p_m, the unbounded score again.

Sampling: Geant4's loops, uncapped (a ``lax.while_loop`` that runs until its condition holds, as Geant4's ``do while``):
the facet draw is repeated until it is accepted and faces the photon, the facet path until the photon leaves. Both
terminate with probability 1, with geometric tails: every draw is accepted with a probability a(V) > 0, and every facet
lets the photon leave with a probability q(V) > 0, for every direction V above the surface and every width > 0 (a facet
near the normal faces any V above the surface and mirrors it upwards; at grazing V every facing facet sends it up), and
both are continuous on the closed hemisphere, so bounded below. The draws' counters are uint32: a counter repeats a key
only after 2^32 draws of one loop.

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
from lumen.lobes import Lobe
from lumen.rules.choices import DEFENSIVE_SHARE, FLOOR32
from lumen.rules.choices import UNIFIED_TABLE as UT
from lumen.units import TWO_PI

sg = jax.lax.stop_gradient
HALF_PI = 0.5 * math.pi
LOG_TINY = math.log(FLOOR32)  # a log density off the support
F_MAX_PER_SIGMA = 4.0  # Geant4's f_max = min(1, 4 sigma_alpha) (G4OpBoundaryProcess::GetFacetNormal)
UP = (0.0, 0.0, 1.0)  # the local normal


def _up():
    return literal(UP)


# unified's tables: F on (x, log sigma), x uniform on [0, 1]; log sigma = log(1/4) + j log(2)/8, j in [J_LO, J_HI]
# (resolutions and their measured errors: lumen.rules.choices, UNIFIED_TABLE)
TABLE_NX = UT["nx"]
TABLE_Y0, TABLE_DY, TABLE_J_LO, TABLE_J_HI = UT["y0"], UT["dy"], UT["j_lo"], UT["j_hi"]
TABLE_NY = TABLE_J_HI - TABLE_J_LO + 1
_GX, _GW = np.polynomial.legendre.leggauss(UT["order"])


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
    return np.exp(-alpha * alpha / (2.0 * sigma * sigma)) * np.minimum(np.sin(alpha), min(1.0, F_MAX_PER_SIGMA * sigma))


def tilt_norm(sigma, panels=UT["panels_z"]):
    """Return Z(sigma), the integral of ``tilt_density`` over (0, pi/2), split where the min switches."""
    top = min(HALF_PI, UT["cut_sigmas"] * sigma)  # the integrand is negligible past the cut
    af = min(math.asin(min(1.0, F_MAX_PER_SIGMA * sigma)), top)
    tot = 0.0
    for a, b in ((0.0, af), (af, top)):
        if b > a:
            x, w = _gauss_legendre(a, b, panels)
            tot += float(np.sum(w * tilt_density(x, sigma)))
    return tot


def front_fraction_x(x, sigma, panels=UT["panels_f"]):
    """Return the share of facets facing the photon, P(V . m > 0), at x = (2/pi) atan2(cos theta, sigma sin theta).

    The facets tilted less than atan(cot theta) all face V; past it the front arc is 2 acos(-cot theta cot alpha).
    With alpha = alpha_k + L tau^2 the arc's square-root edge is smooth in tau; the f_max kink is a panel edge.
    """
    x = np.asarray(x, np.float64)
    c = np.where(x >= 1.0, np.inf, sigma * np.tan(HALF_PI * np.minimum(x, 1.0)))
    ak = np.arctan(c)
    top = np.minimum(HALF_PI, ak + UT["cut_sigmas"] * sigma)
    span = top - ak
    tf = np.sqrt(np.clip((math.asin(min(1.0, F_MAX_PER_SIGMA * sigma)) - ak) / np.where(span > 0, span, 1.0), 0.0, 1.0))
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
    d, de = UT["fd_step"], UT["fd_end"]
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


def table_lz(y, T=None):
    """Return log zeta = log(Z / sigma^2) at the table coordinate y = log sigma, C1 cubic Hermite (``T``: the tables'
    operands, else constants)."""  # noqa: D205, D209
    T = unified_tables() if T is None else T
    j, t = _cell((y - TABLE_Y0) / TABLE_DY - TABLE_J_LO, TABLE_NY)
    a0, b0, a1, b1 = _hermite(t)
    v, dv = jnp.asarray(T["lz"]), jnp.asarray(T["lzy"])
    return a0 * v[j] + a1 * v[j + 1] + TABLE_DY * (b0 * dv[j] + b1 * dv[j + 1])


def table_front(x, y, T=None):
    """Return F at the table coordinates (x, y = log sigma), bicubic Hermite (``T``: the tables' operands)."""
    T = unified_tables() if T is None else T
    i, tx = _cell(x * (TABLE_NX - 1), TABLE_NX)
    j, ty = _cell((y - TABLE_Y0) / TABLE_DY - TABLE_J_LO, TABLE_NY)
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


def log_zeta(sigma, T=None):
    """Return log(Z(sigma) / sigma^2), C1 cubic Hermite in log sigma (``T``: the tables' operands, else constants)."""
    return table_lz(jnp.log(sigma), T)


def front_fraction(sigma, cos_t, sin_t, T=None):
    """Return F(theta, sigma), the share of facets facing the photon: bicubic Hermite on (x, log sigma)."""
    return table_front(jnp.arctan2(cos_t, sigma * sin_t) / HALF_PI, jnp.log(sigma), T)


def _unified_log_density(sigma, Vl, M, T=None):
    """Return log D(m) / F(theta, sigma) for the local facet direction M (any length); the caller masks the support.

    D(m) = exp(-alpha^2 / 2 sigma^2) min(1, f_max / sin alpha) / (2 pi Z(sigma)), f_max = min(1, 4 sigma).
    """
    inv = jax.lax.rsqrt(jnp.sum(M * M, -1))
    ch = M[..., 2] * inv
    s = safe_sqrt(M[..., 0] ** 2 + M[..., 1] ** 2) * inv  # sin(alpha), from the tangential part
    alpha = jnp.arctan2(s, ch)
    lf = jnp.minimum(0.0, jnp.log(F_MAX_PER_SIGMA * sigma))  # log f_max
    ls = jnp.log(jnp.where(s > jnp.exp(lf), s, jnp.exp(lf)))
    cos_t = Vl[..., 2]
    sin_t = safe_sqrt(Vl[..., 0] ** 2 + Vl[..., 1] ** 2)
    return (-alpha * alpha / (2.0 * sigma * sigma) + (lf - ls) - math.log(TWO_PI) - 2.0 * jnp.log(sigma)
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


def _half_vector(wi, n, wo):
    """Return (V, the half vector H = V + wo (the local normal where it is off the support), on the support, V . h)
    of one facet reflection in the local frame, V = -wi; h = unit(H)."""  # noqa: D205, D209
    Vl = frames.to_local(-wi, n)
    H = Vl + frames.to_local(wo, n)
    good = (H[..., 2] > 0.0) & (jnp.sum(Vl * H, -1) > 0.0)
    Hs = jnp.where(good[..., None], H, _up())
    vh = jnp.sum(jnp.where(good[..., None], Vl, Hs) * Hs, -1) * jax.lax.rsqrt(jnp.sum(Hs * Hs, -1))
    return Vl, Hs, good, vh


def unified_facet_logpdf(sigma, wi, n, wo, T=None):
    """Return log p1(wo | wi) = log D(h) / (4 F V.h), one unified facet reflection; LOG_TINY off the support.

    h = unit(V + wo), V = -wi: over the whole sphere, with an integrable 1 / angle singularity at wo = wi.
    """
    Vl, Hs, ok, vh = _half_vector(wi, n, wo)
    return jnp.where(ok, _unified_log_density(sigma, Vl, Hs, T) - jnp.log(4.0 * vh), LOG_TINY)


def glisur_log_area(r, h):
    """Return the log area of the sphere |M - n| = r on the photon's side of V . M = 0 (h = V . n)."""
    return jnp.log(jnp.where(h >= r, 4.0 * math.pi * r * r, TWO_PI * r * (r + jnp.minimum(h, r))))


def _glisur_log_proposal(r, Vl, M, e):
    """Return (log of (1 - e) p_m + e p_shell, inside the support) for the local facet direction M (any length)."""
    lp, ok = _glisur_log_density(r, Vl, M)
    m2 = jnp.sum(M * M, -1)
    c = jnp.where(ok, M[..., 2] * jax.lax.rsqrt(m2), 1.0)
    q2 = jnp.where(ok, r * r - (M[..., 0] ** 2 + M[..., 1] ** 2) / m2, r * r)
    ls = jnp.log(2.0 * r) + jnp.log(c * c + q2) - 0.5 * jnp.log(q2) - glisur_log_area(r, Vl[..., 2])
    return jnp.logaddexp(math.log1p(-e) + lp, math.log(e) + ls), ok


def glisur_proposal_logpdf(e):
    """Return log q1(wo | wi) of one GLISUR facet reflection drawn with the shell share e; LOG_TINY off the support."""

    def logpdf(r, wi, n, wo, T=None):
        Vl, Hs, good, vh = _half_vector(wi, n, wo)
        lq, ok = _glisur_log_proposal(r, Vl, Hs, e)
        ok = ok & good
        return jnp.where(ok, lq - jnp.log(4.0 * jnp.where(ok, vh, 1.0)), LOG_TINY)

    return logpdf


def glisur_facet_logpdf(r, wi, n, wo, T=None):
    """Return log p1(wo | wi) = log p_m(h) - log(4 V . h), one GLISUR facet reflection; LOG_TINY off the support."""
    Vl, Hs, good, vh = _half_vector(wi, n, wo)
    lp, ok = _glisur_log_density(r, Vl, Hs)
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
    ratio = Vl[2] * jnp.cos(a) / jnp.maximum(sv * jnp.sin(a), FLOOR32)
    return jnp.arccos(-jnp.minimum(ratio, 1.0))


def unified_facet(sigma, Vl, key):
    """Return a unified facet normal (local frame, Vl_z > 0) drawn from Geant4's distribution given V . m > 0.

    The tilt by rejection from a Rayleigh(sigma) proposal with acceptance min(sin a, f_max) / a * arc(a) / pi, the
    azimuth uniform in the front arc (the accepted draw's third uniform); redrawn until accepted, as Geant4.
    """
    fmax = jnp.minimum(1.0, F_MAX_PER_SIGMA * sigma)

    def body(st):
        t, _, _, _ = st
        uu = rng.uniforms(jax.random.fold_in(key, t), 3)
        al = sigma * jnp.sqrt(-2.0 * jnp.log(uu[0]))
        ok = (al < HALF_PI) & (uu[1] * al * math.pi < jnp.minimum(jnp.sin(al), fmax) * _front_arc(Vl, al))
        return t + 1, al, uu[2], ok

    init = (jnp.uint32(0), jnp.float32(0.0), jnp.float32(0.5), jnp.bool_(False))
    _, a, u, _ = jax.lax.while_loop(lambda st: ~st[3], body, init)
    psi = (2.0 * u - 1.0) * _front_arc(Vl, a)
    ph = jnp.arctan2(Vl[1], Vl[0]) + psi
    sa = jnp.sin(a)
    return jnp.stack([sa * jnp.cos(ph), sa * jnp.sin(ph), jnp.cos(a)])


def glisur_facet(r, Vl, key, e=0.0):
    """Return a GLISUR facet normal (local frame): M = e_z + r s, s uniform in the unit ball, redrawn until V.M > 0.

    ``e`` > 0: s on the unit sphere with probability e (the derivative's measure, the module docstring), the choice
    made once, before the redraws, from its own stream (``rng.SHELL_STREAM``): each part is then cut by V . M > 0 on its
    own, so the facet's density is (1 - e) times the cut ball's plus e times the cut sphere's (redrawing the choice with
    s would weight the parts by their acceptances).
    """
    shell = (rng.uniforms(jax.random.fold_in(key, rng.SHELL_STREAM), 1)[0] < e) if e else False

    def body(st):
        t, m, _ = st
        uu = rng.uniforms(jax.random.fold_in(key, t), 3)
        z = 1.0 - 2.0 * uu[1]
        phi = TWO_PI * uu[2]
        sxy = safe_sqrt(1.0 - z * z)
        rad = jnp.where(shell, 1.0, jnp.cbrt(uu[0])) if e else jnp.cbrt(uu[0])
        M = _up() + r * rad * jnp.stack([sxy * jnp.cos(phi), sxy * jnp.sin(phi), z])
        ok = jnp.sum(Vl * M) > 0.0
        return t + 1, jnp.where(ok, M, m), ok

    init = (jnp.uint32(0), _up().astype(Vl.dtype), jnp.bool_(False))
    _, m, _ = jax.lax.while_loop(lambda st: ~st[2], body, init)
    return m * jax.lax.rsqrt(jnp.sum(m * m))


def reflect_local(Vl, m):
    """Return the mirror of the local direction -Vl about the facet m."""
    return 2.0 * jnp.sum(Vl * m) * m - Vl


def _path_of(facet):
    def path(p, wi, n, rand, visit, carry):
        """Draw a facet path at p and fold ``visit`` over its facets: return (carry, w_prev, wo, k).

        The draws are held (p, wi and n detached): Geant4's loop, a facet for the current direction and the mirror
        reflection about it until the direction leaves the surface, uncapped. ``visit(carry, w0, w1)`` gets each
        facet's incoming and outgoing world directions (w0 = wi as given, live, for the first facet) and may carry
        live values. wo is the last outgoing direction, w_prev the last facet's incoming one, k the facets drawn.
        """
        width, wd, nd = sg(p["sigma_alpha"]), sg(wi), sg(n)
        key = seed_key(rand)

        def body(st):
            k, V, _, w, c, done = st
            # a lane already out (batched: vmap runs the body for every lane) draws at the normal, never at a direction
            # into the surface, for which a facet draw would not end; its result is discarded
            V = jnp.where(done, _up().astype(V.dtype), V)
            L = reflect_local(V, facet(width, V, jax.random.fold_in(key, k)))
            wo = frames.to_world(L, nd)
            return k + 1, -L, w, wo, visit(c, w, wo), L[2] >= 0.0

        init = (jnp.uint32(0), frames.to_local(-wd, nd), wi, wi, carry, jnp.bool_(False))
        k, _, w_prev, wo, c, _ = jax.lax.while_loop(lambda st: ~st[5], body, init)
        return c, w_prev, wo, k

    return path


def _facet_logpdf_of(facet_lp):
    def facet_logpdf(p, wi, n, wo):
        """Return one facet reflection's log density log p1(wo | wi) at the lobe parameters p."""
        return facet_lp(p["sigma_alpha"], wi, n, wo, p.get("tables"))

    return facet_logpdf


def _facet_path_lobe(name, facet, facet_lp, proposal_lp=None):
    path = _path_of(facet)

    def end(p, wi, n, rand):
        return path(p, wi, n, rand, lambda c, w0, w1: c, ())[2]

    return Lobe(name, surface=True, n_rand=3, params=("sigma_alpha",), sample=end, logpdf=None, path=path,
                facet_logpdf=_facet_logpdf_of(facet_lp),
                facet_proposal_logpdf=None if proposal_lp is None else _facet_logpdf_of(proposal_lp))  # fmt: skip


# Geant4's unified lobe (width sigma_alpha, valid in [SIGMA_MIN, SIGMA_MAX]) and GLISUR's (width r = 1 - polish,
# carried under the name "sigma_alpha"); chosen by the surface type's ``glossy`` name. A facet lobe whose width can
# move its support's edge declares its covering density: (its facet drawn with a share e of it, the mixture's density)
FACETS = {"unified": (unified_facet, unified_facet_logpdf, None),
          "glisur": (glisur_facet, glisur_facet_logpdf,
                     lambda e: (functools.partial(glisur_facet, e=e), glisur_proposal_logpdf(e)))}  # fmt: skip


def covered(name):
    """Return the facet lobe ``name`` drawn from the one defensive mixture (1 - e) p + e cover (``FACETS``' cover,
    e = ``DEFENSIVE_SHARE``): its weight carries the density drawn from (``facet_proposal_logpdf``)."""  # noqa: D205, D209
    facet, lp, cover = FACETS[name]
    draw, proposal = cover(DEFENSIVE_SHARE)
    return _facet_path_lobe(f"{name}+cover", draw, lp, proposal)


FACET_LOBES = {k: _facet_path_lobe(k, f, lp) for k, (f, lp, _) in FACETS.items()} | {
    f"{k}+cover": covered(k) for k, v in FACETS.items() if v[2] is not None
}
UNIFIED, GLISUR = FACET_LOBES["unified"], FACET_LOBES["glisur"]


def lobe_name(stype):
    """Return the name in ``FACET_LOBES`` of a surface type's GLOSSY facet path (its ``glossy`` lobe, or covered).

    Covered ("+cover") where a parameter of the type moves its support's edge (role "support") and the lobe declares a
    cover. The role is structure (a parameter, not a constant), so a row's lobe never depends on a value.
    """
    support = any(stype.role(p) == "support" for p in stype.params)
    return f"{stype.glossy}+cover" if support and FACETS[stype.glossy][2] is not None else stype.glossy
