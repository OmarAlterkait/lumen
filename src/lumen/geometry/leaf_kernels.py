"""The device kernels of one leaf along a line: a quadric's pieces (``quad_line``) and a torus's intervals.

Membership along a ray is half-open, [lo, hi) per interval; an empty slot is (INF, INF); an unbounded end is
-INF / +INF and never a crossing (|t| >= BIG). Gradients: the order of endpoints is detached, their values live; the
torus roots are found by detached bracketing and bisection and made live by one attached Newton step (the implicit
function theorem): exact first derivatives in every parameter.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from lumen.geometry.csg_build import INF

BIG = 1e29  # |t| at or beyond this is an unbounded end, not a crossing
LINEAR_REL = 1e-12  # a quadric with |alpha| below this times max |A_ij| is linear along the line
TINY = 1e-30  # a divisor below this is zero
TANGENT_ULPS = 16.0  # a discriminant within this many ulps of its terms is a tangency, not a chord
TORUS_BRACKET_ULPS = 64.0  # the torus bracket sphere is padded by this many ulps
TORUS_AXIAL = 1e-12  # a direction with a squared xy part below this is along the torus axis
BISECT_F64, BISECT_F32 = 64, 40  # bisection steps per dtype
sg = jax.lax.stop_gradient


def _ldot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _laxpy(y, t, d):
    return [y[a] + t * d[a] for a in range(3)]


def _linear_test(alpha, sa):
    if isinstance(sa, float):
        return jnp.abs(alpha) < max(LINEAR_REL * sa, TINY)
    return jnp.abs(alpha) < jnp.maximum(LINEAR_REL * sa, TINY)


def _roots(alpha, beta, gam, t_c, lin, eps, inf):  # noqa: PLR0913 -- the quadratic's terms
    """Return (rmin, rmax, tl, Dpos, b_small): the quadric's roots about t_c, the linear root and the root tests."""
    D = beta * beta - alpha * gam
    # a double root (tangent ray, a cone apex ON the ray) is only known to D's rounding, ~eps (beta^2 + |alpha gamma|):
    # a chord shorter than its own uncertainty is a tangency, not material
    Dpos = D > TANGENT_ULPS * eps * (beta * beta + jnp.abs(alpha * gam))
    sD = jnp.sqrt(jnp.where(Dpos, D, 1.0))
    sgn = jnp.where(beta >= 0.0, 1.0, -1.0)
    s_ = -(beta + sgn * jnp.where(Dpos, sD, 0.0))
    s_ok = jnp.abs(s_) > TINY
    r1 = jnp.where(lin, inf, s_ / jnp.where(lin, 1.0, alpha))
    r2 = jnp.where(s_ok, gam / jnp.where(s_ok, s_, 1.0), 0.0)
    rmin = jnp.minimum(r1, r2) + t_c
    rmax = jnp.maximum(r1, r2) + t_c
    b_small = jnp.abs(beta) < TINY
    tl = t_c - gam / jnp.where(b_small, 1.0, 2.0 * beta)
    return rmin, rmax, tl, Dpos, b_small


def _pieces(alpha, beta, gam, lin, roots, inf):
    """Return (L0, R0, L0q, R0q, L1, R1) from the roots: piece 0 [L0, R0] and piece 1 [L1, +inf)."""
    rmin, rmax, tl, Dpos, b_small = roots
    lin_ent = lin & (~b_small) & (beta < 0.0)
    lin_ext = lin & (~b_small) & (beta > 0.0)
    lin_all = lin & b_small & (gam <= 0.0)
    pos = (~lin) & (alpha > 0.0)
    neg = (~lin) & (alpha < 0.0)
    L0 = jnp.where(pos, jnp.where(Dpos, rmin, inf),
                   jnp.where(neg, -inf, jnp.where(lin_ent, tl, jnp.where(lin_ext | lin_all, -inf, inf))))  # fmt: skip
    R0 = jnp.where(pos, jnp.where(Dpos, rmax, -inf),
                   jnp.where(neg, jnp.where(Dpos, rmin, inf),
                             jnp.where(lin_ext, tl, jnp.where(lin_ent | lin_all, inf, -inf))))  # fmt: skip
    L0q = (pos & Dpos) | lin_ent
    R0q = (pos & Dpos) | (neg & Dpos) | lin_ext
    has1 = neg & Dpos
    L1 = jnp.where(has1, rmax, inf)
    R1 = jnp.where(has1, inf, -inf)
    return L0, R0, L0q, R0q, L1, R1


def quad_line(Av, bd, c, y, d, sa, eps, inf, dot=_ldot, axpy=_laxpy):  # noqa: PLR0913, N803 -- the quadric's terms; Av = A v, the docstring's Notation
    """Return the quadric part of a leaf along the line y + t d: (L0, R0, L0q, R0q, L1, R1).

    The pieces [L0, R0] and [L1, +inf) of {q <= 0} before the planes clip them (L0q / R0q: the end is a quadric
    root). ``Av(v)`` gives A v (a list of 3), ``bd(v)`` b . v, ``c`` the constant; y is the line's origin in the
    quadric's own frame (``quad_frame``), |d| = 1, ``sa`` = max |A_ij| (the linear test's scale), ``eps`` the dtype's
    machine epsilon, ``inf`` its INF. Notation: Av, Ad, Ay the matrix products; D the discriminant.

    Re-origination: from the closest approach t_o to the frame's origin, rho = alpha gamma_o / beta_o^2; at rho <= 1/2
    the roots are taken at t_o (no cancellation), else at the vertex, which then lies no farther from t_o than ~3.4x
    the roots themselves. Both bound D's relative rounding by a few eps for every A, b, c. The origin is detached.
    """
    t_o = -dot(y, d)
    y0 = axpy(y, t_o, d)
    Ad = Av(d)
    alpha = dot(Ad, d)
    lin = _linear_test(alpha, sa)
    bdd = bd(d)
    Ay0 = Av(y0)
    beta0 = dot(Ay0, d) + bdd
    gam0 = dot(y0, Ay0) + 2.0 * bd(y0) + c
    near = (~lin) & (alpha * gam0 > 0.5 * beta0 * beta0)
    t_c = sg(t_o + jnp.where(near, -beta0 / jnp.where(near, alpha, 1.0), 0.0))
    yc = axpy(y, t_c, d)
    Ayc = Av(yc)
    beta = dot(Ayc, d) + bdd
    gam = dot(yc, Ayc) + 2.0 * bd(yc) + c
    roots = _roots(alpha, beta, gam, t_c, lin, eps, inf)
    return _pieces(alpha, beta, gam, lin, roots, inf)


# ------------------------------------------------------------------------------------------------- the torus
def _torus_g(R, r, oc, d, s):  # noqa: N803 -- R, r the torus radii
    """Return g(s) = (rho - R)^2 + z^2 - r^2 at oc + s d (inside where g <= 0), cancellation-reduced."""
    x = oc[0] + s * d[0]
    y = oc[1] + s * d[1]
    z = oc[2] + s * d[2]
    rho2 = x * x + y * y
    rho = jnp.sqrt(jnp.maximum(rho2, 0.0))
    dr = (rho2 - R * R) / (rho + R)  # rho - R without cancellation of rho, R
    return dr * dr + z * z - r * r


def _torus_dg(R, r, oc, d, s):  # noqa: N803 -- R, r the torus radii
    x = oc[0] + s * d[0]
    y = oc[1] + s * d[1]
    z = oc[2] + s * d[2]
    rho2 = x * x + y * y
    rho = jnp.sqrt(jnp.maximum(rho2, jnp.finfo(rho2.dtype).tiny))
    dr = (rho2 - R * R) / (rho + R)
    drho = (x * d[0] + y * d[1]) / rho
    return 2.0 * dr * drho + 2.0 * z * d[2]


def _torus_bracket(Rd, rd, od, dd, dt):  # noqa: N803 -- Rd, rd the detached radii
    """Return (oc, s_lo, s_hi, hit_b, p1, p2): the bracket on the bounding sphere and the inflection points."""
    t_c = -jnp.dot(od, dd)
    oc = od + t_c * dd
    q0 = jnp.dot(oc, oc)
    # the bounding sphere |p| <= R + r, padded: the outer equator lies on it, so unpadded rounding puts the end inside
    Rb = (Rd + rd) * (1.0 + TORUS_BRACKET_ULPS * jnp.finfo(dt).eps)
    # oc is not exactly perpendicular to d in finite precision: solve the bracket with the residual b0 = oc . d
    b0 = jnp.dot(oc, dd)
    disc = Rb * Rb - q0 + b0 * b0
    hit_b = disc > 0.0
    sq = jnp.sqrt(jnp.where(hit_b, disc, 1.0))
    s_lo, s_hi = -b0 - sq, -b0 + sq
    a = dd[0] * dd[0] + dd[1] * dd[1]
    bxy = oc[0] * dd[0] + oc[1] * dd[1]
    cxy = oc[0] * oc[0] + oc[1] * oc[1]
    a_ok = a > TORUS_AXIAL
    a_s = jnp.where(a_ok, a, 1.0)
    s_v = jnp.where(a_ok, -bxy / a_s, 0.0)
    m2 = jnp.maximum(jnp.where(a_ok, cxy - bxy * bxy / a_s, cxy), 0.0)
    rho_i2 = jnp.cbrt(Rd * a * m2) ** 2
    w2 = (rho_i2 - m2) / a_s
    has_w = a_ok & (w2 > 0.0)
    w = jnp.sqrt(jnp.where(has_w, w2, 0.0))
    p1 = jnp.clip(s_v - w, s_lo, s_hi)
    p2 = jnp.clip(s_v + w, s_lo, s_hi)
    return oc, s_lo, s_hi, hit_b, p1, p2


def _bisect_extrema(dg, s_lo, s_hi, p1, p2, n_bisect):
    """Return the extrema: min on [s_lo, p1] (g' increasing), max on [p1, p2], min on [p2, s_hi]."""
    lo_e = jnp.stack([s_lo, p1, p2])
    hi_e = jnp.stack([p1, p2, s_hi])
    incr = jnp.asarray([True, False, True])

    def ext_body(_, lh):
        lo, hi = lh
        mid = 0.5 * (lo + hi)
        gm = jax.vmap(dg)(mid)
        go_right = jnp.where(incr, gm < 0.0, gm > 0.0)
        return jnp.where(go_right, mid, lo), jnp.where(go_right, hi, mid)

    el, eh = jax.lax.fori_loop(0, n_bisect, ext_body, (lo_e, hi_e))
    return 0.5 * (el + eh)


def _bisect_roots(g, xl, xr, n_bisect):
    """Return (the roots of g on each monotone piece [xl, xr], the membership at xl, at xr)."""
    inl = jax.vmap(g)(xl) <= 0.0
    inr = jax.vmap(g)(xr) <= 0.0

    def root_body(_, lh):
        lo, hi = lh
        mid = 0.5 * (lo + hi)
        im = jax.vmap(g)(mid) <= 0.0
        same = im == inl  # move the end with lo's membership
        return jnp.where(same, mid, lo), jnp.where(same, hi, mid)

    rl, rh = jax.lax.fori_loop(0, n_bisect, root_body, (xl, xr))
    return 0.5 * (rl + rh), inl, inr


def _polished(R, r, o, d, s_star):  # noqa: N803 -- R, r the torus radii
    """Return the roots with one attached Newton step: live in R, r, o, d; the value stays the bisection root."""
    t_c_l = -jnp.dot(o, d)
    oc_l = o + t_c_l * d

    def polish(s0):
        gv = _torus_g(R, r, oc_l, d, s0)
        gp = _torus_dg(R, r, oc_l, d, s0)
        ok = jnp.abs(gp) > TINY * (1.0 + jnp.abs(gv))
        return s0 - jnp.where(ok, gv / jnp.where(ok, gp, 1.0), 0.0)

    s_live = jax.vmap(polish)(s_star)
    return s_star + (s_live - sg(s_live)), t_c_l


def torus_set(R, r, o, d, n_bisect=None):  # noqa: N803 -- R, r the torus radii
    """Return the interval set (two slots) of a solid torus about z; o relative to its centre (torus frame), |d| = 1.

    Root isolation is exact up to rounding: g(s) = |p|^2 - 2 R rho + R^2 - r^2 has at most two inflection points, in
    closed form; they split the bounding-sphere bracket into convex / concave / convex parts, one extremum each (found
    by bisection on g'), giving at most six monotone pieces with at most one root each (bisection on the membership).
    Detached; one attached Newton step on g makes each root live. Notation: R, r the swept and tube radii.
    """
    dt = o.dtype
    inf = jnp.asarray(INF, dt)
    if n_bisect is None:
        n_bisect = BISECT_F64 if dt == jnp.float64 else BISECT_F32
    Rd, rd, od, dd = sg(R), sg(r), sg(o), sg(d)
    oc, s_lo, s_hi, hit_b, p1, p2 = _torus_bracket(Rd, rd, od, dd, dt)

    def g(s):
        return _torus_g(Rd, rd, oc, dd, s)

    def dg(s):
        return _torus_dg(Rd, rd, oc, dd, s)

    e = _bisect_extrema(dg, s_lo, s_hi, p1, p2, n_bisect)
    knots = jnp.stack([s_lo, e[0], p1, e[1], p2, e[2], s_hi])
    xl, xr = knots[:-1], knots[1:]
    s_star, inl, inr = _bisect_roots(g, xl, xr, n_bisect)
    has_root = (inl != inr) & (xr > xl) & hit_b
    s_root, t_c_l = _polished(R, r, o, d, s_star)
    t_root = jnp.where(has_root, s_root + t_c_l, inf)
    order = jnp.argsort(sg(t_root))
    ts = t_root[order]
    # outside at the bracket's start (g >= 0 on the bounding sphere): roots alternate entry / exit
    lo = jnp.stack([ts[0], ts[2]])
    hi = jnp.stack([ts[1], ts[3]])
    ok = hi < BIG
    lo = jnp.where(ok, lo, inf)
    hi = jnp.where(ok, hi, inf)
    f = jnp.where(ok, 0, -1).astype(jnp.int32)
    return dict(lo=lo, hi=hi, flo=f, fhi=f)
