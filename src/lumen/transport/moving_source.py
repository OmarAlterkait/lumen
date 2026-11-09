"""The moving-source method: derivatives for parameters that move where light starts (a source's position, direction).

The rule (Reynolds transport on path space). The expected count is an integral over paths, held in some coordinates;
its derivative is the expectation of the integrand's derivative at fixed coordinates plus a boundary term for every
discontinuity (sensor rim, wall, the switch between scattering and reaching a wall) that moves in them. The sampled
coordinates (directions, free paths) move every vertex when the source moves. This method holds the VERTICES instead:
a fixed discontinuity is then fixed, and its term is exactly zero.

How. A vertex whose direction is drawn from a density starts a CHAIN with 2 free coordinates u. The deterministic
outcomes after it (straight, refraction, mirror) are functions of u: the photon carries the chain's sensitivities --
A = dx/du, B = dd/du (live), the second order CA, CB (pairs 11, 12, 22; values), the log weight's (h), the time's (ht)
and the polarisation's (Er, Ei) first-order dependence on u -- propagated by forward mode of the step's own rules. At
the chain's END (a deposit point, the next drawn vertex) u is moved so the end stays where it was drawn (a 2x2 solve
on a surface, 3x3 with the free path in the volume) and the weight takes the change of measure (|n . Y1 x Y2|,
|det[Y1, Y2, d]|, Y = dy/du). Every correction is exactly zero in value.

The gauge (float32): u is re-anchored at every deterministic vertex (``anchor``, ``rebase``), a value-zero move that
holds the vertex just met, so every carried tangent stays O(1) on an expanding chain (a convex mirror multiplies
dy/du ~100 per bounce). A volume end near a moving origin co-moves with it, V = w U, w = exp(-r^2 / R0^2), R0 =
min(CO_MOVING_RADIUS, a quarter of the origin's clearance); its divergence enters the weight.

Known limits (SPEC section 9): the moving shadow edge between a moving source and its first held vertex is not
carried; the caustic at a held end (refraction through a PMT window) has an infinite-variance tail; an ill-conditioned
end solve after a grazing hit loses float32 precision; a zero-length segment in a medium opaque on the band's scale
carries no attenuation or time derivative.

Notes
-----
Ported from the baseline's lumen.chart (float32 with the per-vertex anchor) and the chart parts of lumen.step.
"""

# ruff: noqa: N803 -- the baseline's math names (Vl, M, W, Y, ...) as arguments, as in the module docstring
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from lumen._core import frames

sg = jax.lax.stop_gradient
PAIRS = ((0, 0), (0, 1), (1, 1))
TINY_DET = 1e-12  # a Jacobian below this: the end is not held (weight factor 1)
ANCHOR_COND = 1e-6  # the anchor's 2x2 determinant, relative to |A1| |A2|, below which the gauge is not moved
CO_MOVING_RADIUS = 0.05  # m: the largest co-moving radius about a moving origin (the baseline's chart_r0)
CLEARANCE_SHARE = 0.25  # the co-moving radius is at most this share of the origin's clearance


class Chain(NamedTuple):
    """The chain's sensitivities to its free coordinates u (see the module docstring); ``xo`` its live origin."""

    A: jax.Array  # (3, 2) dx / du
    B: jax.Array  # (3, 2) dd / du
    CA: jax.Array  # (3, 3) d2x / du_i du_j, pairs 11, 12, 22 (values)
    CB: jax.Array  # (3, 3) d2d / du_i du_j
    h: jax.Array  # (2,) d log(weight) / du (values)
    ht: jax.Array  # (2,) d time / du (values)
    Er: jax.Array  # (3, 2) d er / du
    Ei: jax.Array  # (3, 2) d ei / du
    xo: jax.Array  # (3,)


def sensitivity(state_fn, d):
    """Return d(er, ei)/du of a polarisation state_fn(d) -> (er, ei) along free_start's coordinates of d (values)."""
    e1, e2 = frames.basis(sg(d))
    tans = [jax.jvp(state_fn, (sg(d),), (e,))[1] for e in (e1, e2)]
    return tuple(sg(jnp.stack([t[c] for t in tans], 1)) for c in range(2))


def free_start(x, d, g=None, E=None):
    """Return the chain of a direction d drawn from a density in solid angle at x.

    u are the tangent-plane coordinates of d(u) = unit(d + u1 e1 + u2 e2); g the drawing density's log gradient in
    the outgoing direction; E = (Er, Ei) the polarisation's sensitivity to the direction (None: it has none).
    """
    d = sg(d)
    e1, e2 = frames.basis(d)
    z3 = jnp.zeros(3, d.dtype)
    h = jnp.zeros(2, d.dtype) if g is None else jnp.stack([jnp.sum(g * e1), jnp.sum(g * e2)])
    Er, Ei = (jnp.zeros((3, 2), d.dtype),) * 2 if E is None else E
    return Chain(jnp.zeros((3, 2), d.dtype), jnp.stack([e1, e2], 1), jnp.zeros((3, 3), d.dtype),
                 jnp.stack([-d, z3, -d], 1), h, jnp.zeros(2, d.dtype), Er, Ei, x)  # fmt: skip


def none_start(x, d):
    """Return the chain of a direction with no free coordinates (a pencil beam): nothing is held."""
    z = jnp.zeros((3, 2), d.dtype)
    return Chain(z, z, jnp.zeros((3, 3), d.dtype), jnp.zeros((3, 3), d.dtype), jnp.zeros(2, d.dtype),
                 jnp.zeros(2, d.dtype), z, z, x)  # fmt: skip


def jet(f, x, d, ch):
    """Return (f, [f_1, f_2], [f_11, f_12, f_22]) of f(x, d) along the chain's columns.

    First derivatives live (their parameter dependence is carried), second derivatives values (they only multiply
    zero-valued corrections).
    """
    V = [(ch.A[:, i], ch.B[:, i]) for i in range(2)]
    C = [(ch.CA[:, k], ch.CB[:, k]) for k in range(3)]
    out = f(x, d)
    F1 = [jax.jvp(f, (x, d), v)[1] for v in V]
    xs, ds = sg(x), sg(d)
    Vs = [(sg(a), sg(b)) for a, b in V]
    F2 = []
    for k, (i, j) in enumerate(PAIRS):

        def inner(x_, d_, i=i):
            return jax.jvp(f, (x_, d_), Vs[i])[1]

        t2 = jax.jvp(inner, (xs, ds), Vs[j])[1]
        t1 = jax.jvp(f, (xs, ds), (sg(C[k][0]), sg(C[k][1])))[1]
        F2.append(jax.tree_util.tree_map(lambda a, b: a + b, t2, t1))
    return out, F1, F2


def _det3(a, b, c):
    return jnp.sum(a * jnp.cross(b, c))


def _solve2(M, r):
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    ok = jnp.abs(det) > TINY_DET
    det = jnp.where(ok, det, 1.0)
    x = jnp.stack([M[1, 1] * r[0] - M[0, 1] * r[1], -M[1, 0] * r[0] + M[0, 0] * r[1]]) / det
    return jnp.where(ok, x, 0.0)


def surface_end(ch, y, n, Y, Y2, valid):
    """Return (du, weight factor) that hold the end y (live, on a fixed surface of normal n) by moving u.

    Y = (dy/du_1, dy/du_2) live, Y2 the second derivatives (values); du has value 0, the factor value 1.
    """
    t1, t2 = frames.basis(sg(n))
    Ys = [sg(v) for v in Y]
    M = jnp.stack([jnp.stack([jnp.sum(t * v) for v in Ys]) for t in (t1, t2)])
    dy = y - sg(y)
    du = -_solve2(M, jnp.stack([jnp.sum(t1 * dy), jnp.sum(t2 * dy)]))
    nn = sg(n)
    J = jnp.sum(nn * jnp.cross(Y[0], Y[1]))
    Js = sg(J)
    dJ = jnp.stack([jnp.sum(nn * (jnp.cross(sg(Y2[0]), Ys[1]) + jnp.cross(Ys[0], sg(Y2[1])))),
                    jnp.sum(nn * (jnp.cross(sg(Y2[1]), Ys[1]) + jnp.cross(Ys[0], sg(Y2[2]))))])  # fmt: skip
    ok = valid & (jnp.abs(Js) > TINY_DET)
    Jsafe = jnp.where(ok, Js, 1.0)
    lw = -(jnp.where(ok, J, 1.0) / Jsafe - 1.0) - jnp.sum(dJ / Jsafe * du) + jnp.sum(sg(ch.h) * du)
    du = jnp.where(ok, du, 0.0)
    return du, jnp.where(ok, jnp.exp(lw - sg(lw)), 1.0)


def anchor(ch, y, n, valid):
    """Return (du, ok): the value-zero move of u that holds y (on the fixed surface of normal n), the new gauge.

    surface_end's solve without its weight factor; an ill-conditioned solve gives 0 (the identity gauge).
    """
    t1, t2 = frames.basis(sg(n))
    Ys = [sg(ch.A[:, i]) for i in range(2)]
    M = jnp.stack([jnp.stack([jnp.sum(t * v) for v in Ys]) for t in (t1, t2)])
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    scale = jnp.sqrt(jnp.sum(Ys[0] * Ys[0]) * jnp.sum(Ys[1] * Ys[1]))  # |A1| |A2|: det's own units
    ok = valid & (jnp.abs(det) > ANCHOR_COND * scale) & (scale > 0.0)
    dy = y - sg(y)
    du = -_solve2(jnp.where(ok, M, jnp.eye(2, dtype=M.dtype)), jnp.stack([jnp.sum(t1 * dy), jnp.sum(t2 * dy)]))
    return jnp.where(ok, du, 0.0), ok


def rebase(ch, du):
    """Return the chain's first-order fields at u + du (A, B moved by CA, CB; value unchanged, du of value 0)."""
    CA, CB = sg(ch.CA), sg(ch.CB)

    def m(C):
        return jnp.stack([C[:, 0] * du[0] + C[:, 1] * du[1], C[:, 1] * du[0] + C[:, 2] * du[1]], 1)

    return ch._replace(A=ch.A + m(CA), B=ch.B + m(CB))


def volume_end(ch, x, d, s, V, valid):
    """Return (du, ds, factor) that hold the free-path end x + s d by moving (u, s), the end's own motion V."""
    Y = [ch.A[:, i] + s * ch.B[:, i] for i in range(2)]
    Ys = [sg(v) for v in Y]
    ds_ = sg(d)
    y = x + s * d
    rhs = V - (y - sg(y))
    det = _det3(Ys[0], Ys[1], ds_)
    ok = valid & (jnp.abs(det) > TINY_DET)
    dd = jnp.where(ok, det, 1.0)  # Cramer: [Y1, Y2, d] (du1, du2, ds) = rhs
    du = jnp.stack([_det3(rhs, Ys[1], ds_), _det3(Ys[0], rhs, ds_)]) / dd
    dsv = _det3(Ys[0], Ys[1], rhs) / dd
    J = _det3(Y[0], Y[1], d)
    Js = jnp.where(ok, sg(J), 1.0)
    Bs = [sg(ch.B[:, i]) for i in range(2)]
    Y2 = [sg(ch.CA[:, k]) + s * sg(ch.CB[:, k]) for k in range(3)]
    dJu = jnp.stack([_det3(Y2[0], Ys[1], ds_) + _det3(Ys[0], Y2[1], ds_) + _det3(Ys[0], Ys[1], Bs[0]),
                     _det3(Y2[1], Ys[1], ds_) + _det3(Ys[0], Y2[2], ds_) + _det3(Ys[0], Ys[1], Bs[1])])  # fmt: skip
    dJs = _det3(Bs[0], Ys[1], ds_) + _det3(Ys[0], Bs[1], ds_)
    lw = -(jnp.where(ok, J, 1.0) / Js - 1.0) - jnp.sum(dJu / Js * du) - dJs / Js * dsv + jnp.sum(sg(ch.h) * du)
    return jnp.where(ok, du, 0.0), jnp.where(ok, dsv, 0.0), jnp.where(ok, jnp.exp(lw - sg(lw)), 1.0)


def co_move(ch, y_held, r0):
    """Return (V, div V) at a volume point, both of value 0: V = w U, U the origin's motion, w = exp(-r^2 / r0^2)."""
    U = ch.xo - sg(ch.xo)
    e = y_held - sg(ch.xo)
    on = sg(r0) > 0.0
    # r0 = 0 (no clearance known): no field, exactly (a floor such as max(r0, 1e-30) squares to 0 in float32)
    r0s = jnp.where(on, sg(r0), 1.0)
    w = jnp.where(on, jnp.exp(-jnp.sum(e * e) / (r0s * r0s)), 0.0)
    return w * U, jnp.where(on, -2.0 * w / (r0s * r0s) * jnp.sum(e * U), 0.0)


def moved(v, dv_du, du):
    """Return a quantity at the chain's end with u moved: v + (dv/du) du (value v)."""
    return v + jnp.sum(sg(dv_du) * du, -1)


def fresh_state(ch, d, er0, ei0, unpolarised):
    """Return the chain with a state just drawn for an unpolarised photon carried along d (its sensitivity)."""

    def tr(E):
        return -sg(d)[:, None] * jnp.sum(sg(E)[:, None] * ch.B, 0)[None, :]

    return ch._replace(Er=jnp.where(unpolarised, tr(er0), ch.Er), Ei=jnp.where(unpolarised, tr(ei0), ch.Ei))


def free_path_end(ch, ray, s, field, pol0, valid):
    """Return the free-path end held in the volume (co-moving near a moving origin).

    ``ray`` = (x, d), ``field`` the clearance field (None: none known); ``pol0`` = (er0, ei0). Returns (weight
    factor, s, wi, er, ei, time shift, x) at the end.
    """
    x, d = ray
    y_s = x + s * d
    clear = field(None, sg(ch.xo)) if field is not None else jnp.float32(0.0)
    V, divV = co_move(ch, sg(y_s), jnp.minimum(CO_MOVING_RADIUS, CLEARANCE_SHARE * sg(clear)))
    du_s, ds_s, f_s = volume_end(ch, x, d, s, V, valid)
    er0, ei0 = pol0
    return (f_s * (1.0 + divV), s + ds_s, moved(d, ch.B, du_s), moved(er0, ch.Er, du_s), moved(ei0, ch.Ei, du_s),
            jnp.sum(sg(ch.ht) * du_s), sg(y_s) + V)  # fmt: skip
