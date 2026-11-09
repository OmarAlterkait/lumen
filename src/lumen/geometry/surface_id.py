"""Surface identity: the surface a photon sits on, packed in one int32; the live crossing distance; mesh hits.

The photon's side of the surface it sits on is its state (its volume), so at the next query only the direction counts
(Geant4's fEntering / fExiting). The code (``srf``) holds the item met with its leaf, face and outward sign:

- a plain item (or -1): the item itself, with no face (the band rules);
- a volume's face: ``su + ((item * LM + leaf) * FM + face) * 4 + 2 o + e``;
- a mesh triangle k: ``sm + 4 k + 2 o + e``;

where o says the outward normal is +grad and e that the event entered the item. Notation: su, sm the first unit and
mesh codes, LM the most leaves of a kind, FM the most faces of a leaf.

Gradients: the traversal is detached; the winning (item, leaf, face) is re-evaluated live, ``dt = -g . (do + t dd) /
(g . d)`` (a custom_jvp), g the winning leaf's gradient (a mesh triangle's normal).

Notes
-----
Ported from the baseline's lumen.geometry.navigator (srf_pack, srf_decode, hit_grad, live_t, the mesh hit's plane).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from lumen.rules import navigation as rules

F32, I32 = jnp.float32, jnp.int32
INF = float(jnp.finfo(jnp.float32).max)
NORM_FLOOR = 1e-30  # a gradient's norm floor
SLOPE_REL = 1e-12  # the live t's denominator g . d below this times |g| is no crossing (zero derivative)
MESH_PARALLEL = 1e-6  # a mesh hit's plane is used where |n . d| is above this
sg = lax.stop_gradient


def srf_decode(nav, s):
    """Return (item, leaf, face, o, known, e) of a surface code; a plain item (or -1): leaf -1, not known."""
    su, sm, LM, FM = nav.SRF
    D = nav.dev()
    s = jnp.asarray(s, I32)
    isu, ism = (s >= su) & (s < sm), s >= sm
    u = jnp.maximum(s - su, 0)
    eu, u = u % 2, u // 2
    ou, u = u % 2, u // 2
    fu, u = u % FM, u // FM
    lu, iu = u % LM, u // LM
    km = jnp.maximum(s - sm, 0)
    em, km = km % 2, km // 2
    om, km = km % 2, km // 2
    im = nav.CH0 + D["tchunk"][jnp.clip(km, 0, D["tchunk"].shape[0] - 1)]
    item = jnp.where(isu, iu, jnp.where(ism, im, s))
    leaf = jnp.where(isu, lu, jnp.where(ism, km, -1))
    return (item, leaf, jnp.where(isu, fu, -1), jnp.where(isu, ou, om) == 1, isu | ism,
            jnp.where(isu, eu, em) == 1)  # fmt: skip


def srf_pack(nav, item, leaf, face, o, e):
    """Return the surface code of (item, leaf, face, o, e); without a leaf (or a unit face) the plain item."""
    su, sm, LM, FM = nav.SRF
    mesh = item >= nav.CH0
    b2 = 2 * o.astype(I32) + e.astype(I32)
    cu = su + ((item * LM + leaf) * FM + face) * 4 + b2
    cm = sm + leaf * 4 + b2
    return jnp.where((item < 0) | (leaf < 0) | (~mesh & (face < 0)), item, jnp.where(mesh, cm, cu)).astype(I32)


def hit_grad(nav, item, leaf, face, x):
    """Return the outward-or-inward gradient (world) of the winning surface at x: a volume's leaf, or a triangle."""
    D = nav.dev()
    i = jnp.clip(item, 0, nav.n - 1)
    R, t = D["R"][i], D["t"][i]
    g = R @ nav.E.leaf_grad(D["kind"][i], leaf, face, R.T @ (x - t))
    if not nav.n_mesh:
        return g
    return jnp.where(item >= nav.CH0, D["N"][jnp.clip(leaf, 0, D["N"].shape[0] - 1)], g)


def _live_rule(nav):
    """Return the custom_jvp rule of the live t: the value is t0 itself; its derivative the implicit function's."""

    @jax.custom_jvp
    def live(o_, d_, it_, lf_, fc_, t0_):
        return t0_

    @live.defjvp
    def live_jvp(primals, tangents):
        o_, d_, it_, lf_, fc_, t0_ = primals
        do_, dd_ = tangents[0], tangents[1]
        # t inside the rule is live itself (value t0_), and so is the primal it returns: a derivative of the rule
        # (second order) carries d t / d (o, d) through the hit point, the t dd term and every product with t
        t_ = live(o_, d_, it_, lf_, fc_, t0_)
        g = hit_grad(nav, it_, lf_, fc_, o_ + t_ * d_)
        den = jnp.dot(g, d_)
        ok = jnp.abs(den) > SLOPE_REL * jnp.maximum(jnp.linalg.norm(g), NORM_FLOOR)
        dt = jnp.where(ok, -jnp.dot(g, do_ + t_ * dd_) / jnp.where(ok, den, 1.0), 0.0)
        return t_, dt

    return live


def live_t(nav, item, leaf, face, o, d, bt):  # noqa: PLR0913 -- the winner and the ray
    """Return bt (detached) as the value; under differentiation the implicit-function derivative of the winner.

    ``dt = -(g . (do + t dd)) / (g . d)`` (custom_jvp: the forward kernel computes nothing for it).
    """
    if nav._live is None:
        nav._live = _live_rule(nav)
    t0 = sg(jnp.where(bt < INF, bt, 0.0))
    return jnp.where(bt < INF, nav._live(o, d, item, leaf, face, t0), bt)


def mesh_plane_t(nav, win, od, dd, xa):
    """Return a mesh hit's t from its plane, ``FACET_SHORT_ULPS`` ulps short of it (the facet rule).

    ``win`` = (live, item, leaf, t). The plane is taken from the triangle's vertex nearest the start (V - o small,
    float32 holds it): the hit point never lies past the plane, behind the triangle, where a neighbouring face of the
    frame would be met from its back.
    """
    live, bi, bl, bt = win
    D = nav.dev()
    km = jnp.clip(bl, 0, D["N"].shape[0] - 1)
    nk = D["N"][km]
    nd = jnp.dot(nk, dd)
    ok_m = live & (bi >= nav.CH0) & (jnp.abs(nd) > F32(MESH_PARALLEL))
    P0, P1, P2 = D["V0"][km], D["V1"][km], D["V2"][km]
    m0, m1, m2 = (jnp.max(jnp.abs(P_ - od)) for P_ in (P0, P1, P2))
    Pn = jnp.where((m0 <= m1) & (m0 <= m2), P0, jnp.where(m1 <= m2, P1, P2))
    t_pl = jnp.dot(nk, Pn - od) / jnp.where(ok_m, nd, F32(1.0))
    ulp = F32(rules.FLOAT32_EPS) * (xa + jnp.abs(bt))
    short = F32(rules.FACET_SHORT_ULPS) * ulp / jnp.abs(jnp.where(ok_m, nd, F32(1.0)))
    return jnp.where(ok_m, jnp.maximum(t_pl - short, F32(0.0)), bt)
