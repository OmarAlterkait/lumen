"""A proven lower bound of the distance to every surface, on a grid: the diffusion jump's clearance.

The jump (``lumen.transport.fastpath``) moves a photon to a point of a sphere around it; that sphere must cross no
surface a photon can meet, visible or not. ``bound`` gives, at the nodes of a regular grid, a number no larger than
the distance from the node to the nearest surface of any volume the navigator can be in (``navigator_build.
navigable``: its per-classes output, so no transparent volume it dissolved, whose faces bound nothing a photon sees)
and of every mesh, from the navigator's own arrays (the
leaf coefficients the device evaluates, in its own precision, and the volumes' padded world boxes), in float64:

- a volume whose box does not hold the node: the distance to that box (the solid lies inside it);
- else the minimum over the volume's CSG leaves of a lower bound of the distance to the leaf's surfaces (the surface
  of a boolean of leaves lies in the union of the leaves' surfaces; a leaf as the ray scan evaluates it is gated by
  its leaf box, whose faces are counted too):
    plane n.y = h:                    |n.y - h| / |n|                                         (exact)
    quadric f = y'Ay + 2b'y + c:      the root r >= 0 of ||A|| r^2 + |grad f| r = |f|          (|f(y + e) - f(y)| <=
                                      |grad f| |e| + ||A|| |e|^2, so f keeps its sign within r)
    torus (rho - R)^2 + z^2 = r^2 in u = M y:  | sqrt((rho - R)^2 + z^2) - r | / ||M||  (exact when M is a rotation)
    leaf box:                         the distance to its faces (inside) or to the box (outside);
- every mesh (``mesh_boxes``: the world box of its float32 triangles, which hold its surface): 0 inside the box, the
  distance to the box outside.
Minus the navigator's band at the grid's coordinate scale (``lumen.rules.navigation.band``, float32's in either
precision: the jump itself works in float32, so a float64 navigator only makes this bound looser), rounded down to
float32. The device reads c(x) = clear[node] - |x - node| at the node nearest
x, again a lower bound (the distance is 1-Lipschitz), so any spacing gives a proven bound: the spacing changes only
how tight it is, i.e. how often and how far the jump goes (speed), never what it is allowed to do.

It runs on the device in float64 (``jax.experimental.enable_x64``) over flat evaluation lists, each volume and mesh
only at the nodes it can bound below ``cap`` and below what the leaf bounds already gave (``bound``).

    g = grid(nav, volumes, h)              # lo, n of a grid over the volumes' boxes
    clear = bound(nav, g["lo"], h, g["n"], cap)
"""

from __future__ import annotations

import functools

import numpy as np

from lumen.geometry.navigator_build import navigable
from lumen.rules import navigation as rules
from lumen.rules.tolerances import FLOAT64_TINY, WORLD_BOX_PAD_ABS, WORLD_BOX_PAD_REL


def grid(nav, volumes, h):
    """Return dict(lo, n) of the grid of spacing h (m) whose nodes cover the union of the volumes' world boxes."""
    lo = np.min(np.asarray(nav.lo_w, np.float64)[list(volumes)], 0)
    hi = np.max(np.asarray(nav.hi_w, np.float64)[list(volumes)], 0)
    n = np.floor((hi - lo) / h).astype(np.int64) + 2  # the last node at or past hi
    return dict(lo=lo, n=n)


def mesh_boxes(nav):
    """Return (lo, hi) (n_mesh, 3) float64: each mesh's world box over the float32 world triangles the device tests
    (``host`` V0, V1, V2, by chunk), padded as the navigator pads its boxes. A mesh's surface lies in it.
    """  # noqa: D205
    A = nav.arrays
    nm = int(A.n_mesh)
    lo, hi = np.full((nm, 3), np.inf), np.full((nm, 3), -np.inf)
    if not nm:
        return lo, hi
    H = A.host
    V = np.stack([np.asarray(H[k], np.float64) for k in ("V0", "V1", "V2")], 1)  # (triangles, vertex, axis)
    for s, k, m in zip(np.asarray(H["cstart"]), np.asarray(H["cn"]), np.asarray(H["cmesh"]), strict=True):
        t = V[int(s) : int(s) + int(k)].reshape(-1, 3)
        lo[m], hi[m] = np.minimum(lo[m], t.min(0)), np.maximum(hi[m], t.max(0))
    pad = WORLD_BOX_PAD_REL * (np.abs(lo) + np.abs(hi)) + WORLD_BOX_PAD_ABS
    return lo - pad, hi + pad


def _leaf_tables(T):  # noqa: N803 -- T the CSG tables
    """Return the leaves' bound parameters as padded float64 host arrays (one row per leaf)."""
    G = len(T["typ"])
    P = max(1, int(np.max(T["np_"])) if G else 1)
    f = lambda k: np.asarray(T[k], np.float64)  # noqa: E731
    PN, PH, PM = np.zeros((G, P, 3)), np.zeros((G, P)), np.zeros((G, P), bool)
    for g in range(G):
        for j, q in enumerate(range(int(T["p0"][g]), int(T["p0"][g] + T["np_"][g]))):
            PN[g, j] = (T["n0"][q], T["n1"][q], T["n2"][q])
            PH[g, j], PM[g, j] = T["h"][q], True
    A = np.stack([np.stack([f(f"A{a}{b}") for b in range(3)], -1) for a in range(3)], -2)  # (G, 3, 3)
    M = np.stack([np.stack([f(f"m{a}{b}") for b in range(3)], -1) for a in range(3)], -2)
    an = np.array([np.max(np.abs(np.linalg.eigvalsh(0.5 * (a + a.T)))) for a in A]) if G else np.zeros(0)
    mn = np.array([np.linalg.norm(m, 2) for m in M]) if G else np.zeros(0)
    BL = np.stack([f(f"bl{a}") for a in range(3)], -1)
    BH = np.stack([f(f"bh{a}") for a in range(3)], -1)
    return dict(O=np.stack([f("o0"), f("o1"), f("o2")], -1), PN=PN, PH=PH, PM=PM,
                PNn=np.maximum(np.linalg.norm(PN, axis=-1), FLOAT64_TINY), BL=BL, BH=BH, HB=np.all(BL <= BH, -1),
                TYP=np.asarray(T["typ"], np.int64), Q=np.stack([f("q0"), f("q1"), f("q2")], -1), A=A,
                B=np.stack([f("b0"), f("b1"), f("b2")], -1), C=f("c"), AN=an, M=M, MN=np.maximum(mn, FLOAT64_TINY),
                RR=f("R"), rr=f("r"))  # fmt: skip


def _leaf_eval(L, g, xl):  # noqa: N803 -- L the leaf tables
    """Return (K, N) lower bounds of the distance from points xl (K, N, 3; each row in its volume's frame) to the
    surfaces of leaves g (K,) (the rule of the module docstring, vectorised; ``L`` the leaf tables on the device).
    """  # noqa: D205
    import jax.numpy as jnp

    y0 = xl - L["O"][g][:, None, :]
    dp = jnp.abs(jnp.einsum("kni,kpi->knp", y0, L["PN"][g]) - L["PH"][g][:, None, :]) / L["PNn"][g][:, None, :]
    cv = jnp.min(jnp.where(L["PM"][g][:, None, :], dp, jnp.inf), -1)
    bl, bh = L["BL"][g][:, None, :], L["BH"][g][:, None, :]
    out = jnp.maximum(jnp.maximum(bl - xl, xl - bh), 0.0)
    dout = jnp.sqrt(jnp.sum(out * out, -1))
    din = jnp.min(jnp.minimum(xl - bl, bh - xl), -1)
    cv = jnp.where(L["HB"][g][:, None], jnp.minimum(cv, jnp.where(dout > 0, dout, jnp.maximum(din, 0.0))), cv)
    y = xl - L["Q"][g][:, None, :]
    A, b = L["A"][g], L["B"][g][:, None, :]
    f = jnp.einsum("kni,kij,knj->kn", y, A, y) + 2.0 * jnp.sum(y * b, -1) + L["C"][g][:, None]
    gr = jnp.sqrt(jnp.sum((2.0 * (jnp.einsum("kni,kji->knj", y, A) + b)) ** 2, -1))
    an, af = L["AN"][g][:, None], jnp.abs(f)
    r = jnp.where(an > 0, 2.0 * af / (gr + jnp.sqrt(gr * gr + 4.0 * an * af)),
                  jnp.where(gr > 0, af / jnp.maximum(gr, FLOAT64_TINY), jnp.where(af > 0, jnp.inf, 0.0)))  # fmt: skip
    cv = jnp.where((L["TYP"][g] == 1)[:, None], jnp.minimum(cv, r), cv)
    u = jnp.einsum("kni,kji->knj", y0, L["M"][g])
    rho = jnp.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2)
    tor = jnp.sqrt((rho - L["RR"][g][:, None]) ** 2 + u[..., 2] ** 2)
    tor = jnp.abs(tor - L["rr"][g][:, None]) / L["MN"][g][:, None]
    return jnp.where((L["TYP"][g] == 2)[:, None], jnp.minimum(cv, tor), cv)


EVALS = 1 << 27  # bytes of leaf planes per device call: memory per call only, the result does not depend on it


@functools.cache
def _kernels():
    """The bound's jitted passes (float64: traced inside ``jax.experimental.enable_x64``). Each evaluates a chunk of
    E flat evaluations e0 + arange(E) of the concatenated per-item node ranges (item k: ``start[k]`` its first
    evaluation, ``i0[k]`` its first node index, ``dim[k]`` its range) and min-reduces them into the flat grid.
    """  # noqa: D205, D401
    import jax
    import jax.numpy as jnp

    def nodes(e0, E, start, i0, dim, lo, h, n):  # noqa: N803, PLR0913 -- E the chunk size; the chunk's items, node indices and coordinates
        e = e0 + jnp.arange(E, dtype=jnp.int64)
        k = jnp.clip(jnp.searchsorted(start, e, side="right") - 1, 0, start.shape[0] - 1)
        loc = e - start[k]
        d = dim[k]
        ijk = i0[k] + jnp.stack([loc // (d[:, 1] * d[:, 2]), (loc // d[:, 2]) % d[:, 1], loc % d[:, 2]], -1)
        ok = (e < start[-1]) & jnp.all((ijk >= 0) & (ijk < n), -1)
        ijk = jnp.clip(ijk, 0, n - 1)
        return k, ok, lo + h * ijk.astype(jnp.float64), (ijk[:, 0] * n[1] + ijk[:, 1]) * n[2] + ijk[:, 2]

    @functools.partial(jax.jit, static_argnums=1)
    def boxes(out, E, e0, start, i0, dim, vol, lo, h, n, VLO, VHI):  # noqa: N803, PLR0913 -- pass A: outside a box, the distance
        k, ok, x, flat = nodes(e0, E, start, i0, dim, lo, h, n)
        v = vol[k]
        o = jnp.maximum(jnp.maximum(VLO[v] - x, x - VHI[v]), 0.0)
        d = jnp.sqrt(jnp.sum(o * o, -1))
        return out.at[jnp.where(ok & (d > 0), flat, out.shape[0])].min(d, mode="drop")

    @functools.partial(jax.jit, static_argnums=(1, 2, 3))
    def inner(out, E, NL, P, e0, start, i0, dim, vol, mesh, lo, h, n, R, t, kind, l0, nl, L, VLO, VHI):  # noqa: N803, PLR0913 -- pass B
        k, ok, x, flat = nodes(e0, E, start, i0, dim, lo, h, n)
        v = vol[k]
        o = jnp.maximum(jnp.maximum(VLO[v] - x, x - VHI[v]), 0.0)
        ok = ok & (jnp.sum(o * o, -1) == 0.0)  # inside the box (pass A takes the others: the same arithmetic)
        xl = jnp.einsum("ei,eij->ej", x - t[v], R[v])  # R^T (x - t): the volume's frame
        kv = kind[v]
        slot = jnp.arange(NL)
        L = dict(L, PN=L["PN"][:, :P], PH=L["PH"][:, :P], PM=L["PM"][:, :P], PNn=L["PNn"][:, :P])  # the chunk's planes
        g = jnp.clip(l0[kv][:, None] + slot[None], 0, L["TYP"].shape[0] - 1)  # (E, NL) the kind's leaves
        cv = _leaf_eval(L, g.reshape(-1), jnp.repeat(xl, NL, 0)[:, None, :])[:, 0].reshape(E, NL)
        cv = jnp.min(jnp.where(slot[None] < nl[kv][:, None], cv, jnp.inf), -1)
        val = jnp.where(mesh[k], 0.0, cv)
        return out.at[jnp.where(ok, flat, out.shape[0])].min(val, mode="drop")

    return boxes, inner


def _ranges(blo, bhi, lo, h, n, pad):
    """Return (first node index, range dims) per box of the nodes within ``pad`` of it, plus one node each side
    (a superset: whatever rounding gives the node coordinates).
    """  # noqa: D205
    a = np.floor((blo - pad - lo) / h).astype(np.int64) - 1
    b = np.ceil((bhi + pad - lo) / h).astype(np.int64) + 1
    a, b = np.clip(a, 0, n - 1), np.clip(b, 0, n - 1)
    return a, np.maximum(b - a + 1, 0) * (bhi >= blo).all(-1, keepdims=True)


BLOCK = 16  # nodes along each axis of the blocks pass A's ranges are cut into (speed only: the result is the same)


def _near(U, it, i0, dim, blo, bhi, lo, h, n):  # noqa: N803, PLR0913 -- U the bound so far
    """Return (box, first node index, range dims) of the pieces of pass A's ranges worth evaluating.

    Pass A's ranges (boxes ``it``, ranges i0, dim) are cut into ``BLOCK``-node blocks; a block is kept where the
    distance between the box and the block's nodes' box is at most the largest bound so far there (U, at the nodes)
    plus one spacing.

    A block left out holds no node where the box's term is below the bound so far: the term is the node's distance to
    the box to within float64 rounding (relative 1e-15 of the coordinates), and that distance exceeds U there by a
    spacing. So the minimum, the result, is unchanged.
    """
    nb = -(-np.asarray(n) // BLOCK)
    Um = np.pad(U, [(0, int(x)) for x in nb * BLOCK - n]).reshape(nb[0], BLOCK, nb[1], BLOCK, nb[2], BLOCK)
    Um = Um.max((1, 3, 5)) + h  # U >= 0: the padding raises no maximum
    b0 = [np.arange(int(x)) * BLOCK for x in nb]  # per axis, the blocks' first and last node indices
    b1 = [np.minimum(x + BLOCK, int(m)) - 1 for x, m in zip(b0, n, strict=True)]
    gap, over = [], []
    for ax in range(3):
        g = np.maximum(
            np.maximum(blo[it, ax, None] - (lo[ax] + h * b1[ax]), lo[ax] + h * b0[ax] - bhi[it, ax, None]), 0
        )
        gap.append(g)
        over.append((b0[ax] < (i0[:, ax] + dim[:, ax])[:, None]) & (b1[ax] >= i0[:, ax, None]))
    out = [(it[:0], i0[:0], dim[:0])]
    step = max(1, (1 << 22) // int(np.prod(nb)))
    for s0 in range(0, len(it), step):
        sl = slice(s0, s0 + step)
        d2 = gap[0][sl, :, None, None] ** 2 + gap[1][sl, None, :, None] ** 2 + gap[2][sl, None, None, :] ** 2
        keep = (
            (d2 <= Um[None] ** 2) & over[0][sl, :, None, None] & over[1][sl, None, :, None] & over[2][sl, None, None, :]
        )
        r, *bk = np.nonzero(keep)
        r = r + s0
        a0 = np.stack([np.maximum(i0[r, ax], b0[ax][bk[ax]]) for ax in range(3)], -1)
        a1 = np.stack([np.minimum(i0[r, ax] + dim[r, ax] - 1, b1[ax][bk[ax]]) for ax in range(3)], -1)
        out.append((it[r], a0, a1 - a0 + 1))
    return tuple(np.concatenate([o[k] for o in out]) for k in range(3))


def bound(nav, lo, h, n, cap):
    """Return the proven clearance (float32, shape n) at the nodes lo + h i, at most ``cap`` (m) before the band.

    On the device in float64 (``jax.experimental.enable_x64``), in two passes over flat evaluation lists: (B) at the
    nodes of its box that lie inside it, a volume's leaf bound, a mesh's 0; (A) for every volume and mesh, the distance
    to its box at the nodes within ``cap`` of it that lie outside it, where it may be below the bound pass B left
    (``_near``) (inside: the box distance is 0, with the same arithmetic as pass A, so each node takes exactly one of
    the two terms of each volume). The ranges are padded by one node (a superset whatever the rounding). A node's
    value is min(cap, every term): a term >= cap, or above the bound so far, cannot lower it. The volumes are those
    the navigator can be in (``navigator_build.navigable``), and every mesh.
    """
    import jax.numpy as jnp
    from jax.experimental import enable_x64

    A = nav.arrays
    H = A.host
    T = {k: np.asarray(v, np.float64) for k, v in A.csg.T.items()}
    l0, nl = np.asarray(A.csg.K["l0"], np.int64), np.asarray(A.csg.K["nl"], np.int64)
    nv = int(A.n)
    mlo, mhi = mesh_boxes(nav)  # boxes nv.. are the meshes': 0 inside, the box distance outside
    lo_w = np.concatenate([np.asarray(A.lo_w, np.float64)[:nv], mlo])
    hi_w = np.concatenate([np.asarray(A.hi_w, np.float64)[:nv], mhi])
    R, tt, kind = np.asarray(H["R"], np.float64), np.asarray(H["t"], np.float64), np.asarray(H["kind"], np.int64)
    lo, n = np.asarray(lo, np.float64), np.asarray(n, np.int64)
    cap = float(cap)
    nvm = len(lo_w)
    kind_all = np.concatenate([kind[:nv], np.zeros(nvm - nv, np.int64)])
    seen = np.zeros(nv, bool)
    seen[navigable(A)] = True
    with enable_x64():
        boxes, inner = _kernels()
        Ld = {k: jnp.asarray(v) for k, v in _leaf_tables(T).items()}
        out = jnp.full(int(np.prod(n)) + 1, cap)  # the last cell takes the dropped evaluations
        dev = dict(lo=jnp.asarray(lo), h=jnp.asarray(h, jnp.float64), n=jnp.asarray(n), VLO=jnp.asarray(lo_w),
                   VHI=jnp.asarray(hi_w), R=jnp.asarray(R), t=jnp.asarray(tt), kind=jnp.asarray(kind_all),
                   l0=jnp.asarray(l0), nl=jnp.asarray(nl))  # fmt: skip
        npl = np.asarray(T["np_"], np.int64)  # per kind: its leaves (NL) and the most planes of one of them (P)
        pk = np.array([max(1, int(npl[a : a + b].max(initial=1))) for a, b in zip(l0, nl, strict=True)], np.int64)
        for p, pad in ((1, 0.0), (0, cap)):  # pass B, then pass A
            i0, dim = _ranges(lo_w, hi_w, lo, h, n, pad)
            cnt = np.prod(dim, -1)
            cnt[:nv] = np.where(seen, cnt[:nv], 0)  # a volume the navigator cannot be in bounds nothing it meets
            items = np.flatnonzero(cnt > 0)
            if p == 0:  # pass A: the pieces of its ranges that may lower the bound so far
                vol, i0, dim = _near(np.asarray(out[:-1]).reshape(tuple(n)), items, i0[items], dim[items], lo_w, hi_w,
                                     lo, h, n)  # fmt: skip
                groups = [(0, 0, vol, i0, dim)]
            else:
                key = np.stack(
                    [np.where(items < nv, nl[kind_all[items]], 1), np.where(items < nv, pk[kind_all[items]], 1)], -1
                )
                groups = [(NL, P, items[g], i0[items[g]], dim[items[g]])  # one group per (leaves, planes)
                          for NL, P in np.unique(key, axis=0) for g in [np.all(key == (NL, P), -1)]]  # fmt: skip
            for NL, P, it, gi0, gdim in groups:  # its calls sized to it
                start = np.concatenate([[0], np.cumsum(np.prod(gdim, -1))]).astype(np.int64)
                sd, i0d, dimd = jnp.asarray(start), jnp.asarray(gi0), jnp.asarray(gdim)
                vol, mesh = jnp.asarray(it), jnp.asarray(it >= nv)
                E = 1 << 22 if p == 0 else 1 << max(10, int(np.log2(EVALS // (32 * int(NL) * int(P)))))
                for e0 in range(0, int(start[-1]), E):
                    if p == 0:
                        out = boxes(out, E, e0, sd, i0d, dimd, vol, dev["lo"], dev["h"], dev["n"], dev["VLO"],
                                    dev["VHI"])  # fmt: skip
                    else:
                        out = inner(out, E, int(NL), int(P), e0, sd, i0d, dimd, vol, mesh, dev["lo"], dev["h"],
                                    dev["n"], dev["R"], dev["t"], dev["kind"], dev["l0"], dev["nl"], Ld, dev["VLO"],
                                    dev["VHI"])  # fmt: skip
        out = np.asarray(out[:-1]).reshape(tuple(n))
    scale = float(np.max(np.abs(np.concatenate([lo, lo + h * (n - 1)])))) + float(cap)
    c = np.maximum(out - float(rules.band(scale)), 0.0)
    c32 = c.astype(np.float32)
    up = c32.astype(np.float64) > c
    c32[up] = np.nextafter(c32[up], np.float32(0))
    return c32
