"""BVHs and frame groups (host): the navigator's daughter boxes, binned-SAH trees built level by level.

    tree = build_bvh(lo, hi)                       # dict(lo, hi (m, 3) float32, skip, start, count, items)
    trees = build_many(lo, hi, bounds)             # the trees of the box ranges [bounds[i], bounds[i + 1]) at once
    gkey, n_groups, rotations, accepted = frame_groups_all(R, hb, mother, daughter)

A tree is stored depth first: a node's left child follows it, ``skip`` is the node after its subtree (the stackless
order), and a leaf (``count`` > 0) covers ``items[start:start + count]``.

Notes
-----
Ported from the baseline's lumen.geometry.navigator (its BVH builder, after lucid.csg.bvh's level-synchronous
rewrite).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

LEAF_SIZE = 2  # a node of at most this many boxes is a leaf
BINS = 16  # SAH bins per axis
MIN_GROUP = 8  # a rotation shared by at least this many daughters (or by all) may be a frame group
GROUP_GAIN = 4.0  # ... when its daughters' world boxes are at least this times their own (summed surface areas)
ROT_DECIMALS = 6  # rotations are grouped by their entries rounded to this many decimals
IDENT_ATOL = 1e-9  # np.allclose(R, I, atol=1e-9): an identity rotation is no frame group
IDENT_RTOL = 1e-5


def build_bvh(lo, hi, leaf_size=LEAF_SIZE, bins=BINS):
    """Return the binned-SAH BVH of the boxes (lo, hi): dict(lo, hi, skip, start, count, items)."""
    out = build_many(lo, hi, [0, len(lo)], leaf_size, bins)
    del out["tree0"], out["n_levels"]
    return out


def _area3(a, b):
    """Return the surface areas of component-major boxes (3, ...)."""
    e = np.maximum(b - a, 0.0)
    return 2.0 * (e[0] * e[1] + e[1] * e[2] + e[2] * e[0])


def _scan(a, op, direction):
    """Return the inclusive prefix (direction 1) or suffix (-1) op.accumulate along the last axis, by doubling."""
    a = a.copy()
    n, k = a.shape[-1], 1
    while k < n:
        if direction > 0:
            a[..., k:] = op(a[..., k:], a[..., :-k])
        else:
            a[..., :-k] = op(a[..., :-k], a[..., k:])
        k *= 2
    return a


def _axis_cost(seg, ax, bins):
    """Return (cost of the best plane per segment, its bin, each item's bin, the items' mask) on one axis, or None.

    ``seg`` = (sgi, c, clo, chi, loi, hii, Si): the inner segments' items' segment, centroids, their segments' centroid
    bounds, the items' boxes and the number of segments.
    """
    sgi, c, clo, chi, loi, hii, Si = seg
    ext = chi[:, ax] - clo[:, ax]
    ok = ext > 0
    if not ok.any():
        return None
    okp = ok[sgi]
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.minimum(((c[:, ax] - clo[sgi, ax]) / ext[sgi] * bins).astype(int), bins - 1)
    b = np.where(okp, b, 0)
    key = (sgi * bins + b)[okp]
    cnt = np.bincount(key, minlength=Si * bins).reshape(Si, bins)
    # per-bin bounds, component-major (3, Si, bins); prefix / suffix bounds by doubling (exact: min, max)
    blo_ = np.full((3, Si * bins), np.inf)
    bhi_ = np.full((3, Si * bins), -np.inf)
    for q in range(3):
        np.minimum.at(blo_[q], key, loi[okp, q])
        np.maximum.at(bhi_[q], key, hii[okp, q])
    blo_, bhi_ = blo_.reshape(3, Si, bins), bhi_.reshape(3, Si, bins)
    ll, lh = _scan(blo_, np.minimum, 1), _scan(bhi_, np.maximum, 1)
    rl, rh = _scan(blo_, np.minimum, -1), _scan(bhi_, np.maximum, -1)
    lc, rc = np.cumsum(cnt, 1), np.cumsum(cnt[:, ::-1], 1)[:, ::-1]
    cost = _area3(ll[..., :-1], lh[..., :-1]) * lc[:, :-1] + _area3(rl[..., 1:], rh[..., 1:]) * rc[:, 1:]
    cost = np.where((lc[:, :-1] > 0) & (rc[:, 1:] > 0), cost, np.inf)
    k = np.argmin(cost, 1)
    return cost[np.arange(Si), k], k, b, ok


def _best_split(seg, bins):
    """Return (has a plane per segment, each item goes left) over the three axes (the first axis on ties)."""
    sgi, Si = seg[0], seg[6]
    best_c = np.full(Si, np.inf)
    best_left = np.zeros(len(sgi), bool)
    has = np.zeros(Si, bool)
    for ax in range(3):
        r = _axis_cost(seg, ax, bins)
        if r is None:
            continue
        ck, k, b, ok = r
        upd = ok & (ck < best_c)
        best_c = np.where(upd, ck, best_c)
        has |= upd
        best_left = np.where(upd[sgi], b <= k[sgi], best_left)
    return has, best_left


def _split_level(level, items, boxes, leaf_size, bins):
    """Split one level's segments: return (items' new order, children segment starts and ends, l_id, r_id, n_new)."""
    seg_s, seg_e, S, size, off, sg_, gpos, ix = level
    lo, hi, cen = boxes
    inner = size > leaf_size
    k_in = np.flatnonzero(inner)
    keep = inner[sg_]
    Si = len(k_in)
    sgi = (np.cumsum(inner) - 1)[sg_[keep]]  # the inner segments' items: their inner index
    ixi = ix[keep]
    offi = np.r_[0, np.cumsum(size[k_in])[:-1]]
    c = cen[ixi]
    clo, chi = np.minimum.reduceat(c, offi, 0), np.maximum.reduceat(c, offi, 0)
    has, best_left = _best_split((sgi, c, clo, chi, lo[ixi], hi[ixi], Si), bins)
    hasf = np.zeros(S, bool)
    hasf[k_in] = has
    bl = np.zeros(len(ix), bool)
    bl[keep] = best_left
    pos = np.arange(len(ix)) - off[sg_]
    left = np.where(hasf[sg_], bl, pos < (size // 2)[sg_])  # no plane: the list's halves
    # a stable partition of every inner segment: its left items first (leaves keep their order)
    items[gpos] = ix[np.argsort(sg_ * 2 + (keep & ~left), kind="stable")]
    nl = np.bincount(sg_[keep & left], minlength=S)[k_in]
    mid = seg_s[k_in] + nl
    ns_ = np.empty(2 * len(k_in), np.int64)
    ne_ = np.empty(2 * len(k_in), np.int64)
    ns_[0::2], ne_[0::2] = seg_s[k_in], mid
    ns_[1::2], ne_[1::2] = mid, seg_e[k_in]
    return ns_, ne_, k_in


def _levels(lo, hi, bounds, leaf_size, bins):
    """Build every tree level by level; return (per level: starts, ends, lo, hi, left, right; the roots; items)."""
    n = len(lo)
    cen = 0.5 * (lo + hi)
    items = np.arange(n)
    N_s, N_e, N_lo, N_hi, N_l, N_r = [], [], [], [], [], []
    seg_s, seg_e = bounds[:-1].copy(), bounds[1:].copy()
    n_roots = n_nodes = len(seg_s)
    while len(seg_s):
        S = len(seg_s)
        size = seg_e - seg_s
        off = np.r_[0, np.cumsum(size)[:-1]]
        sg_ = np.repeat(np.arange(S), size)
        gpos = seg_s[sg_] + np.arange(len(sg_)) - off[sg_]  # the segments' positions in items, in order
        ix = items[gpos]
        blo, bhi = np.minimum.reduceat(lo[ix], off, 0), np.maximum.reduceat(hi[ix], off, 0)
        N_s.append(seg_s)
        N_e.append(seg_e)
        N_lo.append(blo)
        N_hi.append(bhi)
        l_id = np.full(S, -1, np.int64)
        r_id = np.full(S, -1, np.int64)
        if np.any(size > leaf_size):
            seg_s, seg_e, k_in = _split_level((seg_s, seg_e, S, size, off, sg_, gpos, ix), items, (lo, hi, cen),
                                              leaf_size, bins)  # fmt: skip
            l_id[k_in] = n_nodes + 2 * np.arange(len(k_in))
            r_id[k_in] = l_id[k_in] + 1
            n_nodes += 2 * len(k_in)
        else:
            seg_s, seg_e = np.zeros(0, np.int64), np.zeros(0, np.int64)
        N_l.append(l_id)
        N_r.append(r_id)
    return (N_s, N_e, N_lo, N_hi, N_l, N_r), n_roots, items


def _depth_first(N_l, L_, R_, n_roots):  # noqa: N803 -- per-level children, as _levels makes them
    """Return (pre, sub): each node's depth-first index and its subtree size."""
    m = len(L_)
    lev = np.cumsum([0] + [len(x) for x in N_l])  # each level's nodes: ids lev[d]:lev[d + 1]
    sub = np.ones(m, np.int64)  # subtree sizes, deepest level first
    for d in range(len(N_l) - 1, -1, -1):
        v = np.arange(lev[d], lev[d + 1])
        v = v[L_[v] >= 0]
        sub[v] = 1 + sub[L_[v]] + sub[R_[v]]
    pre = np.zeros(m, np.int64)  # depth-first index: left child next, right after its subtree
    pre[:n_roots] = np.r_[0, np.cumsum(sub[:n_roots])[:-1]]
    for d in range(len(N_l)):
        v = np.arange(lev[d], lev[d + 1])
        v = v[L_[v] >= 0]
        pre[L_[v]] = pre[v] + 1
        pre[R_[v]] = pre[v] + 1 + sub[L_[v]]
    return pre, sub


def build_many(lo, hi, bounds, leaf_size=LEAF_SIZE, bins=BINS):
    """Return the BVHs of the box ranges [bounds[i], bounds[i + 1]) built in one pass.

    One dict as ``build_bvh``'s, the trees' nodes concatenated in range order (each depth first; skip and start
    absolute; ``tree0``: each tree's first node; ``n_levels``: the deepest tree's depth), items a permutation within
    each range. Each tree is the recursive builder's exactly: every node of one depth (of every tree) is one segment
    of ``items``, binned, costed and partitioned in the same numpy calls (the cheapest of the bins - 1 planes per axis,
    the first axis on ties, a stable partition; all centroids coincident: the list's halves; a leaf at <= leaf_size).
    """
    lo, hi = np.asarray(lo), np.asarray(hi)
    bounds = np.asarray(bounds, np.int64)
    n = len(lo)
    if len(bounds) < 2 or np.any(np.diff(bounds) <= 0) or bounds[0] != 0 or bounds[-1] != n:
        raise ValueError("bvh.build_many: empty or unordered box ranges")
    (N_s, N_e, N_lo, N_hi, N_l, N_r), n_roots, items = _levels(lo, hi, bounds, leaf_size, bins)
    s_all, e_all = np.concatenate(N_s), np.concatenate(N_e)
    lo_all, hi_all = np.concatenate(N_lo), np.concatenate(N_hi)
    L_, R_ = np.concatenate(N_l), np.concatenate(N_r)
    m = len(s_all)
    pre, sub = _depth_first(N_l, L_, R_, n_roots)
    leaf = L_ < 0
    out = dict(lo=np.zeros((m, 3), np.float32), hi=np.zeros((m, 3), np.float32), skip=np.zeros(m, np.int32),
               start=np.zeros(m, np.int32), count=np.zeros(m, np.int32))  # fmt: skip
    out["lo"][pre], out["hi"][pre] = lo_all.astype(np.float32), hi_all.astype(np.float32)
    out["skip"][pre] = (pre + sub).astype(np.int32)
    out["start"][pre] = np.where(leaf, s_all, 0)
    out["count"][pre] = np.where(leaf, e_all - s_all, 0)
    out["items"] = items.astype(np.int32)
    out["tree0"] = pre[:n_roots].astype(np.int64)  # each tree's first node
    out["n_levels"] = len(N_l)  # the deepest tree's depth
    return out


def _surface_terms(h):
    return h[:, 0] * h[:, 1] + h[:, 1] * h[:, 2] + h[:, 2] * h[:, 0]


def frame_groups_all(R, hb, vm, km, min_group=MIN_GROUP, gain=GROUP_GAIN):  # noqa: N803 -- R: the Notation line
    """Return every volume's daughters grouped by rotation: (gkey, G, the groups' rotations (G, 3, 3), accepted).

    ``vm``, ``km``: each daughter's mother and itself (a mother's in its daughters' order); ``gkey`` per daughter: its
    group's key (a rotation group's index, or G + vm for its mother's world-frame group: sorting by (vm, gkey) gives
    the groups in order). Per mother: a rotation shared by at least min_group daughters (or by all) whose world boxes
    are at least gain times their own boxes (summed surface areas: what a BVH pays for) is a group of its own, in the
    order of the rotations' keys; the rest one world-frame group, last. Notation: R the volumes' rotations, G the
    number of rotation keys.
    """
    N = len(km)
    if not N:
        return np.zeros(0, np.int64), 0, np.zeros((0, 3, 3)), np.zeros(0, bool)
    rows = np.column_stack([vm.astype(np.float64), np.round(R[km].reshape(N, 9), ROT_DECIMALS)])
    _, inv, cnt = np.unique(rows, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    G = len(cnt)
    first = np.full(G, N, np.int64)
    np.minimum.at(first, inv, np.arange(N))
    nk = np.bincount(vm)
    Rg = R[km[first]]
    eye = np.eye(3)
    ident = np.all(np.abs(Rg - eye) <= IDENT_ATOL + IDENT_RTOL * eye, axis=(1, 2))  # np.allclose(Rg, I, atol=1e-9)
    cand = ((cnt >= min_group) | (cnt == nk[vm[first]])) & ~ident
    acc = np.zeros(G, bool)
    if cand.any():
        hw = np.einsum("nij,nj->ni", np.abs(R[km]), hb[km])
        tw, tb = _surface_terms(hw), _surface_terms(hb[km])
        aw, ab = np.zeros(G), np.zeros(G)
        one = cand & (cnt == 1)  # a one-term sum is the term
        aw[one], ab[one] = tw[first[one]], tb[first[one]]
        order = np.argsort(inv, kind="stable")
        st = np.r_[0, np.cumsum(cnt)[:-1]]
        for g in np.flatnonzero(cand & (cnt > 1)):  # np.sum's own order (pairwise beyond 8 terms)
            m = order[st[g] : st[g] + cnt[g]]
            aw[g], ab[g] = np.sum(tw[m]), np.sum(tb[m])
        acc = cand & ~(aw < gain * ab)
    gkey = np.where(acc[inv], inv, G + vm)
    return gkey, G, Rg, acc


def slab(lo, hi, o, inv):
    """Return the (entry, exit) parameters of the ray o + t / inv through the box [lo, hi] (device)."""
    ta, tb = (lo - o) * inv, (hi - o) * inv
    return jnp.max(jnp.minimum(ta, tb)), jnp.min(jnp.maximum(ta, tb))
