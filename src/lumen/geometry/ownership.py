"""Which volume a point belongs to where placements overlap or protrude: one owner per point, level by level.

Geant4 takes its placements as disjoint siblings inside their mothers; a detector description may still overlap them
(wire planes lying in their sibling active volume) or let a daughter stick out of its mother (a vessel's end caps out of
its cryostat). lumen gives every point one owner, by one rule, everywhere in the code: descending from the world, at
each level the point belongs to the first sibling containing it in the rule's order -- a solid proved inside a sibling
ahead of it, else the one placed first (a solid whose holes a lowering dropped, ``Scene.hole_fillers``, after the
siblings filling them, so the holes are theirs). So a volume holds only what lies inside its mother, and a region
overlapping siblings share is its owner's, with its daughters.

"Proved" has one meaning here (``Proofs``, ``relate_poses``): the solids' parameters show it, by interval bounds over
bisected boxes (``relation``), never by sampling; what is not proved is not assumed.

The rule needs no list of which pairs overlap: the navigator decides each point by containment at run time, and is
given, per volume, every surface that may bound its region (``resolve``):

- ``prio``: a total order of the units (lower first) that agrees with the rule wherever two of them may hold one point:
  the depth-first order of the tree with each mother's daughters in their ownership order;
- ``owners``: per unit the siblings of it, or of an ancestor, ahead of it in that order that may meet it (a candidate
  entered as a daughter);
- ``clips``: per unit the ancestors it is not proved to lie inside (a candidate whose face is an exit).

A pair is linked where the boxes meet deeper than the contact tolerance (``lumen.rules.navigation.contact``: a contact
is no overlap) and the solids are not proved apart; a box shrunk by that tolerance inside a solid, or a solid proved
inside, is no clip.

    own = resolve(units)            # Ownership: prio, owners, clips, the linked sibling pairs, the protrusions
    proofs = Proofs(units)          # proofs(inner, outer, "inside" | "apart"): per pair of units, proved or not

docs/dev/decisions.md, 23 ("A point has one owner, level by level"), has the rule's reasons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lumen.geometry import scene as sc
from lumen.geometry import solids as so
from lumen.rules import navigation as rules
from lumen.rules.tolerances import CONVEX_TOL, FLOAT64_TINY, PIECES, POSE_ULPS, SAME_POSE, SPLITS

PROOF_BATCH = 256  # pairs proved together (their pieces: at most PIECES each)
TILE_EXTENT = 2.0  # the side of a sibling sweep's tiles across its axis, in median box extents
MAX_TILES = 256  # ... and at most this many tiles along each of those axes
TILE_COPIES = 8  # ... coarser while a box is in more of them than this, on average


def _is_mesh(node):
    return isinstance(node, sc.Mesh)


def box_of(node):
    """Return a solid's box (centre, half) in its frame (m); None when unbounded."""
    if _is_mesh(node):
        p = node.tris.reshape(-1, 3)
        lo, hi = p.min(0), p.max(0)
    else:
        bb = so.node_bbox(node)
        if bb is None:
            return None
        lo, hi = np.asarray(bb[0], float), np.asarray(bb[1], float)
    return (lo + hi) / 2, (hi - lo) / 2


# ------------------------------------------------------------------------------------------------ boxes in solids
def _leaf_boxes(lf, c, Q, h):  # noqa: N803 -- Q the boxes' axes
    """Return per oriented box (centres c (m, 3), axes Q (m, 3, 3) as columns -- None: the solid's axes --, half h
    (m, 3), in the solid's frame) whether the leaf may hold a point of it and whether it surely holds all of it
    (first-order bounds of its plane and quadric functions over the box, the quadric's curvature bounding the rest; a
    torus: never sure)."""  # noqa: D205, D209
    m = len(c)
    if lf.kind != "q":
        return np.ones(m, bool), np.zeros(m, bool)
    may, sure = np.ones(m, bool), np.ones(m, bool)
    if lf.N.shape[0]:
        v = (c - lf.origin) @ lf.N.T - lf.H
        r = h @ np.abs(lf.N).T if Q is None else (np.abs(np.matmul(lf.N, Q)) @ h[:, :, None])[:, :, 0]
        may &= np.all(v - r <= 0.0, 1)
        sure &= np.all(v + r <= 0.0, 1)
    if lf.has_q:
        qo, bq, cq = so.quad_frame(lf)
        A = 0.5 * (lf.A + lf.A.T)
        ev = np.linalg.eigvalsh(A)
        y = c - lf.origin - qo
        yA = y @ A
        q = np.sum(yA * y, 1) + 2.0 * y @ bq + cq
        gr = 2.0 * (yA + bq)
        g = np.sum(np.abs(gr if Q is None else np.matmul(gr[:, None, :], Q)[:, 0, :]) * h, 1)
        hh = np.sum(h * h, 1)
        may &= q - g + min(float(ev.min()), 0.0) * hh <= 0.0
        sure &= q + g + max(float(ev.max()), 0.0) * hh <= 0.0
    return may, sure


def _expr_boxes(e, leaves):
    """Return (may, sure) per box of a CSG expression over its leaves' (may, sure)."""
    if isinstance(e, (int, np.integer)):
        return leaves[int(e)]
    (ma, sa), (mb, sb) = _expr_boxes(e[1], leaves), _expr_boxes(e[2], leaves)
    if e[0] == "union":
        return ma | mb, sa | sb
    if e[0] == "inter":
        return ma & mb, sa & sb
    return ma & ~sb, sa & ~mb


def boxes_inside(node, c, Q, h, may=False):  # noqa: N803 -- Q the boxes' axes
    """Return per oriented box (``_leaf_boxes``) whether the CSG solid surely holds all of it (``may``: whether it may
    hold a point of it); a mesh: never surely, always maybe."""  # noqa: D205, D209
    if _is_mesh(node):
        return np.full(len(c), bool(may))
    expr, leaves = so.flatten(node)
    return _expr_boxes(expr, [_leaf_boxes(lf, c, Q, h) for lf in leaves])[0 if may else 1]


_SIGNS = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])


def _corners(c, Q, h):  # noqa: N803 -- Q the boxes' axes
    """Return the eight corners (m, 8, 3) of oriented boxes (centres c, axes Q as columns -- None: the frame's --,
    half h)."""  # noqa: D205, D209
    if Q is None:
        return c[:, None, :] + _SIGNS[None] * h[:, None, :]
    return c[:, None, :] + np.matmul(_SIGNS, np.transpose(Q * h[:, None, :], (0, 2, 1)))


def flat(node):
    """Return (expr, leaves, per leaf whether it is convex) of a CSG solid, the form ``relation`` reads; None for a
    mesh. Convex: planes, and a quadric of non-negative curvature."""  # noqa: D205, D209
    if _is_mesh(node):
        return None
    expr, leaves = so.flatten(node)
    convex = [lf.kind == "q" and (not lf.has_q or float(np.linalg.eigvalsh(0.5 * (lf.A + lf.A.T)).min())
                                  > -CONVEX_TOL) for lf in leaves]  # fmt: skip
    return expr, leaves, convex


def relation(fl, c, Q, h):  # noqa: N803 -- Q the boxes' axes
    """Return per oriented box (centres c (m, 3), axes Q (m, 3, 3) as columns -- None: the solid's --, half h (m, 3), in
    the solid's frame) (may the solid hold a point of it, does it surely hold all of it); ``fl`` the solid's ``flat``.
    A leaf: the
    interval bounds of ``_leaf_boxes``, and a convex leaf holds a box exactly when it holds its eight corners; a union
    holds a box one side holds, an intersection one both hold, a difference one its first holds and its second
    misses."""  # noqa: D205, D209
    expr, leaves, convex = fl
    out = []
    for lf, cv in zip(leaves, convex, strict=True):
        may, sure = _leaf_boxes(lf, c, Q, h)
        if cv:
            x = _corners(c, Q, h)
            sure = sure | so._leaf_inside(lf, x.reshape(-1, 3)).reshape(len(c), 8).all(1)
        out.append((may, sure))
    return _expr_boxes(expr, out)


def _sub_chain(node):
    """Return (base, subtrahends) of a solid's top chain of subtractions ``((base - s1) - s2) - ...``, in order."""
    subs = []
    while isinstance(node, so.Op) and node.op == "sub":
        subs.append(node.b)
        node = node.a
    return node, subs[::-1]


def _regions(box, L, t):  # noqa: N803 -- L the rotations
    """Return per pose the box (lo, hi (m, 3)) holding a solid's box (centre, half) placed by ``x = L[k] y + t[k]``."""
    c, h = np.asarray(box[0], float), np.asarray(box[1], float)
    w = np.einsum("mij,kj->mki", L, so.corners(c - h, c + h)) + t[:, None, :]
    return w.min(1), w.max(1)


def relate_poses(fi, bi, fo, L, t, want):  # noqa: N803 -- L the rotations
    """Return per pose whether a solid is proved apart from (``want`` "apart") or inside ("inside") another.

    The solid: ``flat`` fi (None: a mesh, known only by its box bi = (centre, half)), placed in the other's frame by
    ``x = L[k] y + t[k]``; ``fo`` the other's ``flat`` (None: a mesh, nothing is proved).

    The proof: the inner box is bisected along its longest side, up to ``SPLITS`` times and ``PIECES`` pieces per
    pose, keeping the pieces undecided; a piece is decided when the inner solid holds none of it, or the outer none of
    it ("apart"), all of it ("inside") (``relation``); a piece the inner solid surely fills and the outer surely fills
    (misses) disproves it. A proof, never a sample: undecided is not proved. All poses at once (one array of pieces).
    """
    L, t = np.asarray(L, float).reshape(-1, 3, 3), np.asarray(t, float).reshape(-1, 3)
    m = len(L)
    res = np.zeros(m, bool)
    if fo is None or not m:
        return res
    live = np.ones(m, bool)
    task = np.arange(m)
    c = np.repeat(np.asarray(bi[0], float)[None], m, 0)
    h = np.repeat(np.asarray(bi[1], float)[None], m, 0)
    for _ in range(SPLITS + 1):
        Lk = L[task]
        may, sure = relation(fo, np.matmul(Lk, c[:, :, None])[:, :, 0] + t[task], Lk, h)
        ok, bad = (~may, sure) if want == "apart" else (sure, ~may)
        if fi is not None:
            imay, isure = relation(fi, c, None, h)
            live[task[bad & isure]] = False
            ok = ok | ~imay
        keep = ~ok & live[task]
        cnt = np.bincount(task[keep], minlength=m)
        res |= live & (cnt == 0)  # every piece decided
        live &= (cnt > 0) & (2 * cnt <= PIECES)
        keep &= live[task]
        c, h, task = c[keep], h[keep], task[keep]
        if not len(c):
            break
        ax = np.argmax(h, 1)
        dh = np.zeros_like(h)
        dh[np.arange(len(h)), ax] = h[np.arange(len(h)), ax] / 2
        c, h, task = np.concatenate([c - dh, c + dh]), np.concatenate([h - dh, h - dh]), np.r_[task, task]
    return res


class Proofs:
    """The containment proofs between placed units (``Units``): each solid flattened once, each (kinds, relative pose,
    want) proved once."""  # noqa: D205, D209

    def __init__(self, u):
        self.u, self.flats, self.memo, self.chains, self.nears = u, {}, {}, {}, {}

    def _flat(self, k):
        if k not in self.flats:
            self.flats[k] = flat(self.u.nodes[k])
        return self.flats[k]

    def _chain(self, k):
        """Return (base, subtrahends, their boxes' lo and hi (s, 3)) of solid k's top chain of subtractions; an
        unbounded subtrahend's box is all space."""  # noqa: D205, D209
        if k not in self.chains:
            node = self.u.nodes[k]
            base, subs = (node, []) if _is_mesh(node) else _sub_chain(node)
            bb = [so.node_bbox(b) for b in subs]
            lo = np.array([np.full(3, -np.inf) if b is None else b[0] for b in bb], float).reshape(-1, 3)
            hi = np.array([np.full(3, np.inf) if b is None else b[1] for b in bb], float).reshape(-1, 3)
            self.chains[k] = (base, subs, lo, hi)
        return self.chains[k]

    def _kept(self, k, lo, hi):
        """Return per box [lo, hi] (m, 3) of solid k's frame which subtrahends of its top chain meet it (m, s)."""
        _, _, slo, shi = self._chain(k)
        return np.all(slo[None] <= hi[:, None], 2) & np.all(lo[:, None] <= shi[None], 2)

    def _near(self, k, kept):
        """Return solid k's ``flat`` with only the subtrahends ``kept`` (a mask) of its top chain: the solid on the
        boxes they meet, a superset of it elsewhere."""  # noqa: D205, D209
        if kept.all():
            return self._flat(k)
        key = (k, kept.tobytes())
        if key not in self.nears:
            node, subs, _, _ = self._chain(k)
            for b, keep in zip(subs, kept, strict=True):
                if keep:
                    node = so.Op("sub", node, b)
            self.nears[key] = flat(node)
        return self.nears[key]

    def _prove(self, a, b, L, t, want):  # noqa: N803 -- L the rotations
        """Return per pose of solid a in solid b's frame (``x = L[k] y + t[k]``) whether a is proved apart from /
        inside b: each solid without the subtrahends of its top chain that miss the other's box at that pose, the
        poses keeping the same ones proved together, ``PROOF_BATCH`` at a time."""  # noqa: D205, D209
        got = np.zeros(len(L), bool)
        if not len(L):
            return got
        box_a, box_b = self.u.box[a], self.u.box[b]
        ko = self._kept(b, *_regions(box_a, L, t))
        if box_b is None:
            ki = np.ones((len(L), len(self._chain(a)[1])), bool)
        else:  # b's box in a's frame: x_a = L^T (x_b - t)
            ki = self._kept(a, *_regions(box_b, np.transpose(L, (0, 2, 1)), -np.einsum("mji,mj->mi", L, t)))
        rows = np.c_[ki, ko]
        grp = np.unique(rows, axis=0, return_inverse=True)[1].ravel() if rows.shape[1] else np.zeros(len(L), int)
        for g in range(int(grp.max()) + 1):
            idx = np.flatnonzero(grp == g)
            fi, fo = self._near(a, ki[idx[0]]), self._near(b, ko[idx[0]])
            for s0 in range(0, len(idx), PROOF_BATCH):
                js = idx[s0 : s0 + PROOF_BATCH]
                got[js] = relate_poses(fi, box_a, fo, L[js], t[js], want)
        return got

    def __call__(self, inner, outer, want):
        """Return per pair (inner[k], outer[k]) of units whether inner is proved apart from / inside outer."""
        u = self.u
        inner, outer = np.asarray(inner, np.int64).ravel(), np.asarray(outer, np.int64).ravel()
        out = np.zeros(len(inner), bool)
        if not len(inner):
            return out
        Ro = np.asarray(u.R, float)[outer]
        L = np.einsum("mji,mjk->mik", Ro, np.asarray(u.R, float)[inner])  # inner's frame in outer's: Ro^T Ri
        t = np.einsum("mji,mj->mi", Ro, np.asarray(u.t, float)[inner] - np.asarray(u.t, float)[outer])
        ki, ko = np.asarray(u.kind)[inner], np.asarray(u.kind)[outer]
        if want == "inside":  # one solid at one pose: coincident
            same = (ki == ko) & np.all(np.abs(L - np.eye(3)) <= SAME_POSE, (1, 2)) & np.all(np.abs(t) <= SAME_POSE, 1)
            out |= same
        # poses equal to SAME_POSE share one proof (the first's): the key is the pose on a SAME_POSE grid
        key = np.c_[ki, ko, np.rint(L.reshape(-1, 9) / SAME_POSE), np.rint(t / SAME_POSE)]
        uk, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.ravel()
        got = np.zeros(len(uk), bool)
        todo = []
        for j in range(len(first)):
            mk = (want, uk[j].tobytes())
            if mk in self.memo:
                got[j] = self.memo[mk]
            else:
                todo.append(j)
        todo_a = np.asarray(todo, np.int64)
        pair = np.c_[ki[first[todo_a]], ko[first[todo_a]]] if len(todo_a) else np.zeros((0, 2), np.int64)
        for a, b in {(int(x), int(y)) for x, y in pair}:
            sel = todo_a[(pair[:, 0] == a) & (pair[:, 1] == b)]
            if u.box[a] is None:  # an unbounded solid: nothing is proved
                sel = sel[:0]
            got[sel] = self._prove(a, b, L[first[sel]], t[first[sel]], want)
            for j in sel:
                self.memo[(want, uk[j].tobytes())] = bool(got[j])
        return out | got[inv]

    def apart(self, a, b):
        """Return per pair whether units a[k] and b[k] are proved apart (either way round)."""
        return self(a, b, "apart") | self(b, a, "apart")


def sibling_order(place, before, forced):
    """Return per unit its place in the rule's order among its siblings (lower first; owners ahead of what they own).

    The placement order (``place``, per unit), each unit moved ahead of the siblings it must precede: ``before`` (m, 2)
    pairs (first, second) -- a solid proved inside a sibling -- and ``forced`` (holed, filler) pairs -- a filler ahead
    of the solid whose hole it fills. A topological order, the lowest index first among the units free to go; a
    cycle (the proofs are pairwise) is broken at its first placed.
    """
    import heapq

    place = np.asarray(place, np.int64)
    n = len(place)
    edges = [(int(a), int(b)) for a, b in np.asarray(before, np.int64).reshape(-1, 2)]
    edges += [(int(f), int(h_)) for h_, f in np.asarray(forced, np.int64).reshape(-1, 2)]
    nxt = [[] for _ in range(n)]
    deg = np.zeros(n, np.int64)
    for a, b in set(edges):
        nxt[a].append(b)
        deg[b] += 1
    heap = [(int(place[i]), i) for i in range(n) if deg[i] == 0]
    heapq.heapify(heap)
    pos = np.full(n, -1, np.int64)
    k = 0
    while k < n:
        if not heap:  # a cycle: its first placed goes first
            rest = np.flatnonzero(pos < 0)
            i = int(rest[np.argmin(place[rest])])
            deg[i] = 0
        else:
            i = heapq.heappop(heap)[1]
            if pos[i] >= 0:
                continue
        pos[i] = k
        k += 1
        for j in nxt[i]:
            deg[j] -= 1
            if deg[j] == 0 and pos[j] < 0:
                heapq.heappush(heap, (int(place[j]), j))
    return pos


def owners_of(proofs, pairs, forced):
    """Return (per unit its place in the rule's order among its siblings, the pairs (non-owner, owner)) of linked
    sibling pairs (``pairs`` (m, 2), the lowered holes' ``forced`` pairs (holed, filler) apart): a solid proved inside
    its sibling owns, else the one placed first (``sibling_order``)."""  # noqa: D205, D209
    P = np.asarray(pairs, np.int64).reshape(-1, 2)
    ab, ba = proofs(P[:, 0], P[:, 1], "inside"), proofs(P[:, 1], P[:, 0], "inside")
    before = np.concatenate([P[ab & ~ba], P[ba & ~ab][:, ::-1]])
    pos = sibling_order(proofs.u.place, before, forced)
    own = np.where((pos[P[:, 0]] < pos[P[:, 1]])[:, None], P[:, ::-1], P)  # (non-owner, owner)
    return pos, own


# ------------------------------------------------------------------------------------------------ the units
@dataclass(frozen=True)
class Units:
    """The placed solids ownership is decided on: volumes in depth-first order (a mother before its daughters), then
    meshes (no daughters).

    ``parent`` per unit its mother unit (-1: none); ``kind`` its solid (an index into ``nodes``: CSG trees or meshes);
    ``box`` per kind (centre, half) in its frame (m); ``R``, ``t`` the world poses (``x_world = R x + t``); ``place``
    per unit its place in the placement order (the scene's walk: siblings as placed); ``forced`` (m, 2) pairs
    (non-owner, owner) decided by construction (a lowered hole and the sibling filling it).
    """  # noqa: D205

    parent: np.ndarray
    kind: np.ndarray
    nodes: list
    box: tuple
    R: np.ndarray
    t: np.ndarray
    place: np.ndarray
    forced: np.ndarray
    prec: object = rules.FLOAT32  # the navigator's precision: its contact tolerance


@dataclass(frozen=True)
class Ownership:
    """Who may own what (``resolve``).

    ``prio`` per unit its place in the rule's order (lower first); ``owners`` per unit the units ahead of it, of its
    own or an ancestor's siblings, whose box meets its own; ``clips`` per unit the ancestors it is not proved inside;
    ``pairs`` (m, 2): the sibling pairs linked (non-owner, owner), ``forced`` marking the lowered holes; ``protrude``
    (m, 2): (unit, its mother) where it is not proved inside; ``thin`` (m, 2): the sibling pairs (i < j) that may
    share a point but not deeper than the contact tolerance (``_meet`` at the poses' rounding, not linked) --
    touching for the navigator, maybe overlapping by less than it.
    """

    prio: np.ndarray
    owners: tuple
    clips: tuple
    pairs: np.ndarray
    forced: np.ndarray
    protrude: np.ndarray
    thin: np.ndarray


def _world_boxes(u):
    """Return per unit its box (centre, half) in its frame and its world box (centre, half)."""
    c = np.stack([np.asarray(u.box[k][0], float) for k in u.kind]) if len(u.kind) else np.zeros((0, 3))
    h = np.stack([np.asarray(u.box[k][1], float) for k in u.kind]) if len(u.kind) else np.zeros((0, 3))
    cw = np.einsum("nij,nj->ni", u.R, c) + u.t
    hw = np.einsum("nij,nj->ni", np.abs(u.R), h)
    return c, h, cw, hw


def _sweep(lo, hi):
    """Return the index pairs of boxes [lo, hi] overlapping on every axis (open), each once and in the sweep's order.

    A pair is (i, j) with i before j in the order of their low faces along the sweep axis (the axis of least relative
    extent; ties by index), the pairs in that order. The sweep runs within tiles across that axis: a box is in each
    tile it covers, and a pair is kept in the tile of its common box's low corner. A box long across the tiles meets
    the boxes of its tiles only, not every box beside it along the axis (a fiber through a block of cubes). The tiles
    coarsen while the boxes are in more than ``TILE_COPIES`` tiles each on average (plates across a block).
    """
    n = len(lo)
    empty = np.zeros((0, 2), np.int64)
    if n < 2:
        return empty
    ext = (hi - lo).mean(0) / np.maximum(hi.max(0) - lo.min(0), FLOAT64_TINY)
    ax = int(np.argmin(ext))
    o = np.argsort(lo[:, ax], kind="stable")
    rank = np.empty(n, np.int64)
    rank[o] = np.arange(n)
    u, w = (k for k in range(3) if k != ax)
    origin = lo.min(0)
    size = np.maximum(TILE_EXTENT * np.median(hi - lo, axis=0), (hi.max(0) - origin) / MAX_TILES)
    size = np.where(size > 0, size, 1.0)
    while True:
        t0 = np.floor((lo - origin) / size).astype(np.int64)
        t1 = np.floor((hi - origin) / size).astype(np.int64)
        cu, cw = t1[:, u] - t0[:, u] + 1, t1[:, w] - t0[:, w] + 1
        cnt = cu * cw
        if cnt.sum() <= TILE_COPIES * n:
            break
        size = 2 * size
    nw = int(t1[:, w].max()) + 1
    box = np.repeat(np.arange(n), cnt)
    k = np.arange(len(box)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    tile = (t0[box, u] + k % cu[box]) * nw + t0[box, w] + k // cu[box]
    s = np.lexsort((rank[box], tile))
    tile, box = tile[s], box[s]
    key = tile * (n + 1) + rank[box]
    below = np.searchsorted(lo[o, ax], hi[box, ax], side="left")  # the ranks whose low face is below this high face
    end = np.searchsorted(key, tile * (n + 1) + below, side="left")
    c = np.maximum(end - np.arange(len(box)) - 1, 0)
    if not c.sum():
        return empty
    ii = np.repeat(np.arange(len(box)), c)
    jj = ii + 1 + (np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c))
    a, b, tq = box[ii], box[jj], tile[ii]
    keep = np.all((lo[a] < hi[b]) & (lo[b] < hi[a]), 1)
    a, b, tq = a[keep], b[keep], tq[keep]
    corner = np.floor((np.maximum(lo[a], lo[b]) - origin) / size).astype(np.int64)
    home = corner[:, u] * nw + corner[:, w] == tq
    a, b = a[home], b[home]
    order = np.lexsort((rank[b], rank[a]))
    return np.stack([a[order], b[order]], 1)


def _obb_depth(c1, h1, R1, c2, h2, R2):  # noqa: N803 -- R the boxes' rotations
    """Return the smallest overlap (m) of oriented box pairs over the 15 separating axes (< 0: separated)."""
    axes = [R1[:, :, k] for k in range(3)] + [R2[:, :, k] for k in range(3)]
    for i in range(3):
        for j in range(3):
            a = np.cross(R1[:, :, i], R2[:, :, j])
            nrm = np.linalg.norm(a, axis=1, keepdims=True)
            axes.append(np.divide(a, nrm, out=np.zeros_like(a), where=nrm > 0))
    d = c2 - c1
    best = np.full(len(c1), np.inf)
    for a in axes:
        r1 = np.sum(np.abs(np.einsum("mij,mi->mj", R1, a)) * h1, 1)
        r2 = np.sum(np.abs(np.einsum("mij,mi->mj", R2, a)) * h2, 1)
        best = np.where(np.linalg.norm(a, axis=1) > 0, np.minimum(best, r1 + r2 - np.abs(np.sum(d * a, 1))), best)
    return best


def _tolerance(cw, hw, zero=False, prec=rules.FLOAT32):
    """Return the contact tolerance (m) of boxes (world centre, half) at their largest coordinate (precision ``prec``).

    ``zero``: only the float64 rounding of composed poses there, ``POSE_ULPS`` ulps (a face shared exactly is still a
    contact).
    """
    scale = np.max(np.abs(cw) + hw, axis=-1)
    return POSE_ULPS * np.finfo(float).eps * scale if zero else rules.contact(scale, prec)


def _meet(boxes, a, b, u=None, zero=False):
    """Return per pair (a[k], b[k]) whether they may share a point deeper than the contact tolerance: their oriented
    boxes overlap that deep, and (``u`` given) each solid may hold a point of the common box (b's box in a's frame
    met with a's, shrunk by the tolerance)."""  # noqa: D205, D209
    c, h, cw, hw, R = boxes
    prec = rules.FLOAT32 if u is None else u.prec
    tol = np.maximum(_tolerance(cw[a], hw[a], zero, prec), _tolerance(cw[b], hw[b], zero, prec))
    ok = _obb_depth(cw[a], h[a], R[a], cw[b], h[b], R[b]) > tol
    if u is None or not ok.any():
        return ok
    k = np.flatnonzero(ok)
    a, b, tol = a[k], b[k], tol[k]
    Rab = np.einsum("mji,mjk->mik", R[a], R[b])  # b's frame in a's
    cb = np.einsum("mji,mj->mi", R[a], cw[b] - cw[a]) + c[a]  # b's box centre in a's frame
    hb = np.einsum("mij,mj->mi", np.abs(Rab), h[b])
    lo, hi = np.maximum(c[a] - h[a], cb - hb) + tol[:, None], np.minimum(c[a] + h[a], cb + hb) - tol[:, None]
    may = np.all(hi > lo, 1)
    cc, hc = (lo + hi) / 2, np.maximum(hi - lo, 0.0) / 2
    eye = np.broadcast_to(np.eye(3), (len(a), 3, 3))
    for x, cx, Qx in ((a, cc, eye), (b, None, None)):
        if cx is None:  # the common box in b's frame: x_b = Rab^T (x_a - c_a) + c_b's own centre offset
            cx = np.einsum("mji,mj->mi", Rab, cc - cb) + c[b]
            Qx = np.transpose(Rab, (0, 2, 1))
        for kd in np.unique(u.kind[x]):
            sel = np.flatnonzero(may & (u.kind[x] == kd))
            if len(sel):
                may[sel] = boxes_inside(u.nodes[int(kd)], cx[sel], Qx[sel], hc[sel], may=True)
    ok[k] = may
    return ok


def _sibling_pairs(u, boxes, zero=False):
    """Return the sibling pairs (i < j) whose oriented boxes overlap deeper than the contact tolerance (``zero``: any).

    Per mother, its daughters' boxes are swept in the frame most of them share (the wires of one plane), then tested
    as oriented boxes.
    """
    c, h, _, _, _ = boxes
    par = np.asarray(u.parent)
    out = []
    order = np.argsort(par, kind="stable")
    bounds = np.flatnonzero(np.r_[True, par[order][1:] != par[order][:-1], True])
    for g0, g1 in zip(bounds[:-1], bounds[1:], strict=True):
        s = order[g0:g1]
        if len(s) < 2:
            continue
        _, inv, cnt = np.unique(u.R[s].reshape(len(s), 9), axis=0, return_inverse=True, return_counts=True)
        Rm = u.R[s][np.flatnonzero(inv.ravel() == np.argmax(cnt))[0]]
        M = np.einsum("ji,njk->nik", Rm, u.R[s])
        cg = np.einsum("nij,nj->ni", M, c[s]) + u.t[s] @ Rm
        hg = np.einsum("nij,nj->ni", np.abs(M), h[s])
        p = _sweep(cg - hg, cg + hg)
        if len(p):
            out.append(s[p])
    P = np.sort(np.concatenate(out), 1) if out else np.zeros((0, 2), np.int64)
    return P[_meet(boxes, P[:, 0], P[:, 1], u, zero)] if len(P) else P


def hole_kinds(scene, defs):
    """Return per definition name of ``defs`` (whether it is holed, whether it fills a hole): ``Definition.holed``, and
    its logical volume among ``Scene.hole_fillers``."""  # noqa: D205, D209
    fill = set(scene.hole_fillers)
    d_ = [scene.definitions[k] for k in defs]
    holed = np.array([x.holed for x in d_], bool)
    flag = np.array([(x.origin or k) in fill for k, x in zip(defs, d_, strict=True)], bool)
    return holed, flag


def hole_pairs(holed, flag, parent, lo, hi):
    """Return the (holed solid, filler) unit pairs of the lowered holes: per ``holed`` unit, its siblings placed after
    it that fill a hole (``flag``, ``hole_kinds``) whose world box (lo, hi) meets its own."""  # noqa: D205, D209
    n = len(holed)
    holed, flag = np.asarray(holed, bool), np.asarray(flag, bool)
    par = np.asarray(parent)
    out = []
    for u in np.flatnonzero(holed):
        sib = np.flatnonzero((par == par[u]) & flag & (np.arange(n) > u))
        meet = sib[np.all((lo[sib] <= hi[u]) & (hi[sib] >= lo[u]), axis=1)]
        out.extend((int(u), int(i)) for i in meet)
    return np.asarray(out, np.int64).reshape(-1, 2)


def _preorder(parent, pos):
    """Return per unit its index in the depth-first order of the tree, each mother's daughters by ``pos``."""
    n = len(parent)
    ch = [[] for _ in range(n + 1)]
    for i in np.argsort(pos, kind="stable"):
        ch[int(parent[i]) if parent[i] >= 0 else n].append(int(i))
    out = np.zeros(n, np.int64)
    stack, k = list(reversed(ch[n])), 0
    while stack:
        i = stack.pop()
        out[i] = k
        k += 1
        stack.extend(reversed(ch[i]))
    return out


def _below(u):
    """Return per unit the units below it (volumes: a contiguous depth-first range; meshes: by their mother)."""
    n = len(u.parent)
    ch = [[] for _ in range(n)]
    for i, p in enumerate(u.parent):
        if p >= 0:
            ch[int(p)].append(i)
    out = [None] * n
    for i in range(n - 1, -1, -1):
        out[i] = np.asarray(sorted(ch[i] + [x for c in ch[i] for x in out[c]]), np.int64)
    return out


def _clips(u, boxes, proofs):
    """Return per unit the ancestors it is not proved inside: its mother and those its mother is not proved inside,
    each tested on the unit's box shrunk by the contact tolerance, then by the containment proof (``Proofs``)."""  # noqa: D205, D209
    c, h, cw, hw, R = boxes
    n = len(u.parent)
    tol = _tolerance(cw, hw, prec=u.prec)
    clips = [()] * n
    depth = np.zeros(n, np.int64)
    for i in range(n):  # volumes before their daughters; meshes last
        depth[i] = depth[u.parent[i]] + 1 if u.parent[i] >= 0 else 0
    for lv in range(1, int(depth.max(initial=0)) + 1):
        pv = [(int(v), int(m)) for v in np.flatnonzero(depth == lv)
              for m in (int(u.parent[v]), *clips[int(u.parent[v])])]  # fmt: skip
        if not pv:
            continue
        v_, m_ = np.array([x[0] for x in pv]), np.array([x[1] for x in pv])
        inside = np.zeros(len(pv), bool)
        for k in np.unique(u.kind[m_]):
            sel = np.flatnonzero(u.kind[m_] == k)
            Rm = np.transpose(R[m_[sel]], (0, 2, 1))
            cm = np.einsum("mij,mj->mi", Rm, np.einsum("mij,mj->mi", R[v_[sel]], c[v_[sel]]) + u.t[v_[sel]]
                           - u.t[m_[sel]])  # fmt: skip
            inside[sel] = boxes_inside(u.nodes[int(k)], cm, Rm @ R[v_[sel]], np.maximum(h[v_[sel]] - tol[v_[sel], None],
                                                                                         0.0))  # fmt: skip
        rest = np.flatnonzero(~inside)
        inside[rest] = proofs(v_[rest], m_[rest], "inside")
        out = {}
        for (v, m), ok in zip(pv, inside, strict=True):
            if not ok:
                out.setdefault(v, []).append(m)
        for v, ms in out.items():
            clips[v] = tuple(sorted(ms))
    return tuple(clips)


def _linked(u, boxes, proofs):
    """Return (the linked sibling pairs (i < j), the pairs deeper than the contact tolerance): boxes meeting deeper
    than the tolerance (``_sibling_pairs``), the lowered holes' ``forced`` pairs apart, the pairs proved apart out."""  # noqa: D205, D209
    forced = np.asarray(u.forced, np.int64).reshape(-1, 2)
    P = _sibling_pairs(u, boxes)
    fset = {tuple(sorted(map(int, p))) for p in forced}
    P = np.array([p for p in P if tuple(map(int, p)) not in fset], np.int64).reshape(-1, 2)
    deep = {tuple(map(int, p)) for p in P} | fset  # deeper than the tolerance: linked, or proved apart
    return P[~proofs.apart(P[:, 0], P[:, 1])], deep  # a link proved apart shares no point


def _boxes(u):
    c, h, cw, hw = _world_boxes(u)
    return c, h, cw, hw, np.asarray(u.R, float)


def sibling_rank(u):
    """Return per unit its place in the rule's order among its siblings (lower first), as ``resolve`` orders them:
    the proofs run on the linked pairs only."""  # noqa: D205, D209
    proofs = Proofs(u)
    return owners_of(proofs, _linked(u, _boxes(u), proofs)[0], np.asarray(u.forced, np.int64).reshape(-1, 2))[0]


def resolve(u):
    """Return the Ownership of the units (``Ownership``; module docstring)."""
    n = len(u.parent)
    boxes = _boxes(u)
    proofs = Proofs(u)
    forced = np.asarray(u.forced, np.int64).reshape(-1, 2)
    P, deep = _linked(u, boxes, proofs)
    T = np.array([p for p in _sibling_pairs(u, boxes, zero=True) if tuple(map(int, p)) not in deep],
                 np.int64).reshape(-1, 2)  # fmt: skip
    pos, own = owners_of(proofs, P, forced)
    pairs = np.concatenate([own, forced])
    prio = _preorder(np.asarray(u.parent), pos)
    below = _below(u)
    owners = [set() for _ in range(n)]
    for a, b in pairs:
        owners[int(a)].add(int(b))
    if len(pairs):  # below a non-owner: the units its owner may meet
        d = np.concatenate([below[int(a)] for a, _ in pairs])
        w = np.repeat(pairs[:, 1], [len(below[int(a)]) for a, _ in pairs])
        meet = _meet(boxes, d, w, u) if len(d) else np.zeros(0, bool)
        k = np.flatnonzero(meet)
        meet[k] = ~proofs.apart(d[k], w[k])
        for k_, b in zip(d[meet], w[meet], strict=True):
            owners[int(k_)].add(int(b))
    clips = _clips(u, boxes, proofs)
    protrude = [(v, int(u.parent[v])) for v in range(n) if u.parent[v] >= 0 and int(u.parent[v]) in clips[v]]
    return Ownership(prio=prio, owners=tuple(tuple(sorted(x)) for x in owners), clips=clips, pairs=pairs,
                     forced=np.r_[np.zeros(len(own), bool), np.ones(len(forced), bool)],
                     protrude=np.asarray(protrude, np.int64).reshape(-1, 2), thin=T)  # fmt: skip
