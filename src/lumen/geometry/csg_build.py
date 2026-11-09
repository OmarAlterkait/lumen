"""The CSG tables (host): every solid of a scene as structure-of-arrays leaf tables and a postfix boolean program.

Every kind (one distinct solid) runs the same device instructions on different data (``lumen.geometry.csg``):

- leaves: scalar tables (type, origin, A, b, c, planes, torus R / r / inverse frame), gathered by leaf id;
- a kind: its leaf range and its boolean as a postfix program (``op`` 0 pushes leaf ``arg``; 1 union, 2 inter,
  3 sub), the operand stack held in the bits of one int32 (depth at most ``MAX_DEPTH``);
- each leaf's box in the kind frame (outside it the leaf is proved false): a leaf whose box the ray's window misses is
  skipped.

    tables = csg_tables([solid, ...])          # CsgTables: T (leaves), K (kinds), G (programs), kind boxes

Notes
-----
Ported from the baseline's lumen.geometry.evaluator (its host part), itself from LUCiD's lucid.csg.uniform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lumen.build import solids

INF = 1e30  # "no crossing"; compared, never squared
OPS = {"union": 1, "inter": 2, "sub": 3}
MAX_DEPTH = 31  # the operand stack lives in one int32's bits
BOX_GRID = 8  # leaf boxes: cells per axis of the grid over the kind's box
BOX_LEVELS = 2  # ... refined this many times on the leaf's own box
BOX_PAD_REL = 1e-5  # ... padded by this times the kind box's extent
BOX_MIN_LEAVES = 3  # kinds with fewer leaves get no leaf boxes (every leaf is evaluated)
TINY_HALF = 1e-12  # m: a box half-size floor


@dataclass(frozen=True)
class CsgTables:
    """The CSG tables of a scene's kinds.

    ``T`` leaf tables (one entry per leaf, plus one empty leaf past the end; planes ``n0..n2``, ``h`` per plane),
    ``K`` per kind (``l0``, ``nl``: its leaf range; ``p0``, ``pl``: its program range), ``G`` the programs (``op``,
    ``arg``); ``kind_box`` per kind (centre, half) in its frame; ``maxnl`` the largest leaf count; ``empty_leaf`` the
    index every finished leaf loop reads; ``has_torus`` whether the torus kernel is needed.
    """

    T: dict
    K: dict
    G: dict
    kind_box: tuple
    n_leaves: tuple
    n_kinds: int
    maxnl: int
    empty_leaf: int
    has_torus: bool


# ------------------------------------------------------------------------------------------------- leaf boxes
def _quad_cells(lf, lo, hi, grid, n):
    """Refine a quadric leaf's box (lo, hi) to the cells of an n^3 grid where it may be true; empty: (inf, -inf)."""
    qo, bq, cq = solids.quad_frame(lf)
    lmin = float(np.min(np.linalg.eigvalsh(0.5 * (lf.A + lf.A.T)))) if lf.has_q else 0.0
    for _ in range(BOX_LEVELS):
        if lo[0] > hi[0]:
            break
        ck, hk = 0.5 * (lo + hi), np.maximum(0.5 * (hi - lo), TINY_HALF)
        cu = ck + grid * hk
        hc = hk / n
        ok = np.ones(len(cu), bool)
        if lf.N.shape[0]:
            v = (cu - lf.origin) @ lf.N.T - lf.H - np.abs(lf.N) @ hc  # min over the cell of n.x - h
            ok &= np.all(v <= 0.0, 1)
        if lf.has_q:
            y = cu - lf.origin - qo
            q = np.einsum("ni,ij,nj->n", y, lf.A, y) + 2.0 * y @ bq + cq
            gq = 2.0 * (y @ lf.A.T + bq)
            ok &= q - np.abs(gq) @ hc + min(lmin, 0.0) * float(hc @ hc) <= 0.0
        if not ok.any():
            return np.inf, -np.inf
        lo, hi = (cu[ok] - hc).min(0), (cu[ok] + hc).max(0)
    return lo, hi


def leaf_boxes(leaves, c, h, n=BOX_GRID):
    """Return per leaf its box (lo, hi), each (L, 3), in the kind frame (centre c, half h of the kind's box).

    The AABB of the cells of an n^3 grid over the kind's box where the leaf is possibly true (first-order interval
    bounds of its plane and quadric functions; a torus by its canonical box), refined on the leaf's own box, padded by
    ``BOX_PAD_REL`` times the extent. Outside the kind's box the solid is false whatever its leaves, so a leaf box may
    stop there; an empty box has lo > hi.
    """
    g = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    c, h = np.asarray(c, float), np.maximum(np.asarray(h, float), TINY_HALF)
    nl = len(leaves)
    lo, hi = np.tile(c - h, (nl, 1)), np.tile(c + h, (nl, 1))
    for k, lf in enumerate(leaves):
        if lf.kind != "q":  # torus: its canonical box through lin
            ext = np.array([lf.R + lf.r, lf.R + lf.r, lf.r])
            cs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * ext
            w = cs @ lf.lin.T + lf.origin
            lo[k], hi[k] = np.maximum(lo[k], w.min(0)), np.minimum(hi[k], w.max(0))
            continue
        lo[k], hi[k] = _quad_cells(lf, lo[k], hi[k], grid, n)
    pad = BOX_PAD_REL * (np.linalg.norm(c) + np.linalg.norm(h))
    ok = lo[:, 0] <= hi[:, 0]
    lo[ok] -= pad
    hi[ok] += pad
    return lo, hi


# ------------------------------------------------------------------------------------------------- the tables
def _depth(e):
    if isinstance(e, (int, np.integer)):
        return 1
    return max(_depth(e[1]), _depth(e[2]) + 1)


def _emit(e, prog):
    """Append the postfix program of a flattened expression to ``prog``: (op, arg) pairs."""
    if isinstance(e, (int, np.integer)):
        prog.append((0, int(e)))
    else:
        _emit(e[1], prog)
        _emit(e[2], prog)
        prog.append((OPS[e[0]], 0))


def _leaf_row(lf, blo, bhi, planes):
    """Return a leaf's table row; its planes are appended to ``planes``."""
    row = dict(typ=0, o=np.asarray(lf.origin, float), A=np.zeros((3, 3)), b=np.zeros(3), c=0.0, sa=1.0,
               p0=len(planes), np_=0, R=0.0, r=0.0, minv=np.eye(3), blo=blo, bhi=bhi)  # fmt: skip
    row["q"] = row["o"]
    if lf.kind == "q":
        qo, bq, cq = solids.quad_frame(lf)
        row.update(typ=1 if lf.has_q else 0, A=lf.A, b=bq, c=cq, q=row["o"] + qo,
                   sa=max(float(np.max(np.abs(lf.A))), 1e-30), np_=lf.N.shape[0])  # fmt: skip
        planes += [(lf.N[k], lf.H[k]) for k in range(lf.N.shape[0])]
    else:
        row.update(typ=2, R=lf.R, r=lf.r, minv=np.linalg.inv(lf.lin))
    return row


def _kind(node, rows, planes, prog):
    """Flatten one solid into the tables; return (its box centre and half, its leaf count, its K row)."""
    expr, leaves = solids.flatten(node)
    leaves = solids.common_origin_and_shared_planes(leaves)
    bb = solids.node_bbox(node)
    if bb is None or not (np.all(np.isfinite(bb[0])) and np.all(np.isfinite(bb[1]))):
        raise ValueError("csg tables: an unbounded solid (no finite box)")
    c, h = 0.5 * (np.asarray(bb[0]) + np.asarray(bb[1])), 0.5 * (np.asarray(bb[1]) - np.asarray(bb[0]))
    if len(leaves) >= BOX_MIN_LEAVES:
        blo, bhi = leaf_boxes(leaves, c, h)
    else:
        blo, bhi = np.full((len(leaves), 3), -INF), np.full((len(leaves), 3), INF)
    l0 = len(rows)
    for li, lf in enumerate(leaves):
        rows.append(_leaf_row(lf, blo[li], bhi[li], planes))
    p0 = len(prog)
    _emit(expr, prog)
    if _depth(expr) > MAX_DEPTH:
        raise ValueError(f"csg tables: a boolean deeper than {MAX_DEPTH} operands")
    return (c, h), len(leaves), (l0, len(leaves), p0, len(prog) - p0)


def _f32(v):
    return np.asarray(np.asarray(v, np.float64), np.float32)


def _i32(v):
    return np.asarray(np.asarray(v, np.int64), np.int32)


def _leaf_table(rows, planes):
    t = dict(typ=_i32([r["typ"] for r in rows]), sa=_f32([r["sa"] for r in rows]), c=_f32([r["c"] for r in rows]),
             p0=_i32([r["p0"] for r in rows]), np_=_i32([r["np_"] for r in rows]), R=_f32([r["R"] for r in rows]),
             r=_f32([r["r"] for r in rows]))  # fmt: skip
    for a in range(3):
        t[f"o{a}"] = _f32([r["o"][a] for r in rows])
        t[f"b{a}"] = _f32([r["b"][a] for r in rows])
        t[f"q{a}"] = _f32([r["q"][a] for r in rows])
        t[f"bl{a}"] = _f32([np.clip(r["blo"][a], -INF, INF) for r in rows])
        t[f"bh{a}"] = _f32([np.clip(r["bhi"][a], -INF, INF) for r in rows])
        for b in range(3):
            t[f"A{a}{b}"] = _f32([r["A"][a, b] for r in rows])
            t[f"m{a}{b}"] = _f32([r["minv"][a, b] for r in rows])
    planes = planes or [(np.zeros(3), 0.0)]
    for a in range(3):
        t[f"n{a}"] = _f32([p[0][a] for p in planes])
    t["h"] = _f32([p[1] for p in planes])
    return t


def csg_tables(kinds) -> CsgTables:
    """Return the CSG tables of the solids ``kinds`` (one ``lumen.build.solids`` node per kind, its frame, m)."""
    rows, planes, prog, krows, boxes, nls = [], [], [], [], [], []
    for node in kinds:
        box, nl, krow = _kind(node, rows, planes, prog)
        boxes.append(box)
        nls.append(nl)
        krows.append(krow)
    has_torus = any(r["typ"] == 2 for r in rows)
    empty_leaf = len(rows)  # past-the-end index of every leaf loop (a finished lane still runs the body there)
    empty = dict(typ=0, o=np.zeros(3), A=np.zeros((3, 3)), b=np.zeros(3), c=0.0, sa=1.0, p0=len(planes), np_=0,
                 R=0.0, r=0.0, minv=np.eye(3), q=np.zeros(3), blo=np.full(3, INF), bhi=np.full(3, -INF))  # fmt: skip
    rows.append(empty)
    kcols = {k: [x[j] for x in krows] for j, k in enumerate(("l0", "nl", "p0", "pl"))}
    maxnl = int(max(kcols["nl"])) if kcols["nl"] else 0
    prog = prog or [(0, 0)]
    return CsgTables(
        T=_leaf_table(rows, planes),
        K={k: _i32(v) for k, v in kcols.items()},
        G=dict(op=_i32([g[0] for g in prog]), arg=_i32([g[1] for g in prog])),
        kind_box=tuple(boxes),
        n_leaves=tuple(nls),
        n_kinds=len(krows),
        maxnl=maxnl,
        empty_leaf=empty_leaf,
        has_torus=has_torus,
    )
