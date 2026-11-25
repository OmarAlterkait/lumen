"""The navigator's arrays (host): a lowered Scene as volumes, daughter BVHs in frame groups, meshes and CSG tables.

    arrays = build(scene)                      # NavArrays: host arrays, names, definitions, statistics
    arrays = finish(prepare(scene), classes)   # the same: the class-free stage (ownership), then the classes'
    arrays = arrays.with_classes(vol_cls, mesh_cls, outside_cls)     # another class per volume (a new NavArrays)

- **Volumes:** one per instance of a definition with a CSG member, in the canonical order
  (``lumen.geometry.scene.instances``, the walk every consumer reads); a member-less definition (an assembly) places
  its daughters in the nearest ancestor volume.
- **Meshes** (opaque tessellated parts): pseudo-volumes (a mesh holds no point; a facet met, from either side, is an
  entry into it); their triangle chunks are daughter items of their mother volume (watertight ray-triangle test).
- **Frame groups:** per volume its daughters sharing a rotation (at least ``MIN_GROUP``, or all) whose world boxes are
  at least ``GROUP_GAIN`` times their own get a BVH in that rotation's frame (diagonal wire planes); the rest one
  world-frame BVH.
- **Overlaps and protrusions** (``lumen.geometry.ownership``: one owner per point, level by
  level): a sibling that may own part of a volume (or of an ancestor; a mesh as its triangle chunks; the filler of a
  lowered hole) is a candidate of that volume (where it holds a point, the point is not the volume's); an
  ancestor a volume is not proved inside is a candidate whose face is its exit (``size``: the volumes below each, so
  an ancestor is known); ``prio``, the rule's order, decides a locate and an entry tie. The containment tests decide
  every point at run time.
- **Transparent volumes** (``classes`` given): a volume of the class of every volume listing it as a candidate (or
  one interchangeable with it: ``lumen.build.tracked.interchangeable``), no sensor, not of a class with a surface
  between two of its volumes (a self pair), owning no part of a mesh, with everything below it proved inside it,
  bounds nothing a photon sees: it is no candidate, its candidates are its listers' instead, and a volume below it
  exits to its nearest ancestor that is not transparent (``host["parent"]``). The tree (names, ``size``) is
  unchanged.

The navigator follows Geant4's tracking, with no configuration.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from lumen.geometry import bvh, ownership
from lumen.geometry import scene as sc
from lumen.geometry.csg_build import CsgTables, csg_tables
from lumen.rules import navigation as rules
from lumen.rules.tolerances import DIVISOR_FLOOR, WORLD_BOX_PAD_ABS, WORLD_BOX_PAD_REL

STACK = 32  # the traversal's stack: one entry per BVH level
MESH_CHUNK = 8  # triangles per mesh chunk (one BVH item)
MORTON_BITS = 10  # mesh chunks: triangles ordered along a Morton curve of this many bits per axis
SRF_LIMIT = 2**31  # surface codes are int32


@dataclass(frozen=True, eq=False)
class NavArrays:
    """The navigator's host arrays and what they index.

    ``host``: the device arrays' host values (BVH nodes, frame groups, volume poses, kinds, classes, meshes);
    ``csg``: the CSG tables; ``n`` volumes, ``n_mesh`` meshes, ``chunk`` triangles per mesh chunk; ``ch0`` the first
    mesh-chunk item; ``srf`` = (su, sm, FM), the surface codes' layout (``host["ub"]``: per volume its first leaf
    slot); ``names``, ``defs`` per volume its placement path and definition; ``origins`` per volume, the outside
    ("OUT") and per mesh, the logical volume its definition was lowered from (its definition when none is recorded);
    ``pieces`` per volume the split union instance it is a piece of (-1 none: ``Definition.piece_of``); ``owned``:
    the overlapping siblings (non-owner, owner, forced: a lowered hole and its filler; a mesh k as its pseudo-volume
    n + 1 + k); ``clipped``: the (volume, mother) pairs where a volume reaches out of its mother; ``thin``: the
    sibling pairs that may share a point, not deeper than the contact tolerance (touching here); ``kids``: per volume
    (and the outside, then each mesh) its candidate items: daughters, owners, the ancestors it reaches out of;
    ``lo_w``, ``hi_w``: the volumes' padded world boxes (float64); ``probe``: host arrays only the placement check
    reads (``uc``, ``uh`` each volume's padded box in its frame, ``cmoth`` each mesh chunk's mother), never device
    operands of the navigator; ``precision``: the navigator's (``lumen.rules.navigation.Precision``), its float arrays
    in that dtype.
    """

    host: dict
    csg: CsgTables
    n: int
    n_mesh: int
    chunk: int
    ch0: int
    srf: tuple
    names: tuple
    defs: tuple
    stats: dict
    owned: tuple
    clipped: tuple
    kids: tuple
    lo_w: np.ndarray
    hi_w: np.ndarray
    probe: dict
    origins: tuple
    pieces: tuple
    thin: tuple
    precision: str = "float32"

    def with_classes(self, vol_cls, mesh_cls=(), outside_cls=None):
        """Return these arrays with another class per volume, mesh and the outside (``host["cls"]``)."""
        c = self.host["cls"].copy()
        c[: self.n] = np.asarray(vol_cls, np.int32)
        if outside_cls is not None:
            c[self.n] = int(outside_cls)
        if self.n_mesh:
            c[self.n + 1 :] = np.asarray(mesh_cls, np.int32)
        return dataclasses.replace(self, host=dict(self.host, cls=c))


def navigable(arrays: NavArrays) -> np.ndarray:
    """Return the volumes the navigator can be in (ascending): those reachable from the outside's candidates through
    the candidates of the volumes reached (``kids``). A transparent volume (dissolved: module docstring) is none of
    them: it is no candidate of a volume reached (its own row keeps its daughters, but no walk visits it). What the
    navigator can meet is these volumes and every mesh (``lumen.geometry.clearance``)."""  # noqa: D205, D209
    n = int(arrays.n)
    seen, todo = np.zeros(n, bool), [n]
    while todo:
        for x in arrays.kids[todo.pop()]:
            if x < n and not seen[x]:
                seen[x] = True
                todo.append(int(x))
    return np.flatnonzero(seen)


# ------------------------------------------------------------------------------------------------- the scene walk
def morton(c, bits=MORTON_BITS):
    """Return the Morton codes (uint64) of points c (n, 3) over their box."""
    lo, hi = c.min(0), c.max(0)
    q = ((c - lo) / np.maximum(hi - lo, 1e-12) * (2**bits - 1)).astype(np.uint64)
    code = np.zeros(len(c), np.uint64)
    for b in range(bits):
        for a in range(3):
            code |= ((q[:, a] >> np.uint64(b)) & np.uint64(1)) << np.uint64(3 * b + a)
    return code


def _check_definitions(scene, T):  # noqa: N803 -- T the instance table
    """Fail closed on what the navigator cannot place: a mesh with daughters or a mesh sensor."""
    for dname in set(T.defn):
        D = scene.definitions[dname]
        if D.member is not None and isinstance(D.member.solid, sc.Mesh):
            if D.placements:
                raise NotImplementedError(f"navigator: tessellated volume {dname!r} has daughters: a mesh is "
                                          "opaque (first hit only), so place them in its mother")  # fmt: skip
            if int(D.member.label.sensor) != -1:
                raise NotImplementedError(f"navigator: tessellated volume {dname!r} is a sensor: a mesh reports "
                                          "its first hit only; select a volume of a CSG solid")  # fmt: skip


def _volumes(scene, T):  # noqa: N803 -- T the instance table
    """Return (V: per volume its kind, pose, medium, parent, depth, name, definition, origin, piece; kinds; meshes).

    The volumes are the instances with a CSG member, in the canonical order (``lumen.geometry.scene.instances``); a
    volume's (and a mesh's) mother is the nearest volume above it.
    """
    defs = scene.definitions
    _check_definitions(scene, T)
    n_all = len(T.defn)
    mem = [defs[d_].member for d_ in T.defn]
    mesh = np.array([m is not None and isinstance(m.solid, sc.Mesh) for m in mem], bool)
    vol = np.array([m is not None for m in mem], bool) & ~mesh
    vo = np.flatnonzero(vol)
    vidx = np.full(n_all, -1, np.int64)
    vidx[vo] = np.arange(len(vo))
    moth = np.full(n_all, -1, np.int64)
    for i in range(1, n_all):  # parents come first in the canonical order
        p = int(T.parent[i])
        moth[i] = vidx[p] if vol[p] else moth[p]
    V = dict(kind=[], R=T.R[vo], t=T.t[vo], med=[], parent=moth[vo].tolist(), depth=T.depth[vo].tolist(),
             name=[T.path[i_] for i_ in vo], defn=[T.defn[i_] for i_ in vo])  # fmt: skip
    V["origin"] = [defs[d_].origin or d_ for d_ in V["defn"]]
    whole: dict = {}  # a split union's instance (the instance placing its pieces) -> its group
    V["piece"] = [whole.setdefault(int(T.parent[i_]), len(whole)) if defs[d_].piece_of else -1
                  for i_, d_ in zip(vo, V["defn"], strict=True)]  # fmt: skip
    kinds, kid_of = [], {}
    for i_ in vo:  # the kinds (first met), media: per volume
        m = mem[i_]
        k = id(m.solid)
        if k not in kid_of:
            kid_of[k] = len(kinds)
            kinds.append(m.solid)
        V["kind"].append(kid_of[k])
        V["med"].append(int(m.label.medium))
    meshes = [
        (mem[i_].solid.tris @ T.R[i_].T + T.t[i_], int(mem[i_].label.medium), int(moth[i_]), T.defn[i_], int(i_))
        for i_ in np.flatnonzero(mesh)
    ]  # (world triangles, medium, mother, definition, instance)
    V["inst"] = vo.tolist()
    return V, kinds, meshes


# ------------------------------------------------------------------------------------------------- links
def _pad(c, h):
    return WORLD_BOX_PAD_REL * (np.abs(c) + h) + WORLD_BOX_PAD_ABS


def _ownership(scene, V, kinds, tables, geo, prec):  # noqa: N803 -- V the volumes' columns
    """Return the Ownership (``lumen.geometry.ownership``) of the volumes, then the meshes (units n, n + 1, ...), at
    the contact tolerance of precision ``prec``."""  # noqa: D205, D209
    kind, parent, Rh, th, lo, hi, meshes = geo
    n = len(kind)
    nodes = list(kinds) + [sc.Mesh(m[0]) for m in meshes]  # a mesh: its world triangles
    boxes = tuple(tables.kind_box) + tuple(ownership.box_of(nd) for nd in nodes[len(kinds) :])
    defn = list(V["defn"]) + [m[3] for m in meshes]
    ukind = np.r_[np.asarray(kind, np.int64), len(kinds) + np.arange(len(meshes), dtype=np.int64)]
    upar = np.r_[np.where(parent[:n] >= n, -1, parent[:n]), [m[2] for m in meshes]].astype(np.int64)
    eye = np.broadcast_to(np.eye(3), (len(meshes), 3, 3))
    mlo = np.array([m[0].reshape(-1, 3).min(0) for m in meshes]).reshape(-1, 3)
    mhi = np.array([m[0].reshape(-1, 3).max(0) for m in meshes]).reshape(-1, 3)
    place = np.r_[np.asarray(V["inst"], np.int64), [m[4] for m in meshes]].astype(np.int64)
    units = ownership.Units(parent=upar, kind=ukind, nodes=nodes, box=boxes, R=np.concatenate([Rh, eye]),
                            t=np.concatenate([th, np.zeros((len(meshes), 3))]), place=place,
                            forced=ownership.hole_pairs(*ownership.hole_kinds(scene, defn), upar, np.r_[lo, mlo],
                                                        np.r_[hi, mhi]), prec=prec)  # fmt: skip
    return ownership.resolve(units)


def _transparent(n, parent, own, classes):
    """Return per volume whether it is transparent (module docstring): ``classes`` = (per volume, per mesh, the
    outside's class, the volumes never transparent, per class the class it is interchangeable with)."""  # noqa: D205, D209
    vol_cls, mesh_cls, out_cls, kept, canon = classes
    cls = np.asarray(canon, np.int64)[np.r_[np.asarray(vol_cls, np.int64), [int(out_cls)],
                                             np.asarray(mesh_cls, np.int64)].astype(np.int64)]  # fmt: skip
    ok = np.ones(n, bool)
    ok[np.asarray(kept, np.int64)] = False
    for u in range(n):
        ok[[a for a in own.clips[u] if a < n]] = False  # something below reaches out of it
        if cls[parent[u]] != cls[u]:
            ok[u] = False
        for w in own.owners[u]:  # it lists its owners: they must be of its class
            if w < n and cls[u] != cls[w]:
                ok[w] = False
    for a, b in own.pairs:  # an owner of part of a mesh (a mesh: unit n + k, class n + 1 + k)
        if a >= n and b < n:
            ok[b] = False
    return ok


def _dissolve(kids, parent, clear, clips):
    """Return (the candidates, the exit parents) with the transparent volumes ``clear`` replaced by their candidates
    (its daughters and owners; not the ancestors it is not proved inside: its listers lie inside them)."""  # noqa: D205, D209
    n = len(clear)
    memo = {}

    def expand(k):
        if k >= n or not clear[k]:
            return [k]
        if k not in memo:
            memo[k] = [x for j in kids[k] if j not in clips[k] for x in expand(j)]
        return memo[k]

    out = []
    for v, ks in enumerate(kids):
        seen, row = set(), []
        for x in [y for k in ks for y in expand(k)] if v >= n or not clear[v] else list(ks):
            if x not in seen:
                seen.add(x)
                row.append(x)
        out.append(row)
    up = np.asarray(parent).copy()
    for i in range(n):  # depth-first order: a mother's exit parent is final before its daughters'
        p = up[i]
        if p < n and clear[p]:
            up[i] = up[p]
    return out, up


def _nav_id(i, n):
    """Return the navigator's id of ownership unit i: a volume as is, mesh k (unit n + k) as its pseudo-volume."""
    return int(i) if i < n else int(i) + 1


def _mesh_chunks(meshes, n, mesh_chunk, kids):
    """Cut every mesh into Morton-ordered chunks of at most mesh_chunk triangles, items of their mother volume."""
    out_ = n
    ch0 = n + 1 + len(meshes)
    Vt, cst, cnn, cmesh, chm = [], [], [], [], []
    for m_, (T, _med, mother, *_) in enumerate(meshes):
        T = T[np.argsort(morton(T.mean(1)), kind="stable")]
        k0 = sum(len(x) for x in Vt)
        for s in range(0, len(T), mesh_chunk):
            cst.append(k0 + s)
            cnn.append(min(mesh_chunk, len(T) - s))
            cmesh.append(m_)
            chm.append(out_ if mother < 0 else mother)
        Vt.append(T)
    Vt = np.concatenate(Vt) if Vt else np.zeros((1, 3, 3))
    cst, cnn = np.asarray(cst, np.int64), np.asarray(cnn, np.int64)
    if len(cst):
        clo = np.minimum.reduceat(Vt.reshape(-1, 3), 3 * cst, axis=0)  # the chunks: consecutive triangles
        chi = np.maximum.reduceat(Vt.reshape(-1, 3), 3 * cst, axis=0)
    else:
        clo = chi = np.zeros((0, 3))
    for c in range(len(cst)):  # each chunk: an item of its mesh's mother (Geant4: not of the volumes below it)
        kids[chm[c]].append(ch0 + c)
    nrm = np.cross(Vt[:, 1] - Vt[:, 0], Vt[:, 2] - Vt[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), DIVISOR_FLOOR)
    return dict(Vt=Vt, cst=cst, cnn=cnn, cmesh=cmesh, chm=chm, clo=clo, chi=chi, nrm=nrm, ch0=ch0)


# ------------------------------------------------------------------------------------------------- frame groups
def _entries(kids, n, mesh, rot):
    """Return the daughter entries of every volume ordered (mother, group key, daughters' order) and their groups."""
    Rh, hb = rot
    nv = n + 1 + mesh["n_mesh"]
    lens = np.array([len(x) for x in kids], np.int64)
    kf = np.concatenate([np.asarray(x, np.int64) for x in kids]) if lens.sum() else np.zeros(0, np.int64)
    vf = np.repeat(np.arange(len(kids)), lens)
    un = kf < n
    gkey, G, Rg_all = bvh.frame_groups_all(Rh, hb, vf[un], kf[un])
    # one sequence: (mother, group key, kids order); a mother's mesh chunks after its unit groups
    ent_v = np.r_[vf[un], vf[~un]]
    ent_k = np.r_[gkey, np.full(int((~un).sum()), 3 * (G + nv), np.int64)]
    ent_i = np.r_[kf[un], kf[~un]]
    o_ = np.lexsort((np.arange(len(ent_v)), ent_k, ent_v))
    ent_v, ent_k, ent_i = ent_v[o_], ent_k[o_], ent_i[o_]
    newg = np.r_[True, (ent_v[1:] != ent_v[:-1]) | (ent_k[1:] != ent_k[:-1])] if len(ent_v) else np.zeros(0, bool)
    gs = np.flatnonzero(newg)  # each group's first entry
    g_mesh = ent_i[gs] >= mesh["ch0"]
    g_rot = ~g_mesh & (ent_k[gs] < G)
    GR = [np.eye(3)] * len(gs)
    for j in np.flatnonzero(g_rot):
        GR[j] = Rg_all[ent_k[gs][j]]
    return dict(v=ent_v, i=ent_i, gs=gs, g_v=ent_v[gs], g_mesh=g_mesh, g_rot=g_rot, GR=GR, nv=nv)


def _group_boxes(E, n, mesh, poses):  # noqa: N803 -- E the daughter entries
    """Return the entries' boxes (lo, hi), each in its group's frame (world for the world-frame and mesh groups)."""
    Rh, th, cb, hb = poses
    ent_i, gs, g_rot, GR = E["i"], E["gs"], E["g_rot"], E["GR"]
    blo = np.zeros((len(ent_i), 3))
    bhi = np.zeros((len(ent_i), 3))
    unit = ent_i < n
    wk = unit & ~np.repeat(g_rot, np.diff(np.r_[gs, len(ent_i)]))  # the world-frame groups' units
    if wk.any():  # Rg = I: the per-group formulas, batched
        kg = ent_i[wk]
        Ig = np.eye(3)
        M = np.einsum("ji,njk->nik", Ig, Rh[kg])
        cg = np.einsum("nij,nj->ni", M, cb[kg]) + th[kg] @ Ig
        hg = np.einsum("nij,nj->ni", np.abs(M), hb[kg])
        pg = _pad(cg, hg)
        blo[wk], bhi[wk] = cg - hg - pg, cg + hg + pg
    ge = np.r_[gs, len(ent_i)]
    for j in np.flatnonzero(g_rot):  # a rotation group: in its own frame
        sl = slice(ge[j], ge[j + 1])
        kg, Rg = ent_i[sl], GR[j]
        M = np.einsum("ji,njk->nik", Rg, Rh[kg])  # Rg^T R_i
        cg = np.einsum("nij,nj->ni", M, cb[kg]) + th[kg] @ Rg  # Rg^T (R_i c + t_i)
        hg = np.einsum("nij,nj->ni", np.abs(M), hb[kg])
        pg = _pad(cg, hg)
        blo[sl], bhi[sl] = cg - hg - pg, cg + hg + pg
    ch = ~unit
    if ch.any():  # mesh chunks: their world boxes
        kc = ent_i[ch] - mesh["ch0"]
        cp = WORLD_BOX_PAD_REL * (np.abs(mesh["clo"][kc]) + np.abs(mesh["chi"][kc])) + WORLD_BOX_PAD_ABS
        blo[ch], bhi[ch] = mesh["clo"][kc] - cp, mesh["chi"][kc] + cp
    return blo, bhi, ge


def _trees(E, boxes):  # noqa: N803 -- E the daughter entries
    """Return every group's BVH, built in one level-synchronous pass: the node arrays, each group's first and
    past-the-end node (g0, g1) and the depth; None when there are no groups."""  # noqa: D205, D209
    blo, bhi, ge = boxes
    if not len(E["gs"]):
        return None
    B_ = bvh.build_many(blo, bhi, ge)
    t0 = np.r_[B_["tree0"], len(B_["skip"])]
    return dict(lo=B_["lo"], hi=B_["hi"], skip=B_["skip"], start=B_["start"], count=B_["count"],
                items=E["i"][B_["items"]].astype(np.int32), g0=t0[:-1], g1=t0[1:],
                depth=int(B_["n_levels"]))  # fmt: skip


def _node(trees, k, shape, dt):
    """Return a node array of the BVHs as ``dt`` (a placeholder of ``shape`` when there are none)."""
    return np.zeros(shape, dt) if trees is None else np.asarray(trees[k]).astype(dt)


def _host(E, trees, vols, mesh, bg, fdt):  # noqa: N803 -- E the daughter entries
    """Return the host arrays of the navigator, its float arrays in ``fdt``."""
    V, kind, parent, Rh, th, prio = vols
    nv, g_v, g_mesh = E["nv"], E["g_v"], E["g_mesh"]
    av = np.arange(nv)
    vg0 = np.searchsorted(g_v, av, "left").astype(np.int32)
    vg1 = np.searchsorted(g_v, av, "right").astype(np.int32)
    vg1u = (vg0 + np.bincount(g_v[~g_mesh], minlength=nv)[:nv]).astype(np.int32)
    out_ = len(kind)
    mmed = np.asarray([x[1] for x in mesh["meshes"]], np.int32)
    mpar = np.asarray([out_ if x[2] < 0 else x[2] for x in mesh["meshes"]], np.int32)
    Vt, cst, cnn, cmesh = mesh["Vt"], mesh["cst"], mesh["cnn"], mesh["cmesh"]
    GR = E["GR"]
    return dict(lo=_node(trees, "lo", (1, 3), fdt), hi=_node(trees, "hi", (1, 3), fdt),
                skip=_node(trees, "skip", (1,), np.int32), start=_node(trees, "start", (1,), np.int32),
                count=_node(trees, "count", (1,), np.int32), items=_node(trees, "items", (1,), np.int32),
                gR=(np.stack(GR) if GR else np.eye(3)[None]).astype(fdt),
                g0=_node(trees, "g0", (1,), np.int32), g1=_node(trees, "g1", (1,), np.int32), vg0=vg0, vg1=vg1,
                vg1u=np.maximum(vg1u, vg0),
                parent=np.concatenate([parent, [out_], mpar]).astype(np.int32),  # per navigator id
                cls=np.concatenate([np.asarray(V["med"], np.int32), [bg], mmed]).astype(np.int32),
                kind=kind.astype(np.int32), R=Rh.astype(fdt), t=th.astype(fdt),
                prio=np.asarray(prio, np.int32),
                V0=Vt[:, 0].astype(fdt), V1=Vt[:, 1].astype(fdt), V2=Vt[:, 2].astype(fdt),
                N=mesh["nrm"].astype(fdt), cstart=np.asarray(cst if len(cst) else [0], np.int32),
                cn=np.asarray(cnn if len(cnn) else [0], np.int32),
                cmesh=np.asarray(cmesh if cmesh else [0], np.int32))  # fmt: skip


def _srf_layout(tables, kind, ch0, mesh):
    """Return the surface codes' layout (su, sm, FM), the per-volume leaf base ``ub`` and the triangle -> chunk table.

    Above the items (a plain item: the band rules): units su + ((ub[item] + leaf) * FM + face) * 4 + 2 o + e, with
    ub[i] the leaves of the volumes before i (each volume spans its own kind's leaves: the code space is the scene's
    leaves, not the volumes times the largest solid's); mesh triangles sm + 4 k + 2 o + e (o: the outward normal is
    +grad; e: the event entered the item).
    """
    FM = 1 + int(np.max(tables.T["np_"]))
    nl = np.maximum(np.asarray(tables.K["nl"], np.int64)[np.asarray(kind, np.int64)], 1)
    ub = np.concatenate([[0], np.cumsum(nl)])
    cst, Vt = mesh["cst"], mesh["Vt"]
    su = ch0 + len(cst)
    sm = su + int(ub[-1]) * FM * 4
    if sm + 4 * len(Vt) >= SRF_LIMIT:
        raise ValueError(f"navigator: the surface codes overflow int32 ({sm + 4 * len(Vt)}): the detector volume "
                         "holds too many solid leaves and mesh triangles; simplify its largest solids or "
                         "meshes")  # fmt: skip
    tchunk = ((np.searchsorted(cst, np.arange(len(Vt)), side="right") - 1).clip(0).astype(np.int32)
              if len(cst) else np.zeros(1, np.int32))  # fmt: skip
    return (su, sm, FM), ub.astype(np.int32), tchunk


def _volume_boxes(tables, V):  # noqa: N803 -- V the volumes' columns
    """Return (kind, Rh, th, cb, hb, lo, hi): each volume's kind, pose, kind box and padded world box."""
    kind = np.asarray(V["kind"], np.int64)
    Rh = np.asarray(V["R"], float).reshape(-1, 3, 3)
    th = np.asarray(V["t"], float).reshape(-1, 3)
    cb = np.stack([tables.kind_box[k][0] for k in kind])
    hb = np.stack([tables.kind_box[k][1] for k in kind])
    cw = np.einsum("nij,nj->ni", Rh, cb) + th
    hw = np.einsum("nij,nj->ni", np.abs(Rh), hb)
    pad = _pad(cw, hw)
    return kind, Rh, th, cb, hb, cw - hw - pad, cw + hw + pad


def _candidates(own, n, kids, mesh, n_mesh):
    """Add each volume's owners (a mesh: its triangle chunks) and the ancestors it is not proved inside to its
    candidates."""  # noqa: D205, D209
    chunks_of = [[] for _ in range(n_mesh)]
    for c, m_ in enumerate(mesh["cmesh"]):
        chunks_of[m_].append(mesh["ch0"] + c)
    for u in range(n):  # an owner is a candidate of its non-owner and of the volumes below it it reaches (a mesh: its
        # triangle chunks); an ancestor u reaches out of is a candidate too (its exit is u's)
        kids[u].extend(w if w < n else c for w in own.owners[u] for c in ([w] if w < n else chunks_of[w - n]))
        kids[u].extend(own.clips[u])


@dataclass(frozen=True, eq=False)
class Stage:
    """The class-free part of a build (``prepare``): the volumes, CSG tables, boxes, meshes, ownership and candidates.

    It reads no optics, so one Stage serves every set of classes (``finish``); its ownership (``own``) is the costly
    part, which the build cache keeps per scene (``lumen.geometry.cache``).
    """

    scene: object
    V: dict
    kinds: list
    meshes: list
    tables: CsgTables
    boxes: tuple  # (kind, Rh, th, cb, hb, lo, hi)
    parent: np.ndarray
    kids: list  # per volume, the outside, then each mesh: its candidate items (before any dissolution)
    own: object  # lumen.geometry.ownership.Ownership
    mesh: dict
    prec: object = rules.FLOAT32  # the navigator's precision


def prepare(scene, own=None, precision="float32", instances=None) -> Stage:
    """Return the class-free Stage of a lowered Scene in ``precision`` ("float32" or "float64"); ``own``: its
    Ownership if already known (a Stage's ``own``, of the same precision); ``instances``: the scene's
    ``scene.instances``, if already listed."""  # noqa: D205, D209
    prec = rules.precision(precision)
    V, kinds, meshes = _volumes(scene, sc.instances(scene) if instances is None else instances)
    n = len(V["kind"])
    if n == 0:
        raise ValueError("navigator: a scene without volumes")
    tables = csg_tables(kinds, prec.np)
    kind, Rh, th, cb, hb, lo, hi = _volume_boxes(tables, V)
    parent = np.where(np.asarray(V["parent"]) < 0, n, np.asarray(V["parent"])).astype(np.int32)
    kids = [[] for _ in range(n + 1)]
    for i in range(n):
        kids[parent[i]].append(i)
    if own is None:
        own = _ownership(scene, V, kinds, tables, (kind, parent, Rh, th, lo, hi, meshes), prec)
    mesh = _mesh_chunks(meshes, n, MESH_CHUNK, kids)
    mesh.update(n_mesh=len(meshes), meshes=meshes)
    _candidates(own, n, kids, mesh, len(meshes))
    return Stage(scene, V, kinds, meshes, tables, (kind, Rh, th, cb, hb, lo, hi), parent, kids, own, mesh, prec)


class Classes(NamedTuple):
    """The classes a navigator reports and dissolves by (``lumen.build.tracked``)."""

    volume: tuple  # per volume
    mesh: tuple  # per mesh
    outside: int
    kept: tuple  # the volumes never transparent (a sensor's, a self pair class's)
    canon: tuple  # per class, the class it is interchangeable with


def build(scene, classes: Classes | None = None, precision="float32", instances=None):
    """Return the NavArrays of a lowered Scene (``lumen.geometry.scene``): ``finish(prepare(scene, precision=))``.

    ``classes``: the classes reported, and the transparent volumes dissolved (module docstring); None: a label's
    medium is the class reported, nothing dissolved. ``precision``: the navigator's, "float32" or "float64".
    ``instances``: the scene's ``scene.instances``, if already listed.
    """
    return finish(prepare(scene, precision=precision, instances=instances), classes)


def finish(st: Stage, classes: Classes | None = None):
    """Return the NavArrays of a Stage for ``classes`` (``build``): the transparent volumes dissolved, the BVHs."""
    scene, V, kinds, meshes, tables, own, mesh, parent = st.scene, st.V, st.kinds, st.meshes, st.tables, st.own, \
        dict(st.mesh), st.parent  # fmt: skip
    kind, Rh, th, cb, hb, lo, hi = st.boxes
    n = len(V["kind"])
    kids = [list(k) for k in st.kids]
    clear = _transparent(n, parent, own, classes) if classes is not None else np.zeros(n, bool)
    kids, up = _dissolve(kids, parent, clear, own.clips) if clear.any() else (kids, parent)
    E = _entries(kids, n, mesh, (Rh, hb))
    trees = _trees(E, _group_boxes(E, n, mesh, (Rh, th, cb, hb)))
    # the rule's order per navigator id (volumes, the outside last, then each mesh as its unit)
    prio = np.r_[own.prio[:n], [len(own.prio)], own.prio[n:]]
    host = _host(E, trees, (V, kind, up, Rh, th, prio), mesh, int(scene.outside.medium), st.prec.np)
    for k, m_ in enumerate(meshes):  # a mesh in a transparent volume: its exit parent is that volume's
        if m_[2] >= 0 and clear[m_[2]]:
            host["parent"][n + 1 + k] = up[m_[2]]
    size = np.ones(n, np.int64)  # the volumes below each, itself included: depth-first order makes them contiguous
    for i in range(n - 1, -1, -1):
        if parent[i] < n:
            size[parent[i]] += size[i]
    host["size"] = np.concatenate([size, np.zeros(1 + len(meshes), np.int64)]).astype(np.int32)
    host["piece"] = np.concatenate([np.asarray(V["piece"], np.int64), np.full(1 + len(meshes), -1)]).astype(np.int32)
    dmax = (
        0 if trees is None else trees["depth"]
    )  # the deepest BVH: the traversal's stack holds one entry per level; a deeper one would drop nodes
    if dmax > STACK:
        raise ValueError(f"navigator: a BVH of depth {dmax} exceeds the traversal stack ({STACK}): a volume has "
                         f"too many daughters ({max(len(k_) for k_ in kids)}): group them in sub-volumes")  # fmt: skip
    srf, host["ub"], host["tchunk"] = _srf_layout(tables, kind, mesh["ch0"], mesh)
    stats = dict(volumes=n, kinds=len(kinds), meshes=len(meshes), chunks=len(mesh["cst"]), groups=len(E["GR"]),
                 framed=int(E["g_rot"].sum()), hole_links=int(own.forced.sum()), links=int((~own.forced).sum()),
                 owner_links=sum(len(x) for x in own.owners), clip_links=sum(len(x) for x in own.clips),
                 transparent=int(clear.sum()), max_daughters=int(max(len(k_) for k_ in kids)),
                 bvh_depth=dmax)  # fmt: skip
    defs_, chm = scene.definitions, mesh["chm"]
    probe = dict(uc=cb.astype(st.prec.np), uh=(hb + _pad(cb, hb)).astype(st.prec.np),
                 cmoth=np.asarray(chm if chm else [n], np.int32))  # fmt: skip
    arrays = NavArrays(host=host, csg=tables, n=n, n_mesh=len(meshes), chunk=MESH_CHUNK, ch0=mesh["ch0"], srf=srf,
                     names=tuple(V["name"]), defs=tuple(V["defn"]), stats=stats,
                     owned=tuple((_nav_id(a, n), _nav_id(b, n), bool(f))
                                 for (a, b), f in zip(own.pairs, own.forced, strict=True)),
                     clipped=tuple((_nav_id(a, n), _nav_id(b, n)) for a, b in own.protrude),
                     kids=tuple(tuple(k_) for k_ in kids), lo_w=lo, hi_w=hi, probe=probe,
                     origins=tuple(V["origin"]) + ("OUT",) + tuple(defs_[m[3]].origin or m[3] for m in meshes),
                     pieces=tuple(V["piece"]),
                     thin=tuple((_nav_id(a, n), _nav_id(b, n)) for a, b in own.thin),
                     precision=st.prec.name)  # fmt: skip
    return arrays if classes is None else arrays.with_classes(classes.volume, classes.mesh, classes.outside)
