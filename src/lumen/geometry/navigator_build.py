"""The navigator's arrays (host): a lowered Scene as volumes, daughter BVHs in frame groups, meshes and CSG tables.

    arrays = build(scene)                      # NavArrays: host arrays, names, definitions, statistics
    arrays = arrays.with_classes(vol_cls, mesh_cls, outside_cls)     # another class per volume (a new NavArrays)

- **Volumes:** one per placed single-member definition, in the canonical depth-first order (the oracle's instances);
  a member-less definition (an assembly) places its daughters in the nearest ancestor volume; a definition of several
  members fails closed.
- **Meshes** (opaque tessellated parts): pseudo-volumes (a mesh holds no point); their triangle chunks are daughter
  items of their mother volume (watertight ray-triangle test).
- **Frame groups:** per volume its daughters sharing a rotation (at least ``MIN_GROUP``, or all) whose world boxes are
  at least ``GROUP_GAIN`` times their own get a BVH in that rotation's frame (SBND's diagonal wires); the rest one
  world-frame BVH.
- **Overlapping siblings** need no configuration (``lumen.rules.navigation``); **coincident siblings** (one mother,
  one solid, one pose) get their partner ``coin_hi``; the import's lowered holes (a solid kept without its holes, the
  siblings filling them, ``Scene.hole_fillers``) make those siblings daughter candidates of the holed solid (``fill``).

Notes
-----
Ported from the baseline's lumen.geometry.navigator (the constructor), from LUCiD's vtrack prototype. The baseline's
``overlaps=`` links and ``mesh_descend`` are not ported: the navigator follows Geant4's tracking, no configuration.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from lumen.build import scene as sc
from lumen.geometry import bvh
from lumen.geometry.csg_build import CsgTables, csg_tables

STACK = 32  # the traversal's stack: one entry per BVH level
MESH_CHUNK = 8  # triangles per mesh chunk (one BVH item)
BOX_PAD_REL = 1e-6  # world boxes are padded by this times (|centre| + half) ...
BOX_PAD_ABS = 1e-9  # ... plus this (m)
NORMAL_FLOOR = 1e-30  # a degenerate triangle's normal length floor
MORTON_BITS = 10  # mesh chunks: triangles ordered along a Morton curve of this many bits per axis
SRF_LIMIT = 2**31  # surface codes are int32
PATH_RE = r"\|pc\d+(\||$)"  # a split union's piece: '<definition>|pc<k>'
HOLES = "|holes:"  # a definition variant with its holes dropped


@dataclass(frozen=True, eq=False)
class NavArrays:
    """The navigator's host arrays and what they index.

    ``host``: the device arrays' host values (BVH nodes, frame groups, volume poses, kinds, classes, meshes);
    ``csg``: the CSG tables; ``n`` volumes, ``n_mesh`` meshes, ``chunk`` triangles per mesh chunk; ``ch0`` the
    first mesh-chunk item; ``srf`` = (su, sm, LM, FM), the surface codes' layout; ``names``, ``defs`` per volume its
    placement path and definition; ``coincident``: groups of coincident siblings; ``kids``: per volume (and the
    outside, then each mesh) its daughter items; ``lo_w``, ``hi_w``: the volumes' padded world boxes (float64).
    """

    host: dict
    csg: CsgTables
    n: int
    n_mesh: int
    chunk: int
    ch0: int
    srf: tuple
    bg: int
    names: tuple
    defs: tuple
    stats: dict
    coincident: tuple
    kids: tuple
    lo_w: np.ndarray
    hi_w: np.ndarray

    def with_classes(self, vol_cls, mesh_cls=(), outside_cls=None):
        """Return these arrays with another class per volume, mesh and the outside (``host["cls"]``)."""
        c = self.host["cls"].copy()
        c[: self.n] = np.asarray(vol_cls, np.int32)
        if outside_cls is not None:
            c[self.n] = int(outside_cls)
        if self.n_mesh:
            c[self.n + 1 :] = np.asarray(mesh_cls, np.int32)
        return dataclasses.replace(self, host=dict(self.host, cls=c))


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


def _placements(d):
    """Return a definition's placements as arrays: (R, t, sensor, step name, definition name)."""
    return (np.stack([np.asarray(p.R, float) for p in d.placements]),
            np.stack([np.asarray(p.t, float) for p in d.placements]),
            np.array([int(p.sensor) for p in d.placements]),
            [p.name or f"{p.definition}#{k}" for k, p in enumerate(d.placements)],
            [p.definition for p in d.placements])  # fmt: skip


def _place_level(scene, dix, cache, prev):
    """Return the next level of the walk: every (instance, placement) pair of one definition at one depth in a batch."""
    dn, Rl, tl, ml, pl_ = prev
    out = []
    dnames = list(scene.definitions)
    for d_ in np.unique(dn):
        D = scene.definitions[dnames[d_]]
        if not D.placements:
            continue
        if d_ not in cache:
            Rp, tp, sp, nm_, dp = _placements(D)
            cache[d_] = (Rp, tp, sp, nm_, np.array([dix[x] for x in dp]))
        Rp, tp, sp, nm_, dp = cache[d_]
        ix = np.flatnonzero(dn == d_)
        P = len(Rp)
        Rc = np.matmul(Rl[ix][:, None], Rp[None])  # Rw @ R_
        tc = np.matmul(Rl[ix][:, None], tp[None, :, :, None])[..., 0] + tl[ix][:, None]  # Rw @ t_ + tw
        out.append((np.repeat(ix, P), np.tile(np.arange(P), len(ix)), Rc.reshape(-1, 3, 3), tc.reshape(-1, 3),
                    np.tile(dp, len(ix)), np.where(sp[None] >= 0, sp[None], ml[ix][:, None]).ravel(),
                    [(pl_[i_] + "/" + q_) if pl_[i_] else q_ for i_ in ix for q_ in nm_]))  # fmt: skip
    return out


def _walk(scene):
    """Walk the scene level by level; return dict of per-instance arrays (definition, pose, parent, ...) and levels."""
    dix = {d_: i_ for i_, d_ in enumerate(scene.definitions)}
    cache = {}
    L = dict(dn=[np.array([dix[scene.root]])], R=[np.eye(3)[None]], t=[np.zeros((1, 3))], par=[np.array([-1])],
             sib=[np.array([0])], msen=[np.array([-1])], path=[[""]], lev=[np.array([0])])  # fmt: skip
    n_all = 1
    while True:
        base = n_all - len(L["dn"][-1])
        out = _place_level(scene, dix, cache, (L["dn"][-1], L["R"][-1], L["t"][-1], L["msen"][-1], L["path"][-1]))
        if not out:
            break
        L["par"].append(base + np.concatenate([o[0] for o in out]))
        L["sib"].append(np.concatenate([o[1] for o in out]))
        L["R"].append(np.concatenate([o[2] for o in out]))
        L["t"].append(np.concatenate([o[3] for o in out]))
        L["dn"].append(np.concatenate([o[4] for o in out]))
        L["msen"].append(np.concatenate([o[5] for o in out]))
        L["path"].append([x for o in out for x in o[6]])
        L["lev"].append(np.full(len(L["par"][-1]), len(L["lev"])))
        n_all += len(L["par"][-1])
    W = {k: np.concatenate(v) for k, v in L.items() if k != "path"}
    W["paths"] = [x for p_ in L["path"] for x in p_]
    W["lv_off"] = np.cumsum([0] + [len(x) for x in L["dn"]])
    W["n_all"] = n_all
    return W


def _preorder(W):  # noqa: N803 -- W the walk's arrays
    """Return the depth-first order of the instances (preorder, children by placement order)."""
    n_all, par, sib, lv_off = W["n_all"], W["par"], W["sib"], W["lv_off"]
    n_lev = len(lv_off) - 1
    sub = np.ones(n_all, np.int64)
    for L_ in range(n_lev - 1, 0, -1):  # subtree sizes bottom-up
        c_ = np.arange(lv_off[L_], lv_off[L_ + 1])
        np.add.at(sub, par[c_], sub[c_])
    pre = np.zeros(n_all, np.int64)
    for L_ in range(1, n_lev):  # then each child's offset
        c_ = np.arange(lv_off[L_], lv_off[L_ + 1])
        o_ = c_[np.lexsort((sib[c_], par[c_]))]  # by parent, then placement order
        ex = np.cumsum(sub[o_]) - sub[o_]  # the earlier siblings' (and others') sizes
        ps = par[o_]
        first = np.r_[True, ps[1:] != ps[:-1]]
        ex = ex - np.maximum.accumulate(np.where(first, ex, 0))
        pre[o_] = pre[ps] + 1 + ex
    return np.argsort(pre, kind="stable")


def _check_definitions(scene, dn):
    """Fail closed on what the navigator cannot place: several members, a mesh with daughters or a mesh sensor."""
    dnames = list(scene.definitions)
    for d_ in np.unique(dn):
        D = scene.definitions[dnames[d_]]
        if len(D.members) > 1:
            raise NotImplementedError(f"navigator: definition {dnames[d_]!r} has {len(D.members)} members (one solid "
                                      "per volume: the lowering gives one member per logical volume)")  # fmt: skip
        if D.members and isinstance(D.members[0].solid, sc.Mesh):
            if D.placements:
                raise NotImplementedError(f"navigator: mesh definition {dnames[d_]!r} with daughters")
            if int(D.members[0].label.sensor) != -1:
                raise NotImplementedError(f"navigator: mesh definition {dnames[d_]!r} is a sensor")


def _mothers(W, vol, order):  # noqa: N803 -- W the walk's arrays
    """Return (the volumes' instances in depth-first order, each instance's mother volume: the nearest above it)."""
    vo = order[vol[order]]
    vidx = np.full(W["n_all"], -1, np.int64)
    vidx[vo] = np.arange(len(vo))
    moth = np.full(W["n_all"], -1, np.int64)
    lv_off, par = W["lv_off"], W["par"]
    for L_ in range(1, len(lv_off) - 1):
        c_ = np.arange(lv_off[L_], lv_off[L_ + 1])
        p_ = par[c_]
        moth[c_] = np.where(vol[p_], vidx[p_], moth[p_])
    return vo, moth


def _volumes(scene, W):  # noqa: N803 -- W the walk's arrays
    """Return (V: per volume its kind, pose, medium, sensor, parent, depth, name, definition; kinds; meshes)."""
    defs, dnames = scene.definitions, list(scene.definitions)
    dn = W["dn"]
    order = _preorder(W)
    _check_definitions(scene, dn)
    is_mesh = np.array([bool(defs[d_].members) and isinstance(defs[d_].members[0].solid, sc.Mesh) for d_ in dnames])
    has_m = np.array([bool(defs[d_].members) for d_ in dnames])
    vol = has_m[dn] & ~is_mesh[dn]
    vo, moth = _mothers(W, vol, order)
    V = dict(kind=[], R=W["R"][vo], t=W["t"][vo], med=[], sen=[], parent=moth[vo].tolist(), depth=W["lev"][vo].tolist(),
             name=[W["paths"][i_] for i_ in vo], defn=[dnames[d_] for d_ in dn[vo]])  # fmt: skip
    kinds, kid_of, path_id = [], {}, 0
    for i_ in vo:  # the kinds (first met), sensors, media: per volume
        m = defs[dnames[dn[i_]]].members[0]
        k = id(m.solid)
        if k not in kid_of:
            kid_of[k] = len(kinds)
            kinds.append(m.solid)
        s = int(m.label.sensor)
        if s == sc.PATH:
            s = path_id
            path_id += 1
        elif s == sc.PLACEMENT:
            s = int(W["msen"][i_])
        V["kind"].append(kid_of[k])
        V["med"].append(int(m.label.medium))
        V["sen"].append(s)
    meshes = []  # (world triangles, medium, mother)
    for i_ in order[(has_m[dn] & is_mesh[dn])[order]]:
        m = defs[dnames[dn[i_]]].members[0]
        meshes.append((m.solid.tris @ W["R"][i_].T + W["t"][i_], int(m.label.medium), int(moth[i_])))
    return V, kinds, meshes


# ------------------------------------------------------------------------------------------------- links
def _pad(c, h):
    return BOX_PAD_REL * (np.abs(c) + h) + BOX_PAD_ABS


def _hole_links(scene, V, parent, lo, hi, kids):  # noqa: N803 -- V the volumes' columns
    """Link the import's hole fillers as daughter candidates of the holed solid whose box they meet; return the count.

    The filler siblings (``Scene.hole_fillers``, placed after) win where they hold a point, as any later sibling.
    """
    n = len(V["defn"])
    ov = set(scene.hole_fillers)
    ud, inv_d = np.unique(np.asarray(V["defn"], dtype=object).astype(str), return_inverse=True)  # per definition
    holed = np.array([HOLES in d_ for d_ in ud], bool)[inv_d.ravel()]
    flag = (np.array([any(d_ == o_ or d_.startswith(o_ + "|") for o_ in ov) for d_ in ud], bool)[inv_d.ravel()]
            if ov else np.zeros(n, bool))  # fmt: skip
    n_hole = 0
    for u in np.flatnonzero(holed):
        sib = np.flatnonzero((parent[:n] == parent[u]) & flag & ~holed & (np.arange(n) > u))
        meet = sib[np.all((lo[sib] <= hi[u]) & (hi[sib] >= lo[u]), axis=1)]
        kids[u].extend(int(i) for i in meet)
        n_hole += len(meet)
    return n_hole


def kind_signatures(tables):
    """Return per kind an id equal for kinds of the same solid (leaf tables, boolean program).

    Coincident siblings may be placements of two definitions with one shape (MicroBooNE's PMT lens and underside).
    """
    T, K, G = tables.T, tables.K, tables.G
    nL = len(T["typ"])
    lf = [f for f in sorted(T) if f != "p0" and np.shape(T[f])[0] == nL]
    seen, out = {}, np.zeros(tables.n_kinds, np.int64)
    for k in range(tables.n_kinds):
        l0, nl, p0, pl = (int(K[x][k]) for x in ("l0", "nl", "p0", "pl"))
        rows = []
        for l_ in range(l0, l0 + nl):
            pp = slice(int(T["p0"][l_]), int(T["p0"][l_]) + int(T["np_"][l_]))
            rows.append(np.r_[[T[f][l_] for f in lf], T["n0"][pp], T["n1"][pp], T["n2"][pp], T["h"][pp]].tobytes())
        key = (tuple(rows), np.asarray(G["op"][p0 : p0 + pl]).tobytes(), np.asarray(G["arg"][p0 : p0 + pl]).tobytes())
        out[k] = seen.setdefault(key, len(seen))
    return out


def _coincident(tables, kind, parent, Rh, th):  # noqa: N803 -- Rh the volumes' rotations
    """Return (coin_hi, groups): per volume its coincident partner (-1 none), and the coincident groups.

    One mother, one solid, one pose: the lowest index fills the region; leaving a member, Geant4 locates the photon in
    the highest-index other member (LevelLocate excludes only the volume left).
    """
    n = len(kind)
    coin_hi = np.full(n, -1, np.int32)
    ks = kind_signatures(tables)  # one id per distinct solid (two definitions of one shape: one)
    ck = np.c_[parent[:n], ks[kind], Rh.reshape(n, 9), th]
    _, ci, cc = np.unique(ck, axis=0, return_inverse=True, return_counts=True)
    ci = ci.ravel()
    for g_ in np.flatnonzero(cc > 1):
        m_ = np.flatnonzero(ci == g_)
        coin_hi[m_[:-1]] = m_[-1]
        coin_hi[m_[-1]] = m_[-2]
    return coin_hi, tuple(tuple(np.flatnonzero(ci == g_).tolist()) for g_ in np.flatnonzero(cc > 1))


def _mesh_chunks(meshes, n, mesh_chunk, kids):
    """Cut every mesh into Morton-ordered chunks of at most mesh_chunk triangles, items of their mother volume."""
    out_ = n
    ch0 = n + 1 + len(meshes)
    Vt, cst, cnn, cmesh, chm = [], [], [], [], []
    for m_, (T, _med, mother) in enumerate(meshes):
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
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), NORMAL_FLOOR)
    return dict(Vt=Vt, cst=cst, cnn=cnn, cmesh=cmesh, chm=chm, clo=clo, chi=chi, nrm=nrm, ch0=ch0)


# ------------------------------------------------------------------------------------------------- frame groups
def _entries(kids, n, mesh, rot, min_group, gain):
    """Return the daughter entries of every volume ordered (mother, group key, daughters' order) and their groups."""
    Rh, hb = rot
    nv = n + 1 + mesh["n_mesh"]
    lens = np.array([len(x) for x in kids], np.int64)
    kf = np.concatenate([np.asarray(x, np.int64) for x in kids]) if lens.sum() else np.zeros(0, np.int64)
    vf = np.repeat(np.arange(len(kids)), lens)
    un = kf < n
    gkey, G, Rg_all, _ = bvh.frame_groups_all(Rh, hb, vf[un], kf[un], min_group, gain)
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
        cp = BOX_PAD_REL * (np.abs(mesh["clo"][kc]) + np.abs(mesh["chi"][kc])) + BOX_PAD_ABS
        blo[ch], bhi[ch] = mesh["clo"][kc] - cp, mesh["chi"][kc] + cp
    return blo, bhi, ge


def _trees(E, boxes, leaf_size):  # noqa: N803 -- E the daughter entries
    """Return every group's BVH, built in one level-synchronous pass: (node arrays, g0, g1, depth)."""
    blo, bhi, ge = boxes
    if not len(E["gs"]):
        return [], [], [], [], [], [], [], [], 0
    B_ = bvh.build_many(blo, bhi, ge, leaf_size)
    t0 = np.r_[B_["tree0"], len(B_["skip"])]
    return ([B_["lo"]], [B_["hi"]], [B_["skip"]], [B_["start"]], [B_["count"]],
            [E["i"][B_["items"]].astype(np.int32)], t0[:-1].tolist(), t0[1:].tolist(), int(B_["n_levels"]))  # fmt: skip


def _cat(xs, shape, dt):
    return np.concatenate(xs).astype(dt) if xs else np.zeros(shape, dt)


def _host(E, trees, vols, mesh, bg):  # noqa: N803 -- E the daughter entries
    """Return the host arrays of the navigator."""
    L, H, SK, ST, CN, IT, G0, G1, _ = trees
    V, kind, parent, Rh, th, cb, hb, coin_hi, fill = vols
    nv, g_v, g_mesh = E["nv"], E["g_v"], E["g_mesh"]
    av = np.arange(nv)
    vg0 = np.searchsorted(g_v, av, "left").astype(np.int32)
    vg1 = np.searchsorted(g_v, av, "right").astype(np.int32)
    vg1u = (vg0 + np.bincount(g_v[~g_mesh], minlength=nv)[:nv]).astype(np.int32)
    n_mesh = mesh["n_mesh"]
    out_ = len(kind)
    mmed = np.asarray([x[1] for x in mesh["meshes"]], np.int32)
    mpar = np.asarray([out_ if x[2] < 0 else x[2] for x in mesh["meshes"]], np.int32)
    Vt, cst, cnn, cmesh, chm = mesh["Vt"], mesh["cst"], mesh["cnn"], mesh["cmesh"], mesh["chm"]
    GR = E["GR"]
    return dict(lo=_cat(L, (1, 3), np.float32), hi=_cat(H, (1, 3), np.float32), skip=_cat(SK, (1,), np.int32),
                start=_cat(ST, (1,), np.int32), count=_cat(CN, (1,), np.int32), items=_cat(IT, (1,), np.int32),
                gR=(np.stack(GR) if GR else np.eye(3)[None]).astype(np.float32),
                g0=np.asarray(G0 or [0], np.int32), g1=np.asarray(G1 or [0], np.int32), vg0=vg0, vg1=vg1,
                vg1u=np.maximum(vg1u, vg0),
                parent=np.concatenate([parent, mpar]).astype(np.int32),
                cls=np.concatenate([np.asarray(V["med"], np.int32), [bg], mmed]).astype(np.int32),
                sen=np.concatenate([np.asarray(V["sen"], np.int32), np.full(1 + n_mesh, -1)]).astype(np.int32),
                kind=kind.astype(np.int32), R=Rh.astype(np.float32), t=th.astype(np.float32), coin_hi=coin_hi,
                uc=cb.astype(np.float32), uh=(hb + 1e-6 * (np.abs(cb) + hb) + 1e-9).astype(np.float32),
                V0=Vt[:, 0].astype(np.float32), V1=Vt[:, 1].astype(np.float32), V2=Vt[:, 2].astype(np.float32),
                N=mesh["nrm"].astype(np.float32), cstart=np.asarray(cst if len(cst) else [0], np.int32),
                cn=np.asarray(cnn if len(cnn) else [0], np.int32),
                cmesh=np.asarray(cmesh if cmesh else [0], np.int32),
                cmoth=np.asarray(chm if chm else [out_], np.int32), fill=fill)  # fmt: skip


def _srf_layout(tables, n, ch0, mesh):
    """Return the surface codes' layout (su, sm, LM, FM) and the triangle -> chunk table.

    Above the items (a plain item: the band rules): units su + ((item * LM + leaf) * FM + face) * 4 + 2 o + e, mesh
    triangles sm + 4 k + 2 o + e (o: the outward normal is +grad; e: the event entered the item).
    """
    LM, FM = max(int(tables.maxnl), 1), 1 + int(np.max(tables.T["np_"]))
    cst, Vt = mesh["cst"], mesh["Vt"]
    su = ch0 + len(cst)
    sm = su + n * LM * FM * 4
    if sm + 4 * len(Vt) >= SRF_LIMIT:
        raise ValueError(f"navigator: srf codes overflow int32 ({sm + 4 * len(Vt)})")
    tchunk = ((np.searchsorted(cst, np.arange(len(Vt)), side="right") - 1).clip(0).astype(np.int32)
              if len(cst) else np.zeros(1, np.int32))  # fmt: skip
    return (su, sm, LM, FM), tchunk


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


def build(scene, min_group=bvh.MIN_GROUP, gain=bvh.GROUP_GAIN, leaf_size=bvh.LEAF_SIZE, mesh_chunk=MESH_CHUNK):
    """Return the NavArrays of a lowered Scene (``lumen.build.scene``; a label's medium is the class reported)."""
    W = _walk(scene)
    V, kinds, meshes = _volumes(scene, W)
    n = len(V["kind"])
    if n == 0:
        raise ValueError("navigator: a scene without volumes")
    tables = csg_tables(kinds)
    kind, Rh, th, cb, hb, lo, hi = _volume_boxes(tables, V)
    parent = np.where(np.asarray(V["parent"]) < 0, n, np.asarray(V["parent"])).astype(np.int32)
    kids = [[] for _ in range(n + 1)]
    for i in range(n):
        kids[parent[i]].append(i)
    n_hole = _hole_links(scene, V, parent, lo, hi, kids)
    # the volumes with a daughter candidate that is not their daughter (a sibling filling a lowered hole): where that
    # candidate holds a point, the point is not theirs (fill)
    fill = np.array([any(k_ < n and parent[k_] != u for k_ in kids[u]) for u in range(n)] or [False], bool)
    coin_hi, coincident = _coincident(tables, kind, parent, Rh, th)
    mesh = _mesh_chunks(meshes, n, mesh_chunk, kids)
    mesh.update(n_mesh=len(meshes), meshes=meshes)
    E = _entries(kids, n, mesh, (Rh, hb), min_group, gain)
    trees = _trees(E, _group_boxes(E, n, mesh, (Rh, th, cb, hb)), leaf_size)
    host = _host(E, trees, (V, kind, parent, Rh, th, cb, hb, coin_hi, fill), mesh, int(scene.outside.medium))
    dmax = trees[-1]  # the deepest BVH: the traversal's stack holds one entry per level; a deeper one would drop nodes
    if dmax > STACK:
        raise ValueError(f"navigator: a BVH of depth {dmax} > the traversal stack ({STACK})")
    srf, host["tchunk"] = _srf_layout(tables, n, mesh["ch0"], mesh)
    stats = dict(volumes=n, kinds=len(kinds), meshes=len(meshes), chunks=len(mesh["cst"]), groups=len(E["GR"]),
                 framed=int(E["g_rot"].sum()), hole_links=n_hole, max_daughters=int(max(len(k_) for k_ in kids)),
                 bvh_depth=dmax)  # fmt: skip
    return NavArrays(host=host, csg=tables, n=n, n_mesh=len(meshes), chunk=int(mesh_chunk), ch0=mesh["ch0"], srf=srf,
                     bg=int(scene.outside.medium), names=tuple(V["name"]), defs=tuple(V["defn"]), stats=stats,
                     coincident=coincident, kids=tuple(tuple(k_) for k_ in kids), lo_w=lo, hi_w=hi)  # fmt: skip
