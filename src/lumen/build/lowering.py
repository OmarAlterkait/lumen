"""Lowering: a detector file's geometry to the Scene every geometry query is built from.

    loaded = lower(df, detector="auto:argon,lar", sensors=Sensors(aux=("PhotonDetector",)))
    loaded.scene      # lumen.build.scene.Scene
    loaded.media      # GDML material name -> medium id (optics are attached by medium id)
    loaded.sensors    # sensor id -> the instance's placement path (one id per world instance: scene.PATH)
    loaded.report     # what was decided: the detector volume, the sensors found and how, the normalisations

    scene = split_unions(loaded.scene, 16)                        # large unions placed as their pieces (exact)
    scene = place(base, part, into="world", R=R, t=t, prefix="frame")  # compose a part into a scene
    scale = scene_scale(scene)                                    # m

One Definition per logical volume (assemblies are member-less definitions; a reflected placement folds into a
mirrored variant of its definition), Geant4's conventions throughout:

- a physvol rotation is passive: ``p_mother = G^T p_daughter + t`` with ``G = Rz Ry Rx`` (``rot_from_angles``
  returns ``G^T``, the placement matrix); a physvol scale multiplies its columns (a negative one reflects);
- a boolean's or multi-union's rotation is active: ``p_first = G p_second + t`` (the transpose);
- a tessellated solid is an opaque ``Mesh`` member (first hit only), never silently dropped;
- an unsupported solid type is an error, unless ``unsupported="assembly"`` keeps its volume as a transparent
  assembly (its daughters stay, its own solid goes) and lists it in the approximations.

What a detector must declare (nothing else is per-detector): which volume is the detector (GDML files ship halls,
rock and ground; photons never come back from them), which volumes are sensors (GDML files mark them by an aux
SensDet tag, an EFFICIENCY surface, or only by name: the rule is declared and the report lists what it matched),
and the optics of each medium, attached by ``media``.

Notes
-----
Ported from lumen.scene.mapping, build, normalise and load (rebuild/next ec8de54c), themselves from LUCiD's gdml_v2
(geom-vec a0579a25). The input is now a ``lumen.format.schema.DetectorFile`` in metres, so the mm to m conversions
are gone; every other operation is the baseline's, in the same order.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from scipy.spatial import cKDTree

from lumen.build import scene as sc
from lumen.build import solids as so

TWOPI = 2 * np.pi
IDENT = np.eye(3)
MIRROR_Z = np.diag([1.0, 1.0, -1.0])  # the reflection a mirrored definition variant is built with
HOLE_SEARCH = 1e-6  # m: a sibling filling a hole is searched for this close to the hole's position
HOLE_POSE_TOL = 1e-9  # a sibling fills a hole when its rotation and position agree to this (position: relative, m)
HOLE_KEY_DECIMALS = 9  # a hole's position rounded to this many decimals (m) names it
BOX_TOUCH = 1e-12  # m: boxes overlapping by less than this touch, they do not overlap
SIBLING_SEARCH_PAD = 1e-9  # m: added to the sibling search radius
DIGEST_CHARS = 8  # the length of the digest naming a definition variant with holes dropped
SENSOR_AUX_SHOWN = 6  # the report lists at most this many sensor aux values
UNSUPPORTED_MODES = ("error", "assembly")
ROOT, DETECTOR = "__root__", "__detector__"
OP_OF = {"union": "union", "subtraction": "sub", "intersection": "inter"}


# ======================================================================================================== poses
def rot_from_angles(x, y, z):
    """Return the physvol (passive) matrix of GDML rotation angles (rad).

    G4GDML builds ``Rz(z) Ry(y) Rx(x)`` and places with its inverse, so this returns ``(Rz Ry Rx)^T``; booleans use
    its transpose.
    """
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (rz @ ry @ rx).T


def _rotation(angles):
    """Return the physvol matrix of angles: the identity when no rotation is given (all +0), as the baseline read it."""
    if all(a == 0.0 and math.copysign(1.0, a) > 0 for a in angles):
        return IDENT
    return rot_from_angles(*angles)


def placement_pose(p):
    """Return ``(m, t)`` of a ``schema.Placement``: the placement matrix (scale applied to its columns) and position."""
    m = _rotation(p.rotation)
    if p.scale is not None:
        m = m * np.asarray(p.scale, float)[None, :]
    return np.asarray(m, float), np.asarray(p.position, float)


def boolean_pose(tr):
    """Return ``(L, t)`` of a boolean's or multi-union's ``schema.Transform``: active, the transpose."""
    return np.asarray(_rotation(tr.rotation), float).T, np.asarray(tr.position, float)


# ================================================================================================ solids mapping
def _params(s, *keys, **defaults):
    return [s.params.get(k, defaults.get(k, 0.0)) for k in keys]


def _phi(s):
    return tuple(_params(s, "startphi", "deltaphi", deltaphi=TWOPI))


SOLIDS = {
    "box": lambda s, mp: so.g4_box(*_params(s, "x", "y", "z")),
    "tube": lambda s, mp: so.g4_tube(*_params(s, "rmin", "rmax", "z"), *_phi(s)),
    "cutTube": lambda s, mp: so.g4_cuttube(*_params(s, "rmin", "rmax", "z"), *_phi(s),
                                           _params(s, "lowx", "lowy", "lowz"), _params(s, "highx", "highy", "highz")),
    "cone": lambda s, mp: so.g4_cone(*_params(s, "rmin1", "rmax1", "rmin2", "rmax2", "z"), _phi(s)),
    "polycone": lambda s, mp: so.g4_polycone(*_phi(s), s.planes),
    "polyhedra": lambda s, mp: so.g4_polyhedra(*_phi(s), *_params(s, "numsides"), s.planes),
    "sphere": lambda s, mp: so.g4_sphere(*_params(s, "rmin", "rmax"), _phi(s),
                                         tuple(_params(s, "starttheta", "deltatheta", deltatheta=np.pi))),
    "orb": lambda s, mp: so.g4_orb(*_params(s, "r")),
    "ellipsoid": lambda s, mp: so.g4_ellipsoid(*_params(s, "ax", "by", "cz", "zcut1", "zcut2")),
    "eltube": lambda s, mp: so.g4_eltube(*_params(s, "dx", "dy", "dz")),
    "elcone": lambda s, mp: so.g4_elcone(*_params(s, "dx", "dy", "zmax", "zcut")),
    "paraboloid": lambda s, mp: so.g4_paraboloid(*_params(s, "rlo", "rhi", "dz")),
    "hype": lambda s, mp: so.g4_hype(*_params(s, "rmin", "rmax", "inst", "outst", "z")),
    "torus": lambda s, mp: so.g4_torus(*_params(s, "rmin", "rmax", "rtor"), *_phi(s)),
    "trd": lambda s, mp: so.g4_trd(*_params(s, "x1", "x2", "y1", "y2", "z")),
    "para": lambda s, mp: so.g4_para(*_params(s, "x", "y", "z", "alpha", "theta", "phi")),
    "trap": lambda s, mp: so.g4_trap(*_params(s, "z", "theta", "phi"), _params(s, "y1", "x1", "x2", "alpha1"),
                                     _params(s, "y2", "x3", "x4", "alpha2")),
    "arb8": lambda s, mp: so.g4_arb8([tuple(_params(s, f"v{i}x", f"v{i}y")) for i in range(1, 9)],
                                     *_params(s, "dz")),
    # the importer applied G4GDMLReadSolids::TetRead's second lunit scaling to the vertices
    "tet": lambda s, mp: so.g4_tet(*s.vertices),
    "xtru": lambda s, mp: so.g4_xtru(list(s.vertices), list(s.sections)),
    "tessellated": lambda s, mp: sc.Mesh(np.array(mp.df.meshes[s.mesh], float)),
}  # fmt: skip


def _balanced_union(nodes):
    """Return a balanced union tree (depth log2 n) of the nodes, each union GDML-level (SK unites ~7000 PMTs)."""
    while len(nodes) > 1:
        nxt = []
        for i in range(0, len(nodes) - 1, 2):
            nxt.append(so.Op("union", nodes[i], nodes[i + 1]))
            nxt[-1].g4bool = True
        nodes = nxt + nodes[len(nodes) - len(nodes) % 2 :]
    return nodes[0]


class SolidMapper:
    """GDML solids of a detector file to CSG trees (``lumen.build.solids``) with Geant4's semantics.

    An unsupported solid type raises NotImplementedError (the lowering decides what that means) and is counted in
    ``unsupported``. A GDML-level boolean is marked ``g4bool`` (decomposable by ``split_unions``), unlike the booleans
    inside a constructor (a tube's phi wedge).
    """

    def __init__(self, df):
        self.df = df
        self.unsupported = {}

    def solid(self, name):
        """Return the CSG tree (or Mesh) of a solid."""
        s = self.df.solids[name]
        if s.type in OP_OF:
            (m1, t1), (m2, t2) = (boolean_pose(tr) for tr in s.transforms)
            na = so.Xf(m1, t1, self.solid(s.refs[0]))
            nb = so.Xf(m2, t2, self.solid(s.refs[1]))
            out = so.Op(OP_OF[s.type], na, nb)
            out.g4bool = True  # a GDML-level boolean (decomposable), not a constructor internal
            return out
        if s.type == "multiUnion":
            nodes = [so.Xf(*boolean_pose(tr), self.solid(r)) for r, tr in zip(s.refs, s.transforms, strict=True)]
            return _balanced_union(nodes)
        if s.type == "scaledSolid":
            return so.Xf(np.asarray(np.diag(np.asarray(s.scale, float)), float), np.zeros(3), self.solid(s.refs[0]))
        build = SOLIDS.get(s.type)
        if build is None:
            self.unsupported[s.type] = self.unsupported.get(s.type, 0) + 1
            raise NotImplementedError(s.type)
        return build(s, self)

    def holes(self, name, match):
        """Return the holes of a subtraction spine ``((A - B1) - B2) ...``.

        Each subtrahend is split through its unions and multi-unions until a part satisfies ``match(ref, L, t)`` or is
        no union. Returns ``[(path, ref, L, t, matched)]`` in the solid's frame (``x = L x_hole + t``, m); ``path``
        names the part for ``solid_without``. Empty when the solid is no subtraction.

        Notation: L a linear map (``x = L x_hole + t``).
        """
        out = []

        def sub(nm, lin, t, path):
            s = self.df.solids[nm]
            if match(nm, lin, t):
                out.append((path, nm, lin, t, True))
            elif s.type in ("union", "multiUnion"):
                for k, (r, tr) in enumerate(zip(s.refs, s.transforms, strict=True)):
                    m, tt = boolean_pose(tr)
                    sub(r, lin @ m, lin @ tt + t, path + (k,))
            else:
                out.append((path, nm, lin, t, False))

        def spine(nm, lin, t, path):
            s = self.df.solids[nm]
            if s.type != "subtraction":
                return
            (m1, t1), (m2, t2) = (boolean_pose(tr) for tr in s.transforms)
            spine(s.refs[0], lin @ m1, lin @ t1 + t, path + (0,))
            sub(s.refs[1], lin @ m2, lin @ t2 + t, path + (1,))

        spine(name, np.eye(3), np.zeros(3), ())
        return out

    def solid_without(self, name, drop):
        """Return ``solid(name)`` with the hole parts whose paths (``holes``) are in ``drop`` left out.

        A union with nothing left goes, a subtrahend with nothing left goes; the minuend stays.
        """
        pre = {d[:k] for d in drop for k in range(len(d) + 1)}

        def sub(nm, path):  # -> node, or None when everything is dropped
            if path in drop:
                return None
            s = self.df.solids[nm]
            if s.type in ("union", "multiUnion") and path in pre:
                kids = []
                for k, (r, tr) in enumerate(zip(s.refs, s.transforms, strict=True)):
                    c = sub(r, path + (k,))
                    if c is not None:
                        kids.append(so.Xf(*boolean_pose(tr), c))
                return _balanced_union(kids) if kids else None
            return self.solid(nm)

        def spine(nm, path):
            s = self.df.solids[nm]
            if s.type != "subtraction":
                return self.solid(nm)
            (m1, t1), (m2, t2) = (boolean_pose(tr) for tr in s.transforms)
            a = so.Xf(m1, t1, spine(s.refs[0], path + (0,)))
            b = sub(s.refs[1], path + (1,))
            if b is None:
                return a
            out = so.Op("sub", a, so.Xf(m2, t2, b))
            out.g4bool = True
            return out

        return spine(name, ())


# ============================================================================================ the detector volume
class DetectorVolume(NamedTuple):
    """The detector volume: its LV, the world pose of its first placement, its mother LV and physvol name."""

    lv: str
    R: np.ndarray
    t: np.ndarray
    mother: str | None
    placement: str | None


def _first_paths(df):
    """LV -> the first placement path from the world [(lv, L, t, physvol name)], depth-first."""
    paths = {}

    def walk(lv, lin, t, path, pv=None):
        if lv in paths:
            return
        paths[lv] = path + [(lv, lin, t, pv)]
        for p in df.volumes[lv].daughters:
            pl = df.placements[p]
            lc, tc = placement_pose(pl)
            walk(pl.volume, lin @ lc, lin @ tc + t, paths[lv], pl.name)

    walk(df.world, np.eye(3), np.zeros(3), [])
    return paths


def _auto_detector(df, paths, materials, mapper):
    """Return the outermost volume of the main bath of a material.

    The largest volume (by its solid's box) of a matching material, then up its placement chain while the mother is
    of a matching material.
    """
    rx = re.compile("|".join(re.escape(a) for a in materials.split(",")), re.I)  # comma: alternatives
    matching = lambda lv: bool(df.volumes[lv].material) and rx.search(df.volumes[lv].material) is not None  # noqa: E731 -- a one-line predicate
    hits = [lv for lv in paths if matching(lv)]
    if not hits:
        raise ValueError(f"gdml: no volume of material matching {materials!r}")
    # volumes of that material elsewhere (a pipe, a second vessel) do not pull the choice up to the world
    vol = {}
    for h in hits:
        v = df.volumes[h]
        try:
            bb = so.node_bbox(mapper.solid(v.solid)) if v.solid else None
        except NotImplementedError:
            bb = None
        vol[h] = float(np.prod(bb[1] - bb[0])) if bb is not None else 0.0
    lv = max(hits, key=lambda h: vol[h])
    chain = [e[0] for e in paths[lv]]
    k = len(chain) - 1
    while k > 0 and matching(chain[k - 1]):
        k -= 1
    return chain[k]


def find_detector(df, spec, mapper=None):
    """Return the detector volume to keep (``DetectorVolume``).

    ``spec`` is an LV name (or the start of one), or ``"auto:<names>"``: the outermost volume of the main bath of a
    material (one of the comma-separated names, a case-insensitive substring, e.g. ``"auto:argon,lar"``).
    """
    paths = _first_paths(df)
    if spec.startswith("auto:"):
        lv = _auto_detector(df, paths, spec[len("auto:") :], mapper or SolidMapper(df))
    else:
        lv = spec
        if lv not in paths:
            cand = [n for n in paths if n.startswith(spec)]
            if not cand:
                raise ValueError(f"gdml: no volume {spec!r}")
            lv = cand[0]
    chain = paths[lv]
    _, rot, t, pv = chain[-1]
    mother = chain[-2][0] if len(chain) > 1 else None
    return DetectorVolume(lv, rot, t, mother, pv)


# ==================================================================================================== lowering
def flat_aux(volume):
    """Return a volume's auxiliary (type, value) pairs as the baseline read them: each tag and its direct children."""
    out = []
    for a in volume.aux:
        out.append((a.type, a.value))
        out += [(c.type, c.value) for c in a.children]
    return out


@dataclass
class _Context:
    """The definition being built: its LV and Definition, whether mirrored, and its sibling-hole lowering."""

    lv: str
    d: sc.Definition
    mirrored: bool
    dropped: dict
    held: set


class _Lowering:
    """The state of one lowering (see ``lower``): definitions, media, and what was decided."""

    def __init__(self, df, options):
        self.df, self.opt = df, options
        self.mp = SolidMapper(df)
        self.mats, self.unsupported, self.defs = {}, {}, {}
        self.lowered, self.overlays = [], set()
        self.aliases, self.dissolved, self.has_holes = {}, {}, {}
        self.contained, self.unproven, self.overlapping = {}, {}, {}  # dissolve: proven inside; kept envelopes
        self.sib, self.lv_box_ = {}, {}
        self.skin_lvs = {s.volume for s in df.skins}
        self.border_of = {}  # physvol name -> the physvol names it shares a border with
        for b in df.borders:
            self.border_of.setdefault(b.pv1, set()).add(b.pv2)
            self.border_of.setdefault(b.pv2, set()).add(b.pv1)
        self.placed_as = {}  # LV -> the physvol names placing it
        for p in df.placements:
            self.placed_as.setdefault(p.volume, set()).add(p.name)
        self._pose = {}

    # ---- helpers
    def mid(self, name):
        """Return the medium id of a material, numbered in first-use order from 1."""
        return self.mats.setdefault(name, len(self.mats) + 1)

    def pose(self, p):
        """Return the (m, t) of placement index p."""
        if p not in self._pose:
            self._pose[p] = placement_pose(self.df.placements[p])
        return self._pose[p]

    def vol(self, lv):
        """Return the schema.Volume of an LV."""
        return self.df.volumes[lv]

    def _bbox_of(self, solid):
        nd = self.mp.solid(solid)
        if isinstance(nd, sc.Mesh):
            pts = nd.tris.reshape(-1, 3)
            return pts.min(0), pts.max(0)
        return so.node_bbox(nd)

    # ---- dissolve: proofs
    def inside_all(self, c):
        """Return whether every daughter of volume c is proven inside c's solid (by its solid's box)."""
        try:
            env = self.mp.solid(c.solid)
        except NotImplementedError:
            return False
        if isinstance(env, sc.Mesh):
            return False

        def ok(q, lin, t):
            if q.assembly or not q.solid:
                return all(ok(self.vol(self.df.placements[k].volume), lin @ self.pose(k)[0], lin @ self.pose(k)[1] + t)
                           for k in q.daughters)  # fmt: skip
            try:
                bb = self._bbox_of(q.solid)
            except NotImplementedError:
                return False
            if bb is None:
                return False
            return so.box_inside(env, so.corners(*bb) @ lin.T + t)

        return all(ok(self.vol(self.df.placements[k].volume), *self.pose(k)) for k in c.daughters)

    def lv_box(self, lv):
        """Return the box (lo, hi) of an LV's own solid, or of its daughters for an assembly (m; None: unbounded)."""
        if lv in self.lv_box_:
            return self.lv_box_[lv]
        v = self.vol(lv)
        bb = None
        if v.assembly or not v.solid:
            los, his = [], []
            for k in v.daughters:
                b = self.lv_box(self.df.placements[k].volume)
                if b is None:
                    self.lv_box_[lv] = None
                    return None
                m, t = self.pose(k)
                c = so.corners(*b) @ m.T + t
                los.append(c.min(0))
                his.append(c.max(0))
            bb = (np.min(los, 0), np.max(his, 0)) if los else None
        else:
            try:
                bb = self._bbox_of(v.solid)
            except NotImplementedError:
                bb = None
        self.lv_box_[lv] = bb
        return bb

    def _siblings(self, mother):
        if mother not in self.sib:
            pls = list(self.vol(mother).daughters)
            boxes = np.full((len(pls), 2, 3), np.nan)
            for k, q in enumerate(pls):
                b = self.lv_box(self.df.placements[q].volume)
                if b is not None:
                    m, t = self.pose(q)
                    c = so.corners(*b) @ m.T + t
                    boxes[k] = c.min(0), c.max(0)
            fin = np.all(np.isfinite(boxes.reshape(len(pls), -1)), 1)
            cen, rad = 0.5 * (boxes[:, 0] + boxes[:, 1]), 0.5 * np.linalg.norm(boxes[:, 1] - boxes[:, 0], axis=1)
            self.sib[mother] = (pls, boxes, fin, cKDTree(np.where(fin[:, None], cen, 0.0)), np.where(fin, rad, 0.0),
                                {q: k for k, q in enumerate(pls)})  # fmt: skip
        return self.sib[mother]

    def disjoint_from_siblings(self, mother, p):
        """Return whether placement p (in LV mother) is proven to overlap none of its siblings.

        Notation: B the siblings' boxes (n, 2, 3); Li, Lj placement matrices.
        """
        pls, B, fin, kd, rad, at = self._siblings(mother)
        i = at[p]
        if not fin[i] or not np.all(fin):
            return False  # an unbounded sibling (or this one): not provable
        cand = kd.query_ball_point(0.5 * (B[i, 0] + B[i, 1]), rad[i] + rad.max() + SIBLING_SEARCH_PAD)
        Li, ti = self.pose(p)
        child = self.df.placements[p].volume
        env = self.mp.solid(self.vol(child).solid)
        for j in cand:
            if j == i or not (np.all(B[i, 0] < B[j, 1] - BOX_TOUCH) and np.all(B[j, 0] < B[i, 1] - BOX_TOUCH)):
                continue  # boxes apart (touching faces included)
            q = pls[j]
            Lj, tj = self.pose(q)
            cj = so.corners(*self.lv_box(self.df.placements[q].volume)) @ Lj.T + tj  # sibling j's box, mother frame
            if not isinstance(env, sc.Mesh) and so.box_outside(env, (cj - ti) @ Li):
                continue
            if not self._sibling_misses(q, child, Li, ti):
                return False
        return True

    def _sibling_misses(self, q, child, Li, ti):  # noqa: N803 -- a placement matrix, as in the caller
        """Return whether sibling placement q is proven to miss the box of ``child`` placed at (Li, ti).

        Notation: Li, Lj placement matrices.
        """
        s = self.vol(self.df.placements[q].volume)
        if not (s.solid and not s.assembly):
            return False
        try:
            nj = self.mp.solid(s.solid)
        except NotImplementedError:
            return False
        Lj, tj = self.pose(q)
        ci = so.corners(*self.lv_box(child)) @ Li.T + ti
        return not isinstance(nj, sc.Mesh) and so.box_outside(nj, (ci - tj) @ Lj)

    def dissolvable(self, mother, p, gmother=None):
        """Return whether placement p (in LV mother) is a navigation envelope (``lower``'s docstring)."""
        if not self.opt.dissolve:
            return False
        c, m = self.vol(self.df.placements[p].volume), self.vol(mother)
        if c.assembly or not c.solid or not c.material or c.material != m.material or c.name in self.skin_lvs:
            return False
        if (self.opt.keep is not None and self.opt.keep(c.name)) or any(t == "SensDet" for t, _ in flat_aux(c)):
            return False
        if c.name not in self.contained:
            self.contained[c.name] = self.inside_all(c)
        if not self.contained[c.name]:
            self.unproven[c.name] = self.unproven.get(c.name, 0) + 1
            return False
        if not self.disjoint_from_siblings(gmother or mother, p):  # its siblings: those of its own GDML mother
            self.overlapping[c.name] = self.overlapping.get(c.name, 0) + 1
            return False
        if c.name not in self.has_holes:  # holes lowered among its daughters: that ordering is its own
            self.has_holes[c.name] = bool(self.lowering(c, False, record=False)[0])
        if self.has_holes[c.name]:
            return False
        return self._borders_allow(mother, p, c)

    def _borders_allow(self, mother, p, c):
        partners = self.border_of.get(self.df.placements[p].name, set())
        if partners:
            mine = self.placed_as.get(mother, set())
            kids = {self.df.placements[k].name for k in c.daughters}
            if partners & mine or len(mine) != 1 or not partners <= kids:
                return False
        return True

    # ---- sibling holes
    def _sibling_holes(self, v, i, pose, by):
        """Return the holes of local placement i's subtraction that are sibling placements (dropped, filled by, n)."""
        lp, tp = pose[i]
        hit = {}

        def match(ref, lh, th):
            lv_, tv = lp @ lh, lp @ th + tp
            if ref not in by:
                return False
            js, kd = by[ref]
            for q in kd.query_ball_point(tv, HOLE_SEARCH):
                j = js[q]
                lj, tj = pose[j]
                if j != i and np.abs(lj - lv_).max() <= HOLE_POSE_TOL and \
                        np.abs(tj - tv).max() <= HOLE_POSE_TOL * max(1.0, np.abs(tv).max()):  # fmt: skip
                    hit[(ref, tuple(np.round(tv, HOLE_KEY_DECIMALS)))] = j
                    return True
            return False

        cs = self.vol(self.df.placements[v.daughters[i]].volume).solid
        hs = self.mp.holes(cs, match)
        return frozenset(h[0] for h in hs if h[4]), hit, len(hs)

    def lowering(self, v, mirrored, record=True):
        """Return ``(dropped, order, held)`` of a volume's placements.

        Per local placement the holes dropped, the placement order (subtracted volumes before their siblings) and the
        siblings that fill a hole (never dissolved).
        """
        pls = list(range(len(v.daughters)))
        if not self.opt.lower_holes or mirrored:
            return {}, pls, set()
        pose = {i: self.pose(v.daughters[i]) for i in pls}
        by = {}
        for j in pls:
            cs = self.vol(self.df.placements[v.daughters[j]].volume).solid
            if cs and np.linalg.det(pose[j][0]) > 0:
                by.setdefault(cs, []).append(j)
        by = {cs: (js, cKDTree(np.array([pose[j][1] for j in js]))) for cs, js in by.items()}
        drop, first, held = {}, {}, set()
        for i in pls:
            child = self.df.placements[v.daughters[i]].volume
            cs = self.vol(child).solid
            if not cs or self.df.solids[cs].type != "subtraction" or np.linalg.det(pose[i][0]) < 0:
                continue
            d, hit, n_holes = self._sibling_holes(v, i, pose, by)
            if d:
                drop[i], first[i] = d, min(hit.values())
                held.update(hit.values())  # siblings that fill a hole: never dissolved
                if record:
                    self.lowered.append((v.name, child, len(d), n_holes))
                    self.overlays.update(self.df.placements[v.daughters[j]].volume for j in hit.values())
        order = list(pls)
        for i in sorted(drop):  # only if a sibling it holds precedes it: just before the first one
            if first[i] < i:
                order.remove(i)
                order.insert(order.index(first[i]), i)
        return drop, order, held

    # ---- definitions
    def _member(self, v, lv, mirrored, drop):
        try:
            s = self.mp.solid_without(v.solid, drop) if drop else self.mp.solid(v.solid)
        except NotImplementedError as e:
            if self.opt.unsupported == "error":
                msg = (
                    f"gdml: solid {v.solid!r} of volume {lv!r}: type {e} is not supported (lower(..., "
                    "unsupported='assembly') keeps the LV as an assembly)"
                )
                raise NotImplementedError(msg) from None
            self.unsupported[str(e)] = self.unsupported.get(str(e), 0) + 1
            return None
        if isinstance(s, sc.Mesh):  # an opaque free-form part; mirrored: reflect and keep the winding
            s = sc.Mesh((s.tris @ MIRROR_Z.T)[:, ::-1]) if mirrored else s
        else:
            s = so.Xf(MIRROR_Z, np.zeros(3), s) if mirrored else s
        return sc.Member(s, sc.Label(self.mid(v.material), -1, lv), 0)

    def definition(self, lv, mirrored=False, drop=None, key=None):
        """Return the key of the Definition of an LV (built on first use): mirrored, or with holes dropped."""
        key = key or lv + ("|mirror" if mirrored else "")
        if key in self.defs:
            return key
        v = self.vol(lv)
        d = sc.Definition(key)
        self.defs[key] = d
        if not v.assembly and v.solid:
            m = self._member(v, lv, mirrored, drop)
            if m is not None:
                d.members.append(m)
        dropped, order, held = self.lowering(v, mirrored)
        ctx = _Context(lv, d, mirrored, dropped, held)
        for i in order:
            p = v.daughters[i]
            self.emit(ctx, p, *self.pose(p), i)
        return key

    def place(self, ctx, p, lin, t, i=None, group=""):
        """Place placement p in the definition being built, at (lin, t)."""
        pl = self.df.placements[p]
        if ctx.mirrored:
            lin, t = MIRROR_Z @ lin, MIRROR_Z @ t
        refl = np.linalg.det(lin) < 0
        if i is not None and i in ctx.dropped:
            tag = hashlib.sha1(repr(sorted(ctx.dropped[i])).encode()).hexdigest()[:DIGEST_CHARS]  # one per drop set
            child = self.definition(pl.volume, drop=ctx.dropped[i], key=f"{pl.volume}|holes:{tag}")
            self.overlays.add(child)
        else:
            child = self.definition(pl.volume, mirrored=refl)
        rot = lin @ MIRROR_Z if refl else lin
        ctx.d.placements.append(sc.Placement(child, rot, t, name=pl.name or "", group=group))

    def emit(self, ctx, p, lin, t, i=None, group="", gm=None):  # noqa: PLR0913 -- the placement and its pose
        """Place placement p, or, when it is a navigation envelope, its daughters in its place (recursively)."""
        free = i is None or (i not in ctx.dropped and i not in ctx.held)
        if free and not ctx.mirrored and np.linalg.det(lin) > 0 and self.dissolvable(ctx.lv, p, gm):
            pl = self.df.placements[p]
            c = self.vol(pl.volume)
            if self.border_of.get(pl.name):
                self.aliases[pl.name] = next(iter(self.placed_as[ctx.lv]))
            self.dissolved[c.name] = self.dissolved.get(c.name, 0) + 1
            g = f"{group}/{c.name}@{self.dissolved[c.name]}"  # this envelope instance: its daughters share it
            for q in c.daughters:
                lq, tq = self.pose(q)
                self.emit(ctx, q, lin @ lq, lin @ tq + t, group=g, gm=c.name)
            return
        self.place(ctx, p, lin, t, i, group)

    def notes(self):
        """Return the approximations and decisions, in words."""
        return ([f"gdml (exact): {c} placed in {m or '?'}: {n} of {nh} holes are sibling placements -- left out of "
                 "the solid, the siblings outrank it" for m, c, n, nh in self.lowered] +
                [f"gdml (exact): dissolved {n} placements of the envelope {c} (its mother's material, no surface)"
                 for c, n in self.dissolved.items()] +
                [f"gdml: envelope {c} kept (x{n}): a daughter is not proven inside it"
                 for c, n in self.unproven.items()] +
                [f"gdml: envelope {c} kept (x{n}): not proven disjoint from its siblings"
                 for c, n in self.overlapping.items()] +
                [f"gdml: unsupported, LV kept as a transparent assembly: {k} x{v}"
                 for k, v in self.unsupported.items()])  # fmt: skip

    def run(self):
        """Lower the geometry; return the Scene."""
        self.defs[ROOT] = sc.Definition(ROOT)
        if self.opt.detector is not None:
            det = find_detector(self.df, self.opt.detector, mapper=self.mp)
            dk = self.definition(det.lv)
            # the placement keeps its GDML physvol name: border surfaces on the detector volume name it
            top = sc.Placement(dk, det.R, det.t, name=det.placement or det.lv)
            self.defs[DETECTOR] = sc.Definition(DETECTOR, placements=[top])
            mother = det.mother
            out_med = self.mid(self.vol(mother).material) if mother is not None and self.vol(mother).material else 0
            notes = [f"gdml: kept only the detector volume {det.lv} (outside: {mother})"] + self.notes()
            return sc.Scene(self.defs, DETECTOR, sc.Label(out_med, -1, mother or "World"), approximations=notes,
                            aliases=dict(self.aliases), overlays=sorted(self.overlays))  # fmt: skip
        wk = self.definition(self.df.world)
        if not self.opt.part:
            self.defs[wk].members = []  # the world solid is the outside
        return sc.Scene(self.defs, wk, sc.Label(0, -1, "World"), approximations=self.notes(),
                        aliases=dict(self.aliases), overlays=sorted(self.overlays))  # fmt: skip


# ======================================================================================================= loading
@dataclass
class Sensors:
    """The sensor rule: an LV is a sensor when it carries one of these aux SensDet values, or its name matches.

    Every world instance of a sensor volume gets its own id (``scene.PATH``, the canonical order).
    """

    aux: tuple[str, ...] = ("PhotonDetector",)
    names: str | None = None


@dataclass
class Options:
    """The lowering's choices (``lower``)."""

    detector: str | None = None
    part: bool = False
    lower_holes: bool = True
    dissolve: bool = False
    keep: object = None
    unsupported: str = "error"


@dataclass
class Loaded:
    """A lowered detector: the scene, material name -> medium id, sensor id -> instance path, and the report."""

    scene: sc.Scene
    media: dict[str, int]
    sensors: dict[int, str] = field(default_factory=dict)
    report: list[str] = field(default_factory=list)
    detector_file: object = None

    def __str__(self):
        """Return the report, one line each."""
        return "\n".join(self.report)


def lower_scene(df, options):
    """Return ``(scene, media, unsupported, mapper)`` of a detector file under ``Options`` (no sensors, no report)."""
    if options.unsupported not in UNSUPPORTED_MODES:
        raise ValueError(f"unsupported={options.unsupported!r} (one of {UNSUPPORTED_MODES})")
    low = _Lowering(df, options)
    scene = low.run()
    return scene, low.mats, low.unsupported, low.mp


def lower(df, detector=None, sensors=None, **options):
    """Return the ``Loaded`` scene of a detector file (``lumen.format.schema.DetectorFile``).

    Parameters
    ----------
    df : DetectorFile
        The full GDML world (``lumen.gdml.importer.read_gdml``).
    detector : str, optional
        None (the whole world), an LV name, or ``"auto:<material names>"`` (``find_detector``): keep only that
        volume and its contents, at its world placement; its outside is its mother's material (photons never return
        from beyond the detector: the hall, rock, building and ground are dropped).
    sensors : Sensors, optional
        The sensor rule (None: no sensors). A sensor volume is never dissolved.
    **options
        ``part`` (the file is one object whose world volume is the part, for ``place``: the world keeps its own
        solid), ``lower_holes``, ``dissolve``, ``keep`` (``keep(lv)``: never dissolve it; default the sensor name
        rule) and ``unsupported`` ("error" or "assembly").

    Notes
    -----
    ``lower_holes``: a hole of a subtraction (A - B, B split through its unions and multi-unions) that is a sibling
    placement -- the same solid at the same pose in the same mother (WCSim's black sheets: the PMTs cut out of the
    sheet they sit in) -- is left out of A, and the subtracted volume is placed before those siblings, which then
    outrank it (Geant4's locate: later siblings first). Exact: every point of the hole lies in the sibling, which
    wins it either way; a hole matching no sibling stays subtracted.

    ``dissolve``: a placed volume of its mother's material with no skin surface, no border surface with its mother,
    no holes, not kept (sensors), every daughter proven inside it (``solids.box_inside`` on the daughter's box:
    Geant4 never sees the part of a daughter outside its mother, so a protruding daughter -- MicroBooNE's wires out
    of their plane -- would change the detector if lifted) and proven disjoint from its siblings (where siblings
    overlap, Geant4's locate picks one -- MicroBooNE's volTPCActive hides the wire planes it overlaps -- and
    dissolving the one that wins would expose the other) is a Geant4 navigation envelope (a PMT mask's virtual,
    water in water). Its daughters are placed in the mother directly (poses composed, in its place in the sibling
    order) and its region is the mother's, which it already was optically. Border surfaces naming it name the
    mother's placement instead (``Scene.aliases``; only when the mother has one placement name and the borders are
    with its own daughters). Both are listed in the approximations (as exact).
    """
    opt = Options(detector=detector, **options)
    if opt.keep is None and sensors is not None and sensors.names:  # a sensor volume is never dissolved
        rx_ = re.compile(sensors.names)
        opt.keep = lambda lv: rx_.search(lv) is not None
    scene, mats, _, _ = lower_scene(df, opt)
    source = df.provenance.get("source", {}).get("file", df.name)
    rep = [f"gdml {source}: {len(scene.definitions)} definitions, {len(mats)} materials"]
    rep += [f"  {a}" for a in df.report]
    rep += [f"  {a}" for a in scene.approximations]
    n_mesh = sum(isinstance(m.solid, sc.Mesh) for d in scene.definitions.values() for m in d.members)
    if n_mesh:
        rep.append(f"  {n_mesh} tessellated solids as opaque meshes (first hit only)")
    sen = {}
    if sensors is not None:
        scene, sen, line = _sensors(df, scene, sensors)
        rep.append(line)
    rep.append(f"  scene scale {scene_scale(scene):.3g} m")
    return Loaded(scene, dict(mats), sen, rep, df)


def _sensors(df, scene, sensors):
    aux = {lv for lv, v in df.volumes.items() if any(t == "SensDet" and x in sensors.aux for t, x in flat_aux(v))}
    rx = re.compile(sensors.names) if sensors.names else None
    scene, sen = assign_sensors(scene, lambda lv: lv in aux or (rx is not None and rx.search(lv) is not None))
    vols = {v.rsplit("/", 1)[-1] for v in sen.values()}
    shown = f"{sorted(aux)[:SENSOR_AUX_SHOWN]}{'...' if len(aux) > SENSOR_AUX_SHOWN else ''}"
    return scene, sen, (f"  sensors: {len(sen)} instances (ids by world-instance path) of {len(vols)} volumes "
                        f"(aux {shown}, names {sensors.names!r})")  # fmt: skip


def assign_sensors(scene, is_sensor):
    """Mark the sensor members (``is_sensor(volume name)``) with ``scene.PATH``: one id per world instance.

    Ids are numbered in the canonical depth-first order, unique even inside repeated parents. An opaque mesh is
    never a sensor: its first hit only is reported, not what lies behind a cathode.

    Returns
    -------
    tuple
        (the new Scene, {sensor id: instance path name}).
    """

    def relabel(m):
        if is_sensor(m.label.volume) and not isinstance(m.solid, sc.Mesh):
            return sc.Member(m.solid, sc.Label(m.label.medium, sc.PATH, m.label.volume), m.rank)
        return m

    defs = {k: sc.Definition(d.name, members=[relabel(m) for m in d.members], placements=list(d.placements))
            for k, d in scene.definitions.items()}  # fmt: skip
    out = {}

    def walk(d, path):
        for m in sorted(defs[d].members, key=lambda m_: m_.rank):
            if m.label.sensor == sc.PATH:
                out[len(out)] = "/".join(path + (m.label.volume,))
        for k, pl in enumerate(defs[d].placements):
            walk(pl.definition, path + (pl.name or f"{pl.definition}#{k}",))

    walk(scene.root, ())
    return dataclasses.replace(scene, definitions=defs, approximations=list(scene.approximations)), out


def place(base, part, into, pose=None, prefix="part", clip=True, sensor=-1):  # noqa: PLR0913 -- the composition's choices
    """Return ``base`` with the root definition of ``part`` (all its definitions, names prefixed) placed in ``into``.

    ``pose`` is (R, t) (None: the identity). A Mesh part is never clipped (``scene.Mesh``); ``clip`` True makes a CSG
    part a Geant4 daughter of ``into`` (clipped to it), False an overlay.
    """
    rot, t = (None, None) if pose is None else pose
    defs = dict(base.definitions)
    ren = {k: f"{prefix}/{k}" for k in part.definitions}
    for k, d in part.definitions.items():
        defs[ren[k]] = sc.Definition(ren[k], members=list(d.members),
                                     placements=[sc.Placement(ren[p.definition], p.R, p.t, p.sensor, p.name, p.clip,
                                                              p.group) for p in d.placements])  # fmt: skip
    host = defs[into]
    defs[into] = sc.Definition(host.name, members=list(host.members), placements=list(host.placements) + [
        sc.Placement(ren[part.root], np.eye(3) if rot is None else np.asarray(rot, float),
                     np.zeros(3) if t is None else np.asarray(t, float), sensor, prefix, clip)])  # fmt: skip
    return dataclasses.replace(base, definitions=defs, approximations=list(base.approximations) +
                               [f"{prefix}: {a}" for a in part.approximations])  # fmt: skip


def world_boxes(scene):
    """Return the world boxes (lo, hi) of every member instance (CSG solids by their tree box, meshes by vertices)."""
    out = []

    def walk(d, rw, tw):
        for m in scene.definitions[d].members:
            if isinstance(m.solid, sc.Mesh):
                pts = m.solid.tris.reshape(-1, 3) @ rw.T + tw
                out.append((pts.min(0), pts.max(0)))
                continue
            bb = so.node_bbox(m.solid)
            if bb is None or not np.all(np.isfinite(bb[0])) or not np.all(np.isfinite(bb[1])):
                continue
            cs = so.corners(*bb) @ rw.T + tw
            out.append((cs.min(0), cs.max(0)))
        for pl in scene.definitions[d].placements:
            walk(pl.definition, rw @ np.asarray(pl.R, float), rw @ np.asarray(pl.t, float) + tw)

    walk(scene.root, np.eye(3), np.zeros(3))
    return out


def scene_scale(scene):
    """Return the scene scale (m): the largest extent of the scene's finite member boxes.

    The transport's epsilon policy reads it (push = max(1e-5, 2.5e-6 scale) m).
    """
    bx = world_boxes(scene)
    lo = np.min([b[0] for b in bx], 0)
    hi = np.max([b[1] for b in bx], 0)
    return float(np.max(hi - lo))


# ============================================================================================ exact normalisation
def _is_identity(lin, t):
    return np.allclose(lin, np.eye(3)) and np.allclose(t, 0)


def _wrap(lin, t, node):
    return node if _is_identity(lin, t) else so.Xf(lin, t, node)


def _sub_spine(node, lin, t):
    """Return (base, [subtrahends]) of a chain of GDML-level subtractions, transforms composed."""
    if isinstance(node, so.Xf):
        return _sub_spine(node.child, lin @ node.L, lin @ node.t + t)
    if isinstance(node, so.Op) and node.op == "sub" and getattr(node, "g4bool", False):
        base, subs = _sub_spine(node.a, lin, t)
        return base, subs + [_wrap(lin, t, node.b)]
    return _wrap(lin, t, node), []


def decompose(node, lin=None, t=None):
    """Return the pieces of a solid's GDML-level booleans: unions split, ``(U P_i) - S`` distributed (exact).

    Constructor-internal booleans (a tube's phi wedge) are never split (they carry no ``g4bool`` mark).
    """
    lin = np.eye(3) if lin is None else lin
    t = np.zeros(3) if t is None else t
    if isinstance(node, so.Xf):
        return decompose(node.child, lin @ node.L, lin @ node.t + t)
    if isinstance(node, so.Op) and getattr(node, "g4bool", False):
        if node.op == "union":
            return decompose(node.a, lin, t) + decompose(node.b, lin, t)
        if node.op == "sub":
            base, subs = _sub_spine(node, lin, t)
            out = []
            for pc in decompose(base):
                for b in subs:
                    pc = so.Op("sub", pc, b)
                out.append(pc)
            return out
    return [_wrap(lin, t, node)]


def n_leaves(node):
    """Return the number of leaves of a CSG tree."""
    return len(so.flatten(node)[1])


def split_unions(scene, min_leaves=16):
    """Return a Scene where each large union solid is a member-less definition placing its pieces.

    A large union solid is one member, no daughters, no sensor, ``min_leaves`` leaves or more (a CPA frame of ~150
    unioned tubes and boxes minus a few boxes): evaluated whole it costs every ray meeting its box a pass over all
    its leaves; placed as its pieces (``decompose``), each piece is bounded and evaluated alone. Each piece is a
    definition of one member with the same label: the region and every label are unchanged (a crossing between two
    pieces is one of the same label, a point in two overlapping pieces is that label either way). Listed in the
    approximations (as exact).
    """
    defs, n_split, n_pc = dict(scene.definitions), 0, 0
    for name, d in list(scene.definitions.items()):
        if len(d.members) != 1 or d.placements:
            continue
        m = d.members[0]
        if isinstance(m.solid, sc.Mesh) or int(m.label.sensor) != -1 or n_leaves(m.solid) < min_leaves:
            continue
        pcs = decompose(m.solid)
        if len(pcs) < 2:  # a split needs two pieces
            continue
        pl = []
        for k, pc in enumerate(pcs):
            pn = f"{name}|pc{k}"
            defs[pn] = sc.Definition(pn, [sc.Member(pc, copy.copy(m.label), m.rank)], [])
            pl.append(sc.Placement(pn, name=f"pc{k}"))
        defs[name] = sc.Definition(name, [], pl)
        n_split += 1
        n_pc += len(pcs)
    if not n_split:
        return scene
    return sc.Scene(defs, scene.root, scene.outside, list(scene.approximations) +
                    [f"split_unions (exact): {n_split} union solids placed as {n_pc} pieces"],
                    list(scene.overlays), dict(scene.aliases))  # fmt: skip
