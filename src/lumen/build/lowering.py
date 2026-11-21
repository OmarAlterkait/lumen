"""Lowering: a detector file's geometry to the Scene every geometry query is built from.

    loaded = lower(df, detector=df.detector_volume, sensors=sensor_set.is_sensor)
    loaded.scene      # lumen.geometry.scene.Scene
    loaded.media      # GDML material name -> medium id (optics are attached by medium id)
    loaded.sensors    # sensor id -> the instance's placement path (one id per world instance: scene.PATH)

    scene = split_unions(loaded.scene)                            # large unions placed as their pieces (exact)

One Definition per logical volume (assemblies are member-less definitions; a reflected placement folds into a
mirrored variant of its definition), Geant4's conventions throughout:

- a physvol rotation is passive: ``p_mother = G^T p_daughter + t`` with ``G = Rz Ry Rx`` (``rot_from_angles``
  returns ``G^T``, the placement matrix); a physvol scale multiplies its columns (a negative one reflects);
- a boolean's or multi-union's rotation is active: ``p_first = G p_second + t`` (the transpose);
- a tessellated solid is an opaque ``Mesh`` member (first hit only), never silently dropped;
- an unsupported solid type is refused.

What a detector must declare (nothing else is per-detector): which volume is the detector (GDML files ship halls,
rock and ground; photons never come back from them), which volumes are sensors (GDML files mark them by an aux
SensDet tag, an EFFICIENCY surface, or only by name: the recipe's sensor groups declare the rule), and the optics of
each medium, attached by ``media``.

The input is a ``lumen.format.schema.DetectorFile`` in metres.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from scipy.spatial import cKDTree

from lumen.format.schema import placement_path, placements_of
from lumen.geometry import scene as sc
from lumen.geometry import solids as so
from lumen.rules.tolerances import HOLE_POSE_TOL, HOLE_SEARCH

IDENT = np.eye(3)
MIRROR_Z = np.diag([1.0, 1.0, -1.0])  # the reflection a mirrored definition variant is built with
DIGEST_CHARS = 8  # the length of the digest naming a definition variant with holes dropped
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
    """Return the physvol matrix of angles: the identity when no rotation is given (all +0)."""
    if all(a == 0.0 and math.copysign(1.0, a) > 0 for a in angles):
        return IDENT
    return rot_from_angles(*angles)


# G4ReflectionFactory's fScalePrecision: 10 kCarTolerance (1e-9 mm), in mm as its 4 x 4 sum compares it
SCALE_PRECISION = 10 * 1e-9


class ScaleRefused(ValueError):  # noqa: N818 -- named like OpticsRefused, ImportRefused
    """A physvol scale Geant4 aborts on (G4ReflectionFactory::CheckScale, GeomVol0002)."""


def placement_pose(p):
    """Return ``(m, t)`` of a ``schema.Placement``: its rotation (reflected under a negative scale) and position.

    Geant4 places a physvol through G4ReflectionFactory::Place, which decomposes rotation x scale and keeps only the
    rotation and the reflection: a positive scale's magnitude is dropped (the volume is placed unscaled), and a
    reflecting scale must be a unit one (else GeomVol0002, a fatal exception: refused). So ``m`` is orthogonal, and
    the navigator's inverse R^T is exact.
    """
    m = _rotation(p.rotation)
    if p.scale is not None:
        s = np.asarray(p.scale, float)
        if np.prod(s) < 0.0 and np.sum(np.abs(np.abs(s) - 1.0)) > SCALE_PRECISION:
            raise ScaleRefused(f"placement {p.name or p.volume!r}: scale {tuple(s)} reflects and is not a unit scale: "
                               "Geant4 aborts (G4ReflectionFactory::CheckScale, GeomVol0002)")  # fmt: skip
        m = m * np.sign(s)[None, :]  # the magnitude is not placed (G4ReflectionFactory::Place)
    return np.asarray(m, float), np.asarray(p.position, float)


def boolean_pose(tr):
    """Return ``(L, t)`` of a boolean's or multi-union's ``schema.Transform``: active, the transpose."""
    return np.asarray(_rotation(tr.rotation), float).T, np.asarray(tr.position, float)


# ================================================================================================ solids mapping
def _params(s, *keys, **defaults):
    return [s.params.get(k, defaults.get(k, 0.0)) for k in keys]


def _phi(s):  # an omitted attribute is G4GDMLReadSolids' 0 (lumen.format.solids.OPEN_ANGLE_ABORTS)
    return tuple(_params(s, "startphi", "deltaphi"))


SOLIDS = {
    "box": lambda s, mp: so.g4_box(*_params(s, "x", "y", "z")),
    "tube": lambda s, mp: so.g4_tube(*_params(s, "rmin", "rmax", "z"), *_phi(s)),
    "cutTube": lambda s, mp: so.g4_cuttube(*_params(s, "rmin", "rmax", "z"), *_phi(s),
                                           _params(s, "lowX", "lowY", "lowZ"), _params(s, "highX", "highY", "highZ")),
    "cone": lambda s, mp: so.g4_cone(*_params(s, "rmin1", "rmax1", "rmin2", "rmax2", "z"), _phi(s)),
    "polycone": lambda s, mp: so.g4_polycone(*_phi(s), s.planes),
    "polyhedra": lambda s, mp: so.g4_polyhedra(*_phi(s), *_params(s, "numsides"), s.planes),
    "sphere": lambda s, mp: so.g4_sphere(*_params(s, "rmin", "rmax"), _phi(s),
                                         tuple(_params(s, "starttheta", "deltatheta"))),
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
    """Return a balanced union tree (depth log2 n) of the nodes, each union GDML-level (thousands of nodes: shallow)."""
    while len(nodes) > 1:
        nxt = []
        for i in range(0, len(nodes) - 1, 2):
            nxt.append(so.Op("union", nodes[i], nodes[i + 1]))
            nxt[-1].g4bool = True
        nodes = nxt + nodes[len(nodes) - len(nodes) % 2 :]
    return nodes[0]


class SolidMapper:
    """GDML solids of a detector file to CSG trees (``lumen.geometry.solids``) with Geant4's semantics.

    An unsupported solid type is refused (NotImplementedError, naming it). A GDML-level boolean is marked ``g4bool``
    (decomposable by ``split_unions``), unlike the booleans inside a constructor (a tube's phi wedge).
    """

    def __init__(self, df):
        self.df = df

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
            raise NotImplementedError(f"solid {name!r} is a {s.type}, which lumen does not trace (it traces "
                                      f"{', '.join(sorted({*SOLIDS, *OP_OF, 'multiUnion', 'scaledSolid'}))}): "
                                      "replace it in the recipe")  # fmt: skip
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
SHOWN = 4  # a refusal names at most this many placements


class DetectorVolume(NamedTuple):
    """The detector volume: its LV, the world pose of its one placement, its mother LV and physvol name."""

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


def solid_box(node):
    """Return the axis-aligned box ``(lo, hi)`` of a lowered solid in its own frame (a mesh by its triangles); None
    when unbounded."""  # noqa: D205, D209
    if isinstance(node, sc.Mesh):
        pts = node.tris.reshape(-1, 3)
        return pts.min(0), pts.max(0)
    return so.node_bbox(node)


def _auto_detector(df, paths, materials, mapper):
    """Return the outermost volume of the main bath of a material.

    The largest volume (by its solid's box) of one of the materials (comma-separated names, matched whole, ignoring
    case), then up its placement chain while the mother is of one of them.
    """
    names = {a.strip().lower() for a in materials.split(",")}

    def matching(lv):
        return (df.volumes[lv].material or "").lower() in names

    hits = [lv for lv in paths if matching(lv)]
    if not hits:
        found = sorted({v.material for v in df.volumes.values() if v.material})
        raise ValueError(f"no volume of a material named {sorted(names)} (ignoring case): name one of the detector's "
                         f"materials ({found[:SHOWN]}{', ...' if len(found) > SHOWN else ''})")  # fmt: skip
    # volumes of that material elsewhere (a pipe, a second vessel) do not pull the choice up to the world
    vol = {}
    for h in hits:
        v = df.volumes[h]
        bb = solid_box(mapper.solid(v.solid)) if v.solid else None
        if bb is None:
            raise ValueError(f"volume {h!r} of material {v.material!r} has no bounded solid: auto: chooses by size; "
                             "name the detector volume instead")  # fmt: skip
        vol[h] = float(np.prod(bb[1] - bb[0]))
    lv = max(hits, key=lambda h: vol[h])
    chain = [e[0] for e in paths[lv]]
    k = len(chain) - 1
    while k > 0 and matching(chain[k - 1]):
        k -= 1
    return chain[k]


def common_mother(df, chains) -> str:
    """Return the innermost volume holding every one of the placement chains: its LV when that is placed once (or is
    the world), else the physvol path of its placement that holds them (a detector volume ``find_detector`` takes)."""  # noqa: D205, D209
    n = 0
    while all(len(c) > n + 1 and c[n] == chains[0][n] for c in chains):
        n += 1
    if n == 0:
        return df.world
    lv = df.placements[chains[0][n - 1]].volume
    return lv if len(placements_of(df, lv)) == 1 else placement_path(df, chains[0][:n])


def find_detector(df, spec, mapper=None):
    """Return the detector volume to keep (``DetectorVolume``).

    ``spec`` is an LV name (as stored: a GDML name without its pointer suffix, ``lumen.gdml.importer``), a physvol
    name or a physvol path (its names from the world's daughter down, joined by "/"), or ``"auto:<names>"``: the
    outermost volume of the main bath of a material (one of the comma-separated material names, matched whole,
    ignoring case: ``"auto:lar"``). It must select one placement: one placement is simulated, never the first of several
    silently. An LV placed more than once is refused, naming its placements' paths and their common mother
    (``common_mother``: give it to simulate every placement, or a path to simulate one).
    """
    if spec == df.world:
        return DetectorVolume(spec, np.eye(3), np.zeros(3), None, None)
    if spec.startswith("auto:"):
        spec = _auto_detector(df, _first_paths(df), spec[len("auto:") :], mapper or SolidMapper(df))
    chains = placements_of(df, spec)
    if not chains:
        raise ValueError(f"no volume, physvol or physvol path {spec!r} placed under the world")
    if len(chains) > 1:
        paths = [placement_path(df, c) for c in chains]
        distinct = list(dict.fromkeys(paths))
        more = ", ..." if len(distinct) > SHOWN else ""
        one = ("or give the physvol path of the one to simulate" if len(distinct) == len(paths) else
               "their physvol paths are not distinct, so none selects one placement")  # fmt: skip
        shown = ", ".join(distinct[:SHOWN])
        raise ValueError(f"detector volume {spec!r} is placed {len(chains)} times ({shown}{more}): lumen simulates "
                         "one placement of the detector volume; to simulate all of them give their "
                         f"common mother, detector volume {common_mother(df, chains)!r}; {one}")  # fmt: skip
    rot, t = np.eye(3), np.zeros(3)
    for j in chains[0]:
        lc, tc = placement_pose(df.placements[j])
        rot, t = rot @ lc, rot @ tc + t
    last = df.placements[chains[0][-1]]
    return DetectorVolume(last.volume, rot, t, last.mother, last.name)


# ==================================================================================================== lowering
@dataclass
class _Context:
    """The definition being built: its Definition, whether mirrored, and the holes dropped per local placement."""

    d: sc.Definition
    mirrored: bool
    dropped: dict


class _Lowering:
    """The state of one lowering (see ``lower``): definitions, media and the hole fillers."""

    def __init__(self, df, detector):
        self.df, self.detector = df, detector
        self.mp = SolidMapper(df)
        self.mats, self.defs = {}, {}
        self.hole_fillers = set()
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

    # ---- sibling holes
    def _sibling_holes(self, v, i, pose, by):
        """Return the holes of local placement i's subtraction that are sibling placements, the siblings filling
        them, and the unmatched holes that miss the solid's box (``_disjoint_holes``)."""  # noqa: D205, D209
        lp, tp = pose[i]
        hit = set()

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
                    hit.add(j)
                    return True
            return False

        cs = self.vol(self.df.placements[v.daughters[i]].volume).solid
        hs = self.mp.holes(cs, match)
        return frozenset(h[0] for h in hs if h[4]), hit, self._disjoint_holes(cs, hs)

    def _disjoint_holes(self, cs, hs):
        """Return the paths of the unmatched holes whose boxes miss the solid's box: A - B = A when B misses A.

        Exact, whatever the holes are (a sheet cut into pieces that each subtract one multi-union of every sensor: a
        narrow piece holds a few of them and misses the rest).
        """
        box = so.sector_bbox(self.mp.solid(cs))  # a subtraction's box is its minuend's
        if box is None:
            return frozenset()
        lo, hi = box
        cache, out = {}, set()
        for path, ref, lin, t, matched in hs:
            if matched:
                continue
            if ref not in cache:
                cache[ref] = so.node_bbox(self.mp.solid(ref))
            b = cache[ref]
            if b is None:
                continue
            w = so.corners(*b) @ np.asarray(lin).T + t
            if np.any(w.max(0) < lo) or np.any(w.min(0) > hi):
                out.add(path)
        return frozenset(out)

    def _hole_plan(self, v, mirrored):
        """Return ``(dropped, order)`` of a volume's placements.

        Per local placement the holes dropped, and the placement order (subtracted volumes before their siblings).
        """
        pls = list(range(len(v.daughters)))
        if mirrored:
            return {}, pls
        pose = {i: self.pose(v.daughters[i]) for i in pls}
        by = {}
        for j in pls:
            cs = self.vol(self.df.placements[v.daughters[j]].volume).solid
            if cs and np.linalg.det(pose[j][0]) > 0:
                by.setdefault(cs, []).append(j)
        by = {cs: (js, cKDTree(np.array([pose[j][1] for j in js]))) for cs, js in by.items()}
        drop, first = {}, {}
        for i in pls:
            child = self.df.placements[v.daughters[i]].volume
            cs = self.vol(child).solid
            if not cs or self.df.solids[cs].type != "subtraction" or np.linalg.det(pose[i][0]) < 0:
                continue
            d, hit, apart = self._sibling_holes(v, i, pose, by)
            if d or apart:
                drop[i] = d | apart
                if hit:
                    first[i] = min(hit)
                self.hole_fillers.update(self.df.placements[v.daughters[j]].volume for j in hit)
        order = list(pls)
        for i in sorted(first):  # only if a sibling it holds precedes it: just before the first one
            if first[i] < i:
                order.remove(i)
                order.insert(order.index(first[i]), i)
        return drop, order

    # ---- definitions
    def _member(self, v, lv, mirrored, drop):
        try:
            s = self.mp.solid_without(v.solid, drop) if drop else self.mp.solid(v.solid)
        except NotImplementedError as e:
            raise NotImplementedError(f"volume {lv!r}: {e}") from None
        if isinstance(s, sc.Mesh):  # an opaque free-form part; mirrored: reflect and keep the winding
            s = sc.Mesh((s.tris @ MIRROR_Z.T)[:, ::-1]) if mirrored else s
        else:
            s = so.Xf(MIRROR_Z, np.zeros(3), s) if mirrored else s
        return sc.Member(s, sc.Label(self.mid(v.material), -1, lv))

    def definition(self, lv, mirrored=False, drop=None, key=None):
        """Return the key of the Definition of an LV (built on first use): mirrored, or with holes dropped."""
        key = key or lv + ("|mirror" if mirrored else "")
        if key in self.defs:
            return key
        v = self.vol(lv)
        d = sc.Definition(key, origin=lv, holed=bool(drop))
        self.defs[key] = d
        if not v.assembly and v.solid:
            d.member = self._member(v, lv, mirrored, drop)
        dropped, order = self._hole_plan(v, mirrored)
        ctx = _Context(d, mirrored, dropped)
        for i in order:
            p = v.daughters[i]
            self.place(ctx, p, *self.pose(p), i)
        return key

    def place(self, ctx, p, lin, t, i):
        """Place placement p (local placement i) in the definition being built, at (lin, t)."""
        pl = self.df.placements[p]
        if ctx.mirrored:
            lin, t = MIRROR_Z @ lin, MIRROR_Z @ t
        refl = np.linalg.det(lin) < 0
        if i in ctx.dropped:
            tag = hashlib.sha1(repr(sorted(ctx.dropped[i])).encode()).hexdigest()[:DIGEST_CHARS]  # one per drop set
            child = self.definition(pl.volume, drop=ctx.dropped[i], key=f"{pl.volume}|holes:{tag}")
        else:
            child = self.definition(pl.volume, mirrored=refl)
        rot = lin @ MIRROR_Z if refl else lin
        ctx.d.placements.append(sc.Placement(child, rot, t, name=pl.name or "", source=p))

    def run(self):
        """Lower the geometry; return the Scene."""
        self.defs[ROOT] = sc.Definition(ROOT)
        if self.detector is not None:
            det = find_detector(self.df, self.detector, mapper=self.mp)
            dk = self.definition(det.lv)
            # the placement keeps its GDML physvol name: border surfaces on the detector volume name it
            top = sc.Placement(dk, det.R, det.t, name=det.placement or det.lv)
            self.defs[DETECTOR] = sc.Definition(DETECTOR, placements=[top])
            mother = det.mother
            out_med = self.mid(self.vol(mother).material) if mother is not None and self.vol(mother).material else 0
            return sc.Scene(self.defs, DETECTOR, sc.Label(out_med, -1, mother or "World"),
                            hole_fillers=sorted(self.hole_fillers))  # fmt: skip
        wk = self.definition(self.df.world)
        self.defs[wk].member = None  # the world solid is the outside
        return sc.Scene(self.defs, wk, sc.Label(0, -1, "World"), hole_fillers=sorted(self.hole_fillers))


# ======================================================================================================= loading
@dataclass
class Loaded:
    """A lowered detector: the scene, material name -> medium id, sensor id -> instance path."""

    scene: sc.Scene
    media: dict[str, int]
    sensors: dict[int, str] = field(default_factory=dict)


def lower(df, detector=None, sensors=None):
    """Return the ``Loaded`` scene of a detector file (``lumen.format.schema.DetectorFile``).

    Parameters
    ----------
    df : DetectorFile
        The full GDML world (``lumen.gdml.importer.read_gdml``).
    detector : str, optional
        None (the whole world), or the detector volume (``find_detector``): keep only that volume and its contents, at
        its world placement; its outside is its mother's material (photons never return from beyond the detector: the
        hall, rock, building and ground are dropped).
    sensors : callable, optional
        ``sensors(lv) -> bool``: whether a logical volume is a sensor volume (``SensorSet.is_sensor``; None: no
        sensors). Every world instance of one gets its own id (``scene.PATH``, the canonical order).

    Notes
    -----
    A hole of a subtraction (A - B, B split through its unions and multi-unions) that is a sibling placement is left
    out of A, and A is placed before those siblings (docs/dev/decisions.md, 14).
    """
    low = _Lowering(df, detector)
    scene, sen = low.run(), {}
    if sensors is not None:
        scene, sen = assign_sensors(scene, sensors)
    return Loaded(scene, dict(low.mats), sen)


def assign_sensors(scene, is_sensor):
    """Mark the sensor members (``is_sensor(volume name)``) with ``scene.PATH``: one id per world instance.

    Ids are numbered in the canonical depth-first order, unique even inside repeated parents; a path names each
    instance as the volume tree does (``scene.instances``), siblings of one name told apart by their index. An opaque
    mesh is never a sensor: its first hit only is reported, not what lies behind a cathode.

    Returns
    -------
    tuple
        (the new Scene, {sensor id: instance path name}).
    """

    def relabel(m):
        if m is not None and is_sensor(m.label.volume) and not isinstance(m.solid, sc.Mesh):
            return sc.Member(m.solid, sc.Label(m.label.medium, sc.PATH, m.label.volume))
        return m

    defs = {k: dataclasses.replace(d, member=relabel(d.member), placements=list(d.placements))
            for k, d in scene.definitions.items()}  # fmt: skip
    out = dataclasses.replace(scene, definitions=defs)
    T = sc.instances(out)  # the volume tree's order and PV names, so an id is its sensor's PV name
    ids = [i for i, d in enumerate(T.defn) if defs[d].member is not None and defs[d].member.label.sensor == sc.PATH]
    return out, {k: T.name(out, i) for k, i in enumerate(ids)}


# ============================================================================================ exact normalisation
def _is_identity(lin, t):
    return np.array_equal(lin, IDENT) and not np.any(t)


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


SPLIT_LEAVES = 16  # a union of this many leaves or more is placed as its pieces (every detector)


def split_unions(scene, min_leaves=SPLIT_LEAVES):
    """Return a Scene where each large union solid is a member-less definition placing its pieces.

    A large union solid is one member, no daughters, no sensor, ``min_leaves`` leaves or more (a CPA frame of ~150
    unioned tubes and boxes minus a few boxes): evaluated whole it costs every ray meeting its box a pass over all
    its leaves; placed as its pieces (``decompose``), each piece is bounded and evaluated alone. Each piece is a
    definition of one member with the same label, so where pieces overlap one of them owns
    the region (``lumen.geometry.ownership``): the region and every label are unchanged (a crossing between two pieces
    is one of the same label).
    """
    defs, n_split = dict(scene.definitions), 0
    for name, d in list(scene.definitions.items()):
        m = d.member
        if m is None or d.placements:
            continue
        if isinstance(m.solid, sc.Mesh) or int(m.label.sensor) != -1 or n_leaves(m.solid) < min_leaves:
            continue
        pcs = decompose(m.solid)
        if len(pcs) < 2:  # a split needs two pieces
            continue
        pl = []
        for k, pc in enumerate(pcs):
            pn = f"{name}|pc{k}"
            defs[pn] = sc.Definition(pn, sc.Member(pc, copy.copy(m.label)), [], d.origin, piece_of=name)
            pl.append(sc.Placement(pn, name=f"pc{k}"))
        defs[name] = sc.Definition(name, None, pl, d.origin)
        n_split += 1
    if not n_split:
        return scene
    return sc.Scene(defs, scene.root, scene.outside, list(scene.hole_fillers))
