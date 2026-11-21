"""The overlap-free GDML of a detector: lumen's ownership of overlapping siblings written out for Geant4.

    df2, notes = owned(df, sensor_set)        # the detector file with every shared region given to its owner
    report = export("microboone.lumen", "microboone_owned.gdml")      # lumen.gdml.export of owned(...)
    python -m lumen.compare export microboone.lumen microboone_owned.gdml   # also microboone_owned.json

The JSON beside the GDML records the export's sha256, the content digest of the detector file it was written from,
whether it is lossless, and the notes: a g4ref run on the export is a reference product that names them
(``lumen.compare.references.original``).

lumen gives every point one owner, level by level (``lumen.geometry.ownership``); Geant4 takes siblings as disjoint and
daughters as inside their mothers. Where the detector file already means that (daughters inside their mothers,
siblings disjoint up to the contact tolerance), the export is the detector file as it is: its own solids and
placements. Only what really overlaps is rewritten, and one source decides what does: lumen's placement check
(``lumen.build.placements.check`` and ``check_meshes``, the report ``lumen.convert`` records), points just inside each
volume's surface, a contact (a shared face, an overlap thinner than the contact tolerance) no overlap -- the
tolerance the navigator links its pairs by. A pair the navigator links but the check finds apart (surfaces that
coincide, a containment the proofs cannot decide) means the same in Geant4 and is written as it is: an intersection or
subtraction there only adds coincident surfaces for Geant4 to step through. The rewriting follows the navigator's
lists (its linked pairs, the ancestors a volume is not proved inside, ``lumen.geometry.ownership.Proofs``), for the
pairs and protrusions the check found:

- the non-owner of an overlapping pair becomes a subtraction solid, itself less its owner (an owner that protrudes
  from it is represented as it is: nothing is dropped or approximated); a volume below the non-owner that its owner
  may reach is cut alike; a tessellated solid alike; a volume proved inside its owner holds nothing and is left out;
- a daughter reaching out of its mother becomes an intersection solid, itself within the mother's solid, and so does
  each volume below it that the mother is not proved to hold (it would reach out of the cut daughter);
- a logical volume written as several variants carries its skin on each, and a border surface names each copy of
  its placements (every pair of copies: the border's meaning, a step from one into the other).

A tool for comparing with Geant4, never a path the photons take: lumen's own conversion keeps the geometry as written.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter

import numpy as np

import lumen
from lumen import gdml
from lumen.build import lowering
from lumen.build import placements as pc
from lumen.format import schema
from lumen.format.sensorsets import SensorSet
from lumen.geometry import navigate
from lumen.geometry import ownership as own
from lumen.geometry import scene as sc
from lumen.rules.tolerances import GIMBAL_COS, SAME_POSE

SUFFIX = "_owned"  # a logical volume or solid made here: <name>_owned<k>


class OwnershipExportError(ValueError):
    """An ownership the detector file cannot express (a reflected placement, an owner reached through an assembly)."""


def gdml_angles(L):  # noqa: N803 -- L a rotation matrix
    """Return GDML rotation angles (x, y, z) (rad) with ``Rz(z) Ry(y) Rx(x) = L`` (``lowering.rot_from_angles``)."""
    L = np.asarray(L, float)
    sy = -L[2, 0]
    y = math.asin(max(-1.0, min(1.0, sy)))
    if abs(math.cos(y)) > GIMBAL_COS:
        x, z = math.atan2(L[2, 1], L[2, 2]), math.atan2(L[1, 0], L[0, 0])
    else:  # gimbal lock: z = 0
        x, z = math.atan2(-L[1, 2], L[1, 1]), 0.0
    return tuple(0.0 if a == 0.0 else a for a in (x, y, z))  # +0, never -0 (an unrotated placement)


def _pose(df, p):
    m, t = lowering.placement_pose(df.placements[p])
    if np.linalg.det(m) < 0 or not np.allclose(m.T @ m, np.eye(3), atol=SAME_POSE):
        raise OwnershipExportError(f"placement {df.placements[p].name!r} is reflected or scaled: its ownership is not "
                                   "written out")  # fmt: skip
    return m, t


def _chain(df, path):
    """Return the pose (m, t) of the end of a placement path in the frame of its first placement's mother."""
    m, t = np.eye(3), np.zeros(3)
    for p in path:
        mp, tp = _pose(df, p)
        m, t = m @ mp, m @ tp + t
    return m, t


def _same(df, a, b):
    """Return whether solids a and b are one solid (the same type, parameters and references)."""
    return a == b or dataclasses.replace(df.solids[a], name="") == dataclasses.replace(df.solids[b], name="")


class _Solids:
    """The detector file's solids as CSG trees (``lowering.SolidMapper``), each built and flattened once."""

    def __init__(self, df):
        self.df, self.mapper, self.nodes, self.memo = df, lowering.SolidMapper(df), {}, {}

    def node(self, name):
        """Return (the solid's tree or mesh, its ``_flat`` (None: a mesh), its box) or None (unsupported)."""
        if name not in self.nodes:
            try:
                n = self.mapper.solid(name)
            except NotImplementedError:
                n = None
            bx = own.box_of(n) if n is not None else None
            self.nodes[name] = None if bx is None else (n, own.flat(n), bx)
        return self.nodes[name]

    def relate(self, inner, outer, L, t, want):  # noqa: N803 -- L a rotation matrix
        """Return whether ``inner`` (placed in ``outer``'s frame by ``x = L y + t``) is proved apart from or inside it.

        ``want`` is "apart" or "inside"; the proof is the navigator's (``lumen.geometry.ownership.relate_poses``), and
        one solid at one pose is inside itself. Memoised per pose.
        """
        key = (inner, outer, np.asarray(L, float).tobytes(), np.asarray(t, float).tobytes(), want)
        if key not in self.memo:
            self.memo[key] = self._relate(inner, outer, L, t, want)
        return self.memo[key]

    def _relate(self, inner, outer, L, t, want):  # noqa: N803 -- L a rotation matrix
        if want == "inside" and _same(self.df, inner, outer) and np.allclose(L, np.eye(3), rtol=0, atol=own.SAME_POSE) \
                and np.allclose(t, 0.0, rtol=0, atol=own.SAME_POSE):  # fmt: skip
            return True  # one solid at one pose: coincident
        ni, no = self.node(inner), self.node(outer)
        if ni is None or no is None:
            return False
        return bool(own.relate_poses(ni[1], ni[2], no[1], np.asarray(L, float)[None], np.asarray(t, float)[None],
                                     want)[0])  # fmt: skip


def _paths(scene, nav):
    """Return per navigator id (a volume, n: none, then each mesh at n + 1 + k) its detector-file placement path (the
    lowering's ``Placement.source``s), in the navigator's order (``scene.instances``)."""  # noqa: D205, D209
    T = sc.instances(scene)
    path: list = []
    vols, meshes = [], []
    for i, dname in enumerate(T.defn):
        p = int(T.parent[i])
        here = (path[p] if p >= 0 else ()) + ((int(T.source[i]),) if T.source[i] >= 0 else ())
        path.append(here)
        m = scene.definitions[dname].member
        if m is not None:
            (meshes if isinstance(m.solid, sc.Mesh) else vols).append(here)
    if len(vols) != nav.n or len(meshes) != nav.n_mesh:
        raise OwnershipExportError(f"the scene walk gives {len(vols)} volumes and {len(meshes)} meshes, the navigator "
                                   f"{nav.n} and {nav.n_mesh}")  # fmt: skip
    return vols + [None] + meshes


def found(nav, backend="cpu"):
    """Return what the placement check finds overlapping: (sibling pairs as frozensets of ids, protruding ids).

    Navigator ids: a mesh k as its pseudo-volume n + 1 + k. ``lumen.build.placements.check`` and ``check_meshes`` on
    the CPU (``backend``), as ``placements.conversion`` runs them, so the pairs are those the detector file's report
    names. A mesh passing through a volume pairs with the volume's ancestor (or itself) among the mesh's siblings, the
    unit the navigator links.
    """
    n = nav.n
    c = pc.check(nav, backend=backend)
    m = pc.check_meshes(nav, backend=backend)
    par = np.asarray(nav.host["parent"])  # each volume's mother (a navigator built without classes dissolves none)
    pairs = {frozenset(map(int, p)) for p in np.asarray(c["pairs"]).reshape(-1, 2)}
    mother = {int(k): int(mv) for k, mv in zip(nav.host["cmesh"], nav.arrays.probe["cmoth"], strict=False)}
    for k, v in m["pairs"]:
        while v < n and par[v] != mother[k] and par[v] < n:
            v = int(par[v])
        pairs.add(frozenset((n + 1 + int(k), int(v))))
    prot = {int(u) for u in c["protrude"]} | {n + 1 + int(k) for k in m["protrude"]}
    return pairs, prot


def _cuts(nav, real):
    """Return the cuts of the pairs and protrusions found (``real``, from ``found``): (volume, owner or ancestor,
    non-owner it stems from, op) -- the non-owner of an overlapping pair and each volume below it its owner may reach,
    less the owner ("subtraction"); a protruding volume, within its mother, and each volume below it not proved inside
    that mother, within it alike ("intersection")."""  # noqa: D205, D209
    n = nav.n
    size = np.asarray(nav.host["size"])
    cmesh = np.asarray(nav.host["cmesh"])
    par = np.asarray(nav.host["parent"])  # each volume's mother (``found``)
    seen = [set(k) for k in nav.kids[:n]]
    pairs, prot = real

    def reaches(d, w):
        if w < n:
            return w in seen[d]
        return any(c >= nav.ch0 and cmesh[c - nav.ch0] == w - n - 1 for c in seen[d])

    out = []
    linked = set()
    for u, w, forced in nav.owned:
        linked.add(frozenset((u, w)))
        if forced or frozenset((u, w)) not in pairs:
            continue
        out.append((u, w, u, "subtraction"))
        if u < n:
            out += [(d, w, u, "subtraction") for d in range(u + 1, u + int(size[u])) if reaches(d, w)]
    clips = {v: {a for a in seen[v] if a < v < a + int(size[a])} for v in range(n)}
    clips |= {v: {a} for v, a in nav.clipped if v > n}
    lost = sorted(tuple(sorted(p)) for p in pairs - linked) + sorted((v,) for v in prot if not clips.get(v))
    if lost:
        raise OwnershipExportError(f"the placement check finds overlaps the navigator does not link ({lost[:4]}): "
                                   "not written out")  # fmt: skip

    def kept(v, a):  # v reaches out of a: it protrudes from its mother a, or lies in a volume that does
        return a in clips.get(v, ()) and (v in prot if par[v] == a or v > n else kept(int(par[v]), a))

    out += [(v, a, a, "intersection") for v in sorted(clips) for a in sorted(clips[v]) if kept(v, a)]
    return out


@dataclasses.dataclass
class _Plan:
    """What changes in one logical volume's contents.

    ``cuts``: per path below it (placement indices) the solids to combine it with, (op, solid, L, t) with the solid
    placed in the path's frame by ``x = L y + t``.
    """

    cuts: dict = dataclasses.field(default_factory=dict)


def _plan(df, nav, paths, real):
    """Return per logical volume its _Plan: the cuts of the overlaps found (``_cuts``: a subtraction per owner, an
    intersection per ancestor reached out of), and a volume proved inside its owner left out ("drop")."""  # noqa: D205, D209
    sol = _Solids(df)
    plans: dict[str, _Plan] = {}
    by_pair, clips = {}, []
    for v, w, u, op in _cuts(nav, real):
        pv, pw = paths[v], paths[w]
        if op == "subtraction" and pv[: len(pw)] == pw:  # pieces of one split union, or a piece and a daughter
            continue  # of it: the union is written whole
        if op == "intersection":  # w an ancestor: its path a prefix of v's
            if pv[: len(pw)] != pw or len(pv) <= len(pw):
                raise OwnershipExportError(f"{nav.names[min(v, nav.n - 1)]!r} reaches out of an ancestor through an "
                                           "assembly: not written out")  # fmt: skip
            clips.append((df.placements[pv[len(pw)]].mother, pv[len(pw) :], w))
            continue
        k = len(pw) - 1
        if k < 0 or pv[:k] != pw[:k] or len(pv) <= k:
            raise OwnershipExportError(f"an owner owns part of a volume across an assembly ({pv}, {pw}): not written "
                                       "out")  # fmt: skip
        mother = df.placements[pw[-1]].mother
        if df.placements[pv[k]].mother != mother:
            raise OwnershipExportError(f"a volume and its owner have different mothers in the detector file ({pv}, "
                                       f"{pw})")  # fmt: skip
        by_pair.setdefault((mother, paths[u][k:], pw[-1]), set()).add(pv[k:])
    for (mother, _, q), targets in by_pair.items():
        mw, tw = _pose(df, q)
        w_sol = df.volumes[df.placements[q].volume].solid
        for path in sorted(targets):
            mp_, tp_ = _chain(df, path)
            L, t = mp_.T @ mw, mp_.T @ (tw - tp_)  # the owner in the path's frame
            v_sol = df.volumes[df.placements[path[-1]].volume].solid
            inside = sol.relate(v_sol, w_sol, L.T, -L.T @ t, "inside")
            cut = ("drop", w_sol, L, t) if inside else ("subtraction", w_sol, L, t)
            plans.setdefault(mother, _Plan()).cuts.setdefault(path, []).append(cut)
    _clip(df, sol, paths, clips, plans)
    return plans


def _clip(df, sol, paths, clips, plans):
    """Add to ``plans`` each clip (ancestor's volume, path, ancestor): the path's end within it."""
    for lv, path, a in clips:
        mp_, tp_ = _chain(df, path)  # the path's end in lv's frame; lv is the ancestor's volume
        if paths[a] and df.placements[paths[a][-1]].volume != lv:  # (the detector volume has no placement path)
            raise OwnershipExportError(f"the ancestor's volume is not the path's mother {lv!r}")
        plans.setdefault(lv, _Plan()).cuts.setdefault(path, []).append(("intersection", df.volumes[lv].solid,
                                                                         mp_.T, -mp_.T @ tp_))  # fmt: skip


def _dropped(cuts):
    return any(op == "drop" for op, _, _, _ in cuts)


class _Writer:
    """The detector file being rewritten: new solids, volumes and placements, named once; per new volume the original
    it is a variant of (``base``), per new placement the original it copies (``orig``)."""  # noqa: D205, D209

    def __init__(self, df):
        self.df = df
        self.solids = dict(df.solids)
        self.volumes = dict(df.volumes)
        self.placements = list(df.placements)
        self.k = 0
        self.memo = {}
        self.base = {lv: lv for lv in df.volumes}
        self.orig = list(range(len(df.placements)))

    def fresh(self, base, taken):
        while True:
            name = f"{base}{SUFFIX}{self.k}"
            self.k += 1
            if name not in taken:
                return name

    def cut_solid(self, solid, cuts):
        name = solid
        for op, second, L, t in cuts:
            new = self.fresh(solid, self.solids)
            self.solids[new] = schema.Solid(
                new,
                op,
                refs=(name, second),
                transforms=(
                    schema.Transform(),
                    schema.Transform(rotation=gdml_angles(L), position=tuple(float(x) for x in t)),
                ),
            )
            name = new
        return name

    def placement(self, p, volume, mother):
        pl = dataclasses.replace(self.placements[p], volume=volume, mother=mother)
        self.placements.append(pl)
        self.orig.append(self.orig[p])
        return len(self.placements) - 1

    def variant(self, lv, mods):
        """Return the name of lv's variant: its solid cut by mods[()] (owners, ancestors), its daughters' variants,
        a daughter whose mods[(p,)] drop it left out."""  # noqa: D205, D209
        key = (lv, repr(sorted((k, [(op, q, np.asarray(L).tolist(), np.asarray(t).tolist()) for op, q, L, t in c])
                               for k, c in mods.items())))  # fmt: skip
        if not mods:
            return lv
        if key in self.memo:
            return self.memo[key]
        v = self.volumes[lv]
        name = self.fresh(lv, self.volumes)
        self.memo[key] = name
        self.base[name] = self.base[lv]
        solid = self.cut_solid(v.solid, mods.get((), [])) if mods.get(()) else v.solid
        daughters = []
        for p in v.daughters:
            sub = {k[1:]: c for k, c in mods.items() if k and k[0] == p}
            if not _dropped(sub.get((), [])):
                daughters.append(self.placement(p, self.variant(self.placements[p].volume, sub), name))
        self.volumes[name] = dataclasses.replace(v, name=name, solid=solid, daughters=tuple(daughters))
        return name


def owned(df, sensor_set=None):
    """Return (the detector file with lumen's ownership written out, the notes saying what was changed)."""
    sset = sensor_set or SensorSet(())
    loaded = lowering.lower(df, detector=df.detector_volume, sensors=sset.is_sensor)
    scene = lowering.split_unions(loaded.scene)
    nav = navigate.Navigator.from_scene(scene)
    linked = any(not f for _, _, f in nav.owned) or nav.clipped
    real = found(nav) if linked else (set(), set())
    plans = _plan(df, nav, _paths(scene, nav), real) if real[0] or real[1] else {}
    if not plans:
        return df, ["no overlapping siblings, no daughter reaching out of its mother: written as it is"]
    w = _Writer(df)
    notes = []
    rank = _post_order(df)
    for mother, plan in sorted(plans.items(), key=lambda kv: rank[kv[0]]):  # inner mothers first
        keep = []
        for p in w.volumes[mother].daughters:
            mods = {k[1:]: c for k, c in plan.cuts.items() if k[0] == p}
            if _dropped(mods.get((), [])):
                continue
            keep.append(p)
            child = w.variant(w.placements[p].volume, mods)
            if child != w.placements[p].volume:
                w.placements[p] = dataclasses.replace(w.placements[p], volume=child)
        w.volumes[mother] = dataclasses.replace(w.volumes[mother], daughters=tuple(keep))
        for path, cs in sorted(plan.cuts.items()):
            name = f"{df.placements[path[-1]].volume} in {mother}"
            if _dropped(cs):
                notes.append(f"{name} left out: inside {', '.join(sorted({q for op, q, _, _ in cs if op == 'drop'}))}")
                continue
            less = sorted({sol for op, sol, _, _ in cs if op == "subtraction"})
            within = sorted({sol for op, sol, _, _ in cs if op == "intersection"})
            notes.append(name + (f" less {', '.join(less)}" if less else "")
                         + (f" within {', '.join(within)}" if within else ""))  # fmt: skip
    return _tidy(df, w), notes


def _post_order(df):
    """Return per logical volume its rank in a post-order from the world: every volume after all below it."""
    rank, seen = {}, set()

    def visit(lv):
        if lv in seen:
            return
        seen.add(lv)
        for p in df.volumes[lv].daughters:
            visit(df.placements[p].volume)
        rank[lv] = len(rank)

    visit(df.world)
    return rank


def _tidy(df, w):
    """Return the rewritten detector file: placements renumbered, an original no placement uses any more dropped, a
    variant that replaced every use of its original under the original's name, the skins and borders on every
    variant and copy (``_surfaces``)."""  # noqa: D205, D209
    lvs = _post_lvs(w)  # each volume after those below it: the order a GDML file holds its placements in
    live = [p for lv in lvs for p in w.volumes[lv].daughters]
    used = {w.placements[p].volume for p in live} | {df.world}
    was = {pl.volume for pl in df.placements}
    gone = {lv for lv in was if lv not in used}  # replaced everywhere by variants
    rename = {}
    for name in lvs:
        base = name.rsplit(SUFFIX, 1)[0] if SUFFIX in name else None
        if base in gone and base not in rename.values():
            rename[name] = base
    order = [p for lv in lvs if lv not in gone for p in w.volumes[lv].daughters]
    index = {p: k for k, p in enumerate(order)}

    def rn(x):
        return rename.get(x, x)

    pls = [dataclasses.replace(w.placements[p], volume=rn(w.placements[p].volume), mother=rn(w.placements[p].mother))
           for p in order]  # fmt: skip
    vols = {rn(lv): dataclasses.replace(w.volumes[lv], name=rn(lv),
                                        daughters=tuple(index[p] for p in w.volumes[lv].daughters))
            for lv in lvs if lv not in gone}  # fmt: skip
    skins, borders = _surfaces(df, w, {rn(lv): w.base[lv] for lv in lvs if lv not in gone}, pls,
                               [w.orig[p] for p in order])  # fmt: skip
    return dataclasses.replace(df, solids=w.solids, volumes=vols, placements=tuple(pls), skins=skins, borders=borders)


def _surfaces(df, w, base, pls, orig):
    """Return (skins, borders) of the rewritten file; rename (in ``pls``) the copies of a placement a border names.

    A skin on a logical volume is on each of its variants (``base``: per written volume its original). A border names
    placements by name: each copy (``orig``: per written placement the original it copies) of a placement a border
    names gets a name of its own (the first keeps the original's), and the border is written for every pair of copies,
    so it holds wherever the original pair meets (a pair that never meets is inert in Geant4). A border whose placement
    was left out everywhere holds nowhere: dropped. A skin on a volume left out goes with it.
    """
    skins = []
    for k in df.skins:
        for j, lv in enumerate(sorted(x for x, b in base.items() if b == k.volume)):
            skins.append(dataclasses.replace(k, volume=lv, name=k.name if j == 0 else w.fresh(k.name, ())))
    count = Counter(pl.name for pl in df.placements)
    named = {pv for b in df.borders for pv in (b.pv1, b.pv2) if count[pv] == 1}  # (an ambiguous one: the exporter's)
    copies: dict = {}
    for i, o in enumerate(orig):
        nm = df.placements[o].name
        if nm in named:
            copies.setdefault(nm, []).append(i)
    taken = {pl.name for pl in pls}
    for nm, ii in copies.items():
        for i in ii[1:]:
            pls[i] = dataclasses.replace(pls[i], name=w.fresh(nm, taken))
            taken.add(pls[i].name)
    borders = []
    for b in df.borders:
        if b.pv1 not in named or b.pv2 not in named:  # no placement, or several, of that name: the exporter's to judge
            borders.append(b)
            continue
        pairs = [(pls[i].name, pls[j].name) for i in copies.get(b.pv1, ()) for j in copies.get(b.pv2, ())]
        for n_, (a, c) in enumerate(pairs):
            borders.append(dataclasses.replace(b, pv1=a, pv2=c, name=b.name if n_ == 0 else w.fresh(b.name, ())))
    return tuple(skins), tuple(borders)


def _post_lvs(w):
    """Return every logical volume, each after every volume below it (daughters in order; the world's first)."""
    out, seen = [], set()

    def visit(lv):
        if lv in seen:
            return
        seen.add(lv)
        for p in w.volumes[lv].daughters:
            visit(w.placements[p].volume)
        out.append(lv)

    for lv in [w.df.world, *w.volumes]:
        visit(lv)
    return out


def export(det_or_path, path, **kw):
    """Write the overlap-free GDML of a detector (a ``.lumen`` path or a loaded detector) at ``path``.

    ``kw`` goes to ``lumen.gdml.export``. Returns (the ExportReport, the notes of ``owned``).
    """
    det = lumen.load(det_or_path) if not hasattr(det_or_path, "file") else det_or_path
    df2, notes = owned(det.file, det.sensor_set)
    return gdml.export(df2, path, **kw), notes


def main(argv=None) -> int:
    """Write a detector file's overlap-free GDML and, beside it, its record (``<gdml>.json``; module docstring)."""
    import argparse
    import json
    import sys
    from pathlib import Path

    from lumen.format import io

    ap = argparse.ArgumentParser(prog="python -m lumen.compare export", description=main.__doc__)
    ap.add_argument("detector_file")
    ap.add_argument("gdml")
    a = ap.parse_args(argv)
    det = lumen.load(a.detector_file)
    report, notes = export(det, a.gdml)
    record = dict(tool="lumen.compare.ownership", lumen=lumen.__version__, detector=det.file.name,
                  detector_digest=io.content_digest(io.to_files(det.file)), sha256=report.sha256,
                  lossless=report.lossless, notes=notes)  # fmt: skip
    Path(a.gdml).with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    sys.stdout.write(f"{a.gdml}: {report.sha256}, {len(notes)} notes, lossless {report.lossless}\n")
    return 0 if report.lossless else 1
