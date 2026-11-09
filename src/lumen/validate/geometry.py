"""The geometry-import check: lumen's point location on a detector file against Geant4's navigator on the GDML.

Points: uniform in the detector volume's world box, and pairs straddling surfaces (a random solid instance near the
detector, a random ray through its box, one of its leaf roots, the points at +- ``DELTA`` along the ray). lumen's
answer comes from the whole world as the detector file lowers it (every placement kept, as Geant4 has it) and the
float64 oracle; Geant4's from g4ref ``--probe`` (G4Navigator::LocateGlobalPointAndSetup) on the same GDML, or its
normalised copy when the reader normalised an exporter defect. Compared per point: the material, and the placement
name where both place the point inside some placed volume (both stripped of the ``0x`` suffix, as Geant4 strips
them). Then the scene as used (the detector volume only, envelopes dissolved, the sensors kept): its material where
it places the point inside the detector.

    python -m lumen.validate.geometry DETECTOR_FILE --gdml FILE --g4ref BINARY --out DIR [--uniform N --surface N]

Writes ``DIR/geometry.json`` (the counts and the most common mismatches) and ``DIR/mismatches.npz``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

DELTA = 1e-6  # m: the surface pairs' offset along the ray
SEED = 0
MM_PER_M = 1000.0


def detector_box(df, lowering, solids):
    """Return (detector LV, world box low corner, high corner) of the detector volume (m)."""
    mapper = lowering.SolidMapper(df)
    det = lowering.find_detector(df, df.detector_volume, mapper=mapper)
    lv, rot, t = det.lv, det.R, det.t
    lo, hi = solids.node_bbox(mapper.solid(df.volumes[lv].solid))
    cs = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ rot.T + t
    return lv, cs.min(0), cs.max(0)


def surface_points(orc, blo, bhi, n, rng):
    """Return n points: pairs +- DELTA along random rays at leaf roots of solid instances whose box meets the box."""
    fin = np.all(np.isfinite(orc.lo), 1) & np.all(orc.lo <= bhi, 1) & np.all(orc.hi >= blo, 1)
    cand = np.flatnonzero(fin)
    out: list = []
    while len(out) < n:
        i = int(rng.choice(cand))
        lo, hi = np.maximum(orc.lo[i], blo), np.minimum(orc.hi[i], bhi)
        o = rng.uniform(lo, hi)
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        ol, dl = orc.R[i].T @ (o - orc.t[i]), orc.R[i].T @ d
        r = orc.solids[orc.sol[i]].roots(ol, dl)
        r = r[np.isfinite(r) & (np.abs(r) < np.linalg.norm(hi - lo) + 1e-6)]
        if not len(r):
            continue
        x = o + rng.choice(r) * d
        if np.all((x >= blo) & (x <= bhi)):
            out += [x - DELTA * d, x + DELTA * d]
    return np.asarray(out[:n])


def geant4_probe(g4ref, gdml, points, prefix):
    """Return Geant4's (material, placement, depth) per point (names stripped) and the seconds it took."""
    from lumen.gdml.reader import g4_strip

    np.savetxt(f"{prefix}_points.txt", points * MM_PER_M, fmt="%.9f")
    t0 = time.time()
    r = subprocess.run([str(g4ref), "--gdml", str(gdml), "--probe", f"{prefix}_points.txt", "--out", prefix],
                       capture_output=True, text=True)  # fmt: skip
    if r.returncode != 0:
        raise RuntimeError(f"g4ref failed ({r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    rows = [ln.split(",") for ln in Path(f"{prefix}_probe.csv").read_text().splitlines()[1:]]
    mat = np.array([g4_strip(x[1]) for x in rows], object)
    pv = np.array([g4_strip(x[2]) for x in rows], object)
    return mat, pv, np.array([int(x[4]) for x in rows]), time.time() - t0


def ours(orc, loaded, points, world_material):
    """Return lumen's (material, placement, instance) per point (names stripped)."""
    from lumen.gdml.reader import g4_strip

    med, _, inst = orc.label_many(points)
    name = {v: k for k, v in loaded.media.items()}
    mat = np.array([g4_strip(name.get(int(m), world_material)) if i >= 0 else g4_strip(world_material)
                    for m, i in zip(med, inst, strict=True)], object)  # fmt: skip
    pv = np.array([g4_strip(orc.inst_placement[i]) if i >= 0 else "" for i in inst], object)
    return mat, pv, inst


def check(path, gdml, g4ref, out, n_uniform=20000, n_surface=20000) -> dict:
    """Run the check on a detector file; return the result (also written to ``out/geometry.json``)."""
    from lumen.build import lowering, solids
    from lumen.format import io, sensorsets
    from lumen.gdml import reader
    from lumen.validate import oracle

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = io.read(path)
    world = lowering.lower(df)
    orc = oracle.Oracle(world.scene)
    lv, blo, bhi = detector_box(df, lowering, solids)
    rng = np.random.default_rng(SEED)
    pts = np.concatenate([rng.uniform(blo, bhi, (n_uniform, 3)), surface_points(orc, blo, bhi, n_surface, rng)])
    kind = np.r_[np.zeros(n_uniform, int), np.ones(len(pts) - n_uniform, int)]
    t1 = time.time()
    world_mat = df.volumes[df.world].material
    o_mat, o_pv, inst = ours(orc, world, pts, world_mat)
    t_ours = time.time() - t1
    doc = reader.read(gdml)
    g4file = gdml
    if any(r.startswith("normalised") for r in doc.report):
        g4file = out / (Path(gdml).stem + "_normalised.gdml")
        reader.write_normalised(gdml, g4file)
    g_mat, g_pv, g_dep, t_g4 = geant4_probe(g4ref, g4file, pts, str(out / "g4"))
    bad_m = o_mat != g_mat
    meshes = {lv for lv, v in df.volumes.items() if v.solid and df.solids[v.solid].type == "tessellated"}
    mesh_pv = {reader.g4_strip(p.name) for p in df.placements if p.volume in meshes}
    in_mesh = np.array([p in mesh_pv for p in g_pv])
    both_in = (inst >= 0) & (g_dep > 0)
    bad_p = both_in & (o_pv != g_pv)
    t2 = time.time()
    sset = sensorsets.read(path)
    used = lowering.lower(df, detector=df.detector_volume, dissolve=True,
                          sensors=lowering.Sensors(aux=sset.volumes.aux, names=sset.volumes.names))  # fmt: skip
    u_mat, _, u_inst = ours(oracle.Oracle(used.scene), used, pts, "")
    used_in = u_inst >= 0
    bad_u = used_in & (u_mat != g_mat)
    res = dict(detector=df.name, gdml=str(gdml), g4_gdml=str(g4file), detector_volume=lv, points=int(len(pts)),
               uniform=n_uniform, surface=int(len(pts) - n_uniform), material_mismatch=int(bad_m.sum()),
               material_mismatch_uniform=int((bad_m & (kind == 0)).sum()),
               material_mismatch_surface=int((bad_m & (kind == 1)).sum()),
               material_mismatch_in_mesh=int((bad_m & in_mesh).sum()), placement_compared=int(both_in.sum()),
               placement_mismatch=int(bad_p.sum()), as_used_compared=int(used_in.sum()),
               as_used_mismatch=int(bad_u.sum()),
               mismatch_outside_meshes=int(((bad_m | bad_p | bad_u) & ~in_mesh).sum()),
               seconds=dict(load=round(t1 - t0, 1), ours=round(t_ours, 1), g4=round(t_g4, 1),
                            as_used=round(time.time() - t2, 1)),
               material_pairs=Counter(f"{a} | g4 {b}" for a, b in zip(o_mat[bad_m], g_mat[bad_m], strict=True))
               .most_common(12),
               placement_pairs=Counter(f"{a} | g4 {b}" for a, b in zip(o_pv[bad_p], g_pv[bad_p], strict=True))
               .most_common(12))  # fmt: skip
    (out / "geometry.json").write_text(json.dumps(res, indent=1, default=str))
    np.savez(out / "mismatches.npz", points=pts, kind=kind, bad_material=bad_m, in_mesh=in_mesh, bad_placement=bad_p,
             bad_as_used=bad_u)  # fmt: skip
    return res


def main(argv=None) -> int:
    """Run the check from the command line; exit 1 when any point disagrees outside the opaque meshes.

    A mesh (tessellated solid) is an opaque part that holds no point by design, so a point Geant4 places inside one
    is lumen's mother volume: those points are counted and listed, never a failure.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("detector_file")
    ap.add_argument("--gdml", required=True, help="the GDML the detector file was converted from")
    ap.add_argument("--g4ref", required=True, help="the g4ref binary (Geant4 10.7.4)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--uniform", type=int, default=20000)
    ap.add_argument("--surface", type=int, default=20000)
    a = ap.parse_args(argv)
    res = check(a.detector_file, a.gdml, a.g4ref, a.out, a.uniform, a.surface)
    sys.stdout.write(json.dumps(res, indent=1, default=str) + "\n")
    return 1 if res["mismatch_outside_meshes"] else 0


if __name__ == "__main__":
    sys.exit(main())
