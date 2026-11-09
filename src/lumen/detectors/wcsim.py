"""Recipes for the WCSim detectors (SK, WCTE, ANNIE), from the GDML WCSim itself exports.

WCSim semantics these recipes follow (WCSim e2dc7a48 source):

- The hit. WCSim's sensitive detector is the PMT glass; WCSimWCSD::ProcessHits counts a photon killed in the glass on
  a step whose post volume is InteriorWCPMT, i.e. absorbed at the glass to interior border surface
  (GlassCathodeSurface), whether Geant4 calls it Detection or Absorption. So the sensors are the InteriorWCPMT
  volumes, counted across their GDML border surfaces (rule ``border_into``) with every absorption entering detected
  (``efficiency_on_entry`` 1). The reverse border (AirCathodeSurface, where the GDML has one: WCTE) counts with its
  EFFICIENCY, credited to the interior it leaves (WCSimOpBoundaryProcess::DoAbsorption and ProcessHits_boundary).
- Quantum and collection efficiency are off in the reference runs; they are applied after tracking.
- Only WCSim's photocathode model 0 (= G4OpBoundaryProcess) is implemented; models 1 and 2 (thin-film cathodes) are
  refused, as they would not show in the GDML.
- A material whose ABSLENGTH is below ``ABSORB_AT_ENTRY`` at every knot, used by a tessellated (mesh) volume, is
  opaque in effect (ANNIE's StainlessSteel: Geant4 refracts a photon in and absorbs it within about 1e-11 m); it is
  bound opaque, its border and skin surfaces still apply.

ANNIE's quantum efficiencies (Multi_Tank_Types) are read from WCSim's C++ source at conversion time
(``annie_qe``): the 20-point tables of each tube type, its maximum and the tuning's QEratio, stored with the detector
file.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np

from lumen.format import schema
from lumen.format.sensorsets import SensorGroup
from lumen.recipe import Recipe

DETECTOR = "WCBarrel"  # WCSim's water volume (WC is air, expHall vacuum)
SENSOR_LV = "InteriorWCPMT"
CATHODE_MODEL = 0  # the reference's /WCSim/tuning/PMTCathodePara model
ABSORB_AT_ENTRY = 1e-6  # m: an ABSLENGTH below this at every knot makes a mesh volume's material opaque in effect
WAVELENGTHS = (300.0, 600.0)  # nm
ANNIE_TANK_TYPES = ("R7081", "D784KFLB", "R5912HQE", "R7081HQE")  # SetANNIEPhase2Geometryv7's tank collections
QE_POINTS = 20  # WCSim's QE tables have 20 points


def _detector_lvs(df: schema.DetectorFile, root: str) -> set:
    lvs, stack = set(), [root]
    while stack:
        lv = stack.pop()
        if lv not in lvs:
            lvs.add(lv)
            stack += [df.placements[i].volume for i in df.volumes[lv].daughters]
    return lvs


def opaque_mesh_materials(df: schema.DetectorFile, inputs: dict) -> schema.DetectorFile:
    """Bind opaque every mesh volume's material that absorbs at entry (see the module docstring)."""
    if CATHODE_MODEL != 0:
        raise ValueError(f"WCSim photocathode model {CATHODE_MODEL}: only model 0 is implemented")
    lvs = _detector_lvs(df, df.detector_volume)
    meshes = {df.volumes[lv].material for lv in lvs
              if df.volumes[lv].solid and df.solids[df.volumes[lv].solid].type == "tessellated"}  # fmt: skip
    mats, rep = dict(df.materials), list(df.report)
    for name in sorted(m for m in meshes if m in mats):
        a = mats[name].properties.get("ABSLENGTH")
        if a is not None and a.energy is not None and float(np.max(a.value)) < ABSORB_AT_ENTRY:
            note = f"mesh material with ABSLENGTH < {ABSORB_AT_ENTRY} m at every knot: absorbed at entry"
            mats[name] = dataclasses.replace(mats[name], optics=schema.OpticsModel("lumen:opaque", 1, {"medium": name},
                                                                               "recipe", note))  # fmt: skip
            rep.append(f"recipe: material {name} bound opaque ({note})")
    return dataclasses.replace(df, materials=mats, report=tuple(rep))


def recipe(name: str, detector_volume: str = DETECTOR, decisions=(), inputs=(), **kw) -> Recipe:
    """Return a WCSim detector's recipe."""
    group = SensorGroup("pmt", "border_into", "lumen:geant4", lv_pattern=f"^{SENSOR_LV}",
                        settings={"efficiency_on_entry": 1.0, "name_prefix": "pmt"})  # fmt: skip
    return Recipe(name=name, detector_volume=detector_volume, wavelength_range=WAVELENGTHS, groups=(group,),
                  decisions=(opaque_mesh_materials, *decisions), inputs=tuple(inputs), **kw)  # fmt: skip


# ------------------------------------------------------------------------------- ANNIE: tube types and QE tables
def _cxx(path) -> str:
    """Return a C++ source without its comments."""
    s = Path(path).read_text()
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", s, flags=re.S))


def wcsim_pmt_qe(types, source) -> dict:
    """Return {type: (wavelength[20] nm, QE[20], maxQE)} as WCSim defines them.

    Notes
    -----
    CreatePMTObject's type to class (WCSimDetectorConstruction.cc), then the class's GetQEWavelength, GetQE and
    GetmaxQE (WCSimPMTObject.cc).
    """
    dc, po = _cxx(f"{source}/src/WCSimDetectorConstruction.cc"), _cxx(f"{source}/src/WCSimPMTObject.cc")
    out = {}
    for ty in types:
        cls = re.search(rf'PMTType\s*==\s*"{ty}"\s*\)\s*\{{\s*WCSimPMTObject\s*\*\s*PMT\s*=\s*new\s+(\w+)', dc).group(1)

        def body(fn, cls=cls):
            return re.search(rf"\b{cls}::{fn}\s*\(\s*\)\s*\{{(.*?)\n\}}", po, re.S).group(1)

        def arr(fn):
            v = re.search(r"\{([^{}]*)\}", body(fn)).group(1)
            return [float(x) for x in v.replace("\n", " ").split(",") if x.strip()]

        wl, qe = arr("GetQEWavelength"), arr("GetQE")
        mx = float(re.search(r"maxQE\s*=\s*([0-9.eE+-]+)", body("GetmaxQE")).group(1))
        if len(wl) != QE_POINTS or len(qe) != QE_POINTS:
            raise ValueError(f"WCSim PMT {ty} ({cls}): QE tables of {len(wl)} / {len(qe)} points, not {QE_POINTS}")
        out[ty] = (wl, qe, mx)
    return out


def wcsim_tuning(path) -> dict:
    """Return {command: value} of a WCSim tuning macro (/WCSim/tuning/<command> value)."""
    out = {}
    for ln in Path(path).read_text().splitlines():
        f = ln.split("#")[0].split()
        if len(f) >= 2 and f[0].startswith("/WCSim/tuning/"):
            out[f[0].rsplit("/", 1)[1]] = f[1]
    return out


def tube_types(df: schema.DetectorFile, paths) -> list:
    """Return each sensor's tube type: the suffix of the glass face placed beside its interior in the same tube."""
    holder = {}
    for name, v in df.volumes.items():
        for i in v.daughters:
            holder.setdefault(df.placements[i].volume, set()).add(name)
    out = []
    for p in paths:
        lv = str(p).rsplit("/", 1)[-1].split("#")[0]
        tys = {
            m.group(1)
            for n in holder.get(lv, set())
            for i in df.volumes[n].daughters
            for m in [re.match(r".*glassFaceWCPMT_([A-Za-z0-9]+?)0x[0-9a-f]+$", df.placements[i].volume)]
            if m
        }
        if len(tys) != 1:
            raise ValueError(f"tube_types: {p!r}: interior LV {lv!r} has glass faces of types {sorted(tys)}")
        out.append(tys.pop())
    return out


def annie_columns(df, loaded, tree, inputs) -> dict:
    """Return the id map's per-sensor tube types."""
    return {"tube_type": tube_types(df, [loaded.sensors[i] for i in sorted(loaded.sensors)])}


def annie_qe(df, sensor_set, inputs) -> dict:
    """ANNIE's Multi_Tank_Types quantum efficiency data, read from WCSim's source at conversion time."""
    tun = wcsim_tuning(inputs["wcsim_tuning"])
    if tun.get("PMTwiseQE", "0") not in ("0", "false"):
        raise ValueError("ANNIE: PMTwiseQE on (pmt_qe.txt by WCSim tube id) is not implemented")
    tabs = wcsim_pmt_qe(ANNIE_TANK_TYPES, inputs["wcsim_source"])
    return {
        "annie_wcsim_qe": dict(
            qe_ratio=float(tun["QEratio"]),
            tank_types=list(ANNIE_TANK_TYPES),
            tables={t: dict(wavelength_nm=w, qe=q, max_qe=m) for t, (w, q, m) in tabs.items()},
            source="WCSim annie e88314b: WCSimPMTObject.cc, WCSimDetectorConstruction.cc",
        )
    }


SK = recipe("sk")
WCTE = recipe("wcte")
ANNIE = recipe("annie", inputs=("wcsim_source", "wcsim_tuning"), sensor_columns=annie_columns, calibrations=annie_qe)
