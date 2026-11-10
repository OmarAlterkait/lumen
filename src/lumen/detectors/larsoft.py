"""Recipes for the LArSoft detectors: SBND (its full-simulation GDML) and MicroBooNE (LArSoft's public optics).

SBND (sbnd_v02_00_full_simulation.gdml, sbndcode 4e41b3d8): the GDML holds every optical property SBND's full
optical simulation gives Geant4. The recipe's decisions, each checked against the GDML first:

- The TPB film (volCPATPB, 23 nm on one face of each CPA foil) is lowered by the general film rule
  (``lumen.build.import_rules``): TPB bound to LAr, the foil's skin a ``lumen:wls_film`` on the film's face.
- The outside: the detector volume (volCryostat) is exactly the inner box of volCryoMembrane (steel), placed at the
  same position, so the outside carries the membrane's skin (checked).
- Every material without RINDEX is one opaque medium (Geant4 kills a photon there).
- The sensors are the volumes with the SensDet aux value PhotonDetector: 120 PMTs (volOpDetSensitive) and 192
  X-ARAPUCA windows (volOpDetSensitiveXArapucaFilter), counted on entry; their physvol copy number is the OpDet channel,
  typed by sbndcode's sbnd_pds_mapping.json (input ``pds_mapping``).

MicroBooNE (microboone.gdml): only LAr carries optical properties in LArSoft's MicroBooNE simulation (LArSoft's
tables, ``lumen.detectors.larsoft_optics``); every other material kills a photon at its boundary, except the
stainless steel, which carries LArSoft's reflector table. Photons are counted on entering the volOpDetSensitive
discs. The tables keep their source tokens (the numbers in eV and cm, as LArSoft hands them to Geant4), so a GDML
export makes Geant4 read the doubles LArSoft's Geant4 reads.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from lumen.format import schema
from lumen.format.sensorsets import SensorGroup
from lumen.recipe import Recipe

LAR = "LAr"
CRYOSTAT_LV, MEMBRANE_LV, ENCLOSURE_LV = "volCryostat", "volCryoMembrane", "volDetEnclosure"
SBND_WAVELENGTHS = (100.0, 900.0)  # nm: LAr scintillation (104-172) and TPB emission (350-826)
GEOMETRY_TOLERANCE = 1e-12  # m: two GDML lengths that agree to this are equal (1e-9 mm)


def _daughters(df, lv):
    return {df.placements[i].volume: df.placements[i] for i in df.volumes[lv].daughters}


def membrane_skin(df: schema.DetectorFile) -> str:
    """Check that the detector volume is enclosed by the membrane's inner box; return the membrane's skin surface."""
    msol = df.solids[df.volumes[MEMBRANE_LV].solid]
    inner = df.solids.get(msol.refs[1]) if msol.type == "subtraction" else None
    cbox = df.solids[df.volumes[CRYOSTAT_LV].solid]
    pos = {p.volume: p.position for p in _daughters(df, ENCLOSURE_LV).values()}
    skins = {k.volume: k.surface for k in df.skins}
    ok = (inner is not None and cbox.type == inner.type == "box"
          and all(abs(cbox.params[k] - inner.params[k]) < GEOMETRY_TOLERANCE for k in "xyz")
          and np.allclose(pos[CRYOSTAT_LV], pos[MEMBRANE_LV]) and MEMBRANE_LV in skins)  # fmt: skip
    if not ok:
        raise ValueError(f"{CRYOSTAT_LV} is not enclosed by {MEMBRANE_LV}'s inner box")
    return skins[MEMBRANE_LV]


def sbnd_optics(df: schema.DetectorFile, inputs: dict) -> schema.DetectorFile:
    """SBND's optics decisions (see the module docstring), each recorded with origin "recipe"."""
    outside = membrane_skin(df)
    mats, rep = {}, list(df.report)
    for name, m in df.materials.items():
        if "RINDEX" not in m.properties:
            o = schema.OpticsModel("lumen:opaque", 1, {"medium": "opaque"}, "recipe", "no RINDEX: one opaque medium")
        else:
            o = m.optics
        mats[name] = dataclasses.replace(m, optics=o)
    rep.append(f"recipe: the outside carries {MEMBRANE_LV}'s skin {outside}")
    return dataclasses.replace(df, materials=mats, outside_surface=outside, report=tuple(rep))


def channel_of(df: schema.DetectorFile, mother_lv: str, child_lv: str) -> int:
    """Return the copy number of the one physvol of ``child_lv`` inside ``mother_lv`` (an OpDet channel)."""
    hits = [p.copy for p in _daughters_all(df, mother_lv) if p.volume == child_lv]
    if len(hits) != 1:
        raise ValueError(f"{mother_lv!r}: {len(hits)} placements of {child_lv!r} (want 1)")
    return hits[0]


def _daughters_all(df, lv):
    return [df.placements[i] for i in df.volumes[lv].daughters]


def sbnd_columns(df, loaded, tree, inputs) -> dict:
    """Return the id map's per-sensor OpDet channel, type and logical volume."""
    types = {int(r["channel"]): r["pd_type"] for r in json.loads(Path(inputs["pds_mapping"]).read_text())}
    t = tree.tree_from_geometry(loaded)
    ch, ty, lv = [], [], []
    for i in sorted(loaded.sensors):
        p = t.by_name[loaded.sensors[i]]
        c = channel_of(df, t.by_name[p.mother].lv, p.lv)
        ch.append(c)
        ty.append(types[c])
        lv.append(p.lv)
    if len(set(ch)) != len(ch):
        raise ValueError("sbnd: OpDet channels are not unique")
    return {"channel": ch, "type": ty, "lv": lv}


# Neither SBND's nor MicroBooNE's GDML has an exporter defect Geant4 refuses (lumen.gdml.reader.REPAIRS): no repairs.
SBND = Recipe(
    name="sbnd", detector_volume="auto:argon,lar", wavelength_range=SBND_WAVELENGTHS,
    groups=(SensorGroup("pmt", "any_entry", "lumen:counting", lv="volOpDetSensitive", settings={"eff": 1.0}),
            SensorGroup("xarapuca", "any_entry", "lumen:counting", lv="volOpDetSensitiveXArapucaFilter",
                        settings={"eff": 1.0})),
    decisions=(sbnd_optics,), inputs=("pds_mapping",), sensor_columns=sbnd_columns)  # fmt: skip


# ------------------------------------------------------------------------------------------------------- MicroBooNE
STEEL = "STEEL_STAINLESS_Fe7Cr2Ni"
STEEL_SURFACE = "steel"
MICROBOONE_WAVELENGTHS = (100.0, 180.0)  # nm: the LAr emission spectrum (7.2-11.9 eV)


def microboone_optics(df: schema.DetectorFile, inputs: dict) -> schema.DetectorFile:
    """MicroBooNE's optics: LArSoft's LAr tables, LArSoft's steel reflector, every other material opaque."""
    from lumen.detectors import larsoft_optics

    mats = {}
    for name, m in df.materials.items():
        if name == LAR:
            o = schema.OpticsModel("lumen:larsoft_lar", 1, {"medium": "lar"}, "recipe", "LArSoft's LAr tables")
            mats[name] = dataclasses.replace(m, properties=larsoft_optics.lar_tables(), optics=o)
        else:
            o = schema.OpticsModel(
                "lumen:opaque", 1, {"medium": "opaque"}, "recipe", "no optical properties in LArSoft"
            )
            mats[name] = dataclasses.replace(m, optics=o)
    o = schema.OpticsModel("lumen:larsoft_reflector", 1, {}, "recipe", "LArSoft's ReflectiveSurfaces")
    steel = schema.Surface(STEEL_SURFACE, "dielectric_metal", "unified", "ground", 0.0,
                      larsoft_optics.reflector_tables(STEEL), o)  # fmt: skip
    skins = tuple(schema.Skin(f"steel:{lv}", STEEL_SURFACE, lv, "recipe", "LArSoft's steel reflector")
                  for lv, v in df.volumes.items() if v.material == STEEL)  # fmt: skip
    rep = list(df.report) + ["recipe: LAr from LArSoft's tables; the steel carries LArSoft's reflector; every other "
                             "material is opaque"]  # fmt: skip
    toks = {
        **df.tokens,
        "materials": {**df.tokens.get("materials", {}), LAR: larsoft_optics.lar_tokens()},
        "surfaces": {**df.tokens.get("surfaces", {}), STEEL_SURFACE: larsoft_optics.reflector_tokens(STEEL)},
    }
    return dataclasses.replace(df, materials=mats, surfaces={**df.surfaces, STEEL_SURFACE: steel}, skins=skins,
                               report=tuple(rep), tokens=toks)  # fmt: skip


MICROBOONE = Recipe(
    name="microboone", detector_volume="auto:argon,lar", wavelength_range=MICROBOONE_WAVELENGTHS,
    groups=(SensorGroup("pmt", "any_entry", "lumen:counting", lv="volOpDetSensitive", settings={"eff": 1.0}),),
    decisions=(microboone_optics,))  # fmt: skip
