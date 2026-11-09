"""Recipes for the LArSoft detectors: SBND (its full-simulation GDML) and MicroBooNE (LArSoft's public optics).

SBND (sbnd_v02_00_full_simulation.gdml, sbndcode 4e41b3d8): the GDML holds every optical property SBND's full
optical simulation gives Geant4. The recipe's decisions, each checked against the GDML first:

- The TPB film: volCPATPB is a 23 nm TPB box on one face of each CPA foil (inside volCPA_FoilTPB, LAr, exactly foil
  plus film thick). It is index-matched to LAr (equal RINDEX tables, checked), so TPB is bound to LAr's medium and the
  foil's skin becomes a wavelength-shifting film over the foil's own surface on the film's face (``lumen:wls_film``:
  the film's WLSABSLENGTH, thickness, Geant4's WLSCOMPONENT sampler and isotropic emission), crossed layer by layer as
  Geant4 tracks it. The foil's back face and edges are the bare foil surface.
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

LAR, TPB = "LAr", "TPB"
TPB_LV, FOIL_LV, FOILTPB_LV = "volCPATPB", "volCPAFoil", "volCPA_FoilTPB"
CRYOSTAT_LV, MEMBRANE_LV, ENCLOSURE_LV = "volCryostat", "volCryoMembrane", "volDetEnclosure"
SBND_WAVELENGTHS = (100.0, 900.0)  # nm: LAr scintillation (104-172) and TPB emission (350-826)
FILM_SURFACE = "cpa_tpb"
FILM_AXES = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]  # the CPA films face +-x (the drift axis), checked at load
SPLIT_UNIONS = 16  # unions of more pieces are placed as their pieces in the navigator's scene
GEOMETRY_TOLERANCE = 1e-12  # m: two GDML lengths that agree to this are equal (1e-9 mm)


def _box(df: schema.DetectorFile, lv: str) -> np.ndarray:
    s = df.solids[df.volumes[lv].solid]
    if s.type != "box":
        raise ValueError(f"{lv}: solid {s.name} is a {s.type}, expected a box")
    return np.array([s.params["x"], s.params["y"], s.params["z"]], np.float64)


def _daughters(df, lv):
    return {df.placements[i].volume: df.placements[i] for i in df.volumes[lv].daughters}


def tpb_facts(df: schema.DetectorFile) -> dict:
    """Check the film geometry the SBND recipe relies on; return the film thickness d (m) and side (+1 / -1 in x)."""
    film, foil, both = _box(df, TPB_LV), _box(df, FOIL_LV), _box(df, FOILTPB_LV)
    pv = _daughters(df, FOILTPB_LV)
    names = {TPB_LV, FOILTPB_LV} | {p.name for p in df.placements if p.volume in (TPB_LV, FOILTPB_LV)}
    if any(k.volume in (TPB_LV, FOILTPB_LV) for k in df.skins) or any(
        b.pv1 in names or b.pv2 in names for b in df.borders
    ):
        raise ValueError(f"{TPB_LV} / {FOILTPB_LV} carry an optical surface: the film lowering does not apply")
    if set(pv) != {TPB_LV, FOIL_LV} or df.volumes[FOILTPB_LV].material != LAR:
        raise ValueError(f"{FOILTPB_LV}: expected LAr holding exactly {FOIL_LV} and {TPB_LV}")
    xf, xt = pv[FOIL_LV].position[0], pv[TPB_LV].position[0]
    side = 1.0 if xt > xf else -1.0
    tol = GEOMETRY_TOLERANCE
    ok = (abs(film[0] + foil[0] - both[0]) < tol and np.allclose(film[1:], foil[1:]) and np.allclose(foil[1:], both[1:])
          and abs((xt - side * film[0] / 2) - (xf + side * foil[0] / 2)) < tol
          and abs(abs(xt) + film[0] / 2 - both[0] / 2) < tol)  # fmt: skip
    if not ok:
        raise ValueError(f"TPB film geometry is not a {film[0]} m layer on one face of the foil filling {FOILTPB_LV}")
    if df.volumes[TPB_LV].material != TPB:
        raise ValueError(f"{TPB_LV} is not of material {TPB}")
    a, b = df.materials[TPB].properties["RINDEX"], df.materials[LAR].properties["RINDEX"]
    if not (np.array_equal(a.energy, b.energy) and np.array_equal(a.value, b.value)):
        raise ValueError("TPB RINDEX differs from LAr's: the film is not index-matched")
    return dict(d=float(film[0]), side=side)


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
    film = tpb_facts(df)
    outside = membrane_skin(df)
    mats, rep = {}, list(df.report)
    for name, m in df.materials.items():
        if name == TPB:
            o = schema.OpticsModel(
                "lumen:same_as", 1, {"material": LAR}, "recipe", "index-matched film on the CPA foil"
            )
        elif "RINDEX" not in m.properties:
            o = schema.OpticsModel("lumen:opaque", 1, {"medium": "opaque"}, "recipe", "no RINDEX: one opaque medium")
        else:
            o = m.optics
        mats[name] = dataclasses.replace(m, optics=o)
    backing = next(k.surface for k in df.skins if k.volume == FOIL_LV)
    settings = dict(film_material=TPB, front_material=LAR, backing=backing, thickness_m=film["d"],
                    local_axis=[film["side"], 0.0, 0.0], expected_axes=FILM_AXES)  # fmt: skip
    surf = schema.Surface(
        FILM_SURFACE,
        "",
        "",
        "",
        0.0,
        {},
        schema.OpticsModel("lumen:wls_film", 1, settings, "recipe", "TPB film on the foil's drift-volume face"),
    )
    skins = df.skins + (schema.Skin(FILM_SURFACE, FILM_SURFACE, FOIL_LV, "recipe", "replaces the bare foil's skin"),)
    rep += [f"recipe: TPB bound to LAr; the CPA foil's skin is a {film['d']} m TPB film over {backing}",
            f"recipe: the outside carries {MEMBRANE_LV}'s skin {outside}"]  # fmt: skip
    return dataclasses.replace(df, materials=mats, surfaces={**df.surfaces, FILM_SURFACE: surf}, skins=skins,
                               outside_surface=outside,
                               build_options={**df.build_options, "split_unions": SPLIT_UNIONS},
                               report=tuple(rep))  # fmt: skip


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
