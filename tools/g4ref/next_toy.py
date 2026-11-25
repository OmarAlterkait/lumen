"""The NEXT-like toy detectors of lumen's NEXT tests, and the reduction of their Geant4 runs.

    python tools/g4ref/next_toy.py write tests/data/next_toy      # the GDML cases (committed)
    python tools/g4ref/next_toy.py reduce tests/data/next_toy RUNDIR/<case>  # the reference counts

One geometry at NEXT-100's scales (a metre-scale teflon light tube coated with 3 um of TPB whose inner face lies on
the active gas; a SiPM board at one end, its teflon mask carrying a 4 um TPB coating whose face lies on the gas
through the mask and the board, four SiPM windows with sensitive areas; a PMT at the other end), in four cases that
switch on one kind of surface each:

- ``plain``: the sensors' own skins only (polished dielectric_metal with EFFICIENCY, Geant4's InvokeSD) and the light
  tube's unified ground skin;
- ``sigma1``: ``plain`` and the mask's unified ground skin with sigma_alpha = 1 rad;
- ``glisur``: ``plain`` and the glisur ground dielectric_dielectric borders gas <-> TPB (polish 0.01), which the
  volumes meet across coincident faces only;
- ``all``: every surface together.

The surfaces are nexus's NEXT-100 kinds (``gas_tpb_teflon_surf``, ``SIPM_BOARD_MASK_OPSURF``, ``PHOTOCATHODE``,
``SIPM_OPSURF``, ``refl_Surf``); there is no wavelength shifting in the toy, so one photon energy runs through.

The Geant4 reference of each case is g4ref on its own GDML, from the two point sources of ``SOURCES`` (mm, photons,
eV):

    tools/g4ref/build.sh g4ref                       # against the Geant4 whose release goes into the manifest
    for c in plain sigma1 glisur all; do
      g4ref --gdml tests/data/next_toy/next_toy_$c.gdml --config sources.cfg \
            --sens-regex '^(PMT_PHOTOCATHODE|SIPM_SENSE)$' --seed 31 --batch 1000 --out run_$c
      python tools/g4ref/next_toy.py reduce tests/data/next_toy run_$c
    done

``reduce`` writes ``tests/data/next_toy/geant4.json``: per case, the detected photons per sensor group (pmt, sipm)
and per SiPM, as fractions of the photons launched, with their binomial errors, and the Geant4 release.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

SOURCES = ((0.0, 0.0, 0.0, 1000000, 2.95), (0.0, 0.0, -550.0, 1000000, 2.95))  # x y z N E_eV, mm
GROUPS = {"pmt": "PMT_PHOTOCATHODE", "sipm": "SIPM_SENSE"}

GDML = r"""<?xml version="1.0" encoding="UTF-8" ?>
<!-- A NEXT-like toy at NEXT-100's scales, small enough for a test: a metre-scale teflon light tube coated with
     3 um of TPB whose inner face lies on the gas, a SiPM board at one end (a teflon mask carrying a TPB coating
     whose face lies on the gas, four SiPM windows with sensitive areas) and a PMT at the other, both read by
     their own skin surface. The surfaces are nexus's NEXT-100 kinds: a glisur ground dielectric_dielectric border
     gas <-> TPB (polish 0.01), a unified ground dielectric_metal skin on the mask (sigma_alpha 1 rad), a unified
     ground skin on the tube, and polished dielectric_metal skins with EFFICIENCY on the photocathode and the
     sensitive areas. Lengths in mm, as nexus writes them. The surfaces are filled in per case
     (tools/g4ref/next_toy.py). -->
<gdml>
<define>
  <matrix name="E2" coldim="2" values="1.0*eV 1.0 12.0*eV 1.0"/>
  <matrix name="RI_GAS" coldim="2" values="1.0*eV 1.01 12.0*eV 1.01"/>
  <matrix name="RI_TPB" coldim="2" values="1.0*eV 1.67 12.0*eV 1.67"/>
  <matrix name="RI_QUARTZ" coldim="2" values="1.0*eV 1.47 12.0*eV 1.47"/>
  <matrix name="ABS_LONG" coldim="2" values="1.0*eV 1.0e6 12.0*eV 1.0e6"/>
  <matrix name="ABS_TPB" coldim="2" values="1.0*eV 1.0e4 12.0*eV 1.0e4"/>
  <matrix name="REFL_TEFLON" coldim="2" values="1.0*eV 0.97 12.0*eV 0.97"/>
  <matrix name="REFL_ZERO" coldim="2" values="1.0*eV 0.0 12.0*eV 0.0"/>
  <matrix name="LOBE_ONE" coldim="2" values="1.0*eV 1.0 12.0*eV 1.0"/>
  <matrix name="EFF_SIPM" coldim="2" values="1.0*eV 0.4 12.0*eV 0.4"/>
  <matrix name="EFF_PMT" coldim="2" values="1.0*eV 0.3 12.0*eV 0.3"/>
</define>
<materials>
  <element name="toyXe" formula="Xe" Z="54"> <atom value="131.29"/> </element>
  <element name="toyC" formula="C" Z="6"> <atom value="12.011"/> </element>
  <material name="GXe" state="gas"> <property name="RINDEX" ref="RI_GAS"/>
    <property name="ABSLENGTH" ref="ABS_LONG"/> <D value="0.09"/> <fraction n="1.0" ref="toyXe"/> </material>
  <material name="TPB" state="solid"> <property name="RINDEX" ref="RI_TPB"/>
    <property name="ABSLENGTH" ref="ABS_TPB"/> <D value="1.08"/> <fraction n="1.0" ref="toyC"/> </material>
  <material name="Quartz" state="solid"> <property name="RINDEX" ref="RI_QUARTZ"/>
    <property name="ABSLENGTH" ref="ABS_LONG"/> <D value="2.2"/> <fraction n="1.0" ref="toyC"/> </material>
  <material name="Teflon" state="solid"> <D value="2.2"/> <fraction n="1.0" ref="toyC"/> </material>
  <material name="Copper" state="solid"> <D value="8.96"/> <fraction n="1.0" ref="toyC"/> </material>
</materials>
<solids>
  <box name="WorldBox" x="3000" y="3000" z="3000" lunit="mm"/>
  <tube name="GasTube" rmax="700" z="1600" deltaphi="360" aunit="deg" lunit="mm"/>
  <tube name="ActiveTube" rmax="500" z="1200" deltaphi="360" aunit="deg" lunit="mm"/>
  <tube name="TubeSolid" rmin="500" rmax="520" z="1200" deltaphi="360" aunit="deg" lunit="mm"/>
  <tube name="CoatSolid" rmin="500" rmax="500.003" z="1200" deltaphi="360" aunit="deg" lunit="mm"/>
  <box name="BoardBox" x="400" y="400" z="20" lunit="mm"/>
  <box name="MaskBox" x="200" y="200" z="6" lunit="mm"/>
  <box name="MaskCoatBox" x="200" y="200" z="0.004" lunit="mm"/>
  <box name="SipmBox" x="60" y="60" z="1" lunit="mm"/>
  <box name="SipmSenseBox" x="30" y="30" z="0.2" lunit="mm"/>
  <tube name="PmtBody" rmax="100" z="40" deltaphi="360" aunit="deg" lunit="mm"/>
  <tube name="PmtCath" rmax="95" z="0.2" deltaphi="360" aunit="deg" lunit="mm"/>
{solids}</solids>
<structure>
  <volume name="SIPM_SENSE"> <materialref ref="Copper"/> <solidref ref="SipmSenseBox"/> </volume>
  <volume name="SIPM"> <materialref ref="Quartz"/> <solidref ref="SipmBox"/>
    <physvol name="sipm_sense"> <volumeref ref="SIPM_SENSE"/>
      <position name="pss" unit="mm" x="0" y="0" z="-0.4"/> </physvol>
  </volume>
  <volume name="MASK_COAT"> <materialref ref="TPB"/> <solidref ref="MaskCoatBox"/> </volume>
  <volume name="MASK"> <materialref ref="Teflon"/> <solidref ref="MaskBox"/>
    <physvol name="mask_coat"> <volumeref ref="MASK_COAT"/>
      <position name="pmc" unit="mm" x="0" y="0" z="2.998"/> </physvol>
  </volume>
  <volume name="BOARD"> <materialref ref="Copper"/> <solidref ref="BoardBox"/>
    <physvol name="mask"> <volumeref ref="MASK"/> <position name="pm" unit="mm" x="0" y="0" z="7"/> </physvol>
    <physvol name="sipm1"> <volumeref ref="SIPM"/> <position name="ps1" unit="mm" x="-150" y="-150" z="9.5"/>
    </physvol>
    <physvol name="sipm2"> <volumeref ref="SIPM"/> <position name="ps2" unit="mm" x="150" y="-150" z="9.5"/>
    </physvol>
    <physvol name="sipm3"> <volumeref ref="SIPM"/> <position name="ps3" unit="mm" x="-150" y="150" z="9.5"/>
    </physvol>
    <physvol name="sipm4"> <volumeref ref="SIPM"/> <position name="ps4" unit="mm" x="150" y="150" z="9.5"/>
    </physvol>
  </volume>
  <volume name="PMT_PHOTOCATHODE"> <materialref ref="Copper"/> <solidref ref="PmtCath"/> </volume>
  <volume name="PMT"> <materialref ref="Quartz"/> <solidref ref="PmtBody"/>
    <physvol name="pmt_cath"> <volumeref ref="PMT_PHOTOCATHODE"/>
      <position name="pcath" unit="mm" x="0" y="0" z="-19.9"/> </physvol>
  </volume>
  <volume name="TUBE_COAT"> <materialref ref="TPB"/> <solidref ref="CoatSolid"/> </volume>
  <volume name="LIGHT_TUBE"> <materialref ref="Teflon"/> <solidref ref="TubeSolid"/>
    <physvol name="tube_coat"> <volumeref ref="TUBE_COAT"/> </physvol>
  </volume>
  <volume name="ACTIVE"> <materialref ref="GXe"/> <solidref ref="ActiveTube"/> </volume>
  <volume name="GAS"> <materialref ref="GXe"/> <solidref ref="GasTube"/>
    <physvol name="active"> <volumeref ref="ACTIVE"/> </physvol>
    <physvol name="light_tube"> <volumeref ref="LIGHT_TUBE"/> </physvol>
    <physvol name="board"> <volumeref ref="BOARD"/> <position name="pb" unit="mm" x="0" y="0" z="-620"/> </physvol>
    <physvol name="pmt"> <volumeref ref="PMT"/> <position name="pp" unit="mm" x="0" y="0" z="630"/> </physvol>
  </volume>
  <volume name="WORLD"> <materialref ref="Teflon"/> <solidref ref="WorldBox"/>
    <physvol name="gas"> <volumeref ref="GAS"/> </physvol>
  </volume>
{structure}</structure>
<setup name="Default" version="1.0"> <world ref="WORLD"/> </setup>
</gdml>
"""


SENSOR_SOLIDS = """\
  <opticalsurface name="PHOTOCATHODE" model="unified" finish="polished" type="dielectric_metal" value="1">
    <property name="REFLECTIVITY" ref="REFL_ZERO"/> <property name="EFFICIENCY" ref="EFF_PMT"/>
  </opticalsurface>
  <opticalsurface name="SIPM_OPSURF" model="unified" finish="polished" type="dielectric_metal" value="1">
    <property name="REFLECTIVITY" ref="REFL_ZERO"/> <property name="EFFICIENCY" ref="EFF_SIPM"/>
  </opticalsurface>
  <opticalsurface name="TUBE_REFL" model="unified" finish="ground" type="dielectric_metal" value="0.01">
    <property name="REFLECTIVITY" ref="REFL_TEFLON"/> <property name="SPECULARLOBECONSTANT" ref="LOBE_ONE"/>
  </opticalsurface>
"""
SENSOR_SKINS = """\
  <skinsurface name="pmt_cathode_skin" surfaceproperty="PHOTOCATHODE">
    <volumeref ref="PMT_PHOTOCATHODE"/> </skinsurface>
  <skinsurface name="sipm_skin" surfaceproperty="SIPM_OPSURF"> <volumeref ref="SIPM_SENSE"/> </skinsurface>
  <skinsurface name="light_tube_skin" surfaceproperty="TUBE_REFL"> <volumeref ref="LIGHT_TUBE"/> </skinsurface>
"""
MASK_SOLIDS = """\
  <opticalsurface name="MASK_OPSURF" model="unified" finish="ground" type="dielectric_metal" value="1">
    <property name="REFLECTIVITY" ref="REFL_TEFLON"/> <property name="SPECULARLOBECONSTANT" ref="LOBE_ONE"/>
  </opticalsurface>
"""
MASK_SKINS = """\
  <skinsurface name="mask_skin" surfaceproperty="MASK_OPSURF"> <volumeref ref="MASK"/> </skinsurface>
"""
GLISUR_SOLIDS = """\
  <opticalsurface name="TPB_GLISUR" model="glisur" finish="ground" type="dielectric_dielectric" value="0.01"/>
"""
GLISUR_BORDERS = """\
  <bordersurface name="active_tube_coat" surfaceproperty="TPB_GLISUR">
    <physvolref ref="active"/> <physvolref ref="tube_coat"/> </bordersurface>
  <bordersurface name="tube_coat_active" surfaceproperty="TPB_GLISUR">
    <physvolref ref="tube_coat"/> <physvolref ref="active"/> </bordersurface>
  <bordersurface name="gas_mask_coat" surfaceproperty="TPB_GLISUR">
    <physvolref ref="gas"/> <physvolref ref="mask_coat"/> </bordersurface>
  <bordersurface name="mask_coat_gas" surfaceproperty="TPB_GLISUR">
    <physvolref ref="mask_coat"/> <physvolref ref="gas"/> </bordersurface>
"""
CASES = {
    "plain": ("", ""),
    "sigma1": (MASK_SOLIDS, MASK_SKINS),
    "glisur": (GLISUR_SOLIDS, GLISUR_BORDERS),
    "all": (MASK_SOLIDS + GLISUR_SOLIDS, MASK_SKINS + GLISUR_BORDERS),
}


def write(out: Path):
    """Write every case's GDML under ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    for case, (solids, structure) in CASES.items():
        text = GDML.format(solids=SENSOR_SOLIDS + solids, structure=SENSOR_SKINS + structure)
        (out / f"next_toy_{case}.gdml").write_text(text)
        print(f"wrote {out / f'next_toy_{case}.gdml'}")
    (out / "sources.cfg").write_text(
        "# x y z N E_eV (mm)\n" + "".join(f"{x} {y} {z} {n} {e}\n" for x, y, z, n, e in SOURCES)
    )


def geant4_version(name: str) -> str:
    """Return the release of a G4Version string ("$Name: geant4-11-03-patch-00 [MT]$": "11.3.0")."""
    m = re.search(r"geant4-(\d+)-(\d+)(?:-patch-(\d+))?", name)
    if m is None:
        raise ValueError(f"not a Geant4 version string: {name!r}")
    return f"{int(m[1])}.{int(m[2])}.{int(m[3] or 0)}"


def reduce(out: Path, run: Path):
    """Add one case's counts (the g4ref run with prefix ``run``) to ``out/geant4.json``."""
    case = run.name.split("_", 1)[1]
    summary = (run.parent / f"{run.name}_summary.txt").read_text()
    rows = np.load(run.parent / f"{run.name}_hits.npy")
    launched = sum(n for *_, n, _ in SOURCES)
    table = (run.parent / f"{run.name}_sensors.csv").read_text().splitlines()[1:]
    path_of = {int(line.split(",")[0]): line.split(",")[1] for line in table if line}
    paths = [path_of[int(i)] for i in rows["sensor"]]
    lv_of = {int(line.split(",")[0]): line.split(",")[2] for line in table if line}
    lvs = [lv_of[int(i)] for i in rows["sensor"]]
    counts = {g: int(sum(x == lv for x in lvs)) for g, lv in GROUPS.items()}
    sipms: dict = {}
    for p in paths:
        m = re.search(r"/sipm(\d)", p)
        if m:
            sipms[f"sipm{m[1]}"] = sipms.get(f"sipm{m[1]}", 0) + 1
    rec = {
        "geant4": geant4_version(re.search(r"^geant4 (.*)$", summary, re.M)[1]),
        "launched": launched,
        "gdml_sha256": _sha(out / f"next_toy_{case}.gdml"),
        "source_sha256": _sha(Path(__file__).with_name("g4ref.cc")),
    }
    for k, c in sorted(counts.items()) + sorted(sipms.items()):
        rec[k] = c / launched
        rec[k + "_error"] = float(np.sqrt(c * (1 - c / launched))) / launched
        rec[k + "_counts"] = c
    path = out / "geant4.json"
    all_rec = json.loads(path.read_text()) if path.exists() else {}
    all_rec[case] = rec
    path.write_text(json.dumps(all_rec, indent=1, sort_keys=True) + "\n")
    print(f"{case}: " + ", ".join(f"{k} {rec[k]:.5g}" for k in sorted(counts)))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


if __name__ == "__main__":
    what, out = sys.argv[1], Path(sys.argv[2])
    if what == "write":
        write(out)
    elif what == "reduce":
        reduce(out, Path(sys.argv[3]))
    else:
        raise SystemExit(__doc__)
