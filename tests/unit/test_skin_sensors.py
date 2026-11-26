"""Sensors whose detecting surface is their volume's own skin (nexus's photocathodes and SiPMs, read with
G4OpticalParameters::SetBoundaryInvokeSD): a ``border_into`` group's volume that no border sensor enters.

What is checked:

- at the surface lookup, the skin is the sensor where Geant4 picks it: entering from the mother (the daughter's skin
  first), or from a volume without a skin; a sibling's own skin, or a border on the crossing, is read instead;
  leaving a transparent skin sensor its skin is the surface, without detection;
- through simulate, a pencil beam into a sensor slab whose skin has R = 0 and EFFICIENCY 0.4 is detected at 0.4;
- a border_into group with neither a border into its volumes nor a detecting skin is refused.
"""

import math

import numpy as np
import pytest
from test_surfaces import compile_, kern, library, tree

from lumen.build.lookup import SensorBinding
from lumen.rules.entries import SurfaceEntry
from lumen.surfaces import types

PMT = (
    ("World", "WorldLV", "Water", None),
    ("Pmt", "PmtLV", "Acrylic", "World"),  # the glass
    ("Cath", "CathLV", "LAr", "Pmt"),  # the cathode, its skin the sensor (transparent here, to test the exit)
    ("Side", "SideLV", "Water2", "Pmt"),  # a sibling of the cathode without a skin
    ("Skinned", "SkinnedLV", "Water2", "Pmt"),  # one with its own skin
)


def _lib():
    lib = library()
    lib.surfaces["mirror"] = SurfaceEntry(types.no_detection(types.SPECULAR_METAL), {"R": 1.0})
    return lib


def test_the_skin_is_the_sensor_where_geant4_picks_it():
    t = tree(*PMT)
    cath = (SensorBinding("pmt", 1, pv="Cath", entry="skin"),)
    o = compile_(t, lib=_lib(), skins={"SkinnedLV": "refl", "CathLV": "refl2"}, sensors=cath)
    assert kern(o, "Pmt", "Cath")[1] == "sensor:pmt"  # from the mother: the daughter's skin
    assert kern(o, "Side", "Cath")[1] == "sensor:pmt"  # from a sibling without a skin: the cathode's
    assert kern(o, "Skinned", "Cath")[1] == "surface:refl"  # from a sibling with its own: that one first
    # leaving the cathode its skin is the surface, without detection (Geant4's InvokeSD: the volume entered's SD)
    assert kern(o, "Cath", "Pmt")[1] == "surface:refl2" and kern(o, "Cath", "Pmt")[0].detects is False
    assert kern(o, "Cath", "Skinned")[1] == "surface:refl2"  # into a sibling with a skin: the cathode's first
    o = compile_(t, lib=_lib(), borders={("Pmt", "Cath"): "mirror"},
                 sensors=(SensorBinding("pmt", 1, pv="Cath", entry="skin"),))  # fmt: skip
    assert kern(o, "Pmt", "Cath")[1] == "surface:mirror"  # a border before any skin


def test_daughters_and_touching_placements_of_a_skin_sensor():
    """Geant4 10.7.4 with SetBoundaryInvokeSD (skins' EFFICIENCY 0.4 and 0.7 on two touching slabs): a crossing from
    one into the other is read by the skin of the volume left and detected by the volume entered; a daughter without a
    skin leads back into its mother through the mother's."""
    t = tree(*PMT[:3], ("Cath2", "CathLV", "LAr", "Pmt"), ("Core", "CoreLV", "Water2", "Cath"),
             ("Other", "OtherLV", "Water2", "Pmt"))  # fmt: skip
    o = compile_(t, lib=_lib(), skins={"CathLV": "refl2", "OtherLV": "refl"},
                 sensors=(SensorBinding("pmt", 2, lv="CathLV", entry="skin"),))  # fmt: skip
    assert kern(o, "Cath", "Cath2")[1] == "sensor:pmt"  # the one skin: detected by the placement entered
    assert o.deposit_pre is None
    assert kern(o, "Pmt", "Cath2")[1] == "sensor:pmt" and len(o.sensor_pvs) == 2
    assert kern(o, "Cath", "Core")[1] == "surface:refl2"  # into a daughter without a skin: its mother's
    assert kern(o, "Core", "Cath")[1] == "sensor:pmt"
    assert kern(o, "Other", "Cath")[1] == "surface:refl"  # another skin first


SLAB = """<?xml version="1.0" encoding="UTF-8" ?>
<gdml>
<define>
  <matrix name="R1" coldim="2" values="2.0*eV 1.0 4.5*eV 1.0"/>
  <matrix name="R0" coldim="2" values="2.0*eV 0.0 4.5*eV 0.0"/>
  <matrix name="E4" coldim="2" values="2.0*eV 0.4 4.5*eV 0.4"/>
</define>
<materials>
  <element name="O" formula="O" Z="8"> <atom value="15.999"/> </element>
  <material name="Gas" state="gas"> <property name="RINDEX" ref="R1"/> <D value="0.001"/> <fraction n="1.0" ref="O"/>
  </material>
  <material name="Metal" state="solid"> <D value="2.7"/> <fraction n="1.0" ref="O"/> </material>
</materials>
<solids>
  <box name="WorldBox" x="300" y="300" z="1" lunit="m"/>
  <box name="GasBox" x="200" y="200" z="0.02" lunit="m"/>
  <box name="CathBox" x="200" y="200" z="0.001" lunit="m"/>
  <opticalsurface name="CATHODE" model="unified" finish="polished" type="dielectric_metal" value="1">
    <property name="REFLECTIVITY" ref="R0"/> <property name="EFFICIENCY" ref="E4"/> </opticalsurface>
</solids>
<structure>
  <volume name="CathLV"> <materialref ref="Metal"/> <solidref ref="CathBox"/> </volume>
  <volume name="GasLV"> <materialref ref="Gas"/> <solidref ref="GasBox"/>
    <physvol name="cath"> <volumeref ref="CathLV"/> <position name="pc" unit="m" x="0" y="0" z="-0.0095"/> </physvol>
  </volume>
  <volume name="WorldLV"> <materialref ref="Metal"/> <solidref ref="WorldBox"/>
    <physvol name="gas"> <volumeref ref="GasLV"/> </physvol>
  </volume>
  {skin}
</structure>
<setup name="Default" version="1.0"> <world ref="WorldLV"/> </setup>
</gdml>
"""
SKIN = '<skinsurface name="CathSkin" surfaceproperty="CATHODE"> <volumeref ref="CathLV"/> </skinsurface>'


def _convert(tmp, text):
    import lumen
    from lumen.format.sensorsets import SensorGroup
    from lumen.recipe import Recipe

    (tmp / "slab.gdml").write_text(text)
    recipe = Recipe("slab", "GasLV", groups=(SensorGroup("cath", rule="border_into", lv="CathLV"),))
    return lumen.load(lumen.convert(tmp / "slab.gdml", recipe, tmp / "slab.lumen"))


@pytest.mark.slow  # converts a toy and compiles the photon loop: about a minute on 4 CPUs
def test_through_simulate_a_skin_sensor_detects_with_its_efficiency(tmp_path):
    import lumen
    from lumen.sources import FromArrays

    det = _convert(tmp_path, SLAB.format(skin=SKIN))
    assert len(det.sensor_table().name) == 1
    n, th = 4096, math.radians(30.0)
    src = FromArrays(np.tile([0.0, 0.0, 0.0], (n, 1)), np.tile([math.sin(th), 0.0, -math.cos(th)], (n, 1)),
                     np.zeros(n), np.full(n, 400.0), np.full(n, 1.0 / n))  # fmt: skip
    light = lumen.simulate(det, src, seed=1)
    assert abs(float(light.arrivals.sum()) - 0.4) < 1e-5  # absorbed (R = 0), detected with EFFICIENCY 0.4


def test_a_border_into_group_with_nothing_to_read_is_refused(tmp_path):
    from lumen.build.library import LibraryError

    with pytest.raises(LibraryError, match="would never detect"):
        _convert(tmp_path, SLAB.format(skin=""))


def test_a_skin_without_efficiency_is_no_sensor(tmp_path):
    """Nothing to detect with (no EFFICIENCY, no efficiency of the group's): refused, as a group with no skin is."""
    from lumen.build.library import LibraryError

    with pytest.raises(LibraryError, match="would never detect"):
        _convert(tmp_path, SLAB.format(skin=SKIN).replace('<property name="EFFICIENCY" ref="E4"/>', ""))
