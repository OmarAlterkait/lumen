"""Border surfaces between volumes that meet only across coincident faces (G4LogicalBorderSurface binds any two
physical volumes; the photon crosses from one straight into the other where a daughter's face lies on its mother's).

NEXT-100 has both patterns: the TPB coating inside the teflon light tube against the active gas beside the tube (the
coating's inner face on the tube's), and the SiPM board's coating, inside the mask inside the board, against the gas.
What is checked, at the surface lookup (the navigator reports such a crossing as p -> q, the zero-thickness volumes
between skipped):

- the border is read on the crossing it names, both ways, and nowhere else (a second coated tube without borders
  keeps the default interface);
- coincident faces compose through opaque volumes (coating -> mask -> board -> gas);
- a border between volumes that no coincident face joins is refused when the optics compile.
"""

import pytest
from test_surfaces import compile_, kern, tree

from lumen.build.lookup import OpticsError, nearest_instances
from lumen.surfaces import types

TUBE = (
    ("World", "WorldLV", "Water", None),  # the gas
    ("Active", "ActiveLV", "Water2", "World"),  # the active gas, beside the tube
    ("Tube", "TubeLV", "Steel", "World"),  # the light tube (opaque)
    ("Coat", "CoatLV", "Acrylic", "Tube"),  # its coating, the inner face on the tube's
    ("Tube2", "TubeLV", "Steel", "World"),  # another coated tube, no border
    ("Coat2", "CoatLV", "Acrylic", "Tube2"),
)
BOARD = (
    ("World", "WorldLV", "Water", None),  # the gas
    ("Board", "BoardLV", "Steel", "World"),  # the board (opaque)
    ("Mask", "MaskLV", "Steel304", "Board"),  # the mask (opaque), its face on the board's
    ("Coat", "CoatLV", "Acrylic", "Mask"),  # the coating, its face on the mask's
)


def test_a_border_across_a_coincident_face_is_read_both_ways_and_nowhere_else():
    t = tree(*TUBE)
    o = compile_(t, borders={("Active", "Coat"): "refl", ("Coat", "Active"): "refl2"})
    k, origin, reasons = kern(o, "Active", "Coat")
    assert k is types.REFLECTOR and origin == "surface:refl", (k, origin, reasons)
    assert any("coincident via" in r for r in reasons), reasons
    k, origin, _ = kern(o, "Coat", "Active")
    assert k.name.startswith("painted") and origin == "surface:refl2"
    # the tube without borders: the default interface (water -> acrylic), the coatings in two classes
    assert o.class_of["Coat"] != o.class_of["Coat2"]
    k, origin, _ = kern(o, "World", "Coat2")
    assert k is types.DIELECTRIC and origin is None
    assert nearest_instances(t, "Active", "Coat") == frozenset({("Active", "Coat")})


def test_a_daughter_against_its_mothers_sibling_is_a_coincident_border():
    """In (in Box) against Box2 (Box's sibling): where In's face lies on Box's, the photon steps from In into Box2."""
    t = tree(("World", "WorldLV", "Water", None), ("Box", "BoxLV", "Acrylic", "World"),
             ("Box2", "Box2LV", "Acrylic", "World"), ("In", "InLV", "Water", "Box"))  # fmt: skip
    o = compile_(t, borders={("In", "Box2"): "refl"})
    assert kern(o, "In", "Box2")[1] == "surface:refl" and kern(o, "In", "Box")[1] is None


def test_coincident_faces_compose_through_opaque_volumes():
    """coating -> gas across the mask's and the board's faces: (mask -> gas via the board) carries the chain."""
    o = compile_(tree(*BOARD), borders={("Coat", "World"): "refl", ("World", "Coat"): "refl2"})
    assert kern(o, "Coat", "World")[1] == "surface:refl"
    assert kern(o, "World", "Coat")[1] == "surface:refl2"


def test_a_border_no_coincident_face_joins_is_refused():
    """two coatings in two tubes: neither touches the other, nor across one face (a daughter's on its mother's)"""
    with pytest.raises(OpticsError, match="neither touch .* nor meet across a coincident face"):
        compile_(tree(*TUBE), borders={("Coat", "Coat2"): "refl"})


# ---------------------------------------------------------------------------------------------- through simulate
GDML_HEAD = """<?xml version="1.0" encoding="UTF-8" ?>
<gdml>
<define>
  <matrix name="R1" coldim="2" values="2.0*eV 1.0 4.5*eV 1.0"/>
  <matrix name="R15" coldim="2" values="2.0*eV 1.5 4.5*eV 1.5"/>
</define>
<materials>
  <element name="O" formula="O" Z="8"> <atom value="15.999"/> </element>
  <material name="Gas" state="gas"> <property name="RINDEX" ref="R1"/> <D value="0.001"/> <fraction n="1.0" ref="O"/>
  </material>
  <material name="Coating" state="solid"> <property name="RINDEX" ref="R15"/> <D value="1.0"/>
    <fraction n="1.0" ref="O"/> </material>
  <material name="Black" state="solid"> <D value="1.0"/> <fraction n="1.0" ref="O"/> </material>
  <material name="Kapton" state="solid"> <D value="1.4"/> <fraction n="1.0" ref="O"/> </material>
  <material name="Teflon" state="solid"> <D value="2.2"/> <fraction n="1.0" ref="O"/> </material>
</materials>
"""
MIRROR = '  <opticalsurface name="Mirror" model="glisur" finish="polished" type="dielectric_metal" value="1"/>\n'
# the board: gas over a board, its mask and the mask's coating, their top faces at z = -5 mm; a source 5 mm above
BOARD_GDML = (
    GDML_HEAD
    + """<solids>
  <box name="WorldBox" x="300" y="300" z="1" lunit="m"/>
  <box name="GasBox" x="200" y="200" z="0.022" lunit="m"/>
  <box name="BoardBox" x="200" y="200" z="0.004" lunit="m"/>
  <box name="MaskBox" x="200" y="200" z="0.002" lunit="m"/>
  <box name="CoatBox" x="200" y="200" z="0.0005" lunit="m"/>
  <box name="SenseBox" x="200" y="200" z="0.001" lunit="m"/>
{surfaces}</solids>
<structure>
  <volume name="CoatLV"> <materialref ref="Coating"/> <solidref ref="CoatBox"/> </volume>
  <volume name="MaskLV"> <materialref ref="Teflon"/> <solidref ref="MaskBox"/>
    <physvol name="coat"> <volumeref ref="CoatLV"/> <position name="pc" unit="m" x="0" y="0" z="0.00075"/> </physvol>
  </volume>
  <volume name="BoardLV"> <materialref ref="Kapton"/> <solidref ref="BoardBox"/>
    <physvol name="mask"> <volumeref ref="MaskLV"/> <position name="pm" unit="m" x="0" y="0" z="0.001"/> </physvol>
  </volume>
  <volume name="SenseLV"> <materialref ref="Gas"/> <solidref ref="SenseBox"/> </volume>
  <volume name="GasLV"> <materialref ref="Gas"/> <solidref ref="GasBox"/>
    <physvol name="board"> <volumeref ref="BoardLV"/> <position name="pb" unit="m" x="0" y="0" z="-0.007"/> </physvol>
    <physvol name="sense"> <volumeref ref="SenseLV"/> <position name="ps" unit="m" x="0" y="0" z="0.0105"/> </physvol>
  </volume>
  <volume name="WorldLV"> <materialref ref="Black"/> <solidref ref="WorldBox"/>
    <physvol name="gas"> <volumeref ref="GasLV"/> </physvol>
  </volume>
{borders}</structure>
<setup name="Default" version="1.0"> <world ref="WorldLV"/> </setup>
</gdml>
"""
)
BOARD_BORDER = '  <bordersurface name="GasCoat" surfaceproperty="Mirror"> <physvolref ref="gas"/> ' \
               '<physvolref ref="coat"/> </bordersurface>\n'  # fmt: skip
# the tube: active gas in a teflon tube whose coating's inner face is the tube's; counting caps on its ends
TUBE_GDML = (
    GDML_HEAD
    + """<solids>
  <box name="WorldBox" x="4" y="4" z="4" lunit="m"/>
  <tube name="GasTube" rmax="0.5" z="1.2" deltaphi="360" aunit="deg" lunit="m"/>
  <tube name="ActiveTube" rmax="0.1" z="1.0" deltaphi="360" aunit="deg" lunit="m"/>
  <tube name="WallTube" rmin="0.1" rmax="0.12" z="1.0" deltaphi="360" aunit="deg" lunit="m"/>
  <tube name="CoatTube" rmin="0.1" rmax="0.101" z="1.0" deltaphi="360" aunit="deg" lunit="m"/>
  <tube name="CapTube" rmax="0.12" z="0.001" deltaphi="360" aunit="deg" lunit="m"/>
{surfaces}</solids>
<structure>
  <volume name="CoatLV"> <materialref ref="Coating"/> <solidref ref="CoatTube"/> </volume>
  <volume name="WallLV"> <materialref ref="Teflon"/> <solidref ref="WallTube"/>
    <physvol name="coat"> <volumeref ref="CoatLV"/> </physvol>
  </volume>
  <volume name="ActiveLV"> <materialref ref="Gas"/> <solidref ref="ActiveTube"/> </volume>
  <volume name="CapLV"> <materialref ref="Gas"/> <solidref ref="CapTube"/> </volume>
  <volume name="GasLV"> <materialref ref="Gas"/> <solidref ref="GasTube"/>
    <physvol name="active"> <volumeref ref="ActiveLV"/> </physvol>
    <physvol name="wall"> <volumeref ref="WallLV"/> </physvol>
    <physvol name="cap_top" copynumber="1"> <volumeref ref="CapLV"/>
      <position name="pt" unit="m" x="0" y="0" z="0.5005"/> </physvol>
    <physvol name="cap_bottom" copynumber="2"> <volumeref ref="CapLV"/>
      <position name="pd" unit="m" x="0" y="0" z="-0.5005"/> </physvol>
  </volume>
  <volume name="WorldLV"> <materialref ref="Black"/> <solidref ref="WorldBox"/>
    <physvol name="gas"> <volumeref ref="GasLV"/> </physvol>
  </volume>
{borders}</structure>
<setup name="Default" version="1.0"> <world ref="WorldLV"/> </setup>
</gdml>
"""
)
TUBE_BORDER = '  <bordersurface name="ActiveCoat" surfaceproperty="Mirror"> <physvolref ref="active"/> ' \
              '<physvolref ref="coat"/> </bordersurface>\n'  # fmt: skip


def _counted(tmp, name, text, sensor_lv, source):
    """Return (the share of a one-photon source's light that the counting sensors of ``sensor_lv`` count, the share
    lost)."""
    import lumen
    from lumen.format.sensorsets import SensorGroup
    from lumen.recipe import Recipe

    (tmp / f"{name}.gdml").write_text(text)
    recipe = Recipe(name, "GasLV", groups=(SensorGroup("caps", rule="any_entry", lv=sensor_lv,
                                                       settings={"entry_efficiency": 1.0}),))  # fmt: skip
    det = lumen.load(lumen.convert(tmp / f"{name}.gdml", recipe, tmp / f"{name}.lumen"))
    light = lumen.simulate(det, source, samples=1 << 15, seed=3)
    return float(light.arrivals.sum()), float(light.health.lost[0])


@pytest.mark.slow  # converts and simulates four toys: minutes on 4 CPUs
@pytest.mark.parametrize("toy", ["board", "tube"])
def test_through_simulate_a_mirror_border_across_coincident_faces_keeps_all_the_light(tmp_path, toy):
    """With a mirror on the coincident face every photon reaches the counting sensors (the board's: a slab above
    everything; the tube's: its end caps), none lost; without the border the coating's interface lets light into the
    opaque volume behind it."""
    from lumen.sources import Point

    gdml, border, lv = (BOARD_GDML, BOARD_BORDER, "SenseLV") if toy == "board" else (TUBE_GDML, TUBE_BORDER, "CapLV")
    src = Point((0.0, 0.0, 0.0), n_photons=1.0, lam=400.0)
    kept, lost = _counted(tmp_path, f"{toy}_mirror", gdml.format(surfaces=MIRROR, borders=border), lv, src)
    assert abs(kept - 1.0) < 1e-3 and lost == 0.0, (toy, kept, lost)
    plain, _ = _counted(tmp_path, f"{toy}_plain", gdml.format(surfaces="", borders=""), lv, src)
    assert plain < 0.9, (toy, plain)
