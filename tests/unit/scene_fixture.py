"""A small GDML exercising the lowering (mm; materials are names only), and a loader for it.

  World (Air, 2 m box)
    Tank (Water, 1.6 m box) tank_pv
      Env (Water box 400) env_pv at (500, 0, 0), env2_pv at (500, 450, 0)    -- navigation envelopes (dissolvable)
        Pmt (Glass orb 100) pmt_pv at (0, 0, 50)
          Cath (Vacuum orb 50) cath_pv                                       -- the sensor (by name)
      Flag (Steel: a 200 x 20 x 20 bar + a 20 mm cube at (90, 30)) flag_pv at (-500, 0, 0), rotation z = 90 deg
                                                         (passive: the cube ends up at (-500, 0) + (30, -90))
      Bool (Steel: a 20 mm cube united with the flag solid rotated z = 90 deg as a boolean second: active, the cube at
            (-30, 90)) bool_pv at (0, 500, 0)
      Flag mirrored: refl_pv at (0, -500, 0), scale (-1, 1, 1)               (the cube at (-90, 30))
      Asm (an assembly) asm_pv at (0, 0, 500): Small (Steel 50 mm cube) small_pv at (100, 0, 0)
      Sheet (Steel: 600 x 600 x 20 box minus a 50 mm orb at (100, 0, 0)) sheet_pv at (0, 0, -500)
      Plug (Glass, the same orb) plug_pv at (100, 0, -500)                   -- the sheet's hole is this sibling
      Frame (Steel tessellated tetrahedron) frame_pv at (-300, -300, 300)   -- an opaque mesh

Ported from the baseline's tests/scene_fixture.py.
"""

from lumen.build import lowering
from lumen.gdml import importer

GDML = """<?xml version="1.0"?>
<gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="gdml.xsd">
 <define>
  <rotation name="rz90" z="90" unit="deg"/>
  <scale name="mirx" x="-1" y="1" z="1"/>
  <position name="v0" x="0" y="0" z="0"/><position name="v1" x="100" y="0" z="0"/>
  <position name="v2" x="0" y="100" z="0"/><position name="v3" x="0" y="0" z="100"/>
 </define>
 <materials/>
 <solids>
  <box name="wbox" x="2000" y="2000" z="2000" lunit="mm"/>
  <box name="tankbox" x="1600" y="1600" z="1600" lunit="mm"/>
  <box name="envbox" x="400" y="400" z="400" lunit="mm"/>
  <orb name="pmtorb" r="100" lunit="mm"/>
  <orb name="cathorb" r="50" lunit="mm"/>
  <box name="bar" x="200" y="20" z="20" lunit="mm"/>
  <box name="cube" x="20" y="20" z="20" lunit="mm"/>
  <union name="flagsol"><first ref="bar"/><second ref="cube"/>
   <position name="fp" x="90" y="30" z="0" unit="mm"/></union>
  <union name="boolsol"><first ref="cube"/><second ref="flagsol"/><rotationref ref="rz90"/></union>
  <box name="smallbox" x="50" y="50" z="50" lunit="mm"/>
  <box name="sheetbox" x="600" y="600" z="20" lunit="mm"/>
  <orb name="holeorb" r="50" lunit="mm"/>
  <subtraction name="sheetsol"><first ref="sheetbox"/><second ref="holeorb"/>
   <position name="hp" x="100" y="0" z="0" unit="mm"/></subtraction>
  <tessellated name="tetmesh" lunit="mm">
   <triangular vertex1="v0" vertex2="v2" vertex3="v1"/><triangular vertex1="v0" vertex2="v1" vertex3="v3"/>
   <triangular vertex1="v0" vertex2="v3" vertex3="v2"/><triangular vertex1="v1" vertex2="v2" vertex3="v3"/>
  </tessellated>
 </solids>
 <structure>
  <volume name="Cath"><materialref ref="Vacuum"/><solidref ref="cathorb"/></volume>
  <volume name="Pmt"><materialref ref="Glass"/><solidref ref="pmtorb"/>
   <physvol name="cath_pv"><volumeref ref="Cath"/></physvol></volume>
  <volume name="Env"><materialref ref="Water"/><solidref ref="envbox"/>
   <physvol name="pmt_pv"><volumeref ref="Pmt"/><position name="pp" x="0" y="0" z="50" unit="mm"/></physvol></volume>
  <volume name="Flag"><materialref ref="Steel"/><solidref ref="flagsol"/></volume>
  <volume name="Bool"><materialref ref="Steel"/><solidref ref="boolsol"/></volume>
  <volume name="Small"><materialref ref="Steel"/><solidref ref="smallbox"/></volume>
  <assembly name="Asm"><physvol name="small_pv"><volumeref ref="Small"/>
   <position name="sp" x="100" y="0" z="0" unit="mm"/></physvol></assembly>
  <volume name="Sheet"><materialref ref="Steel"/><solidref ref="sheetsol"/></volume>
  <volume name="Plug"><materialref ref="Glass"/><solidref ref="holeorb"/></volume>
  <volume name="Frame"><materialref ref="Steel"/><solidref ref="tetmesh"/></volume>
  <volume name="Tank"><materialref ref="Water"/><solidref ref="tankbox"/>
   <physvol name="env_pv"><volumeref ref="Env"/><position name="e1" x="500" y="0" z="0" unit="mm"/></physvol>
   <physvol name="env2_pv"><volumeref ref="Env"/><position name="e2" x="500" y="450" z="0" unit="mm"/></physvol>
   <physvol name="flag_pv"><volumeref ref="Flag"/><position name="f1" x="-500" y="0" z="0" unit="mm"/>
    <rotationref ref="rz90"/></physvol>
   <physvol name="bool_pv"><volumeref ref="Bool"/><position name="b1" x="0" y="500" z="0" unit="mm"/></physvol>
   <physvol name="refl_pv"><volumeref ref="Flag"/><position name="r1" x="0" y="-500" z="0" unit="mm"/>
    <scaleref ref="mirx"/></physvol>
   <physvol name="asm_pv"><volumeref ref="Asm"/><position name="a1" x="0" y="0" z="500" unit="mm"/></physvol>
   <physvol name="sheet_pv"><volumeref ref="Sheet"/><position name="s1" x="0" y="0" z="-500" unit="mm"/></physvol>
   <physvol name="plug_pv"><volumeref ref="Plug"/><position name="g1" x="100" y="0" z="-500" unit="mm"/></physvol>
   <physvol name="frame_pv"><volumeref ref="Frame"/><position name="m1" x="-300" y="-300" z="300" unit="mm"/></physvol>
  </volume>
  <volume name="World"><materialref ref="Air"/><solidref ref="wbox"/>
   <physvol name="tank_pv"><volumeref ref="Tank"/></physvol></volume>
 </structure>
 <setup name="Default" version="1.0"><world ref="World"/></setup>
</gdml>
"""

# (point [m], material, placement) under Geant4's conventions
POINTS = [
    ((0.0, 0.0, 0.0), "Water", "tank_pv"),
    ((0.5, 0.0, 0.05), "Vacuum", "cath_pv"),
    ((0.5, 0.0, 0.12), "Glass", "pmt_pv"),
    ((0.5, 0.45, 0.05), "Vacuum", "cath_pv"),
    ((-0.47, -0.09, 0.0), "Steel", "flag_pv"),  # physvol rotation: passive
    ((-0.53, 0.09, 0.0), "Water", "tank_pv"),
    ((-0.03, 0.59, 0.0), "Steel", "bool_pv"),  # boolean rotation: active
    ((0.03, 0.41, 0.0), "Water", "tank_pv"),
    ((-0.09, -0.47, 0.0), "Steel", "refl_pv"),  # reflected placement
    ((0.09, -0.47, 0.0), "Water", "tank_pv"),
    ((0.1, 0.0, 0.5), "Steel", "small_pv"),  # assembly
    ((0.1, 0.0, -0.5), "Glass", "plug_pv"),  # the sheet's hole: the plug
    ((0.0, 0.0, -0.5), "Steel", "sheet_pv"),
    ((0.95, 0.95, 0.95), "Air", None),  # outside the tank
]


def write(tmp_path, name="scene.gdml", text=GDML):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def load(tmp_path, text=GDML, name="scene.gdml", **kw):
    """The fixture (or a variant of its text) imported and lowered: ``lowering.lower(read_gdml(path), **kw)``."""
    return lowering.lower(importer.read_gdml(write(tmp_path, name, text)), **kw)
