"""The overlap-free GDML (lumen.compare.ownership): lumen's ownership of overlapping siblings written
out in Geant4's terms -- the same materials everywhere, containment and separation proved from the solids'
parameters, skins and borders kept on every variant and copy, a split union written whole."""

import dataclasses
import math

import jax
import numpy as np

from lumen.build import lowering
from lumen.compare import ownership as owned
from lumen.gdml import importer
from lumen.geometry import navigate

I3 = np.eye(3)


def write(tmp_path, text, name="toy.gdml"):
    p = tmp_path / name
    p.write_text(text)
    return p


TOY = """<?xml version="1.0"?>
<gdml>
 <materials>
  <element name="H" formula="H" Z="1"><atom value="1.008"/></element>
  <material name="MW"><D value="1"/><fraction n="1" ref="H"/></material>
  <material name="MA"><D value="1"/><fraction n="1" ref="H"/></material>
  <material name="MB"><D value="1"/><fraction n="1" ref="H"/></material>
  <material name="MC"><D value="1"/><fraction n="1" ref="H"/></material>
 </materials>
 <solids>
  <box name="wbox" x="4000" y="4000" z="4000" lunit="mm"/>
  <box name="tank" x="2000" y="2000" z="2000" lunit="mm"/>
  <box name="big" x="400" y="400" z="400" lunit="mm"/>
  <box name="plane" x="10" y="400" z="404" lunit="mm"/>
  <tube name="lens" rmax="100" z="60" deltaphi="360" aunit="deg" lunit="mm"/>
  <tube name="under" rmax="100" z="60" deltaphi="360" aunit="deg" lunit="mm"/>
 </solids>
 <structure>
  <volume name="volPlane"><materialref ref="MB"/><solidref ref="plane"/></volume>
  <volume name="volBig"><materialref ref="MA"/><solidref ref="big"/></volume>
  <volume name="volLens"><materialref ref="MA"/><solidref ref="lens"/></volume>
  <volume name="volUnder"><materialref ref="MC"/><solidref ref="under"/></volume>
  <volume name="volTank"><materialref ref="MW"/><solidref ref="tank"/>
   <physvol name="pPlane"><volumeref ref="volPlane"/><position name="xp" x="150" y="0" z="0" unit="mm"/></physvol>
   <physvol name="pBig"><volumeref ref="volBig"/></physvol>
   <physvol name="pLens"><volumeref ref="volLens"/><position name="xl" x="600" y="0" z="0" unit="mm"/></physvol>
   <physvol name="pUnder"><volumeref ref="volUnder"/><position name="xu" x="600" y="0" z="0" unit="mm"/></physvol>
  </volume>
  <volume name="volWorld"><materialref ref="MW"/><solidref ref="wbox"/>
   <physvol name="pTank"><volumeref ref="volTank"/></physvol>
  </volume>
 </structure>
 <setup name="Default" version="1.0"><world ref="volWorld"/></setup>
</gdml>
"""


def test_the_overlap_free_export_holds_the_same_materials_everywhere(tmp_path):
    """A plane (placed first: the owner) inside a box but 2 mm past it in z: the box becomes a subtraction; a lens and
    an underside coincident (each inside the other: the lens, placed first, owns): the underside holds nothing and is
    left out. Read back, every point is in a volume of the same material."""
    df = dataclasses.replace(importer.read_gdml(write(tmp_path, TOY)), detector_volume="volTank")
    df2, notes = owned.owned(df)
    assert any("volBig in volTank less plane" in n for n in notes), notes  # the solids it is less
    assert any("volUnder in volTank left out: inside lens" in n for n in notes), notes
    assert "volUnder" not in {p.volume for p in df2.placements}
    assert df2.solids[df2.volumes["volBig"].solid].type == "subtraction"
    from lumen import gdml

    gdml.export(df2, tmp_path / "owned.gdml")
    back = dataclasses.replace(importer.read_gdml(tmp_path / "owned.gdml"), detector_volume="volTank")

    def nav_media(d):
        ld = lowering.lower(d, detector="volTank")
        nav = navigate.Navigator.from_scene(lowering.split_unions(ld.scene))
        return nav, {v: k for k, v in ld.media.items()}

    (n1, m1), (n2, m2) = nav_media(df), nav_media(back)
    x = np.random.default_rng(5).uniform(-0.99, 0.99, size=(20000, 3)).astype(np.float32)
    x[:5000, 1:] *= 0.2  # through the box, the plane and the PMT
    mat = [np.array([m.get(int(c), "?") for c in np.asarray(n.host["cls"])[np.asarray(jax.jit(jax.vmap(n.locate))(x))]])
           for n, m in ((n1, m1), (n2, m2))]  # fmt: skip
    assert np.array_equal(*mat), np.unique(np.stack(mat, 1)[mat[0] != mat[1]], axis=0)
    assert {"MA", "MB", "MW"} <= set(mat[0]) and "MC" not in set(mat[0])  # the underside holds nothing


def test_gdml_angles_invert_the_lowerings_rotation():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.uniform(-math.pi, math.pi, 3) * np.array([1, 0.5, 1])
        L = lowering.rot_from_angles(*a).T  # a boolean's active matrix
        assert np.allclose(lowering.rot_from_angles(*owned.gdml_angles(L)).T, L, atol=1e-12)


PROOF_TOY = """<?xml version="1.0"?>
<gdml>
 <materials>
  <element name="H" formula="H" Z="1"><atom value="1.008"/></element>
  <material name="MW"><D value="1"/><fraction n="1" ref="H"/></material>
 </materials>
 <solids>
  <box name="wbox" x="10" y="10" z="12" lunit="m"/>
  <tube name="cryo" rmax="193" z="1091.485" deltaphi="360" aunit="deg" lunit="cm"/>
  <tube name="wall" rmin="190.46" rmax="192.9" z="1086.485" deltaphi="360" aunit="deg" lunit="cm"/>
  <tube name="wire" rmax="0.0075" z="465.97" deltaphi="360" aunit="deg" lunit="cm"/>
  <box name="tpc" x="260" y="256" z="1200" lunit="cm"/>
  <tube name="lens" rmax="10.16" z="6.35" deltaphi="360" aunit="deg" lunit="cm"/>
  <tube name="under" rmax="10.16" z="6.35" deltaphi="360" aunit="deg" lunit="cm"/>
 </solids>
 <structure>
  <volume name="volWorld"><materialref ref="MW"/><solidref ref="wbox"/></volume>
 </structure>
 <setup name="Default" version="1.0"><world ref="volWorld"/></setup>
</gdml>
"""


def test_the_export_proves_containment_and_separation_from_the_parameters(tmp_path):
    """A 0.15 mm wire 4.66 m long at 30 degrees, 1.28 m off the axis of a 1.93 m cryostat: inside it, apart from its
    1.9 m steel wall (the box test alone proves neither: the wire's box is 4.66 m long); a TPC box longer than the
    cryostat: neither; one solid at one pose: inside; a wire touching the wall's inner face from inside: not proved
    apart (a contact is not decided by bisection)."""
    sol = owned._Solids(importer.read_gdml(write(tmp_path, PROOF_TOY)))
    L = lowering.rot_from_angles(math.pi / 6, 0.0, 0.0).T
    t = np.array([-1.28, 0.0, 2.6])  # its ends at y = +-1.165 m, z = 0.58 and 4.62 m
    assert sol.relate("wire", "cryo", L, t, "inside") and not sol.relate("wire", "cryo", L, t, "apart")
    assert sol.relate("wire", "wall", L, t, "apart") and not sol.relate("wire", "wall", L, t, "inside")
    assert not sol.relate("tpc", "cryo", I3, np.zeros(3), "inside")
    assert not sol.relate("tpc", "cryo", I3, np.zeros(3), "apart")
    assert sol.relate("under", "lens", I3, np.zeros(3), "inside")
    assert not sol.relate("under", "lens", I3, np.array([0.0, 0.0, 1e-3]), "inside")
    up = np.array([0.0, 1.9046 - 0.000075, 0.0])  # the wire along z, its surface on the wall's inner radius
    assert not sol.relate("wire", "wall", I3, up, "apart")
    assert sol.relate("wire", "wall", I3, up - [0.0, 1e-4, 0.0], "apart")


BORDER_TOY = """<?xml version="1.0"?>
<gdml>
 <define>
  <matrix name="REF" coldim="2" values="2.0*eV 0.9 4.5*eV 0.8"/>
 </define>
 <materials>
  <element name="H" formula="H" Z="1"><atom value="1.008"/></element>
  <material name="MW"><D value="1"/><fraction n="1" ref="H"/></material>
  <material name="MA"><D value="1"/><fraction n="1" ref="H"/></material>
  <material name="MB"><D value="1"/><fraction n="1" ref="H"/></material>
 </materials>
 <solids>
  <box name="wbox" x="4000" y="4000" z="4000" lunit="mm"/>
  <box name="tank" x="2000" y="2000" z="2000" lunit="mm"/>
  <box name="cell" x="400" y="400" z="400" lunit="mm"/>
  <box name="small" x="100" y="100" z="100" lunit="mm"/>
  <opticalsurface name="Foil" model="unified" finish="ground" type="dielectric_metal" value="0.1">
   <property name="REFLECTIVITY" ref="REF"/>
  </opticalsurface>
 </solids>
 <structure>
  <volume name="volGlass"><materialref ref="MA"/><solidref ref="small"/></volume>
  <volume name="volInner"><materialref ref="MB"/><solidref ref="small"/></volume>
  <volume name="volPlate"><materialref ref="MB"/><solidref ref="small"/></volume>
  <volume name="volCell"><materialref ref="MW"/><solidref ref="cell"/>
   <physvol name="pGlass"><volumeref ref="volGlass"/></physvol>
   <physvol name="pInner"><volumeref ref="volInner"/><position name="xi" x="100" y="0" z="0" unit="mm"/></physvol>
  </volume>
  <volume name="volTank"><materialref ref="MW"/><solidref ref="tank"/>
   <physvol name="pPlate"><volumeref ref="volPlate"/><position name="xq" x="680" y="0" z="0" unit="mm"/></physvol>
   <physvol name="pCellA"><volumeref ref="volCell"/><position name="xa" x="-500" y="0" z="0" unit="mm"/></physvol>
   <physvol name="pCellB"><volumeref ref="volCell"/><position name="xb" x="500" y="0" z="0" unit="mm"/></physvol>
  </volume>
  <volume name="volWorld"><materialref ref="MW"/><solidref ref="wbox"/>
   <physvol name="pTank"><volumeref ref="volTank"/></physvol>
  </volume>
  <skinsurface name="CellSkin" surfaceproperty="Foil"><volumeref ref="volCell"/></skinsurface>
  <bordersurface name="GlassInner" surfaceproperty="Foil"><physvolref ref="pGlass"/><physvolref ref="pInner"/>
  </bordersurface>
 </structure>
 <setup name="Default" version="1.0"><world ref="volWorld"/></setup>
</gdml>
"""


def test_the_export_keeps_skins_and_borders_on_every_variant_and_copy(tmp_path):
    """The plate (placed first) owns its overlap with cell B and with B's inner box: cell B becomes a variant of
    volCell, its glass and inner boxes copies of pGlass and pInner. Each copy gets a name of its own, the border names
    every pair of copies (so it holds in both cells), the skin is on volCell and its variant, and the export is
    accepted."""
    df = dataclasses.replace(importer.read_gdml(write(tmp_path, BORDER_TOY)), detector_volume="volTank")
    df2, notes = owned.owned(df)
    assert any("volCell in volTank less small" in n for n in notes), notes
    names = [p.name for p in df2.placements]
    glass = [p.name for p in df2.placements if p.volume == "volGlass"]
    inner = [p.name for p in df2.placements if p.volume.startswith("volInner")]
    assert len(glass) == len(inner) == 2 and all(names.count(n) == 1 for n in glass + inner), names
    assert sorted((b.pv1, b.pv2) for b in df2.borders) == sorted((g, i) for g in glass for i in inner)
    assert {b.surface for b in df2.borders} == {"Foil"} and len({b.name for b in df2.borders}) == 4
    cells = {p.volume for p in df2.placements if p.name in ("pCellA", "pCellB")}
    assert len(cells) == 2 and {k.volume for k in df2.skins} == cells
    from lumen import gdml

    gdml.export(df2, tmp_path / "owned.gdml")  # the binding check passes


def test_the_export_writes_a_split_union_whole(tmp_path):
    """A union of 16 overlapping boxes (placed by lumen as its pieces, which overlap: the first placed owns) and a
    box overlapping it: the union is cut by the box only, never by its own pieces, and nothing is left out."""
    sol = ['<box name="b" x="100" y="100" z="100" lunit="mm"/>', '<box name="probe" x="50" y="50" z="50" lunit="mm"/>']
    prev = "b"
    for k in range(1, 16):
        sol.append(f'<union name="u{k}"><first ref="{prev}"/><second ref="b"/>'
                   f'<position name="pu{k}" x="{20 * k}" y="0" z="0" unit="mm"/></union>')  # fmt: skip
        prev = f"u{k}"
    text = (
        TOY.replace('<box name="big" x="400" y="400" z="400" lunit="mm"/>', "\n".join(sol))
        .replace('<solidref ref="big"/>', f'<solidref ref="{prev}"/>')
        .replace('<solidref ref="plane"/>', '<solidref ref="probe"/>')
    )
    df = dataclasses.replace(importer.read_gdml(write(tmp_path, text)), detector_volume="volTank")
    _, notes = owned.owned(df)
    big = [n for n in notes if n.startswith("volBig ")]
    assert big == ["volBig in volTank less probe"], notes


FLUSH_TOY = (
    TOY.replace(
        '<tube name="under" rmax="100" z="60" deltaphi="360" aunit="deg" lunit="mm"/>',
        '<tube name="under" rmax="100" z="60" deltaphi="360" aunit="deg" lunit="mm"/>\n'
        '  <tube name="inner" rmax="100" z="30" deltaphi="360" aunit="deg" lunit="mm"/>\n'
        '  <box name="side" x="100" y="100" z="100" lunit="mm"/>',
    )
    .replace(
        '<volume name="volLens"><materialref ref="MA"/><solidref ref="lens"/></volume>',
        '<volume name="volInner"><materialref ref="MB"/><solidref ref="inner"/></volume>\n'
        '  <volume name="volSide"><materialref ref="MB"/><solidref ref="side"/></volume>\n'
        '  <volume name="volLens"><materialref ref="MA"/><solidref ref="lens"/>\n'
        '   <physvol name="pInner"><volumeref ref="volInner"/>'
        '<position name="xn" x="0" y="0" z="15" unit="mm"/></physvol>\n'
        "  </volume>",
    )
    .replace(
        '<physvol name="pUnder"><volumeref ref="volUnder"/><position name="xu" x="600" y="0" z="0" unit="mm"/>'
        "</physvol>",
        '<physvol name="pSide"><volumeref ref="volSide"/><position name="xs" x="250" y="0" z="0" unit="mm"/></physvol>',
    )
)


def test_the_export_writes_what_already_holds_in_geant4_as_it_is(tmp_path):
    """A disc flush with its lens's face and rim (surfaces that coincide: not proved inside, but the placement check
    finds no point of it outside) and a box touching the big box face to face: the export is the detector file as it
    is; only the plane, overlapping the big box 2 mm deep, is rewritten."""
    df = dataclasses.replace(importer.read_gdml(write(tmp_path, FLUSH_TOY)), detector_volume="volTank")
    df2, notes = owned.owned(df)
    assert notes == ["volBig in volTank less plane"], notes
    assert df2.volumes["volInner"] == df.volumes["volInner"] and df2.volumes["volLens"] == df.volumes["volLens"]
    assert df2.volumes["volSide"] == df.volumes["volSide"]
    apart = FLUSH_TOY.replace('name="xp" x="150"', 'name="xp" x="-700"')  # the plane away from the big box
    df = dataclasses.replace(importer.read_gdml(write(tmp_path, apart, "apart.gdml")), detector_volume="volTank")
    assert owned.owned(df) == (
        df,
        ["no overlapping siblings, no daughter reaching out of its mother: written as it is"],
    )
