"""GDML toys for the exporter's tests (gdml_export.md section 7.2, toys a-k, and one per supported solid type).

Each toy is a GDML string; ``TOYS`` maps a name to it. They are small enough to load in Geant4 as well
(tools/port/check_export.py writes them out for the Geant4 load and probe check).
"""

HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
"""
MATERIALS = """ <materials>
  <isotope name="U235" Z="92" N="235"><atom unit="g/mole" value="235.0439"/></isotope>
  <isotope name="U238" Z="92" N="238"><atom value="238.0508"/></isotope>
  <element name="enriched_U"><fraction ref="U235" n="0.9"/><fraction ref="U238" n="0.1"/></element>
  <element name="H" formula="H" Z="1"><atom value="1.00794"/></element>
  <element name="O" formula="O" Z="8"><atom unit="g/mole" value="15.9994"/></element>
  <element name="C" formula="C" Z="6."><atom value="12.0107"/></element>
  <material name="Water" state="liquid">
   <T unit="K" value="293.15"/><P unit="pascal" value="101325.1"/><MEE unit="eV" value="78.0000000001"/>
   <D value="1.0" unit="g/cm3"/><composite n="2" ref="H"/><composite n="1" ref="O"/>
   <property name="RINDEX" ref="RIX"/><property name="ABSLENGTH" ref="ABS"/>{water_props}
  </material>
  <material name="Air" state="gas"><D value="0.00120479"/><fraction n="0.7" ref="O"/><fraction n="0.3" ref="C"/>
   <property name="RINDEX" ref="RIX1"/></material>
  <material name="Fuel"><D value="19.0"/><fraction n="1" ref="enriched_U"/></material>
  <material name="Mix" state="solid"><D value="1.5"/><fraction n="0.25" ref="Water"/>
   <fraction n="0.75" ref="Air"/></material>
  <material name="Graphite" Z="6"><D value="2.26"/><atom value="12.011"/></material>
  <material name="Black"><D value="1.0"/><fraction n="1" ref="C"/></material>
 </materials>
"""
DEFINE = """ <define>
  <matrix name="RIX" coldim="2" values="1.5*eV 1.33 2.5*eV 1.34 4.5e-06 1.35"/>
  <matrix name="RIX1" coldim="2" values="1.5*eV 1.0 4.5*eV 1.0"/>
  <matrix name="ABS" coldim="2" values="1.5*eV 10*m 4.5*eV 25000.5"/>
  <matrix name="REF" coldim="2" values="1.5*eV 0.9 4.5*eV 0.85"/>
  <matrix name="EFF" coldim="2" values="1.5*eV 0.2 4.5*eV 0.3"/>
  <matrix name="C1" coldim="1" values="0.25"/>
  <matrix name="LEN1" coldim="1" values="12.5"/>
{define}
 </define>
"""


def toy(solids="", volumes="", physvols="", define="", structure="", water_props="", tail=""):
    """Return a GDML file: the materials above, a 4 m world box of Air and the given parts."""
    return (HEAD + DEFINE.format(define=define) + MATERIALS.format(water_props=water_props)
            + f""" <solids>
  <box name="sWorld" x="4" y="4" z="4" lunit="m"/>
{solids}
 </solids>
 <structure>
{volumes}
  <volume name="World"><materialref ref="Air"/><solidref ref="sWorld"/>
{physvols}
  </volume>
{structure}
 </structure>
{tail} <setup name="Default" version="1.0"><world ref="World"/></setup>
</gdml>
""")  # fmt: skip


def vol(name, solid, material="Water", inner=""):
    """Return a <volume> element."""
    return f'  <volume name="{name}"><materialref ref="{material}"/><solidref ref="{solid}"/>{inner}</volume>'


def pv(lv, name=None, pos=None, rot=None, scale=None, copy=None):
    """Return a <physvol> element (position in mm, rotation in rad)."""
    a = (f' name="{name}"' if name else "") + (f' copynumber="{copy}"' if copy is not None else "")
    inner = f'<volumeref ref="{lv}"/>'
    if pos:
        inner += '<position name="p" unit="mm" x="{}" y="{}" z="{}"/>'.format(*pos)
    if rot:
        inner += '<rotation name="r" unit="rad" x="{}" y="{}" z="{}"/>'.format(*rot)
    if scale:
        inner += '<scale name="s" x="{}" y="{}" z="{}"/>'.format(*scale)
    return f"   <physvol{a}>{inner}</physvol>"


SURFACES = """  <opticalsurface name="Glisur" model="glisur" finish="ground" type="dielectric_metal" value="0.7">
   <property name="REFLECTIVITY" ref="REF"/></opticalsurface>
  <opticalsurface name="Unified" model="unified" finish="ground" type="dielectric_dielectric" value="0.2">
   <property name="REFLECTIVITY" ref="REF"/><property name="SPECULARLOBECONSTANT" ref="C1"/></opticalsurface>
  <opticalsurface name="Sensor" model="1" finish="0" type="0" value="1">
   <property name="EFFICIENCY" ref="EFF"/><property name="REFLECTIVITY" ref="REF"/></opticalsurface>"""

# every solid type the exporter writes, with all its attributes (GDML spelling; lengths mm, angles rad)
SOLIDS = {
    "box": '<box name="s_box" x="10" y="20" z="30" lunit="mm"/>',
    "tube": '<tube name="s_tube" rmin="1" rmax="5" z="10" startphi="0.1" deltaphi="6" aunit="rad" lunit="mm"/>',
    "cutTube": '<cutTube name="s_cutTube" rmin="1" rmax="5" z="10" startphi="0" deltaphi="6.283185307179586" '
               'lowX="0" lowY="-0.1" lowZ="-1" highX="0.1" highY="0" highZ="1" aunit="rad" lunit="mm"/>',
    "cone": '<cone name="s_cone" rmin1="0" rmax1="5" rmin2="1" rmax2="3" z="8" startphi="0" deltaphi="360" '
            'aunit="deg" lunit="mm"/>',
    "polycone": '<polycone name="s_polycone" startphi="0" deltaphi="6.283185307179586" aunit="rad" lunit="mm">'
                '<zplane rmin="0" rmax="3" z="-5"/><zplane rmin="1" rmax="4" z="0"/><zplane rmin="0" rmax="2" z="5"/>'
                "</polycone>",
    "genericPolycone": '<genericPolycone name="s_genericPolycone" startphi="0" deltaphi="6.283185307179586" '
                       'aunit="rad" lunit="mm"><rzpoint r="1" z="-5"/><rzpoint r="3" z="-5"/><rzpoint r="3" z="5"/>'
                       '<rzpoint r="1" z="5"/></genericPolycone>',
    "polyhedra": '<polyhedra name="s_polyhedra" startphi="0" deltaphi="6.283185307179586" numsides="6" aunit="rad" '
                 'lunit="mm"><zplane rmin="0" rmax="3" z="-5"/><zplane rmin="0" rmax="3" z="5"/></polyhedra>',
    "genericPolyhedra": '<genericPolyhedra name="s_genericPolyhedra" startphi="0" deltaphi="6.283185307179586" '
                        'numsides="5" aunit="rad" lunit="mm"><rzpoint r="1" z="-5"/><rzpoint r="3" z="-5"/>'
                        '<rzpoint r="3" z="5"/></genericPolyhedra>',
    "sphere": '<sphere name="s_sphere" rmin="1" rmax="5" startphi="0" deltaphi="6" starttheta="0.1" deltatheta="2" '
              'aunit="rad" lunit="mm"/>',
    "orb": '<orb name="s_orb" r="4.5" lunit="cm"/>',
    "ellipsoid": '<ellipsoid name="s_ellipsoid" ax="3" by="4" cz="5" zcut1="-4" zcut2="4" lunit="mm"/>',
    "eltube": '<eltube name="s_eltube" dx="3" dy="4" dz="5" lunit="mm"/>',
    "elcone": '<elcone name="s_elcone" dx="0.5" dy="0.25" zmax="10" zcut="5" lunit="mm"/>',
    "paraboloid": '<paraboloid name="s_paraboloid" rlo="1" rhi="3" dz="4" lunit="mm"/>',
    "hype": '<hype name="s_hype" rmin="1" rmax="3" inst="0.1" outst="0.2" z="10" aunit="rad" lunit="mm"/>',
    "torus": '<torus name="s_torus" rmin="0" rmax="1" rtor="5" startphi="0" deltaphi="6.283185307179586" '
             'aunit="rad" lunit="mm"/>',
    "trd": '<trd name="s_trd" x1="2" x2="4" y1="3" y2="5" z="6" lunit="mm"/>',
    "para": '<para name="s_para" x="2" y="3" z="4" alpha="0.1" theta="0.2" phi="0.3" aunit="rad" lunit="mm"/>',
    "trap": '<trap name="s_trap" z="10" theta="0" phi="0" y1="4" x1="3" x2="3" alpha1="0" y2="4" x3="3" x4="3" '
            'alpha2="0" aunit="rad" lunit="mm"/>',
    "arb8": '<arb8 name="s_arb8" dz="5" v1x="-3" v1y="-3" v2x="-3" v2y="3" v3x="3" v3y="3" v4x="3" v4y="-3" '
            'v5x="-2" v5y="-2" v6x="-2" v6y="2" v7x="2" v7y="2" v8x="2" v8y="-2" lunit="mm"/>',
    "xtru": '<xtru name="s_xtru" lunit="mm"><twoDimVertex x="-3" y="-3"/><twoDimVertex x="-3" y="3"/>'
            '<twoDimVertex x="3" y="3"/><twoDimVertex x="3" y="-3"/>'
            '<section zOrder="0" zPosition="-5" xOffset="0" yOffset="0" scalingFactor="1"/>'
            '<section zOrder="1" zPosition="5" xOffset="1" yOffset="0.5" scalingFactor="0.5"/></xtru>',
    "tet": '<tet name="s_tet" vertex1="t1" vertex2="t2" vertex3="t3" vertex4="t4" lunit="cm"/>',
    # a closed 10 mm cube: quadrangles, one face as two RELATIVE triangles, and one face on a -0.0 copy of a corner
    "tessellated": '<tessellated name="s_tessellated">'
                   '<quadrangular vertex1="c0" vertex2="c3" vertex3="c2" vertex4="c1"/>'
                   '<quadrangular vertex1="c4" vertex2="c5" vertex3="c6" vertex4="c7" type="ABSOLUTE"/>'
                   '<quadrangular vertex1="c0" vertex2="c1" vertex3="c5" vertex4="c4"/>'
                   '<quadrangular vertex1="c3" vertex2="c7" vertex3="c6" vertex4="c2"/>'
                   '<quadrangular vertex1="c0m" vertex2="c4" vertex3="c7" vertex4="c3"/>'
                   '<triangular vertex1="c1" vertex2="ra" vertex3="rb" type="RELATIVE"/>'
                   '<triangular vertex1="c1" vertex2="rb" vertex3="rc" type="RELATIVE"/></tessellated>',
    "scaledSolid": '<scaledSolid name="s_scaledSolid"><solidref ref="s_box"/><scale name="sc" x="1" y="2" z="0.5"/>'
                   "</scaledSolid>",
    "subtraction": '<subtraction name="s_subtraction"><first ref="s_box"/><second ref="s_orb"/>'
                   '<position name="sp" unit="mm" x="1" y="2" z="3"/><rotation name="sr" unit="rad" x="0.3" y="-0.7" '
                   'z="1.1"/><firstposition name="fp" unit="mm" x="-1" y="0" z="0"/>'
                   '<firstrotation name="fr" unit="deg" x="10" y="20" z="30"/></subtraction>',
    "union": '<union name="s_union"><first ref="s_box"/><second ref="s_tube"/></union>',
    "intersection": '<intersection name="s_intersection"><first ref="s_box"/><second ref="s_orb"/>'
                    '<positionref ref="pos1"/></intersection>',
    "multiUnion": '<multiUnion name="s_multiUnion"><multiUnionNode name="n1"><solid ref="s_box"/></multiUnionNode>'
                  '<multiUnionNode name="n2"><solid ref="s_orb"/><position name="np" unit="mm" x="5" y="0" z="0"/>'
                  '<rotation name="nr" unit="rad" x="0.3" y="-0.7" z="1.1"/></multiUnionNode></multiUnion>',
}  # fmt: skip
SOLID_DEFINES = """  <position name="t1" unit="mm" x="0" y="0" z="0"/><position name="t2" unit="mm" x="1" y="0" z="0"/>
  <position name="t3" unit="mm" x="0" y="1" z="0"/><position name="t4" unit="mm" x="0" y="0" z="1"/>
  <position name="c0" unit="mm" x="0" y="0" z="0"/><position name="c1" unit="mm" x="10" y="0" z="0"/>
  <position name="c2" unit="mm" x="10" y="10" z="0"/><position name="c3" unit="mm" x="0" y="10" z="0"/>
  <position name="c4" unit="mm" x="0" y="0" z="10"/><position name="c5" unit="mm" x="10" y="0" z="10"/>
  <position name="c6" unit="mm" x="10" y="10" z="10"/><position name="c7" unit="mm" x="0" y="10" z="10"/>
  <position name="c0m" unit="mm" x="-0.0" y="0" z="0"/>
  <position name="ra" unit="mm" x="0" y="10" z="0"/><position name="rb" unit="mm" x="0" y="10" z="10"/>
  <position name="rc" unit="mm" x="0" y="0" z="10"/>
  <position name="pos1" unit="cm" x="0.1" y="0" z="0"/>"""


def _all_solids():
    names = list(SOLIDS)
    volumes = "\n".join(vol(f"lv_{t}", f"s_{t}") for t in names)
    physvols = "\n".join(pv(f"lv_{t}", f"pv_{t}", pos=(100 * (i % 6) - 250, 100 * (i // 6) - 250, 0))
                         for i, t in enumerate(names))  # fmt: skip
    return toy("\n".join("  " + SOLIDS[t] for t in names), volumes, physvols, SOLID_DEFINES)


# gdml_export.md section 7.2's export toys
TOYS = {
    "a_angles": toy('  <box name="b" x="100" y="200" z="300" lunit="mm"/>', vol("lvB", "b"),
                    pv("lvB", "pB", pos=(10, -20, 30.5), rot=(0.3, -0.7, 1.1))),
    "b_booleans": toy("  " + SOLIDS["box"] + "\n  " + SOLIDS["orb"] + "\n  " + SOLIDS["tube"] + "\n  "
                      + SOLIDS["subtraction"] + "\n  " + SOLIDS["multiUnion"],
                      vol("lvS", "s_subtraction") + "\n" + vol("lvM", "s_multiUnion"),
                      pv("lvS", "pS", pos=(-500, 0, 0)) + "\n" + pv("lvM", "pM", pos=(500, 0, 0)), SOLID_DEFINES),
    "c_reflection": toy('  <trd name="t" x1="100" x2="200" y1="100" y2="150" z="300" lunit="mm"/>\n' + SURFACES,
                        vol("lvT", "t", "Black") + "\n" + vol("lvT2", "t", "Black"),
                        pv("lvT", "pT", pos=(-500, 0, 0), scale=(1, 1, -1))
                        + "\n" + pv("lvT2", "pT2", pos=(500, 0, 0), scale=(1, -1, 1), rot=(0, 0, 0.5)),
                        structure='  <skinsurface name="sk" surfaceproperty="Glisur"><volumeref ref="lvT2"/>'
                                  "</skinsurface>"),
    "d_tessellated": toy("  " + SOLIDS["tessellated"], vol("lvMesh", "s_tessellated", "Black"),
                         pv("lvMesh", "pMesh", pos=(0, 0, 100)), SOLID_DEFINES),
    "e_tet": toy("  " + SOLIDS["tet"], vol("lvTet", "s_tet", "Black"), pv("lvTet", "pTet"), SOLID_DEFINES),
    "f_solids": toy("\n".join("  " + SOLIDS[t] for t in ("cutTube", "xtru", "polycone", "polyhedra")),
                    "\n".join(vol(f"lv_{t}", f"s_{t}") for t in ("cutTube", "xtru", "polycone", "polyhedra")),
                    "\n".join(pv(f"lv_{t}", f"pv_{t}", pos=(200 * i - 300, 0, 0))
                              for i, t in enumerate(("cutTube", "xtru", "polycone", "polyhedra")))),
    "g_skins": toy('  <box name="big" x="1000" y="1000" z="1000" lunit="mm"/>\n  <box name="small" x="100" y="100" '
                   'z="100" lunit="mm"/>\n' + SURFACES,
                   vol("lvSmall", "small", "Water") + "\n" + vol("lvBig", "big", "Water", pv("lvSmall", "pSmall")),
                   pv("lvBig", "pBig"),
                   structure='  <skinsurface name="skBig" surfaceproperty="Unified"><volumeref ref="lvBig"/>'
                             '</skinsurface>\n  <skinsurface name="skSmall" surfaceproperty="Glisur">'
                             '<volumeref ref="lvSmall"/></skinsurface>'),
    "h_borders": toy('  <box name="half" x="500" y="1000" z="1000" lunit="mm"/>\n' + SURFACES,
                     vol("lvA", "half") + "\n" + vol("lvB", "half", "Mix"),
                     pv("lvA", "pA", pos=(-250, 0, 0)) + "\n" + pv("lvB", "pB", pos=(250, 0, 0)),
                     structure='  <bordersurface name="AB" surfaceproperty="Glisur"><physvolref ref="pA"/>'
                               '<physvolref ref="pB"/></bordersurface>\n  <bordersurface name="BA" '
                               'surfaceproperty="Sensor"><physvolref ref="pB"/><physvolref ref="pA"/></bordersurface>'),
    "i_surfaces": toy('  <box name="b" x="100" y="100" z="100" lunit="mm"/>\n' + SURFACES,
                      vol("lvG", "b") + "\n" + vol("lvU", "b"),
                      pv("lvG", "pG", pos=(-200, 0, 0)) + "\n" + pv("lvU", "pU", pos=(200, 0, 0)),
                      structure='  <skinsurface name="skG" surfaceproperty="Glisur"><volumeref ref="lvG"/>'
                                '</skinsurface>\n  <skinsurface name="skU" surfaceproperty="Unified">'
                                '<volumeref ref="lvU"/></skinsurface>'),
    "j_names": toy('  <box name="b0x1234" x="100" y="100" z="100" lunit="mm"/>',
                   vol("lv0x55aa", "b0x1234"),
                   pv("lv0x55aa", pos=(-200, 0, 0)) + "\n" + pv("lv0x55aa", pos=(200, 0, 0), copy=7)),
    "k_scintillation": toy('  <box name="b" x="100" y="100" z="100" lunit="mm"/>', vol("lvS", "b"), pv("lvS", "pS"),
                           water_props='<property name="FASTCOMPONENT" ref="REF"/><property name="SLOWCOMPONENT" '
                                       'ref="REF"/><property name="FASTTIMECONSTANT" ref="C1"/><property '
                                       'name="YIELDRATIO" ref="C1"/><property name="SCINTILLATIONYIELD" ref="LEN1"/>'
                                       '<property name="RAYLEIGH" ref="LEN1"/>'),
    "all_solids": _all_solids(),
    "aux_userinfo": toy('  <box name="b" x="100" y="100" z="100" lunit="mm"/>',
                        vol("lvS", "b", inner='<auxiliary auxtype="SensDet" auxvalue="PhotonDetector"/>'
                            '<auxiliary auxtype="Region" auxvalue="R"><auxiliary auxtype="cut" auxvalue="1" '
                            'auxunit="mm"><auxiliary auxtype="deep" auxvalue="x"/></auxiliary></auxiliary>'),
                        pv("lvS", "pS"),
                        tail=' <userinfo><auxiliary auxtype="Property" auxvalue="p1"><auxiliary auxtype="k" '
                             'auxvalue="v" auxunit="cm"/></auxiliary><auxiliary auxtype="Tag" auxvalue="t"/>'
                             "</userinfo>\n"),
}  # fmt: skip
