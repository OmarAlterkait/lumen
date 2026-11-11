"""The film toy: CPA-like foil holders in a liquid tank, each a WLS film on one face of a painted foil.

``film_gdml(holders, ...)`` returns the GDML: a 2 m liquid tank (light leaving it ends) in a vacuum world; each
holder is a liquid box exactly foil (1 mm, material Steel, skin ``paint``: groundfrontpainted) plus film (20 nm,
material Shifter: index-matched to the liquid, WLSABSLENGTH 1e-11 mm above 7.5 eV, emission at 2.5-3.4 eV) thick,
400 mm square, the film on the foil's local +x face; ``holders`` gives each holder's (position mm, GDML rotation
deg). Two 1 cm counting slabs at y = +-0.7 m (Slab0LV, Slab1LV) see both sides of every holder. Variants for the
fail-closed rules: ``films`` ("+x", "both": a film on each face), ``film_side`` (the film's square side, mm:
smaller is partial coverage) and ``bare`` (one more foil placed alone in the tank).
"""

from __future__ import annotations

from lumen.format.sensorsets import SensorGroup
from lumen.recipe import Recipe

FOIL, FILM, SIDE = 1.0, 2e-5, 400.0  # mm
BAND = (110.0, 600.0)  # nm: inside the tables (1-12 eV)
SLABS = tuple(SensorGroup(f"slab{k}", "any_entry", "lumen:counting", lv=f"Slab{k}LV", settings={"efficiency": 1.0})
              for k in (0, 1))  # fmt: skip
RECIPE = Recipe("film_toy", "TankLV", BAND, SLABS)

HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<gdml>
<define>
  <matrix name="RIX" coldim="2" values="1.0*eV 1.30 12.0*eV 1.30"/>
  <matrix name="ABS" coldim="2" values="1.0*eV 1000*m 12.0*eV 1000*m"/>
  <matrix name="WLA" coldim="2" values="1*eV 1e5*mm 5*eV 1e5*mm 7.5*eV 1e-11*mm 12*eV 1e-11*mm"/>
  <matrix name="WLE" coldim="2" values="2.5*eV 0 2.8*eV 1 3.1*eV 0.6 3.4*eV 0"/>
  <matrix name="WLT" coldim="1" values="0.0*ns"/>
  <matrix name="R" coldim="2" values="1.0*eV 0.5 12.0*eV 0.5"/>
</define>
<materials>
  <element name="Hyd" formula="H" Z="1"><atom value="1.008"/></element>
  <material name="Liq"><D value="1.4"/><fraction n="1" ref="Hyd"/>
    <property name="RINDEX" ref="RIX"/><property name="ABSLENGTH" ref="ABS"/></material>
  <material name="Shifter"><D value="1.0"/><fraction n="1" ref="Hyd"/>
    <property name="RINDEX" ref="RIX"/><property name="WLSABSLENGTH" ref="WLA"/>
    <property name="WLSCOMPONENT" ref="WLE"/><property name="WLSTIMECONSTANT" ref="WLT"/></material>
  <material name="Steel"><D value="8"/><fraction n="1" ref="Hyd"/></material>
  <material name="Vac"><D value="1e-9"/><fraction n="1" ref="Hyd"/></material>
</materials>
"""


def film_gdml(holders=(((0.0, 0.0, 0.0), None),), films="+x", film_side=SIDE, bare=False) -> str:
    """Return the film toy's GDML (see the module docstring)."""
    n = 2 if films == "both" else 1
    thick = FOIL + n * FILM
    xf = -thick / 2 + (FILM if n == 2 else 0.0) + FOIL / 2  # the foil's centre; the film(s) beyond it
    film_pvs = "".join(f'<physvol name="film{k}"><volumeref ref="FilmLV"/><position name="pf{k}" unit="mm" '
                       f'x="{xf + s * (FOIL + FILM) / 2!r}" y="0" z="0"/></physvol>'
                       for k, s in enumerate((1.0, -1.0)[:n]))  # fmt: skip
    pvs = []
    for k, (pos, rot) in enumerate(holders):
        r = "" if rot is None else f'<rotation name="r{k}" unit="deg" x="{rot[0]}" y="{rot[1]}" z="{rot[2]}"/>'
        pvs.append(f'<physvol name="holder{k}" copynumber="{k}"><volumeref ref="HolderLV"/><position name="p{k}" '
                   f'unit="mm" x="{pos[0]}" y="{pos[1]}" z="{pos[2]}"/>{r}</physvol>')  # fmt: skip
    if bare:
        pvs.append('<physvol name="bare"><volumeref ref="FoilLV"/><position name="pb" unit="mm" x="0" y="0" '
                   'z="-800"/></physvol>')  # fmt: skip
    pvs += [f'<physvol name="slab{k}"><volumeref ref="Slab{k}LV"/><position name="ps{k}" unit="mm" x="0" y="{y}" '
            'z="0"/></physvol>' for k, y in enumerate((700, -700))]  # fmt: skip
    body = f"""<solids>
  <box name="WorldS" x="3000" y="3000" z="3000" lunit="mm"/>
  <box name="TankS" x="2000" y="2000" z="2000" lunit="mm"/>
  <box name="FoilS" x="{FOIL}" y="{SIDE}" z="{SIDE}" lunit="mm"/>
  <box name="FilmS" x="{FILM}" y="{film_side}" z="{film_side}" lunit="mm"/>
  <box name="HolderS" x="{thick!r}" y="{SIDE}" z="{SIDE}" lunit="mm"/>
  <box name="SlabS" x="1800" y="10" z="1800" lunit="mm"/>
  <opticalsurface name="paint" model="unified" finish="groundfrontpainted" type="dielectric_dielectric" value="0.8">
    <property name="REFLECTIVITY" ref="R"/></opticalsurface>
</solids>
<structure>
  <volume name="FoilLV"><materialref ref="Steel"/><solidref ref="FoilS"/></volume>
  <skinsurface name="foilskin" surfaceproperty="paint"><volumeref ref="FoilLV"/></skinsurface>
  <volume name="FilmLV"><materialref ref="Shifter"/><solidref ref="FilmS"/></volume>
  <volume name="HolderLV"><materialref ref="Liq"/><solidref ref="HolderS"/>
    <physvol name="foil"><volumeref ref="FoilLV"/><position name="pfoil" unit="mm" x="{xf!r}" y="0" z="0"/></physvol>
    {film_pvs}</volume>
  <volume name="Slab0LV"><materialref ref="Liq"/><solidref ref="SlabS"/></volume>
  <volume name="Slab1LV"><materialref ref="Liq"/><solidref ref="SlabS"/></volume>
  <volume name="TankLV"><materialref ref="Liq"/><solidref ref="TankS"/>
    {"".join(pvs)}</volume>
  <volume name="WorldLV"><materialref ref="Vac"/><solidref ref="WorldS"/>
    <physvol name="tank"><volumeref ref="TankLV"/></physvol></volume>
</structure>
<setup name="Default" version="1.0"><world ref="WorldLV"/></setup>
</gdml>
"""
    return HEAD + body
