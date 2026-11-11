"""A toy GDML exercising every optics rule, read by the importer.

Every material names a composition (an element Ar), the border's second volume is a placed physvol ("other"), and the
file has a setup naming the world, as the importer requires.
"""

import numpy as np

from lumen.build.tree import PV, Tree
from lumen.gdml import importer

TOY = """<?xml version="1.0" encoding="UTF-8" ?>
<gdml>
<define>
  <constant name="TWO" value="2"/>
  <quantity name="L0" value="TWO*5" unit="cm"/>
  <matrix name="RIX0x55aa" coldim="2" values="2.0*eV 1.30  3.0*eV 1.31  6.0*eV 1.35  9.0*eV 1.45"/>
  <matrix name="ABS" coldim="2" values="2*eV 2000*cm 10*eV L0"/>
  <matrix name="RAY" coldim="2" values="2*eV 100*m 10*eV 1*m"/>
  <matrix name="WLA" coldim="2" values="1*eV 1e5*mm 5*eV 1e5*mm 7.5*eV 1e-3*mm 50*eV 1e-11*mm"/>
  <matrix name="WLE" coldim="2" values="2.5*eV 0 2.8*eV 1 3.1*eV 0.6 3.4*eV 0"/>
  <matrix name="WLT" coldim="1" values="0.0*ns"/>
  <matrix name="FTC" coldim="1" values="6." unit="ns"/>
  <matrix name="YR" coldim="1" values="0.27"/>
  <matrix name="SY" coldim="1" values="720/MeV"/>
  <matrix name="R" coldim="2" values="1.77*eV 0.66 2.953*eV 0.59 7*eV 0.27 15*eV 0.25"/>
  <matrix name="HALF" coldim="2" values="1*eV 0.5 20*eV 0.5"/>
</define>
<materials>
  <element name="Ar" formula="Ar" Z="18"> <atom value="39.95"/> </element>
  <material name="Liq"><D value="1.4"/><fraction n="1" ref="Ar"/>
    <property name="RINDEX" ref="RIX0x55aa"/><property name="ABSLENGTH" ref="ABS"/><property name="RAYLEIGH" ref="RAY"/>
    <property name="FASTCOMPONENT" ref="WLE"/><property name="SLOWCOMPONENT" ref="WLE"/>
    <property name="FASTTIMECONSTANT" ref="FTC"/><property name="SLOWTIMECONSTANT" ref="FTC"/>
    <property name="YIELDRATIO" ref="YR"/><property name="SCINTILLATIONYIELD" ref="SY"/>
    <property name="RADLENGTH" ref="YR"/></material>
  <material name="Shifter"><D value="1.0"/><fraction n="1" ref="Ar"/>
    <property name="RINDEX" ref="RIX0x55aa"/><property name="WLSABSLENGTH" ref="WLA"/>
    <property name="WLSCOMPONENT" ref="WLE"/><property name="WLSTIMECONSTANT" ref="WLT"/></material>
  <material name="Steel"><D value="8"/><fraction n="1" ref="Ar"/></material>
</materials>
<solids>
  <box name="b" x="10" y="20" z="30" lunit="mm"/>
  <box name="big" x="100" y="200" z="300" lunit="mm"/>
  <opticalsurface name="glis" model="glisur" finish="ground" type="dielectric_metal" value="0.5">
    <property name="REFLECTIVITY" ref="R"/></opticalsurface>
  <opticalsurface name="glispol" model="glisur" finish="polished" type="dielectric_metal" value="0.5"/>
  <opticalsurface name="glis1" model="glisur" finish="ground" type="dielectric_metal" value="1"/>
  <opticalsurface name="uni" model="unified" finish="ground" type="dielectric_metal" value="0.2">
    <property name="REFLECTIVITY" ref="R"/><property name="SPECULARLOBECONSTANT" ref="HALF"/></opticalsurface>
  <opticalsurface name="paint" model="unified" finish="groundfrontpainted" type="dielectric_dielectric" value="0.8">
    <property name="REFLECTIVITY" ref="R"/></opticalsurface>
  <opticalsurface name="paintT" model="glisur" finish="polishedfrontpainted" type="dielectric_dielectric" value="1">
    <property name="REFLECTIVITY" ref="HALF"/><property name="TRANSMITTANCE" ref="HALF"/></opticalsurface>
  <opticalsurface name="die" model="glisur" finish="polished" type="dielectric_dielectric" value="1"/>
  <opticalsurface name="lut" model="LUT" finish="polishedteflonair" type="dielectric_LUT" value="1"/>
  <opticalsurface name="dich" model="dichroic" finish="polished" type="dielectric_dichroic" value="1"/>
  <opticalsurface name="back" model="unified" finish="groundbackpainted" type="dielectric_dielectric" value="0.1"/>
  <opticalsurface name="gdie" model="unified" finish="ground" type="dielectric_dielectric" value="0.1"/>
  <opticalsurface name="pdieR" model="glisur" finish="polished" type="dielectric_dielectric" value="1">
    <property name="REFLECTIVITY" ref="HALF"/></opticalsurface>
  <opticalsurface name="eff" model="glisur" finish="polished" type="dielectric_metal" value="1">
    <property name="EFFICIENCY" ref="HALF"/></opticalsurface>
  <opticalsurface name="mT" model="glisur" finish="polished" type="dielectric_metal" value="1">
    <property name="TRANSMITTANCE" ref="HALF"/></opticalsurface>
  <opticalsurface name="cplx" model="glisur" finish="polished" type="dielectric_metal" value="1">
    <property name="REALRINDEX" ref="HALF"/><property name="IMAGINARYRINDEX" ref="HALF"/></opticalsurface>
  <opticalsurface name="usig" model="unified" finish="ground" type="dielectric_metal" value="0.9"/>
  <opticalsurface name="neg" model="glisur" finish="ground" type="dielectric_metal" value="-0.1"/>
  <opticalsurface name="enum" model="1" finish="4" type="1" value="0.1">
    <property name="REFLECTIVITY" ref="R"/></opticalsurface>
</solids>
<structure>
  <volume name="Blk"><materialref ref="Steel"/><solidref ref="b"/></volume>
  <skinsurface name="sk" surfaceproperty="glis"><volumeref ref="Blk"/></skinsurface>
  <volume name="X"><materialref ref="Liq"/><solidref ref="b"/></volume>
  <volume name="W"><materialref ref="Liq"/><solidref ref="big"/>
    <physvol name="blk1" copynumber="7"><volumeref ref="Blk"/><position name="p" unit="cm" x="1" y="0" z="0"/></physvol>
    <physvol name="other"><volumeref ref="X"/><position name="q" unit="cm" x="-3" y="0" z="0"/></physvol>
  </volume>
  <bordersurface name="bs" surfaceproperty="paint"><physvolref ref="blk1"/><physvolref ref="other"/></bordersurface>
</structure>
<setup name="Default" version="1.0"> <world ref="W"/> </setup>
</gdml>
"""


def read_toy(tmp_path_factory, text=TOY, name="toy.gdml"):
    """Return the detector file of a toy GDML written to a temporary file."""
    p = tmp_path_factory.mktemp("gdml") / name
    p.write_text(text)
    return importer.read_gdml(str(p))


def tree(*specs):
    """Return a Tree of (name, lv, material, mother, copy) specs."""
    return Tree(tuple(PV(*s) for s in specs))


def flat(value, energies=(1.0, 20.0)):
    """Return (energies, values) of a flat table."""
    return np.asarray(energies, np.float64), np.full(len(energies), float(value))
