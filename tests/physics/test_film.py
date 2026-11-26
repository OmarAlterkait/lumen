"""A one-sided WLS slab through the step (the physics tier): the film toy (a 25.4 um slab of a WLS material on a
painted foil, as SBND's split TPB) with its holder turned 90 degrees about z.

VUV light (128 nm, absorbed within 1e-11 mm by the slab: bulk WLS) from 5 cm in front of the slab face is re-emitted
into the visible band; from 5 cm behind the foil's back face no light can reach the film (the tank's walls absorb), so
no visible light is detected at all, and the bare back face reflects VUV light (paint, R = 0.5) the film face absorbs.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import toys
from toys import simulate

import lumen
from lumen.compare import photons
from lumen.format.sensorsets import SensorGroup
from lumen.recipe import Recipe

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))  # the unit tier's film toy
from film_toy import RECIPE, film_gdml

N = 1 << 16
CENTRE_MM = (0, 0, 500)  # off the origin
VUV, SPLIT = 128.0, 300.0  # nm


@pytest.fixture(scope="module")
def film(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("film")
    (tmp / "film.gdml").write_text(film_gdml(((CENTRE_MM, (0, 0, 90)),)))
    det = lumen.load(lumen.convert(tmp / "film.gdml", RECIPE, tmp / "film.lumen"))
    return toys.setup(det, (-np.inf, SPLIT, np.inf))


def charge(s, side):
    """Return (VUV, visible) detected per photon from 5 cm off the film's face (side +1) or the back face (-1)."""
    axis = np.array([0.0, -1.0, 0.0])  # the slab's face: the local +x face, turned 90 degrees about z (GDML, passive)
    x = jnp.tile(jnp.asarray(np.asarray(CENTRE_MM) * 1e-3 + 0.05 * side * np.asarray(axis), jnp.float32), (N, 1))
    rows = photons.isotropic(x, jnp.full(N, VUV, jnp.float32), jax.random.PRNGKey(3))
    q = np.asarray(simulate(s, photons.packed(s.det), rows)["charge:0"]).reshape(-1, 2).sum(0) / N
    return float(q[0]), float(q[1])


def test_only_the_film_face_re_emits_and_the_back_face_reflects(film):
    vuv_front, vis_front = charge(film, 1.0)
    vuv_back, vis_back = charge(film, -1.0)
    assert vis_back == 0.0 and vis_front > 0.05
    assert vuv_back > vuv_front > 0.0


# ------------------------------------------------------------ a film flush in its holder, and the same as a sibling
FILM_M, FOIL_M, FACE_M = 1e-4, 0.02, -0.2  # m: the film, the foil, and the film's outer face (off the origin)
FLUSH_HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<gdml>
<define>
  <matrix name="R1" coldim="2" values="1.0*eV 1.0 12.0*eV 1.0"/>
  <matrix name="R15" coldim="2" values="1.0*eV 1.5 12.0*eV 1.5"/>
  <matrix name="WLA" coldim="2" values="1*eV 1e5*mm 6*eV 1e5*mm 6.001*eV 5e-4*mm 12*eV 5e-4*mm"/>
  <matrix name="WLE" coldim="2" values="2.8*eV 1.0 3.0*eV 1.0"/>
  <matrix name="WLT" coldim="1" values="0.0*ns"/>
  <matrix name="RR" coldim="2" values="1.0*eV 0.5 12.0*eV 0.5"/>
</define>
<materials>
  <element name="Hyd" formula="H" Z="1"><atom value="1.008"/></element>
  <material name="Gas" state="gas"><D value="0.001"/><fraction n="1" ref="Hyd"/>
    <property name="RINDEX" ref="R1"/></material>
  <material name="Shifter"><D value="1.0"/><fraction n="1" ref="Hyd"/>
    <property name="RINDEX" ref="R15"/><property name="WLSABSLENGTH" ref="WLA"/>
    <property name="WLSCOMPONENT" ref="WLE"/><property name="WLSTIMECONSTANT" ref="WLT"/></material>
  <material name="Steel"><D value="8"/><fraction n="1" ref="Hyd"/></material>
  <material name="Vac"><D value="1e-9"/><fraction n="1" ref="Hyd"/></material>
</materials>
"""


def flush_gdml(flush: bool) -> str:
    """Return the GDML of a WLS film on a painted foil in a gas tank, the film's outer face at ``FACE_M``.

    ``flush``: the film is the foil box's daughter, its outer face coincident with the box's own (a WLS film placed
    flush in its holder, as SBND's TPB); else the same film is the foil box's sibling, on top of it. The two are one
    geometry for light: the film's outer face meets the gas (index 1.5 to 1), its inner face the painted foil
    (groundfrontpainted, R = 0.5), and the foil is opaque. VUV (128 nm) is absorbed within 1e-11 mm of the outer face
    (0.5 um, a couple of bands there) and re-emitted at 2.8-3.0 eV.
    """
    film_z = FACE_M - FILM_M / 2 if flush else FACE_M + FILM_M / 2  # the foil box's own top is at FACE_M
    inside = f'<physvol name="film"><volumeref ref="FilmLV"/><position name="pf" unit="m" x="0" y="0" z="{FOIL_M / 2 - FILM_M / 2!r}"/></physvol>'  # noqa: E501
    outside = f'<physvol name="film"><volumeref ref="FilmLV"/><position name="pf" unit="m" x="0" y="0" z="{film_z!r}"/></physvol>'  # noqa: E501
    foil = f'<physvol name="foil"><volumeref ref="FoilLV"/><position name="pb" unit="m" x="0" y="0" z="{FACE_M - FOIL_M / 2!r}"/></physvol>'  # noqa: E501
    slab = f'<physvol name="slab"><volumeref ref="SlabLV"/><position name="ps" unit="m" x="0" y="0" z="{FACE_M + 1.0!r}"/></physvol>'  # noqa: E501
    return (
        FLUSH_HEAD
        + f"""<solids>
  <box name="WorldS" x="8" y="8" z="8" lunit="m"/>
  <box name="TankS" x="4" y="4" z="4" lunit="m"/>
  <box name="FoilS" x="2" y="2" z="{FOIL_M!r}" lunit="m"/>
  <box name="FilmS" x="1.9" y="1.9" z="{FILM_M!r}" lunit="m"/>
  <box name="SlabS" x="3.9" y="3.9" z="0.001" lunit="m"/>
  <opticalsurface name="paint" model="unified" finish="groundfrontpainted" type="dielectric_dielectric" value="1">
    <property name="REFLECTIVITY" ref="RR"/></opticalsurface>
</solids>
<structure>
  <volume name="FilmLV"><materialref ref="Shifter"/><solidref ref="FilmS"/></volume>
  <volume name="FoilLV"><materialref ref="Steel"/><solidref ref="FoilS"/>{inside if flush else ""}</volume>
  <skinsurface name="foilskin" surfaceproperty="paint"><volumeref ref="FoilLV"/></skinsurface>
  <volume name="SlabLV"><materialref ref="Gas"/><solidref ref="SlabS"/></volume>
  <volume name="TankLV"><materialref ref="Gas"/><solidref ref="TankS"/>
    {foil}
    {"" if flush else outside}
    {slab}
  </volume>
  <volume name="WorldLV"><materialref ref="Vac"/><solidref ref="WorldS"/>
    <physvol name="tank"><volumeref ref="TankLV"/></physvol></volume>
</structure>
<setup name="Default" version="1.0"><world ref="WorldLV"/></setup>
</gdml>
"""
    )


FLUSH_RECIPE = Recipe("flush_film", "TankLV",
                      groups=(SensorGroup("slab", rule="any_entry", lv="SlabLV",
                                                       settings={"entry_efficiency": 1.0}),))  # fmt: skip


def test_a_film_flush_in_its_holder_shifts_the_same_light_as_one_beside_it(tmp_path):
    """One geometry placed two ways detects the same light: a WLS film flush in its holder (its outer face coincident
    with the holder's) and the same film as the holder's sibling.

    A re-emission within the band of coincident faces is on them: it must leave the film into the gas (index 1.5 to
    1: most of it is held by total internal reflection), not be read on the holder's own face there (its paint) or
    without a face at all (no refraction, so nothing held). Both are wrong by a factor on the flush film alone.
    """
    n = 1 << 16
    out = {}
    for flush in (True, False):
        src = tmp_path / f"flush{int(flush)}.gdml"
        src.write_text(flush_gdml(flush))
        det = lumen.load(lumen.convert(src, FLUSH_RECIPE, tmp_path / f"flush{int(flush)}.lumen"))
        light = lumen.simulate(det, lumen.sources.Point((0.0, 0.0, FACE_M + 0.5), n_photons=1.0, lam=VUV,
                                                        direction=(0.0, 0.0, -1.0)), samples=n, seed=5)  # fmt: skip
        out[flush] = (float(light.arrivals.sum()), float(light.arrivals_error.sum()))
    (qf, ef), (qs, es) = out[True], out[False]
    assert qs > 0.05 and qf > 0.05, out
    assert abs(qf - qs) < 4.0 * (ef + es) + 0.02 * qs, out  # the same light, within the runs' own errors
