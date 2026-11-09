"""lumen.build.legacy_library: a detector file's recorded optics decisions as the region compiler's library and
bindings -- every material, surface and sensor encoding, the WLS-film split, the outside surface, a recipe skin
replacing the GDML's, and the parameter sources."""

import dataclasses

import numpy as np
import pytest
from optics_toy import read_toy, tree

from lumen.build import _legacy_regions as regions
from lumen.build import legacy_library as glue
from lumen.format import schema
from lumen.format.sensorsets import SensorGroup, SensorSet
from lumen.surfaces import _legacy as legacy

LAM = (300.0, 600.0)


class Inst:
    def __init__(self, pv, R):
        self.pv, self.R = pv, np.asarray(R, np.float64)


def model(name, **settings):
    return schema.OpticsModel(name, 1, settings, "recipe", "test")


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    return read_toy(tmp_path_factory)


def decided(df):
    mats = dict(df.materials)
    mats["Steel"] = dataclasses.replace(mats["Steel"], optics=model("lumen:opaque", medium="opaque"))
    mats["Shifter"] = dataclasses.replace(mats["Shifter"], optics=model("lumen:same_as", material="Liq"))
    settings = dict(film_material="Shifter", front_material="Liq", backing="paint", thickness_m=1e-6,
                    local_axis=[1.0, 0.0, 0.0])  # fmt: skip
    film = schema.Surface("film", "", "", "", 0.0, {}, model("lumen:wls_film", **settings))
    skins = df.skins + (schema.Skin("film", "film", "Foil", "recipe"),)
    return dataclasses.replace(df, materials=mats, surfaces={**df.surfaces, "film": film}, skins=skins,
                               outside_surface="glis", wavelength_range=LAM)  # fmt: skip


TREE = (("<outside>", "<outside>", "<outside>"), ("w/W", "W", "Liq", "<outside>"),
        ("w/blk1/Blk", "Blk", "Steel", "w/W"), ("w/other/X", "X", "Liq", "w/W"),
        ("w/f1/Foil", "Foil", "Steel", "w/W"), ("w/f2/Foil", "Foil", "Steel", "w/W"),
        ("w/sh/S", "S", "Shifter", "w/W"), ("w/sen/Sen", "Sen", "Liq", "w/W", 0))  # fmt: skip
INSTANCES = (Inst("w/f1/Foil", np.eye(3)), Inst("w/f2/Foil", np.diag([-1.0, -1.0, 1.0])))
COUNTING = SensorSet("default",
                     (SensorGroup("sen", "any_entry", "lumen:counting", lv="Sen", settings={"eff": 0.5}),))  # fmt: skip
PLUS, MINUS = "Foil|film(np.float64(1.0), np.float64(0.0), np.float64(0.0))", \
    "Foil|film(np.float64(-1.0), np.float64(0.0), np.float64(0.0))"  # fmt: skip


def test_every_encoding_on_the_toy(toy):
    t, lib, b, src = glue.library_and_bindings(decided(toy), tree(*TREE), INSTANCES, COUNTING)
    assert {t.by_name["w/f1/Foil"].lv, t.by_name["w/f2/Foil"].lv} == {PLUS, MINUS}
    assert np.array_equal(t.film_axes[MINUS], [-1.0, 0.0, 0.0])
    assert b.materials == {"<outside>": "outside", "Liq": "Liq", "Steel": "opaque", "Shifter": "Liq"}
    assert set(lib.media) == {"outside", "Liq", "opaque"} and lib.media["opaque"].medium.opaque
    assert b.skins == {"Blk": "glis", "<outside>": "glis", PLUS: f"film:{PLUS}", MINUS: f"film:{MINUS}"}
    film = lib.surfaces[f"film:{PLUS}"]
    # the film's side, thickness and yield are values the model fixes (OUT): folded into one type per side
    assert film.kind.name == "coated:Shifter:painted{axis=(1,0,0),b:f_spec=0,d=1e-06,eta=1}"
    assert lib.surfaces[f"film:{MINUS}"].kind.name == "coated:Shifter:painted{axis=(-1,0,0),b:f_spec=0,d=1e-06,eta=1}"
    assert "axis" not in film.params and "d" not in film.params and "L" in film.params
    assert (
        b.borders == {("w/blk1/Blk", "w/other/X"): "paint"} and lib.surfaces["paint"].kind.name == "painted{f_spec=0}"
    )
    assert lib.sensors["sen"].params == {"eff": 0.5} and b.sensors == (regions.SensorBinding("sen", 1, lv="Sen"),)
    o = regions.compile(lib, t, b, LAM)
    assert o.class_of["w/sh/S"] == o.class_of["w/W"]  # the index-matched film volume is the liquid
    assert o.kernel("w/W", "w/f1/Foil").kind is film.kind and o.kernel("w/W", "<outside>").kind is legacy.GLISUR_METAL
    # the sources, for every parameter leaf of the library
    named = regions.named_params(o, lib)
    assert src[("media", "Liq", "index", "value")] == glue.Source("materials/Liq/RINDEX", "value")
    assert src[("media", "Liq", "rayleigh", "length", "energy")] == glue.Source("materials/Liq/RAYLEIGH", "energy")
    assert src[("surfaces", "glis", "polish")] == glue.Source("surfaces/glis/value", "constant")
    assert src[("surfaces", f"film:{PLUS}", "d")].key == "surfaces/film/settings/thickness_m"
    assert src[("surfaces", f"film:{PLUS}", "b:R", "value")] == glue.Source("surfaces/paint/REFLECTIVITY", "value")
    assert src[("surfaces", f"film:{PLUS}", "eta")].column == "fixed"
    assert src[("sensors", "sen", "eff")] == glue.Source("sensors/sen/eff", "setting")
    leaves = {(s, n) + p for s in ("surfaces", "sensors") for n, ps in named[s].items() for p in _paths(ps)}
    assert leaves <= set(src) and not [k for k, v in src.items() if v.note == "unrecorded"]


def _paths(p, path=()):
    for k, v in p.items():
        yield from (_paths(v, path + (k,)) if isinstance(v, dict) else [path + (k,)])


def test_a_recipe_skin_replaces_the_gdml_skin(toy):
    df = decided(toy)
    df = dataclasses.replace(df, skins=df.skins + (schema.Skin("mirror", "glispol", "Blk", "recipe"),))
    _, lib, b, _ = glue.library_and_bindings(df, tree(*TREE), INSTANCES, COUNTING)
    assert b.skins["Blk"] == "glispol" and lib.surfaces["glispol"].kind.name.split("{")[0] == legacy.SPECULAR_METAL.name


def test_border_sensors(toy):
    df = dataclasses.replace(decided(toy), borders=(schema.Border("bs", "pdieR", "blk1", "other"),))
    group = SensorGroup("pmt", "border_into", "lumen:geant4", lv_pattern="^X$",
                        settings={"efficiency_on_entry": 1.0, "name_prefix": "tube"})  # fmt: skip
    sset = SensorSet("default", (group,) + COUNTING.groups)
    _, lib, b, src = glue.library_and_bindings(df, tree(*TREE), INSTANCES, sset)
    e = lib.sensors["tube:pdieR"]
    assert e.kind.name == "dielectric_rt_sensor{T=0}" and e.params["eff"] == 1.0  # Geant4's default T 0: fixed (OUT)
    assert b.borders == {("w/blk1/Blk", "w/other/X"): "tube:pdieR"}
    assert regions.SensorBinding("tube:pdieR", 1, lv="X", entry="border") in b.sensors
    assert src[("sensors", "tube:pdieR", "eff")] == glue.Source("sensors/pmt/efficiency_on_entry", "setting")


def test_larsoft_encodings(toy):
    from lumen.detectors import larsoft_optics as lar

    mats = dict(toy.materials)
    mats["Liq"] = dataclasses.replace(mats["Liq"], properties=lar.lar_tables(),
                                      optics=model("lumen:larsoft_lar", medium="lar", scatter=False))  # fmt: skip
    mats["Steel"] = dataclasses.replace(mats["Steel"], optics=model("lumen:opaque", medium="opaque"))
    steel = schema.Surface(
        "steel",
        "dielectric_metal",
        "unified",
        "ground",
        0.0,
        lar.reflector_tables("STEEL_STAINLESS_Fe7Cr2Ni"),
        model("lumen:larsoft_reflector"),
    )
    df = dataclasses.replace(
        toy,
        materials=mats,
        surfaces={**toy.surfaces, "steel": steel},
        borders=(),
        skins=(schema.Skin("s", "steel", "Blk", "recipe"),),
        wavelength_range=(100.0, 180.0),
    )
    t = tree(*TREE[:4])
    _, lib, b, src = glue.library_and_bindings(df, t, (), SensorSet("default", ()))
    assert b.materials["Liq"] == "lar" and not lib.media["lar"].medium.channels
    assert lib.surfaces["steel"].params["R"] == 0.25 and b.skins == {"Blk": "steel"}
    assert src[("surfaces", "steel", "R")].column == "derived"


def test_refusals(toy):
    df = decided(toy)
    bad = dataclasses.replace(df, materials={**df.materials, "Liq": dataclasses.replace(
        df.materials["Liq"], optics=model("lumen:mystery"))})  # fmt: skip
    with pytest.raises(glue.LibraryError, match="mystery"):
        glue.library_and_bindings(bad, tree(*TREE), INSTANCES, COUNTING)
    two = dataclasses.replace(df, materials={**df.materials, "Shifter": dataclasses.replace(
        df.materials["Shifter"], optics=model("geant4", medium="Liq"))})  # fmt: skip
    with pytest.raises(glue.LibraryError, match="defined by another material"):
        glue.library_and_bindings(two, tree(*TREE), INSTANCES, COUNTING)
