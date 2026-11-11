"""lumen.rules.optics, the "geant4" optics model: property reading through the importer (CLHEP units, the ignored
matrix unit attribute, constants, '0x' names, Geant4-10 keys inert under 11, enum-valued surface attributes), media
and surfaces with Geant4 semantics (each supported case, each refused one), Geant4's GROUPVEL group index and WLS /
scintillation sampler against literal numpy translations, the WLS film (its numbers and checks), spectral surface
params through the compiler, and border surfaces on a tree."""

import dataclasses

import helpers
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from optics_toy import TOY, read_toy, tree

from lumen.build import library as glue
from lumen.build import surfaces as regions
from lumen.format import schema
from lumen.media import bulk, laws, spectra
from lumen.rules import optics as rules
from lumen.surfaces import types as legacy
from lumen.surfaces.outcomes import ABSORB, BACKSCATTER, DETECT, DIFFUSE, GLOSSY, SPECULAR, STRAIGHT
from lumen.units import HC

F32 = jnp.float32
Z_MAX = 4.5


@pytest.fixture(scope="module")
def df(tmp_path_factory):
    return read_toy(tmp_path_factory)


def test_reading_units_and_keys(df):
    liq = df.materials["Liq"].properties
    assert np.allclose(liq["RINDEX"].energy, [2, 3, 6, 9]) and liq["RINDEX"].matrix == "RIX0x55aa"
    assert np.allclose(liq["ABSLENGTH"].value, [20.0, 0.10])  # cm and a quantity, in m
    assert float(liq["FASTTIMECONSTANT"].value) == 6.0  # the matrix unit attribute is ignored
    assert abs(float(liq["SCINTILLATIONYIELD"].value) - 720.0) < 1e-9  # 720 / MeV
    assert np.allclose(df.materials["Shifter"].properties["WLSABSLENGTH"].value, [100, 100, 1e-6, 1e-14])
    assert [(k.surface, k.volume) for k in df.skins] == [("glis", "Blk")]
    assert [(b.surface, b.pv1, b.pv2) for b in df.borders] == [("paint", "blk1", "other")]
    txt = "\n".join(helpers.key_report(df.materials))
    print("\n" + txt)
    assert "Geant4-10 keys" in txt and "RADLENGTH" not in txt  # a user key: kept inert by default, unread
    assert "RADLENGTH" in df.materials["Liq"].inert


def test_enum_valued_surface_attributes(df):
    """a GDML export writes Geant4's enum values: unified, groundfrontpainted, dielectric_dielectric"""
    assert rules.surface_names(df.surfaces["enum"]) == ("unified", "groundfrontpainted", "dielectric_dielectric")
    assert rules.surface(df.surfaces["enum"]).surface_type is legacy.PAINTED_T  # T at Geant4's default 0


def test_scintillation_reading_conventions(df):
    with pytest.raises(rules.OpticsRefused, match="inert under Geant4 >= 11"):
        rules.scintillation(df.materials["Liq"], geant4="11")
    s = rules.scintillation(df.materials["Liq"], geant4="10.7")
    assert s["f_fast"] == 0.27 and s["tau_fast"] == 6.0 and s["same_spectra"] and abs(s["yield_per_MeV"] - 720) < 1e-9


def test_media_read_geant4s_rindex_law_and_wls_spectrum(df):
    """The wiring: rules.medium reads RINDEX with laws.geant4_rindex and WLSCOMPONENT with spectra.geant4_spectrum
    (each checked against Geant4's own code in test_laws.py and test_spectra.py)."""
    r = df.materials["Liq"].properties["RINDEX"]
    p = {"energy": jnp.asarray(r.energy, F32), "value": jnp.asarray(r.value, F32)}
    lam = jnp.asarray(np.linspace(HC / (r.energy[-1] * 1.05), HC / (r.energy[0] * 0.95), 401), F32)
    law = rules.medium(df.materials["Liq"]).medium.index
    assert np.array_equal(np.asarray(law(p, lam)), np.asarray(laws.geant4_rindex()(p, lam)))
    assert np.array_equal(np.asarray(laws.group_index(law)(p, lam)),
                          np.asarray(laws.group_index(laws.geant4_rindex())(p, lam)))  # fmt: skip
    w = df.materials["Shifter"].properties["WLSCOMPONENT"]
    got = rules.medium(df.materials["Shifter"], "shifter").medium.channels[0].reemit
    want = spectra.geant4_spectrum("shifter.WLSCOMPONENT", w.energy, w.value)
    grid = np.linspace(w.energy[0], w.energy[-1], 101)
    assert np.array_equal(np.asarray(got.density_energy(grid)), np.asarray(want.density_energy(grid)))


def test_media(df):
    e = rules.medium(df.materials["Liq"])
    assert e.medium.index.form == "g4_rindex" and e.medium.absorption is not None
    assert [c.name for c in e.medium.channels] == ["rayleigh"]
    w = rules.medium(df.materials["Shifter"], "shifter")
    assert [c.name for c in w.medium.channels] == ["wls"] and w.params["wls"]["yield"] == 1.0
    assert w.medium.name == "shifter" and repr(w.medium.channels[0].reemit) == "EnergySpectrum(shifter.WLSCOMPONENT)"
    assert rules.medium(df.materials["Steel"]).medium.opaque
    bulk.validate(e.medium, e.params, 100.0, 900.0)
    ev = bulk.evaluate(e.medium, jax.tree_util.tree_map(lambda x: jnp.asarray(x, F32), e.params), F32(HC / 6.0))
    assert abs(float(ev["n"]) - 1.35) < 1e-5 and abs(float(ev["mu_abs"]) - 1 / np.interp(6, [2, 10], [20, 0.1])) < 1e-4
    with pytest.raises(rules.OpticsRefused, match="index must be"):
        rules.medium(df.materials["Liq"], index="cauchy")


def _with(material, **props):
    return dataclasses.replace(material, properties={**material.properties, **props})


def test_refused_media(df):
    liq = df.materials["Liq"]
    one = schema.Property("X", np.asarray(1.0))
    # MIEHG is read now (test_mie.py)
    for key, why in (("WLSABSLENGTH2", "second WLS"), ("WLSMEANNUMBERPHOTONS", "Poisson"),
                     ("REALRINDEX", "complex index")):  # fmt: skip
        with pytest.raises(rules.OpticsRefused, match=why):
            rules.medium(_with(liq, **{key: one}))
    with pytest.raises(rules.OpticsRefused, match="must be a table"):
        rules.medium(_with(liq, ABSLENGTH=schema.Property("ABSLENGTH", np.asarray(1.0))))
    sh = df.materials["Shifter"]
    with pytest.raises(rules.OpticsRefused, match="'delta'"):
        rules.medium(_with(sh, WLSTIMECONSTANT=schema.Property("WLSTIMECONSTANT", np.asarray(2.0))))
    rules.medium(_with(sh, WLSTIMECONSTANT=schema.Property("WLSTIMECONSTANT", np.asarray(2.0))),
                 wls_time_profile="exponential")  # fmt: skip
    without = dataclasses.replace(sh, properties={k: v for k, v in sh.properties.items() if k != "WLSCOMPONENT"})
    with pytest.raises(rules.OpticsRefused, match="without WLSCOMPONENT"):
        rules.medium(without)


def test_geant4_rayleigh_for_water(df):
    """a material named Water (its '0x' suffix stripped) without RAYLEIGH gets Geant4's table; others none."""
    liq = df.materials["Liq"]
    no_ray = dataclasses.replace(liq, properties={k: v for k, v in liq.properties.items() if k != "RAYLEIGH"})
    assert rules.geant4_rayleigh(no_ray) is None and not rules.medium(no_ray).medium.channels
    water = dataclasses.replace(no_ray, name="Water0x1234", temperature=None)
    ray = rules.geant4_rayleigh(water)
    e, n2 = ray.energy, np.asarray(no_ray.properties["RINDEX"].value) ** 2
    c1 = 7.658e-23 * 1e9 * 293.15 * 8.617333262e-11 / (6 * np.pi)
    want = 1e-3 / (c1 * (2 * np.pi * e / 1.239841984e-3) ** 4 * ((n2 - 1) * (n2 + 2) / 3) ** 2)
    assert np.allclose(ray.value, want, rtol=1e-12) and [c.name for c in rules.medium(water).medium.channels] == [
        "rayleigh"
    ]
    beta = schema.Property("ISOTHERMAL_COMPRESSIBILITY", np.asarray(7.658e-23 * 1e9))
    scale = schema.Property("RS_SCALE_FACTOR", np.asarray(2.0))
    other = _with(
        dataclasses.replace(no_ray, temperature=293.15), ISOTHERMAL_COMPRESSIBILITY=beta, RS_SCALE_FACTOR=scale
    )
    assert np.allclose(rules.geant4_rayleigh(other).value, ray.value / 2.0, rtol=1e-12)


def test_groupvel_export_is_accepted_a_user_table_not_read(df):
    """G4GDMLWriteMaterials writes Geant4's computed GROUPVEL: that table is an export and is accepted; any other
    GROUPVEL is a user table, not read, with a warning (photons timed by RINDEX's group index, unlike Geant4)."""
    liq = df.materials["Liq"]
    r = liq.properties["RINDEX"]
    eg, ng = (
        np.asarray(x, np.float64) for x in laws.geant4_group_index_table(jnp.asarray(r.energy), jnp.asarray(r.value))
    )
    exported = _with(liq, GROUPVEL=schema.Property("GROUPVEL", 299.792458 / ng, eg))
    assert rules.geant4_groupvel_export(exported.properties)
    assert rules.medium(exported).medium.index.form == "g4_rindex"
    user = _with(liq, GROUPVEL=schema.Property("GROUPVEL", 0.99 * 299.792458 / ng, eg))
    with pytest.warns(UserWarning, match="user GROUPVEL"):
        assert rules.medium(user).medium.index.form == "g4_rindex"


SUPPORTED = {"glis": legacy.GLISUR_METAL, "glispol": legacy.SPECULAR_METAL, "glis1": legacy.SPECULAR_METAL,
             "uni": legacy.UNIFIED_METAL, "paint": legacy.PAINTED_T, "paintT": legacy.PAINTED_T,
             "die": legacy.DIELECTRIC_RT, "pdieR": legacy.DIELECTRIC_RT}  # fmt: skip
# a dielectric_dielectric type never depends on whether R or T is present: Geant4 reads them at its defaults (R 1, T 0)
REFUSED = {"lut": "model", "dich": "model", "back": "finish", "gdie": "ground dielectric", "eff": "EFFICIENCY",
           "mT": "TRANSMITTANCE", "cplx": "REALRINDEX", "usig": "sigma_alpha", "neg": "polish"}  # fmt: skip


def test_surfaces_supported_and_refused(df):
    s = df.surfaces
    for name, kind in SUPPORTED.items():
        assert rules.surface(s[name]).surface_type is kind, name
    assert rules.surface(s["die"]).ignored_between_same_material
    assert not rules.surface(s["paint"]).ignored_between_same_material
    assert rules.surface(s["glis"]).params["polish"] == 0.5 and rules.surface(s["paint"]).params["f_spec"] == 0.0
    assert rules.surface(s["paintT"]).params["f_spec"] == 1.0
    rt = rules.surface(s["pdieR"])  # absorption before Fresnel (WCSim's PMT cathode surface)
    assert rt.ignored_between_same_material and rt.params["T"] == 0.0 and set(rt.params) == {"R", "T"}
    sen = helpers.rt_sensor(s["pdieR"], eff=1.0)
    assert sen.surface_type is legacy.DIELECTRIC_RT_SENSOR and sen.params["eff"] == 1.0
    assert helpers.rt_sensor(s["pdieR"]).params["eff"] == 0.0  # the surface's EFFICIENCY (none here)
    with pytest.raises(rules.OpticsRefused, match="not a polished dielectric_dielectric"):
        helpers.rt_sensor(dataclasses.replace(s["eff"], name="effm"))
    for name, why in REFUSED.items():
        with pytest.raises(rules.OpticsRefused, match=why):
            rules.surface(s[name])


def test_painted_and_unified_kernels_are_geant4s_cuts():
    """PAINTED_T: rand > R + T absorbed, rand > R transmitted, else reflected (constants summing past 1 included)."""
    N = 1_000_000
    rg = np.random.default_rng(2)
    ctx = dict(lam=F32(400.0), wi=jnp.asarray([0.0, 0.6, -0.8], F32), n=jnp.asarray([0.0, 0.0, 1.0], F32),
               n1=F32(1.3), n2=F32(1.0))  # fmt: skip
    for R, T in ((0.5, 0.3), (0.7, 0.6), (0.0, 0.2)):
        pr = np.asarray(legacy.PAINTED_T.probs({"R": F32(R), "T": F32(T), "f_spec": F32(0.0)}, ctx))
        u = rg.random(N)
        freq = np.array([np.mean(~(u > R)), np.mean((u > R) & ~(u > R + T)), np.mean(u > R + T)])
        want = pr[[DIFFUSE, STRAIGHT, ABSORB]]
        z = (freq - want) / np.sqrt(np.maximum(want * (1 - want), 1e-12) / N)
        assert np.max(np.abs(z)) < Z_MAX and abs(pr.sum() - 1) < 1e-6
    pu = np.asarray(legacy.UNIFIED_METAL.probs({"R": F32(0.8), "p_spike": F32(0.1), "p_lobe": F32(0.5),
                                                "p_back": F32(0.1), "sigma_alpha": F32(0.2)}, ctx))  # fmt: skip
    assert np.allclose(pu[[SPECULAR, GLOSSY, BACKSCATTER, DIFFUSE, ABSORB]], [0.08, 0.4, 0.08, 0.24, 0.2], atol=1e-6)
    assert pu[DETECT] == 0


def test_compile_packs_spectral_kernel_params(df):
    """three metal rows: two REFLECTIVITY tables of different lengths and one scalar; the packed laws evaluate to each
    row's table at any wavelength (Geant4's clamping included)."""
    t = tree(("<outside>", "<outside>", "Air"), ("w/W", "W", "Liq", "<outside>"),
             ("w/blk1/Blk", "Blk", "Steel", "w/W"), ("w/blk2/Blk2", "Blk2", "Steel", "w/W"),
             ("w/blk3/Blk3", "Blk3", "Steel", "w/W"), ("w/s/Sen", "Sen", "Liq", "w/W", 0))  # fmt: skip
    glis = rules.surface(df.surfaces["glis"])
    short = regions.SurfaceEntry(legacy.GLISUR_METAL, {"R": (rules.TABLE_LAWS["fraction"], {
        "energy": np.array([2.0, 8.0]), "value": np.array([0.9, 0.1])}), "polish": 0.2})  # fmt: skip
    flat = regions.SurfaceEntry(legacy.GLISUR_METAL, {"R": 0.33, "polish": 0.7})
    lib = regions.Library(media={"liq": rules.medium(df.materials["Liq"]),
                                 "opaque": regions.MediumEntry(bulk.Medium("opaque", None), {})},
                          surfaces={"glis": glis, "short": short, "flat": flat},
                          sensors={"s": regions.SensorEntry(legacy.COUNTING, {"eff": 1.0})})  # fmt: skip
    pvs = ("w/blk1/Blk", "w/blk2/Blk2", "w/blk3/Blk3", "w/s/Sen")
    b = regions.Bindings(materials={"Air": "opaque", "Liq": "liq", "Steel": "opaque"},
                         skins={"Blk": "glis", "Blk2": "short", "Blk3": "flat"},
                         sensors=(regions.SensorBinding("s", 1, lv="Sen"),),
                         disjoint=tuple((a, c) for i, a in enumerate(pvs) for c in pvs[i + 1:]))  # fmt: skip
    o = regions.compile(lib, t, b, (100.0, 900.0))
    assert ("glisur_metal", "R") in o.type_laws
    blk = o.params["surface_types"]["glisur_metal"]["R"]
    law = o.type_laws[("glisur_metal", "R")]
    lam = np.linspace(100, 900, 50)
    for i, origin in enumerate(o.rows["glisur_metal"]):
        got = np.asarray(law({k: jnp.asarray(v[i]) for k, v in blk.items()}, jnp.asarray(lam, F32)))
        e = lib.surfaces[origin.split(":", 1)[1]].params["R"]
        want = np.full_like(lam, e) if not isinstance(e, tuple) else np.interp(HC / lam, e[1]["energy"], e[1]["value"])
        assert np.max(np.abs(got - want)) < 1e-6, origin
        assert abs(float(o.params["surface_types"]["glisur_metal"]["polish"][i])
                   - lib.surfaces[origin.split(":", 1)[1]].params["polish"]) < 1e-7  # fmt: skip
    # named params and pack reproduce the compiled kernel exactly
    packed = regions.pack(o, regions.named_params(o, lib))
    for k in ("energy", "value"):
        assert np.array_equal(np.asarray(packed["surface_types"]["glisur_metal"]["R"][k]), blk[k])


def test_border_bindings_on_a_tree(df):
    pvs = (("<outside>", "<outside>", "Air"), ("w/W", "W", "Liq", "<outside>"),
           ("w/blk1/Blk", "Blk", "Steel", "w/W"), ("w/other/X", "X", "Liq", "w/W"))  # fmt: skip
    assert glue.border_bindings(df, tree(*pvs)) == {("w/blk1/Blk", "w/other/X"): "paint"}
    with pytest.raises(regions.OpticsError, match="no adjacent"):
        glue.border_bindings(df, tree(*pvs[:3]))


def _cut(*borders):
    """A detector file's geometry cut at a placement: world Wld > tank (T) > det (D, the detector volume), sib (S);
    det > d1 (X); with the given borders (pv1, pv2), and the tree of the detector volume."""
    from types import SimpleNamespace as Ns

    pl = [Ns(name=n, volume=v) for n, v in (("tank", "T"), ("det", "D"), ("sib", "S"), ("d1", "X"))]
    vols = {"Wld": Ns(daughters=[0]), "T": Ns(daughters=[1, 2]), "D": Ns(daughters=[3]), "S": Ns(daughters=[]),
            "X": Ns(daughters=[])}  # fmt: skip
    bs = tuple(Ns(name=f"b{i}", surface=f"s{i}", pv1=a, pv2=b) for i, (a, b) in enumerate(borders))
    t = tree(("<outside>", "<outside>", "Air"), ("det/D", "D", "Liq", "<outside>"), ("det/d1/X", "X", "Liq", "det/D"))
    return Ns(placements=pl, volumes=vols, borders=bs), t


@pytest.mark.parametrize(("border", "want"), [
    (("d1", "det"), {("det/d1/X", "det/D"): "s0"}),    # a daughter -> the detector placement
    (("det", "tank"), {("det/D", "<outside>"): "s0"}),  # the detector placement -> its mother
    (("tank", "det"), {("<outside>", "det/D"): "s0"}),  # the mother -> the detector placement (never crossed)
])  # fmt: skip
def test_borders_at_the_detector_volume_are_bound_as_geant4_binds_them(border, want):
    assert glue.border_bindings(*_cut(border)) == want


@pytest.mark.parametrize("border", [("sib", "tank"), ("sib", "det")])  # wholly outside; from outside into it
def test_a_border_from_outside_the_detector_volume_is_dropped(border):
    with pytest.warns(UserWarning, match="needs no action"):
        assert glue.border_bindings(*_cut(border)) == {}


def test_a_border_from_the_detector_volume_to_a_sibling_raises():
    with pytest.raises(regions.OpticsError, match="outside_surface"):
        glue.border_bindings(*_cut(("det", "sib")))


def test_coverage_lists_tables_short_of_the_range(df):
    lines = rules.coverage(df, (100.0, 900.0))
    assert any("material 'Liq'.RINDEX" in x for x in lines) and not any("WLSABSLENGTH" in x for x in lines)


# ------------------------------------------------------------------------------------------------- the WLS film
def test_wls_film_reads_its_numbers_and_checks(df):
    m, s = df.materials, df.surfaces
    e = rules.wls_film(m["Shifter"], m["Liq"], s["paint"], 1e-6, (0.0, 0.0, 1.0))
    assert e.surface_type.stack and e.surface_type.name == "coated:Shifter:painted" and e.params["eta"] == 1.0
    assert e.params["d"] == 1e-6 and e.params["tau"] == 0.0 and e.params["b:f_spec"] == 0.0
    assert np.array_equal(e.params["b:R"][1]["value"], s["paint"].properties["REFLECTIVITY"].value)
    assert np.array_equal(e.params["L"][1]["value"], m["Shifter"].properties["WLSABSLENGTH"].value)
    types = {}
    a = rules.wls_film(m["Shifter"], m["Liq"], s["paint"], 1e-6, (1.0, 0.0, 0.0), types)
    b = rules.wls_film(m["Shifter"], m["Liq"], s["paint"], 1e-6, (-1.0, 0.0, 0.0), types)
    assert a.surface_type is b.surface_type and len(types) == 1
    with pytest.raises(rules.OpticsRefused, match="index-matched"):
        rules.wls_film(m["Shifter"], m["Steel"], s["paint"], 1e-6, (0, 0, 1))
    with pytest.raises(rules.OpticsRefused, match="absorbs its own emission"):
        rules.wls_film(m["Shifter"], m["Liq"], s["paint"], 1e-3, (0, 0, 1))
    with pytest.raises(rules.OpticsRefused, match="not lowered onto a zero-thickness film"):
        rules.wls_film(_with(m["Shifter"], ABSLENGTH=m["Liq"].properties["ABSLENGTH"]), m["Liq"], s["paint"], 1e-6,
                       (0, 0, 1))  # fmt: skip


@pytest.mark.parametrize("mean", [0.7, 1.4])
def test_wls_mean_number_of_photons_is_the_yield(tmp_path_factory, mean):
    """WLSMEANNUMBERPHOTONS on the film (Geant4: Poisson(mean) photons per absorption) is read as eta."""
    toy = TOY.replace('<matrix name="WLT" coldim="1" values="0.0*ns"/>',
                      f'<matrix name="WLT" coldim="1" values="0.0*ns"/>\n  <matrix name="WLM" coldim="1" '
                      f'values="{mean}"/>')  # fmt: skip
    toy = toy.replace('<property name="WLSTIMECONSTANT" ref="WLT"/></material>',
                      '<property name="WLSTIMECONSTANT" ref="WLT"/><property name="WLSMEANNUMBERPHOTONS" ref="WLM"/>'
                      "</material>", 1)  # fmt: skip
    dfm = read_toy(tmp_path_factory, toy, "toy_mean.gdml")
    e = rules.wls_film(dfm.materials["Shifter"], dfm.materials["Liq"], dfm.surfaces["paint"], 1e-6, (0, 0, 1))
    assert abs(float(e.params["eta"]) - mean) < 1e-12
    with pytest.raises(rules.OpticsRefused, match="Poisson"):  # a bulk medium refuses the count
        rules.medium(dfm.materials["Shifter"])
