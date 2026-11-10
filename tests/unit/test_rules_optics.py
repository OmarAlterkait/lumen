"""lumen.rules.optics, the "geant4" optics model: property reading through the importer (CLHEP units, the ignored
matrix unit attribute, constants, '0x' names, Geant4-10 keys inert under 11, enum-valued surface attributes), media
and surfaces with Geant4 semantics (each supported case, each refused one), Geant4's GROUPVEL group index and WLS /
scintillation sampler against literal numpy translations, the WLS film (its numbers and checks), spectral surface
params through the compiler, and border surfaces on a tree (the baseline's tests/test_gdml_optics.py and the import
parts of tests/test_coated.py)."""

import dataclasses
import math

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
from lumen.surfaces import _legacy as legacy
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
    txt = "\n".join(rules.key_report(df.materials))
    print("\n" + txt)
    assert "Geant4-10 keys" in txt and "RADLENGTH" in txt


def test_enum_valued_surface_attributes(df):
    """a GDML export writes Geant4's enum values: unified, groundfrontpainted, dielectric_dielectric"""
    assert rules.surface_names(df.surfaces["enum"]) == ("unified", "groundfrontpainted", "dielectric_dielectric")
    assert rules.surface(df.surfaces["enum"]).kind is legacy.PAINTED_T  # T at Geant4's default 0


def test_scintillation_reading_conventions(df):
    with pytest.raises(rules.OpticsRefused, match="inert under Geant4 >= 11"):
        rules.scintillation(df.materials["Liq"], geant4="11")
    s = rules.scintillation(df.materials["Liq"], geant4="10.7")
    assert s["f_fast"] == 0.27 and s["tau_fast"] == 6.0 and s["same_spectra"] and abs(s["yield_per_MeV"] - 720) < 1e-9


def g4_sampler(energy, intensity, u):
    """G4OpWLS::BuildPhysicsTable (trapezoid CII at the knots) + G4PhysicsVector::GetEnergy (linear inverse)."""
    cii = np.concatenate([[0.0], np.cumsum(0.5 * np.diff(energy) * (intensity[1:] + intensity[:-1]))])
    v = u * cii[-1]
    b = np.clip(np.searchsorted(cii, v, side="left") - 1, 0, len(cii) - 2)
    d = cii[b + 1] - cii[b]
    return energy[b] + np.where(d > 0, (v - cii[b]) * (energy[b + 1] - energy[b]) / np.where(d > 0, d, 1), 0.0)


def chi2_two(a, b, edges):
    """two-sample chi2/ndf for samples of any sizes."""
    ha, hb = np.histogram(a, edges)[0].astype(float), np.histogram(b, edges)[0].astype(float)
    ka, kb = math.sqrt(hb.sum() / ha.sum()), math.sqrt(ha.sum() / hb.sum())
    ok = ha + hb > 0
    return float(np.sum((ka * ha[ok] - kb * hb[ok]) ** 2 / (ha[ok] + hb[ok])) / max(ok.sum() - 1, 1))


def test_geant4_spectrum_sampler_is_geant4s(df):
    """WLSCOMPONENT through the Geant4 sampler vs the literal Geant4 inverse (two-sample chi2 in energy)."""
    p = df.materials["Shifter"].properties["WLSCOMPONENT"]
    sp = spectra.geant4_spectrum("t", p.energy, p.value)
    u = jax.random.uniform(jax.random.PRNGKey(0), (400_000,), F32, 1e-7, 1 - 1e-7)
    E = HC / np.asarray(jax.vmap(sp.sample)(u), np.float64)
    Eg = g4_sampler(p.energy, p.value, np.random.default_rng(1).random(400_000))
    c2 = chi2_two(E, Eg, np.linspace(2.5, 3.4, 61))
    grid = np.linspace(2.5, 3.4, 9001)
    norm = np.trapezoid(sp.density_energy(grid), grid)
    print(f"\n[g4 spectrum] chi2/ndf vs Geant4 {c2:.2f}; density integral {norm:.5f}")
    assert c2 < 1.6 and abs(norm - 1) < 2e-3 and E.min() >= 2.5 - 1e-4 and E.max() <= 3.4 + 1e-4


def g4_groupvel(E, n):
    """CalculateGROUPVEL, literally (float64): -> (energies, v_g / c)."""
    eg, vg = [E[0]], [1.0 / (n[0] + (n[1] - n[0]) / math.log(E[1] / E[0]))]
    if vg[0] < 0 or vg[0] > 1.0 / n[0]:
        vg[0] = 1.0 / n[0]
    E0, n0, E1, n1 = E[0], n[0], E[1], n[1]
    for i in range(2, len(E)):
        v = 1.0 / (0.5 * (n0 + n1) + (n1 - n0) / math.log(E1 / E0))
        if v < 0 or v > 1.0 / (0.5 * (n0 + n1)):
            v = 1.0 / (0.5 * (n0 + n1))
        eg.append(0.5 * (E0 + E1))
        vg.append(v)
        E0, n0, E1, n1 = E1, n1, E[i], n[i]
    v = 1.0 / (n1 + (n1 - n0) / math.log(E1 / E0))
    if v < 0 or v > 1.0 / n1:
        v = 1.0 / n1
    eg.append(E1)
    vg.append(v)
    return np.array(eg), np.array(vg)


def test_geant4_rindex_group_index_is_geant4s_groupvel(df):
    """medium(index='geant4'): n = RINDEX (linear in E, clamped) and group_index = c / GROUPVEL; finite gradients."""
    r = df.materials["Liq"].properties["RINDEX"]
    E, n = r.energy, r.value
    law = rules.medium(df.materials["Liq"]).medium.index
    p = {"energy": jnp.asarray(E, F32), "value": jnp.asarray(n, F32)}
    lam = np.linspace(HC / (E[-1] * 1.05), HC / (E[0] * 0.95), 4001)
    got_n = np.asarray(law(p, jnp.asarray(lam, F32)), np.float64)
    got_g = np.asarray(laws.group_index(law)(p, jnp.asarray(lam, F32)), np.float64)
    eg, vg = g4_groupvel(np.asarray(E, np.float64), np.asarray(n, np.float64))
    want_g = 1.0 / np.interp(HC / lam, eg, vg)
    want_n = np.interp(HC / lam, E, n)
    gr = jax.grad(lambda v: jnp.sum(laws.group_index(law)({"energy": p["energy"], "value": v},
                                                          jnp.asarray(lam[::50], F32))))(p["value"])  # fmt: skip
    assert np.max(np.abs(got_n - want_n)) < 1e-5 and np.max(np.abs(got_g / want_g - 1)) < 1e-5
    assert np.all(np.isfinite(np.asarray(gr)))


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


def test_groupvel_export_is_accepted_a_user_table_refused(df):
    """G4GDMLWriteMaterials writes Geant4's computed GROUPVEL: that table is an export and is accepted; any other
    GROUPVEL is a user table and is refused."""
    liq = df.materials["Liq"]
    r = liq.properties["RINDEX"]
    eg, ng = (
        np.asarray(x, np.float64) for x in laws.geant4_group_index_table(jnp.asarray(r.energy), jnp.asarray(r.value))
    )
    exported = _with(liq, GROUPVEL=schema.Property("GROUPVEL", 299.792458 / ng, eg))
    assert rules.geant4_groupvel_export(exported.properties)
    assert rules.medium(exported).medium.index.form == "g4_rindex"
    user = _with(liq, GROUPVEL=schema.Property("GROUPVEL", 0.99 * 299.792458 / ng, eg))
    with pytest.raises(rules.OpticsRefused, match="user GROUPVEL"):
        rules.medium(user)


SUPPORTED = {"glis": legacy.GLISUR_METAL, "glispol": legacy.SPECULAR_METAL, "glis1": legacy.SPECULAR_METAL,
             "uni": legacy.UNIFIED_METAL, "paint": legacy.PAINTED_T, "paintT": legacy.PAINTED_T,
             "die": legacy.DIELECTRIC_RT, "pdieR": legacy.DIELECTRIC_RT}  # fmt: skip
# a dielectric_dielectric type never depends on whether R or T is present: Geant4 reads them at its defaults (R 1, T 0)
REFUSED = {"lut": "model", "dich": "model", "back": "finish", "gdie": "ground dielectric", "eff": "EFFICIENCY",
           "mT": "TRANSMITTANCE", "cplx": "REALRINDEX", "usig": "sigma_alpha", "neg": "polish"}  # fmt: skip


def test_surfaces_supported_and_refused(df):
    s = df.surfaces
    for name, kind in SUPPORTED.items():
        assert rules.surface(s[name]).kind is kind, name
    assert rules.surface(s["die"]).ignored_between_same_material
    assert not rules.surface(s["paint"]).ignored_between_same_material
    assert rules.surface(s["glis"]).params["polish"] == 0.5 and rules.surface(s["paint"]).params["f_spec"] == 0.0
    assert rules.surface(s["paintT"]).params["f_spec"] == 1.0
    rt = rules.surface(s["pdieR"])  # absorption before Fresnel (WCSim's PMT cathode surface)
    assert rt.ignored_between_same_material and rt.params["T"] == 0.0 and set(rt.params) == {"R", "T"}
    sen = rules.rt_sensor(s["pdieR"], eff=1.0)
    assert sen.kind is legacy.DIELECTRIC_RT_SENSOR and sen.params["eff"] == 1.0
    assert rules.rt_sensor(s["pdieR"]).params["eff"] == 0.0  # the surface's EFFICIENCY (none here)
    with pytest.raises(rules.OpticsRefused, match="not a polished dielectric_dielectric"):
        rules.rt_sensor(dataclasses.replace(s["eff"], name="effm"))
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


def test_coverage_lists_tables_short_of_the_range(df):
    lines = rules.coverage(df, (100.0, 900.0))
    assert any("material 'Liq'.RINDEX" in x for x in lines) and not any("WLSABSLENGTH" in x for x in lines)


# ------------------------------------------------------------------------------------------------- the WLS film
def test_wls_film_reads_its_numbers_and_checks(df):
    m, s = df.materials, df.surfaces
    e = rules.wls_film(m["Shifter"], m["Liq"], s["paint"], 1e-6, (0.0, 0.0, 1.0))
    assert e.kind.stack and e.kind.name == "coated:Shifter:painted" and e.params["eta"] == 1.0
    assert e.params["d"] == 1e-6 and e.params["tau"] == 0.0 and e.params["b:f_spec"] == 0.0
    assert np.array_equal(e.params["b:R"][1]["value"], s["paint"].properties["REFLECTIVITY"].value)
    assert np.array_equal(e.params["L"][1]["value"], m["Shifter"].properties["WLSABSLENGTH"].value)
    types = {}
    a = rules.wls_film(m["Shifter"], m["Liq"], s["paint"], 1e-6, (1.0, 0.0, 0.0), types)
    b = rules.wls_film(m["Shifter"], m["Liq"], s["paint"], 1e-6, (-1.0, 0.0, 0.0), types)
    assert a.kind is b.kind and len(types) == 1
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
