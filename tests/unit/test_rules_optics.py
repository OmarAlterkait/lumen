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
from surface_reference import ABSORB, DETECT
from test_surface_types import full

from lumen.build import library as glue
from lumen.build import lookup as regions
from lumen.format import schema
from lumen.format.calibrations import Table
from lumen.media import bulk, laws
from lumen.rules import optics as rules
from lumen.surfaces import types
from lumen.surfaces.outcomes import BACKSCATTER, DIFFUSE, GLOSSY, SPECULAR, STRAIGHT
from lumen.tools.stats import z_limit
from lumen.units import HC

F32 = jnp.float32


@pytest.fixture(scope="module")
def df(tmp_path_factory):
    return read_toy(tmp_path_factory)


def test_reading_units_and_keys(df):
    liq = df.materials["Liq"].properties
    assert np.allclose(liq["RINDEX"].energy, [2, 3, 6, 9])
    assert np.allclose(liq["ABSLENGTH"].value, [20.0, 0.10])  # cm and a quantity, in m
    assert float(liq["SCINTILLATIONTIMECONSTANT1"].value) == 6.0  # the matrix unit attribute is ignored
    assert abs(float(liq["SCINTILLATIONYIELD"].value) - 720.0) < 1e-9  # 720 / MeV
    assert np.allclose(df.materials["Shifter"].properties["WLSABSLENGTH"].value, [100, 100, 1e-6, 1e-14])
    assert [(k.surface, k.volume) for k in df.skins] == [("glis", "Blk")]
    assert [(b.surface, b.pv1, b.pv2) for b in df.borders] == [("paint", "blk1", "other")]
    txt = "\n".join(helpers.key_report(df.materials))
    print("\n" + txt)
    assert "Geant4-10 keys" not in txt and "RADLENGTH" not in txt  # read as Geant4 11's; a user key kept inert
    assert "RADLENGTH" in df.materials["Liq"].inert


def test_enum_valued_surface_attributes(df):
    """A GDML's enum values (model 1, finish 4, type 1) are named at import: unified, groundfrontpainted,
    dielectric_dielectric; a value beyond Geant4's enum is refused."""
    s = df.surfaces["enum"]
    assert (s.model, s.finish, s.type) == ("unified", "groundfrontpainted", "dielectric_dielectric")
    assert rules.surface(df.surfaces["enum"]).surface_type == types.no_detection(types.PAINTED_T)  # T at its default 0
    from lumen.gdml import importer

    with pytest.raises(importer.ImportRefused, match="finish 40 is no value of Geant4's enum"):
        importer._enum_name("s", "finish", importer.FINISHES, "40")


def test_scintillation_reading_conventions(df):
    """The GDML's Geant4-10 keys are read as Geant4 11's at import (FAST, SLOW as components 1, 2; YIELDRATIO as the
    yields): Geant4 10.7's light. An incomplete 10.7 reading (two components without YIELDRATIO, where Geant4 10.7
    aborts) is left as it is, inert under Geant4 11, and the source refuses it, naming the keys."""
    from lumen.rules import keys
    from lumen.sources import scintillation as sc

    m = df.materials["Liq"].properties
    assert not any(k in m for k in keys.G4_10_KEYS)
    vals = {f"materials/Liq/{k}": (Table(v.energy, v.value) if v.energy is not None else float(v.value))
            for k, v in m.items()}  # fmt: skip
    s = sc.ScintillationPoint.of(vals, "Liq")
    y, (f, taus, _, rises) = s.light_of(vals)
    assert rises is None and not s.rises  # no rise time: the decay alone (Geant4's default)
    assert np.allclose(f, [0.27, 0.73]) and float(taus[0]) == 6.0 and abs(float(y) - 720) < 1e-9
    old = {k.replace("SCINTILLATIONCOMPONENT1", "FASTCOMPONENT").replace("SCINTILLATIONCOMPONENT2", "SLOWCOMPONENT")
           .replace("SCINTILLATIONTIMECONSTANT1", "FASTTIMECONSTANT").replace("SCINTILLATIONTIMECONSTANT2",
                                                                         "SLOWTIMECONSTANT"): v
           for k, v in m.items() if not k.startswith("SCINTILLATIONYIELD") or k == "SCINTILLATIONYIELD"}  # fmt: skip
    assert keys.geant4_11_scintillation(old) == (old, None)  # no YIELDRATIO: kept as it is
    vals = {f"materials/Liq/{k}": (Table(v.energy, v.value) if v.energy is not None else float(v.value))
            for k, v in old.items()}  # fmt: skip
    with pytest.raises(KeyError, match="inert under Geant4 >= 11"):
        sc.ScintillationPoint.of(vals, "Liq")


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
    e = rules.medium(df.materials["Shifter"], "shifter")
    assert repr(e.medium.channels[0].emission) == "EmissionRecord(shifter.wls)"
    got = e.params["wls"]["spectrum"]  # a parameter: the re-emission spectrum has a derivative
    assert np.array_equal(got["energy"], w.energy) and np.array_equal(got["value"], w.value)


def test_media(df):
    e = rules.medium(df.materials["Liq"])
    assert e.medium.index.form == "g4_rindex" and e.medium.absorption is not None
    assert [c.name for c in e.medium.channels] == ["rayleigh"]
    w = rules.medium(df.materials["Shifter"], "shifter")
    assert [c.name for c in w.medium.channels] == ["wls"] and w.params["wls"]["yield"] == 1.0
    assert w.medium.name == "shifter" and w.medium.channels[0].emission.conserving
    assert rules.medium(df.materials["Steel"]).medium.opaque
    bulk.validate(e.medium, e.params)
    ev = bulk.evaluate(e.medium, jax.tree_util.tree_map(lambda x: jnp.asarray(x, F32), e.params), F32(HC / 6.0))
    assert abs(float(ev["n"]) - 1.35) < 1e-5 and abs(float(ev["mu_abs"]) - 1 / np.interp(6, [2, 10], [20, 0.1])) < 1e-4


def _with(material, **props):
    return dataclasses.replace(material, properties={**material.properties, **props})


def test_refused_media(df):
    liq = df.materials["Liq"]
    one = schema.Property("X", np.asarray(1.0))
    # MIEHG is read now (test_mie.py)
    for key, why in (("WLSABSLENGTH2", "second WLS"), ("WLSMEANNUMBERPHOTONS2", "second WLS"),
                     ("REALRINDEX", "complex index")):  # fmt: skip
        with pytest.raises(rules.OpticsRefused, match=why):
            rules.medium(_with(liq, **{key: one}))
    with pytest.raises(rules.OpticsRefused, match="must be a table"):
        rules.medium(_with(liq, ABSLENGTH=schema.Property("ABSLENGTH", np.asarray(1.0))))
    sh = df.materials["Shifter"]
    slow = _with(sh, WLSTIMECONSTANT=schema.Property("WLSTIMECONSTANT", np.asarray(2.0)))
    # Geant4's default profile, delta (a fixed delay tau), is read when the material records none
    ch = rules.medium(slow).medium.channels[-1]
    assert ch.emission.profile.components == ("delta",) and rules.medium(slow).params["wls"]["tau"] == 2.0
    # the material records the Geant4 run's exponential profile (a recipe decision), and the build reads it
    expo = dataclasses.replace(slow, optics=schema.OpticsModel("geant4", settings={"wls_time_profile": "exponential"}))
    assert rules.medium(expo).params["wls"]["tau"] == 2.0 and rules.medium(expo).medium.channels[
        -1
    ].emission.profile.components == ("decay",)
    from lumen.build import library

    name, entry, src = library.geant4_medium(expo)
    assert entry.params["wls"]["tau"] == 2.0 and src[("wls", "tau")].key == f"materials/{sh.name}/WLSTIMECONSTANT"
    other = dataclasses.replace(slow, optics=schema.OpticsModel("geant4", settings={"wls_time_profile": "gaussian"}))
    with pytest.raises(rules.OpticsRefused, match="not one of G4OpWLS's"):
        rules.medium(other)
    without = dataclasses.replace(sh, properties={k: v for k, v in sh.properties.items() if k != "WLSCOMPONENT"})
    with pytest.raises(rules.OpticsRefused, match="without WLSCOMPONENT"):
        rules.medium(without)


# Geant4 10.7.4's G4OpRayleigh tables (BuildPhysicsTable, then Value at the knots) for RINDEX 1.33 / 1.34 at 2 / 3 eV,
# material temperature 293.15 K: a material named "Water" (no compressibility given) and one named "Other" with
# ISOTHERMAL_COMPRESSIBILITY 7.658e-23 m3/MeV. Dumped by a probe linking Geant4 (G4OpRayleigh), printed %.10g.
G4_RAYLEIGH_M = {"Water": (1024.461817, 186.3585258), "Other": (989.5151409, 180.001421)}


def test_geant4_rayleigh_follows_g4oprayleigh(df):
    """G4OpRayleigh's order: a RAYLEIGH table wins; else a material named Water (its '0x' suffix stripped) gets
    Geant4's compressibility of water at 283.15 K whatever its own; else ISOTHERMAL_COMPRESSIBILITY at the material's
    temperature (RS_SCALE_FACTOR scales both); else none. Checked against Geant4's own tables."""
    liq = df.materials["Liq"]
    rindex = schema.Property("RINDEX", np.array([1.33, 1.34]), energy=np.array([2.0, 3.0]))
    base = dataclasses.replace(liq, temperature_k=293.15, properties={"RINDEX": rindex})
    assert rules.geant4_rayleigh(base) is None and not rules.medium(base).medium.channels
    beta = schema.Property("ISOTHERMAL_COMPRESSIBILITY", np.asarray(7.658e-23))  # m3 / MeV
    for name, props in (("Water0x1234", {}), ("Water", {"ISOTHERMAL_COMPRESSIBILITY": beta}),
                        ("Other", {"ISOTHERMAL_COMPRESSIBILITY": beta})):  # fmt: skip
        m = _with(dataclasses.replace(base, name=name), **props)
        ray = rules.geant4_rayleigh(m)
        rel = np.abs(ray.value / G4_RAYLEIGH_M[name.split("0x")[0]] - 1.0)
        assert np.all(rel <= 4 * np.finfo(np.float32).eps), (name, ray.value)  # float32, the engine's precision
        assert [c.name for c in rules.medium(m).medium.channels] == ["rayleigh"]
        hot = rules.geant4_rayleigh(dataclasses.replace(m, temperature_k=350.0))
        assert np.array_equal(hot.value, ray.value) == (name.split("0x")[0] == "Water")  # Water ignores its T
        scaled = rules.geant4_rayleigh(_with(m, RS_SCALE_FACTOR=schema.Property("RS_SCALE_FACTOR", np.asarray(2.0))))
        assert np.allclose(scaled.value, ray.value / 2.0, rtol=4 * np.finfo(np.float32).eps)
    own = schema.Property("RAYLEIGH", np.array([5.0, 6.0]), energy=np.array([2.0, 3.0]))
    water = _with(dataclasses.replace(base, name="Water"), RAYLEIGH=own)
    assert rules.medium(water).params["rayleigh"]["length"]["value"].tolist() == [5.0, 6.0]  # a table wins


def test_a_groupvel_table_times_the_photons_else_rindex_s(df):
    """A GROUPVEL table in the file times the photons, as in Geant4 (n_g = c / GROUPVEL, linear in energy, clamped),
    a parameter as any table; without one the photons are timed by RINDEX's group index; a constant GROUPVEL is inert
    (G4Track reads the property vector); a speed <= 0 is refused."""
    liq = df.materials["Liq"]
    r = liq.properties["RINDEX"]
    assert rules.medium(liq).medium.index.form == "g4_rindex"
    gv_e, gv_v = np.array([2.0, 3.0, 4.0]), np.array([0.18, 0.2, 0.15])  # m / ns
    user = _with(liq, GROUPVEL=schema.Property("GROUPVEL", gv_v, energy=gv_e))
    entry = rules.medium(user)
    assert entry.medium.index.form == "g4_rindex_groupvel"
    e = np.array([1.5, 2.0, 2.5, 3.5, 4.0, 5.0])  # eV: clamped outside the knots
    out = bulk.evaluate(entry.medium, entry.params, jnp.asarray(HC / e, jnp.float32))
    want = 0.299792458 / np.interp(e, gv_e, gv_v)
    assert np.allclose(np.asarray(out["n_g"]), want, rtol=1e-6)
    assert np.allclose(np.asarray(out["n"]), np.interp(e, r.energy, r.value), rtol=1e-6)  # the index is unchanged
    const = _with(liq, GROUPVEL=schema.Property("GROUPVEL", np.asarray(0.2)))
    assert rules.medium(const).medium.index.form == "g4_rindex"
    with pytest.raises(rules.OpticsRefused, match=r"\(0, c\] in m / ns"):
        rules.medium(_with(liq, GROUPVEL=schema.Property("GROUPVEL", -gv_v, energy=gv_e)))


def test_a_group_velocity_faster_than_light_is_refused(df):
    """A GROUPVEL in mm / ns (Geant4's internal unit, lumen's before) is 1000 times faster than light: refused at
    load, table or constant, naming m / ns; params validation bounds GROUPVEL by c too (lumen.units.C_VACUUM, exact);
    c itself is a speed light has. A speed derived from RINDEX is Geant4's, not refused."""
    from lumen.rules.keys import PARAMS
    from lumen.units import C_VACUUM

    liq = df.materials["Liq"]
    gv_e, mm_ns = np.array([2.0, 3.0, 4.0]), np.array([180.0, 200.0, 150.0])  # water's group velocity, mm / ns
    for gv in (schema.Property("GROUPVEL", mm_ns, energy=gv_e), schema.Property("GROUPVEL", np.asarray(200.0))):
        with pytest.raises(rules.OpticsRefused, match=r"m / ns.*1000 times too fast"):
            rules.medium(_with(liq, GROUPVEL=gv))
    assert rules.medium(_with(liq, GROUPVEL=schema.Property("GROUPVEL", np.full(3, C_VACUUM), energy=gv_e))) is not None
    assert PARAMS["GROUPVEL"].range == (0.0, C_VACUUM) and PARAMS["GROUPVEL"].unit == "m/ns"
    r = liq.properties["RINDEX"]  # a RINDEX Geant4 accepts is not refused: its derived speed is Geant4's (c / n)
    assert rules.medium(_with(liq, RINDEX=schema.Property("RINDEX", np.full(len(r.energy), 0.9), energy=r.energy)))


SUPPORTED = {"glis": types.GLISUR_METAL, "glispol": types.SPECULAR_METAL, "glis1": types.SPECULAR_METAL,
             "uni": types.UNIFIED_METAL, "paint": types.PAINTED_T, "paintT": types.PAINTED_T,
             "die": types.DIELECTRIC_RT, "pdieR": types.DIELECTRIC_RT,
             "mT": types.transmitting(types.SPECULAR_METAL), "cplx": types.PHOTOCATHODE}  # fmt: skip
SUPPORTED = {k: types.no_detection(v) for k, v in SUPPORTED.items()}  # a surface is the type with eff fixed at 0
# a dielectric_dielectric type never depends on whether R or T is present: Geant4 reads them at its defaults (R 1, T 0)
# Geant4 reads a metal's TRANSMITTANCE (straight on) and a complex index on any surface: both are read, not refused
REFUSED = {"lut": "model", "dich": "model", "back": "finish", "gdie": "ground dielectric", "usig": "sigma_alpha",
           "neg": "polish"}  # fmt: skip


def test_surfaces_supported_and_refused(df):
    s = df.surfaces
    for name, kind in SUPPORTED.items():
        assert rules.surface(s[name]).surface_type == kind, name
    assert rules.surface(s["die"]).ignored_between_same_material
    assert not rules.surface(s["paint"]).ignored_between_same_material
    assert rules.surface(s["glis"]).params["polish"] == 0.5 and rules.surface(s["paint"]).params["f_spec"] == 0.0
    assert rules.surface(s["paintT"]).params["f_spec"] == 1.0
    rt = rules.surface(s["pdieR"])  # absorption before Fresnel (WCSim's PMT cathode surface)
    assert rt.ignored_between_same_material and rt.params["T"] == 0.0 and set(rt.params) == {"R", "T"}
    sen = helpers.rt_sensor(s["pdieR"], eff=1.0)
    assert sen.surface_type is types.DIELECTRIC_RT and sen.params["eff"] == 1.0
    assert helpers.rt_sensor(s["pdieR"]).params["eff"] == 0.0  # the surface's EFFICIENCY (none here)
    with pytest.raises(rules.OpticsRefused, match="not a polished dielectric_dielectric"):
        helpers.rt_sensor(dataclasses.replace(s["eff"], name="effm"))
    for name, why in REFUSED.items():
        with pytest.raises(rules.OpticsRefused, match=why):
            rules.surface(s[name])
    # EFFICIENCY on a surface no sensor is bound to is read as Geant4 reads it there: detection and absorption both
    # end the photon, so it changes no light (the surface is its type with eff fixed at 0) and is no parameter
    assert rules.surface(s["eff"]).surface_type.name.endswith("{eff=0}")


def test_painted_and_unified_kernels_are_geant4s_cuts():
    """PAINTED_T: rand > R + T absorbed, rand > R transmitted, else reflected (constants summing past 1 included)."""
    N = 1_000_000
    rg = np.random.default_rng(2)
    ctx = dict(lam=F32(400.0), wi=jnp.asarray([0.0, 0.6, -0.8], F32), n=jnp.asarray([0.0, 0.0, 1.0], F32),
               n1=F32(1.3), n2=F32(1.0))  # fmt: skip
    for R, T in ((0.5, 0.3), (0.7, 0.6), (0.0, 0.2)):
        pr = np.asarray(full(types.PAINTED_T, {"R": F32(R), "T": F32(T), "f_spec": F32(0.0), "eff": F32(0.0)}, ctx))
        u = rg.random(N)
        freq = np.array([np.mean(~(u > R)), np.mean((u > R) & ~(u > R + T)), np.mean(u > R + T)])
        want = pr[[DIFFUSE, STRAIGHT, ABSORB]]
        z = (freq - want) / np.sqrt(np.maximum(want * (1 - want), 1e-12) / N)
        assert np.max(np.abs(z)) < z_limit(9) and abs(pr.sum() - 1) < 1e-6  # 3 outcomes x 3 settings
    pu = np.asarray(full(types.UNIFIED_METAL, {"R": F32(0.8), "p_spike": F32(0.1), "p_lobe": F32(0.5),
                                                "p_back": F32(0.1), "sigma_alpha": F32(0.2), "eff": F32(0.0)},
                                               ctx))  # fmt: skip
    assert np.allclose(pu[[SPECULAR, GLOSSY, BACKSCATTER, DIFFUSE, ABSORB]], [0.08, 0.4, 0.08, 0.24, 0.2], atol=1e-6)
    assert pu[DETECT] == 0


def test_compile_packs_spectral_kernel_params(df):
    """three metal rows: two REFLECTIVITY tables of different lengths and one scalar; the packed laws evaluate to each
    row's table at any wavelength (Geant4's clamping included)."""
    t = tree(("<outside>", "<outside>", "Air"), ("w/W", "W", "Liq", "<outside>"),
             ("w/blk1/Blk", "Blk", "Steel", "w/W"), ("w/blk2/Blk2", "Blk2", "Steel", "w/W"),
             ("w/blk3/Blk3", "Blk3", "Steel", "w/W"), ("w/s/Sen", "Sen", "Liq", "w/W", 0))  # fmt: skip
    glis = rules.surface(df.surfaces["glis"])
    short = regions.SurfaceEntry(types.no_detection(types.GLISUR_METAL), {"R": (rules.TABLE_LAWS["fraction"], {
        "energy": np.array([2.0, 8.0]), "value": np.array([0.9, 0.1])}), "polish": 0.2})  # fmt: skip
    flat = regions.SurfaceEntry(types.no_detection(types.GLISUR_METAL), {"R": 0.33, "polish": 0.7})
    lib = regions.Library(media={"liq": rules.medium(df.materials["Liq"]),
                                 "opaque": regions.MediumEntry(bulk.Medium("opaque", None), {})},
                          surfaces={"glis": glis, "short": short, "flat": flat},
                          sensors={"s": regions.SensorEntry(types.COUNTING, {"eff": 1.0})})  # fmt: skip
    b = regions.Bindings(materials={"Air": "opaque", "Liq": "liq", "Steel": "opaque"},
                         skins={"Blk": "glis", "Blk2": "short", "Blk3": "flat"},
                         sensors=(regions.SensorBinding("s", 1, lv="Sen"),))  # fmt: skip
    o = regions.compile(lib, t, b)
    assert "R" in o.table.laws and len(o.table.laws["R"]) == 1
    blk = o.params["surfaces"]["R"]
    law = o.table.laws["R"][0]
    lam = np.linspace(100, 900, 50)
    for i, (st, origin) in enumerate(o.rows):
        if st.name != "glisur_metal{eff=0}":
            continue
        got = np.asarray(law({k: jnp.asarray(v[i]) for k, v in blk.items()}, jnp.asarray(lam, F32)))
        e = lib.surfaces[origin.split(":", 1)[1]].params["R"]
        want = np.full_like(lam, e) if not isinstance(e, tuple) else np.interp(HC / lam, e[1]["energy"], e[1]["value"])
        assert np.max(np.abs(got - want)) < 1e-6, origin
        assert abs(float(o.params["surfaces"]["polish"][i])
                   - lib.surfaces[origin.split(":", 1)[1]].params["polish"]) < 1e-7  # fmt: skip
    # named params and pack reproduce the compiled kernel exactly
    packed = regions.pack(o, regions.named_params(o, lib))
    for k in ("energy", "value"):
        assert np.array_equal(np.asarray(packed["surfaces"]["R"][k]), blk[k])


def test_border_bindings_on_a_tree(df):
    pvs = (("<outside>", "<outside>", "Air"), ("w/W", "W", "Liq", "<outside>"),
           ("w/blk1/Blk", "Blk", "Steel", "w/W"), ("w/other/X", "X", "Liq", "w/W"))  # fmt: skip
    assert glue.border_bindings(df, tree(*pvs)) == {("blk1", "other"): "paint"}
    with pytest.raises(regions.OpticsError, match="no adjacent"):
        glue.border_bindings(df, tree(*pvs[:3]))


def _cut(*borders):
    """A detector file's geometry cut at a placement: world Wld > tank (T) > det (D, the detector volume), sib (S);
    det > d1 (X); with the given borders (pv1, pv2), and the tree of the detector volume."""
    from types import SimpleNamespace as Ns

    pl = [Ns(name=n, volume=v, mother=m) for n, v, m in (("tank", "T", "Wld"), ("det", "D", "T"), ("sib", "S", "T"),
                                                          ("d1", "X", "D"))]  # fmt: skip
    vols = {
        "Wld": Ns(daughters=[0], solid=None),
        "T": Ns(daughters=[1, 2], solid=None),
        "D": Ns(daughters=[3], solid=None),
        "S": Ns(daughters=[], solid=None),
        "X": Ns(daughters=[], solid=None),
    }
    bs = tuple(Ns(name=f"b{i}", surface=f"s{i}", pv1=a, pv2=b) for i, (a, b) in enumerate(borders))
    t = tree(("<outside>", "<outside>", "Air"), ("det/D", "D", "Liq", "<outside>"), ("det/d1/X", "X", "Liq", "det/D"))
    return Ns(placements=pl, volumes=vols, borders=bs, detector_volume="D", world="Wld"), t


@pytest.mark.parametrize(("border", "want"), [
    (("d1", "det"), {("d1", "det"): "s0"}),            # a daughter -> the detector placement
    (("det", "tank"), {("det", "<outside>"): "s0"}),   # the detector placement -> its mother
    (("tank", "det"), {("<outside>", "det"): "s0"}),   # the mother -> the detector placement (never crossed)
])  # fmt: skip
def test_borders_at_the_detector_volume_are_bound_as_geant4_binds_them(border, want):
    assert glue.border_bindings(*_cut(border)) == want


@pytest.mark.parametrize("border", [("sib", "tank"), ("sib", "det")])  # wholly outside; from outside into it
def test_a_border_from_outside_the_detector_volume_is_dropped(border):
    with pytest.warns(UserWarning, match="needs no action"):
        assert glue.border_bindings(*_cut(border)) == {}


def test_a_border_from_the_detector_volume_to_a_sibling_raises():
    with pytest.raises(regions.OpticsError, match="outside is one region"):
        glue.border_bindings(*_cut(("det", "sib")))


@pytest.mark.parametrize("mean", [0.7, 1.4])
def test_wls_mean_number_of_photons_is_the_yield(tmp_path_factory, mean):
    """WLSMEANNUMBERPHOTONS (Geant4: Poisson(mean) photons per absorption) is the expected yield."""
    toy = TOY.replace('<matrix name="WLT" coldim="1" values="0.0*ns"/>',
                      f'<matrix name="WLT" coldim="1" values="0.0*ns"/>\n  <matrix name="WLM" coldim="1" '
                      f'values="{mean}"/>')  # fmt: skip
    toy = toy.replace('<property name="WLSTIMECONSTANT" ref="WLT"/></material>',
                      '<property name="WLSTIMECONSTANT" ref="WLT"/><property name="WLSMEANNUMBERPHOTONS" ref="WLM"/>'
                      "</material>", 1)  # fmt: skip
    dfm = read_toy(tmp_path_factory, toy, "toy_mean.gdml")
    assert abs(float(rules.medium(dfm.materials["Shifter"]).params["wls"]["yield"]) - mean) < 1e-12


def test_a_scintillation_rise_time_is_read_only_under_geant4s_switch(df):
    """SCINTILLATIONRISETIMEi without the optics setting scint_finite_rise_time (Geant4's finite-rise-time switch, off
    by default) is refused, never silently ignored; the switch without an emitted component's rise time is refused
    (G4Scintillation aborts); with both, the source's delays are the decay plus the formation (mean tau + tau tau_r /
    (tau + tau_r) per component)."""
    from lumen.sources import scintillation as sc

    liq = df.materials["Liq"]
    rise = {f"SCINTILLATIONRISETIME{i}": schema.Property(f"SCINTILLATIONRISETIME{i}", np.asarray(r))
            for i, r in ((1, 2.0), (2, 0.0))}  # fmt: skip
    with pytest.raises(rules.OpticsRefused, match="without the optics setting scint_finite_rise_time"):
        rules.medium(_with(liq, **rise))
    on = schema.OpticsModel("geant4", settings={"scint_finite_rise_time": True})
    with pytest.raises(rules.OpticsRefused, match=r"without \['SCINTILLATIONRISETIME2'\]"):
        rules.medium(dataclasses.replace(_with(liq, SCINTILLATIONRISETIME1=rise["SCINTILLATIONRISETIME1"]), optics=on))
    m = dataclasses.replace(_with(liq, **rise), optics=on)
    assert rules.medium(m) is not None
    vals = {f"materials/Liq/{k}": (Table(v.energy, v.value) if v.energy is not None else float(v.value))
            for k, v in m.properties.items()}  # fmt: skip
    s = sc.ScintillationPoint.of(vals, "Liq", energy=1.0)
    _, (f, taus, _, rises) = s.light_of(vals)
    assert s.rises and [float(r) for r in rises] == [2.0, 0.0]
    v = {**vals, "position": jnp.zeros(3, F32), "energy": F32(1.0), "time": F32(0.0)}
    out = s.draw(v, v, jax.random.PRNGKey(0), 200_000, 0, 200_000)
    t = np.asarray(out.t)
    want = sum(float(fi) * (float(ti) + float(ti) * float(ri) / (float(ti) + float(ri)))
               for fi, ti, ri in zip(f, taus, rises, strict=True))  # fmt: skip
    assert abs(t.mean() - want) < 5 * t.std() / np.sqrt(t.size), (t.mean(), want)
