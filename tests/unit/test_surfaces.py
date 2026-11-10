"""The surface lookup (lumen.build.surfaces, lumen.rules.boundary; the baseline's tests/test_classes.py): classes
from symbolic signatures, Geant4 border/skin precedence and defaults, the guards (surface conflicts in a class pair,
invisible self-crossings), fail-closed bindings, sensor tags, and static tables that do not depend on parameter values
or input order."""

import dataclasses
import random

import numpy as np
import pytest

from lumen.build import surfaces as regions
from lumen.build.surfaces import (
    Bindings,
    DescriptionError,
    Library,
    MediumEntry,
    SensorBinding,
    SensorEntry,
    SurfaceEntry,
)
from lumen.build.tree import PV, Tree, TreeError
from lumen.media import bulk, laws, spectra
from lumen.surfaces import _legacy as legacy

LAM = (300.0, 600.0)
IDX, LEN = laws.constant("index"), laws.constant("length")
WATER = bulk.Medium("water", IDX, LEN)
ACRYLIC = bulk.Medium("acrylic", IDX, LEN)
LAR = bulk.Medium("lar", IDX, LEN)
STEEL = bulk.Medium("steel", None)
STEEL2 = bulk.Medium("steel2", None)
MATS = {
    "Water": "water",
    "Water2": "water",
    "Acrylic": "acrylic",
    "LAr": "lar",
    "Steel": "steel",
    "Steel304": "steel",
    "Steel316": "steel",
}


def refl(R, fs=0.3):
    return SurfaceEntry(legacy.REFLECTOR, {"R": R, "f_spec": fs, "f_glossy": 0.5, "sigma_alpha": 0.1})


def library(n_water=1.33, R=0.9, R2=0.5, eff=0.03, qe=0.25):
    def med(m, n, L=50.0):
        return MediumEntry(m, {"index": {"value": n}, "absorption": {"value": L}})

    return Library(
        media={
            "water": med(WATER, n_water),
            "acrylic": med(ACRYLIC, 1.5, 5.0),
            "lar": med(LAR, 1.23, 20.0),
            "steel": MediumEntry(STEEL, {}),
            "steel2": MediumEntry(STEEL2, {}),
        },
        surfaces={
            "refl": refl(R),
            "refl2": refl(R2, 0.0),
            "black": SurfaceEntry(legacy.ABSORBER, {}),
            "polish": SurfaceEntry(legacy.DIELECTRIC, {}, ignored_between_same_material=True),
            "polish_on": SurfaceEntry(legacy.DIELECTRIC, {}),
            "matched": SurfaceEntry(legacy.PASS, {}),
        },
        sensors={
            "arapuca": SensorEntry(legacy.COUNTING, {"eff": eff}),
            "pmt": SensorEntry(
                legacy.PHOTOCATHODE,
                {"nr": 2.7, "nk": 1.5, "qe": qe, "f_spec": 1.0, "f_glossy": 0.0, "sigma_alpha": 0.1},
            ),
        },
    )


def tree(*specs):
    return Tree(tuple(PV(*s) for s in specs))


def compile_(t, lib=None, mats=None, **kw):
    used = {p.material for p in t.pvs}
    mats = mats if mats is not None else {m: MATS[m] for m in used}
    return regions.compile(lib or library(), t, Bindings(mats, **kw), LAM)


def kern(o, pre, post):
    p = o.kernel(pre, post)
    return None if p is None else (p.kind, p.origin, p.reasons)


TANK = (
    ("World", "WorldLV", "Water", None),
    ("Box", "BoxLV", "Acrylic", "World"),
    ("Box2", "Box2LV", "Acrylic", "World"),
)


# ----------------------------------------------------------------------------------------------- classes / sensors
def test_lar_sensor_gets_its_own_class_and_the_active_volume_is_invisible():
    dets = tuple((f"det{i}", "SensLV", "LAr", "Bulk", 3 - i) for i in range(4))
    t = tree(
        ("World", "WorldLV", "LAr", None),
        ("Cryo", "CryoLV", "Steel", "World"),
        ("Bulk", "BulkLV", "LAr", "Cryo"),
        ("TPC", "TPCLV", "LAr", "Bulk"),
        *dets,
    )
    o = compile_(t, sensors=(SensorBinding("arapuca", 4, lv="SensLV"),), disjoint_placements=("SensLV",))
    c = o.class_of
    assert c["World"] == c["Bulk"] == c["TPC"] and c["det0"] != c["Bulk"] and c["Cryo"] not in (c["Bulk"], c["det0"])
    assert len({c[f"det{i}"] for i in range(4)}) == 1 and o.n_classes == 3
    k, origin, why = kern(o, "Bulk", "det1")
    assert k is legacy.COUNTING and origin == "sensor:arapuca" and why == ("sensor",)
    assert kern(o, "TPC", "det2")[0] is legacy.COUNTING  # sibling entry: same class pair, same kernel
    assert kern(o, "det0", "Bulk")[0] is legacy.PASS
    assert o.kernel("Bulk", "TPC") is None and o.kernel("World", "Bulk") is None  # one class: no boundary
    assert kern(o, "Bulk", "Cryo")[0] is legacy.ABSORBER and o.kernel("Cryo", "Bulk") is None
    # nothing entering the LAr class absorbs or detects (the active volume is plain LAr)
    for (a, b), p in o.pairs.items():
        if b == c["TPC"]:
            assert p.kind is legacy.PASS, (a, b, p.kind)
    assert o.type_of[c["Bulk"], c["det0"]] == o.surface_types.index(legacy.COUNTING)
    eff = o.params["surface_types"]["counting"]["eff"][o.row_of[c["Bulk"], c["det0"]]]
    assert abs(float(eff) - 0.03) < 1e-7
    # tags by copy number, 0..N-1
    assert [o.tag_of[f"det{i}"] for i in range(4)] == [3, 2, 1, 0] and o.tag_pvs[0] == "det3"
    assert o.tag_sensors == ("arapuca",) * 4
    assert o.row_of.dtype == np.int32 and o.type_of.dtype == np.int32 and o.medium_of.dtype == np.int32
    assert np.all(np.diag(o.row_of) == -1) and np.all(np.diag(o.type_of) == -1)


def test_touching_sensors_of_one_class_need_a_disjoint_declaration():
    dets = tuple((f"det{i}", "SensLV", "LAr", "World") for i in range(3))
    t = tree(("World", "WorldLV", "LAr", None), *dets)
    with pytest.raises(regions.OpticsError, match="touches itself"):
        compile_(t, sensors=(SensorBinding("arapuca", 3, lv="SensLV"),))
    compile_(t, sensors=(SensorBinding("arapuca", 3, lv="SensLV"),), disjoint_placements=("SensLV",))
    compile_(
        t,
        sensors=(SensorBinding("arapuca", 3, lv="SensLV"),),
        disjoint=(("det0", "det1"), ("det1", "det2"), ("det2", "det0")),
    )
    with pytest.raises(regions.OpticsError, match="touches itself"):  # one pair left touching
        compile_(t, sensors=(SensorBinding("arapuca", 3, lv="SensLV"),), disjoint=(("det0", "det1"), ("det1", "det2")))


def test_two_steels_one_entry_one_class_two_entries_two_classes():
    t = tree(
        ("World", "WorldLV", "Water", None),
        ("TankA", "ALV", "Steel304", "World"),
        ("TankB", "BLV", "Steel316", "World"),
        ("Pool", "PoolLV", "Water2", "World"),
    )
    o = compile_(t)
    assert o.class_of["TankA"] == o.class_of["TankB"]
    assert o.class_of["Pool"] == o.class_of["World"]  # two waters, one medium, no surfaces
    o2 = compile_(t, mats={"Water": "water", "Water2": "water", "Steel304": "steel", "Steel316": "steel2"})
    assert o2.class_of["TankA"] != o2.class_of["TankB"] and o2.n_classes == o.n_classes + 1
    assert o2.media_names == ("steel", "steel2", "water")
    for oo in (o, o2):
        assert kern(oo, "World", "TankA")[0] is legacy.ABSORBER and kern(oo, "World", "TankB")[0] is legacy.ABSORBER


def test_value_only_changes_leave_partition_and_tables_identical():
    t = tree(*TANK, ("Pmt", "PmtLV", "Acrylic", "World"), ("Can", "CanLV", "Steel", "World"))
    kw = dict(
        skins={"BoxLV": "refl", "Box2LV": "refl2"},
        borders={("World", "Can"): "refl"},
        sensors=(SensorBinding("pmt", 1, pv="Pmt"),),
    )
    a = compile_(t, **kw)
    lib = library(n_water=1.40, R=0.7, R2=0.2, eff=0.5, qe=0.3)
    b = compile_(t, lib=lib, **kw)
    for f in ("class_names", "signatures", "class_of", "class_pvs", "media_names", "rows", "tag_of", "report"):
        assert getattr(a, f) == getattr(b, f), f
    assert a.surface_types == b.surface_types and a.media == b.media
    for f in ("type_of", "row_of", "medium_of"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f
    ra, rb = a.params["surface_types"]["reflector"]["R"], b.params["surface_types"]["reflector"]["R"]
    assert ra.shape == rb.shape == (2,) and not np.allclose(ra, rb)
    assert float(b.params["media"]["water"]["index"]["value"]) == pytest.approx(1.40)
    assert float(b.params["surface_types"]["photocathode"]["qe"][0]) == pytest.approx(0.3)
    # re-packing params for a value-changed library reproduces them without recompiling
    rp = regions.pack(a, regions.named_params(a, lib))
    assert np.array_equal(np.asarray(rp["surface_types"]["reflector"]["R"]), rb)
    # rows gather the right surface
    cw, cb, cb2 = a.class_of["World"], a.class_of["Box"], a.class_of["Box2"]
    assert a.params["surface_types"]["reflector"]["R"][a.row_of[cw, cb]] == pytest.approx(0.9)
    assert a.params["surface_types"]["reflector"]["R"][a.row_of[cw, cb2]] == pytest.approx(0.5)
    assert a.media[a.medium_of[cw]] is WATER


# ------------------------------------------------------------------------------------------------------ precedence
PRECEDENCE = [
    # (name, skins, borders, pre, post, kind, origin, reason)
    (
        "border beats post skin",
        {"BoxLV": "refl2"},
        {("World", "Box"): "refl"},
        "World",
        "Box",
        legacy.REFLECTOR,
        "surface:refl",
        "border",
    ),
    (
        "border beats pre skin",
        {"WorldLV": "refl2"},
        {("World", "Box"): "refl"},
        "World",
        "Box",
        legacy.REFLECTOR,
        "surface:refl",
        "border",
    ),
    ("border is directional", {}, {("Box", "World"): "refl"}, "World", "Box", legacy.DIELECTRIC, None, "default"),
    (
        "border applies to its own order",
        {},
        {("Box", "World"): "refl"},
        "Box",
        "World",
        legacy.REFLECTOR,
        "surface:refl",
        "border",
    ),
    (
        "entering a daughter: post skin first",
        {"BoxLV": "refl2", "WorldLV": "refl"},
        {},
        "World",
        "Box",
        legacy.REFLECTOR,
        "surface:refl2",
        "skin(post)",
    ),
    (
        "entering a daughter: then pre skin",
        {"WorldLV": "refl"},
        {},
        "World",
        "Box",
        legacy.REFLECTOR,
        "surface:refl",
        "skin(pre)",
    ),
    (
        "leaving into the mother: pre skin first",
        {"BoxLV": "refl2", "WorldLV": "refl"},
        {},
        "Box",
        "World",
        legacy.REFLECTOR,
        "surface:refl2",
        "skin(pre)",
    ),
    (
        "leaving into the mother: then post skin",
        {"WorldLV": "refl"},
        {},
        "Box",
        "World",
        legacy.REFLECTOR,
        "surface:refl",
        "skin(post)",
    ),
    (
        "sibling: pre skin first",
        {"BoxLV": "refl", "Box2LV": "refl2"},
        {},
        "Box",
        "Box2",
        legacy.REFLECTOR,
        "surface:refl",
        "skin(pre)",
    ),
    (
        "sibling: pre skin first, reversed",
        {"BoxLV": "refl", "Box2LV": "refl2"},
        {},
        "Box2",
        "Box",
        legacy.REFLECTOR,
        "surface:refl2",
        "skin(pre)",
    ),
    (
        "sibling: then post skin",
        {"Box2LV": "refl2"},
        {},
        "Box",
        "Box2",
        legacy.REFLECTOR,
        "surface:refl2",
        "skin(post)",
    ),
    (
        "sibling border beats both skins",
        {"BoxLV": "refl", "Box2LV": "refl"},
        {("Box", "Box2"): "refl2"},
        "Box",
        "Box2",
        legacy.REFLECTOR,
        "surface:refl2",
        "border",
    ),
    ("default: different indices", {}, {}, "World", "Box", legacy.DIELECTRIC, None, "default"),
    ("paramless surface kind", {"BoxLV": "black"}, {}, "World", "Box", legacy.ABSORBER, None, "skin(post)"),
]


@pytest.mark.parametrize("case", PRECEDENCE, ids=[c[0] for c in PRECEDENCE])
def test_geant4_precedence(case):
    _, skins, borders, pre, post, kind, origin, reason = case
    o = compile_(tree(*TANK), skins=skins, borders=borders)
    k, org, why = kern(o, pre, post)
    assert (k, org) == (kind, origin) and why == (reason,), (k, org, why)


def test_same_medium_without_surfaces_is_pass_across_classes():
    t = tree(("World", "WorldLV", "Water", None), ("A", "ALV", "Water", "World"), ("B", "BLV", "Water", "A"))
    o = compile_(t, skins={"ALV": "refl"})
    assert o.class_of["B"] == o.class_of["World"] != o.class_of["A"]
    assert kern(o, "A", "B")[0] is legacy.REFLECTOR  # pre skin (B has none), daughter entry
    o = compile_(t, borders={("World", "A"): "refl"})
    assert kern(o, "A", "B") == (legacy.PASS, None, ("default",))


def test_sensor_kind_on_entry_beats_the_mother_skin():
    t = tree(("World", "WorldLV", "Water", None), ("Pmt", "PmtLV", "Acrylic", "World"))
    o = compile_(t, skins={"WorldLV": "refl"}, sensors=(SensorBinding("pmt", 1, pv="Pmt"),))
    assert kern(o, "World", "Pmt") == (legacy.PHOTOCATHODE, "sensor:pmt", ("sensor",))
    assert kern(o, "Pmt", "World")[:2] == (legacy.REFLECTOR, "surface:refl")  # leaving: ordinary precedence
    with pytest.raises(DescriptionError, match="ambiguous"):
        compile_(t, skins={"PmtLV": "refl"}, sensors=(SensorBinding("pmt", 1, pv="Pmt"),))
    with pytest.raises(DescriptionError, match="ambiguous"):
        compile_(t, borders={("World", "Pmt"): "refl"}, sensors=(SensorBinding("pmt", 1, pv="Pmt"),))


def test_no_rindex_absorbs_and_opaque_media_are_never_left():
    t = tree(
        ("World", "WorldLV", "Water", None), ("Can", "CanLV", "Steel", "World"), ("Can2", "Can2LV", "Steel", "World")
    )
    o = compile_(t)
    assert kern(o, "World", "Can") == (legacy.ABSORBER, None, ("default: NoRINDEX",))
    assert o.kernel("Can", "World") is None
    o = compile_(t, borders={("World", "Can"): "polish_on"})
    assert kern(o, "World", "Can") == (legacy.ABSORBER, None, ("border: NoRINDEX",))
    o = compile_(t, skins={"CanLV": "refl"})
    assert kern(o, "World", "Can")[:2] == (legacy.REFLECTOR, "surface:refl")
    assert kern(o, "World", "Can2")[0] is legacy.ABSORBER
    with pytest.raises(regions.OpticsError, match="opaque"):
        compile_(t, skins={"CanLV": "matched"})


def test_a_frontpainted_transmission_into_an_opaque_medium_is_absorbed_where_it_lands():
    """Geant4's frontpainted finish skips the RINDEX check: T transmits into the steel, killed at the next step."""
    import jax.numpy as jnp

    from lumen.surfaces.outcomes import ABSORB, STRAIGHT

    t = tree(("World", "WorldLV", "Water", None), ("Can", "CanLV", "Steel", "World"),
             ("Box", "BoxLV", "Acrylic", "World"))  # fmt: skip
    lib = library()
    paint = SurfaceEntry(legacy.PAINTED_T, {"R": 0.4, "T": 0.25, "f_spec": 1.0})
    lib = dataclasses.replace(lib, surfaces={**lib.surfaces, "paint": paint})
    o = compile_(t, lib=lib, skins={"CanLV": "paint", "BoxLV": "paint"})
    into = o.kernel("World", "Can")
    assert into.kind is legacy.into_opaque(legacy.PAINTED_T) and into.origin == "surface:paint"
    assert "absorbed" in into.reasons[0] and set(into.kind.params) == {"R", "T", "f_spec"}
    assert o.kernel("World", "Box").kind is legacy.PAINTED_T  # into a medium with RINDEX it transmits
    p = {k: jnp.float32(v) for k, v in paint.params.items()}
    pr, base = into.kind.probs(p, {}), legacy.PAINTED_T.probs(p, {})
    assert float(pr[STRAIGHT]) == 0.0 and float(pr[ABSORB]) == pytest.approx(float(base[ABSORB] + base[STRAIGHT]))
    assert float(jnp.sum(pr)) == pytest.approx(1.0) and STRAIGHT not in into.kind.outcomes
    assert legacy.into_opaque(legacy.PAINTED_T) is into.kind  # one object per base: one row block


def test_surface_ignored_between_the_same_material():
    t = tree(("World", "WorldLV", "Water", None), ("Box", "BoxLV", "Water", "World"))
    o = compile_(t, skins={"BoxLV": "polish"})
    k, org, why = kern(o, "World", "Box")
    assert k is legacy.PASS and "same-material" in why[0] and kern(o, "Box", "World")[0] is legacy.PASS
    o = compile_(t, skins={"BoxLV": "polish_on"})  # not flagged: acts
    assert kern(o, "World", "Box") == (legacy.DIELECTRIC, None, ("skin(post)",))
    # Geant4 compares materials, not media: a different material bound to the same medium keeps the surface
    t2 = tree(("World", "WorldLV", "Water", None), ("Box", "BoxLV", "Water2", "World"))
    o = compile_(t2, skins={"BoxLV": "polish"})
    assert kern(o, "World", "Box") == (legacy.DIELECTRIC, None, ("skin(post)",))


# ---------------------------------------------------------------------------------------------------------- guards
def test_self_touching_skinned_placements():
    t = tree(
        ("World", "WorldLV", "Water", None), ("P1", "BoxLV", "Acrylic", "World"), ("P2", "BoxLV", "Acrylic", "World")
    )
    with pytest.raises(regions.OpticsError, match="touches itself"):
        compile_(t, skins={"BoxLV": "refl"})
    o = compile_(t, skins={"BoxLV": "refl"}, disjoint=(("P2", "P1"),))
    assert o.class_of["P1"] == o.class_of["P2"] and kern(o, "World", "P1")[0] is legacy.REFLECTOR
    compile_(t, skins={"BoxLV": "refl"}, disjoint_placements=("BoxLV",))
    o = compile_(t)  # unskinned: PASS inside one class is fine
    assert o.class_of["P1"] == o.class_of["P2"]


def test_border_on_one_placement_splits_only_that_placement():
    t = tree(("World", "WorldLV", "Water", None), *((f"P{i}", "BoxLV", "Acrylic", "World") for i in (1, 2, 3)))
    o = compile_(t, borders={("P1", "World"): "refl"})
    c = o.class_of
    assert c["P2"] == c["P3"] != c["P1"] and c["World"] not in (c["P1"], c["P2"])
    assert kern(o, "P1", "World") == (legacy.REFLECTOR, "surface:refl", ("border",))
    assert kern(o, "P2", "World") == (legacy.DIELECTRIC, None, ("default",))
    assert kern(o, "World", "P1") == (legacy.DIELECTRIC, None, ("default",))
    assert kern(o, "P1", "P2") == (legacy.PASS, None, ("default",))
    assert o.pairs[(c["P2"], c["World"])].n_pv_pairs == 2


def test_class_pair_with_two_kernels_raises():
    # entering a daughter takes the post skin; the same class pair between siblings takes the pre skin
    t = tree(
        ("World", "WorldLV", "Water", None),
        ("M", "MLV", "Acrylic", "World"),
        ("D", "DLV", "Water", "M"),
        ("M2", "MLV", "Acrylic", "World"),
        ("D2", "D2LV", "Water", "World"),
    )
    with pytest.raises(regions.OpticsError, match="missed a distinction"):
        compile_(t, skins={"MLV": "refl", "DLV": "refl2", "D2LV": "refl2"}, disjoint=(("M", "D2"), ("M", "M2")))


def test_border_sensor_is_credited_to_the_sensor_pv_it_leaves():
    """WCSim's AirCathodeSurface: a detection leaving the sensor PV (into an untagged one) counts on that sensor."""
    t = tree(("World", "WorldLV", "Water", None), ("Pmt", "PmtLV", "Acrylic", "World"))
    sb = (SensorBinding("pmt", 1, pv="Pmt"),)
    plain = compile_(t, sensors=sb)
    assert plain.deposit_pre is None
    o = compile_(t, borders={("Pmt", "World"): "arapuca"}, sensors=sb)
    c = o.class_of
    k, origin, why = kern(o, "Pmt", "World")
    assert (k, origin) == (legacy.COUNTING, "sensor:arapuca") and why == ("border sensor arapuca (credited to pre)",)
    assert kern(o, "World", "Pmt")[:2] == (legacy.PHOTOCATHODE, "sensor:pmt")
    assert o.deposit_pre.dtype == bool and o.deposit_pre.shape == (o.n_classes,) * 2
    assert o.deposit_pre[c["Pmt"], c["World"]] and o.deposit_pre.sum() == 1  # entering still credits post
    with pytest.raises(DescriptionError, match="not a sensor PV"):  # nothing to credit
        compile_(t, borders={("Pmt", "World"): "arapuca"})
    both = library()
    both.surfaces["arapuca"] = SurfaceEntry(legacy.ABSORBER, {})
    with pytest.raises(DescriptionError, match="both a library surface and a library sensor"):
        compile_(t, lib=both, borders={("Pmt", "World"): "arapuca"}, sensors=sb)


def test_fail_closed_bindings():
    t = tree(*TANK)
    with pytest.raises(DescriptionError, match="without a binding"):
        compile_(t, mats={"Water": "water"})
    with pytest.raises(DescriptionError, match="do not occur"):
        compile_(t, mats={"Water": "water", "Acrylic": "acrylic", "Steel": "steel"})
    with pytest.raises(DescriptionError, match="not a library medium"):
        compile_(t, mats={"Water": "water", "Acrylic": "pmma"})
    with pytest.raises(DescriptionError, match="does not occur"):
        compile_(t, skins={"NoLV": "refl"})
    with pytest.raises(DescriptionError, match="not a library surface"):
        compile_(t, skins={"BoxLV": "gold"})
    with pytest.raises(DescriptionError, match="never touch"):
        compile_(t, borders={("Box", "Box2"): "refl"}, disjoint=(("Box", "Box2"),))
    with pytest.raises(DescriptionError, match="unknown PV"):
        compile_(t, borders={("Box", "Nope"): "refl"})
    with pytest.raises(DescriptionError, match="both touching and disjoint"):
        compile_(t, touching=(("Box", "Box2"),), disjoint=(("Box2", "Box"),))
    with pytest.raises(DescriptionError, match="siblings"):
        compile_(t, disjoint=(("World", "Box"),))
    bad = library()
    bad.surfaces["half"] = SurfaceEntry(legacy.REFLECTOR, {"R": 0.9})
    with pytest.raises(DescriptionError, match="missing"):
        compile_(t, lib=bad)
    bad = library()
    bad.surfaces["cathode"] = SurfaceEntry(legacy.COUNTING, {"eff": 0.1})
    with pytest.raises(DescriptionError, match="sensor binding"):
        compile_(t, lib=bad)
    with pytest.raises(TreeError):
        tree(("World", "WorldLV", "Water", None), ("A", "ALV", "Water", "World"), ("B", "ALV", "Acrylic", "World"))
    with pytest.raises(TreeError):
        tree(("World", "WorldLV", "Water", None), ("W2", "WLV", "Water", None))


def test_sensor_count_and_binding_errors():
    t = tree(("World", "WorldLV", "Water", None), *((f"pmt{i}", "PmtLV", "Acrylic", "World") for i in range(3)))
    ok = dict(disjoint_placements=("PmtLV",))
    o = compile_(t, sensors=(SensorBinding("pmt", 3, lv="PmtLV"),), **ok)
    assert sorted(o.tag_of.values()) == [0, 1, 2]
    with pytest.raises(DescriptionError, match="expected 4 instances, found 3"):
        compile_(t, sensors=(SensorBinding("pmt", 4, lv="PmtLV"),), **ok)
    with pytest.raises(DescriptionError, match="two sensors"):
        compile_(t, sensors=(SensorBinding("pmt", 3, lv="PmtLV"), SensorBinding("arapuca", 1, pv="pmt1")), **ok)
    with pytest.raises(DescriptionError, match="not a library sensor"):
        compile_(t, sensors=(SensorBinding("sipm", 3, lv="PmtLV"),), **ok)
    with pytest.raises(DescriptionError, match="exactly one"):
        compile_(t, sensors=(SensorBinding("pmt", 3),), **ok)


def test_laws_are_validated_over_the_wavelength_domain():
    lib = library()
    lib.media["water"] = MediumEntry(WATER, {"index": {"value": 1.33}, "absorption": {"value": -1.0}})
    with pytest.raises(laws.LawError):
        compile_(tree(*TANK), lib=lib)
    tab = bulk.Medium("water_tab", laws.tabulated("index", 2.0, 3.0))  # [413, 620] nm
    lib = library()
    lib.media["water"] = MediumEntry(tab, {"index": {"energy": np.array([2.0, 3.0]), "value": np.array([1.33, 1.34])}})
    with pytest.raises(laws.LawError):
        compile_(tree(*TANK), lib=lib)
    lib.media["water"] = MediumEntry(WATER, {"index": {"value": 1.33}})
    with pytest.raises(DescriptionError, match="missing"):
        compile_(tree(*TANK), lib=lib)


# ------------------------------------------------------------------------------------------- determinism / report
def _rich():
    specs = [
        ("World", "WorldLV", "Water", None),
        ("Can", "CanLV", "Steel", "World"),
        ("Inner", "InnerLV", "Water", "Can"),
        ("Box", "BoxLV", "Acrylic", "Inner"),
        ("Box2", "Box2LV", "Acrylic", "Inner"),
        ("Rod", "RodLV", "Steel", "Inner"),
    ]
    specs += [(f"pmt{i}", "PmtLV", "Acrylic", "Inner", (7 * i) % 5) for i in range(5)]
    kw = dict(
        skins={"BoxLV": "refl", "CanLV": "refl2"},
        borders={("Inner", "Box2"): "refl2"},
        sensors=(SensorBinding("pmt", 5, lv="PmtLV"),),
        disjoint_placements=("PmtLV",),
        disjoint=(("Box", "pmt0"),),
    )
    return specs, kw


def _shuffled(d, rnd):
    items = list(d.items())
    rnd.shuffle(items)
    return dict(items)


def _same(a, b):
    for f in (
        "class_names",
        "signatures",
        "class_of",
        "class_pvs",
        "media_names",
        "rows",
        "tag_of",
        "tag_pvs",
        "report",
    ):
        assert getattr(a, f) == getattr(b, f), f
    assert a.surface_types == b.surface_types and a.media == b.media
    for f in ("type_of", "row_of", "medium_of"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f
    for k in a.params["surface_types"]:
        for p in a.params["surface_types"][k]:
            assert np.array_equal(a.params["surface_types"][k][p], b.params["surface_types"][k][p])


def test_determinism_under_shuffled_input_and_idempotence():
    specs, kw = _rich()
    ref = compile_(tree(*specs), **kw)
    assert ref.class_of["World"] == 0
    rnd = random.Random(7)
    for _ in range(5):
        s = list(specs)
        rnd.shuffle(s)
        k2 = dict(kw, skins=_shuffled(kw["skins"], rnd), sensors=tuple(reversed(kw["sensors"])))
        mats = _shuffled({m: MATS[m] for m in ("Water", "Steel", "Acrylic")}, rnd)
        _same(ref, compile_(tree(*s), mats=mats, **k2))
    lib, t, b = library(), tree(*specs), Bindings({m: MATS[m] for m in ("Water", "Steel", "Acrylic")}, **kw)
    _same(regions.compile(lib, t, b, LAM), regions.compile(lib, t, b, LAM))
    assert b.skins == kw["skins"] and len(lib.surfaces) == len(library().surfaces)


def test_report_explains_every_pair():
    specs, kw = _rich()
    o = compile_(tree(*specs), **kw)
    r = o.report
    for word in ("border", "skin(post)", "skin(pre)", "sensor", "default", "NoRINDEX", "pmt: 5 instances"):
        assert word in r, word
    assert all(o.class_names[c] in r for c in range(o.n_classes))
    assert o.tag_of == {f"pmt{i}": t for t, i in enumerate(sorted(range(5), key=lambda i: ((7 * i) % 5, i)))}


def test_coated_skin_on_an_opaque_backing_compiles():
    """a WLS film skin (interface.coated) on an opaque volume: its STRAIGHT is the film crossing onto the backing, so
    it never puts the photon into the opaque medium (the SBND CPA foil); a non-stacked transmitting kind there still
    fails closed."""
    em = spectra.wavelength_spectrum("em", np.array([400.0, 450.0, 500.0]), np.array([0.0, 1.0, 0.0]))
    film = legacy.coated("coated:test", legacy.REFLECTOR, em)
    p = {
        "L": 1e-6,
        "d": 1e-6,
        "eta": 1.0,
        "tau": 0.0,
        "axis": np.array([1.0, 0.0, 0.0]),
        **{f"b:{k}": v for k, v in refl(0.9).params.items()},
    }
    lib = library()
    lib = dataclasses.replace(lib, surfaces={**lib.surfaces, "film": SurfaceEntry(film, p)})
    t = tree(("World", "WorldLV", "Water", None), ("Foil", "FoilLV", "Steel", "World"))
    o = compile_(t, lib, skins={"FoilLV": "film"})
    assert o.kernel("World", "Foil").kind is film
    lib2 = dataclasses.replace(lib, surfaces={**lib.surfaces, "pass": SurfaceEntry(legacy.PASS, {})})
    with pytest.raises(regions.OpticsError, match="opaque medium"):
        compile_(t, lib2, skins={"FoilLV": "pass"})


def test_a_coincident_crossing_past_a_pass_pair_takes_the_border():
    """a sensor's shell (one material with the glass around it, its own class only because it carries the cathode
    border to the cavity) thinner than a float32 band: the geometry merges glass -> shell -> cavity into one crossing
    glass -> cavity (via the shell). glass -> shell is a PASS (the shell is optically the glass), so the only
    interface is shell -> cavity: the cathode -- as Geant4, which steps into any shell thicker than its tolerance
    (g4ref 2026-11-07: 1 nm to 0.1 mm, 81.1k detections of 2e5; only an exactly coincident face skips the border)"""
    t = tree(
        ("World", "WorldLV", "Water", None),
        ("Glass", "GlassLV", "Acrylic", "World"),
        ("Body", "BodyLV", "Acrylic", "Glass"),
        ("Inner", "InnerLV", "LAr", "Body"),
    )
    o = compile_(t, sensors=(SensorBinding("pmt", 1, pv="Inner", entry="border"),), borders={("Body", "Inner"): "pmt"})
    c = o.class_of
    assert c["Glass"] != c["Body"]
    assert kern(o, "Glass", "Body")[0] is legacy.PASS
    assert kern(o, "Body", "Inner")[0].name == "photocathode"
    k, origin, reasons = kern(o, "Glass", "Inner")
    assert k.name == "photocathode" and origin == "sensor:pmt", (k, origin, reasons)
    assert any(r.startswith("pass via") for r in reasons), reasons
    assert o.tag_of["Inner"] == 0 and not (o.deposit_pre is not None and o.deposit_pre[c["Glass"], c["Inner"]])
