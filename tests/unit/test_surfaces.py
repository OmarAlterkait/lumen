"""The surface lookup (lumen.build.lookup, lumen.rules.boundary): classes
from symbolic signatures, Geant4 border/skin precedence and defaults, crossings between two volumes of one class on the
class table's diagonal, the guard (surface conflicts in a class pair), fail-closed bindings, sensor tags, and static
tables that do not depend on parameter values or input order."""

import dataclasses
import random

import jax
import numpy as np
import pytest
from test_surface_types import full

from lumen.build import lookup as regions
from lumen.build import tracked
from lumen.build.lookup import (
    Bindings,
    DescriptionError,
    Library,
    MediumEntry,
    SensorBinding,
    SensorEntry,
    SurfaceEntry,
)
from lumen.build.tree import PV, Tree, TreeError
from lumen.media import bulk, laws
from lumen.rules.boundary import FAR_ABSORB, FAR_KILL
from lumen.surfaces import types

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
    return SurfaceEntry(types.REFLECTOR, {"R": R, "f_spec": fs})


def library(n_water=1.33, R=0.9, R2=0.5, eff=0.03, eff_pmt=0.25):
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
            "black": SurfaceEntry(types.ABSORBER, {}),
            "polish": SurfaceEntry(types.DIELECTRIC, {}, ignored_between_same_material=True, far_side=FAR_KILL),
            "polish_on": SurfaceEntry(types.DIELECTRIC, {}, far_side=FAR_KILL),
            "matched": SurfaceEntry(types.PASS, {}),
        },
        sensors={
            "arapuca": SensorEntry(types.COUNTING, {"eff": eff}),
            "pmt": SensorEntry(
                types.PHOTOCATHODE,
                {"nr": 2.7, "nk": 1.5, "eff": eff_pmt},
            ),
        },
    )


def tree(*specs):
    return Tree(tuple(PV(*s) for s in specs))


def compile_(t, lib=None, mats=None, **kw):
    used = {p.material for p in t.pvs}
    mats = mats if mats is not None else {m: MATS[m] for m in used}
    return regions.compile(lib or library(), t, Bindings(mats, **kw))


def kern(o, pre, post):
    p = o.pair(pre, post)
    return None if p is None else (p.surface_type, p.origin, p.reasons)


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
    o = compile_(t, sensors=(SensorBinding("arapuca", 4, lv="SensLV"),))
    c = o.class_of
    assert c["World"] == c["Bulk"] == c["TPC"] and c["det0"] != c["Bulk"] and c["Cryo"] not in (c["Bulk"], c["det0"])
    assert len({c[f"det{i}"] for i in range(4)}) == 1 and o.n_classes == 3
    k, origin, why = kern(o, "Bulk", "det1")
    assert k is types.COUNTING and origin == "sensor:arapuca" and why == ("sensor",)
    assert kern(o, "TPC", "det2")[0] is types.COUNTING  # sibling entry: same class pair, same kernel
    assert kern(o, "det0", "Bulk")[0] is types.PASS
    assert o.pair("Bulk", "TPC") is None and o.pair("World", "Bulk") is None  # one class: no boundary
    assert kern(o, "Bulk", "Cryo")[0] is types.ABSORBER and o.pair("Cryo", "Bulk") is None
    # nothing entering the LAr class absorbs or detects (the active volume is plain LAr)
    for (a, b), p in o.pairs.items():
        if b == c["TPC"]:
            assert p.surface_type is types.PASS, (a, b, p.surface_type)
    assert o.rows[o.row_of[c["Bulk"], c["det0"]]].surface_type == types.COUNTING
    eff = o.params["surfaces"]["eff"][o.row_of[c["Bulk"], c["det0"]]]
    assert abs(float(eff) - 0.03) < 1e-7
    # tags by copy number, 0..N-1
    assert [o.sensor_index[f"det{i}"] for i in range(4)] == [3, 2, 1, 0] and o.sensor_pvs[0] == "det3"
    assert o.sensor_entries == ("arapuca",) * 4
    assert o.row_of.dtype == np.int32 and o.medium_of.dtype == np.int32
    # the diagonal: one sensor entered from another (touching siblings) counts; every other class's own pair is a pass
    on_diag = {k for k in range(o.n_classes) if o.row_of[k, k] >= 0}
    assert on_diag == {c["det0"]} and kern(o, "det1", "det2")[0] is types.COUNTING
    assert o.pair("Bulk", "TPC") is None and o.pairs[(c["det0"], c["det0"])].n_pv_pairs == 4 * 3


def test_touching_sensors_of_one_class_cross_on_the_diagonal():
    """Siblings of one class may touch: the crossing from one into the other is the class pair (c, c), its surface the
    one Geant4 applies there (entering an any-entry sensor: the sensor's type), and the navigator stops there
    (``tracked.vertices``' diagonal). No declaration, no geometry: a pair that never touches is never crossed."""
    dets = tuple((f"det{i}", "SensLV", "LAr", "World") for i in range(3))
    t = tree(("World", "WorldLV", "LAr", None), *dets)
    o = compile_(t, sensors=(SensorBinding("arapuca", 3, lv="SensLV"),))
    c = o.class_of
    assert len({c[f"det{i}"] for i in range(3)}) == 1
    k, origin, why = kern(o, "det0", "det1")
    assert (k, origin, why) == (types.COUNTING, "sensor:arapuca", ("sensor",))
    assert o.pairs[(c["det0"], c["det0"])].n_pv_pairs == 6 and kern(o, "World", "det2")[0] is types.COUNTING
    assert np.diag(tracked.vertices(o)).tolist() == [k == c["det0"] for k in range(o.n_classes)]
    plain = compile_(tree(("World", "WorldLV", "LAr", None), *((f"b{i}", "BLV", "LAr", "World") for i in range(3))))
    assert not np.diag(tracked.vertices(plain)).any() and plain.pair("b0", "b1") is None  # a pass: no crossing


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
        assert kern(oo, "World", "TankA")[0] is types.ABSORBER and kern(oo, "World", "TankB")[0] is types.ABSORBER


def test_value_only_changes_leave_partition_and_tables_identical():
    t = tree(*TANK, ("Pmt", "PmtLV", "Acrylic", "World"), ("Can", "CanLV", "Steel", "World"))
    kw = dict(
        skins={"BoxLV": "refl", "Box2LV": "refl2"},
        borders={("World", "Can"): "refl"},
        sensors=(SensorBinding("pmt", 1, pv="Pmt"),),
    )
    a = compile_(t, **kw)
    lib = library(n_water=1.40, R=0.7, R2=0.2, eff=0.5, eff_pmt=0.3)
    b = compile_(t, lib=lib, **kw)
    for f in ("class_names", "signatures", "class_of", "class_pvs", "media_names", "rows", "sensor_index", "report"):
        assert getattr(a, f) == getattr(b, f), f
    assert a.media == b.media
    for f in ("row_of", "medium_of"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f
    refl = [i for i, r in enumerate(a.rows) if r.surface_type == types.REFLECTOR]
    ra, rb = a.params["surfaces"]["R"][refl], b.params["surfaces"]["R"][refl]
    assert ra.shape == rb.shape == (2,) and not np.allclose(ra, rb)
    assert float(b.params["media"]["water"]["index"]["value"]) == pytest.approx(1.40)
    pc = next(i for i, r in enumerate(b.rows) if r.surface_type == types.PHOTOCATHODE)
    assert float(b.params["surfaces"]["eff"][pc]) == pytest.approx(0.3)
    # re-packing params for a value-changed library reproduces them without recompiling
    rp = regions.pack(a, regions.named_params(a, lib))
    assert np.array_equal(np.asarray(rp["surfaces"]["R"]), b.params["surfaces"]["R"])
    # rows gather the right surface
    cw, cb, cb2 = a.class_of["World"], a.class_of["Box"], a.class_of["Box2"]
    assert a.params["surfaces"]["R"][a.row_of[cw, cb]] == pytest.approx(0.9)
    assert a.params["surfaces"]["R"][a.row_of[cw, cb2]] == pytest.approx(0.5)
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
        types.REFLECTOR,
        "surface:refl",
        "border",
    ),
    (
        "border beats pre skin",
        {"WorldLV": "refl2"},
        {("World", "Box"): "refl"},
        "World",
        "Box",
        types.REFLECTOR,
        "surface:refl",
        "border",
    ),
    ("border is directional", {}, {("Box", "World"): "refl"}, "World", "Box", types.DIELECTRIC, None, "default"),
    (
        "border applies to its own order",
        {},
        {("Box", "World"): "refl"},
        "Box",
        "World",
        types.REFLECTOR,
        "surface:refl",
        "border",
    ),
    (
        "entering a daughter: post skin first",
        {"BoxLV": "refl2", "WorldLV": "refl"},
        {},
        "World",
        "Box",
        types.REFLECTOR,
        "surface:refl2",
        "skin(post)",
    ),
    (
        "entering a daughter: then pre skin",
        {"WorldLV": "refl"},
        {},
        "World",
        "Box",
        types.REFLECTOR,
        "surface:refl",
        "skin(pre)",
    ),
    (
        "leaving into the mother: pre skin first",
        {"BoxLV": "refl2", "WorldLV": "refl"},
        {},
        "Box",
        "World",
        types.REFLECTOR,
        "surface:refl2",
        "skin(pre)",
    ),
    (
        "leaving into the mother: then post skin",
        {"WorldLV": "refl"},
        {},
        "Box",
        "World",
        types.REFLECTOR,
        "surface:refl",
        "skin(post)",
    ),
    (
        "sibling: pre skin first",
        {"BoxLV": "refl", "Box2LV": "refl2"},
        {},
        "Box",
        "Box2",
        types.REFLECTOR,
        "surface:refl",
        "skin(pre)",
    ),
    (
        "sibling: pre skin first, reversed",
        {"BoxLV": "refl", "Box2LV": "refl2"},
        {},
        "Box2",
        "Box",
        types.REFLECTOR,
        "surface:refl2",
        "skin(pre)",
    ),
    (
        "sibling: then post skin",
        {"Box2LV": "refl2"},
        {},
        "Box",
        "Box2",
        types.REFLECTOR,
        "surface:refl2",
        "skin(post)",
    ),
    (
        "sibling border beats both skins",
        {"BoxLV": "refl", "Box2LV": "refl"},
        {("Box", "Box2"): "refl2"},
        "Box",
        "Box2",
        types.REFLECTOR,
        "surface:refl2",
        "border",
    ),
    ("default: different indices", {}, {}, "World", "Box", types.DIELECTRIC, None, "default"),
    ("paramless surface kind", {"BoxLV": "black"}, {}, "World", "Box", types.ABSORBER, None, "skin(post)"),
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
    assert kern(o, "A", "B")[0] is types.REFLECTOR  # pre skin (B has none), daughter entry
    o = compile_(t, borders={("World", "A"): "refl"})
    assert kern(o, "A", "B") == (types.PASS, None, ("default",))


def test_sensor_kind_on_entry_beats_the_mother_skin():
    t = tree(("World", "WorldLV", "Water", None), ("Pmt", "PmtLV", "Acrylic", "World"))
    o = compile_(t, skins={"WorldLV": "refl"}, sensors=(SensorBinding("pmt", 1, pv="Pmt"),))
    assert kern(o, "World", "Pmt") == (types.PHOTOCATHODE, "sensor:pmt", ("sensor",))
    assert kern(o, "Pmt", "World")[:2] == (types.REFLECTOR, "surface:refl")  # leaving: ordinary precedence
    with pytest.raises(DescriptionError, match="ambiguous"):
        compile_(t, skins={"PmtLV": "refl"}, sensors=(SensorBinding("pmt", 1, pv="Pmt"),))
    with pytest.raises(DescriptionError, match="ambiguous"):
        compile_(t, borders={("World", "Pmt"): "refl"}, sensors=(SensorBinding("pmt", 1, pv="Pmt"),))


def test_no_rindex_absorbs_and_opaque_media_are_never_left():
    t = tree(
        ("World", "WorldLV", "Water", None), ("Can", "CanLV", "Steel", "World"), ("Can2", "Can2LV", "Steel", "World")
    )
    o = compile_(t)
    assert kern(o, "World", "Can") == (types.ABSORBER, None, ("default: NoRINDEX",))
    assert o.pair("Can", "World") is None
    o = compile_(t, borders={("World", "Can"): "polish_on"})
    assert kern(o, "World", "Can") == (types.ABSORBER, None, ("border: NoRINDEX",))
    o = compile_(t, skins={"CanLV": "refl"})
    assert kern(o, "World", "Can")[:2] == (types.REFLECTOR, "surface:refl")
    assert kern(o, "World", "Can2")[0] is types.ABSORBER
    with pytest.raises(regions.OpticsError, match="opaque"):
        compile_(t, skins={"CanLV": "matched"})


def test_a_frontpainted_transmission_into_an_opaque_medium_is_absorbed_where_it_lands():
    """Geant4's frontpainted finish skips the RINDEX check: T transmits into the steel, killed at the next step."""
    import jax.numpy as jnp
    from surface_reference import ABSORB

    from lumen.surfaces.outcomes import STRAIGHT

    t = tree(("World", "WorldLV", "Water", None), ("Can", "CanLV", "Steel", "World"),
             ("Box", "BoxLV", "Acrylic", "World"))  # fmt: skip
    lib = library()
    surface = types.no_detection(types.PAINTED_T)  # a surface: DoAbsorption's efficiency fixed at 0
    paint = SurfaceEntry(surface, {"R": 0.4, "T": 0.25, "f_spec": 1.0}, far_side=FAR_ABSORB)
    lib = dataclasses.replace(lib, surfaces={**lib.surfaces, "paint": paint})
    o = compile_(t, lib=lib, skins={"CanLV": "paint", "BoxLV": "paint"})
    into = o.pair("World", "Can")
    assert into.surface_type == types.into_opaque(surface) and into.origin == "surface:paint"
    assert "absorbed" in into.reasons[0] and set(into.surface_type.params) == {"R", "T", "f_spec"}
    assert o.pair("World", "Box").surface_type is surface  # into a medium with RINDEX it transmits
    p = {k: jnp.float32(v) for k, v in paint.params.items()}
    pr, base = full(into.surface_type, p, {}), full(surface, p, {})
    assert float(pr[STRAIGHT]) == 0.0 and float(pr[ABSORB]) == pytest.approx(float(base[ABSORB] + base[STRAIGHT]))
    assert float(jnp.sum(pr)) == pytest.approx(1.0) and STRAIGHT not in into.surface_type.outcomes
    assert types.into_opaque(surface) == into.surface_type  # equal types are one type: one row


def test_the_far_side_into_a_medium_without_rindex_follows_the_reading():
    """Geant4 kills a photon at a polished dielectric_dielectric surface whose far side has no RINDEX (NoRINDEX), a
    detecting one before its detection; lumen refused it before. A type with no Geant4 reading is still refused."""
    from lumen.format import schema
    from lumen.rules import optics as rules

    t = tree(("World", "WorldLV", "Water", None), ("Can", "CanLV", "Steel", "World"))
    win = schema.Surface("win", type="dielectric_dielectric", model="unified", finish="polished", value=0.0,
                         properties={"REFLECTIVITY": schema.Property("REFLECTIVITY", np.asarray(0.1))})  # fmt: skip
    e = rules.surface(win)
    assert e.surface_type == types.no_detection(types.DIELECTRIC_RT) and e.far_side == FAR_KILL
    lib = library()
    lib = dataclasses.replace(lib, surfaces={**lib.surfaces, "win": e},
                              sensors={**lib.sensors, "cath": rules.sensor(win, eff=1.0)})  # fmt: skip
    assert kern(compile_(t, lib=lib, skins={"CanLV": "win"}), "World", "Can") == (
        types.ABSORBER, None, ("skin(post): NoRINDEX",))  # fmt: skip
    o = compile_(t, lib=lib, sensors=(SensorBinding("cath", 1, pv="Can"),))
    assert kern(o, "World", "Can") == (types.ABSORBER, None, ("sensor: NoRINDEX",))
    bare = SurfaceEntry(e.surface_type, e.params)  # no far side recorded: refused
    with pytest.raises(regions.OpticsError, match="opaque"):
        compile_(t, lib=dataclasses.replace(lib, surfaces={**lib.surfaces, "bare": bare}), skins={"CanLV": "bare"})


def test_surface_ignored_between_the_same_material():
    t = tree(("World", "WorldLV", "Water", None), ("Box", "BoxLV", "Water", "World"))
    o = compile_(t, skins={"BoxLV": "polish"})
    k, org, why = kern(o, "World", "Box")
    assert k is types.PASS and "same-material" in why[0] and kern(o, "Box", "World")[0] is types.PASS
    o = compile_(t, skins={"BoxLV": "polish_on"})  # not flagged: acts
    assert kern(o, "World", "Box") == (types.DIELECTRIC, None, ("skin(post)",))
    # Geant4 compares materials, not media: a different material bound to the same medium keeps the surface
    t2 = tree(("World", "WorldLV", "Water", None), ("Box", "BoxLV", "Water2", "World"))
    o = compile_(t2, skins={"BoxLV": "polish"})
    assert kern(o, "World", "Box") == (types.DIELECTRIC, None, ("skin(post)",))


# ---------------------------------------------------------------------------------------------------------- guards
def test_self_touching_skinned_placements():
    t = tree(
        ("World", "WorldLV", "Water", None), ("P1", "BoxLV", "Acrylic", "World"), ("P2", "BoxLV", "Acrylic", "World")
    )
    o = compile_(t, skins={"BoxLV": "refl"})  # Geant4: leaving P1 into P2, P1's skin (pre's, not entering a daughter)
    assert o.class_of["P1"] == o.class_of["P2"] and kern(o, "World", "P1")[0] is types.REFLECTOR
    assert kern(o, "P1", "P2") == (types.REFLECTOR, "surface:refl", ("skin(pre)",))
    o = compile_(t)  # unskinned: a pass inside one class, no crossing
    assert o.class_of["P1"] == o.class_of["P2"] and o.pair("P1", "P2") is None


def test_border_on_one_placement_splits_only_that_placement():
    t = tree(("World", "WorldLV", "Water", None), *((f"P{i}", "BoxLV", "Acrylic", "World") for i in (1, 2, 3)))
    o = compile_(t, borders={("P1", "World"): "refl"})
    c = o.class_of
    assert c["P2"] == c["P3"] != c["P1"] and c["World"] not in (c["P1"], c["P2"])
    assert kern(o, "P1", "World") == (types.REFLECTOR, "surface:refl", ("border",))
    assert kern(o, "P2", "World") == (types.DIELECTRIC, None, ("default",))
    assert kern(o, "World", "P1") == (types.DIELECTRIC, None, ("default",))
    assert kern(o, "P1", "P2") == (types.PASS, None, ("default",))
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
        compile_(t, skins={"MLV": "refl", "DLV": "refl2", "D2LV": "refl2"})


def test_border_sensor_is_credited_to_the_sensor_pv_it_leaves():
    """WCSim's AirCathodeSurface: a detection leaving the sensor PV (into an untagged one) counts on that sensor."""
    t = tree(("World", "WorldLV", "Water", None), ("Pmt", "PmtLV", "Acrylic", "World"))
    sb = (SensorBinding("pmt", 1, pv="Pmt"),)
    plain = compile_(t, sensors=sb)
    assert plain.deposit_pre is None
    o = compile_(t, borders={("Pmt", "World"): "arapuca"}, sensors=sb)
    c = o.class_of
    k, origin, why = kern(o, "Pmt", "World")
    assert (k, origin) == (types.COUNTING, "sensor:arapuca") and why == ("border sensor arapuca (credited to pre)",)
    assert kern(o, "World", "Pmt")[:2] == (types.PHOTOCATHODE, "sensor:pmt")
    assert o.deposit_pre.dtype == bool and o.deposit_pre.shape == (o.n_classes,) * 2
    assert o.deposit_pre[c["Pmt"], c["World"]] and o.deposit_pre.sum() == 1  # entering still credits post
    with pytest.raises(DescriptionError, match="not a sensor PV"):  # nothing to credit
        compile_(t, borders={("Pmt", "World"): "arapuca"})
    both = library()
    both.surfaces["arapuca"] = SurfaceEntry(types.ABSORBER, {})
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
    with pytest.raises(DescriptionError, match="unknown physvol"):
        compile_(t, borders={("Box", "Nope"): "refl"})
    bad = library()
    bad.surfaces["half"] = SurfaceEntry(types.REFLECTOR, {"R": 0.9})
    with pytest.raises(DescriptionError, match="missing"):
        compile_(t, lib=bad)
    bad = library()
    bad.surfaces["cathode"] = SurfaceEntry(types.COUNTING, {"eff": 0.1})
    with pytest.raises(DescriptionError, match="sensor binding"):
        compile_(t, lib=bad)
    with pytest.raises(TreeError):
        tree(("World", "WorldLV", "Water", None), ("A", "ALV", "Water", "World"), ("B", "ALV", "Acrylic", "World"))
    with pytest.raises(TreeError):
        tree(("World", "WorldLV", "Water", None), ("W2", "WLV", "Water", None))


def test_sensor_count_and_binding_errors():
    t = tree(("World", "WorldLV", "Water", None), *((f"pmt{i}", "PmtLV", "Acrylic", "World") for i in range(3)))
    o = compile_(t, sensors=(SensorBinding("pmt", 3, lv="PmtLV"),))
    assert sorted(o.sensor_index.values()) == [0, 1, 2]
    with pytest.raises(DescriptionError, match="expected 4 instances, found 3"):
        compile_(t, sensors=(SensorBinding("pmt", 4, lv="PmtLV"),))
    with pytest.raises(DescriptionError, match="two sensors"):
        compile_(t, sensors=(SensorBinding("pmt", 3, lv="PmtLV"), SensorBinding("arapuca", 1, pv="pmt1")))
    with pytest.raises(DescriptionError, match="not a library sensor"):
        compile_(t, sensors=(SensorBinding("sipm", 3, lv="PmtLV"),))
    with pytest.raises(DescriptionError, match="exactly one"):
        compile_(t, sensors=(SensorBinding("pmt", 3),))


def test_laws_are_validated_at_their_knots():
    lib = library()
    lib.media["water"] = MediumEntry(WATER, {"index": {"value": 1.33}, "absorption": {"value": -1.0}})
    with pytest.raises(laws.LawError):
        compile_(tree(*TANK), lib=lib)
    tab = bulk.Medium("water_tab", laws.tabulated("index"))
    lib = library()
    lib.media["water"] = MediumEntry(tab, {"index": {"energy": np.array([2.0, 3.0]), "value": np.array([1.33, 0.9])}})
    with pytest.raises(laws.LawError, match="out of range"):  # an index below 1 at its 413 nm knot
        compile_(tree(*TANK), lib=lib)
    lib.media["water"] = MediumEntry(tab, {"index": {"energy": np.array([2.0, 3.0]), "value": np.array([1.33, 1.34])}})
    compile_(tree(*TANK), lib=lib)  # a table is held at its end values: no wavelength is outside it
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
        "sensor_index",
        "sensor_pvs",
        "report",
    ):
        assert getattr(a, f) == getattr(b, f), f
    assert a.media == b.media
    for f in ("row_of", "medium_of"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f
    for x, y in zip(jax.tree_util.tree_leaves(a.params["surfaces"]), jax.tree_util.tree_leaves(b.params["surfaces"]),
                    strict=True):  # fmt: skip
        assert np.array_equal(x, y)


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
    _same(regions.compile(lib, t, b), regions.compile(lib, t, b))
    assert b.skins == kw["skins"] and len(lib.surfaces) == len(library().surfaces)


def test_report_explains_every_pair():
    specs, kw = _rich()
    o = compile_(tree(*specs), **kw)
    r = o.report
    for word in ("border", "skin(post)", "skin(pre)", "sensor", "default", "NoRINDEX", "pmt: 5 instances"):
        assert word in r, word
    assert all(o.class_names[c] in r for c in range(o.n_classes))
    assert o.sensor_index == {f"pmt{i}": t for t, i in enumerate(sorted(range(5), key=lambda i: ((7 * i) % 5, i)))}


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
    assert kern(o, "Glass", "Body")[0] is types.PASS
    assert kern(o, "Body", "Inner")[0].name == "photocathode"
    k, origin, reasons = kern(o, "Glass", "Inner")
    assert k.name == "photocathode" and origin == "sensor:pmt", (k, origin, reasons)
    assert any(r.startswith("pass via") for r in reasons), reasons
    assert o.sensor_index["Inner"] == 0 and not (o.deposit_pre is not None and o.deposit_pre[c["Glass"], c["Inner"]])


def test_a_border_between_two_physvols_of_many_instances_is_never_expanded_into_instance_pairs():
    """n instances of physvol a and n of physvol b, siblings under one mother, a border a -> b: the bindings hold one
    physvol pair, and the compile counts the n^2 crossings from the instances per class (Geant4's border is the
    physvol pair, every touching instance pair of it) without enumerating them; b -> a and a -> a stay unbordered."""
    n = 3000
    pvs = [PV("World", "WorldLV", "Water", None)]
    pvs += [PV(f"W/a#{k}/A", "ALV", "Water", "World", physvol="a") for k in range(n)]
    pvs += [PV(f"W/b#{k}/B", "BLV", "Water", "World", physvol="b") for k in range(n)]
    o = compile_(Tree(tuple(pvs)), borders={("a", "b"): "refl"})
    a, b = "W/a#0/A", "W/b#7/B"
    assert kern(o, a, b) == (types.REFLECTOR, "surface:refl", ("border",))
    assert kern(o, b, a) == (types.PASS, None, ("default",)) and kern(o, a, "W/a#5/A") is None
    assert o.pair(a, b).n_pv_pairs == n * n and o.pair(b, a).n_pv_pairs == n * n
    assert o.class_of[a] != o.class_of["World"] != o.class_of[b] != o.class_of[a]
