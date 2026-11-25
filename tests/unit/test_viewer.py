"""The GDML viewer (``lumen.viewer``): its meshes against the solids lumen traces, its scene against the detector lumen
builds (sensors, crossings, refusals), and its server."""

import base64
import gzip
import http.client
import json
import math
import re
import threading
from pathlib import Path

import numpy as np
import pytest

from lumen.build import build, lowering
from lumen.format import schema
from lumen.format.sensorsets import SensorGroup
from lumen.geometry import solids as so
from lumen.recipe import Recipe
from lumen.viewer import mesh, scene, server

ROOT = Path(__file__).resolve().parents[2]
TANK = ROOT / "examples" / "data" / "example_tank.gdml"
RECIPE = f"{ROOT / 'examples' / 'example_tank.py'}:RECIPE"
TWOPI = 2 * math.pi
FULL = {"startphi": 0.0, "deltaphi": TWOPI}
HERE = schema.Transform(position=(0, 0, 0), rotation=(0, 0, 0))


# ----------------------------------------------------------------------------------------------------- the meshes
def solid(t, name="s", **p):
    given = {k: p.pop(k) for k in ("planes", "vertices", "sections", "refs", "transforms", "scale") if k in p}
    return schema.Solid(name, t, params=p, **given)


def detector(*solids):
    return schema.DetectorFile(name="t", world="w", detector_volume="w", volumes={}, placements=(), materials={},
                               solids={s.name: s for s in solids})  # fmt: skip


def volume(m):
    a, b, c = np.moveaxis(m.vertices.astype(float)[m.triangles], 1, 0)
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0


def traced_volume(df, name="s", n=200_000):
    node = lowering.SolidMapper(df).solid(name)
    lo, hi = so.node_bbox(node)
    x = np.random.default_rng(1).uniform(lo, hi, (n, 3))
    return so.Compiled(node).inside_np(x).mean() * np.prod(hi - lo)


def arb8(*xy, dz=1.0):
    return solid("arb8", dz=dz, **{f"v{i}{k}": v for i, p in enumerate(xy, 1) for k, v in zip("xy", p, strict=True)})


SQUARE = ((-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0))
SOLIDS = {
    "box": solid("box", x=1.0, y=2.0, z=3.0),
    "thin box": solid("box", x=3e-5, y=2.0, z=3.0),
    "tube": solid("tube", rmin=0.5, rmax=1.0, z=2.0, startphi=0.3, deltaphi=4.5),
    "cut tube": solid("cutTube", rmin=0.2, rmax=0.5, z=2.0, lowX=0.0, lowY=-0.6, lowZ=-0.8, highX=0.6, highY=0.0,
                      highZ=0.8, **FULL),
    "cone": solid("cone", rmin1=0.2, rmax1=1.0, rmin2=0.1, rmax2=0.5, z=1.0, **FULL),
    "polycone": solid("polycone", planes=((0.0, 1.0, -1.0), (0.2, 0.5, 0.0), (0.2, 0.9, 1.0)), startphi=0.1,
                      deltaphi=5.0),
    "polyhedra": solid("polyhedra", planes=((0.2, 1.0, -1.0), (0.2, 0.5, 1.0)), numsides=6, **FULL),
    "sphere": solid("sphere", rmin=0.3, rmax=1.0, startphi=0.1, deltaphi=2.0, starttheta=0.4, deltatheta=2.0),
    "orb": solid("orb", r=1.0),
    "torus": solid("torus", rmin=0.1, rmax=0.2, rtor=1.0, startphi=0.2, deltaphi=3.0),
    "eltube": solid("eltube", dx=0.5, dy=0.2, dz=1.0),
    "elcone cut past its apex": solid("elcone", dx=0.5, dy=0.4, zmax=1.0, zcut=1.5),
    "ellipsoid": solid("ellipsoid", ax=1.0, by=0.8, cz=0.5, zcut1=0.0, zcut2=0.0),
    "ellipsoid cut at 0": solid("ellipsoid", ax=1.0, by=1.0, cz=1.0, zcut1=0.0, zcut2=0.5),
    "ellipsoid cut": solid("ellipsoid", ax=1.0, by=1.0, cz=1.0, zcut1=-0.5, zcut2=0.0),
    "paraboloid": solid("paraboloid", rlo=0.1, rhi=1.0, dz=0.5),
    "hype": solid("hype", rmin=0.2, rmax=0.5, inst=0.3, outst=0.6, z=2.0),
    "trd": solid("trd", x1=2.0, x2=1.0, y1=2.0, y2=1.0, z=1.0),
    "para": solid("para", x=1.0, y=1.0, z=1.0, alpha=0.2, theta=0.3, phi=0.4),
    "trap": solid("trap", z=1.0, theta=0.2, phi=0.3, y1=1.0, x1=1.0, x2=0.8, alpha1=0.1, y2=0.6, x3=0.6, x4=0.48,
                  alpha2=0.1),
    "arb8": arb8(*SQUARE, *((0.5 * x, 0.5 * y) for x, y in SQUARE)),
    "tet": solid("tet", vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))),
    "xtru": solid("xtru", vertices=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 0.3), (0.0, 1.0)),
                  sections=((-1.0, 0.0, 0.0, 1.0), (1.0, 0.2, 0.0, 0.5))),
}  # fmt: skip


@pytest.mark.parametrize("name", SOLIDS)
def test_a_solid_is_drawn_as_lumen_traces_it(name):
    df = detector(SOLIDS[name])
    m = mesh.Tessellator(df).solid("s")
    assert m.exact and m.note == "" and m.refused == ""
    if name != "thin box":  # its Monte Carlo volume is exact too, its relative error float32's
        assert volume(m) == pytest.approx(traced_volume(df), rel=2e-2)


def test_booleans_are_drawn_as_lumen_traces_them():
    shift = schema.Transform(position=(0.4, 0.1, 0.0), rotation=(0.3, 0.2, 0.1))
    ops = [solid(op, op, refs=("b", "t"), transforms=(HERE, shift))
           for op in ("union", "subtraction", "intersection", "multiUnion")]  # fmt: skip
    df = detector(solid("box", "b", x=2.0, y=2.0, z=2.0), solid("tube", "t", rmin=0.0, rmax=0.5, z=4.0, **FULL),
                  solid("scaledSolid", "scaled", refs=("subtraction",), scale=(1.0, 2.0, 0.5)), *ops)  # fmt: skip
    for name in (*(s.name for s in ops), "scaled"):
        m = mesh.Tessellator(df).solid(name)
        assert m.exact, (name, m.note)
        assert volume(m) == pytest.approx(traced_volume(df, name), rel=2e-2)


WRONG = {  # one way a drawing can be wrong each: too big, too small, a wall lost
    "elcone cut past its apex": lambda s, g, seg: mesh._scale_xy(
        mesh.lathe([mesh._orient([(0, -1.5), (2.5, -1.5), (0, 1.0)])], seg=seg), 0.5, 0.4),
    "tube": lambda s, g, seg: mesh.lathe(mesh._ring(0.5, 1.0, 1.0), 0.3, 4.4, seg=seg),
    "cut tube": lambda s, g, seg: mesh.lathe(mesh._ring(0.2, 0.5, 1.0), seg=seg),
    "sphere": lambda s, g, seg: mesh.lathe(mesh._sphere_loop(0.0, 1.0, 0.4, 2.0, seg), 0.1, 2.0, seg=seg),
}  # fmt: skip


@pytest.mark.parametrize("name", WRONG)
def test_a_drawing_that_differs_from_lumen_s_solid_is_marked(name, monkeypatch):
    monkeypatch.setitem(mesh.PRIMITIVES, SOLIDS[name].type, WRONG[name])
    m = mesh.Tessellator(detector(SOLIDS[name])).solid("s")
    assert not m.exact and "not matched" in m.note


@pytest.mark.parametrize("s", [
    solid("genericPolycone", planes=((1.0, -1.0), (1.0, 1.0), (0.0, 1.0), (0.0, -1.0)), **FULL),
    solid("genericPolyhedra", planes=((1.0, -1.0), (1.0, 1.0), (0.0, 1.0), (0.0, -1.0)), numsides=6, **FULL),
    arb8(*SQUARE, *SQUARE[:3], (1.5, -1.0)),  # its faces 2 and 4 are twisted
], ids=lambda s: s.type)  # fmt: skip
def test_a_solid_lumen_does_not_trace_says_so(s):
    m = mesh.Tessellator(detector(s)).solid("s")
    assert not m.exact and "lumen does not trace this solid" in m.refused
    if s.type == "genericPolyhedra":  # its points are corners (G4Polyhedra): a hexagon of circumradius 1, length 2
        assert volume(m) == pytest.approx(3 * math.sqrt(3), rel=1e-6)


def test_repeated_solids_get_fewer_faces():
    assert mesh.segments_for(1) > mesh.segments_for(10_000)


# ------------------------------------------------------------------------------------------------------ the scene
EXTRA = {  # the tutorial tank with what a page could misread
    "<solids>": """<solids>
  <trd name="WedgeTrd" x1="100" x2="50" y1="100" y2="100" z="60" lunit="mm"/>
  <opticalsurface name="BadSurface" model="LUT" finish="polishedlumirrorair" type="dielectric_LUT" value="0"/>""",
    '<property name="ABSLENGTH" ref="WATER_ABSLENGTH"/>': """<property name="ABSLENGTH" ref="WATER_ABSLENGTH"/>
    <property name="SCINTILLATIONCOMPONENT1" ref="WATER_ABSLENGTH"/>
    <property name="SCINTILLATIONYIELD" ref="TEN"/><property name="SCINTILLATIONTIMECONSTANT1" ref="TEN"/>
    <property name="RESOLUTIONSCALE" ref="TEN"/><property name="PROTONSCINTILLATIONYIELD" ref="WATER_ABSLENGTH"/>""",
    "</define>": '<matrix name="TEN" coldim="1" values="10"/></define>',
    '<volume name="TankLV">': """<volume name="Wedge">
    <materialref ref="Photocathode"/><solidref ref="WedgeTrd"/></volume>
  <volume name="Shelf"><materialref ref="Photocathode"/><solidref ref="WedgeTrd"/></volume>
  <volume name="TankLV">""",
    '<physvol name="PMT_0" copynumber="0">': """<physvol name="Wedge_0"><volumeref ref="Wedge"/>
      <position name="w0" x="200" y="150" z="0" unit="mm"/><scale name="w0s" x="1" y="1" z="-1"/></physvol>
    <physvol name="PMT_0" copynumber="0">""",
    '<physvol name="Tank">': """<physvol name="Shelf_0"><volumeref ref="Shelf"/>
      <position name="s0" x="700" y="0" z="0" unit="mm"/></physvol>
    <physvol name="Tank">""",
    "</structure>": """<skinsurface name="wedge_skin" surfaceproperty="CathodeSurface">
    <volumeref ref="Wedge"/></skinsurface>
  <skinsurface name="shelf_skin" surfaceproperty="BadSurface"><volumeref ref="Shelf"/></skinsurface>
</structure>""",
}


def edited(path, changes):
    text = TANK.read_text()
    for old, new in changes.items():
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    path.write_text(text)
    return path


def short_of(payload, kind, name):
    items = payload[kind] if isinstance(payload[kind], list) else payload[kind].values()
    return next(x for x in items if x["short"] == name)


def decoded(payload, key, dtype):
    return np.frombuffer(base64.b64decode(payload["instances"][key]), dtype)


@pytest.fixture(scope="module")
def tank():
    return scene.build(TANK, recipe=RECIPE)


@pytest.fixture(scope="module")
def extra(tmp_path_factory):
    """Return (the scene, the detector it was built from): two groups share the PMT volume, 2 and 6 of its sensors."""
    path = edited(tmp_path_factory.mktemp("viewer") / "extra.gdml", EXTRA)
    base = scene.load_recipe(RECIPE)
    groups = tuple(SensorGroup(n, rule="border_into", lv="PMT", where={"type": n}, values=base.groups[0].values,
                               readout=base.groups[0].readout) for n in ("few", "many"))  # fmt: skip
    rec = Recipe("extra", "TankLV", groups=groups, sensor_columns=lambda df, paths, inputs: {
        "type": ["few" if "PMT_0" in p or "PMT_6" in p else "many" for p in paths]})  # fmt: skip
    kept = []
    load = scene.api.load
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(scene.api, "load", lambda out: kept.append(load(out)) or kept[0])
        view = scene.build(path, recipe=rec)
    return view.payload, kept[0]


def test_tank_scene_as_lumen_reads_it(tank):
    """The tutorial tank with its recipe: the detector volume, the PMT sensors, Water read as a medium."""
    p = tank.payload
    assert tank.errors == []
    assert p["meta"]["detector_volume"] == "TankLV" and not p["meta"]["detector_guessed"] and p["meta"]["built"]
    water = short_of(p, "materials", "Water")
    assert water["status"] == "medium" and "absorption" in water["channels"]
    rindex = next(q for q in water["properties"] if q["key"] == "RINDEX")
    assert any(r["param"].startswith("index") for r in rindex["reads"])
    assert short_of(p, "materials", "Air")["status"] == "unused"  # only the world, outside the detector volume
    assert short_of(p, "volumes", "PMT")["sensors"] == {"pmt": 8} and p["sensor_ids"] == 8
    assert p["surfaces"]["CathodeSurface"]["status"] == "sensor"
    assert len(decoded(p, "matrix", np.float32)) == 12 * p["instances"]["n"]
    json.loads(server.encode(p))  # NaN-free JSON the browser parses


def test_groups_sharing_a_volume_hold_their_own_sensors(extra):
    """The groups' counts and each copy's group are the built detector's (``build.group_tags``), ``where`` included."""
    p, det = extra
    tags = {g.name: len(build.group_tags(g, det.parts, det.sensor_set)) for g in det.sensor_set.groups}
    assert {g["name"]: g["sensors"] for g in p["sensor_groups"]} == tags == {"few": 2, "many": 6}
    assert short_of(p, "volumes", "PMT")["sensors"] == tags
    placed, group = decoded(p, "placement", np.int32), decoded(p, "group", np.int32)
    few = {p["placements"][i]["name"] for i in placed[group == 0]}
    assert few == {"PMT_0", "PMT_6"} and np.count_nonzero(group == 1) == 6


def test_crossings_are_the_built_optics(extra):
    """One surface is a sensor's where a group reads it and a plain surface elsewhere: told apart per crossing."""
    p, det = extra
    for v in p["volumes"]:
        if not v["crossings"]:
            continue
        mother = det.parts.tree.by_name[v["first"]].mother
        for c, (a, b) in zip(v["crossings"], ((mother, v["first"]), (v["first"], mother)), strict=False):
            pair = det.parts.optics.pair(a, b)
            assert (c["outcome"] == "none") == (pair is None)
            assert (c["outcome"] == "sensor") == (pair is not None and pair.surface_type.detects)
            assert pair is None or pair.surface_type.name in c["text"]
    into = {v["short"]: v["crossings"][0] for v in p["volumes"] if v["crossings"]}
    assert into["PMT"]["outcome"] == "sensor" and into["Wedge"]["outcome"] == "surface"
    assert into["PMT"]["surface"] == into["Wedge"]["surface"] == "CathodeSurface"
    assert {u["how"] for u in p["surfaces"]["CathodeSurface"]["uses"]} == {"surface", "sensor"}


def test_what_the_detector_volume_does_not_hold_is_not_refused(extra):
    """A finish lumen refuses, on a volume outside the detector volume: lumen never reads it."""
    p, _ = extra
    assert p["errors"] == [] and p["meta"]["built"]
    assert p["surfaces"]["BadSurface"]["status"] == "unused" and p["surfaces"]["BadSurface"]["message"] == ""
    assert short_of(p, "volumes", "Shelf")["crossings"] is None


def test_the_scintillation_source_reads_its_own_keys_only(extra):
    p, _ = extra
    read = {q["key"]: bool(q["reads"]) for q in short_of(p, "materials", "Water")["properties"]}
    assert read["SCINTILLATIONYIELD"] and read["SCINTILLATIONCOMPONENT1"] and read["SCINTILLATIONTIMECONSTANT1"]
    assert not read["RESOLUTIONSCALE"] and not read["PROTONSCINTILLATIONYIELD"]


def test_without_a_recipe_the_detector_is_guessed():
    p = scene.build(TANK, up="y").payload
    assert p["meta"]["recipe"] is None and p["meta"]["detector_guessed"] and p["meta"]["up"] == "y"
    assert p["meta"]["detector_volume"] == "TankLV"  # the water: the largest volume of a transparent material
    assert p["sensor_groups"] == [] and p["surfaces"]["CathodeSurface"]["status"] == "ok"


@pytest.mark.parametrize("change", [
    {'model="unified" finish="polished" type="dielectric_metal"':
        'model="LUT" finish="polishedlumirrorair" type="dielectric_LUT"'},
    {'<tube name="PMTDisc" rmin="0" rmax="50" z="20" startphi="0" deltaphi="360" aunit="deg" lunit="mm"/>':
        '<genericPolycone name="PMTDisc" startphi="0" deltaphi="360" aunit="deg" lunit="mm"><rzpoint r="0" z="-10"/>'
        '<rzpoint r="50" z="-10"/><rzpoint r="50" z="10"/><rzpoint r="0" z="10"/></genericPolycone>'},
], ids=["surface", "solid"])  # fmt: skip
@pytest.mark.parametrize("recipe", [RECIPE, None], ids=["recipe", "no recipe"])
def test_what_lumen_refuses_is_an_error_and_its_geometry_is_drawn(change, recipe, tmp_path):
    view = scene.build(edited(tmp_path / "refused.gdml", change), recipe=recipe)
    p = view.payload
    assert view.errors and not p["meta"]["built"] and p["instances"]["n"] == 10
    assert all(v["crossings"] is None and v["sensors"] == {} for v in p["volumes"])
    assert all(s["status"] in ("refused", "unread") and not s["uses"] for s in p["surfaces"].values())
    disc = p["meshes"][short_of(p, "volumes", "PMT")["mesh"]]
    assert bool(disc["refused"]) == ("genericPolycone" in str(change))
    if recipe is None and "LUT" in str(change):
        assert p["surfaces"]["CathodeSurface"]["status"] == "refused"
    json.loads(server.encode(p))


def test_a_refused_detector_s_unused_surface_is_not_refused(tmp_path):
    """Of a detector lumen refuses, a surface outside the detector volume is no reason: only the PMTs' is."""
    lut = 'model="LUT" finish="polishedlumirrorair" type="dielectric_LUT"'
    path = edited(tmp_path / "r.gdml", {**EXTRA, 'model="unified" finish="polished" type="dielectric_metal"': lut})
    p = scene.build(path).payload
    assert not p["meta"]["built"]
    assert p["surfaces"]["CathodeSurface"]["status"] == "refused" and p["surfaces"]["BadSurface"]["status"] == "unread"


def test_a_refused_import_still_shows_the_geometry(tmp_path):
    """A GDML the importer refuses (an opticalsurface without its finish): nothing of its optics is claimed."""
    view = scene.build(edited(tmp_path / "bad.gdml", {'finish="polished" ': ""}))
    p = view.payload
    assert view.errors[0][0] == "import" and p["instances"]["n"] == 10
    assert {m["status"] for m in p["materials"].values()} == {"unread"}


def test_a_recipe_file_may_define_a_dataclass(tmp_path):
    f = tmp_path / "mine.py"
    f.write_text("from __future__ import annotations\nimport dataclasses\n\n@dataclasses.dataclass\nclass A:\n"
                 "    x: dataclasses.KW_ONLY\n    y: int = 0\n\nRECIPE = A()\n")  # fmt: skip
    assert scene.load_recipe(f"{f}:RECIPE").y == 0


def test_a_wavelength_range_marks_the_tables_that_do_not_cover_it(tmp_path):
    p = scene.build(TANK, wavelength_range=(200.0, 600.0)).payload
    tables = [q for m in p["materials"].values() for q in m["properties"] if q["kind"] == "table"]
    assert tables and all(q["covers"] is False for q in tables)  # the knots start at 248 nm


def test_a_gdml_geant4_refuses_reaches_the_page_with_its_repair():
    """The demo barrel's GDML (a matrix Geant4 refuses): no geometry, the error in the page's form, the repair named."""
    view = scene.build(ROOT / "examples" / "data" / "demo_barrel.gdml")
    p = view.payload
    assert view.errors and "instances" not in p
    assert p["errors"][0]["stage"] == "import" and "lumen.gdml.repair" in p["errors"][0]["text"]
    json.loads(server.encode(p))


# ----------------------------------------------------------------------------------------------------- the server
@pytest.fixture
def served():
    """Serve a scene of ``{}`` on a free port; yield ask(path, method) -> (status, body, headers)."""
    httpd = server.bind("127.0.0.1", 0)
    threading.Thread(target=server.serve, args=(httpd, b"{}"), daemon=True).start()

    def ask(path, method="GET", **headers):
        c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
        c.request(method, path, headers=headers)
        r = c.getresponse()
        return r.status, r.read(), r.headers

    yield ask
    httpd.shutdown()
    httpd.server_close()


def test_server_sends_the_page_and_the_gzipped_scene(served):
    status, page, _ = served("/")
    assert status == 200
    for name in re.findall(r'(?:src|href)="([^"]+)"', page.decode()):  # everything the page loads comes from here
        assert served("/" + name)[0] == 200, name
    assert served("/scene.json")[1] == b"{}"
    status, body, headers = served("/scene.json", **{"Accept-Encoding": "gzip"})
    assert headers["Content-Encoding"] == "gzip" and gzip.decompress(body) == b"{}"


def test_the_server_sends_no_file_of_the_working_directory(served):
    """Nothing but the page and the scene, by any method: http.server's own handler answers HEAD from the cwd."""
    for method in ("GET", "HEAD", "POST"):
        for path in ("/pyproject.toml", "/../pyproject.toml", "/"):
            assert served(path, method)[0] != 200 or method == "GET" and path == "/", (method, path)


def test_a_port_in_use_is_reported_before_the_scene_is_built(monkeypatch, capsys):
    taken = server.bind("127.0.0.1", 0)
    try:
        monkeypatch.setattr(scene, "build", lambda *a, **k: pytest.fail("the scene was built before the port"))
        assert server.main(["tank.gdml", "--port", str(taken.server_address[1])]) == 2
        assert "cannot listen on" in capsys.readouterr().err
    finally:
        taken.server_close()


def test_the_command_line_parses():
    a = server.parser().parse_args(["lumen_wc:sk", "--port", "9000", "--open", "--up", "y", "--range", "300", "600"])
    assert (a.source, a.port, a.open, a.up, a.range) == ("lumen_wc:sk", 9000, True, "y", [300.0, 600.0])


def test_three_js_ships_with_its_notice():
    static = server.STATIC
    assert "Copyright 2010-2021 Three.js Authors" in (static / "three.min.js").read_text()[:200]
    assert "The MIT License" in (static / "three.LICENSE").read_text()
