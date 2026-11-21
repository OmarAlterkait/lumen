"""The GDML viewer (``lumen.viewer``): display meshes of the solids, and the tutorial tank's scene as lumen reads it."""

import gzip
import json
import math
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from lumen.format import schema
from lumen.viewer import mesh, scene, server

ROOT = Path(__file__).resolve().parents[2]
TANK = ROOT / "examples" / "data" / "example_tank.gdml"
RECIPE = f"{ROOT / 'examples' / 'example_tank.py'}:RECIPE"
manifold = pytest.importorskip("manifold3d")  # the view extra: exact booleans and the volume checks below
TWOPI = 2 * math.pi
IDENTITY = schema.Transform(position=(0, 0, 0), rotation=(0, 0, 0))


def solid(t, **p):
    return schema.Solid("s", t, params=p)


def volume(m):
    v, t = mesh._arrays(m)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c)))) / 6.0


@pytest.mark.parametrize(("s", "exact", "rtol"), [
    (solid("box", x=1.0, y=2.0, z=3.0), 6.0, 1e-6),  # float32 meshes
    (solid("tube", rmin=0.5, rmax=1.0, z=2.0, startphi=0.3, deltaphi=math.pi / 2), 0.75 * math.pi / 2, 4e-3),
    (solid("cone", rmin1=0, rmax1=1, rmin2=0, rmax2=0.5, z=1, startphi=0, deltaphi=TWOPI), math.pi * 1.75 / 3, 4e-3),
    (solid("sphere", rmin=0, rmax=1, startphi=0, deltaphi=TWOPI, starttheta=0, deltatheta=math.pi), 4.18879, 1e-2),
    (solid("trd", x1=2, x2=1, y1=2, y2=1, z=1), 7 / 3, 1e-6),
    (solid("para", x=1, y=1, z=1, alpha=0.2, theta=0.3, phi=0.4), 1.0, 1e-6),
    (solid("torus", rmin=0.1, rmax=0.2, rtor=1, startphi=0, deltaphi=TWOPI), 2 * math.pi**2 * 0.03, 1e-2),
])  # fmt: skip
def test_primitive_volumes(s, exact, rtol):
    """A closed, outward mesh: its volume is the solid's, to the error of its flat faces."""
    assert volume(mesh.primitive(s, None)) == pytest.approx(exact, rel=rtol)


def test_polyhedra_radii_are_apothems():
    """G4Polyhedra's rmax is the distance to the sides: a hexagon of apothem 1 and length 2 has volume 4 sqrt(3)."""
    p = {"startphi": 0.0, "deltaphi": TWOPI, "numsides": 6}
    s = schema.Solid("p", "polyhedra", params=p, planes=((0, 1, -1), (0, 1, 1)))
    assert volume(mesh.primitive(s, None)) == pytest.approx(4 * math.sqrt(3), rel=1e-6)


def test_boolean_subtraction_is_exact():
    """A box with a tube bored through it: the boolean is evaluated on the meshes, not drawn as its first operand."""
    df = schema.DetectorFile(name="t", world="w", detector_volume="w", volumes={}, placements=(), materials={}, solids={
        "b": solid("box", x=2.0, y=2.0, z=2.0),
        "t": solid("tube", rmin=0.0, rmax=0.5, z=4.0, startphi=0.0, deltaphi=2 * math.pi),
        "d": schema.Solid("d", "subtraction", refs=("b", "t"), transforms=(IDENTITY,) * 2),
    })  # fmt: skip
    m = mesh.Tessellator(df).solid("d")
    v = volume((m.vertices.astype(float), m.triangles.astype(np.int64)))
    assert m.exact
    assert v == pytest.approx(8 - math.pi * 0.25 * 2, rel=5e-3)


def test_repeated_solids_get_fewer_faces():
    assert mesh.segments_for(1) > mesh.segments_for(10_000)


def test_tank_scene_as_lumen_reads_it():
    """The tutorial tank with its recipe: the detector volume, the PMT sensors, Water read as a medium."""
    view = scene.build(TANK, recipe=RECIPE)
    p = view.payload
    assert view.errors == []
    assert p["meta"]["detector_volume"] == "TankLV" and not p["meta"]["detector_guessed"]
    assert p["meta"]["up"] == "z"
    water = next(m for m in p["materials"].values() if m["short"] == "Water")
    assert water["status"] == "medium" and "absorption" in water["channels"]
    rindex = next(q for q in water["properties"] if q["key"] == "RINDEX")
    assert any(r["param"].startswith("index") for r in rindex["reads"])
    pmt = next(v for v in p["volumes"] if v["short"] == "PMT")
    assert pmt["sensor"] == "pmt" and pmt["inside"] == 8
    assert next(iter(p["surfaces"].values()))["status"] == "sensor"
    n = p["instances"]["n"]
    assert len(np.frombuffer(__import__("base64").b64decode(p["instances"]["matrix"]), np.float32)) == 12 * n
    json.loads(server.encode(p))  # NaN-free JSON the browser parses


def test_without_a_recipe_the_detector_is_guessed():
    p = scene.build(TANK, up="y").payload
    assert p["meta"]["recipe"] is None and p["meta"]["detector_guessed"] and p["meta"]["up"] == "y"
    assert p["meta"]["detector_volume"] == "TankLV"  # the water: the largest volume of a transparent material
    assert p["sensor_groups"] == []


def test_a_refused_import_still_shows_the_geometry(tmp_path):
    """A GDML the importer refuses (an opticalsurface without its finish): the error is reported, the volumes drawn."""
    text = TANK.read_text()
    bad = tmp_path / "bad.gdml"
    bad.write_text(text.replace('finish="polished" ', "", 1).replace("finish='polished' ", "", 1))
    if bad.read_text() == text:
        pytest.skip("the tutorial GDML's surface spells its finish differently")
    view = scene.build(bad)
    assert view.errors and view.errors[0][0] == "import"
    assert view.payload["volumes"] and view.payload["instances"]["n"] > 1


def test_server_sends_the_page_and_the_gzipped_scene():
    body = server.encode(scene.build(TANK).payload)
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.handler(gzip.compress(body)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}/"
    try:
        assert b"viewer.js" in urllib.request.urlopen(base).read()
        req = urllib.request.Request(base + "scene.json", headers={"Accept-Encoding": "gzip"})
        r = urllib.request.urlopen(req)
        assert r.headers["Content-Encoding"] == "gzip" and gzip.decompress(r.read()) == body
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "../pyproject.toml")
    finally:
        httpd.shutdown()


def test_the_command_line_parses():
    a = server.parser().parse_args(["lumen_wc:sk", "--port", "9000", "--open", "--up", "y", "--range", "300", "600"])
    assert (a.source, a.port, a.open, a.up, a.range) == ("lumen_wc:sk", 9000, True, "y", [300.0, 600.0])
    assert sys.modules["lumen.viewer.server"] is server


def test_a_wavelength_range_marks_the_tables_that_do_not_cover_it():
    p = scene.build(TANK, recipe=RECIPE, wavelength_range=(300.0, 600.0)).payload
    assert p["meta"]["wavelength_range"] == [300.0, 600.0]
    tables = [q for m in p["materials"].values() for q in m["properties"] if q["kind"] == "table"]
    assert tables and all("covers" in q for q in tables if q["key"] not in scene.SPECTRA)


def test_a_gdml_geant4_refuses_reaches_the_page_with_its_repair():
    """The demo barrel's GDML (a matrix Geant4 refuses): no geometry, the error in the page's form, the repair named."""
    view = scene.build(ROOT / "examples" / "data" / "demo_barrel.gdml")
    p = view.payload
    assert view.errors and "instances" not in p
    assert p["errors"][0]["stage"] == "import" and "lumen.gdml.repair" in p["errors"][0]["text"]
    json.loads(server.encode(p))
