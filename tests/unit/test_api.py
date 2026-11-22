"""Convert, load, parameters and calibrations, on a small toy detector."""

import json

import numpy as np
import pytest
from readout_toys import TOY_SET, toy_file
from test_importer import GDML, write

import lumen
from lumen.format import calibrations as cal
from lumen.format.calibrations import Table
from lumen.format.sensorsets import SensorGroup, SensorSet
from lumen.params import UnknownParamKey
from lumen.recipe import Recipe

RECIPE = Recipe(
    name="toy",
    detector_volume="TankLV",
    groups=(SensorGroup("pmt", rule="any_entry", lv="PmtLV", settings={"entry_efficiency": 1.0}),),
)


def _toy():
    start = GDML.index('  <bordersurface name="B1"')
    end = GDML.index("</bordersurface>") + len("</bordersurface>\n")
    text = GDML[:start] + GDML[end:]  # counting sensors take no border surface
    # the world of steel (no RINDEX): Geant4 ends a photon leaving the tank, so the tank is the closed detector volume
    # (lumen.build.binding.detector_volume_closed); in the importer's water world it would grow to the world
    world = '<volume name="WorldLV"> <materialref ref="{}"/>'
    return text.replace(world.format("Water"), world.format("Steel"))


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("toy")
    return lumen.convert(write(tmp, _toy()), RECIPE, tmp / "toy.lumen")


@pytest.mark.filterwarnings("error::DeprecationWarning")  # the path runs as under -W error::DeprecationWarning
def test_a_gdml_constant_is_stored_as_one_value_and_handed_on_as_a_scalar(tmp_path):
    """A coldim="1" matrix is a constant: a scalar in memory, a JSON number in the detector file; the parameters
    and the packing read it with no NumPy array-to-scalar conversion (a deprecation that becomes an error)."""
    toy = _toy().replace('  <matrix name="GV"', '  <matrix name="SY" coldim="1" values="100"/>\n  <matrix name="GV"')
    toy = toy.replace(
        '<property name="ABSLENGTH" ref="ABS0x2"/>',
        '<property name="ABSLENGTH" ref="ABS0x2"/> <property name="SCINTILLATIONYIELD" ref="SY"/>',
    )
    path = lumen.convert(write(tmp_path, toy), RECIPE, tmp_path / "c.lumen")
    stored = json.loads((path / "materials.json").read_text())["materials"]["Water"]["properties"]
    assert stored["SCINTILLATIONYIELD"]["value"] == 100.0  # the stored form: a number, no array file
    det = lumen.load(path)
    assert det.file.materials["Water"].properties["SCINTILLATIONYIELD"].value.shape == ()
    p = det.params()
    assert p["materials/Water/SCINTILLATIONYIELD"] == 100.0
    det.pack(p.set("materials/Water/SCINTILLATIONYIELD", 50.0))


def test_convert_and_load(converted):
    det = lumen.load(converted)
    assert det.file.detector_volume == "TankLV"
    assert len(det.sensor_set.ids) == 2 and det._columns["ids"] == list(det.sensor_set.ids)
    assert "recipe" in det.file.provenance and "toy" in repr(det)
    assert det.parts.optics.n_classes >= 2


def test_params_keys_info_and_precedence(converted):
    det = lumen.load(converted)
    p = det.params()
    assert {"materials/Water/ABSLENGTH", "surfaces/Liner/REFLECTIVITY", "sensors/pmt/entry_efficiency"} <= set(p)
    info = p.info("materials/Water/ABSLENGTH")
    assert info.unit == "m" and info.method == "reweight" and info.available
    assert isinstance(p["materials/Water/ABSLENGTH"], Table)
    assert not p.info("materials/Water/RINDEX").available  # refractive-index derivatives: pending validation
    assert p.explain("materials/Water/ABSLENGTH") == "detector file"
    assert p.explain("sensors/pmt/entry_efficiency") == "sensor model defaults"
    with pytest.raises(UnknownParamKey, match="Did you mean"):
        p["materials/Water/ABSLENGHT"]
    q = p.scale("materials/Water/ABSLENGTH", 2.0)
    np.testing.assert_array_equal(q["materials/Water/ABSLENGTH"].value, [2.0, 40.0])
    with pytest.raises(ValueError, match="outside"):
        p.set("surfaces/Liner/REFLECTIVITY", Table(np.array([2.0, 4.5]), np.array([1.2, 0.8])))
    with pytest.raises(ValueError, match="knot"):
        p.set("materials/Water/ABSLENGTH", Table(np.array([2.0, 3.0, 4.5]), np.array([1.0, 2.0, 3.0])))


def test_a_value_set_again_keeps_the_treedef(converted):
    """The layers ride along a Params' flatten without being part of its treedef: a value set again (another layer)
    retraces nothing; the keys and declarations are the treedef."""
    import jax

    p = lumen.load(converted).params()
    k = "surfaces/Liner/REFLECTIVITY"
    q = p.set(k, p[k])
    assert q.explain(k) == "user" and p.explain(k) != "user"
    assert jax.tree_util.tree_structure(q) == jax.tree_util.tree_structure(p)
    traces = []
    f = jax.jit(lambda x: traces.append(1) or x[k].value.sum())
    f(p), f(q), f(q.set(k, q[k]))
    assert len(traces) == 1
    assert jax.tree_util.tree_unflatten(*reversed(jax.tree_util.tree_flatten(q))).explain(k) == "user"


def test_a_calibration_changes_values_never_the_file(converted):
    det = lumen.load(converted)
    path = cal.save(
        cal.Calibration(
            "abs_x2",
            "toy",
            det.digest,
            {"materials/Water/ABSLENGTH": Table(np.array([2.0, 4.5]), np.array([2.0, 40.0]))},
        ),
        converted / "calibrations",
    )
    assert path.exists()
    det2 = lumen.load(converted, calibration="abs_x2")
    assert det2.params().explain("materials/Water/ABSLENGTH") == "calibration abs_x2"
    a, b = det.pack(), det2.pack()
    water = [k for k in a["media"] if "Water" in k][0]
    np.testing.assert_allclose(b["media"][water]["absorption"]["value"], 2 * a["media"][water]["absorption"]["value"])
    assert lumen.load(converted).params()["materials/Water/ABSLENGTH"].value.tolist() == [1.0, 20.0]


def test_save_calibration_and_its_digest_check(converted, tmp_path):
    """save_calibration writes only the changed values with the file's digest; another file's calibration is refused."""
    det = lumen.load(converted)
    p = det.params().scale("materials/Water/ABSLENGTH", 0.5)
    path = det.save_calibration(p, "half", converted / "calibrations", note="ABSLENGTH x 0.5")
    c = cal.read(path)
    assert set(c.values) == {"materials/Water/ABSLENGTH"} and c.detector_digest == det.digest
    det2 = lumen.load(converted, calibration="half")
    assert det2.params().explain("materials/Water/ABSLENGTH") == "calibration half"
    # a calibration is flat: one saved from a calibrated detector holds every value changed from nominal
    q = det2.params().scale("surfaces/Liner/REFLECTIVITY", 0.5)
    c2 = cal.read(det2.save_calibration(q, "half_liner", converted / "calibrations"))
    assert set(c2.values) == {"materials/Water/ABSLENGTH", "surfaces/Liner/REFLECTIVITY"}
    p2 = lumen.load(converted, calibration="half_liner").params()
    assert all(p2.explain(k) == "calibration half_liner" for k in c2.values)
    assert np.array_equal(p2["materials/Water/ABSLENGTH"].value, q["materials/Water/ABSLENGTH"].value)
    cal.save(cal.Calibration("other", "toy", "sha256:another-file", {}), converted / "calibrations")
    with pytest.raises(cal.CalibrationMismatch, match="made for detector file"):
        lumen.load(converted, calibration="other")


def test_the_detector_and_its_sensor_set_carry_one_version_a_calibration_its_own(converted):
    import json

    from lumen.format import schema

    det = lumen.load(converted)
    path = det.save_calibration(det.params().scale("materials/Water/ABSLENGTH", 0.9), "v", converted / "calibrations")
    heads = [converted / "detector.json", converted / "sensors.json"]
    assert {json.loads(f.read_text())["schema"] for f in heads} == {schema.SCHEMA}
    assert json.loads(path.read_text())["schema"] == cal.SCHEMA == "lumen-calibration/1"


def test_sensor_table_follows_the_arrivals_order(converted):
    det = lumen.load(converted)
    t = det.sensor_table()
    n = len(det.parts.optics.sensor_pvs)
    assert t.name == list(det.parts.optics.sensor_pvs) and t.position.shape == (n, 3)
    np.testing.assert_allclose(np.linalg.norm(t.axis, axis=1), 1.0, rtol=1e-6)
    at = {pv: i for i, pv in enumerate(det._columns["ids"])}  # the id map's columns, in the arrivals' order
    assert all(t.columns[k] == [v[at[pv]] for pv in t.name] for k, v in det._columns.items() if k != "ids")


def test_params_set_and_scale_take_traced_values(converted):
    """A traced factor passes through Params (no range check under tracing); the concrete check still holds."""
    import jax

    p0 = lumen.load(converted).params()
    key = "materials/Water/ABSLENGTH"
    v = jax.jit(lambda s: p0.scale(key, s)[key].value)(2.0)
    np.testing.assert_allclose(np.asarray(v), 2.0 * np.asarray(p0[key].value))
    with pytest.raises(ValueError, match="outside"):
        p0.scale(key, -1.0)


def test_the_nominal_calibration_has_one_name(converted):
    a, b = lumen.load(converted), lumen.load(converted, calibration="nominal")
    assert a.calibration == b.calibration == "nominal" and sorted(a.params()) == sorted(b.params())


def test_simulate_takes_a_source():
    """A callable or a Photons batch is not a source."""
    import jax.numpy as jnp

    from lumen import api
    from lumen.record import Photons

    def rows(z, w=1.0):
        n = 4
        x = jnp.stack([jnp.zeros(n), jnp.zeros(n), jnp.full(n, z)], 1)
        d = jnp.tile(jnp.array([0.0, 0, 1]), (n, 1))
        t, lam, pol = jnp.zeros(n), jnp.full(n, 400.0), jnp.zeros((n, 2, 3))
        return Photons(x, d, t, lam, jnp.full(n, w), pol)

    with pytest.raises(TypeError, match="not a lumen.sources.Source"):
        api.simulate(None, rows(0.0), samples=4, seed=0)


def test_a_run_refuses_more_photons_than_its_ids_hold_and_clashing_diagnostic_names(converted):
    """Photon ids are int32: a run of 2^31 photons is refused. A diagnostic output named as the run's own (a health
    key, a histogram) or as another diagnostic's is refused; so is a probe over several chunks."""
    from dataclasses import dataclass

    import jax.numpy as jnp

    from lumen import sources

    det = lumen.load(converted)
    src = sources.Point(position=tuple(INSIDE), lam=400.0, n_photons=1e6)
    with pytest.raises(ValueError, match="split the run"):
        lumen.simulate(det, src, samples=1 << 31, seed=0)

    @dataclass(frozen=True)
    class Probe:
        name: str

        def outputs(self):
            return {self.name: (1, True)}

        def init(self, ph):
            return jnp.int32(0)

        def update(self, mem, ph, ph2, dep, pid):
            return mem, []

    for dgs in ((Probe("truncated"),), (Probe("charge:0"),), (Probe("a"), Probe("a"))):
        with pytest.raises(ValueError, match="rename the diagnostic"):
            lumen.simulate(det, src, samples=64, seed=0, diagnostics=dgs)
    with pytest.raises(ValueError, match="pass chunk >= samples"):
        lumen.simulate(det, src, samples=64, seed=0, chunk=32, diagnostics=(Probe("a"),))


def test_the_memory_plan_is_the_layout_s_own(converted, monkeypatch):
    """Before launch the device's memory is shared: the rows in flight and the readout's float32 cells (and the
    derivative's tangents) first, then the deposit records and the readout's exact cells, in as many blocks of sensors
    as they need; a device that cannot hold the float cells refuses the run. A launch with values starts at one slot
    per row, then the counts' power of two; under a trace the slots are the most the memory holds; records= is the
    power of two at or above it, a MemoryError when it does not fit; not one slot is RecordsUnknown."""
    from lumen import api
    from lumen import run as runner
    from lumen.readout import fill
    from lumen.readout import grid as ro

    det = lumen.load(converted)
    lay = api._layouts(det, None)
    r = tuple(x.layout for x in lay)
    cells = sum(c for c, _ in fill.readout(r).outputs().values())
    fields = runner.record_fields((), False)
    rb = runner.record_bytes(fields)
    limit = [0.0]
    monkeypatch.setattr(runner, "memory_limit", lambda backend: limit[0])
    floats = cells * ro.CELL_BYTES * 2  # the run's float32 cells, and room for a copy
    n_all = r[0].n_sensors if r[0].index is None else len(r[0].index)
    exact = runner.readout_bytes(sum(fill.sensor_cells(x) * x.n_sensors for x in r) + len(fill.ARRIVALS) * n_all,
                                 False, "auto")  # fmt: skip
    limit[0] = floats + 100 + 4 * exact
    free = api._memory(lay, False, 100, "auto")
    assert free == 4 * exact and api._memory(lay, True, 100, "auto") == 4 * exact - floats  # tangents
    room = runner.slots_in(free, fields, False)
    assert room & (room - 1) == 0 and room * runner.slot_bytes(fields, False) <= free < 2 * room * runner.PEAK * rb
    assert runner.capacity(room, rows=100) == min(128, room)  # values: one slot per row to start
    assert runner.capacity(room, rows=100, counted=2) == 2  # then the counts' power of two
    assert runner.capacity(room) == room  # traced: the most the memory holds
    assert runner.capacity(room, records=3) == 4  # given: the power of two at or above it
    ints = sum(np.dtype(d).itemsize for _, d in fields if np.issubdtype(np.dtype(d), np.integer))
    assert ints == 8  # sensor, photon
    assert runner.slot_bytes(fields, False, 3) == runner.PEAK * rb + 3 * (rb + runner.KEY_BYTES + ints)  # 3 members
    with pytest.raises(runner.RecordsUnknown, match="not one deposit record"):
        runner.capacity(0)
    with pytest.raises(MemoryError, match="records="):
        runner.capacity(room, records=2 * room + 1)
    assert fill.plan(r, free - 4 * rb, False, "auto") == (fill.full(r),)  # everything in one launch
    blocks = fill.plan(
        r, exact, True, "auto"
    )  # half the exact cells' room with tangents: several blocks, every sensor once
    spans = [sp for b in blocks for sp in b.spans if sp is not None]
    assert len(blocks) > 1 and sum(b - a for a, b in spans) == sum(x.n_sensors for x in r)
    assert [b.arrivals for b in blocks] == [True] + [False] * (len(blocks) - 1)
    limit[0] = floats + 100
    with pytest.raises(MemoryError, match="over the device"):
        api._memory(lay, False, 100, "auto")


def test_record_counts_and_readout_cells_never_wrap_int32():
    """tallymap's record counts are int32: a count past the slots, a drop and a wrapped (negative) count are each an
    overflow, and past MAX_RECORDS no launch holds it (review7 F2 #2). The readout's launches keep every output's
    cells below 2^31, so a flat int32 index never wraps (review7 F2 #5)."""
    from lumen import run as runner
    from lumen.readout import fill
    from lumen.readout import grid as ro

    assert runner.overflow(100, 0, 128) == (False, 128)
    assert runner.overflow(200, 5, 128) == (True, 256)
    assert runner.overflow(-5, 7, 128) == (True, None)  # wrapped past 2^31 - 1
    assert runner.overflow(100, 3, 128) == (True, None)  # a drop the count does not show
    assert runner.overflow(runner.MAX_RECORDS + 1, 1, 128) == (True, None)
    lay = fill.HistogramLayout(100, ro.Grid(t0=0.0, dt=1.0, nt=1 << 26))  # 2^26 time cells per sensor
    blocks = fill.plan((lay,), 1e30, False, "auto")  # memory without bound: only the indices limit a launch
    spans = [b.spans[0] for b in blocks]
    assert len(blocks) > 1 and sum(e - s for s, e in spans) == 100
    assert all((e - s) * lay.per(k) < 2**31 for s, e in spans for k in lay.kinds())
    with pytest.raises(MemoryError, match="int32"):
        fill.plan((fill.HistogramLayout(2, ro.Grid(t0=0.0, dt=1.0, nt=1 << 31)),), 1e30, False, "auto")


def test_the_default_chunk_fits_the_records_and_a_given_chunk_is_refused_when_they_do_not(converted, monkeypatch):
    """A launch whose count (tallymap's ``appended``) is more deposit records than fit beside the readout
    (``_record_room``): the default chunk runs those photons again in smaller launches (the run of that chunk given,
    to the launches' float32 sum), a given chunk raises a MemoryError naming the counts."""
    from lumen import api, sources

    det, n = lumen.load(converted), 512
    src = sources.Point(INSIDE, n_photons=1e3, lam=400.0)
    want = lumen.simulate(det, src, samples=n, seed=2, backend="xla")
    assert dict(want.provenance)["chunk"] == n and int(want.health.records) > 64
    monkeypatch.setattr(api, "_record_room", lambda *a, **k: 64)
    got = lumen.simulate(det, src, samples=n, seed=2, backend="xla")
    size = dict(got.provenance)["chunk"]
    assert size < n
    same = lumen.simulate(det, src, samples=n, seed=2, backend="xla", chunk=size)
    for other in (same, want):  # the same photons in other launches: the launches' float32 sum
        np.testing.assert_allclose(np.asarray(got.arrivals), np.asarray(other.arrivals), rtol=1e-6)
    with pytest.raises(MemoryError, match="deposit records, over the 64"):
        lumen.simulate(det, src, samples=n, seed=2, backend="xla", chunk=n)


def _records(det, src, p, n) -> int:
    """Return twice the deposit records of the run with values: a jitted run's static number (``records=``)."""
    return 2 * int(lumen.simulate(det, src, p, samples=n, seed=1, backend="xla").health.records)


@pytest.mark.slow  # minutes on 4 CPUs (measured 2.5-9.5 min on roma, 4 workers)
def test_edge_rows_are_the_detectors_and_a_derivative_does_not_change_the_run(converted):
    """The edge rows are every row a parameter with a domain is read into (the detector's structure, never a tangent or
    a value), so a value-only run, a jitted one and the primal of its jvp are the same run, bitwise."""
    import jax

    from lumen import api, sources

    det, key, n = lumen.load(converted), "surfaces/Liner/REFLECTIVITY", 4096
    p = det.params()
    liner = next(i for i, r in enumerate(det.parts.optics.rows) if r.origin == "surface:Liner")
    assert liner in api._edge_rows(det).surfaces  # the Liner's row
    src = sources.Point(INSIDE, n_photons=1e3, lam=400.0)

    rec = _records(det, src, p, n)

    def arrivals(s):
        return lumen.simulate(det, src, p.scale(key, s), samples=n, seed=1, backend="xla", records=rec).arrivals

    value = np.asarray(arrivals(1.0))
    assert value.sum() > 0.0
    np.testing.assert_array_equal(np.asarray(jax.jit(arrivals)(1.0)), value)
    primal, _ = jax.jvp(arrivals, (1.0,), (1.0,))
    np.testing.assert_array_equal(np.asarray(primal), value)


@pytest.mark.slow  # minutes on 4 CPUs (measured 2.5-9.5 min on roma, 4 workers)
def test_a_traced_value_outside_its_range_is_counted_and_its_derivative_is_nan(converted, monkeypatch):
    """A concrete out-of-range value is refused when set; a traced one (jit, grad) cannot be: health.invalid
    counts its elements and its derivative is NaN (it was a silent 0). Inside the range nothing changes, bit for bit."""
    import jax

    from lumen import sources

    det, key, n = lumen.load(converted), "surfaces/Liner/REFLECTIVITY", 1024  # R in [0, 1], two knots
    p = det.params()
    src = sources.Point(INSIDE, n_photons=1e3, lam=400.0)

    rec = _records(det, src, p, n)

    def light(s):
        return lumen.simulate(det, src, p.scale(key, s), samples=n, seed=1, backend="xla", records=rec)

    with pytest.raises(ValueError, match="outside"):
        p.scale(key, 2.0)
    run = jax.jit(light)
    assert int(run(2.0).health.invalid) == 2 and int(run(1.0).health.invalid) == 0
    assert int(light(1.0).health.invalid) == 0
    grad = jax.grad(lambda s: light(s).arrivals.sum())
    assert np.isnan(float(grad(2.0)))
    inside = np.asarray(run(1.0).arrivals), float(grad(1.0))
    from lumen.build import values

    monkeypatch.setattr(values, "check", lambda d_, p_, s_=None, r_=None: (p_, s_, 0))  # without the check: same bits
    assert np.array_equal(np.asarray(jax.jit(light)(1.0).arrivals), inside[0]) and float(grad(1.0)) == inside[1]


FAR_SET = toy_file(name="toy_b", qe=dict(wavelength_nm=[350.0, 450.0, 550.0], values=[0.2, 0.3, 0.1]))  # another grid


def _two_groups(tmp_path, columns_b=True, kinds=("near", "far")):
    """The toy's two PMTs (one volume) as two modelled groups, split by an id-map column (one sensor each)."""
    from lumen.recipe import Recipe

    def cols(df, paths, inputs):
        return {"kind": list(kinds)[: len(paths)]}

    groups = (
        SensorGroup("near", rule="any_entry", lv="PmtLV", values=TOY_SET,
                    where={"kind": "near"}),
        SensorGroup("far", rule="any_entry", lv="PmtLV", values=FAR_SET,
                    where={"kind": "far"} if columns_b else None),
    )  # fmt: skip
    r = Recipe(name="toy2", detector_volume="TankLV", groups=groups, sensor_columns=cols)
    return lumen.convert(write(tmp_path, _toy()), r, tmp_path / "toy2.lumen")


def test_several_modelled_groups_each_have_their_model_values_and_histograms(tmp_path):
    """Two groups sharing a volume, split by a column: each has its model's keys over its own sensor, its own grid and
    index map, and the photon loop has one histogram set per group (suffixed); one group keeps today's names"""
    from lumen import api
    from lumen.readout import fill

    det = lumen.load(_two_groups(tmp_path))
    p = det.params()
    assert np.shape(p["sensors/near/efficiency"]) == (1,) and np.shape(p["sensors/far/efficiency"]) == (1,)
    assert p.explain("sensors/far/tts/sigma") == "sensor model defaults"
    with pytest.raises(ValueError, match="gate="):  # timed models (hits) and the toy's recipe sets no gate
        api._layouts(det, None)
    lay = api._layouts(det, (0.0, 100.0))
    assert [g.name for g, *_ in lay] == ["near", "far"]
    (near, *_, ln), (far, *_, lf) = lay
    assert sorted([ln.index, lf.index]) == [(-1, 0), (0, -1)]  # one sensor each, the other -1
    assert ln.grid != lf.grid  # each group's own model sets its grid
    names = list(fill.readout((ln, lf)).outputs())
    assert names[:4] == ["charge:0", "time:0", "over_q:0", "over_qt:0"] and "charge:1" in names and "arrival" in names
    with pytest.raises(ValueError, match="selects by the id-map columns"):
        SensorSet((SensorGroup("x", rule="any_entry", lv="L", where={"nope": 1}),), ("a",), {"kind": ["n"]})


def test_groups_sharing_a_volume_must_split_it_by_columns(tmp_path):
    from lumen.build.library import LibraryError

    with pytest.raises(LibraryError, match="select its sensors by the id map's columns"):
        lumen.load(_two_groups(tmp_path, columns_b=False))


def test_groups_sharing_a_volume_cover_it(tmp_path):
    """A sensor of a group-selected volume whose column value no group claims (a new or misspelt type) is refused at
    load, naming it: its light would vanish from every group."""
    with pytest.raises(ValueError, match=r"1 sensors are in no sensor group, e\.g\. .* \{'kind': 'other'\}"):
        lumen.load(_two_groups(tmp_path, kinds=("near", "other")))


def test_convert_takes_gdml_or_yaml(tmp_path):
    """A GDML file (with its recipe) or YAML: only GDML takes a recipe; anything else is refused."""
    import builder_toys as toys

    b = toys.cube_wall(n=2)
    df, sets = b.structure()
    with pytest.raises(TypeError, match="a GDML file or a YAML detector"):
        lumen.convert(df, None, tmp_path / "bare.lumen")
    with pytest.raises(TypeError, match="a GDML file or a YAML detector"):
        lumen.convert(lambda: b, None, tmp_path / "f.lumen")
    yaml_dir = tmp_path / "y"
    yaml_dir.mkdir()
    (yaml_dir / "d.yaml").write_text("schema: lumen-detector-yaml/3\n")
    with pytest.raises(ValueError, match="only a GDML file takes a recipe"):
        lumen.convert(yaml_dir, RECIPE, tmp_path / "x.lumen")
    with pytest.raises(ValueError, match="needs its recipe"):
        lumen.convert(write(tmp_path, _toy()), None, tmp_path / "y.lumen")


def test_load_takes_a_path_only(tmp_path, monkeypatch):
    """A detector is its file: load takes its path; a name resolves nothing, wherever LUMEN_DETECTORS points."""
    import builder
    import builder_toys as toys

    path = builder.save(toys.water_sphere(n_pmt=4), tmp_path / "ball.lumen")
    assert lumen.load(path).path == path == lumen.load(str(path)).path
    monkeypatch.setenv("LUMEN_DETECTORS", str(tmp_path))
    monkeypatch.chdir(tmp_path.parent)
    with pytest.raises(FileNotFoundError, match="convert it first"):
        lumen.load("ball")


# ------------------------------------------------------------------------------- Cherenkov light and diagnostics
HC_EV_NM = 1239.841984
INSIDE = (0.0, 0.0, 0.3)  # in the toy's water


def test_cherenkov_light_spans_the_rindex_table_beyond_the_detector_s_range(converted):
    """A Cherenkov source emits over its medium's RINDEX table (275.5-619.9 nm here), as Geant4 does, not over the
    detector's wavelength range (300-600 nm): simulate takes it with its bounds unset or given as the table's span.
    (lumen 0.2 at 683280e refused every Cherenkov call whose table reached beyond the range: SBND, MicroBooNE and
    ICARUS, whose LAr RINDEX spans 110.8-1045.2 nm.)"""

    import jax

    from lumen import sources

    det = lumen.load(converted)
    p = det.params()
    a, one = np.array([INSIDE]), np.ones(1)
    steps = sources.CherenkovSteps("Water", start=a, end=a + [0.0, 0.0, 0.05], t_start=0 * one, t_end=0.17 * one,
                                   beta_start=0.99 * one, beta_end=0.99 * one, charge=one)  # fmt: skip
    lam = np.asarray(steps.emit(p, jax.random.PRNGKey(0), 4096).lam, np.float64)
    assert HC_EV_NM / 4.5 <= lam.min() < 290.0 and 600.0 < lam.max() <= HC_EV_NM / 2.0 * (1 + 1e-6)
    for src in (steps, sources.CherenkovSegment("Water", start=INSIDE, length=0.05, beta=0.99)):
        light = lumen.simulate(det, src, samples=4096, seed=1, backend="xla")
        assert float(np.asarray(light.arrivals).sum()) > 0.0


def test_a_diagnostic_sums_deposits_in_the_readout_s_units(converted):
    """A diagnostic's Deposit carries the readout's weights: rows of weights 1 and 10 (the photon's weight is relative
    to its row in the loop) give per-sensor diagnostic sums equal to the arrivals, and lost-weight sums equal to the
    health's."""
    from dataclasses import dataclass

    import jax.numpy as jnp

    from lumen import sources

    det, n = lumen.load(converted), 4096
    d = np.random.default_rng(0).normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    w = np.where(np.arange(n) % 2 == 0, 10.0, 1.0)
    src = sources.FromArrays(np.tile(INSIDE, (n, 1)), d, np.zeros(n), np.full(n, 400.0), w)
    s = len(det.parts.optics.sensor_pvs)

    @dataclass(frozen=True)
    class Sums:
        def outputs(self):
            return {"q": (s, True), "lost_w": (1, True)}

        def init(self, ph):
            return jnp.int32(0)

        def update(self, mem, ph, ph2, dep, pid):
            return mem, [("q", jnp.where(dep.sensor >= 0, dep.sensor, -1), dep.q), ("lost_w", jnp.int32(0), dep.lost)]

    light = lumen.simulate(det, src, samples=n, seed=2, backend="xla", diagnostics=(Sums(),))
    q = np.asarray(light.diagnostics["q"], np.float64)
    arrivals = np.asarray(light.arrivals, np.float64)
    assert arrivals.sum() > 0.0
    np.testing.assert_allclose(q, arrivals, rtol=1e-5)
    lost = float(np.asarray(light.diagnostics["lost_w"], np.float64).sum())
    assert lost == pytest.approx(float(np.asarray(light.health.lost, np.float64).sum()), rel=1e-5)


@pytest.mark.slow  # minutes on 4 CPUs (measured 2.5-9.5 min on roma, 4 workers)
def test_the_estimator_follows_the_tangents_with_no_declaration(converted):
    """No ``wrt``: a derivative is taken of whatever carries a tangent (one custom_jvp rule, symbolic zeros kept),
    eagerly, under jit and in reverse mode; a value run with traced values is not differentiated; a source's own
    traced field gets its derivative too; a source function closing over a tracer is refused with a clear error."""
    import jax
    import jax.numpy as jnp

    from lumen import sources

    det, key, n = lumen.load(converted), "surfaces/Liner/REFLECTIVITY", 1024
    p = det.params()
    src = sources.Point(INSIDE, n_photons=1e3, lam=400.0)

    rec = _records(det, src, p, n)

    def total(s):
        return lumen.simulate(det, src, p.scale(key, s), samples=n, seed=1, backend="xla", records=rec).arrivals.sum()

    fwd = float(jax.jvp(total, (1.0,), (1.0,))[1])
    assert np.isfinite(fwd) and fwd != 0.0
    assert float(jax.grad(total)(1.0)) == pytest.approx(fwd, rel=1e-5)
    assert float(jax.jit(jax.grad(total))(1.0)) == pytest.approx(fwd, rel=1e-5)
    assert float(jax.jit(total)(1.0)) == pytest.approx(float(total(1.0)), rel=1e-6)  # traced, not differentiated

    def weighted(s):  # a source's own traced field (not a parameter): its derivative, the total weight
        rows = sources.FromArrays(jnp.tile(jnp.asarray(INSIDE), (n, 1)), jnp.tile(jnp.asarray([0.0, 0.0, 1.0]), (n, 1)),
                                  jnp.zeros(n), jnp.full(n, 400.0), jnp.ones(n) * s)  # fmt: skip
        return lumen.simulate(det, rows, samples=n, seed=1, backend="xla").arrivals.sum()

    v, d = jax.jvp(weighted, (1.0,), (1.0,))
    assert float(d) == pytest.approx(float(v), rel=1e-5)  # every weight times s

    def closing(s):  # a source function closing over a traced value it was not given
        @lumen.source(name="closing")
        def f(v, key, m):
            up = jnp.tile(jnp.asarray([0.0, 0.0, 1.0]), (m, 1))
            return sources.Photons(jnp.tile(jnp.asarray(INSIDE), (m, 1)), up, jnp.zeros(m), jnp.full(m, 400.0),
                                   jnp.ones(m) * s)  # fmt: skip

        return lumen.simulate(det, f, samples=n, seed=1, backend="xla").arrivals.sum()

    with pytest.raises(sources.GradientNotAvailable, match="declare it as a parameter"):
        jax.jvp(closing, (1.0,), (1.0,))


def test_the_time_shape_is_the_detected_light_s_at_the_qe_given_and_differentiable_in_it(tmp_path):
    """The time histogram is weighted by the QE x CE of the values the run is given (an operand), so the time shape
    moves with the QE's shape and its derivative in a QE knot is not zero: the derivative through the run matches a
    finite difference of runs at the same seed (tracking does not read the QE, so the difference is exact per seed)."""
    import jax
    import jax.numpy as jnp

    from lumen import readout, sources

    det = lumen.load(_two_groups(tmp_path))
    p, k, gate = det.params(), "sensors/near/qe", (0.0, 100.0)
    qe0 = jnp.asarray(p[k], jnp.float32)
    tilt = jnp.linspace(-1.0, 1.0, qe0.shape[0], dtype=jnp.float32)
    src = sources.Point(INSIDE, n_photons=1e3, lam=jnp.linspace(300.0, 600.0, 2048, dtype=jnp.float32))

    def mean_time(s):
        light = lumen.simulate(det, src, p.set(k, qe0 * (1.0 + s * tilt)), samples=2048, seed=4,
                               gate=gate, backend="xla")  # fmt: skip
        h = light.group("near")
        q = p.set(k, qe0 * (1.0 + s * tilt))
        nodes, _, _ = readout.detection.masses(h, readout.detection.values(h, q))  # before the time spread
        t = h.grid.t0 + h.grid.dt * jnp.arange(h.grid.nt)
        return jnp.sum(nodes * t) / jnp.sum(nodes)

    _, ad = jax.jvp(mean_time, (0.0,), (1.0,))
    h = 0.05
    fd = (float(mean_time(h)) - float(mean_time(-h))) / (2 * h)
    assert abs(float(ad)) > 1e-3 and float(ad) == pytest.approx(fd, rel=1e-3, abs=1e-5)


@pytest.mark.slow  # minutes on 4 CPUs (measured 2.5-9.5 min on roma, 4 workers)
def test_a_photon_born_inside_a_volume_sensor_is_counted_at_its_emission(tmp_path):
    """An any_entry group declares a sensitive volume: by default (count "volume", Geant4's sensitive detector) a photon
    emitted inside it is counted there with the entry's detection, at its emission time; with count "entry" only
    entries count. A count on a border_into group is refused."""
    from lumen import sources
    from lumen.format.sensorsets import SensorSet

    src = sources.Point((-0.5, 0.0, 0.0), n_photons=1e3, lam=400.0, time=2.5)  # inside the second PMT
    out = {}
    for count in ("volume", "entry"):
        g = SensorGroup("pmt", rule="any_entry", lv="PmtLV", settings={"entry_efficiency": 0.75, "count": count})
        r = Recipe(name=f"toy_{count}", detector_volume="TankLV", groups=(g,))
        (tmp_path / count).mkdir()
        det = lumen.load(lumen.convert(write(tmp_path / count, _toy()), r, tmp_path / f"{count}.lumen"))
        light = lumen.simulate(det, src, samples=1024, seed=2, backend="xla")
        out[count] = (np.asarray(light.arrivals, np.float64), np.asarray(light.mean_arrival_time, np.float64))
    a, t = out["volume"]
    i = int(np.argmax(a))
    assert a[i] == pytest.approx(0.75e3, rel=1e-5) and a.sum() == pytest.approx(a[i]) and t[i] == pytest.approx(2.5)
    assert out["entry"][0].sum() == 0.0
    model = lumen.load(_two_groups(tmp_path))  # under a sensor model the entry's efficiency, 1, is folded into its type
    light = lumen.simulate(model, src, samples=1024, seed=2, gate=(0.0, 100.0), backend="xla")
    a = np.asarray(light.arrivals, np.float64)
    assert a.max() == pytest.approx(1e3, rel=1e-5) and a.sum() == pytest.approx(a.max())
    with pytest.raises(ValueError, match="setting count"):
        SensorSet((SensorGroup("x", rule="border_into", lv="L", settings={"count": "volume"}),))


@pytest.mark.slow  # more than 16 GB with 4 workers (killed on a 16 GB roma job)
def test_jit_and_vmap_over_the_whole_params_run_and_a_derivative_without_a_method_is_refused(converted):
    """A value-only trace of every key packs (the Rayleigh table derived from a traced RINDEX in JAX, as at load); the
    jitted run equals the eager one, at the detector's values and at a new RINDEX; only a derivative is refused where
    its method is not available (RINDEX). Traced, records= is optional: without it the slots are the memory's (here a
    small memory), the same bitwise; a launch past its slots raises RecordsOverflow when it runs."""
    import jax

    from lumen import run as runner
    from lumen import sources

    det = lumen.load(converted)
    p = det.params()
    rk = next(k for k in p if k.endswith("/RINDEX") and det.parts.key_rows.get(
        ("media", k.split("/")[1], "rayleigh", "length", "value"), (None, None))[1] == "derived")  # fmt: skip
    src = sources.Point(INSIDE, n_photons=1e3, lam=400.0)

    rec = _records(det, src, p, 1024)

    def arrivals(q):
        return lumen.simulate(det, src, q, samples=1024, seed=1, backend="xla", records=rec).arrivals

    for q in (p, p.scale(rk, 1.01)):
        np.testing.assert_allclose(np.asarray(jax.jit(arrivals)(q)), np.asarray(arrivals(q)), rtol=1e-6)
        m = rk.split("/")[1]
        ray = lambda pk, m=m: np.asarray(pk["media"][m]["rayleigh"]["length"]["value"])  # noqa: E731
        np.testing.assert_allclose(ray(jax.jit(det.pack)(q)), ray(det.pack(q)), rtol=1e-6)
    both = jax.vmap(lambda s: arrivals(p.scale(rk, s)))(np.array([1.0, 1.01]))
    np.testing.assert_allclose(np.asarray(both[1]), np.asarray(arrivals(p.scale(rk, 1.01))), rtol=1e-6)
    with pytest.raises(sources.GradientNotAvailable, match="not available"):
        jax.jvp(lambda s: arrivals(p.scale(rk, s)).sum(), (1.0,), (1.0,))
    import jax.numpy as jnp

    lim = runner.memory_limit("xla")
    with pytest.MonkeyPatch.context() as mp:  # a small memory: the slots it holds, not the host's gigabytes
        mp.setattr(runner, "memory_limit", lambda backend: min(lim, 2e8))
        free = jax.jit(lambda q: lumen.simulate(det, src, q, samples=1024, seed=1, backend="xla").arrivals)(p)
    assert np.asarray(free).tobytes() == np.asarray(jax.jit(arrivals)(p)).tobytes()
    with pytest.raises(Exception, match="RecordsOverflow|over its 1 slots"):  # past its slots: refused as it runs
        jnp.asarray(jax.jit(lambda q: lumen.simulate(det, src, q, samples=1024, seed=1, backend="xla",
                                                     records=1).arrivals)(p)).block_until_ready()  # fmt: skip


def test_a_direction_is_checked_by_the_one_value_rule_concrete_or_traced(converted):
    """A direction that is not a unit vector raises InvalidValue when concrete and is counted in health.invalid
    when traced (it was not checked under a trace)."""
    import jax
    import jax.numpy as jnp

    from lumen import sources
    from lumen.build import values

    det = lumen.load(converted)

    def invalid(s):
        src = sources.Point(INSIDE, n_photons=1e3, lam=400.0, direction=jnp.array([0.0, 0.0, 1.0]) * s)  # |d| = s
        return lumen.simulate(det, src, samples=256, seed=1, backend="xla", records=1 << 12).health.invalid

    with pytest.raises(values.InvalidValue, match="unit vector"):
        invalid(0.5)
    run = jax.jit(invalid)
    assert int(run(0.5)) == 1 and int(run(1.0)) == 0


def test_an_eager_derivative_sizes_its_records_from_the_count(converted, monkeypatch):
    """Outside jax.jit / jax.vmap a derivative's run (jax.jvp, jax.grad) has values: its launches take their slots
    from tallymap's count (one slot per row to start), never the memory budget of a traced run."""
    import jax

    from lumen import run as runner
    from lumen import sources

    det = lumen.load(converted)
    p = det.params()
    key = next(k for k in p if k.endswith("/ABSLENGTH"))
    src = sources.Point(INSIDE, n_photons=1e3, lam=400.0)
    calls, capacity = [], runner.capacity
    monkeypatch.setattr(runner, "capacity", lambda room, **k: calls.append(k) or capacity(room, **k))
    jax.jvp(lambda s: lumen.simulate(det, src, p.scale(key, s), samples=512, seed=1).arrivals.sum(), (1.0,), (1.0,))
    assert calls and all(k.get("rows") is not None for k in calls), calls


def test_the_class_table_is_the_navigator_s_volumes_by_name(converted):
    """Volume i of the optics is volume i of the navigator, checked by each volume's instance path, not by the counts
    alone: a navigator listing the same number of volumes in another order is refused."""
    from types import SimpleNamespace

    from lumen.build import tracked

    det = lumen.load(converted)
    names = tracked.from_parts(det.parts).nav.names
    assert len(names) > 1
    same = SimpleNamespace(n=len(names), n_mesh=0, names=list(names))
    tracked.classes(same, det.parts.nav_scene, det.parts.optics)
    swapped = SimpleNamespace(n=len(names), n_mesh=0, names=list(names)[::-1])
    with pytest.raises(tracked.GeometryError, match="would attach to other volumes"):
        tracked.classes(swapped, det.parts.nav_scene, det.parts.optics)
