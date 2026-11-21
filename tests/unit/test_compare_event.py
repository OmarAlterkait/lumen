"""lumen.compare.event on the toy detector: a g4ref gun run's event written, read, simulated and compared."""

import numpy as np
import pytest
from test_api import INSIDE, converted  # noqa: F401 -- the toy detector, converted once

import lumen
from lumen.compare import event, photons

STEP = [("evt", "<i4"), ("pdg", "<i4"), ("q", "<f4"), ("x0", "<f4"), ("y0", "<f4"), ("z0", "<f4"), ("x1", "<f4"),
        ("y1", "<f4"), ("z1", "<f4"), ("t0", "<f4"), ("t1", "<f4"), ("b0", "<f4"), ("b1", "<f4"), ("edep", "<f4"),
        ("len", "<f4"), ("nc", "<i4"), ("ns", "<i4"), ("mat", "<i4"), ("ef", "<f4")]  # fmt: skip
HIT = [("evt", "<i4"), ("sensor", "<i4"), ("t", "<f4"), ("E", "<f4"), ("origin", "<i4")]


def _run(tmp_path, centres_m, n_steps=8, at=INSIDE, half_mm=100.0):
    """Write a g4ref-like chunk: event 1 a muon-like track along z through the water at ``at`` (m; mm, ns, MeV in
    the files), event 0 a decoy."""
    st = np.zeros(2 * n_steps, STEP)
    st["evt"][n_steps:] = 1
    z = np.linspace(-half_mm, half_mm, n_steps + 1)
    for k in (0, 1):
        s = st[k * n_steps : (k + 1) * n_steps]
        s["pdg"], s["q"], s["x0"], s["x1"] = 13, -1.0, 1000.0 * at[0], 1000.0 * at[0]
        s["y0"], s["y1"] = 1000.0 * at[1], 1000.0 * at[1]
        s["z0"], s["z1"] = 1000.0 * at[2] + z[:-1], 1000.0 * at[2] + z[1:]
        s["t0"], s["t1"] = z[:-1] / 300.0, z[1:] / 300.0
        s["b0"] = s["b1"] = 0.99
        s["edep"], s["len"], s["nc"], s["mat"] = 0.05, 25.0, 100, 0
    hits = np.zeros(5, HIT)
    last = len(centres_m) - 1
    hits["evt"], hits["sensor"], hits["origin"] = [1, 1, 1, 0, 1], [0, last, last, 0, 0], 2
    hits["t"], hits["E"] = 1.0, 3.0
    np.save(tmp_path / "chunk0_steps.npy", st)
    np.save(tmp_path / "chunk0_hits.npy", hits)
    rows = ["sensor,path,lv,cx_mm,cy_mm,cz_mm"] + [f"{i},/s{i},PmtLV,{c[0] * 1e3},{c[1] * 1e3},{c[2] * 1e3}"
                                                   for i, c in enumerate(centres_m)]  # fmt: skip
    (tmp_path / "chunk0_sensors.csv").write_text("\n".join(rows) + "\n")
    (tmp_path / "chunk0_materials.csv").write_text("index,name\n0,Water\n1,Steel\n")
    (tmp_path / "chunk0_summary.txt").write_text("geant4 test\ngdml /x/toy.gdml\nseed 7\n")
    return tmp_path


@pytest.mark.slow  # minutes on 4 CPUs (measured 2-6 min on roma, 4 workers)
def test_an_event_is_written_read_and_compared(converted, tmp_path):  # noqa: F811
    det = lumen.load(converted)
    centres = photons.sensor_centres(photons.setup(det))
    ev = event.extract(_run(tmp_path, centres), tmp_path / "ev.npz", event=1, meta={"particle": "mu-"})
    assert ev.meta == {"geant4": "test", "gdml": "toy.gdml", "seed": "7", "chunk": 0, "event": 1, "particle": "mu-"}
    assert len(ev.steps["edep"]) == 8 and len(ev.hits["t"]) == 4 and ev.materials == ("Water", "Steel")
    np.testing.assert_allclose(ev.steps["start"][0], [INSIDE[0], 0.0, INSIDE[2] - 0.1], atol=1e-6)  # mm to m
    assert ev.steps["edep"][0] == pytest.approx(5e4) and ev.steps["step_length"][0] == pytest.approx(0.025)
    p = det.params()
    assert list(ev.sources(p)) == [("cherenkov", "Water")] and ev.outside(p) == {}
    a = event.simulate(det, ev, samples=4096, seed=0, backend="xla")
    two = event.add(a, a)
    np.testing.assert_allclose(two.arrivals, 2 * np.asarray(a.arrivals))
    np.testing.assert_allclose(two.arrivals_error, np.sqrt(2) * np.asarray(a.arrivals_error), rtol=1e-6)
    res = event.compare(det, ev, samples=4096, seed=0, nt=50, backend="xla")
    assert res["geant4"].sum() == 4 and res["lumen"].shape == res["geant4"].shape == (len(centres),)
    assert res["lumen"].sum() > 0 and np.isfinite(res["summary"]["ratio"]) and res["times_lumen"].shape == (50,)
    with pytest.raises(ValueError, match="no light"):
        event.simulate(det, ev, lights=("scintillation",), samples=4096, seed=0, backend="xla")


@pytest.mark.slow  # converts the tutorial tank and runs one source
def test_an_event_counts_the_deposit_as_the_charge_check_does_on_a_cathode_read_per_normal(tmp_path):
    """The tutorial tank's cathode at REFLECTIVITY 0.7 (A_n 0.3, read for its sensor model's QE): the deposit is the
    physical absorbed share, so lumen's expected hits are the run's arrivals (``charge.response`` "absorbed"), not
    A_n times them (the convention the comparison applied twice before)."""
    import sys
    from pathlib import Path

    from lumen.build.lookup import normal_absorptance
    from lumen.compare import charge

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
    import example_tank

    text = example_tank.GDML.read_text().replace('values="1.5*eV 0.0 5.0*eV 0.0"', 'values="1.5*eV 0.7 5.0*eV 0.7"')
    (tmp_path / "tank.gdml").write_text(text)
    det = lumen.load(lumen.convert(tmp_path / "tank.gdml", example_tank.RECIPE, tmp_path / "tank.lumen"))
    assert np.allclose(normal_absorptance(det.parts.library, det.parts.optics, [400.0]), 0.3, rtol=1e-6)
    centres = photons.sensor_centres(photons.setup(det))
    (tmp_path / "run").mkdir()
    run = _run(tmp_path / "run", centres, at=(0.0, 0.0, 0.38), half_mm=80.0)  # its cone on the top PMT (z 0.47 m)
    ev = event.extract(run, tmp_path / "ev.npz", event=1)
    res = event.compare(det, ev, samples=4096, seed=0, nt=50, backend="xla")
    assert np.all(charge.response(dict(name="event"), det) == 1.0)
    arrivals = np.asarray(res["light"].arrivals, np.float64)
    assert arrivals.sum() > 0
    np.testing.assert_allclose(res["lumen"], arrivals, rtol=1e-5, atol=1e-6 * arrivals.max())
