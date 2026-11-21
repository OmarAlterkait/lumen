"""No recompile per value (docs/dev/decisions.md, 2): every kind of value enters the photon loop as an operand.

The loop runs once to compile, then again with each kind of value changed: materials, surfaces, sensors, a
calibration, source fields, controls, the seed and the number of samples. No new XLA compilation may happen, except
for a launch whose deposit records take a power of two of slots not seen before (``lumen.run.capacity``: the slots are
the launch's own count, so a handful of record shapes compile, never one per value), and the photon's step is never
traced again (tallymap traces its item at each counted launch; the step inside it is traced once per model and
shapes, ``lumen.run._once``). ``simulate`` compiles once per power of two of rows: a new number of samples, of
FromArrays rows or of CherenkovSteps steps in the same power of two compiles nothing, nor does a new position of a
source whose position is differentiated, nor a new event (new step data in the same shapes). A loop that closes over a
float array is refused.

The physics tier (one GPU, ``backend="auto"``): on a CPU the loop's first compile alone takes minutes. The toy is the
unit tier's (tests/unit/test_api.py: a tank and two tubes).
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))  # the unit tier's toy
from test_api import RECIPE, _toy
from test_importer import write

import lumen
from lumen import run as runner
from lumen.build import tracked
from lumen.compare import photons
from lumen.readout import fill
from lumen.readout.grid import Grid
from lumen.record import Photons
from lumen.transport import step as steps

N = 64
COMPILE = "/jax/core/compile/backend_compile_duration"
SEEN_EVENTS: list = []  # every event since the module's listener was registered (one listener: they are never removed)
jax.monitoring.register_event_duration_secs_listener(lambda name, *a, **k: SEEN_EVENTS.append(name))


class Watch:
    """After a warm-up: the XLA compiles of each call, the records' slots of its launches and the step's traces."""

    def __init__(self, mp):
        self.slots, self.steps, self.bad, self.live = set(), 0, [], False
        trace, step = runner.trace, runner.step

        def counted_step(*a):
            self.steps += self.live  # a Python call of the step is a trace of it (it runs only inside a jit)
            return step(*a)

        def watched_trace(*a, **k):
            t = trace(*a, **k)
            self.launched.add(t.records["sensor"].shape[0])
            return t

        mp.setattr(runner, "step", counted_step)
        mp.setattr(runner, "trace", watched_trace)
        self.launched = set()

    def __call__(self, f, *a, **k):
        """Return f(*a, **k); after ``start``, a compile in a call whose launches took only slots seen is kept."""
        n0, self.launched = len(SEEN_EVENTS), set()
        out = f(*a, **k)
        compiles = [e for e in SEEN_EVENTS[n0:] if e == COMPILE]
        if self.live and compiles and self.launched <= self.slots:
            self.bad.append((len(compiles), sorted(self.launched)))
        self.slots |= self.launched
        return out

    def start(self):
        """Begin watching (after the warm-up)."""
        self.live = True

    def check(self):
        """Refuse a compile with no new records' slots, or a trace of the step, after the warm-up."""
        assert not self.bad and self.steps == 0, (self.bad, self.steps)


def rows(shift=0.0, t=0.0, lam=400.0):
    d = np.random.default_rng(0).normal(size=(N, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    z = jnp.zeros(N, jnp.float32)
    x = jnp.tile(jnp.asarray([0.3 + shift, 0.0, 0.0], jnp.float32), (N, 1))
    return Photons(x, jnp.asarray(d, jnp.float32), z + t, z + lam, z + 1.0, jnp.zeros((N, 2, 3), jnp.float32))


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("norecompile")
    det = lumen.load(lumen.convert(write(tmp, _toy()), RECIPE, tmp / "toy.lumen"))
    model = steps.Model(tracked.from_parts(det.parts), det.parts.optics)
    layout = fill.HistogramLayout(len(det.parts.optics.sensor_pvs), Grid.bands((300.0, 600.0)))
    return det, model, photons.loop(layout)


def test_no_value_recompiles(setup, monkeypatch):
    det, model, spec = setup
    w = Watch(monkeypatch)
    p = det.params()

    def go(*a, **kw):
        return w(run, *a, **kw)

    def run(params=p, controls=None, r=None, seed=0, n_active=N):
        ops = steps.operands(model, det.pack(params), None, controls or steps.Controls())
        key = jax.random.key_data(jax.random.PRNGKey(seed))
        t = runner.run(spec, model, ops, rows() if r is None else r, key, max_vertices=500, backend="auto",
                       n_active=n_active)  # fmt: skip
        return float(np.asarray(t.outputs["charge:0"]).sum())

    q0 = go()
    w.start()
    absorb = p.scale("materials/Water/ABSLENGTH", 0.5)
    calibration = p.with_layer({"materials/Water/ABSLENGTH": absorb["materials/Water/ABSLENGTH"],
                                "sensors/pmt/entry_efficiency": 0.5}, "calibration test")  # fmt: skip
    changed = dict(
        materials=go(absorb),
        surfaces=go(p.scale("surfaces/Liner/REFLECTIVITY", 0.5)),
        sensors=go(p.set("sensors/pmt/entry_efficiency", 0.25)),
        calibration=go(calibration),
        source=go(r=rows(shift=0.1, t=2.0, lam=450.0)),
        controls=go(controls=steps.Controls(0.5, 1e-2)),
        seed=go(seed=7),
        samples=go(n_active=N // 2),
    )
    w.check()
    assert changed["sensors"] < q0 and changed["samples"] < q0, (q0, changed)


def test_a_loop_closing_over_a_float_array_is_refused(setup):
    det, model, spec = setup
    ops = steps.operands(model, det.pack(), None, steps.Controls())
    baked = jnp.linspace(0.5, 1.0, 4, dtype=jnp.float32)  # a value the item closes over instead of reading an operand
    start = runner.start_photon
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(runner, "start_photon", lambda m, o, r: start(m, o, r)._replace(w=start(m, o, r).w * baked[1]))
        with pytest.raises(runner.ClosedOverValues, match="closes over"):
            runner.run(spec, model, ops, rows(), jax.random.key_data(jax.random.PRNGKey(0)), max_vertices=49,
                       backend="auto")  # fmt: skip


def test_simulate_compiles_once_per_power_of_two(setup, monkeypatch):
    """Launches, a source's row arrays and a step dump are padded to a power of two: varying samples (one chunk and
    several, with a new last-chunk count), FromArrays rows and CherenkovSteps steps inside it compiles nothing."""
    from lumen import sources

    det = setup[0]
    p = det.params()

    def listed(n):
        r = rows()
        return sources.FromArrays(*(np.resize(np.asarray(a), (n, *a.shape[1:])) for a in (r.x, r.d, r.t, r.lam, r.w)))

    def steps_of(n):
        a = np.tile([0.2, 0.0, 0.0], (n, 1)) + np.linspace(0.0, 0.1, n)[:, None] * [0.0, 0.0, 1.0]
        one = np.ones(n, np.float32)
        return sources.CherenkovSteps("Water", start=a, end=a + [0.0, 0.0, 0.01], t_start=0 * one, t_end=0.03 * one,
                                      beta_start=0.99 * one, beta_end=0.99 * one, charge=one)  # fmt: skip

    w = Watch(monkeypatch)

    def go(src, samples, chunk=1 << 20):
        return w(one, src, samples, chunk)

    def one(src, samples, chunk):
        return float(np.asarray(lumen.simulate(det, src, samples=samples, seed=1, chunk=chunk).arrivals).sum())

    def fixed(x):
        return w(fixed_one, x)

    def fixed_one(x):  # a fixed reference (simulate(reference=)) at a new concrete position and new values
        src = sources.Point((x, 0.0, 0.0), n_photons=1e3, lam=400.0)
        light = lumen.simulate(det, src, p.scale("materials/Water/ABSLENGTH", 1.0 + x), samples=3000, seed=1,
                               reference=p)  # fmt: skip
        return float(np.asarray(light.arrivals).sum())

    point = sources.Point((0.3, 0.0, 0.0), n_photons=1e3, lam=400.0)
    warm = [go(point, 3000), go(point, 3000, chunk=1024), go(listed(3000), 3000), go(steps_of(100), 3000), fixed(0.3)]
    w.start()
    again = [go(point, 2500), go(point, 4096), go(point, 2900, chunk=1024), go(listed(3500), 3500),
             go(listed(2100), 2100), go(steps_of(120), 2500), go(steps_of(65), 3000), fixed(0.25)]  # fmt: skip
    w.check()
    assert all(v > 0.0 for v in warm + again), (warm, again)


def test_a_new_seed_never_recompiles_a_derivative_run(setup, monkeypatch):
    """``jax.jvp`` through ``simulate`` (a family traced, with and without a fixed reference): the seed is an
    operand, so after the first call a new seed compiles and traces nothing (0.03 s a call on the toy, against 10 s for
    the first; 2026-11-18)."""
    from lumen import sources

    det = setup[0]
    p, key = det.params(), "surfaces/Liner/REFLECTIVITY"
    src = sources.Point((0.3, 0.0, 0.0), n_photons=1e3, lam=400.0)

    w = Watch(monkeypatch)

    def jvp(seed, reference=None):
        return w(one, seed, reference)

    def one(seed, reference):
        def f(s):
            return lumen.simulate(det, src, p.scale(key, s), samples=4096, seed=seed,
                                  reference=reference).arrivals  # fmt: skip

        return float(np.asarray(jax.jvp(f, (np.float32(1.0),), (np.float32(1.0),))[1]).sum())

    warm = [jvp(0), jvp(0, p)]
    w.start()
    again = [jvp(s) for s in (1, 2, 3)] + [jvp(s, p) for s in (1, 2)]
    w.check()
    assert all(np.isfinite(v) for v in warm + again) and len(set(again)) > 1, (warm, again)


def test_a_new_event_compiles_nothing(tmp_path, monkeypatch):
    """A new event is new step data in the same shapes: its sources (CherenkovSteps and ScintillationSteps, new
    tracks, the same step count) are traced values, never static or hashed, so after the first event the second
    compiles and traces nothing, and both lights move (the toy's water given Geant4 11 scintillation keys)."""
    from lumen import sources

    mats = {"SY": ("1", "1000"), "SC1": ("2", "2.0*eV 1 4.5*eV 1"), "ST1": ("1", "5")}
    names = {"SY": "SCINTILLATIONYIELD", "SC1": "SCINTILLATIONCOMPONENT1", "ST1": "SCINTILLATIONTIMECONSTANT1"}
    keys = "".join(f'  <matrix name="{k}" coldim="{c}" values="{v}"/>\n' for k, (c, v) in mats.items())
    props = "".join(f' <property name="{n}" ref="{k}"/>' for k, n in names.items())
    prop = '<property name="ABSLENGTH" ref="ABS0x2"/>'
    toy = _toy().replace('  <matrix name="GV"', keys + '  <matrix name="GV"').replace(prop, prop + props)
    det = lumen.load(lumen.convert(write(tmp_path, toy), RECIPE, tmp_path / "scint.lumen"))
    p = det.params()

    def event(k, n=100):  # a track of n steps along a new direction from a new point: new values, the same shapes
        g = np.random.default_rng(k)
        u = g.normal(size=3)
        a = (
            np.asarray([0.2, 0.0, 0.0])
            + 0.05 * g.uniform(-1, 1, 3)
            + np.linspace(0.0, 0.1, n + 1)[:, None] * u / np.linalg.norm(u)
        )
        one = np.ones(n, np.float32)
        kw = dict(
            start=a[:-1],
            end=a[1:],
            t_start=np.linspace(0.0, 0.3, n),
            t_end=np.linspace(0.003, 0.303, n),
        )
        cher = dict(beta_start=g.uniform(0.9, 0.99) * one, beta_end=g.uniform(0.9, 0.99) * one, charge=one)
        return (
            sources.CherenkovSteps("Water", **kw, **cher),
            sources.ScintillationSteps.of(p, "Water", edep=g.uniform(0.1, 0.3, n), **kw),
        )

    w = Watch(monkeypatch)

    def go(srcs):
        return [w(lambda s: float(np.asarray(lumen.simulate(det, s, samples=3000, seed=1).arrivals).sum()), s)
                for s in srcs]  # fmt: skip

    first = go(event(0))
    w.start()
    second = go(event(1))
    w.check()
    assert all(v > 0.0 for v in first + second) and first[0] != second[0] and first[1] != second[1], (first, second)


class _Total:
    """A records diagnostic of a plain class (equal only to itself): the run's detected weight, one cell."""

    def outputs(self):
        return {"total": (1, True)}

    def emits(self, dep, replica, weights=()):
        return [("total", jnp.where(dep.sensor >= 0, 0, -1), dep.q)]


def test_a_reload_compiles_nothing_and_leaves_nothing(tmp_path, monkeypatch):
    """The detector loaded afresh for every run, with a diagnostic built afresh: the jits see their structures
    (``lumen._core.static``), so after the first run nothing compiles, and no run's Model outlives its detector."""
    import gc

    from lumen import sources

    path = lumen.convert(write(tmp_path, _toy()), RECIPE, tmp_path / "toy.lumen")
    src = sources.Point((0.3, 0.0, 0.0), n_photons=1e3, lam=400.0)
    w = Watch(monkeypatch)

    def models():
        gc.collect()
        return sum(isinstance(o, steps.Model) for o in gc.get_objects())

    def run(seed):
        light = lumen.simulate(lumen.load(path), src, samples=3000, seed=seed, diagnostics=(_Total(),))
        return float(np.asarray(light.diagnostics["total"]).sum())

    before = models()
    w(run, 0)
    w.start()
    totals = [w(run, seed) for seed in (1, 2, 3)]
    w.check()
    assert models() == before and all(t > 0 for t in totals), (models(), before, totals)
