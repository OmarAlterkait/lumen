"""Toy PMT value sets and histograms for the readout tests (every component form, small grids, seconds on a CPU)."""

import hashlib
import json
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np

from lumen.readout import grid as tg
from lumen.sensors import value_sets


def toy_set(**over):
    """Return a value-set dict: a waveform readout with every component; ``over`` replaces top-level records."""
    d = dict(schema=value_sets.SCHEMA, model="lumen:sensor", name="toy", readout="waveform",
             provenance=dict(source="test toy", status="measured"),
             qe=dict(wavelength_nm=[300.0, 400.0, 500.0, 600.0], values=[0.1, 0.3, 0.2, 0.05]),
             ce=dict(theta_deg=[0, 30, 60, 90], values=[1.0, 0.9, 0.6, 0.2]),
             efficiency=0.9, t_offset_ns=5.0, tts=dict(form="gaussian", sigma_ns=1.2),
             charge=dict(form="gauss_exp", sigma_pe=0.3, fraction=0.2, tau_pe=0.4), gain=-20.0,
             threshold=dict(form="table", charge_pe=[0.0, 0.2, 0.5, 1.0], accept=[0.02, 0.1, 0.8, 0.98]),
             time_walk=dict(charge_pe=[0.5, 2.0, 8.0], shift_ns=[0.5, 0.2, 0.0], sigma_ns=[0.6, 0.3, 0.1]),
             pulse=dict(form="bigauss", t_peak_ns=6.0, sigma_rise_ns=1.5, sigma_fall_ns=4.0), dark_rate_hz=2e6,
             frontend=dict(sampling_ns=2.0, unit="ADC", baseline=100.0, noise=1.5, window_ns=200.0,
                           charge_noise_pe=0.05, charge_scale=1.0, charge_step_pe=1.0 / 32, charge_max_pe=32.0),
             ranges={"tts/tau": [0.0, 3.0], "tts/shift": [-1.0, 1.0], "t_offset": [0.0, 10.0],
                     "pulse/t_peak": [4.0, 8.0], "pulse/sigma_rise": [1.0, 3.0], "pulse/sigma_fall": [2.0, 6.0],
                     "pulse/t_start": [0.0, 2.0], "pulse/tau_rise": [0.5, 2.0],
                     "pulse/tau_fall": [3.0, 8.0]})  # fmt: skip
    d.update(over)
    if "ranges" not in over:  # the ranges of the forms chosen (a range naming no value is refused)
        has = lambda c, f: isinstance(d.get(c), dict) and any(f + u in d[c] for u in ("", "_ns", "_pe"))  # noqa: E731
        d["ranges"] = {k: v for k, v in d["ranges"].items() if "/" not in k or has(*k.split("/"))}
    return d


def toy_file(**over) -> str:
    """Return the path of a value-set file holding ``toy_set(**over)``, for a sensor group's ``defaults`` (one file per
    content, in the temporary directory)."""
    text = json.dumps(toy_set(**over), indent=1) + "\n"
    d = Path(tempfile.gettempdir()) / "lumen-test-values"
    d.mkdir(exist_ok=True)
    path = d / f"{hashlib.sha256(text.encode()).hexdigest()[:16]}.json"
    if not path.exists():
        tmp = d / f".{path.name}.{os.getpid()}"
        tmp.write_text(text)
        os.replace(tmp, path)  # whole: another worker may write the same file
    return str(path)


TOY_SET = toy_file()  # the value set the tests' modelled groups read


def model(**over):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return value_sets.model_from_dict(json.loads(json.dumps(toy_set(**over))))


def photons(n, n_sensors=3, t0=20.0, spread=0.0, seed=0):
    """Return a photon list: random sensors, wavelengths in 320-580 nm, crossing points on the unit sphere at cathode
    angles ``hit`` in 0-80 deg (``pos``), times t0 (+ spread)."""
    rng = np.random.default_rng(seed)
    sensor, lam = rng.integers(0, n_sensors, n), rng.uniform(320.0, 580.0, n)  # the draws in today's order
    hit = np.radians(rng.uniform(0.0, 80.0, n))
    return dict(sensor=sensor, lam=lam, hit=hit, pos=np.c_[np.sin(hit), np.zeros(n), np.cos(hit)],
                time=t0 + spread * rng.standard_normal(n), weight=rng.uniform(0.5, 1.5, n))  # fmt: skip


AXIS = np.array([[0.0, 0.0, 1.0]])  # one crossing point on the sensor's axis (cathode angle 0)


def hist(m, n=400, n_sensors=3, gate=(0.0, 120.0), **kw):
    return tg.from_photons(m, gate, n_sensors, photons(n, n_sensors, **kw), group="pmt")


def params(m, n_sensors=3, group="pmt"):
    return {k: np.asarray(v, np.float32) for k, v in m.defaults(group, n_sensors).items()}
