"""Toy PMT value sets and histograms for the readout tests (every component form, small grids, seconds on a CPU)."""

import json
import warnings

import numpy as np

from lumen import readout as rd
from lumen.sensors import pmt


def toy_set(**over):
    """Return a value-set dict: a waveform readout with every component; ``over`` replaces top-level records."""
    d = dict(schema=pmt.SCHEMA, model="lumen:pmt", model_version=1, name="toy", version=1, readout="waveform",
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
             ranges={"tts/sigma": [0.8, 4.0], "tts/tau": [0.0, 3.0], "tts/shift": [-1.0, 1.0], "t_offset": [0.0, 10.0],
                     "pulse/t_peak": [4.0, 8.0], "pulse/sigma_rise": [1.0, 3.0], "pulse/sigma_fall": [2.0, 6.0],
                     "pulse/t_start": [0.0, 2.0], "pulse/tau_rise": [0.5, 2.0],
                     "pulse/tau_fall": [3.0, 8.0]})  # fmt: skip
    d.update(over)
    return d


def model(**over):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pmt.model_from_dict(json.loads(json.dumps(toy_set(**over))))


def photons(n, n_sensors=3, t0=20.0, spread=0.0, seed=0):
    """Return a photon list: random sensors, wavelengths in 320-580 nm, hit angles in 0-80 deg, times t0 (+ spread)."""
    rng = np.random.default_rng(seed)
    return dict(sensor=rng.integers(0, n_sensors, n), lam=rng.uniform(320.0, 580.0, n),
                hit=np.radians(rng.uniform(0.0, 80.0, n)), time=t0 + spread * rng.standard_normal(n),
                weight=rng.uniform(0.5, 1.5, n))  # fmt: skip


def hist(m, n=400, n_sensors=3, gate=(0.0, 120.0), replicas=4, **kw):
    return rd.from_photons(m, gate, n_sensors, photons(n, n_sensors, **kw), replicas=replicas)


def params(m, n_sensors=3, group="pmt"):
    return {k: np.asarray(v, np.float32) for k, v in m.defaults(group, n_sensors).items()}
