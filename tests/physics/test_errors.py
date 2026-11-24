"""The quoted Monte Carlo error of the arrivals against the spread over seeds, on the example tank (two-sided)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

import lumen

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
SEEDS = 64


def test_the_quoted_error_is_the_spread_over_seeds():
    """Mean variance ratio over the 8 PMTs: spread^2 / quoted^2 is 1 with error sqrt(2 / (63 * 8)) = 0.063 for
    independent sensors; 4 of those is the tolerance, so a quoted error 20 % too large or too small fails."""
    spec = importlib.util.spec_from_file_location("example_tank", EXAMPLES / "example_tank.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["example_tank"] = mod
    spec.loader.exec_module(mod)
    det = mod.detector()
    src = lumen.sources.Point(position=(0.0, 0.0, 0.1), lam=400.0, n_photons=2e5)
    runs = [lumen.simulate(det, src, samples=2**14, seed=1000 + i) for i in range(SEEDS)]
    a = np.array([np.asarray(r.arrivals, np.float64) for r in runs])
    e = np.array([np.asarray(r.arrivals_error, np.float64) for r in runs])
    ratio = np.mean(a.var(0, ddof=1) / (e**2).mean(0))
    assert abs(ratio - 1.0) < 0.25, ratio
