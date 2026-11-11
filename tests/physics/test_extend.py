"""The extension API's derivatives (tier b): each extension point, AD against FD with ``lumen.validate.check_gradient``.

All on the shipped toy, ``examples/data/example_tank.gdml`` (a 1 m water cube, 8 disc PMTs, Rayleigh from Geant4's
water formula, lumen:pmt with the tutorial's values), converted once per module. Every derivative is the forward mode
against the finite difference on the same random numbers, over ``SEEDS`` seeds, judged by ``lumen.validate.stats``'s
rules (``GradientReport.passes``). The worked examples of docs/howto/extending.md are ``examples/extending.py``'s
functions, tested here as they are written there.
"""

import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

import lumen
from lumen import sources
from lumen.validate import check_gradient

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
F32 = jnp.float32
SEEDS = 8
N = 1 << 18  # photons per side and seed
PMT7 = 7  # the PMT at (0.25, 0, -0.48), off the LED's axis


def _module(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tank(tmp_path_factory):
    """The tutorial's tank, converted with its own recipe."""
    et = _module("example_tank")
    return lumen.load(lumen.convert(et.GDML, et.RECIPE, tmp_path_factory.mktemp("tank") / "tank.lumen"))


def arrivals(det, src, params=None, seed=0, wrt=(), samples=N):
    return lumen.simulate(det, src, params, samples=samples, seed=seed, wrt=wrt).arrivals


# ------------------------------------------------------------------------------------------------- check_gradient
def test_check_gradient_on_a_built_in_point_position(tank):
    """The Point's position derivative (moving-source method), the procedure every test below uses."""

    def f(dx, seed):
        src = sources.Point(jnp.stack([dx, F32(0.0), F32(0.2)]), n_photons=float(N), lam=420.0)
        return arrivals(tank, src, seed=seed, wrt=("source/position",))

    r = check_gradient(f, 0.0, seeds=SEEDS, h=0.01, stencil="central")
    assert r.passes, r

