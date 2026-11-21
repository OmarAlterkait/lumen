"""A part of your own passed at load: a phase function for one material's scattering channel (a hook of 0.2).

The user's lobe is the Rayleigh phase function of an anisotropic molecule, p(mu) = (1 + a mu^2) / (4 pi (1 + a / 3))
per steradian (a = (1 - g) / (1 + 3 g), g = rho / (2 - rho) from the depolarisation ratio rho; a = 1 is Rayleigh's
1 + mu^2), drawn as the mixture "uniform in mu" with probability 1 / (1 + a / 3), else p(mu) = 3 mu^2 / 2 (mu =
cbrt(2u - 1)): 3 uniforms. It is user code: it imports lumen and nothing private. The water of the guide's lined tank
gets it for its Rayleigh channel, passed at load; the Rayleigh length is scaled to about a metre so the derivative in
``a`` is resolved. Measured numbers are printed (run with -s). The lobe has no polarised pair: it says so
(``unpolarised=True``), and a run records it, as it records that the fast walk is off in that water.
"""

import importlib.util
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lumen
from lumen import Param, sources
from lumen.sources import cone_about
from lumen.tools import check_gradient

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
F32 = jnp.float32
N = 1 << 18
SEEDS = 16
KEY = "materials/Water/rayleigh/a"  # a part's keys are under its channel
SHORT = 1.0  # the Rayleigh table's scale (the fixture's table is 1 m flat)


def _module(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sample(p, wi, n, rand):
    a = p["a"]
    flat = rand[..., 0] * (1.0 + a / 3.0) < 1.0
    u = 2.0 * rand[..., 1] - 1.0
    mu = jnp.where(flat, u, jnp.sign(u) * jnp.abs(u) ** (1.0 / 3.0))
    return cone_about(wi, mu[None], (2.0 * math.pi * rand[..., 2])[None])[0]


def _logpdf(p, wi, n, wo):
    mu = jnp.sum(wo * wi, -1)
    return jnp.log1p(p["a"] * mu * mu) - jnp.log(4.0 * math.pi * (1.0 + p["a"] / 3.0))


DEPOL = lumen.media.phase_function("mylab:rayleigh_depol", params={"a": Param("", (0.0, 3.0), default=0.83)},
                                   n_rand=3, sample=_sample, logpdf=_logpdf, unpolarised=True)  # fmt: skip


@pytest.fixture(scope="module")
def lined(tmp_path_factory):
    """The guide's lined tank, its water given a RAYLEIGH table of 1 m (a key, so it can be scaled)."""
    et = _module("example_tank")  # extending.lined_tank imports it
    ex = _module("extending")
    d = tmp_path_factory.mktemp("lined")
    text = (
        et.GDML.read_text()
        .replace(
            "</define>", '  <matrix name="WATER_RAYLEIGH" coldim="2" values="1.5*eV 1.0*m 5.0*eV 1.0*m"/>\n</define>'
        )
        .replace(
            '<property name="ABSLENGTH" ref="WATER_ABSLENGTH"/>',
            '<property name="ABSLENGTH" ref="WATER_ABSLENGTH"/><property name="RAYLEIGH" ref="WATER_RAYLEIGH"/>',
        )
    )
    (d / "tank_r.gdml").write_text(text)
    et.GDML = d / "tank_r.gdml"
    return ex.lined_tank(d)


def test_a_phase_function_passed_at_load(lined):
    """The part binds to one channel of one material; its parameter is a key; the rest of the detector is unchanged."""
    t0 = time.perf_counter()
    det = lumen.load(lined, parts={"materials/Water/rayleigh": DEPOL})
    print(f"\nload with the part: {time.perf_counter() - t0:.2f} s (includes the sampler-vs-density chi2 check)")
    p = det.params()
    assert p[KEY] == pytest.approx(0.83)
    assert det.file.materials["Water"].optics.name == "mylab:rayleigh_depol@rayleigh"
    with pytest.raises(KeyError):  # a channel the material does not have
        lumen.load(lined, parts={"materials/Water/mie": DEPOL})
    with pytest.raises(ValueError, match="two parts"):
        lumen.load(lined, parts={"materials/Water/rayleigh": DEPOL, "materials/Water": "geant4"})
    with pytest.raises(ValueError, match="unpolarised=True"):  # no polarised pair, not declared
        lumen.media.phase_function("mylab:x", params={}, n_rand=3, sample=_sample, logpdf=_logpdf)
    src = sources.Point((0.05, 0.02, 0.1), n_photons=1024.0, lam=420.0)
    for speed, notes in (("fast", ("unpolarised", "fast walk off")), ("exact", ("unpolarised",))):
        prov = dict(lumen.simulate(det, src, samples=1024, seed=0, speed=speed).provenance)
        assert prov["parts"] == (("materials/Water/rayleigh", "mylab:rayleigh_depol@rayleigh", notes),), speed
    assert "parts" not in dict(lumen.simulate(lumen.load(lined), src, samples=1024, seed=0).provenance)


def test_the_parameter_derivative_ad_vs_fd(lined):
    """AD (forward mode, draws at the reference, weights move) against FD on the same seeds, and per seed the exactness
    against a difference with the draws held (``simulate(reference=)``)."""
    det = lumen.load(lined, parts={"materials/Water/rayleigh": DEPOL})
    p0 = det.params().scale("materials/Water/RAYLEIGH", SHORT)
    src = sources.Point((0.05, 0.02, 0.1), n_photons=float(N), lam=420.0)
    theta0 = 0.83

    def f(theta, seed, reference=None):
        return lumen.simulate(det, src, p0.set(KEY, theta), samples=N, seed=seed, reference=reference).arrivals

    r = check_gradient(f, theta0, seeds=SEEDS, h=0.4, stencil="central")  # the light moves ~0.3 % per unit a
    print("\n" + str(r))
    print("per PMT AD   ", np.array2string(r.ad[:-1], precision=1))
    print("per PMT FD   ", np.array2string(r.fd[:-1], precision=1))
    print("per PMT z    ", np.array2string(r.z[:-1], precision=2))
    print("resolved     ", r.resolved)
    assert r.passes is not False, r
    for seed in (0, 1):  # exactness: the derivative of the light with every draw held at the reference
        ad = float(jax.jvp(lambda t, s=seed: f(t, s).sum(), (F32(theta0),), (F32(1.0),))[1])
        h = 0.05  # float32 totals of ~1e4: a smaller step measures their rounding
        fd = (float(f(F32(theta0 + h), seed, p0).sum()) - float(f(F32(theta0 - h), seed, p0).sum())) / (2 * h)
        print(f"seed {seed}: AD {ad:+.4f}  FD with draws held (h {h}) {fd:+.4f}  rel {abs(ad - fd) / abs(ad):.2e}")
        assert abs(ad - fd) <= 1e-2 * abs(ad), (seed, ad, fd)
