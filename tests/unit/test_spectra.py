"""Emission spectra: Geant4's WLS / scintillation sampler is the literal Geant4 inverse; its density is 0 outside the
table."""

import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import ks_2samp

from lumen.media import spectra
from lumen.tools import stats
from lumen.units import HC

F32 = jnp.float32


def geant4_sampler(energy, intensity, u):
    """G4OpWLS::BuildPhysicsTable (trapezoid integral at the knots) and G4PhysicsVector::GetEnergy (linear inverse)."""
    cii = np.concatenate([[0.0], np.cumsum(0.5 * np.diff(energy) * (intensity[1:] + intensity[:-1]))])
    v = u * cii[-1]
    b = np.clip(np.searchsorted(cii, v, side="left") - 1, 0, len(cii) - 2)
    d = cii[b + 1] - cii[b]
    return energy[b] + np.where(d > 0, (v - cii[b]) * (energy[b + 1] - energy[b]) / np.where(d > 0, d, 1), 0.0)


def chi2_two(a, b, edges):
    ha, hb = np.histogram(a, edges)[0].astype(float), np.histogram(b, edges)[0].astype(float)
    ka, kb = math.sqrt(hb.sum() / ha.sum()), math.sqrt(ha.sum() / hb.sum())
    ok = ha + hb > 0
    return float(np.sum((ka * ha[ok] - kb * hb[ok]) ** 2 / (ha[ok] + hb[ok])) / max(ok.sum() - 1, 1))


def test_geant4_spectrum_sampler_is_geant4s():
    energy = np.array([2.5, 2.6, 2.75, 2.9, 3.0, 3.1, 3.4])
    intensity = np.array([0.0, 0.3, 1.0, 0.7, 0.4, 0.2, 0.0])
    sp = spectra.geant4_spectrum("shifter", energy, intensity)
    u = np.asarray(jax.random.uniform(jax.random.PRNGKey(0), (400_000,), F32, 1e-7, 1 - 1e-7))
    E = HC / np.asarray(spectra.sample(spectra.g4_arrays(sp.energy, sp.intensity), jnp.asarray(u)), np.float64)
    Eg = geant4_sampler(energy, intensity, np.random.default_rng(1).random(400_000))
    grid = np.linspace(2.5, 3.4, 9001)
    assert chi2_two(E, Eg, np.linspace(2.5, 3.4, 61)) < 1.6
    assert abs(np.trapezoid(np.asarray(spectra.density(energy, intensity, grid)), grid) - 1) < 2e-3
    assert float(spectra.density(energy, intensity, 3.5)) == 0.0 == float(spectra.density(energy, intensity, 2.4))
    assert E.min() >= 2.5 - 1e-4 and E.max() <= 3.4 + 1e-4


def test_wls_draw_conserves_energy_as_g4opwls():
    """The energy-conserving draw (the spectrum truncated at the absorbed energy by its inverse CDF, weighted by
    ``kept``, as ``lumen.media.emission`` draws it) is G4OpWLS's redraw loop: the truncated spectrum, and the chance
    that one of 100 draws lands there (else the photon is dropped)."""
    spec = spectra.geant4_spectrum("g4", [2.4, 2.8, 3.0, 3.3, 3.5], [0.0, 1.0, 0.7, 1.2, 0.0])
    arrays = spectra.g4_arrays(spec.energy, spec.intensity)
    lo, hi = HC / spec.energy[-1], HC / spec.energy[0]  # the wavelengths the table spans
    rng = np.random.default_rng(5)
    for lam_in in (lo + 0.3 * (hi - lo), 0.5 * (lo + hi), 300.0, 2000.0):
        # Geant4: redraw until E <= E_in, up to 100 times; a failed photon is dropped
        u = rng.uniform(size=(100, 40_000)).astype(np.float32)
        draws = np.asarray(spectra.sample(arrays, jnp.asarray(u)), np.float64)
        ok = draws >= lam_in
        found = ok.any(0)
        first = draws[ok.argmax(0), np.arange(draws.shape[1])][found]
        f = spectra.cumulative(arrays, F32(lam_in))
        lam = spectra.sample(arrays, jnp.asarray(rng.uniform(size=40_000), F32) * f)
        kept = spectra.kept(f)
        lam = np.asarray(lam, np.float64)
        sd = np.sqrt(found.mean() * (1 - found.mean()) / 40_000)  # binomial; 0 where every or no photon is found
        assert abs(float(np.ravel(kept)[0]) - found.mean()) <= stats.z_limit(4) * sd
        assert float(np.ravel(kept)[0]) == 0.0 or np.all(lam >= lam_in * (1 - 1e-5))  # kept 0: a dropped photon
        if found.sum() > 1000:
            assert ks_2samp(lam, first).pvalue > stats.ALPHA / stats.SUITE_TESTS  # the two samples' laws
