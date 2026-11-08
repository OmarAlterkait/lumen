"""Emission spectra: the wavelength sampler follows its spectrum, and Geant4's WLS / scintillation sampler is the
literal Geant4 inverse."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lumen.media import spectra
from lumen.units import HC

F32 = jnp.float32
TPB = spectra.wavelength_spectrum("tpb_like", [380, 400, 420, 440, 470, 520], [0.0, 0.6, 1.0, 0.8, 0.3, 0.0])


def chi2_vs_spectrum(lam, r, n_bins=25):
    edges = np.linspace(r.wl[0], r.wl[-1], n_bins + 1)
    grid = np.linspace(r.wl[0], r.wl[-1], 20001)
    dens = r.density_at(grid)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    exp = len(lam) * np.diff(np.interp(edges, grid, cdf))
    obs = np.histogram(lam, edges)[0]
    ok = exp > 20
    return float(np.sum((obs[ok] - exp[ok]) ** 2 / exp[ok]) / (ok.sum() - 1))


def test_wavelength_sampler_follows_its_spectrum():
    u = jax.random.uniform(jax.random.PRNGKey(0), (400_000,), F32, 1e-7, 1 - 1e-7)
    lam = np.asarray(jax.vmap(TPB.sample)(u), np.float64)
    assert lam.min() >= 380 and lam.max() <= 520
    assert chi2_vs_spectrum(lam, TPB) < 1.6
    grid = np.linspace(381.0, 519.0, 50)
    back = np.asarray(jax.vmap(TPB.sample)(jnp.asarray(TPB.rand_of(grid), F32)), np.float64)
    np.testing.assert_allclose(back, grid, atol=2e-3)  # rand_of is the inverse of sample


def test_wavelength_spectrum_refuses_bad_tables():
    with pytest.raises(ValueError):
        spectra.wavelength_spectrum("bad", [400.0, 390.0], [1.0, 1.0])


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
    E = HC / np.asarray(jax.vmap(sp.sample)(jnp.asarray(u)), np.float64)
    Eg = geant4_sampler(energy, intensity, np.random.default_rng(1).random(400_000))
    grid = np.linspace(2.5, 3.4, 9001)
    assert chi2_two(E, Eg, np.linspace(2.5, 3.4, 61)) < 1.6
    assert abs(np.trapezoid(sp.density_energy(grid), grid) - 1) < 2e-3
    assert E.min() >= 2.5 - 1e-4 and E.max() <= 3.4 + 1e-4
