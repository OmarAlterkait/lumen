"""Polarisation: the Jones vector of a photon and how surfaces and emission set it.

A photon carries a complex Jones vector E = er + i ei (two real 3-vectors), perpendicular to its direction, with
|E| = 1, defined up to a global phase. E = 0 means unpolarised: a uniform mixture of linear states, drawn the first
time the state matters (``fill``). That is exact, since every observable is linear in the state's density matrix.

Conventions, against Geant4 10.7.4 (G4Cerenkov, G4Scintillation, G4OpWLS, G4OpRayleigh, G4OpBoundaryProcess):

- Emission: Cherenkov light is linear in the plane of the track and the photon (``cherenkov``); scintillation and
  wavelength shifting are unpolarised.
- Dielectric interfaces: Fresnel amplitudes (``lumen.surfaces.fresnel``) on the (s, p) basis, s = unit(wi x n),
  p = k x s for each of the incoming, reflected and transmitted k. Complex amplitudes keep the phases of total
  internal reflection and of absorbing media, which Geant4's real linear vector drops.
- A reflector without a complex index is the perfect mirror -E + 2 (E . n) n (``MIRROR``); a backscatter gives -E.
- Lambertian transmission and re-emission are unpolarised.
"""

import math

import jax
import jax.numpy as jnp

from lumen._core import frames
from lumen._core.safe import sanitize, unit

TWO_PI = 2.0 * math.pi
MIRROR = (-1.0 + 0.0j, 1.0 + 0.0j)  # (r_s, r_p) of a perfect mirror in this module's convention
ZERO_STATE = 0.5  # |E|^2 below this is the unpolarised state E = 0
NORMAL_INCIDENCE = 1e-14  # |wi x n|^2 below this: no plane of incidence
TINY_NORM = 1e-30  # |E|^2 below this normalises to 0


def dot(a, b):
    """Return the dot product over the last axis."""
    return jnp.sum(a * b, -1)


def norm2(er, ei):
    """Return |E|^2 of the Jones vector (er, ei)."""
    return dot(er, er) + dot(ei, ei)


def unpolarised(er, ei):
    """Return whether the state is the unpolarised one (E = 0)."""
    return norm2(er, ei) < ZERO_STATE


def random_linear(d, rand):
    """Return a linear state perpendicular to d at azimuth 2 pi rand; the frame is detached from d's derivative."""
    t1, t2 = frames.basis(jax.lax.stop_gradient(d))
    a = TWO_PI * rand
    return jnp.cos(a) * t1 + jnp.sin(a) * t2


def fill(d, er, ei, rand):
    """Return the state, with an unpolarised one (E = 0) replaced by a random linear state."""
    z = unpolarised(er, ei)
    return jnp.where(z, random_linear(d, rand), er), jnp.where(z, jnp.zeros_like(ei), ei)


def normalise(er, ei):
    """Return E / |E|; 0 stays 0 (unpolarised)."""
    s = norm2(er, ei)
    ok = s > TINY_NORM
    r = 1.0 / jnp.sqrt(sanitize(ok, s, 1.0))
    return jnp.where(ok, er * r, 0.0), jnp.where(ok, ei * r, 0.0)


def project(er, ei, k):
    """Return the part of E perpendicular to k, normalised (0 where nothing is left: the state is unpolarised)."""
    return normalise(er - dot(k, er)[..., None] * k, ei - dot(k, ei)[..., None] * k)


def mirror(er, ei, h):
    """Return -E + 2 (E . h) h: the reflection off a perfect mirror with normal h (either orientation)."""
    return -er + 2.0 * dot(er, h)[..., None] * h, -ei + 2.0 * dot(ei, h)[..., None] * h


def cherenkov(d, axis):
    """Return Geant4's Cherenkov polarisation: in the plane of the track axis and d, perpendicular to d.

    Notes
    -----
    G4Cerenkov: (cos cos phi, cos sin phi, -sin) about the track; the sign is immaterial.
    """
    return unit(dot(axis, d)[..., None] * d - axis)


def s_axis(wi, n, er):
    """Return the s axis unit(wi x n); at normal incidence a fixed perpendicular of wi, detached from wi."""
    c = jnp.cross(wi, n)
    ok = dot(c, c) > NORMAL_INCIDENCE
    t1, _ = frames.basis(jax.lax.stop_gradient(wi))
    return jnp.where(ok[..., None], unit(sanitize(ok[..., None], c, t1)), t1)


def components(er, ei, wi, s):
    """Return the complex (s, p) components (E_s, E_p) of the state, with p = wi x s."""
    p = jnp.cross(wi, s)
    return dot(er, s) + 1j * dot(ei, s), dot(er, p) + 1j * dot(ei, p)


def recombine(e_s, e_p, a_s, a_p, s, k):
    """Return the state a_s E_s s + a_p E_p (k x s), normalised, as (er, ei)."""
    p = jnp.cross(k, s)
    cs, cp = a_s * e_s, a_p * e_p
    er = jnp.real(cs)[..., None] * s + jnp.real(cp)[..., None] * p
    ei = jnp.imag(cs)[..., None] * s + jnp.imag(cp)[..., None] * p
    return normalise(er, ei)


def jones(er, ei, wi, n, k, amplitudes):
    """Return the state after an outcome with amplitudes (a_s, a_p) into direction k (n the facing normal)."""
    s = s_axis(wi, n, er)
    e_s, e_p = components(er, ei, wi, s)
    return recombine(e_s, e_p, amplitudes[0], amplitudes[1], s, k)


def fractions(er, ei, wi, n):
    """Return (|E_s|^2, |E_p|^2), which sum to |E|^2."""
    s = s_axis(wi, n, er)
    e_s, e_p = components(er, ei, wi, s)
    return jnp.abs(e_s) ** 2, jnp.abs(e_p) ** 2
