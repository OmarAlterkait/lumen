"""Fresnel amplitudes and reflectances at an interface between two media.

The amplitudes are complex, in the convention of ``lumen.surfaces.polar`` (p = k x s), so a second medium may
absorb (n2 = nr + i nk: a photocathode or a metal) and total internal reflection keeps its phases. Every function
has a finite value and derivative at normal and grazing incidence, at the critical angle and at n1 = n2.
"""

import jax.numpy as jnp

from lumen._core.safe import safe_sqrt, sanitize
from lumen.surfaces import polar

SMALL = 1e-12  # |z| below this is 0 in the square root and the denominators
NK_FLOOR = 1e-6  # the absorption index used where an absorbing medium is given nk <= 0


def decaying_sqrt(z):
    """Return sqrt(z) with Im >= 0, and 0 near z = 0 with finite derivatives.

    The transmitted wave decays: total internal reflection's evanescent wave, or an absorbing medium with nk > 0.
    One sign for every photon, so successive phases compose.
    """
    small = jnp.abs(z) < SMALL
    q = jnp.sqrt(jnp.where(small, 1.0 + 0.0j, z))
    q = jnp.where(jnp.imag(q) < 0.0, -q, q)
    return jnp.where(small, 0.0 + 0.0j, q)


def amplitudes(cos_i, n1, n2):
    """Return the complex amplitudes (r_s, r_p, t_s, t_p) from n1 into n2 (n2 may be complex) at incidence cos_i.

    Notes
    -----
    Notation: c = cos_i, q = n2 cos_t = sqrt(n2^2 - n1^2 sin_i^2) with Im q >= 0. Then
    r_s = (n1 c - q) / (n1 c + q), t_s = 2 n1 c / (n1 c + q), r_p = (n2^2 c - n1 q) / (n2^2 c + n1 q) and
    t_p = 2 n1 n2 c / (n2^2 c + n1 q). Beyond the critical angle q is imaginary: |r| = 1 with the phases of total
    internal reflection (t is then not used).
    """
    c1 = jnp.clip(cos_i, 0.0, 1.0).astype(jnp.complex64)
    n1c, n2c = jnp.asarray(n1).astype(jnp.complex64), jnp.asarray(n2).astype(jnp.complex64)
    q = decaying_sqrt(n2c * n2c - n1c * n1c * (1.0 - c1 * c1))
    ds, dp = n1c * c1 + q, n2c * n2c * c1 + n1c * q
    ok_s, ok_p = jnp.abs(ds) > SMALL, jnp.abs(dp) > SMALL
    ds, dp = jnp.where(ok_s, ds, 1.0), jnp.where(ok_p, dp, 1.0)
    rs = jnp.where(ok_s, (n1c * c1 - q) / ds, -1.0)
    rp = jnp.where(ok_p, (n2c * n2c * c1 - n1c * q) / dp, 1.0)
    ts = jnp.where(ok_s, 2.0 * n1c * c1 / ds, 0.0)
    tp = jnp.where(ok_p, 2.0 * n1c * n2c * c1 / dp, 0.0)
    return rs, rp, ts, tp


def reflectance(fs, fp, rs, rp):
    """Return |r_s|^2 f_s + |r_p|^2 f_p: the reflectance of a state with (s, p) fractions (f_s, f_p)."""
    return jnp.abs(rs) ** 2 * fs + jnp.abs(rp) ** 2 * fp


def unpolarised_reflectance(cos_i, n1, n2):
    """Return the unpolarised Fresnel reflectance from n1 into n2 at incidence cos_i; 1 beyond the critical angle."""
    eta = n1 / n2
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    tir = sin2_t >= 1.0
    # under total internal reflection (grazing incidence on an index-matched pair included: 0 / 0) the ratios
    # are not evaluated
    ci = sanitize(~tir, cos_i, 1.0)
    ct = sanitize(~tir, safe_sqrt(1.0 - sin2_t), 1.0)
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)
    total = 0.5 * (rs * rs + rp * rp)
    return jnp.where(tir, 1.0, jnp.clip(total, 0.0, 1.0))


def unpolarised_absorbing_reflectance(cos_i, n1, nr, nk):
    """Return the unpolarised reflectance from a real medium n1 into an absorbing one nr + i nk.

    nk = 0 needs the dielectric formula (its total-reflection branch); an nk <= 0 here is replaced by NK_FLOOR.
    """
    nc = nr + 1j * sanitize(nk > 0, nk, NK_FLOOR)
    ci = cos_i.astype(jnp.complex64)
    ct = jnp.sqrt(1.0 - (n1 / nc) ** 2 * (1.0 - ci * ci))
    rs = (n1 * ci - nc * ct) / (n1 * ci + nc * ct)
    rp = (nc * ci - n1 * ct) / (nc * ci + n1 * ct)
    return jnp.clip(0.5 * (jnp.abs(rs) ** 2 + jnp.abs(rp) ** 2), 0.0, 1.0)


def state_reflectance(wi, n, n1, n2, er, ei):
    """Return the dielectric Fresnel reflectance of the photon's own state (er, ei); 1 beyond the critical angle."""
    cos_i = jnp.clip(-jnp.sum(wi * n, -1), 0.0, 1.0)
    eta = n1 / n2
    tir = eta * eta * (1.0 - cos_i * cos_i) >= 1.0
    rs, rp, _, _ = amplitudes(sanitize(~tir, cos_i, 1.0), n1, n2)
    fs, fp = polar.fractions(er, ei, wi, n)
    return jnp.where(tir, 1.0, jnp.clip(reflectance(fs, fp, rs, rp), 0.0, 1.0))
