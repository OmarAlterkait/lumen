"""Fresnel amplitudes and reflectances at an interface between two media.

The amplitudes are complex, in the convention of ``lumen.surfaces.polar`` (p = k x s), so a second medium may
absorb (n2 = nr + i nk: a photocathode or a metal) and total internal reflection keeps its phases. Every function
has a finite value and derivative at normal and grazing incidence, at the critical angle and at n1 = n2.
"""

import jax.numpy as jnp

from lumen.rules.choices import FRESNEL_SMALL


def decaying_sqrt(z):
    """Return sqrt(z) with Im >= 0, and 0 near z = 0 with finite derivatives.

    The transmitted wave decays: total internal reflection's evanescent wave, or an absorbing medium with nk > 0.
    One sign for every photon, so successive phases compose.
    """
    small = jnp.abs(z) < FRESNEL_SMALL
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
    ok_s, ok_p = jnp.abs(ds) > FRESNEL_SMALL, jnp.abs(dp) > FRESNEL_SMALL
    ds, dp = jnp.where(ok_s, ds, 1.0), jnp.where(ok_p, dp, 1.0)
    rs = jnp.where(ok_s, (n1c * c1 - q) / ds, -1.0)
    rp = jnp.where(ok_p, (n2c * n2c * c1 - n1c * q) / dp, 1.0)
    ts = jnp.where(ok_s, 2.0 * n1c * c1 / ds, 0.0)
    tp = jnp.where(ok_p, 2.0 * n1c * n2c * c1 / dp, 0.0)
    return rs, rp, ts, tp


def reflectance(fs, fp, rs, rp):
    """Return |r_s|^2 f_s + |r_p|^2 f_p: the reflectance of a state with (s, p) fractions (f_s, f_p)."""
    return jnp.abs(rs) ** 2 * fs + jnp.abs(rp) ** 2 * fp


def unpolarised_absorbing_reflectance(cos_i, n1, nr, nk):
    """Return the unpolarised reflectance from a real medium n1 into an absorbing one nr + i nk (nk >= 0).

    The amplitudes' (``amplitudes``: q with Im q >= 0, the decaying wave), so nk = 0 is the dielectric's, its total
    reflection included, with its derivative in nk.
    """
    rs, rp, _, _ = amplitudes(cos_i, n1, nr + 1j * nk)
    return jnp.clip(0.5 * (jnp.abs(rs) ** 2 + jnp.abs(rp) ** 2), 0.0, 1.0)
