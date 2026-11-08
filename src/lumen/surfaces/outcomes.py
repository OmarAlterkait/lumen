"""The ten surface outcomes, and the deterministic ones' direction and Jones maps, defined once.

Every surface model produces probabilities over the same fixed outcomes, with these indices:

===========  =========  ==========  ===================================================
index        drawn by   photon on   direction
===========  =========  ==========  ===================================================
STRAIGHT     map        far side    unchanged (index-matched or counting-only surfaces)
REFRACT      map        far side    Snell refraction (the mirror under total reflection)
SPECULAR     map        near side   the mirror direction
BACKSCATTER  map        near side   reversed (Geant4 unified BACKSCATTERCONSTANT)
GLOSSY       lobe       near side   the surface's facet lobe (Geant4 unified or GLISUR)
DIFFUSE      lobe       near side   Lambertian reflection
DIFFUSE_T    lobe       far side    Lambertian transmission (diffusers)
DETECT       terminal   -           the light is detected at the sensor
ABSORB       terminal   -           the light is absorbed
REEMIT       lobe       near side   wavelength-shifted re-emission
===========  =========  ==========  ===================================================

The photon step, and the moving-source method's direction of a path segment, use these maps and never a copy. A
drawn outcome keeps the incoming direction here: its lobe draws the new one.
"""

import jax
import jax.numpy as jnp

from lumen._core import frames
from lumen._core.safe import safe_sqrt, unit
from lumen.surfaces import polar

STRAIGHT, REFRACT, SPECULAR, BACKSCATTER, GLOSSY, DIFFUSE, DIFFUSE_T, DETECT, ABSORB, REEMIT = range(10)
N_OUTCOMES = 10
NAMES = (
    "straight",
    "refract",
    "specular",
    "backscatter",
    "glossy",
    "diffuse",
    "diffuse_t",
    "detect",
    "absorb",
    "reemit",
)
INDEX = {name: i for i, name in enumerate(NAMES)}
DELTA = (True, True, True, True, False, False, False, False, False, False)  # direction by a deterministic map
FAR_SIDE = (True, True, False, False, False, False, True, False, False, False)  # the photon ends beyond the surface
TERMINAL = (False, False, False, False, False, False, False, True, True, False)


def refract(wi, n, eta):
    """Return Snell's refraction of wi at a surface with facing normal n (wi . n < 0), eta = n1 / n2.

    Returns
    -------
    wo : array (..., 3)
        The refracted direction, or the mirror direction under total internal reflection.
    tir : bool array (...)
        Total internal reflection.
    """
    cos_i = -jnp.sum(wi * n, -1, keepdims=True)
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    tir = sin2_t[..., 0] >= 1.0
    cos_t = safe_sqrt(1.0 - sin2_t)
    wt = unit(eta * wi + (eta * cos_i - cos_t) * n)
    wr = wi + 2.0 * cos_i * n
    return jnp.where(tir[..., None], wr, wt), tir


def direction_rules(wi, n, eta):
    """Return one direction rule (a thunk) per outcome for an incoming wi at facing normal n, eta = n1 / n2."""
    rules = [lambda: wi] * N_OUTCOMES
    rules[REFRACT] = lambda: refract(wi, n, eta)[0]
    rules[SPECULAR] = lambda: frames.reflect(wi, n)
    rules[BACKSCATTER] = lambda: -wi
    return rules


def direction(outcome, wi, n, eta):
    """Return the direction after an outcome (a traced index)."""
    return jax.lax.switch(outcome, [lambda _, f=f: f() for f in direction_rules(wi, n, eta)], None)


def jones_rules(components, amplitudes, s_axis, field, wo):
    """Return one Jones-map rule (a thunk) per outcome.

    ``components`` are the state's complex (s, p) components on ``s_axis``, ``amplitudes`` the surface's
    (r_s, r_p, t_s, t_p), ``field`` the Jones vector (er, ei) and ``wo`` the outgoing direction.

    Notes
    -----
    REFRACT takes the transmitted state, SPECULAR the reflected one on the outgoing basis, BACKSCATTER the reversed
    field; every other outcome keeps the state.
    """
    e_s, e_p = components
    er, ei = field
    rs, rp, ts, tp = amplitudes
    maps = [lambda: (er, ei)] * N_OUTCOMES
    maps[REFRACT] = lambda: polar.recombine(e_s, e_p, ts, tp, s_axis, wo)
    maps[SPECULAR] = lambda: polar.recombine(e_s, e_p, rs, rp, s_axis, wo)
    maps[BACKSCATTER] = lambda: (-er, -ei)
    return maps
