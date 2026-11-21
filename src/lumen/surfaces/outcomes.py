"""The six continuing surface outcomes, and the deterministic ones' direction and Jones maps, defined once.

Every surface model produces its continuing shares over the same fixed outcomes, with these indices (the loss is 1 -
their sum), and its detected share, a scalar deposit, separately (a model may deposit a share and let the photon go
on):

===========  =========  ==========  ===================================================
index        drawn by   photon on   direction
===========  =========  ==========  ===================================================
STRAIGHT     map        far side    unchanged (index-matched or counting-only surfaces)
REFRACT      map        far side    Snell refraction (the mirror under total reflection)
SPECULAR     map        near side   the mirror direction
BACKSCATTER  map        near side   reversed (Geant4 unified BACKSCATTERCONSTANT)
GLOSSY       lobe       near side   the surface's facet lobe (Geant4 unified or GLISUR)
DIFFUSE      lobe       near side   Lambertian reflection
===========  =========  ==========  ===================================================

The photon step uses these maps and never a copy. A
drawn outcome keeps the incoming direction here: its lobe draws the new one.
"""

import jax.numpy as jnp

from lumen._core import frames
from lumen._core.safe import safe_sqrt, unit
from lumen.surfaces import polar

STRAIGHT, REFRACT, SPECULAR, BACKSCATTER, GLOSSY, DIFFUSE = range(6)
N_OUTCOMES = 6
NAMES = ("straight", "refract", "specular", "backscatter", "glossy", "diffuse")
INDEX = {name: i for i, name in enumerate(NAMES)}
DELTA = (True, True, True, True, False, False)  # direction by a deterministic map
FAR_SIDE = (True, True, False, False, False, False)  # the photon ends beyond the surface


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
