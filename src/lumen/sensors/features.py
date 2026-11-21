"""Features: plain JAX functions of a deposit record, the axes a sensor's readout is binned or evaluated on.

    def my_feature(dep):                    # dep: the deposit (lumen.transport.step.Deposit) or a truth row's fields
        return dep.pos[2] * 2.0
    my_feature.reads = ("pos",)             # the deposit fields it reads beyond sensor, time and lam

A feature reads the facts transport records at the sensor, in the sensor placement's frame (``Deposit``): ``lam`` (nm),
``pos`` (3, m, the crossing point), ``dir`` (3, the photon's direction), ``normal`` (3, the surface normal there),
``gen`` (the photon's re-emission count). A run carries a field only when a feature of its sensor models reads it
(``reads``), so a feature costs nothing until it is used. ``BUILTIN`` names the features a value-set file may use;
code passes any function with a ``reads`` attribute (no registration).
"""

from __future__ import annotations

import jax.numpy as jnp

FIELDS = ("pos", "dir", "normal", "gen")  # the optional deposit fields a feature may read


def polar_angle(r):
    """Return acos(z / |r|) (rad) of a point r (3,) in a sensor's frame, as atan2(rho, z): finite on the axis."""
    return jnp.arctan2(jnp.sqrt(r[0] * r[0] + r[1] * r[1] + 1e-30), r[2])


def wavelength(dep):
    """Return the photon's wavelength at the sensor (nm)."""
    return dep.lam


def theta(dep):
    """Return the cathode position angle (rad): the polar angle of the crossing point about the placement's +z axis.

    The collection-efficiency angle of WCSim (WCSimWCSD.cc:75-84, e2dc7a4: acos(z / r) of the hit about the PMT's
    axis), at the crossing point itself.
    """
    return polar_angle(dep.pos)


def cos_incidence(dep):
    """Return |cos| of the angle between the photon's direction and the surface normal at the crossing (Geant4's
    angle of incidence, G4OpBoundaryProcess)."""  # noqa: D205, D209
    return jnp.abs(jnp.sum(dep.dir * dep.normal))


def generation(dep):
    """Return the photon's re-emission count (0: never wavelength-shifted in transport), as a float."""
    return jnp.asarray(dep.gen, jnp.float32)


def _coord(i):
    def f(dep):
        return dep.pos[i]

    f.__name__ = "xyz"[i]
    f.__doc__ = f"Return the crossing point's {'xyz'[i]} in the placement's frame (m)."
    f.reads = ("pos",)  # type: ignore[attr-defined]
    return f


x, y, z = (_coord(i) for i in range(3))


def rho(dep):
    """Return the crossing point's distance from the placement's z axis (m)."""
    return jnp.sqrt(dep.pos[0] * dep.pos[0] + dep.pos[1] * dep.pos[1])


wavelength.reads = ()  # type: ignore[attr-defined]
theta.reads = ("pos",)  # type: ignore[attr-defined]
cos_incidence.reads = ("dir", "normal")  # type: ignore[attr-defined]
generation.reads = ("gen",)  # type: ignore[attr-defined]
rho.reads = ("pos",)  # type: ignore[attr-defined]

BUILTIN = {f.__name__: f for f in (wavelength, theta, cos_incidence, generation, x, y, z, rho)}


def reads(fn) -> tuple:
    """Return the deposit fields a feature reads (its ``reads`` attribute), checked."""
    r = tuple(getattr(fn, "reads", None) or ())
    if not hasattr(fn, "reads") or any(f not in FIELDS for f in r):
        raise TypeError(f"feature {getattr(fn, '__name__', fn)!r}: declare the deposit fields it reads as "
                        f"fn.reads, a tuple of {FIELDS} (() for none)")  # fmt: skip
    return r
