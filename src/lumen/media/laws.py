"""Spectral laws: every optical quantity that depends on wavelength.

A ``Law`` is static (hashable by identity; its kind, parameter names and output kind) and is evaluated as
``law(p, lam)`` with ``p`` a dict of arrays and ``lam`` in nm. Output kinds and their valid ranges, checked by
``validate`` over the detector's wavelength range:

=========  ===============  ===================================================
index      n >= 1           refractive indices
length     L > 0 (m)        absorption and scattering lengths
fraction   0 <= f <= 1      reflectances, efficiencies, yields per photon
density    >= 0             emission spectra (normalised by their user)
scale      > 0              dimensionless multipliers
=========  ===============  ===================================================

Tables (Geant4 property vectors) are linear in photon ENERGY between their knots, as Geant4 interpolates
(G4PhysicsVector::Value); outside the knots a law refuses at validation unless ``outside="clamp"`` was declared.
The Geant4 refractive index ``geant4_rindex`` also gives Geant4's group velocity (G4MaterialPropertiesTable::
CalculateGROUPVEL) through ``group_index``.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from lumen.units import HC

RANGES = {
    "index": (1.0, np.inf),
    "length": (0.0, np.inf),
    "fraction": (0.0, 1.0),
    "density": (0.0, np.inf),
    "scale": (0.0, np.inf),
}
STRICT_LOW = {"length", "scale"}  # output kinds whose lower bound is excluded
VALIDATE_POINTS = 2001  # wavelengths at which validate evaluates a law


@dataclass(frozen=True, eq=False)
class Law:
    """A wavelength-dependent quantity: ``fn(p, lam)`` with parameters named ``params``.

    Attributes
    ----------
    kind, out : str
        The form of the law ("tabulated", "cauchy", ...) and its output kind (a key of ``RANGES``).
    params : tuple of str
        The names of the parameters ``fn(p, lam)`` reads from ``p``.
    fn : callable
        ``(p, lam) -> value``.
    domain : (float, float)
        Wavelengths (nm) where the law is defined.
    children : tuple of Law
        The laws this one is built from.
    """

    kind: str
    out: str
    params: tuple[str, ...]
    fn: Callable
    domain: tuple[float, float] = (0.0, np.inf)
    children: tuple["Law", ...] = ()

    def __call__(self, p, lam):
        """Evaluate the law at wavelengths ``lam`` (nm)."""
        return self.fn(p, lam)

    def __repr__(self):
        """Show the law's form and output kind."""
        return f"Law({self.kind}, out={self.out})"


class LawError(ValueError):
    """A law is undefined, non-finite or out of its output range somewhere in the wavelength range."""


def constant(out):
    """Return the law ``value``, independent of wavelength."""
    return Law("constant", out, ("value",), lambda p, lam: p["value"] * jnp.ones_like(lam))


def tabulated(out, e_min, e_max, outside="error"):
    """Return a table linear in photon energy between the knots ``p["energy"]`` (eV, ascending), ``p["value"]``.

    The knot span (e_min, e_max) is static, so the domain is known before any value is. Outside it the law fails
    validation (``outside="error"``) or holds its end values (``outside="clamp"``, as Geant4 does).
    """
    if outside not in ("error", "clamp"):
        raise ValueError(f"outside must be 'error' or 'clamp', got {outside!r}")
    domain = (0.0, np.inf) if outside == "clamp" else (HC / e_max, HC / e_min)

    def fn(p, lam):
        return jnp.interp(HC / lam, p["energy"], p["value"])

    return Law("tabulated", out, ("energy", "value"), fn, domain)


def sellmeier(out="index", terms=3):
    """Return the Sellmeier index: n^2 = 1 + sum_i B_i lam^2 / (lam^2 - C_i), with lam in um and C_i in um^2."""
    names = tuple(f"B{i}" for i in range(1, terms + 1)) + tuple(f"C{i}" for i in range(1, terms + 1))

    def fn(p, lam):
        l2 = (lam * 1e-3) ** 2
        n2 = 1.0 + sum(p[f"B{i}"] * l2 / (l2 - p[f"C{i}"]) for i in range(1, terms + 1))
        return jnp.sqrt(n2)

    return Law("sellmeier", out, names, fn)


def cauchy(out="index"):
    """Return the Cauchy index: n = A + B / lam^2 + C / lam^4, with lam in um."""

    def fn(p, lam):
        l2 = (lam * 1e-3) ** 2
        return p["A"] + p["B"] / l2 + p["C"] / (l2 * l2)

    return Law("cauchy", out, ("A", "B", "C"), fn)


def power(out):
    """Return the power law v0 (lam / lam0)^k; a Rayleigh scattering length has k = 4."""
    return Law("power", out, ("v0", "lam0", "k"), lambda p, lam: p["v0"] * (lam / p["lam0"]) ** p["k"])


def group_index(law):
    """Return the group index n_g = n - lam dn/dlam of an index law: one law gives phase and group velocity."""
    if law.out != "index":
        raise ValueError("group_index needs an index law")

    def fn(p, lam):
        n, dn = jax.jvp(lambda x: law(p, x), (lam,), (jnp.ones_like(lam),))
        return n - lam * dn

    return Law("group_index", "index", law.params, fn, law.domain, (law,))


def validate(law, p, lam_min, lam_max, n=VALIDATE_POINTS, name="law"):
    """Check a law over the wavelength range [lam_min, lam_max] (nm), failing closed.

    The law must be defined there, finite and inside its output range. Returns the evaluated (lam, value) grid.
    """
    lo, hi = law.domain
    if lam_min < lo - 1e-9 or lam_max > hi + 1e-9:
        raise LawError(
            f"{name}: defined on [{lo:.1f}, {hi:.1f}] nm, the detector uses [{lam_min:.1f}, {lam_max:.1f}] nm "
            "(extend the data or declare outside='clamp')"
        )
    missing = [k for k in law.params if k not in p]
    if missing:
        raise LawError(f"{name}: missing parameters {missing}")
    lam = np.linspace(lam_min, lam_max, n)
    v = np.asarray(
        law({k: jnp.asarray(p[k], jnp.float32) for k in law.params}, jnp.asarray(lam, jnp.float32)), np.float64
    )
    if not np.all(np.isfinite(v)):
        raise LawError(f"{name}: non-finite values in [{lam_min}, {lam_max}] nm")
    a, b = RANGES[law.out]
    bad = (v < a) | (v > b) | ((v <= a) if law.out in STRICT_LOW else False)
    if np.any(bad):
        i = int(np.flatnonzero(bad)[0])
        raise LawError(f"{name}: {law.out} out of range at {lam[i]:.1f} nm: {v[i]:.6g} (allowed [{a}, {b}])")
    return lam, v


# ----------------------------------------------------------------------------------- Geant4's index and group index
def geant4_group_index_table(e, n):
    """Return Geant4's GROUPVEL knots as group indices c / v_g: (energies (k + 1,), group indices (k + 1,)).

    Notes
    -----
    G4MaterialPropertiesTable::CalculateGROUPVEL, literally: v_g = c / (n + dn / d log E) at the first knot, the
    midpoints and the last knot, replaced by c / n where it is negative or faster than c / n.
    """
    slope = (n[1:] - n[:-1]) / jnp.log(e[1:] / e[:-1])
    first = jnp.maximum(n[:1] + slope[:1], n[:1])
    mid_n = 0.5 * (n[:-2] + n[1:-1])
    mid = jnp.maximum(mid_n + slope[:-1], mid_n)
    last = jnp.maximum(n[-1:] + slope[-1:], n[-1:])
    eg = jnp.concatenate([e[:1], 0.5 * (e[:-2] + e[1:-1]), e[-1:]])
    return eg, jnp.concatenate([first, mid, last])


def _group_index_knot(e, n, j):
    """Return knot ``j`` of ``geant4_group_index_table``'s group indices alone, with the table's own arithmetic."""
    k = e.shape[0] - 1
    s = jnp.clip(j - 1, 0, k - 1)  # the slope: the first knot's, then the one ending at knot j
    slope = (n[s + 1] - n[s]) / jnp.log(e[s + 1] / e[s])
    base = jnp.where((j == 0) | (j == k), n[j], 0.5 * (n[s] + n[s + 1]))
    return jnp.maximum(base + slope, base)


def geant4_group_index(e, n, lam):
    """Return c / GROUPVEL at wavelengths ``lam`` (nm); Geant4 interpolates v_g linearly in energy, clamped.

    Only the two knots around ``lam`` are evaluated (``jnp.interp`` on that pair is ``jnp.interp`` on the table): in
    the photon loop the whole table's slopes were otherwise recomputed at every step.
    """
    eg = jnp.concatenate([e[:1], 0.5 * (e[:-2] + e[1:-1]), e[-1:]])

    def one(x):
        pair = jnp.clip(jnp.searchsorted(eg, x, side="right"), 1, eg.shape[0] - 1) + jnp.arange(-1, 1)
        ng = jnp.stack([_group_index_knot(e, n, j) for j in pair])
        return 1.0 / jnp.interp(x, eg[pair], 1.0 / ng)

    return jnp.vectorize(one)(HC / lam)


@jax.custom_jvp
def _geant4_n(e, n, lam):
    return jnp.interp(HC / lam, e, n)


@_geant4_n.defjvp
def _geant4_n_jvp(primals, tangents):
    e, n, lam = primals
    de, dn, dlam = tangents
    v, dv = jax.jvp(lambda e_, n_: jnp.interp(HC / lam, e_, n_), (e, n), (de, dn))
    ng = geant4_group_index(e, n, lam)
    return v, dv + (v - ng) / lam * dlam  # so that n - lam dn/dlam is Geant4's group index


def geant4_rindex():
    """Return Geant4's RINDEX law: linear in energy, clamped, with Geant4's group velocity.

    Its wavelength derivative is defined so that ``group_index`` of this law is Geant4's GROUPVEL group index.
    Parameters ``{"energy" (eV), "value"}``.
    """
    return Law("g4_rindex", "index", ("energy", "value"), lambda p, lam: _geant4_n(p["energy"], p["value"], lam))
