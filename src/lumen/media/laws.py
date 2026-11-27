"""Spectral laws: every optical quantity that depends on wavelength.

A ``Law`` is static (hashable by identity; its form, parameter names and output quantity) and is evaluated as
``law(p, lam)`` with ``p`` a dict of arrays and ``lam`` in nm. Output quantities and their valid ranges, checked by
``validate`` at the law's own points (a table's knots) inside the light's domain and at the domain's ends:

=========  ===============  ===================================================
index      n >= 1           refractive indices
length     L >= 0 (m)       absorption and scattering lengths (0: an infinite rate)
fraction   0 <= f <= 1      reflectances, efficiencies, yields per photon
extinction k >= 0           a complex index's imaginary part
=========  ===============  ===================================================

(one table, ``lumen._core.domains.QUANTITIES``, tested by its one predicate ``domains.admits``)

Tables (Geant4 property vectors) are linear in photon ENERGY between their knots, as Geant4 interpolates
(G4PhysicsVector::Value), and hold their end values outside the knots, as Geant4 does: a table's values at its knots
bound it at every wavelength.
The Geant4 refractive index ``geant4_rindex`` also gives Geant4's group velocity (G4MaterialPropertiesTable::
CalculateGROUPVEL) through ``group_index``.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from lumen._core import domains
from lumen.units import C_VACUUM, HC

RANGES = domains.QUANTITIES  # each output quantity's range


@dataclass(frozen=True, eq=False)
class Law:
    """A wavelength-dependent quantity: ``fn(p, lam)`` with parameters named ``params``.

    Attributes
    ----------
    form, out : str
        The form of the law ("tabulated", "constant", ...) and its output quantity (a key of ``RANGES``).
    params : tuple of str
        The names of the parameters ``fn(p, lam)`` reads from ``p``.
    fn : callable
        ``(p, lam) -> value``.
    points : callable, optional
        ``p -> wavelengths`` (nm) whose values bound the law's at every wavelength (a table's knots), where
        ``validate`` checks it; None: the law cannot be checked, and ``validate`` refuses it.
    n_g : callable, optional
        An index law's own group index ``(p, lam) -> n_g`` (``group_index`` returns it); None: n - lam dn/dlam.
    """

    form: str
    out: str
    params: tuple[str, ...]
    fn: Callable
    points: Callable | None = None
    n_g: Callable | None = None

    def __call__(self, p, lam):
        """Evaluate the law at wavelengths ``lam`` (nm)."""
        return self.fn(p, lam)

    def __repr__(self):
        """Show the law's form and output quantity."""
        return f"Law({self.form}, out={self.out})"


class LawError(ValueError):
    """A law is non-finite or out of its output range at one of its points, or has no points to be checked at."""


def knots(p):
    """Return a table's knots as wavelengths (nm): linear in energy between them and held beyond, its values there
    bound it everywhere."""  # noqa: D205, D209
    return HC / np.asarray(p["energy"], np.float64)


def constant(out):
    """Return the law ``value``, independent of wavelength (checked at one wavelength: any is the same)."""
    return Law("constant", out, ("value",), lambda p, lam: p["value"] * jnp.ones_like(lam), lambda p: np.ones(1))


def _table(p, lam):
    """Return the table ``p`` at wavelengths ``lam``: linear in photon energy, held at its end values."""
    return jnp.interp(HC / lam, p["energy"], p["value"])


def tabulated(out):
    """Return a table linear in photon energy between the knots ``p["energy"]`` (eV, ascending), ``p["value"]``,
    held at its end values beyond them (G4PhysicsVector::Value)."""  # noqa: D205, D209
    return Law("tabulated", out, ("energy", "value"), _table, knots)


def group_index(law):
    """Return the group index of an index law: its own ``n_g`` when it has one, else n - lam dn/dlam."""
    if law.out != "index":
        raise ValueError("group_index needs an index law")
    if law.n_g is not None:
        return Law("group_index", "index", law.params, law.n_g)

    def fn(p, lam):
        n, dn = jax.jvp(lambda x: law(p, x), (lam,), (jnp.ones_like(lam),))
        return n - lam * dn

    return Law("group_index", "index", law.params, fn)


def domain_points(lam, domain):
    """Return the wavelengths (nm) where a law is evaluated by light in ``domain``: its points inside, and the ends.

    ``domain``: photon energy intervals (eV; ``lumen.rules.optics.light_domain``), or None (every wavelength: all the
    points). A table is linear in energy between its knots and held beyond them, so its values over the domain are
    bounded by its knots inside it and its values at the domain's ends.
    """
    lam = np.asarray(lam, np.float64)
    if domain is None:
        return lam
    e = np.asarray(domain, np.float64).reshape(-1, 2)
    inside = np.zeros(lam.shape, bool)
    for lo, hi in e:
        inside |= (lam >= HC / hi) & (lam <= HC / lo)
    return np.concatenate([lam[inside], HC / e.ravel()])


def validate(law, p, name="law", domain=None):
    """Check a law where light evaluates it: at its own points (``Law.points``: a table's knots) inside ``domain`` and
    at the domain's ends (``domain_points``), failing closed.

    The law must be finite and inside its output range there; a law without points is refused. Returns the evaluated
    (lam, value) at those wavelengths.
    """  # noqa: D205
    if law.points is None:
        raise LawError(f"{name}: the {law.form} law declares no points to be checked at (Law.points)")
    missing = [k for k in law.params if k not in p]
    if missing:
        raise LawError(f"{name}: missing parameters {missing}")
    lam = domain_points(law.points(p), domain)
    v = np.asarray(
        law({k: jnp.asarray(p[k], jnp.float32) for k in law.params}, jnp.asarray(lam, jnp.float32)), np.float64
    )
    if not np.all(np.isfinite(v)):
        raise LawError(f"{name}: non-finite values at its points")
    a, b = RANGES[law.out]
    bad = ~domains.admits(np, v, a, b)
    if np.any(bad):
        i = int(np.flatnonzero(bad)[0])
        raise LawError(f"{name}: {law.out} out of range at {lam[i]:.1f} nm: {v[i]:.6g} (allowed " f"[{a}, {b}])")
    return lam, v


# ----------------------------------------------------------------------------------- Geant4's index and group index
def _group_index_knot(e, n, j):
    """Return knot ``j`` of Geant4's GROUPVEL table as a group index c / v_g.

    Notes
    -----
    G4MaterialPropertiesTable::CalculateGROUPVEL, literally: v_g = c / (n + dn / d log E) at the first knot, the
    midpoints and the last knot, replaced by c / n where it is negative or faster than c / n.
    """
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


def geant4_rindex():
    """Return Geant4's RINDEX law: linear in energy, clamped, with Geant4's group velocity.

    Its wavelength derivative is the table's segment slope; ``group_index`` of this law is Geant4's GROUPVEL group
    index (its ``n_g``), used for timing only. Parameters ``{"energy" (eV), "value"}``.
    """
    return Law("g4_rindex", "index", ("energy", "value"), _table, knots,
               lambda p, lam: geant4_group_index(p["energy"], p["value"], lam))  # fmt: skip


def geant4_rindex_groupvel():
    """Return Geant4's RINDEX law with a user GROUPVEL: n as ``geant4_rindex``, n_g = c / GROUPVEL.

    Geant4 times a photon by its material's GROUPVEL table (G4Track::CalculateVelocityForOpticalPhoton): a user table
    replaces the one computed from RINDEX whatever the order the two were added in (G4MaterialPropertiesTable::
    AddProperty overwrites it; CalculateGROUPVEL keeps an existing one). The table is linear in energy and clamped.
    Parameters ``{"energy" (eV), "value", "groupvel_energy" (eV), "groupvel" (m / ns)}``.
    """
    return Law("g4_rindex_groupvel", "index", ("energy", "value", "groupvel_energy", "groupvel"),
               _table, knots,
               lambda p, lam: C_VACUUM / jnp.interp(HC / lam, p["groupvel_energy"], p["groupvel"]))  # fmt: skip
