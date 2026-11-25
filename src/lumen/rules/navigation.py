"""Geant4's navigation rules as lumen's navigator applies them: tolerances, surface identity, overlaps, stuck tracks.

The photon carries the placed volume it is in and the surface it sits on (its face and outward sign). Geant4's
G4Navigator decides a point's side of a surface within kCarTolerance by the direction of travel; lumen's tolerance is
the float32 resolution of the coordinates, the **band**. The rules, each with the Geant4 code it follows:

- **Tolerance.** One form, ``band(scale) = 8 eps scale + 1e-9 m`` at a coordinate scale (m), and named multiples:
  the query and the CSG evaluator at ``|x| + |t|`` of a crossing, the step's push unit at a point's ``max |x|``
  (``step_band``), the placement check's step in ``K_ON`` bands (``check_step_in``), a zero-length step below
  ``ZERO_STEP_BANDS``, a WLS box thinner than one band (refused: ``lumen.build.import_rules``).
- **Surface identity.** A face within ``K_ON`` bands of the start (perpendicular to it) is the face the photon sits
  on: the direction decides which side it is on (G4Navigator's fEntering / fExiting). A face that close along the ray
  but farther perpendicularly is looked for up to ``GRAZE`` times farther: a ray leaving its face at an angle theta
  is K_ON bands from it after K_ON / sin(theta) bands, so ``GRAZE`` sets the shallowest angle the start's face is
  still recognised at, asin(1 / GRAZE) = 0.22 deg; below it the face is a new crossing (a zero-length step).
  A bulk event within ``ZERO_STEP_BANDS`` bands of the start leaves the photon on the surface it sat on
  (``lumen.transport.step``: its point has not moved off that face, so its side there is still its direction's), and a
  zero-length crossing whose solid gives no face of its own (a point one ulp beyond the face just crossed) is read on
  that surface too: its normal is the crossing's.
- **Relocation.** A crossing is relocated from one band along the ray and past the face met, along its normal the way
  the photon travels (G4Navigator's LocateGlobalPointAndSetup, up while a solid does not hold the point, then down):
  one band of the point's own coordinates (``step_band``) for a crossing, whose point is within an ulp of the face at
  any angle of the ray; ``K_ON`` + 1 of the query's bands for a zero-length one, whose point is the start's, up to
  ``K_ON`` bands on the near side of the face the on-face rule put the photon on. A volume whose own face coincides
  with the one met (a daughter placed flush in its mother) is then left, or entered, at that one crossing, as Geant4
  leaves a mother whose surface the point is on when the direction leaves it.
- **Provenance of the multiples.** The band and its multiples (``BAND_ULPS``, ``K_ON``, ``GRAZE``,
  ``ZERO_STEP_BANDS``, ``FACET_*``, the push) are stated, with every other geometry tolerance, in
  ``lumen.rules.tolerances``: the band is a few float32 ulps of the coordinates (a root and its point each carry about
  one ulp per operation), and the multiples are margins over it. Geant4 has one length for all of them
  (kCarTolerance, 1e-9 mm, and 100 of it for the push); float32 coordinates cannot resolve that. Their effect on
  light is what the validation measures (docs/explanation/validation.md); none is measured alone.
- **Overlaps and protrusions.** A point has one owner, level by level (``lumen.geometry.ownership``: a solid proved
  inside its sibling, else the one placed first): a sibling that may own part of a volume (or of an ancestor) is a
  candidate whose face the photon enters as a daughter's, ranked by the rule's order (``prio``: lower first) ahead of
  the siblings it owns;
  an ancestor a volume is not proved inside is a candidate whose face is an exit, the earliest exit winning
  (G4Navigator::LocateGlobalPointAndSetup goes up while the mother does not hold the point). Two solids within the
  contact tolerance (``contact``) touch.
- **Ties.** One order decides every tie, the rule's (``prio``): at coincident daughter faces the first in it is
  entered (G4VoxelNavigation::ComputeStep: the first daughter found); leaving a volume, the photon is located in the
  first sibling in it holding the point, then down into its daughters (G4Navigator::LocateGlobalPointAndSetup,
  LevelLocate).
- **StepTooSmall.** G4OpBoundaryProcess returns StepTooSmall at a step <= kCarTolerance (1e-12 m), before any
  material or surface is read. lumen's zero-length step (closer than ``ZERO_STEP_BANDS`` bands) is far longer than
  that, so it is StepTooSmall only where no surface type applies (``lumen.transport.step``). Elsewhere Geant4 takes
  a real step there and applies the surface.
- **Facet rule.** A mesh is an opaque part: it holds no point, and its surface is met from either side as an entry
  into it. A facet is met by a ray entering it (G4TriangularFacet::Intersect, DistanceToIn): a start behind its plane
  by at most ``FACET_BEHIND_BANDS`` bands is on it; the crossing's t comes from the facet's plane, ``FACET_SHORT_ULPS``
  ulps short. A facet met from clearly behind first (a start inside the closed surface) is an entry into the part
  alike.
- **Stuck rule.** After ``ZERO_STEPS_BEFORE_PUSH`` zero-length steps in a row the photon is pushed
  ``PUSH_DISTANCE_TOLERANCES`` bands along its direction and located afresh; at ``ZERO_STEPS_BEFORE_KILL`` it is
  abandoned (G4Navigator's fActionThreshold_NoZeroSteps = 10, fAbandonThreshold_NoZeroSteps = 25; Geant4 pushes
  100 kCarTolerance).

Notes
-----
Geant4 sources: G4Navigator (LocateGlobalPointAndSetup, LevelLocate, the zero-step thresholds), G4VoxelNavigation
(ComputeStep), G4NormalNavigation, G4TriangularFacet.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from lumen.rules import tolerances as tol

# the band and its multiples, stated once in lumen.rules.tolerances; read as rules.<name> here and in the loop
BAND_ULPS, BAND_FLOOR, K_ON, GRAZE = tol.BAND_ULPS, tol.BAND_FLOOR, tol.K_ON, tol.GRAZE
FACET_BEHIND_BANDS, FACET_SHORT_ULPS, ZERO_STEP_BANDS = (
    tol.FACET_BEHIND_BANDS,
    tol.FACET_SHORT_ULPS,
    tol.ZERO_STEP_BANDS,
)
ZERO_STEPS_BEFORE_PUSH, ZERO_STEPS_BEFORE_KILL = tol.ZERO_STEPS_BEFORE_PUSH, tol.ZERO_STEPS_BEFORE_KILL
PUSH_DISTANCE_TOLERANCES = tol.PUSH_DISTANCE_TOLERANCES

FLOAT32_EPS = float(np.finfo(np.float32).eps)
BAND = BAND_ULPS * FLOAT32_EPS  # the band per metre of coordinate scale


def band(scale):
    """Return the float32 band (m) at a coordinate scale (m): for a crossing at ``t`` on a ray whose largest start
    coordinate is ``xa``, ``band(xa + |t|)``. Endpoints closer than this are one crossing; the photon's own surface is
    within it of the start. Takes the scale's precision (float32 in the loop, float64 on the host)."""  # noqa: D205, D209
    return BAND * scale + BAND_FLOOR


def contact(scale):
    """Return the contact tolerance (m) at a coordinate scale (m): ``K_ON`` bands. Two solids closer than this touch;
    they overlap, and a solid reaches out of another, only deeper (``lumen.geometry.ownership``)."""  # noqa: D205, D209
    return K_ON * band(scale)


def check_step_in(xe):
    """Return the placement check's step in from an entry point ``xe`` (3,): the contact tolerance there (m)."""
    return contact(jnp.max(jnp.abs(xe)))


def step_band(x):
    """Return the step's band at a point ``x`` (3,), the stuck rule's push unit (m)."""
    return band(jnp.max(jnp.abs(x)))


def entry_wins(t, rank, best, best_rank):
    """Return whether a daughter's event at ``t`` beats the best so far, by Geant4's entry-tie rule.

    The volume's own exit is visited first and sets the best; a daughter beats it on a tie, and between daughters at
    one t the first in the rule's order (the higher rank) wins. ``best`` is ``(t, item, leaf, face, zero, is_own)``.
    """
    return (t < best[0]) | ((t == best[0]) & (best[5] | (rank > best_rank)))


def level_locate_wins(item, found, excl, n, rank, found_rank):
    """Return whether daughter ``item`` replaces ``found`` as the volume holding a point: the first in the rule's order
    (the higher rank: an owner first).

    ``excl`` is the volume just left (never re-entered here), ``n`` the number of volumes (an item above is no unit).
    """  # noqa: D205
    return ((found < 0) | (rank > found_rank)) & (item != excl) & (item < n)
