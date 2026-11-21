"""What a reading of a material or surface returns: the library's media, surfaces and sensor types.

    entry = lumen.rules.optics.surface(record)       # SurfaceEntry(surface_type, params, ...)

The readings (``lumen.rules.optics``) return these records and the surface
lookup (``lumen.build.lookup``) compiles them; they sit below both, so neither imports the other.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from lumen.media import bulk
from lumen.surfaces import types


class DescriptionError(ValueError):
    """A library, tree and bindings that do not fit together."""


@dataclass(frozen=True, eq=False)
class MediumEntry:
    """A library medium: a bulk.Medium and its params ({"index": ..., "absorption": ..., "<channel>": ...})."""

    medium: bulk.Medium
    params: Mapping


@dataclass(frozen=True, eq=False)
class SurfaceEntry:
    """A library surface: a Geant4 surface type and its params.

    ``ignored_between_same_material``: Geant4 skips the boundary between two volumes of the SAME material, so a
    polished dielectric_dielectric surface declared there never acts. ``far_side``: what Geant4 does when the
    material beyond has no RINDEX, by the surface's (type, finish) (``lumen.rules.boundary``): ``FAR_KILL``,
    ``FAR_ABSORB``, or None (no Geant4 reading says: a crossing that would end beyond is refused).
    """

    surface_type: types.SurfaceType
    params: Mapping
    ignored_between_same_material: bool = False
    far_side: str | None = None


@dataclass(frozen=True, eq=False)
class SensorEntry:
    """A library sensor type: a surface type that can DETECT, its params and its reading's far side and skip.

    ``far_side`` and ``ignored_between_same_material``: as ``SurfaceEntry``'s (the reading of a surface is one, a
    sensor's or not).

    A border sensor records what it reads: ``surface`` the detector file's surface, ``group`` its sensor group, and
    ``exit`` whether it detects leaving its volume (credited to the volume left) rather than entering it. ``born``: a
    photon emitted inside the sensor's volume is counted at its emission with the entry's detection (a group counting
    by volume, ``lumen.rules.sensors.counts_born``).
    """

    surface_type: types.SurfaceType
    params: Mapping
    far_side: str | None = None
    ignored_between_same_material: bool = False
    surface: str | None = None
    group: str | None = None
    exit: bool = False
    born: bool = False
