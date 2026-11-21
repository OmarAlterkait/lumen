"""The efficiency on entry of a sensor group: one rule for both counting rules, declared by the sensor model.

    entry_efficiency(group)        # 1, the group's ``entry_efficiency`` setting, or None (the surface's EFFICIENCY)
    counts_born(group)             # a photon emitted inside the group's volumes is counted at its emission

A sensor model that applies the efficiency after tracking (a group's value set) has every absorption entering counted,
as WCSim's sensitive detector does; otherwise the group's setting, else the surface's own reading
(docs/dev/decisions.md).

Leaving a sensor's volume (an exit sensor: a border out of it whose surface has an EFFICIENCY) the rule is Geant4's
boundary process, whatever the group: the absorbed share is detected with the surface's EFFICIENCY (DoAbsorption,
G4OpBoundaryProcess.hh:289-308, 10.7.4), and only a Detection invokes the sensitive detector
(G4OpBoundaryProcess.cc:519-520); the efficiency on entry is a sensitive detector's counting of the absorptions
entering its volume, which an exit is not.
"""

from __future__ import annotations

from lumen.format.sensorsets import COUNTS
from lumen.sensors import group_model


def after_tracking(group) -> bool:
    """Return whether a sensor group's sensor model applies the detection efficiency after tracking (it declares it)."""
    model = group_model(group)
    return model is not None and bool(model.efficiency_after_tracking)


def entry_efficiency(group) -> float | None:
    """Return the detection efficiency on entry of a sensor group: 1 when its sensor model applies the efficiency after
    tracking (every absorption entering is counted), else its ``entry_efficiency`` setting, else None (the surface's own
    reading: EFFICIENCY)."""  # noqa: D205, D209
    if after_tracking(group):
        return 1.0
    v = group.settings.get("entry_efficiency")
    return None if v is None else float(v)


def counts_born(group) -> bool:
    """Return whether a sensor group counts a photon emitted inside its volumes at its emission.

    An ``any_entry`` group declares a sensitive volume. Its setting ``count`` chooses: "volume" (the default, Geant4's
    sensitive-detector convention, G4VSensitiveDetector::ProcessHits on every step in the volume, the first one of a
    track born there included: the photon is counted where it is emitted, with the entry's detection) or "entry"
    (counted only crossing into the volume; a photon born inside leaves without being counted). A ``border_into``
    group counts at its surfaces, never at an emission.
    """
    return group.rule == "any_entry" and group.settings.get("count", COUNTS[0]) == "volume"
