"""Speed: one static setting, ``"fast"`` or ``"exact"``, for the estimators that make a run faster.

    lumen.simulate(..., speed="fast")      # the default: the diffusion walk, exact in law (other random numbers)
    lumen.simulate(..., speed="exact")     # no fast estimator

- "fast" (the default): the diffusion jump (after a Rayleigh event in a Rayleigh-only medium the photon walks its
  own scatterings inside the ball of its proven clearance, built from the geometry, with no geometry query): the same
  estimator in law with other random numbers; its exact form is the plain Rayleigh walk, so it is not an
  approximation. Where the detector gives it nothing to do (no such medium) the loop is the exact one, bit for bit.
- "exact": no fast estimator.

A level resolves to a static frozenset of lever names (``resolve``) that goes into the step's configuration
(``StepConfig.levers``): "exact" and "fast" are two compiled loops of one detector. A probe diagnostic
(``lumen.diagnostics``: it sees every step) cannot use the jump and runs exact. Such a run is never silently another
level: its ``Light.provenance`` records ``("speed", "exact (<why>)")``, and its values differ at the same seed from a
fast value run (another estimator of the same light, other random numbers); ``speed="exact"`` makes the two runs bitwise
alike. Otherwise the provenance records the level run. A records diagnostic and truth rows (``simulate(...,
truth=lumen.readout.Truth())``) use the jump: the walk is the photon's own path, exact in law, and its factor at the
reference values is the absorption survival exp(-mu_abs L), which the analog photon's open fate reads as Geant4's
absorbed / not absorbed along L (``lumen.transport.step._analog_crossing``). The jump's data (a proven clearance bound
per medium) are built from the detector itself (``lumen.transport.fastpath.prepare``).
"""

from __future__ import annotations

EXACT = frozenset()  # no fast estimator
FAST = frozenset({"jump"})  # the diffusion jump (unbiased)
LEVELS = {"exact": EXACT, "fast": FAST}


def resolve(speed, probes=False) -> tuple[frozenset, str]:
    """Return (the levers a run compiles, the speed it records): the level's levers, or none and "exact (<why>)"
    when the level is "fast" and the run is one the jump cannot serve (a probe diagnostic)."""  # noqa: D205, D209
    if speed not in LEVELS:
        raise ValueError(f"simulate: speed is 'fast' or 'exact', got {speed!r}")
    why = ["diagnostics"] if probes else []
    if speed == "fast" and why:
        return EXACT, f"exact ({', '.join(why)})"
    return LEVELS[speed], speed
