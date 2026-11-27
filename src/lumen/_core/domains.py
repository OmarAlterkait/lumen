"""The admissible ranges of lumen's values, by kind, and the one predicate that tests a value against one.

A range is (lo, hi), both ends included; ``admits`` is the only test of a value against one: the media laws' check of
a file's tables (``lumen.media.laws.validate``), ``Params.set`` and the value checks of every run
(``lumen.build.values``) call it, on NumPy or traced JAX values alike. A value that is not finite (NaN, an infinity)
is never admitted, whatever the range.
"""

import math

FRACTION = (0.0, 1.0)  # a probability or a share
POSITIVE = (0.0, math.inf)  # a yield, a time constant, an extinction index: 0 admitted
LENGTH = (0.0, math.inf)  # a length: >= 0 (0 is an infinite rate: Geant4 interacts at once, G4VDiscreteProcess)
INDEX = (1.0, math.inf)  # a refractive index
QUANTITIES = {"index": INDEX, "length": LENGTH, "fraction": FRACTION, "extinction": POSITIVE}  # a law's output


def admits(xp, v, lo, hi):
    """Return where ``v`` is finite and inside [lo, hi] (``xp``: numpy or jax.numpy)."""
    return xp.isfinite(v) & (v >= lo) & (v <= hi)


__all__ = ["FRACTION", "INDEX", "LENGTH", "POSITIVE", "QUANTITIES", "admits"]
