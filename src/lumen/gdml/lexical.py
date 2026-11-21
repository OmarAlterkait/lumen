"""The GDML lexical rules the reader, the repair and the exporter share (standard library only).

- ``SKIP_LITERALS``: the attributes whose text is a name or a word, never a number Geant4 evaluates;
- ``NUMBER``: a number literal inside an attribute or an expression (not part of a name or a longer number);
- ``BIG_EXPONENT``, ``LONG_LITERAL``: what an out-of-range literal has (a three-digit exponent; else more than
  ``LONG_LITERAL`` characters: a double's range ends near 1e308, so a longer plain literal can be out of it);
- ``TINY``: the smallest normal double: Geant4's evaluator (strtod) refuses a nonzero literal below it (ERANGE).
"""

import re
import sys

SKIP_LITERALS = frozenset({"name", "ref", "formula", "state", "surfaceproperty", "auxtype", "auxvalue", "auxunit",
                           "version"})  # fmt: skip
NUMBER = re.compile(r"(?<![\w.])(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])")
BIG_EXPONENT = re.compile(r"\d[eE][-+]?\d{3}")
LONG_LITERAL = 300
TINY = sys.float_info.min
