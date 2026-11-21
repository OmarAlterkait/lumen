"""The g4ref arguments that read a detector file's sensors as lumen does (and the refuse mark of a lossy export).

    python tools/g4ref/sensor_args.py DETECTOR.lumen [--lossy]

Prints the arguments on one line: ``--sens-regex`` (the sensor groups' selectors, as one regular expression),
``--wcsim-hits`` when a ``border_into`` group's efficiency on entry is 1 (a hit is any photon killed entering the
sensor, as WCSim's sensitive detector counts it: what lumen's library counts under a sensor model that applies its
efficiency after tracking), and ``--refuse-lossy`` with ``--lossy`` (a GDML export that left something out:
``ExportReport.lossless`` False; g4ref does not know the flag and exits, so a lossy export never serves as a
reference).
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys

REFUSE_LOSSY = "--refuse-lossy"  # unknown to g4ref (it exits 2): a lossy export is not a reference


def args(sset, lossless: bool = True) -> list:
    """Return the g4ref arguments of a ``SensorSet`` (None: no sensors), with the refuse mark unless ``lossless``."""
    from lumen.rules.sensors import entry_efficiency

    out: list = []
    if sset is not None:
        pats = list(dict.fromkeys(g.lv_pattern if g.lv is None else f"^{re.escape(g.lv)}$" for g in sset.groups))
        if pats:  # the groups' selectors, as one regular expression
            out += ["--sens-regex", pats[0] if len(pats) == 1 else "|".join(f"({p})" for p in pats)]
        if any(g.rule == "border_into" and entry_efficiency(g) == 1.0 for g in sset.groups):
            out.append("--wcsim-hits")  # a hit is any photon killed entering the sensor, as WCSim's sensitive detector
    if not lossless:
        out.append(REFUSE_LOSSY)
    return out


def main(argv=None) -> int:
    """Print the g4ref arguments of a detector file's sensor set (see the module docstring)."""
    from lumen.format import sensorsets

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("detector", help="the detector file (.lumen)")
    ap.add_argument("--lossy", action="store_true", help="the GDML export is lossy (its report's lossless is False)")
    a = ap.parse_args(argv)
    sys.stdout.write(shlex.join(args(sensorsets.read(a.detector), not a.lossy)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
