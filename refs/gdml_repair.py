"""Write a Geant4-loadable copy of a GDML file: the named exporter-defect repairs applied as text edits, nothing else.

    python refs/gdml_repair.py SRC DST REPAIR [REPAIR ...]

The repairs (and the rule: lumen repairs every defect on import, and every Geant4 reference input made from the same
GDML gets the same repair) are documented in ``lumen.gdml.repair``, which this runs; ``lumen.validate.geometry`` uses
the same module for its Geant4 side.
"""

import sys

from lumen.gdml.repair import main

if __name__ == "__main__":
    main(sys.argv[1:])
