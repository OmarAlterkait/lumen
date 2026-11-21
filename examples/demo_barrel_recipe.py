"""The recipe of the guide's example detector (examples/data/demo_barrel.gdml).

A recipe is a small Python file you keep beside your GDML. It says what the GDML cannot: the detector volume, the
sensor groups. It runs once, when the detector is converted.
"""
# ruff: noqa: T201, TID251, D103  (an example script: it prints, reads its output directory from the environment)
# fmt: off

from lumen import Recipe
from lumen.format.sensorsets import SensorGroup

RECIPE = Recipe(
    name="demo_barrel",
    detector_volume="BarrelLV",              # the logical volume lumen simulates; light leaving it is absorbed
    groups=(
        SensorGroup(
            "pmt",                           # the group's name: parameter keys are sensors/pmt/<field>
            rule="border_into",              # counted across a border surface into the sensor volume
            lv="PMT",                        # or lv_pattern=r"^PMT" (a regular expression on logical-volume names)
            values="examples/data/toy_pmt.json",    # a PMT value set: a JSON file (or <package>:<set>)
            readout={"gate": (-20.0, 80.0)}, # ns after emission
        ),
    ),
)
