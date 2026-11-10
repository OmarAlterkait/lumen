"""The tutorial detector: a 1 m water cube with 8 disc PMTs (examples/data/example_tank.gdml) and its recipe.

    from example_tank import detector
    det = detector()          # converted once into ./lumen-examples (or $LUMEN_EXAMPLES_OUT), then loaded

The recipe says what the GDML cannot: which volume lumen simulates (the detector volume), which volumes are sensors,
how a photon is counted there, and which sensor model turns the light into photoelectrons and waveforms.
"""
# ruff: noqa: T201, TID251, D103  (an example script: it prints, reads its output directory from the environment)
# fmt: off

import hashlib
import os
from pathlib import Path

import lumen
from lumen.format.sensorsets import SensorGroup

DATA = Path(__file__).resolve().parent / "data"
GDML = DATA / "example_tank.gdml"
OUT = Path(os.environ.get("LUMEN_EXAMPLES_OUT", "lumen-examples"))

RECIPE = lumen.Recipe(
    name="example_tank",
    detector_volume="TankLV",                 # the water; light leaving it is absorbed
    wavelength_range=(300.0, 600.0),          # nm: the optics tables are read over this range
    groups=(
        SensorGroup(
            "pmt",
            rule="border_into",               # counted across the CathodeSurface border into each PMT volume
            surface="lumen:geant4", model="lumen:pmt",
            lv="PMT",                         # the logical volume of the 8 PMTs; one sensor per placement
            defaults=str(DATA / "toy_pmt.json"),          # the PMT values: QE, timing, charge, pulse
            readout={"gate": (-10.0, 60.0)},  # ns after emission
        ),
    ),
)


def detector(calibration="nominal"):
    """Return the loaded tutorial detector, converting it on first use (keyed by the GDML's content)."""
    digest = hashlib.sha256(GDML.read_bytes()).hexdigest()[:12]
    path = OUT / f"example_tank-{digest}.lumen"
    if not path.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        lumen.convert(GDML, RECIPE, path)
    return lumen.load(path, calibration=calibration)


def sensor_positions(det):
    """Return each sensor's centre (S, 3) in m, in the order of ``Light.arrivals`` (``Detector.sensor_table``)."""
    return det.sensor_table()["position"]
