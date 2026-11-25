"""The NEXT-like toy (tests/data/next_toy, written by tools/g4ref/next_toy.py) against Geant4, feature by feature.

One geometry at NEXT-100's scales: a metre-scale teflon light tube coated with 3 um of TPB whose inner face lies on
the active gas, a SiPM board whose teflon mask carries a 4 um TPB coating whose face lies on the gas through the mask
and the board, four SiPM windows with sensitive areas, and a PMT; the sensors are read by their own skin surfaces.
Its four cases switch on one kind of surface each, and every one of them is a feature NEXT-100 needs:

- ``plain``: the skin sensors alone (Geant4's InvokeSD: the volume entered detects);
- ``sigma1``: the mask's unified ground skin with sigma_alpha = 1 rad (above lumen 0.2's 0.707 rad);
- ``glisur``: the glisur ground dielectric_dielectric borders gas <-> TPB, which hold across coincident faces only;
- ``all``: every surface together, the end-to-end counts per sensor group.

Each case's detected fractions per group are compared with g4ref's on the same GDML (``tests/data/next_toy``
``geant4.json``, Geant4 11.3.0; the toy reads no property whose Geant4 10.7.4 path differs, so one release stands for
both). Every case is checked in float64, the two that do not touch the coatings in float32 as well, and the
end-to-end counts of ``all`` at both precisions (``test_the_same_counts_at_both_precisions_on_this_toy``: the toy's
coating is 1.9 float32 bands where it sits, so float32 still tracks it).
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from jax.experimental import enable_x64

DATA = Path(__file__).resolve().parents[1] / "data" / "next_toy"
GEANT4 = json.loads((DATA / "geant4.json").read_text()) if (DATA / "geant4.json").exists() else {}
SOURCES = ((0.0, 0.0, 0.0), (0.0, 0.0, -0.55))  # m: in the active gas, and close to the SiPM board
ENERGY_EV = 2.95
SAMPLES = 1 << 18
GROUPS = ("pmt", "sipm")


def _detector(tmp, case, precision="float32"):
    """Convert one case of the toy and load it at ``precision``."""
    import lumen
    from lumen.format.sensorsets import SensorGroup
    from lumen.recipe import Recipe

    groups = (SensorGroup("pmt", rule="border_into", lv="PMT_PHOTOCATHODE"),
              SensorGroup("sipm", rule="border_into", lv="SIPM_SENSE"))  # fmt: skip
    out = tmp / f"next_toy_{case}_{precision}.lumen"
    path = lumen.convert(DATA / f"next_toy_{case}.gdml", Recipe(f"next_toy_{case}", "GAS", groups=groups), out,
                         overwrite=True)  # fmt: skip
    return lumen.load(path, precision=precision)


PV_OF = {"pmt": "pmt_cath", "sipm": "sipm_sense"}  # the group's sensors: the physvol in their placed-volume path


def _fractions(det, seed=11):
    """Return {group: (fraction detected, its error)} of the toy's two point sources, each of one photon."""
    import lumen
    from lumen import units
    from lumen.sources import Point

    pvs = det.sensor_table().name
    rows = {g: [i for i, pv in enumerate(pvs) if PV_OF[g] in pv] for g in GROUPS}
    assert all(rows[g] for g in GROUPS), (rows, pvs)
    tot = {g: (0.0, 0.0) for g in GROUPS}
    for pos in SOURCES:
        light = lumen.simulate(det, Point(pos, n_photons=1.0, lam=units.wavelength_of(ENERGY_EV)),
                               samples=SAMPLES, seed=seed)  # fmt: skip
        a = np.asarray(light.arrivals, np.float64)
        e = np.asarray(light.arrivals_error, np.float64)
        for g in GROUPS:
            f, err = tot[g]
            tot[g] = (f + a[rows[g]].sum() / len(SOURCES),
                      math.hypot(err, np.sqrt((e[rows[g]] ** 2).sum()) / len(SOURCES)))  # fmt: skip
    return tot


def _agree(case, got, which=GROUPS):
    """Check each group's fraction against Geant4's, with both Monte Carlo errors."""
    ref = GEANT4[case]
    bad = []
    for g in which:
        f, e = got[g]
        err = math.hypot(e, ref[f"{g}_error"])
        if abs(f - ref[g]) > 3.0 * err:
            bad.append(
                f"{g}: lumen {f:.6g} +- {e:.3g}, Geant4 {ref[g]:.6g} +- {ref[f'{g}_error']:.3g} "
                f"(ratio {ref[g] / f if f else float('nan'):.4f}, {abs(f - ref[g]) / err:.1f} sigma)"
            )
    assert not bad, f"{case}: " + "; ".join(bad)


needs_reference = pytest.mark.skipif(not GEANT4, reason="no Geant4 reference for the NEXT toy yet")


@pytest.mark.slow  # converts and simulates the toy: minutes on a CPU
@needs_reference
@pytest.mark.parametrize("case", ["plain", "sigma1"])
def test_the_skin_sensors_and_a_sigma_alpha_of_one_match_geant4_in_float32(tmp_path, case):
    """The cases that do not need the coatings: float32 tracks them as Geant4 does."""
    _agree(case, _fractions(_detector(tmp_path, case)))


@pytest.mark.slow
@needs_reference
@pytest.mark.parametrize("case", ["plain", "sigma1", "glisur", "all"])
def test_every_case_matches_geant4_in_float64(tmp_path, case):
    """Every feature together, in float64: the glisur borders across the coatings' coincident faces too."""
    with enable_x64():
        _agree(case, _fractions(_detector(tmp_path, case, "float64")))


@pytest.mark.slow
@needs_reference
def test_the_same_counts_at_both_precisions_on_this_toy(tmp_path):
    """The end-to-end counts of ``all`` at float32 and float64, and how far the toy is from float32's limit.

    The toy's 3 um coating is 1.9 float32 bands where it sits (the band of a ray that started 0.5 m from the origin
    and travelled 1.2 m is 1.6 um), so float32 still tracks it: measured, the two precisions give the same fractions
    to 0.3 % at the PMT and 0.4 % at the SiPMs, inside the Monte Carlo error (lumen / Geant4 1.009 and 0.90 at
    float32, 1.012 and 0.90 at float64). A coating thinner than that, or photons starting farther away, is where
    float32 loses it: NEXT-100's own 2 to 5.4 um coatings, with its 1.3 m drift, are the measured case
    (docs/explanation/known-limits.md, "Thin layers").
    """
    a = _fractions(_detector(tmp_path, "all"))
    with enable_x64():
        b = _fractions(_detector(tmp_path, "all", "float64"))
    for g in GROUPS:
        err = math.hypot(a[g][1], b[g][1])
        assert abs(a[g][0] - b[g][0]) <= 3.0 * err, (g, a[g], b[g])
