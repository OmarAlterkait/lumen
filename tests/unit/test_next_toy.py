"""The NEXT-like toy (tests/data/next_toy, written by tools/g4ref/next_toy.py) against Geant4, feature by feature.

One geometry at NEXT-100's scales: a metre-scale teflon light tube coated with 3 um of TPB whose inner face lies on
the active gas, a SiPM board whose teflon mask carries a 4 um TPB coating whose face lies on the gas through the mask
and the board, four SiPM windows with sensitive areas, and a PMT; the sensors are read by their own skin surfaces.
Its four cases switch on one kind of surface each, and every one of them is a feature NEXT-100 needs:

- ``plain``: the skin sensors alone (Geant4's InvokeSD: the volume entered detects);
- ``sigma1``: the mask's unified ground skin with sigma_alpha = 1 rad (above lumen 0.2's 0.707 rad);
- ``glisur``: the glisur ground dielectric_dielectric borders gas <-> TPB, which hold across coincident faces only;
- ``all``: every surface together, the end-to-end counts per sensor group;
- ``wls``: ``all`` with the coatings shifting the light (as NEXT's TPB does) and 1 um thick, and sensors blind to the
  unshifted light: only a navigator that resolves the coating sees anything, so this is the float32 limit end to end.

Each case's detected fractions per group are compared with g4ref's on the same GDML (``tests/data/next_toy``
``geant4.json``, Geant4 11.3.0; the toy reads no property whose Geant4 10.7.4 path differs, so one release stands for
both). Every case is checked in float64, the two that do not touch the coatings in float32 as well, the end-to-end
counts of ``all`` at both precisions (its 3 um coating is 1.9 float32 bands where it sits, so float32 still tracks
it), and the float32 deficit of ``wls`` (47 % of the PMT light and 46 % of the SiPMs').
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
    path = lumen.convert(DATA / f"next_toy_{case}.gdml", out, recipe=Recipe(f"next_toy_{case}", "GAS", groups=groups),
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
@pytest.mark.parametrize("case", ["plain", "sigma1", "glisur", "all", "wls"])
def test_every_case_matches_geant4_in_float64(tmp_path, case):
    """Every feature together, in float64: the glisur borders across the coatings' coincident faces too."""
    with enable_x64():
        _agree(case, _fractions(_detector(tmp_path, case, "float64")))


@pytest.mark.slow
@needs_reference
def test_the_same_counts_at_both_precisions_where_the_coating_is_thick_enough(tmp_path):
    """The end-to-end counts of ``all`` at float32 and float64.

    Its 3 um coating is 1.9 float32 bands where it sits (the band of a ray that started 0.5 m from the origin and
    travelled 1.2 m is 1.6 um), so float32 still tracks it: measured at 2^22 photons, Geant4 / lumen is 0.998 and
    1.008 at float32 and 0.997 and 1.010 at float64, and the two precisions agree to 0.1 %.
    """
    a = _fractions(_detector(tmp_path, "all"))
    with enable_x64():
        b = _fractions(_detector(tmp_path, "all", "float64"))
    for g in GROUPS:
        err = math.hypot(a[g][1], b[g][1])
        assert abs(a[g][0] - b[g][0]) <= 3.0 * err, (g, a[g], b[g])


@pytest.mark.slow
@needs_reference
def test_float32_loses_the_light_of_a_shifting_coating_and_float64_does_not(tmp_path):
    """The ``wls`` case is the float32 limit end to end, and the reason NEXT-100 needs float64.

    Its coating is 1 um (0.63 of float32's band where it sits) and shifts the light, and the sensors are blind to the
    unshifted light, so a photon counts only if the navigator put it inside the coating. Measured at this test's
    2^18 photons per source, Geant4 / lumen: in float64 0.985 at the PMT (1.1 sigma) and 0.991 at the SiPMs
    (0.2 sigma), so lumen agrees; in float32 1.887 (46 sigma) and 1.865 (14 sigma), so a float32 run loses 47 % of
    the PMT light and 46 % of the SiPMs' (docs/explanation/known-limits.md, "Thin layers"). Before the flush-film
    fix the same run gave 1.033 and 2.445: keeping the surface across a sub-band bulk event, which a flush film
    needs, costs the PMT light this coating never had and gives back some of the SiPMs'. float64's agreement is
    checked with the other cases (``test_every_case_matches_geant4_in_float64``); here the float32 deficit is the
    record.
    """
    ref = GEANT4["wls"]
    got = _fractions(_detector(tmp_path, "wls"))
    for g, least in (("pmt", 1.03), ("sipm", 1.5)):
        ratio = ref[g] / got[g][0]
        assert ratio > least, (g, got[g], ref[g], ratio)
