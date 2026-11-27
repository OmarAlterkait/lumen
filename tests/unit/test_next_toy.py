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
  unshifted light: only a navigator that resolves the coating sees anything.

Each case's detected fractions per group are compared with g4ref's on the same GDML (``tests/data/next_toy``
``geant4.json``, Geant4 11.3.0; the toy reads no property whose Geant4 10.7.4 path differs, so one release stands for
both). Every case is checked in float32 and in float64, and the end-to-end counts of ``all`` at both precisions. The
1 um coatings of ``wls`` are 1.7 to 2.1 float32 bands where they sit: more than one band, so float32 resolves them; a
coating thinner than the band is still stepped over, and the run counts the light pushed through it.
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


def _detector(tmp, case, precision="float32", gdml=None):
    """Convert one case of the toy (or the GDML ``gdml`` in its place) and load it at ``precision``."""
    import lumen
    from lumen.format.sensorsets import SensorGroup
    from lumen.recipe import Recipe

    groups = (SensorGroup("pmt", rule="border_into", lv="PMT_PHOTOCATHODE"),
              SensorGroup("sipm", rule="border_into", lv="SIPM_SENSE"))  # fmt: skip
    out = tmp / f"next_toy_{case}_{precision}.lumen"
    path = lumen.convert(gdml or DATA / f"next_toy_{case}.gdml", out,
                         recipe=Recipe(f"next_toy_{case}", "GAS", groups=groups), overwrite=True)  # fmt: skip
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
@pytest.mark.parametrize("case", ["plain", "sigma1", "glisur", "all", "wls"])
def test_every_case_matches_geant4_in_float32(tmp_path, case):
    """Every feature together in float32, the 1 um shifting coating of ``wls`` too.

    Measured at this test's 2^18 photons per source, Geant4 / lumen on ``wls``: 0.986 +- 0.013 at the PMT and
    0.993 +- 0.044 at the SiPMs; 1.880 and 1.914 before a face more than one band from the photon was met at its own
    distance (the coating's far face, within the on-face tolerance, was taken for the face the photon sat on: the
    photon stepped over the coating and was never shifted).
    """
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
def test_a_run_counts_the_light_it_pushes_through_a_layer_it_does_not_resolve(tmp_path):
    """A photon between the two faces of a coating thinner than float32's band (the ``wls`` light tube's, 0.1 um
    thick here: 0.2 band) takes zero-length steps until the stuck rule pushes it past both: the run counts that
    weight in ``health.pushed``.

    Measured at this test's 2^16 photons from the source by the SiPM board (start weight 1): float32 pushes 0.0020,
    float64, which resolves the coating, none. A sub-band layer is mostly located past instead, which nothing counts:
    float32's light is 3.4 times float64's there. The 1 um coating, which float32 resolves, is pushed 0.0008 (0.33
    before its far face was met at its own distance).
    """
    import lumen
    from lumen import units
    from lumen.sources import Point

    def pushed(det):
        light = lumen.simulate(det, Point(SOURCES[1], n_photons=1.0, lam=units.wavelength_of(ENERGY_EV)),
                               samples=1 << 16, seed=11)  # fmt: skip
        return float(np.sum(np.asarray(light.health.pushed)))

    gdml = tmp_path / "next_toy_wls_sub_band.gdml"
    text = (DATA / "next_toy_wls.gdml").read_text()
    gdml.write_text(text.replace('rmax="500.001"', 'rmax="500.0001"'))
    assert pushed(_detector(tmp_path, "wls", gdml=gdml)) > 1e-3
    with enable_x64():
        assert pushed(_detector(tmp_path, "wls", "float64", gdml=gdml)) == 0.0
