"""Every sampled part's proposal at the reference covers its live density for any keyed value in range (coverage.py).

The parts with a gap on p3/final are marked xfail (strict: a part that becomes covered fails until its mark goes).
"""

import coverage as cv
import numpy as np
import pytest

GAPS = {  # the gaps the harness finds on p3/final after the defensive mixture (p3/physics), and why
    "surface:photocathode": "nk 0 at the reference zeroes the specular share a live nk > 0 gives (contexts into a "
    "denser medium): nk has no defensive role, so with only it keyed nothing proposes the specular outcome",
    "surface:photocathode|T": "the straight share vanishes where the complex index's reflectance reaches 1 (nk 0 under "
    "total reflection, or nr 0): nr and nk have no domain, so with only them keyed the row is no edge row",
    "facet:glisur": "a live polish below the reference's widens the facet ball (radius 1 - polish): the facets outside "
    "the reference's ball are never drawn, by the lobe nor by the shell mixture (same radius)",
}
PARTS = cv.parts()


@pytest.mark.parametrize(
    "part",
    [pytest.param(p, marks=pytest.mark.xfail(strict=True, reason=GAPS[p.name])) if p.name in GAPS else p
     for p in PARTS],
    ids=[p.name for p in PARTS],
)  # fmt: skip
def test_the_proposal_covers_the_live_density(part):
    found = cv.gaps(part)
    assert not found, "\n".join(str(g) for g in found[:10])


def test_the_harness_finds_a_gap():
    """A proposal that drops an outcome the live density reaches is a gap; a covering one is none."""
    live = lambda v: np.array([v["p"], 1.0 - v["p"]])  # noqa: E731
    bare = cv.Part("toy", {"p": (0.0, 0.5, 1.0)}, {"p": 0.5}, live, lambda ref, k: live(ref))
    mixed = cv.Part("toy", {"p": (0.0, 0.5, 1.0)}, {"p": 0.5}, live, lambda ref, k: 0.9 * live(ref) + 0.05)
    assert {(g.reference, g.point) for g in cv.gaps(bare)} == {(0.0, "0"), (1.0, "1")}
    assert cv.gaps(mixed) == []
