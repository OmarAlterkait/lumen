"""The sibling sweep: every pair of boxes overlapping on every axis (open), once, in the order of a plain sweep along
the axis of least relative extent."""

import numpy as np
import pytest

from lumen.geometry import ownership
from lumen.rules.tolerances import FLOAT64_TINY


def plain_sweep(lo, hi):
    """Return the pairs of a sweep over every box along the axis of least relative extent: the order to match."""
    ext = (hi - lo).mean(0) / np.maximum(hi.max(0) - lo.min(0), FLOAT64_TINY)
    ax = int(np.argmin(ext))
    o = np.argsort(lo[:, ax], kind="stable")
    end = np.searchsorted(lo[o, ax], hi[o, ax], side="left")
    cnt = np.maximum(end - np.arange(len(o)) - 1, 0)
    if not cnt.sum():
        return np.zeros((0, 2), np.int64)
    ii = np.repeat(np.arange(len(o)), cnt)
    jj = ii + 1 + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    a, b = o[ii], o[jj]
    keep = np.all((lo[a] < hi[b]) & (lo[b] < hi[a]), 1)
    return np.stack([a[keep], b[keep]], 1)


def brute(lo, hi):
    """Return the overlapping pairs as an unordered set."""
    m = np.all((lo[:, None] < hi[None]) & (lo[None] < hi[:, None]), 2)
    i, j = np.nonzero(np.triu(m, 1))
    return {(int(a), int(b)) for a, b in zip(i, j, strict=True)}


def lattice(n=(6, 4, 6), pitch=1.01, edge=1.0, fiber=0.1):
    """Return the boxes of a block of cubes and of the fibers through it along each axis, as sfgd-optics places
    them (each fiber across its line of cubes, off the cubes' axes)."""
    lo, hi = [], []
    c = [(np.arange(k) + 0.5 - k / 2) * pitch for k in n]
    for x in c[0]:
        for y in c[1]:
            for z in c[2]:
                lo.append(np.array([x, y, z]) - edge / 2)
                hi.append(np.array([x, y, z]) + edge / 2)
    shift = [(0, 0.2, 0.2), (0.2, 0, -0.2), (-0.2, -0.2, 0)]
    for axis in range(3):
        u, w = (k for k in range(3) if k != axis)
        for a in c[u]:
            for b in c[w]:
                p = np.zeros(3)
                p[u], p[w] = a + shift[axis][u], b + shift[axis][w]
                half = np.full(3, fiber / 2)
                half[axis] = n[axis] * pitch / 2 + 2.0
                lo.append(p - half)
                hi.append(p + half)
    return np.array(lo), np.array(hi)


def shuffled(lo, hi, rng):
    p = rng.permutation(len(lo))
    return lo[p], hi[p]


def plates(rng, n=2000, m=100):
    """Return small boxes in a unit cube and plates across it, thin along x: in the tiles of the small boxes, each
    plate would be in thousands of them."""
    lo = rng.uniform(0, 1, (n, 3))
    p, q = np.zeros((m, 3)), np.ones((m, 3))
    p[:, 0] = rng.uniform(0, 1, m)
    q[:, 0] = p[:, 0] + 0.001
    return np.r_[lo, p], np.r_[lo + 0.01, q]


@pytest.mark.parametrize("seed", range(6))
def test_random_boxes(seed):
    rng = np.random.default_rng(seed)
    n = 400
    lo = rng.uniform(-1, 1, (n, 3))
    hi = lo + rng.exponential(0.05, (n, 3)) * rng.choice([1, 1, 1, 20], (n, 3))
    lo[::17, 1] = hi[::17, 1] = 0.3  # flat boxes
    lo[5], hi[5] = lo[4], hi[4]  # a duplicate
    got = ownership._sweep(lo, hi)
    assert np.array_equal(got, plain_sweep(lo, hi))
    assert {tuple(map(int, sorted(p))) for p in got} == brute(lo, hi)
    assert len(got) == len(brute(lo, hi))


@pytest.mark.parametrize("seed", range(3))
def test_cubes_and_fibers(seed):
    lo, hi = shuffled(*lattice(), np.random.default_rng(seed))
    got = ownership._sweep(lo, hi)
    assert np.array_equal(got, plain_sweep(lo, hi))
    assert len(got) == len(brute(lo, hi))


@pytest.mark.parametrize("seed", range(2))
def test_plates_across_the_tiles(seed):
    rng = np.random.default_rng(seed)
    lo, hi = shuffled(*plates(rng), rng)
    got = ownership._sweep(lo, hi)
    assert np.array_equal(got, plain_sweep(lo, hi))
    assert len(got) == len(brute(lo, hi))


def test_degenerate():
    lo = np.zeros((5, 3))
    hi = np.ones((5, 3))
    hi[:, 2] = 0.0  # every box flat in z: no open overlap
    assert len(ownership._sweep(lo, hi)) == 0
    assert np.array_equal(ownership._sweep(lo[:1], hi[:1]), np.zeros((0, 2), np.int64))
    hi[:, 2] = 1.0  # five equal boxes
    assert np.array_equal(ownership._sweep(lo, hi), plain_sweep(lo, hi))
    assert len(ownership._sweep(lo, hi)) == 10
