"""Every number of lumen's optics, transport, readout and checks that is no public constant or Geant4 value.

In one place, with the error it bounds or the choice it is (the geometry's own: ``lumen.rules.tolerances``).

Each entry is one of: **derived** (from float32's resolution, eps32 = 2^-23, or from a stated error model),
**resolution** (a table's grid, with its measured interpolation error), or **choice** (what it trades, and that no
expectation depends on it, or the input it refuses).

**Accepted input ranges.** Geant4 accepts any facet width and any Mie g. lumen refuses only what float32 cannot
represent or a table does not cover:

- ``LOBE_WIDTH_MIN`` (derived): float32 resolves an angle taken from a dot product near 1 to sqrt(eps32) = 3.45e-4
  rad; a lobe narrower than ~3 of that (1e-3 rad = 2.9 sqrt(eps32)) is not represented. It bounds the Mie lobe
  (1 - |g| >= it: ``MIE_G_MAX``) and GLISUR's facet radius (1 - polish >= it, or polish exactly 1: specular).
- ``SIGMA_MIN``, ``SIGMA_MAX`` (resolution; choice): the unified lobe's sigma_alpha is the span of its tables
  (``UNIFIED_TABLE``): 0.25 2^(j / 8), j in [-52, 16] = [0.00276, 1] rad (or exactly 0: specular); below it the tables
  do not reach. The top is G4OpticalSurface's default value, and the width above which Geant4's own draw departs from
  the distribution it states and lumen draws, g(alpha) sin(alpha) on (0, pi/2): GetFacetNormal draws alpha from
  N(0, sigma_alpha) and keeps it where ``u f_max <= sin(alpha)`` and alpha < pi/2, which a negative draw in
  (-2 pi k, -2 pi k + pi), k >= 1, also passes (a facet whose polar angle is alpha + 2 pi k). Their share of the facets
  facing the photon is at most 2.5e-6 at 0.707 rad and 8.5e-4 at 1 rad (grazing incidence; 6.8e-5 at 50 degrees,
  4.9e-6 at normal incidence), and grows fast above: 6.1e-3 at 1.2 rad, 0.15 at 2 rad
  (G4OpBoundaryProcess::GetFacetNormal, 11.3.0; ``tests/unit/test_unified_tables.py``).

**Unified tables** (``lumen.surfaces.facets``; resolution). F(x, log sigma) on ``UNIFIED_TABLE["nx"]`` x nodes by
``UNIFIED_TABLE["dy"]`` steps of log sigma, bicubic Hermite: measured against the float64 quadrature at 136 x 256
off-node points over the span (each cell's midpoint and a random point in it; 128 regular and 128 random x;
``tests/unit/test_unified_tables.py``), max |dF| 4.7e-7 and max |d log zeta| 1.3e-6 (4 and 11 float32 ulps of O(1);
the latter near sigma = 0.75, the interpolation's: the nodes' float32 rounding is 1.4e-8). The
quadrature (Gauss-Legendre of ``order`` points on ``panels_z`` / ``panels_f`` panels) changes by at most 2.2e-16
(Z, relative) and 2.2e-16 (F) when its panels are doubled; the integrands are cut at ``cut_sigmas`` sigma, where
exp(-x^2 / 2) = e^-98 (eps64 is reached at 8.5 sigma). The derivative nodes are central differences of step
``fd_step`` (x and log sigma) and ``fd_end`` (at the x ends): about eps64^(1/3) = 6e-6, the float64 optimum.

**Readout.** ``M_GRID`` (resolution): r(m), the probability that a window of 1 + Poisson(m) photoelectrons is
recorded, is tabulated at m = 0 and ``M_GRID_POINTS`` points geometric in [1e-4, 1e4] and interpolated linearly in m:
measured max |dr| 7.8e-4 (m < 40, threshold and ramp recorded vectors). ``Q_NEGATIVE`` (choice): the charge grid
reaches this far below 0 pe; an SPE density below it is not represented. ``TIME_SPREAD_REACH`` (choice): a time
spread's width (a TTS's ``sigma``) may move to between r and 1 / r times its value set's own (sourced) width; that
range is derived, never declared, and the static time grid resolves its narrow end (``SensorValuesModel.grid``:
nodes sigma r / 4 apart; its wide end sets the grid's reach). It trades the grid's nodes against how far a fit may
move the width; a value outside it is refused.

**Matched indices** (derived): ``INDEX_MATCH``, the relative index difference at or below which a GLISUR facet
refraction keeps the photon's direction (Geant4's refraction between equal indices) and carries no density: its
deflection there is below 128 float32 ulps of a unit vector (1.5e-5 rad), and the facet it implies from the two
directions, (n_a w0 - n_b w1) / |n_a w0 - n_b w1|, is lost to their rounding.

**Float32 floors** (derived): ``FLOOR32``, the floor of a rate, probability, density or divisor the photon loop forms
(and the argument floor of a logarithm): a normal float32 (above 1.2e-38), far below any value of light that matters.
``NORMAL_INCIDENCE``: |wi x n|^2 below this has no plane of incidence, eps32^2 (1.4e-14), the rounding of a cross
product of unit float32 vectors; ``FRESNEL_SMALL``: |z| below this is 0 in the Fresnel square root and its
denominators, about the rounding of a difference of float32 values of order 1 squared.

**Estimator** (``lumen.transport.estimator``; choices: no expectation depends on them, only the variance):
``ROULETTE``, the default roulette's ε (``lumen.Controls``; measured: docs/explanation/known-limits.md, "The roulette's
ε"); ``DEFENSIVE_SHARE`` e, the one defensive mixture's share: a sampled part whose declared parameter can zero a
probability or move a support (``lumen.transport.estimator``: a surface outcome, GLISUR's facet, Mie's lobe, a
scintillation component, a spectrum interval) is drawn from (1 - e) reference + e covering density wherever the
reference does not cover the live density's support. Its one trade: the value pays e of those draws (a weight at
most 1 / (1 - e)), the derivative's weight at a covered draw is at most (its covering density's inverse) / e.

**Inputs.** ``UNIT_TOL`` (choice): a direction given by a user (a source field declared DIRECTION, a source
function's draw) may have | |d| - 1 | up to this: decimal input to about three digits; it is normalised after.

**The diffusion jump** (``lumen.transport.fastpath``; speed only, no result depends on them): ``JUMP_FLOOR``, the floor
of the walk's 1 - (k . e)^2 (a direction along the polarisation has no plane); ``JUMP_STEPS``, the walk's step budget,
below int32's 2^31; ``GRID_COARSEN``, the least factor a clearance grid's spacing grows by while it does not fit
(progress only).

**Checks** (choices). ``GRADIENT_EDGE_STEP`` (``lumen.compare.gradients``) is the one-sided finite difference's step
at a parameter's edge, as a fraction of the central step. The difference's truncation, |FD(h) - FD(h / 4)|, is
added to the gate's tolerance, so the step sets the test's power, not its correctness; 1/16 resolves the steepest
edge family measured (a frame at R = 1: FD(H / 4) -31510, FD(H / 64) -54792 against AD -55285 on one seed). A bulk
lobe's first-use check (``lumen._core.checks.check_lobe``) draws ``LOBE_SAMPLES`` directions into ``LOBE_BINS`` bins
of the cosine and integrates the density per bin by ``LOBE_NODES``-point Gauss-Legendre over ``LOBE_AZIMUTHS``
azimuths; the density integrates to 1 within ``LOBE_NORM_TOL``, and a bin enters the chi^2 when it expects more than
``LOBE_MIN_EXPECTED`` draws (Cochran's rule). ``SOURCE_CHECK_N``: the photons of a source function's first-use check.
``Z_MAX``, ``FD_ROUNDING``, ``NULL_DRAWS``, ``SUITE_TESTS``: the statistical checks' significance, resolution of a
difference, simulated null and count of suite assertions (their reasons: ``lumen.tools.stats``).
``PER_NORMAL_RTOL``, ``PAD_STEP`` (``lumen.build.lookup``): the detection check's float32 tolerance on A_n (derived:
8 eps32) and a padded table's knot spacing past its end.
"""

import math

import numpy as np

EPS32 = float(np.finfo(np.float32).eps)

# float32 floors
FLOOR32 = 1e-30  # a rate, probability, density or divisor below this is this
NORMAL_INCIDENCE = EPS32**2  # |wi x n|^2 below this: no plane of incidence
FRESNEL_SMALL = 1e-12  # |z| below this is 0 in the Fresnel square root and its denominators

# accepted input ranges
LOBE_WIDTH_MIN = 1e-3  # rad: the narrowest lobe, 2.9 sqrt(eps32)
INDEX_MATCH = 128 * EPS32  # |n_a - n_b| / max(n_a, n_b) at or below this: a facet refraction keeps the direction
MIE_G_MAX = 1.0 - LOBE_WIDTH_MIN  # the largest |g| of a Mie lobe (0.999)
GLISUR_R_MIN = LOBE_WIDTH_MIN  # the smallest 1 - polish of a GLISUR facet path
UNIFIED_TABLE = dict(nx=257, y0=math.log(0.25), dy=math.log(2.0) / 8.0, j_lo=-52, j_hi=16, order=16, panels_z=64,
                     panels_f=16, cut_sigmas=14.0, fd_step=1e-5, fd_end=1e-6)  # fmt: skip
SIGMA_MIN = math.exp(UNIFIED_TABLE["y0"] + UNIFIED_TABLE["dy"] * UNIFIED_TABLE["j_lo"])  # rad: 0.00276
SIGMA_MAX = math.exp(UNIFIED_TABLE["y0"] + UNIFIED_TABLE["dy"] * UNIFIED_TABLE["j_hi"])  # rad: 1, Geant4's default

# readout
M_GRID_POINTS = 383  # r(m)'s nodes in [1e-4, 1e4] (and m = 0)
Q_NEGATIVE = 4.0  # pe: the charge grid reaches this far below 0
TIME_SPREAD_REACH = 0.5  # a time spread's width ranges over [r, 1 / r] times its value set's own

# estimator
ROULETTE = 0.01  # the default roulette's epsilon
DEFENSIVE_SHARE = 0.01  # the defensive mixture's share of the covering density

# inputs
UNIT_TOL = 1e-3  # | |d| - 1 | allowed in a user's direction

# the diffusion jump (speed only)
JUMP_FLOOR = 1e-12  # the walk's floor of 1 - (k . e)^2
JUMP_STEPS = 2.0**30  # the walk's step budget, below int32's 2^31
GRID_COARSEN = 1.0 + 2.0**-20  # the least factor a clearance grid's spacing grows by while it does not fit

# checks
GRADIENT_EDGE_STEP = 1.0 / 16  # the one-sided difference's step, as a fraction of the central one
LOBE_SAMPLES = 1 << 14  # the directions a bulk lobe's check draws
LOBE_BINS = 20  # its bins of the cosine
LOBE_NODES = 64  # Gauss-Legendre nodes per bin
LOBE_AZIMUTHS = 16  # azimuths the density is averaged on
LOBE_NORM_TOL = 0.02  # |the density's integral - 1| allowed
LOBE_MIN_EXPECTED = 5.0  # a bin enters the chi^2 when it expects more draws than this (Cochran)
SOURCE_CHECK_N = 256  # photons of a source function's first-use check
Z_MAX = 3.0  # the checks' one significance level, as a normal score (lumen.tools.stats.ALPHA, two-sided)
FD_ROUNDING = 1e-6  # one finite-difference side's float32 rounding relative to its value |F| (lumen.tools.stats)
NULL_DRAWS = 100  # mean_test's simulated null: draws per expected exceedance at its level (p resolved to ~10 %)
SUITE_TESTS = 400  # the test suites' statistical assertions sharing ALPHA (lumen.tools.stats.z_limit): a stated bound
PER_NORMAL_RTOL = 8 * EPS32  # A_n is evaluated in float32: a detection efficiency above it by more is refused
PAD_STEP = 1e-3  # a padded table's knot spacing past its end: this fraction of the last energy, at least this (eV)
