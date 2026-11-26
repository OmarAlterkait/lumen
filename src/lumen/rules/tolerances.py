"""Every tolerance of lumen's geometry and navigation, in one place, with the error each one covers or the choice it is.

Two error models; a tolerance either follows from one of them or is a choice, said so here.

**1. float32 at run time: the band.** The navigator works in float32; a root and its point each carry about one ulp
per operation. The band at a coordinate scale (m) is ``band(scale) = BAND_ULPS eps32 scale + BAND_FLOOR``
(``lumen.rules.navigation.band``: ``BAND`` = 9.5e-7 per metre). From it:

- in bands: ``K_ON`` (a face this close perpendicularly is the one the photon sits on; also the contact tolerance),
  ``ZERO_STEP_BANDS`` (a crossing this close is a zero-length step: below one ulp), ``FACET_BEHIND_BANDS`` (a start
  this far behind a facet is on it), ``PUSH_DISTANCE_TOLERANCES`` (the stuck rule's push), ``RELOCATE_BANDS`` (the
  relocation's point, past the face met: two ulps, so it is past every face coincident with that one to float32 and
  short of the thinnest layer the navigator resolves, one band);
- in float32 ulps of the quantity they guard: ``FACET_SHORT_ULPS`` (a mesh hit's t, short of its plane),
  ``TANGENT_ULPS`` (a discriminant this close to its terms is a tangency), ``TORUS_BRACKET_ULPS`` (the torus bracket
  sphere's pad);
- pads, relative to the extent they pad, each at least one band per metre (a unit test checks it):
  ``WORLD_BOX_PAD_REL`` (and ``WORLD_BOX_PAD_ABS``, the band's floor), ``KIND_BOX_PAD_REL``, ``TORUS_PAD_REL``;
- floors and ends, far outside every value the navigator's float32 arithmetic produces (a normal float32 is above
  1.2e-38; a coordinate below 1e29 m): ``DIVISOR_FLOOR`` (a divisor, a normal's length, a square root's argument
  below it is it), ``TINY_DIR`` (the slab tests' floor: a direction component below it is replaced by +-it in the
  navigator's boxes and parallel to the slab in a leaf's box), ``BIG`` (|t| at or
  beyond it is an unbounded end, not a crossing; the "no crossing" is lumen.units.HUGE, 1e30);
- ``BISECT_F64``, ``BISECT_F32``: the torus bisections per dtype, the mantissa (53, 24) and a margin: from a bracket of
  width W they reach one ulp of the root while W / |root| is below 2^11 (float64) or 2^16 (float32).

``BAND_ULPS`` itself is a margin of a few roundings ("a few ulps" per operation of the root and its point), not a
count of operations; ``BAND_FLOOR`` is a choice (a point at the origin), not derived from eps32.

**2. float64 at build time: k eps64 scale.** The ownership proofs, the lowering and the oracle work in float64:

- ``POSE_ULPS`` (64 ulps of the coordinate: the rounding of composed poses);
- at metre scale, each at least ``POSE_ULPS`` eps64 per metre (1.4e-14 m; a unit test checks it): ``SAME_POSE``
  (a pose composed with its inverse), ``HOLE_POSE_TOL`` (a sibling filling a hole), ``BOX_PAD`` (the oracle's
  instance boxes), ``LINEAR_REL``, ``TORUS_AXIAL``, ``PARALLEL``, ``SLOPE_REL`` (a coefficient this small relative
  to its scale is zero);
- the solids' equality tests: ``FULL_PHI_TOL``, ``ANGLE_TOL`` (1.4 ulp of pi), ``CYLINDER_REL``, ``MERGE_REL``,
  ``SHARE_TOL``, ``DEGENERATE_FACE`` (a few to a few thousand eps64 of the quantity compared); ``Z_TOL`` and
  ``CONVEX_TOL`` are absolute (a length in m, an eigenvalue of an unnormalised A), below one eps64 ulp of a coordinate
  beyond 4.5 m, so an exact-equality test there; ``VERTEX_TOL``, ``TWIST_TOL``, ``HALF_TURN_MARGIN`` (1e-9: 4.5e6
  eps64) are choices of a decimal input's resolution, not float64 error bounds;
- ``FLOAT64_TINY`` (a float64 coefficient or scale below it is zero), ``CUT_NORMAL_FLOOR`` (a cut tube's plane
  normal |n_z| floor in its z extent), ``TINY_HALF`` (a leaf box's half-size floor): at or below ``SAME_POSE``;
- ``POSE_ATOL``: a user-written rotation (decimal text) is orthonormal, and two poses of a binding are equal, to this
  (defined in ``lumen.format.check``, the bottom layer, and read here);
  ``GIMBAL_COS``: an angle extraction is at the gimbal lock where cos y is at or below it (its error ~ eps64 / cos y);
- the float64 oracle's (``lumen.compare.oracle``): ``ORACLE_MERGE`` (candidate points this close are one point),
  ``ORACLE_SEGMENT_PAD``, ``ORACLE_TORUS_FLOOR`` (= ``SAME_POSE``), ``ORACLE_ROOT_IMAG`` (a torus root is real below
  this times its scale: ~ k sqrt(eps64) for a double root), ``ORACLE_NEWTON_STEPS`` (quadratic convergence from 1e-6
  reaches eps64 in 2), ``ORACLE_AMBIGUOUS`` (about one band at 1 m), ``ORACLE_F32_BALL`` (the band's ulps of |x|).

**3. Choices, not error bounds.**

- ``SPLITS``, ``PIECES``: the containment proof's budget (bisections of a box; undecided pieces kept per pair):
  beyond it a pair stays undecided (and is ordered by placement, as Geant4 would).
- ``GRAZE``: the shallowest angle the start's face is still recognised at, asin(1 / GRAZE) = 0.22 deg.
- ``ZERO_STEPS_BEFORE_PUSH``, ``ZERO_STEPS_BEFORE_KILL``: Geant4's G4Navigator thresholds (10, 25).
- ``CONTACT``: Geant4's length tolerance (kCarTolerance, 1e-9 mm) as the equality of two GDML lengths.
- ``HOLE_SEARCH``: the radius a sibling filling a hole is searched within (a search, not a rounding).
- ``MAX_SCAN_PER_LEAF``, ``MAX_SCAN_EXTRA``: the CSG scan's cap, above what it needs (at most 2 intervals per leaf:
  4 endpoints, plus the start's probe and the exit, 4 n + 2 < 4 n + 5): never reached.
- ``MESH_PARALLEL``: a mesh hit's own plane is used only where the ray meets it at |n . d| above this.
- ``IDENT_ATOL``, ``IDENT_RTOL``: a rotation this close to the identity is no BVH frame group (grouping, not a result).
- ``HOLE_KEY_DECIMALS``: a hole's position rounded to this many decimals (m) names it (``HOLE_POSE_TOL``'s 1e-9).
- ``ORACLE_MAX_SKIPS``: invisible label changes the oracle passes before a ray gives up (the referee's budget).
"""

from lumen.format.check import (  # noqa: F401 (the format layer's own, which imports nothing above it)
    GIMBAL_COS,
    POSE_ATOL,
)

# ---------------------------------------------------------------------------------- 1. float32 at run time: the band
BAND_ULPS = 8.0  # the band: this many float32 ulps of the coordinates
BAND_FLOOR = 1e-9  # m: the band's floor (a point at the origin)
K_ON = 4  # a face within this many bands of the start (perpendicular) is AT it
RELOCATE_BANDS = 0.25  # the relocation's point, this many bands past the face met (two ulps: see the band's rules)
ZERO_STEP_BANDS = 0.125  # a crossing within this many bands of the start is a zero-length step (below 1 ulp)
FACET_BEHIND_BANDS = 0.25  # a start behind a facet's plane by at most this many bands is on it
PUSH_DISTANCE_TOLERANCES = 8  # the push, in bands (Geant4: 100 kCarTolerance)
FACET_SHORT_ULPS = 2.0  # a mesh hit's t from its plane, this many ulps short of it
TANGENT_ULPS = 16.0  # a discriminant within this many ulps of its terms is a tangency, not a chord
TORUS_BRACKET_ULPS = 64.0  # the torus bracket sphere is padded by this many ulps
WORLD_BOX_PAD_REL = 1e-6  # world boxes are padded by this times (|centre| + half) ...
WORLD_BOX_PAD_ABS = 1e-9  # ... plus this (m)
KIND_BOX_PAD_REL = 1e-5  # leaf boxes padded by this times the kind box's extent
TORUS_PAD_REL = 1e-4  # the torus pretests' pad, relative to R + r
DIVISOR_FLOOR = 1e-30  # a divisor (a normal's length, a square root's argument) below this is this
TINY_DIR = 1e-30  # the slab tests: a direction component below this is +-this (navigator) or parallel (leaf box)
BIG = 1e29  # |t| at or beyond this is an unbounded end, not a crossing: one decade below units.HUGE
BISECT_F64, BISECT_F32 = 64, 40  # the torus bisections per dtype: the mantissa (53, 24) plus a margin

# ------------------------------------------------------------------------------ 2. float64 at build time: k eps64 scale
POSE_ULPS = 64  # the float64 rounding of composed poses, in ulps of the coordinate: a deeper overlap is one
SAME_POSE = 1e-12  # m, and in a rotation's entries: the float64 rounding of a pose composed with its inverse
HOLE_POSE_TOL = 1e-9  # a sibling fills a hole when its rotation and position agree to this (position: relative, m)
BOX_PAD = 1e-9  # m: the oracle's instance boxes are padded by this
LINEAR_REL = 1e-12  # a quadric with |alpha| below this times max |A_ij| is linear along the line
TORUS_AXIAL = 1e-12  # a direction with a squared xy part below this is along the torus axis
PARALLEL = 1e-12  # a plane with |n . d| below this is parallel to the ray
SLOPE_REL = 1e-12  # the live t's denominator g . d below this times |g| is no crossing (zero derivative)
FULL_PHI_TOL = 1e-12  # rad: a phi range this close to 2 pi is the full circle
ANGLE_TOL = 1e-15  # rad: theta limits this close to 0, pi / 2 or pi are those angles
CYLINDER_REL = 1e-15  # a frustum whose radii agree to this (relative) is a cylinder
Z_TOL = 1e-15  # m: polycone / polyhedra sections thinner than this are dropped
MERGE_REL = 1e-12  # relative: consecutive polycone sections affine to this are one section
SHARE_TOL = 1e-12  # plane normals / quadric coefficients this close are the same surface (relative)
CONVEX_TOL = 1e-15  # an eigenvalue of A above -this is non-negative (a convex quadric)
DEGENERATE_FACE = 1e-14  # a face whose normal is this small (relative to the size squared) has collapsed
VERTEX_TOL = 1e-9  # relative: a polytope vertex beyond a face plane by more than this is not convex
TWIST_TOL = 1e-9  # m: an arb8 side face twisted by more than this is not planar
HALF_TURN_MARGIN = 1e-9  # rad: a polyhedra side spans at most pi - this in its bounding radius
FLOAT64_TINY = 1e-300  # a float64 coefficient or scale below this is zero
CUT_NORMAL_FLOOR = 1e-12  # a cut-tube plane normal with |n_z| below this is treated as this
TINY_HALF = 1e-12  # m: a leaf box's half-size floor
ORACLE_MERGE = 1e-10  # m: the oracle's candidate points closer than this are one point (coincident faces)
ORACLE_SEGMENT_PAD = 1e-12  # m: an oracle box meets a segment when the slab test overlaps by -this or more
ORACLE_TORUS_FLOOR = 1e-12  # m: the smallest torus scale the oracle's real-root test uses
ORACLE_ROOT_IMAG = 1e-6  # a torus root is real when its imaginary part is below this times the torus scale
ORACLE_NEWTON_STEPS = 3  # Newton polish of the oracle's torus roots
ORACLE_AMBIGUOUS = 1e-6  # m: a second change this close after a visible one marks the oracle's hit ambiguous
ORACLE_F32_BALL = BAND_ULPS * 2.0**-23  # the oracle's start_ambiguous: a ball of this many float32 ulps of |x|_inf

# ------------------------------------------------------------------------------------------- 3. choices, not bounds
SPLITS, PIECES = 24, 1024  # a containment proof: bisections of a box, undecided pieces kept (per pair)
GRAZE = 256  # ... looked for up to this many times farther along the ray (the shallowest angle: asin(1 / 256))
ZERO_STEPS_BEFORE_PUSH = 10  # G4Navigator fActionThreshold_NoZeroSteps
ZERO_STEPS_BEFORE_KILL = 25  # G4Navigator fAbandonThreshold_NoZeroSteps
CONTACT = 1e-12  # m: two GDML lengths that agree to this are equal (1e-9 mm)
HOLE_SEARCH = 1e-6  # m: a sibling filling a hole is searched for this close to the hole's position
MAX_SCAN_PER_LEAF = 4  # first_change scans at most this many endpoints per leaf (plus MAX_SCAN_EXTRA)
MAX_SCAN_EXTRA = 5
MESH_PARALLEL = 1e-6  # a mesh hit's plane is used where |n . d| is above this
IDENT_ATOL = 1e-9  # np.allclose(R, I, atol=1e-9): an identity rotation is no frame group
IDENT_RTOL = 1e-5
HOLE_KEY_DECIMALS = 9  # a hole's position rounded to this many decimals (m) names it
ORACLE_MAX_SKIPS = 64  # invisible changes the oracle passes before a ray gives up
