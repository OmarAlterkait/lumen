"""Three toy detectors built in Python (the tests' ``builder``), with no GDML: a sphere, a cube, a cylinder tank.

Each is a function of its sizes that returns a ``Builder``; ``builder.save(water_sphere(), "ball.lumen")``
converts it. The numbers are illustrative, not any experiment's: water (named "ToyWater": Geant4
gives a material named Water its own Rayleigh table) of index 1.33 (flat tables),
black (no RINDEX, so Geant4 kills light entering it) everywhere else, and flat disc PMTs whose cathode is a
``border_into`` surface counting every photon absorbed entering it (efficiency 1).

Closed forms they serve: with an isotropic point at the sphere's centre and no absorption, each of the sphere's PMTs
counts its solid angle over 4 pi (a disc of radius a at distance d: (1 - d / sqrt(d^2 + a^2)) / 2) of the light,
arriving at d n_g / c.
"""

from __future__ import annotations

from builder import Builder, patterns, table

FLAT_EV = (1.5, 5.0)  # eV (827-248 nm): the knots of the flat tables (held at their value beyond them)
WATER_INDEX = 1.33


def flat(value: float):
    """Return a table of one value on ``FLAT_EV`` (Geant4 reads these properties as tables only)."""
    return table(FLAT_EV, (value, value))


CATHODE = dict(REFLECTIVITY=flat(0.0), EFFICIENCY=flat(1.0))  # absorbed at the cathode, each absorption detected


def _common(b: Builder, abs_m: float | None) -> None:
    water = dict(RINDEX=flat(WATER_INDEX)) | ({} if abs_m is None else dict(ABSLENGTH=flat(abs_m)))
    b.material("ToyWater", density=1.0, atoms={"H": 2, "O": 1}, state="liquid", **water)
    b.material("Black", density=1.0, atoms={"C": 1}, state="solid")
    b.surface("Cathode", "dielectric_metal", "glisur", "polished", 1.0, **CATHODE)


def _pmts(b: Builder, into: str, mother_pv: str, poses, radius: float) -> None:
    b.tube("PMTDisc", rmax=radius, z=0.002)
    b.volume("PMT", "PMTDisc", "Black")
    b.place_many("PMT", into, poses, name="pmt_{copy}")
    b.border("ToCathode", "Cathode", mother_pv, "pmt_*")
    b.sensors("pmt", "border_into", lv="PMT")


def water_sphere(radius: float = 1.0, n_pmt: int = 64, pmt_radius: float = 0.04, abs_m: float | None = None):
    """Return a water sphere (radius m) with ``n_pmt`` disc PMTs on its inside wall, facing the centre."""
    b = Builder("toy_sphere", detector_volume="Ball")
    _common(b, abs_m)
    b.box("WorldBox", 3.0 * radius, 3.0 * radius, 3.0 * radius)
    b.solid("BallOrb", "orb", r=radius)
    b.volume("World", "WorldBox", "Black")
    b.volume("Ball", "BallOrb", "ToyWater")
    b.place("Ball", "World", "ball_pv")
    _pmts(b, "Ball", "ball_pv", patterns.sphere(n_pmt, radius - 0.002), pmt_radius)
    return b


def cube_wall(side: float = 1.0, n: int = 4, reflectivity: float = 0.9, abs_m: float | None = None):
    """Return a water cube (side m) with white ground walls and an n x n wall of square sensor tiles at +z."""
    b = Builder("toy_cube", detector_volume="Cube")
    _common(b, abs_m)
    b.box("WorldBox", 2.0 * side, 2.0 * side, 2.0 * side)
    b.box("CubeBox", side, side, side)
    b.volume("World", "WorldBox", "Black")
    b.volume("Cube", "CubeBox", "ToyWater")
    b.place("Cube", "World", "cube_pv")
    b.surface("White", "dielectric_metal", "unified", "ground", 0.2, REFLECTIVITY=flat(reflectivity))
    b.skin("Walls", "White", "Cube")
    tile = side / n
    b.box("TileBox", 0.95 * tile, 0.95 * tile, 0.002)  # gaps between the tiles, and between them and the wall
    b.volume("Tile", "TileBox", "Black")
    b.place_many("Tile", "Cube", patterns.grid(n, n, tile, side / 2 - 0.002, -1.0), name="tile_{copy}")
    b.border("ToTile", "Cathode", "cube_pv", "tile_*")
    b.sensors("tile", "border_into", lv="Tile")
    return b


def small_tank(radius: float = 1.0, height: float = 2.0, rings: int = 4, per_ring: int = 16,
               cap_pitch: float = 0.4, pmt_radius: float = 0.04, abs_m: float | None = 30.0):  # fmt: skip
    """Return a water cylinder (radius, height m) lined black, with PMTs in rings on its wall and on both caps."""
    b = Builder("toy_tank", detector_volume="Tank")
    _common(b, abs_m)
    b.box("WorldBox", 3.0 * radius, 3.0 * radius, 1.5 * height)
    b.tube("TankTube", rmax=radius, z=height)
    b.volume("World", "WorldBox", "Black")
    b.volume("Tank", "TankTube", "ToyWater")
    b.place("Tank", "World", "tank_pv")
    b.surface("Liner", "dielectric_metal", "unified", "ground", 0.1, REFLECTIVITY=flat(0.1))
    b.skin("Liner", "Liner", "Tank")
    zs = [height * ((k + 0.5) / rings - 0.5) for k in range(rings)]
    poses = patterns.cylinder(radius - 0.002, zs, per_ring)
    poses += patterns.disk(radius - 2.0 * pmt_radius, cap_pitch, height / 2 - 0.002, -1.0)
    poses += patterns.disk(radius - 2.0 * pmt_radius, cap_pitch, -height / 2 + 0.002, 1.0)
    _pmts(b, "Tank", "tank_pv", poses, pmt_radius)
    return b


TOYS = {"toy_sphere": water_sphere, "toy_cube": cube_wall, "toy_tank": small_tank}

__all__ = ["TOYS", "cube_wall", "small_tank", "water_sphere"]
