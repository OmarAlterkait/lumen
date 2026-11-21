"""The physics tier's toys: small GDML detectors written here, converted once per session (the conftest's ``toy``).

Each toy is a GDML string and a recipe. Materials have constant tables over 1-12 eV; lengths are in m.

- ``sphere(surface)``: a water ball (radius ``A`` = 1 m) in a 1 cm sensor shell, touching siblings in a vacuum
  world; the border water -> shell is a detecting dielectric_metal surface (REFLECTIVITY, EFFICIENCY 1), the shell
  the one sensor (``border_into``). Optionally a Rayleigh length, and an opaque occluder (a box, or the same cube as
  a tessellated mesh) on the +z axis.
- ``border_pair()``: two touching water boxes A | B in a vacuum world, a detecting border A -> B only; B the sensor.
- ``rt_boxes(r)``: three touching water boxes A | B | C; A -> B a dielectric_dielectric surface with R = r and T = 0
  (a probability at its domain's edge), B -> C detecting; C the sensor.
- ``self_pair(r, t)``: two touching placements A | B of one water box LV, a polishedfrontpainted skin (R = r,
  T = t) on it; a counting slab beyond B. Leaving A into B Geant4 applies the skin (pre's): the class's own pair.
- ``lobe_plate(sigma, **consts)``: a unified ground metal plate (its three constants at 0 unless set) in a water
  detector volume, hit once by a pencil; two counting slabs above it.
- ``overlap(name)``: one of g4ref's overlap toys (tests/data/overlap_toys), its slab counting every entry.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lumen import run as runner
from lumen import sources
from lumen.build import tracked
from lumen.compare import photons
from lumen.format.sensorsets import SensorGroup
from lumen.recipe import Recipe
from lumen.transport import step as steps

A = 1.0  # m: the sphere's radius
SHELL = 0.01  # m: the sensor shell's thickness
N_WATER = 1.33
GLASS = (0.6, 0.65)  # m: the refracting shell's radii
DATA = Path(__file__).resolve().parents[1] / "data" / "overlap_toys"
GPU = jax.default_backend() == "gpu"
gpu_only = pytest.mark.skipif(not GPU, reason="needs a GPU (the comparison is against the CPU)")


def setup(det, lam_edges=None, step=None) -> photons.Setup:
    """Return a ``photons.Setup`` of a loaded detector on a bare Model: no reach bound, ``step`` (default: no edge
    rows), so a test sets the step config itself (``photons.setup`` is the production Model)."""
    geom = tracked.from_parts(det.parts, cache=True)
    edges = tuple(float(v) for v in (lam_edges or (-np.inf, np.inf)))
    model = steps.Model(geom, det.parts.optics, step or steps.StepConfig())
    return photons.Setup(det, model, len(det.parts.optics.sensor_pvs), edges)


def pencil(x, d, lam, n):
    """Return the Photons of ``n`` photons from ``x`` along ``d`` at ``lam`` nm, t = 0, weight 1, unpolarised."""
    return sources.Point(x, n_photons=float(n), lam=lam, direction=d).emit(None, jax.random.PRNGKey(0), n)


def raw_key(seed, *folds):
    """Return the raw key data of ``PRNGKey(seed)`` folded in with each of ``folds`` (the loop's key operand)."""
    k = jax.random.PRNGKey(seed)
    for f in folds:
        k = jax.random.fold_in(k, f)
    return jax.random.key_data(k) if jnp.issubdtype(k.dtype, jax.dtypes.prng_key) else k


HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<gdml>
<define>
{define}
</define>
<materials>
  <element name="Hyd" formula="H" Z="1"><atom value="1.008"/></element>
{materials}
</materials>
<solids>
{solids}
</solids>
<structure>
{structure}
</structure>
<setup name="Default" version="1.0"><world ref="WorldLV"/></setup>
</gdml>
"""
UNITS = {"RINDEX": "", "ABSLENGTH": "*m", "RAYLEIGH": "*m"}


def table(name, value, unit="", key=""):
    return f'  <matrix name="{name}" coldim="2" values="1.0*eV {value}{unit} 12.0*eV {value}{unit}"/>'


def material(name, define, **props):
    """Return a material's XML, adding its tables to ``define`` (a list)."""
    refs = []
    for k, v in props.items():
        define.append(table(f"{name}_{k}", v, UNITS.get(k, "")))
        refs.append(f'<property name="{k}" ref="{name}_{k}"/>')
    return (f'  <material name="{name}" state="liquid"><D value="1.0"/><fraction n="1.0" ref="Hyd"/>'
            f"{''.join(refs)}</material>")  # fmt: skip


def surface(name, define, finish, value, **props):
    """Return a dielectric_metal unified opticalsurface with constant tables."""
    refs = []
    for k, v in props.items():
        define.append(table(f"{name}_{k}", v, key=k))
        refs.append(f'<property name="{k}" ref="{name}_{k}"/>')
    return (f'  <opticalsurface name="{name}" model="unified" finish="{finish}" type="dielectric_metal" '
            f'value="{value}">{"".join(refs)}</opticalsurface>')  # fmt: skip


def shell(name, r_in, r_out):
    """Return a spherical shell as orb minus orb (quadric faces only: no theta or phi planes)."""
    return [
        f'  <orb name="{name}_out" r="{r_out}" lunit="m"/>',
        f'  <orb name="{name}_in" r="{r_in}" lunit="m"/>',
        f'  <subtraction name="{name}"><first ref="{name}_out"/><second ref="{name}_in"/></subtraction>',
    ]


def cube_mesh(name, half_m, centre_m):
    """Return a tessellated cube (12 triangles, outward) and its vertex positions."""
    c, h = [x * 1000.0 for x in centre_m], half_m * 1000.0
    v = {f"{name}_v{i}": (c[0] + (h if i & 1 else -h), c[1] + (h if i & 2 else -h), c[2] + (h if i & 4 else -h))
         for i in range(8)}  # fmt: skip
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    tri = "".join(f'<triangular vertex1="{name}_v{a}" vertex2="{name}_v{b}" vertex3="{name}_v{cc}" type="ABSOLUTE"/>'
                  for f in faces for a, b, cc in ((f[0], f[1], f[2]), (f[0], f[2], f[3])))  # fmt: skip
    pos = [f'  <position name="{k}" unit="mm" x="{x}" y="{y}" z="{z}"/>' for k, (x, y, z) in v.items()]
    return f'  <tessellated name="{name}" aunit="deg" lunit="mm">{tri}</tessellated>', pos


def sphere(r=0.3, finish="polished", sigma_alpha=0.0, absorption=2.0, rayleigh=None, occluder=None, glass=False,
           groupvel=None):  # fmt: skip
    """Return (GDML, recipe) of the sensor sphere.

    ``occluder``: None, or ("box" | "mesh", half side m, z of its centre m); ``glass``: a glass shell (n 1.5, radii
    ``GLASS``) about the centre, inside the water; ``groupvel``: the water's user GROUPVEL (mm / ns, a flat table).
    """
    define = []
    water = dict(RINDEX=N_WATER, ABSLENGTH=absorption) | ({} if rayleigh is None else dict(RAYLEIGH=rayleigh))
    water |= {} if groupvel is None else dict(GROUPVEL=groupvel)
    mats = [material("Liquid", define, **water), material("Vac", define), material("Cathode", define),
            material("Black", define), material("Glass", define, RINDEX=1.5, ABSLENGTH=1e3)]  # fmt: skip
    props = dict(REFLECTIVITY=r, EFFICIENCY=1.0, SPECULARSPIKECONSTANT=0.0, SPECULARLOBECONSTANT=0.0,
                 BACKSCATTERCONSTANT=0.0)  # fmt: skip
    solids = [f'  <orb name="WorldS" r="{A + 0.5}" lunit="m"/>', f'  <orb name="WaterS" r="{A}" lunit="m"/>',
              *shell("ShellS", A, A + SHELL), surface("CathSurface", define, finish, sigma_alpha, **props)]  # fmt: skip
    occ, inner = "", ""
    if occluder is not None:
        kind, half, z = occluder
        if kind == "box":
            solids.append(f'  <box name="OccS" x="{2 * half}" y="{2 * half}" z="{2 * half}" lunit="m"/>')
            inner = f'<physvol name="pvOcc"><volumeref ref="OccLV"/><position name="po" z="{z}" unit="m"/></physvol>'
        else:
            mesh, pos = cube_mesh("OccS", half, (0.0, 0.0, z))
            solids.append(mesh)
            define += pos
            inner = '<physvol name="pvOcc"><volumeref ref="OccLV"/></physvol>'
        occ = '  <volume name="OccLV"><materialref ref="Black"/><solidref ref="OccS"/></volume>\n'
    if glass:
        solids += shell("GlassS", *GLASS)
        occ += '  <volume name="GlassLV"><materialref ref="Glass"/><solidref ref="GlassS"/></volume>\n'
        inner += '<physvol name="pvGlass"><volumeref ref="GlassLV"/></physvol>'
    structure = (f"{occ}  <volume name=\"WaterLV\"><materialref ref=\"Liquid\"/><solidref ref=\"WaterS\"/>{inner}"
                 "</volume>\n"
                 '  <volume name="CathLV"><materialref ref="Cathode"/><solidref ref="ShellS"/></volume>\n'
                 '  <volume name="WorldLV"><materialref ref="Vac"/><solidref ref="WorldS"/>\n'
                 '    <physvol name="pvWater"><volumeref ref="WaterLV"/></physvol>\n'
                 '    <physvol name="pvCath"><volumeref ref="CathLV"/></physvol>\n  </volume>\n'
                 '  <bordersurface name="CathBorder" surfaceproperty="CathSurface"><physvolref ref="pvWater"/>'
                 '<physvolref ref="pvCath"/></bordersurface>')  # fmt: skip
    gdml = HEAD.format(define="\n".join(define), materials="\n".join(mats), solids="\n".join(solids),
                       structure=structure)  # fmt: skip
    group = SensorGroup("cath", rule="border_into", lv="CathLV", settings={"entry_efficiency": 1.0})
    return gdml, Recipe("sphere", "WorldLV", groups=(group,))


def border_pair():
    """Return (GDML, recipe) of two touching water boxes A | B (0.5 m cubes at x = -0.25, +0.25), border A -> B."""
    define = []
    mats = [material("WaterA", define, RINDEX=N_WATER, ABSLENGTH=1e3),
            material("WaterB", define, RINDEX=N_WATER, ABSLENGTH=1e3), material("Vac", define)]  # fmt: skip
    solids = ['  <box name="WorldS" x="4" y="4" z="4" lunit="m"/>',
              '  <box name="CubeS" x="0.5" y="0.5" z="0.5" lunit="m"/>',
              surface("Detect", define, "polished", 0.0, REFLECTIVITY=0.0, EFFICIENCY=1.0)]  # fmt: skip
    structure = ('  <volume name="ALV"><materialref ref="WaterA"/><solidref ref="CubeS"/></volume>\n'
                 '  <volume name="BLV"><materialref ref="WaterB"/><solidref ref="CubeS"/></volume>\n'
                 '  <volume name="WorldLV"><materialref ref="Vac"/><solidref ref="WorldS"/>\n'
                 '    <physvol name="pvA"><volumeref ref="ALV"/><position name="pa" x="-0.25" unit="m"/></physvol>\n'
                 '    <physvol name="pvB"><volumeref ref="BLV"/><position name="pb" x="0.25" unit="m"/></physvol>\n'
                 "  </volume>\n"
                 '  <bordersurface name="AB" surfaceproperty="Detect"><physvolref ref="pvA"/><physvolref ref="pvB"/>'
                 "</bordersurface>")  # fmt: skip
    gdml = HEAD.format(define="\n".join(define), materials="\n".join(mats), solids="\n".join(solids),
                       structure=structure)  # fmt: skip
    group = SensorGroup("b", rule="border_into", lv="BLV", settings={"entry_efficiency": 1.0})
    return gdml, Recipe("border_pair", "WorldLV", groups=(group,))


def junction():
    """Return (GDML, recipe) of a glass box (n 1.5, 0.2 m, at the origin) touching a steel slab (no RINDEX, a
    polished metal skin with R 1) on its +x face (x = 0.1 m), in water; a water shell (r 0.4-0.41 m) counts."""
    define = []
    mats = [material("WaterJ", define, RINDEX=N_WATER, ABSLENGTH=1e3), material("GlassJ", define, RINDEX=1.5,
            ABSLENGTH=1e3), material("Steel", define), material("Vac", define)]  # fmt: skip
    solids = ['  <box name="WorldS" x="1.2" y="1.2" z="1.2" lunit="m"/>',
              '  <box name="TankS" x="1" y="1" z="1" lunit="m"/>',
              '  <box name="GlassS" x="0.2" y="0.2" z="0.2" lunit="m"/>',
              '  <box name="SteelS" x="0.02" y="0.2" z="0.2" lunit="m"/>', *shell("ShellS", 0.4, 0.41),
              surface("Refl", define, "polished", 0.0, REFLECTIVITY=1.0)]  # fmt: skip
    structure = (
        '  <volume name="GlassLV"><materialref ref="GlassJ"/><solidref ref="GlassS"/></volume>\n'
        '  <volume name="SteelLV"><materialref ref="Steel"/><solidref ref="SteelS"/></volume>\n'
        '  <volume name="ShellLV"><materialref ref="WaterJ"/><solidref ref="ShellS"/></volume>\n'
        '  <volume name="TankLV"><materialref ref="WaterJ"/><solidref ref="TankS"/>\n'
        '    <physvol name="pvGlass"><volumeref ref="GlassLV"/></physvol>\n'
        '    <physvol name="pvSteel"><volumeref ref="SteelLV"/><position name="ps" x="0.11" unit="m"/>'
        "</physvol>\n"
        '    <physvol name="pvShell"><volumeref ref="ShellLV"/></physvol>\n  </volume>\n'
        '  <volume name="WorldLV"><materialref ref="Vac"/><solidref ref="WorldS"/>\n'
        '    <physvol name="pvTank"><volumeref ref="TankLV"/></physvol>\n  </volume>\n'
        '  <skinsurface name="ReflSkin" surfaceproperty="Refl"><volumeref ref="SteelLV"/></skinsurface>'
    )
    gdml = HEAD.format(define="\n".join(define), materials="\n".join(mats), solids="\n".join(solids),
                       structure=structure)  # fmt: skip
    group = SensorGroup("shell", rule="any_entry", lv="ShellLV", settings={"entry_efficiency": 1.0})
    return gdml, Recipe("junction", "TankLV", groups=(group,))


def rt_boxes(r=0.5, absorption=1e3):
    """Return (GDML, recipe) of three touching water boxes A | B | C (0.5 m cubes at x = -0.5, 0, 0.5 m).

    The border A -> B is a polished dielectric_dielectric with REFLECTIVITY r and TRANSMITTANCE 0 (DIELECTRIC_RT,
    T at the edge of its domain), B -> C detects every photon (C the sensor). A pencil along +x from A's centre
    crosses A -> B once; with one index on both sides Fresnel refracts it all: Q / N = (r + T) exp(-0.75 / L).
    """
    define = []
    mats = [material(f"Water{n}", define, RINDEX=N_WATER, ABSLENGTH=absorption) for n in "ABC"]
    mats.append(material("Vac", define))
    for k, v in (("RT_R", r), ("RT_T", 0.0), ("Det_R", 0.0), ("Det_E", 1.0)):
        define.append(table(k, v))
    solids = ['  <box name="WorldS" x="4" y="4" z="4" lunit="m"/>',
              '  <box name="CubeS" x="0.5" y="0.5" z="0.5" lunit="m"/>',
              '  <opticalsurface name="RT" model="glisur" finish="polished" type="dielectric_dielectric" value="1">'
              '<property name="REFLECTIVITY" ref="RT_R"/><property name="TRANSMITTANCE" ref="RT_T"/></opticalsurface>',
              '  <opticalsurface name="Detect" model="unified" finish="polished" type="dielectric_metal" value="0">'
              '<property name="REFLECTIVITY" ref="Det_R"/><property name="EFFICIENCY" ref="Det_E"/>'
              "</opticalsurface>"]  # fmt: skip
    vols = "".join(f'  <volume name="{n}LV"><materialref ref="Water{n}"/><solidref ref="CubeS"/></volume>\n'
                   for n in "ABC")  # fmt: skip
    pvs = "".join(f'    <physvol name="pv{n}"><volumeref ref="{n}LV"/><position name="p{n}" x="{x}" unit="m"/>'
                  "</physvol>\n" for n, x in (("A", -0.5), ("B", 0.0), ("C", 0.5)))  # fmt: skip
    structure = (f'{vols}  <volume name="WorldLV"><materialref ref="Vac"/><solidref ref="WorldS"/>\n{pvs}  </volume>\n'
                 '  <bordersurface name="AB" surfaceproperty="RT"><physvolref ref="pvA"/><physvolref ref="pvB"/>'
                 "</bordersurface>\n"
                 '  <bordersurface name="BC" surfaceproperty="Detect"><physvolref ref="pvB"/><physvolref ref="pvC"/>'
                 "</bordersurface>")  # fmt: skip
    gdml = HEAD.format(define="\n".join(define), materials="\n".join(mats), solids="\n".join(solids),
                       structure=structure)  # fmt: skip
    group = SensorGroup("c", rule="border_into", lv="CLV", settings={"entry_efficiency": 1.0})
    return gdml, Recipe("rt_boxes", "WorldLV", groups=(group,))


def self_pair(r=0.3, t=0.5, absorption=1e3):
    """Return (GDML, recipe) of two touching placements A | B of one water box LV (0.5 m cubes at x = -0.25, +0.25 m)
    in a world of index 1, and a counting slab S (x in [0.75, 0.85] m) beyond B.

    The box LV's skin: dielectric_dielectric polishedfrontpainted, REFLECTIVITY r, TRANSMITTANCE t (straight through,
    specular reflection, the rest absorbed). A and B are one class; the crossing A -> B is the class table's diagonal:
    Geant4 applies the skin there (pre's skin, A not entering a daughter), as at A's and B's faces on the world.
    """
    define = []
    mats = [material("Liq", define, RINDEX=N_WATER, ABSLENGTH=absorption), material("Air1", define, RINDEX=1.0)]
    for k, v in (("Paint_R", r), ("Paint_T", t)):
        define.append(table(k, v))
    solids = ['  <box name="WorldS" x="4" y="4" z="4" lunit="m"/>',
              '  <box name="CubeS" x="0.5" y="0.5" z="0.5" lunit="m"/>',
              '  <box name="SlabS" x="0.1" y="1" z="1" lunit="m"/>',
              '  <opticalsurface name="Paint" model="unified" finish="polishedfrontpainted" '
              'type="dielectric_dielectric" value="0"><property name="REFLECTIVITY" ref="Paint_R"/>'
              '<property name="TRANSMITTANCE" ref="Paint_T"/></opticalsurface>']  # fmt: skip
    structure = ('  <volume name="BoxLV"><materialref ref="Liq"/><solidref ref="CubeS"/></volume>\n'
                 '  <volume name="SLV"><materialref ref="Air1"/><solidref ref="SlabS"/></volume>\n'
                 '  <volume name="WorldLV"><materialref ref="Air1"/><solidref ref="WorldS"/>\n'
                 '    <physvol name="pvA"><volumeref ref="BoxLV"/><position name="pa" x="-0.25" unit="m"/></physvol>\n'
                 '    <physvol name="pvB"><volumeref ref="BoxLV"/><position name="pb" x="0.25" unit="m"/></physvol>\n'
                 '    <physvol name="pvS"><volumeref ref="SLV"/><position name="ps" x="0.8" unit="m"/></physvol>\n'
                 "  </volume>\n"
                 '  <skinsurface name="PaintSkin" surfaceproperty="Paint"><volumeref ref="BoxLV"/>'
                 "</skinsurface>")  # fmt: skip
    gdml = HEAD.format(define="\n".join(define), materials="\n".join(mats), solids="\n".join(solids),
                       structure=structure)  # fmt: skip
    group = SensorGroup("slab", rule="any_entry", lv="SLV", settings={"entry_efficiency": 1.0})
    return gdml, Recipe("self_pair", "WorldLV", groups=(group,))


def lobe_plate(sigma=0.2, **consts):
    """Return (GDML, recipe) of a unified ground metal plate in a water detector volume, with two counting slabs above.

    The plate's skin: dielectric_metal, unified, ground, sigma_alpha ``sigma``, REFLECTIVITY 0.9 and the three
    constants (SPECULARSPIKE / SPECULARLOBE / BACKSCATTER, default 0: every reflection Lambertian; ``consts`` sets
    them by Geant4 key). A pencil from (-0.6, 0, 0.3) m along (0.6, 0, -0.8) hits it once; a reflected photon reaches
    a slab or leaves the volume, never the plate again, so the charge is linear in each constant.
    """
    define = []
    mats = [material("Water", define, RINDEX=N_WATER, ABSLENGTH=1e3), material("Metal", define),
            material("SlabA", define, RINDEX=N_WATER, ABSLENGTH=1e3),
            material("SlabB", define, RINDEX=N_WATER, ABSLENGTH=1e3), material("Vac", define)]  # fmt: skip
    props = dict(REFLECTIVITY=0.9, SPECULARSPIKECONSTANT=0.0, SPECULARLOBECONSTANT=0.0, BACKSCATTERCONSTANT=0.0)
    solids = ['  <box name="WorldS" x="6" y="6" z="6" lunit="m"/>',
              '  <box name="WaterS" x="4" y="4" z="4" lunit="m"/>',
              '  <box name="PlateS" x="2" y="2" z="0.02" lunit="m"/>',
              '  <box name="SlabS" x="1" y="1" z="0.05" lunit="m"/>',
              surface("Plate", define, "ground", sigma, **(props | consts))]  # fmt: skip
    structure = ('  <volume name="PlateLV"><materialref ref="Metal"/><solidref ref="PlateS"/></volume>\n'
                 '  <volume name="SlabALV"><materialref ref="SlabA"/><solidref ref="SlabS"/></volume>\n'
                 '  <volume name="SlabBLV"><materialref ref="SlabB"/><solidref ref="SlabS"/></volume>\n'
                 '  <volume name="WaterLV"><materialref ref="Water"/><solidref ref="WaterS"/>\n'
                 '    <physvol name="pvPlate"><volumeref ref="PlateLV"/><position name="pp" z="-0.5" unit="m"/>'
                 "</physvol>\n"
                 '    <physvol name="pvSlabA"><volumeref ref="SlabALV"/><position name="pa" x="0.9" z="0.5" unit="m"/>'
                 "</physvol>\n"
                 '    <physvol name="pvSlabB"><volumeref ref="SlabBLV"/><position name="pb" x="-0.9" z="0.5" unit="m"/>'
                 "</physvol>\n  </volume>\n"
                 '  <volume name="WorldLV"><materialref ref="Vac"/><solidref ref="WorldS"/>\n'
                 '    <physvol name="pvWater"><volumeref ref="WaterLV"/></physvol>\n  </volume>\n'
                 '  <skinsurface name="PlateSkin" surfaceproperty="Plate"><volumeref ref="PlateLV"/>'
                 "</skinsurface>")  # fmt: skip
    gdml = HEAD.format(define="\n".join(define), materials="\n".join(mats), solids="\n".join(solids),
                       structure=structure)  # fmt: skip
    group = SensorGroup("slab", rule="any_entry", lv_pattern="^Slab", settings={"entry_efficiency": 1.0})
    return gdml, Recipe("lobe_plate", "WaterLV", groups=(group,))


def overlap(name):
    """Return (GDML, recipe) of one of g4ref's overlap toys (their world kept, the slab volS counting)."""
    group = SensorGroup("slab", rule="any_entry", lv="volS", settings={"entry_efficiency": 1.0})
    return (DATA / f"{name}.gdml").read_text(), Recipe(name, "volWorld", groups=(group,))


# the toys' roulette ε: below float32's resolution of a photon's start weight, so the closed forms hold to float32
# (a photon ends only at weight 0, by the roulette)
EXACT = 2.0**-24


def simulate(s, params, rows, *, reference=None, seeds=None, controls=None, seed=7, backend="auto", spec=None):
    """Return the outputs of a run of ``rows`` through toy setup ``s`` at packed ``params``; with ``seeds`` K, the rows
    split into K equal runs of their own seeds (seed + j), each output stacked (K, ...): K independent estimates."""
    spec = spec or s.loop()
    ops = photons.operands(s, params, reference, controls or steps.Controls(roulette=EXACT))

    def one(r, j):
        return runner.run(spec, s.model, ops, r, raw_key(seed + j), max_vertices=20000, backend=backend).outputs

    if seeds is None:
        return one(rows, 0)
    return stack_seeds(lambda r, j: one(r, j), rows, seeds)


def stack_seeds(run, rows, k: int) -> dict:
    """Return ``run(rows_j, j)`` of the rows split into ``k`` equal parts, each output stacked (k, ...)."""
    n = rows.x.shape[0]
    if n % k:
        raise ValueError(f"stack_seeds: {n} rows do not split into {k} seeds")
    m = n // k

    def part(j):
        return jax.tree_util.tree_map(lambda a: a[j * m : (j + 1) * m] if jnp.ndim(a) and a.shape[0] == n else a, rows)

    outs = [run(part(j), j) for j in range(k)]
    return {key: jnp.stack([o[key] for o in outs]) for key in outs[0]}
