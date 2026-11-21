"""The structural rules of a detector structure, checked in one place for every input (GDML, YAML, a recipe's result).

    check(df)                                  # raises StructureError naming the first broken rule
    check(df, ImportRefused, decisions=False)  # the GDML importer, before a recipe has decided anything

The rules:

- **references resolve**: the world, every volume's solid and material, every placement's volume and mother, a
  boolean's or scaled solid's references, a tessellated solid's mesh, every component of a material or element, every
  skin's surface and volume and every border's surface;
- **one placement tree**: each placement is listed once, in its mother's daughters; no volume holds itself and no
  placement places the world;
- **solids** of Geant4's catalogue (``lumen.format.solids``) have its attributes only, every number finite (a solid
  outside it is the lowering's to refuse, or keep as an assembly), and a positive deltaphi (deltatheta) where Geant4's
  constructor aborts otherwise (``OPEN_ANGLE_ABORTS``; an omitted attribute is Geant4's 0);
- **materials** name a composition (a positive density with components of non-negative shares, or Z and a molar
  mass);
- **properties** are a finite constant, or a table on >= 2 strictly ascending energies with one value per energy;
- **Geant4's binding rules** (``check_bindings``): one skin per logical volume (the source's skins, and a recipe's or
  import rule's, each among themselves: a recipe's skin may replace the source's, ``lumen.build.binding``), a border's
  physvol name carried by exactly one placement, one border per ordered pair;
- with ``decisions``, the recipe's decision: the detector volume is a volume.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from lumen.format import schema
from lumen.format import solids as catalogue

# the format layer's two geometry tolerances (listed with the others in lumen.rules.tolerances, which reads them here)
POSE_ATOL = 1e-9  # a user-written rotation is orthonormal, two poses of a binding equal, to this
GIMBAL_COS = 1e-12  # an angle extraction is at the gimbal lock where cos y is at or below this


class StructureError(ValueError):
    """A detector structure that breaks a structural rule (see the module docstring)."""


def check_bindings(skins, borders, placement_names, error) -> None:
    """Refuse (raising ``error``) two skins on one logical volume, a border physvol name that is not exactly one
    placement's, and a second border on one ordered pair.

    ``skins``: (name, logical volume) pairs; ``borders``: (name, pv1, pv2); ``placement_names``: every placement's
    name, repeats included. G4LogicalSkinSurface keeps one skin per logical volume (the first registered);
    G4LogicalBorderSurface binds a physvol name to the first placement of that name, and one ordered pair to one
    border: lumen refuses what Geant4 would resolve silently.
    """  # noqa: D205
    lvs: dict = {}
    for name, lv in skins:
        if lv in lvs:
            raise error(f"logical volume {lv!r} has two skin surfaces ({lvs[lv]!r}, {name!r}): Geant4 keeps the first "
                        "registered; refusing the ambiguity")  # fmt: skip
        lvs[lv] = name
    count, pairs = Counter(placement_names), set()
    for name, pv1, pv2 in borders:
        for r in (pv1, pv2):
            if count.get(r, 0) != 1:
                raise error(f"border surface {name!r}: physvol {r!r} names {count.get(r, 0)} physical volumes; Geant4 "
                            "binds a border to the first one only, so the border is ambiguous")  # fmt: skip
        if (pv1, pv2) in pairs:
            raise error(f"border surface {name!r}: the pair {pv1!r} -> {pv2!r} has a border already (Geant4 keeps "
                        "the first)")  # fmt: skip
        pairs.add((pv1, pv2))


def check(df: schema.DetectorFile, error=StructureError, *, decisions: bool = True) -> None:
    """Refuse (raising ``error``) a detector structure that breaks a rule of the module docstring.

    ``decisions=False`` skips the recipe's decisions (a GDML file read before its recipe has none).
    """
    _geometry(df, error)
    _solids(df, error)
    _materials(df, error)
    _surfaces(df, error)
    if decisions:
        _decisions(df, error)


def _volumes(df, error) -> dict[int, str]:
    """Check the volumes; return {placement index: the volume whose daughter it is}."""
    if df.world not in df.volumes:
        raise error(f"the world {df.world!r} is not a volume")
    owner: dict[int, str] = {}
    for name, v in df.volumes.items():
        if not v.assembly and v.solid not in df.solids:
            raise error(f"volume {name!r}: solid {v.solid!r} is not a solid")
        if not v.assembly and v.material not in df.materials:
            raise error(f"volume {name!r}: material {v.material!r} is not a material")
        for i in v.daughters:
            if not 0 <= i < len(df.placements):
                raise error(f"volume {name!r}: daughter {i} names no placement (there are {len(df.placements)})")
            if i in owner:
                raise error(f"placement {i} is a daughter of {owner[i]!r} and of {name!r}")
            owner[i] = name
    return owner


def _geometry(df, error) -> None:
    owner = _volumes(df, error)
    for i, p in enumerate(df.placements):
        if p.volume not in df.volumes:
            raise error(f"placement {p.name!r}: volume {p.volume!r} is not a volume")
        if p.mother not in df.volumes:
            raise error(f"placement {p.name!r}: mother {p.mother!r} is not a volume")
        if owner.get(i) != p.mother:
            raise error(f"placement {p.name!r} (index {i}) is not listed in its mother {p.mother!r}'s daughters")
        nums = (*p.position, *p.rotation, *(p.scale or ()))
        if not (len(p.position) == 3 and len(p.rotation) == 3 and all(math.isfinite(x) for x in nums)):
            raise error(f"placement {p.name!r}: position and rotation are three finite numbers each")
        if p.volume == df.world:
            raise error(f"placement {p.name!r} places the world {df.world!r}: the world is placed by no placement")
    _acyclic(df, error)


def _acyclic(df, error) -> None:
    """Refuse a volume that holds itself through its placements (a cycle Geant4 never ends placing)."""
    state: dict[str, int] = {}  # 1: on the walk, 2: done

    def walk(lv, path):
        state[lv] = 1
        for i in df.volumes[lv].daughters:
            child = df.placements[i].volume
            if state.get(child) == 1:
                raise error(f"volume {child!r} holds itself: {' -> '.join([*path, child])}")
            if child not in state:
                walk(child, [*path, child])
        state[lv] = 2

    for lv in df.volumes:
        if lv not in state:
            walk(lv, [lv])


def _solids(df, error) -> None:
    for name, s in df.solids.items():
        known = catalogue.dimensions(s.type) if s.type in catalogue.ATTRIBUTES else {}  # else the lowering decides
        extra = sorted(set(s.params) - set(known)) if known else []
        if extra:
            raise error(f"solid {name!r} ({s.type}): unknown attributes {extra} (known: {sorted(known)})")
        nums = [*s.params.values(), *(x for p in s.planes for x in p), *(x for v in s.vertices for x in v),
                *(x for c in s.sections for x in c), *(s.scale or ())]  # fmt: skip
        if not np.all(np.isfinite(np.asarray(nums, np.float64))):
            raise error(f"solid {name!r}: a number is not finite")
        for k in catalogue.OPEN_ANGLE_ABORTS.get(s.type, ()):
            if not s.params.get(k, 0.0) > 0.0:  # an omitted attribute is G4GDMLReadSolids' 0
                raise error(f"solid {name!r} ({s.type}): {k} {s.params.get(k, 0.0)} (omitted: 0): Geant4's "
                            f"{s.type} constructor aborts at {k} <= 0 (Invalid dphi / dtheta); give it")  # fmt: skip
        for r in s.refs:
            if r not in df.solids:
                raise error(f"solid {name!r}: reference {r!r} is not a solid")
        if s.type == "tessellated" and s.mesh not in df.meshes:
            raise error(f"solid {name!r}: mesh {s.mesh!r} has no triangles")


def _composition(df, name, c: schema.Composition, error) -> None:
    has = c.density_g_cm3 is not None and (c.components or (c.z is not None and c.molar_mass_g_mol is not None))
    if not has:
        raise error(f"material {name!r} names no composition: give a density and its components (mass fractions or "
                    "atom counts), or Z and a molar mass")  # fmt: skip
    if not (math.isfinite(c.density_g_cm3) and c.density_g_cm3 > 0.0):
        raise error(f"material {name!r}: density {c.density_g_cm3} g/cm3: give a positive, finite density")
    for x in c.components:
        if (x.fraction is not None and not x.fraction >= 0.0) or (x.atoms is not None and x.atoms < 0):
            raise error(f"material {name!r}: component {x.ref!r} has a negative share ({x.fraction or x.atoms}): "
                        "give a mass fraction or an atom count >= 0")  # fmt: skip
        table = {"element": df.elements, "material": df.materials}.get(x.kind)
        if table is None or x.ref not in table:
            raise error(f"material {name!r}: component {x.ref!r} is neither an element nor a material")


def _materials(df, error) -> None:
    for name, e in df.elements.items():
        for x in e.isotopes:
            if x.kind != "isotope" or x.ref not in df.isotopes:
                raise error(f"element {name!r}: isotope {x.ref!r} is not an isotope")
    for name, m in df.materials.items():
        _composition(df, name, m.composition, error)
        for key, p in m.properties.items():
            _property(f"material {name!r}.{key}", p, error)


def _property(what: str, p: schema.Property, error) -> None:
    v = np.asarray(p.value, np.float64)
    if p.energy is None:
        if v.ndim != 0 or not np.isfinite(v):
            raise error(f"{what}: a constant is one finite number")
        return
    e = np.asarray(p.energy, np.float64)
    if e.ndim != 1 or len(e) < 2 or np.any(np.diff(e) <= 0.0):
        raise error(f"{what}: energies must be strictly ascending with >= 2 knots (got {e})")
    if v.shape != e.shape:
        raise error(f"{what}: {len(e)} energies but values of shape {v.shape}")
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(e))):
        raise error(f"{what}: a value is not finite")


def _surfaces(df, error) -> None:
    for name, s in df.surfaces.items():
        if not np.isfinite(s.value):
            raise error(f"surface {name!r}: its value is not finite")
        for key, p in s.properties.items():
            _property(f"surface {name!r}.{key}", p, error)
    for k in df.skins:
        if k.surface not in df.surfaces:
            raise error(f"skin {k.name!r} refers to {k.surface!r}, which is not a surface")
        if k.volume not in df.volumes:
            raise error(f"skin {k.name!r}: volume {k.volume!r} is not a volume")
    for b in df.borders:
        if b.surface not in df.surfaces:
            raise error(f"border {b.name!r} refers to {b.surface!r}, which is not a surface")
    source = [(k.name, k.volume) for k in df.skins if k.origin == "source"]
    decided = [(k.name, k.volume) for k in df.skins if k.origin != "source"]
    check_bindings(source, [(b.name, b.pv1, b.pv2) for b in df.borders], [p.name for p in df.placements], error)
    check_bindings(decided, (), (), error)


def _decisions(df, error) -> None:
    if df.detector_volume != df.world and not schema.placements_of(df, df.detector_volume):
        raise error(f"the detector volume {df.detector_volume!r} is no volume, physvol or physvol path placed under "
                    "the world")  # fmt: skip


__all__ = ["StructureError", "check", "check_bindings"]
