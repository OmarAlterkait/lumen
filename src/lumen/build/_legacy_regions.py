"""The legacy optical-region compiler: a library, a volume tree and bindings to the static tables the engine gathers.

Internal, and used only until the Geant4 surface lookup replaces it; carried over unchanged from the baseline (lumen
rebuild/next ec8de54c: lumen.describe's library, bindings and resolve, lumen.classes, lumen.tables, and the
compile / border parts of lumen.adapters.gdml_optics).

Three inputs, all plain Python:

    Library    named media (bulk.Medium + params), surfaces (a legacy surface type + params) and sensor types (a
               detecting type + params)
    Tree       the physical volumes as world instances (``lumen.build.tree``): each PV names its logical volume,
               material and mother PV
    Bindings   attach a library to one tree by EXACT names: materials -> media, skin surfaces (LV -> surface), border
               surfaces (ordered PV pair -> surface), sensors (LV or PV -> sensor type, expected count) and sibling
               adjacency facts

``resolve`` checks all three against each other and fails closed.

An optical class is a set of PVs with the same SYMBOLIC signature

    (library medium, skin surface of its LV, sensor type, border roles = {(partner class, 'out' | 'in', surface)})

so classes never depend on parameter values. A boundary exists only where the class changes. Each adjacent ordered
PV pair (pre, post) -- mother/daughter always, siblings unless declared disjoint -- resolves to a surface type with
Geant4's precedence:

    entering a sensor PV            the sensor's type
    border surface (pre, post)      the ordered pair only; a border SENSOR (leaving the sensor PV pre: WCSim's
                                    AirCathodeSurface Detection, counted on the tube it leaves) is the sensor's type
    skins                           post daughter of pre: post's LV skin, then pre's; otherwise pre's, then post's
    default                         same medium: PASS; both media have an index: DIELECTRIC; else ABSORBER (NoRINDEX)

A surface flagged ``ignored_between_same_material`` between two PVs of the same tree material is PASS (Geant4's
SameMaterial). A DIELECTRIC into an opaque medium is ABSORBER (NoRINDEX); any other type that would put the photon
into an opaque medium is an error. Pairs leaving an opaque medium are never crossed.

Guards (fail closed): two PV pairs of one class pair resolving to different surfaces; a class adjacent to itself
across a non-PASS surface; laws invalid over the wavelength domain. A DETECT deposit goes to the tag of the PV
entered, except across a border sensor, where it goes to the PV left (``deposit_pre``).

``compile`` also packs spectral surface and sensor params ((Law, params) pairs of the shared clamped tables) as
``Optics.kernel_laws`` (one Law per (type, param), knots padded to a common length with clamped tails).
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from lumen.media import bulk, laws
from lumen.surfaces import _legacy as legacy
from lumen.surfaces.outcomes import DETECT, FAR_SIDE

# the baseline's surface-type order (lumen.interface.KINDS); every other type sorts after these, by name
ORDERED_TYPES = (
    legacy.PASS,
    legacy.ABSORBER,
    legacy.DIELECTRIC,
    legacy.DIELECTRIC_RT,
    legacy.DIELECTRIC_RT_SENSOR,
    legacy.REFLECTOR,
    legacy.SENSOR,
    legacy.PHOTOCATHODE,
    legacy.COUNTING,
    legacy.DIELECTRIC_METAL,
)
MAX_EXAMPLES = 3  # example PV pairs kept per crossed class pair
NAMES_SHOWN = 6  # PV names shown per class in the report
PAD_STEP = 1e-3  # a padded table's knot spacing past its end: this fraction of the last energy, at least this (eV)


class DescriptionError(ValueError):
    """A library, tree and bindings that do not fit together."""


class OpticsError(DescriptionError):
    """A description the class compiler cannot turn into tables."""


# ---------------------------------------------------------------------------------------------------------- library
@dataclass(frozen=True, eq=False)
class MediumEntry:
    """A library medium: a bulk.Medium and its params ({"index": ..., "absorption": ..., "<channel>": ...})."""

    medium: bulk.Medium
    params: Mapping


@dataclass(frozen=True, eq=False)
class SurfaceEntry:
    """A library surface: a legacy surface type and its params.

    ``ignored_between_same_material``: Geant4 skips the boundary between two volumes of the SAME material, so a
    polished dielectric_dielectric surface declared there never acts.
    """

    kind: legacy.SurfaceType
    params: Mapping
    ignored_between_same_material: bool = False


@dataclass(frozen=True, eq=False)
class SensorEntry:
    """A library sensor type: a surface type that can DETECT and its params."""

    kind: legacy.SurfaceType
    params: Mapping


@dataclass(frozen=True, eq=False)
class Library:
    """Named media, surfaces and sensor types."""

    media: Mapping[str, MediumEntry] = field(default_factory=dict)
    surfaces: Mapping[str, SurfaceEntry] = field(default_factory=dict)
    sensors: Mapping[str, SensorEntry] = field(default_factory=dict)


def _check_params(what, kind, params):
    have, want = set(params), set(kind.params)
    if have != want:
        raise DescriptionError(f"{what}: kind {kind.name} takes parameters {sorted(want)}, got {sorted(have)} "
                               f"(missing {sorted(want - have)}, unknown {sorted(have - want)})")  # fmt: skip


def check_library(lib: Library):
    """Check the structure of every entry (the laws are validated by the compiler, for the media used)."""
    for name, e in lib.media.items():
        if not isinstance(e, MediumEntry) or not isinstance(e.medium, bulk.Medium):
            raise DescriptionError(f"medium {name!r}: expected a MediumEntry(bulk.Medium, params)")
    for name, e in lib.surfaces.items():
        if not isinstance(e, SurfaceEntry) or not isinstance(e.kind, legacy.SurfaceType):
            raise DescriptionError(f"surface {name!r}: expected a SurfaceEntry(SurfaceType, params)")
        if DETECT in e.kind.outcomes:
            raise DescriptionError(f"surface {name!r}: kind {e.kind.name} detects; detection needs a sensor binding "
                                   "(library.sensors), not a surface")  # fmt: skip
        _check_params(f"surface {name!r}", e.kind, e.params)
    for name, e in lib.sensors.items():
        if not isinstance(e, SensorEntry) or not isinstance(e.kind, legacy.SurfaceType):
            raise DescriptionError(f"sensor {name!r}: expected a SensorEntry(SurfaceType, params)")
        if DETECT not in e.kind.outcomes:
            raise DescriptionError(f"sensor {name!r}: kind {e.kind.name} cannot detect")
        _check_params(f"sensor {name!r}", e.kind, e.params)


# ------------------------------------------------------------------------------------------------------ the tree
def siblings(tree, a: str, b: str) -> bool:
    """Return whether two PVs of a tree are distinct siblings."""
    pa, pb = tree.by_name[a], tree.by_name[b]
    return a != b and pa.mother is not None and pa.mother == pb.mother


def relation(tree, pre: str, post: str) -> str | None:
    """Return "daughter" (post is a daughter of pre), "mother" (post is pre's mother), "sibling" or None."""
    if tree.by_name[post].mother == pre:
        return "daughter"
    if tree.by_name[pre].mother == post:
        return "mother"
    return "sibling" if siblings(tree, pre, post) else None


# --------------------------------------------------------------------------------------------------------- bindings
@dataclass(frozen=True)
class SensorBinding:
    """A sensor type on every placement of an LV, or on one PV; ``count`` is the number of world instances expected.

    ``entry``: "any" -- every crossing into the PV meets the sensor type; "border" -- only the crossings across a
    border bound to this sensor do, any other entry resolves as if the PV were no sensor (Geant4's photocathode is a
    border surface, not the volume).
    """

    sensor: str
    count: int
    lv: str | None = None
    pv: str | None = None
    entry: str = "any"


@dataclass(frozen=True, eq=False)
class Bindings:
    """A library attached to one tree, by exact names (see the module docstring)."""

    materials: Mapping[str, str]  # tree material -> library medium
    skins: Mapping[str, str] = field(default_factory=dict)  # LV -> surface
    borders: Mapping[tuple[str, str], str] = field(default_factory=dict)  # (pre PV, post PV) -> surface | sensor
    sensors: tuple[SensorBinding, ...] = ()
    touching: tuple[tuple[str, str], ...] = ()  # sibling pairs known to touch
    disjoint: tuple[tuple[str, str], ...] = ()  # sibling pairs known never to touch
    disjoint_placements: tuple[str, ...] = ()  # LVs whose sibling placements never touch


@dataclass(frozen=True, eq=False)
class Description:
    """The checked description: per-PV medium / skin / sensor names, the borders and the disjoint sibling pairs."""

    library: Library
    tree: object
    bindings: Bindings
    medium: Mapping[str, str]  # PV -> library medium
    skin: Mapping[str, str | None]  # PV -> surface of its LV skin (None)
    sensor: Mapping[str, str | None]  # PV -> sensor type (None)
    borders: Mapping[tuple[str, str], str]
    disjoint: frozenset  # of frozenset({a, b})
    disjoint_lvs: frozenset  # LVs whose sibling placements never touch
    sensor_entry: Mapping[str, str] = field(default_factory=dict)  # sensor PV -> "any" | "border"

    def enters_sensor(self, pre, post):
        """Return whether the crossing pre -> post meets post's sensor type."""
        if self.sensor[post] is None:
            return False
        return self.sensor_entry.get(post, "any") == "any" or self.borders.get((pre, post)) == self.sensor[post]

    def is_disjoint(self, a, b):
        """Return whether two sibling PVs are declared never to touch."""
        lv = self.tree.by_name[a].lv
        return frozenset((a, b)) in self.disjoint or (lv == self.tree.by_name[b].lv and lv in self.disjoint_lvs)

    def adjacent(self, pre, post):
        """Return "daughter" / "mother" / "sibling" (siblings touch unless declared disjoint) or None."""
        rel = relation(self.tree, pre, post)
        if rel == "sibling" and self.is_disjoint(pre, post):
            return None
        return rel


def _sibling_pairs(tree, pairs, what):
    out = set()
    for pair in pairs:
        a, b = pair
        for n in (a, b):
            if n not in tree.by_name:
                raise DescriptionError(f"{what} pair {pair}: unknown PV {n!r}")
        if not siblings(tree, a, b):
            raise DescriptionError(f"{what} pair {pair}: adjacency facts are declared between distinct siblings only")
        out.add(frozenset((a, b)))
    return out


def _check_materials(library, tree, b):
    occurring = {p.material for p in tree.pvs}
    unbound = sorted(occurring - set(b.materials))
    if unbound:
        raise DescriptionError(f"materials without a binding: {unbound}")
    dangling = sorted(set(b.materials) - occurring)
    if dangling:
        raise DescriptionError(f"bindings for materials that do not occur in the tree: {dangling}")
    for m, med in sorted(b.materials.items()):
        if med not in library.media:
            raise DescriptionError(f"material {m!r} bound to {med!r}, which is not a library medium")
    for lv, s in sorted(b.skins.items()):
        if lv not in tree.lv_material:
            raise DescriptionError(f"skin surface {s!r} on LV {lv!r}, which does not occur in the tree")
        if s not in library.surfaces:
            raise DescriptionError(f"skin on LV {lv!r}: {s!r} is not a library surface")


def _check_adjacency(tree, b):
    touching = _sibling_pairs(tree, b.touching, "touching")
    disjoint = _sibling_pairs(tree, b.disjoint, "disjoint")
    disjoint_lvs = frozenset(b.disjoint_placements)
    for lv in sorted(disjoint_lvs):
        if lv not in tree.lv_material:
            raise DescriptionError(f"disjoint placements of LV {lv!r}, which does not occur in the tree")

    def never(x, y):
        same = tree.by_name[x].lv == tree.by_name[y].lv and tree.by_name[x].lv in disjoint_lvs
        return same or frozenset((x, y)) in disjoint

    both = sorted(tuple(sorted(p)) for p in touching if never(*p))
    if both:
        raise DescriptionError(f"sibling pairs declared both touching and disjoint: {both}")
    return disjoint, disjoint_lvs, never


def _check_borders(library, tree, b, never):
    for (pre, post), s in sorted(b.borders.items()):
        for n in (pre, post):
            if n not in tree.by_name:
                raise DescriptionError(f"border surface {s!r} ({pre!r} -> {post!r}): unknown PV {n!r}")
        if (s in library.surfaces) == (s in library.sensors):
            what = "both a library surface and a library sensor" if s in library.surfaces else \
                "not a library surface or sensor"  # fmt: skip
            raise DescriptionError(f"border ({pre!r} -> {post!r}): {s!r} is {what}")
        rel = relation(tree, pre, post)
        if rel is None or (rel == "sibling" and never(pre, post)):
            raise DescriptionError(f"border surface {s!r} ({pre!r} -> {post!r}): the volumes never touch "
                                   "(not mother/daughter, not touching siblings)")  # fmt: skip


def _bind_sensors(library, tree, b):
    sensor, entry = {}, {}
    for sb in sorted(b.sensors, key=lambda s: (s.sensor, s.lv or "", s.pv or "")):
        if (sb.lv is None) == (sb.pv is None):
            raise DescriptionError(f"sensor binding {sb}: give exactly one of lv= or pv=")
        if sb.entry not in ("any", "border"):
            raise DescriptionError(f"sensor binding {sb}: entry must be 'any' or 'border'")
        if sb.sensor not in library.sensors:
            raise DescriptionError(f"sensor binding {sb}: {sb.sensor!r} is not a library sensor")
        if sb.lv is not None:
            if sb.lv not in tree.lv_material:
                raise DescriptionError(f"sensor binding {sb}: LV {sb.lv!r} does not occur in the tree")
            covered = tree.placements(sb.lv)
        else:
            if sb.pv not in tree.by_name:
                raise DescriptionError(f"sensor binding {sb}: PV {sb.pv!r} does not occur in the tree")
            covered = (sb.pv,)
        if len(covered) != sb.count:
            raise DescriptionError(f"sensor {sb.sensor!r} on {'LV ' + sb.lv if sb.lv else 'PV ' + sb.pv}: "
                                   f"expected {sb.count} instances, found {len(covered)}")  # fmt: skip
        for n in covered:
            if n in sensor:
                raise DescriptionError(f"PV {n!r} bound to two sensors ({sensor[n]!r}, {sb.sensor!r})")
            sensor[n], entry[n] = sb.sensor, sb.entry
    return sensor, entry


def _check_sensor_borders(library, b, sensor, entry, skin):
    for n in sorted(sensor):
        if skin[n] is not None:
            raise DescriptionError(f"sensor PV {n!r} also has a skin surface {skin[n]!r} on its LV: ambiguous "
                                   "(the sensor kind is the boundary of a sensor)")  # fmt: skip
    for (pre, post), s in sorted(b.borders.items()):
        if post in sensor and not (entry[post] == "border" and s == sensor[post]):
            raise DescriptionError(f"border surface {s!r} ({pre!r} -> {post!r}) into sensor PV {post!r}: ambiguous "
                                   "(the sensor kind is the boundary of a sensor; a border may name the sensor itself "
                                   "when its binding has entry='border')")  # fmt: skip
        if post in sensor:  # the entry border of an entry='border' sensor (above)
            continue
        if s in library.sensors and pre not in sensor:  # an exit sensor: detected leaving pre, credited to pre
            raise DescriptionError(f"border sensor {s!r} ({pre!r} -> {post!r}): {pre!r} is not a sensor PV, so a "
                                   "detection leaving it has no tag to go to")  # fmt: skip
    entered = {post for (_, post), s in b.borders.items() if post in sensor and s == sensor[post]}
    for n in sorted(n for n in sensor if entry[n] == "border" and n not in entered):
        raise DescriptionError(f"sensor PV {n!r} has entry='border' but no border bound to {sensor[n]!r} leads "
                               "into it: it would never detect")  # fmt: skip


def resolve(library: Library, tree, bindings: Bindings) -> Description:
    """Check a library, a tree and bindings against each other (fail closed) and return the Description."""
    check_library(library)
    b = bindings
    _check_materials(library, tree, b)
    disjoint, disjoint_lvs, never = _check_adjacency(tree, b)
    _check_borders(library, tree, b, never)
    sensor, entry = _bind_sensors(library, tree, b)
    skin = {n: b.skins.get(tree.by_name[n].lv) for n in tree.order}
    _check_sensor_borders(library, b, sensor, entry, skin)
    medium = {n: b.materials[tree.by_name[n].material] for n in tree.order}
    return Description(library, tree, b, medium, skin, {n: sensor.get(n) for n in tree.order}, dict(b.borders),
                       frozenset(disjoint), disjoint_lvs, entry)  # fmt: skip


# ------------------------------------------------------------------------------------------------- tables by hand
@dataclass(frozen=True, eq=False)
class Tables:
    """The static optical tables the step reads, built by hand (tests, toys).

    ``kind_of`` / ``row_of`` (C, C) int32: the surface type and its row for the ordered class pair (-1 never
    crossed); ``medium_of`` (C,) int32 into ``media``; ``kernel_laws`` {(type name, param): Law} for spectral params;
    ``deposit_pre`` (C, C) bool or None: pairs whose DETECT deposit goes to the tag of the region left.
    """

    kinds: tuple
    kind_of: np.ndarray
    row_of: np.ndarray
    medium_of: np.ndarray
    media: tuple
    kernel_laws: dict | None = None
    deposit_pre: np.ndarray | None = None

    @property
    def n_classes(self):
        """Return the number of regions."""
        return int(self.medium_of.shape[0])

    @property
    def max_channels(self):
        """Return the largest number of bulk channels of a medium."""
        return max([len(m.channels) for m in self.media] + [0])

    def __hash__(self):
        """Hash by identity (static)."""
        return id(self)


def _by_hand_blocks(full, n_classes):
    kinds, blocks, laws_ = [], {}, {}
    kind_of = np.full((n_classes, n_classes), -1, np.int32)
    row_of = np.full((n_classes, n_classes), -1, np.int32)
    for (a, b), (kind, p) in sorted(full.items(), key=lambda kv: kv[0]):
        if kind not in kinds:
            kinds.append(kind)
            blocks[kind.name] = {k: [] for k in kind.params}
        kind_of[a, b] = kinds.index(kind)
        row_of[a, b] = len(next(iter(blocks[kind.name].values()))) if kind.params else 0
        for k in kind.params:
            v = p[k]
            if isinstance(v, tuple):
                law, lp = v
                if laws_.setdefault((kind.name, k), law) is not law:
                    raise ValueError(f"{kind.name}.{k}: every row of a spectral param must use the same Law")
                blocks[kind.name][k].append({q: np.asarray(x, np.float32) for q, x in lp.items()})
            else:
                blocks[kind.name][k].append(np.asarray(v, np.float32))  # a scalar or a vector
    return kinds, blocks, laws_, kind_of, row_of


def by_hand(media, medium_of, pairs, default=None):
    """Return (Tables, kernel params) from {(class before, class after): (type, params dict)}.

    ``default`` (type, params) applies to every other ordered pair of distinct classes. A param given as (Law, law
    params) is spectral (Tables.kernel_laws).

    Notation: C the number of classes.
    """
    C = len(medium_of)
    full = dict(pairs)
    if default is not None:
        for a in range(C):
            for b in range(C):
                if a != b and (a, b) not in full:
                    full[(a, b)] = default
    kinds, blocks, laws_, kind_of, row_of = _by_hand_blocks(full, C)
    kp = {}
    for name, blk in blocks.items():
        kp[name] = {}
        for k, rows in blk.items():
            if rows and isinstance(rows[0], dict):
                kp[name][k] = {q: np.stack([r[q] for r in rows]) for q in rows[0]}
            else:
                kp[name][k] = np.stack(rows).astype(np.float32)
    return Tables(tuple(kinds), kind_of, row_of, np.asarray(medium_of, np.int32), tuple(media), laws_), kp


# ------------------------------------------------------------------------------------------------- the compiled optics
@dataclass(frozen=True, eq=False)
class Pair:
    """The surface of one crossed class pair and why.

    ``reasons``: "sensor", "border", "skin(post)", "skin(pre)", "default", "same-material", "NoRINDEX", ...;
    ``examples``: PV pairs; ``n_pv_pairs``: the number of adjacent PV pairs.
    """

    kind: legacy.SurfaceType
    origin: str | None  # "surface:<name>" / "sensor:<name>" for types with params, None otherwise
    row: int
    reasons: tuple[str, ...]
    examples: tuple[tuple[str, str], ...]
    n_pv_pairs: int


@dataclass(frozen=True, eq=False)
class Optics:
    """The compiled optics: the classes, the tables, the sensor tags, the packed params and the report."""

    n_classes: int
    class_names: tuple[str, ...]
    signatures: tuple[tuple, ...]
    class_of: Mapping[str, int]  # PV -> class
    class_pvs: tuple[tuple[str, ...], ...]
    kinds: tuple[legacy.SurfaceType, ...]
    kind_of: np.ndarray  # (C, C) int32 index into kinds, -1: never crossed
    row_of: np.ndarray  # (C, C) int32 row in params["kernel"][kind.name], -1: never crossed
    medium_of: np.ndarray  # (C,) int32 index into media
    media: tuple[bulk.Medium, ...]
    media_names: tuple[str, ...]  # library names, keys of params["media"]
    rows: Mapping[str, tuple[str | None, ...]]  # type name -> origin of each row
    pairs: Mapping[tuple[int, int], Pair]
    tag_of: Mapping[str, int]  # sensor PV -> tag 0..N-1
    tag_pvs: tuple[str, ...]
    tag_sensors: tuple[str, ...]
    lam_domain: tuple[float, float]
    params: dict
    report: str
    deposit_pre: np.ndarray | None = None  # (C, C) bool: a DETECT deposit goes to the tag before; None: none do
    kernel_laws: Mapping = field(default_factory=dict)  # (type name, param) -> Law of a spectral param

    def kernel(self, pre, post):
        """Return the Pair crossed from PV pre into PV post (None: never crossed, also inside one class)."""
        return self.pairs.get((self.class_of[pre], self.class_of[post]))


def _f32(tree):
    if isinstance(tree, Mapping):
        return {k: _f32(v) for k, v in tree.items()}
    return np.array(tree, np.float32)


def _entry(library, origin):
    what, name = origin.split(":", 1)
    return (library.surfaces if what == "surface" else library.sensors)[name]


def pack_params(optics: Optics, library: Library) -> dict:
    """Return the traced params pytree for a library whose VALUES may differ from the one compiled."""
    kernel = {}
    for kind in optics.kinds:
        rows = optics.rows[kind.name]
        kp = {}
        for name in kind.params:
            vals = []
            for origin in rows:
                e = _entry(library, origin)
                if e.kind is not kind:
                    raise OpticsError(f"{origin}: kind changed from {kind.name} to {e.kind.name}; recompile")
                vals.append(np.asarray(e.params[name], np.float32))
            if len({v.shape for v in vals}) != 1:
                raise OpticsError(f"{kind.name}.{name}: rows have different shapes {[v.shape for v in vals]}")
            kp[name] = np.stack(vals)
        kernel[kind.name] = kp
    media = {}
    for name, med in zip(optics.media_names, optics.media, strict=False):
        e = library.media[name]
        if e.medium is not med:
            raise OpticsError(f"medium {name!r} changed its Medium object; recompile")
        media[name] = _f32(e.params)
    return {"kernel": kernel, "media": media}


def named_params(optics: Optics, library: Library) -> dict:
    """Return the params of the library entries the optics uses, by NAME: {"media", "surfaces", "sensors"}.

    The calibration space, stable across recompiles (which renumber classes and rows); ``pack`` turns it into the
    engine's rows. A spectral param (Law, params) is its params: the Law is static (``Optics.kernel_laws``).
    """

    def values(p):
        return {k: (v[1] if isinstance(v, tuple) else v) for k, v in p.items()}

    out = {"media": {n: library.media[n].params for n in optics.media_names}, "surfaces": {}, "sensors": {}}
    for kind in optics.kinds:
        for origin in optics.rows[kind.name]:
            if origin is not None:
                what, name = origin.split(":", 1)
                out["surfaces" if what == "surface" else "sensors"][name] = values(_entry(library, origin).params)
    return out


def pad_table(e, v, n):
    """Return an (energy, value) table padded to n knots past its end (flat, clamped), as float32.

    numpy in float64 for concrete values (bit-identical to the compiled tables), jnp for traced ones.
    """
    import jax
    import jax.numpy as jnp

    traced = isinstance(e, jax.core.Tracer) or isinstance(v, jax.core.Tracer)
    xp = jnp if traced else np
    e, v = xp.asarray(e, xp.float32 if traced else np.float64), xp.asarray(v, xp.float32 if traced else np.float64)
    k = n - e.shape[0]
    if k > 0:
        step = xp.maximum(PAD_STEP * xp.abs(e[-1]), PAD_STEP)
        e = xp.concatenate([e, e[-1] + step * xp.arange(1, k + 1)])
        v = xp.concatenate([v, xp.full(k, v[-1])])
    return jnp.asarray(e, jnp.float32), jnp.asarray(v, jnp.float32)


def _packed_table(name, ents):
    import jax.numpy as jnp

    tabs = [e[name] if hasattr(e[name], "items") else {"energy": np.array([1.0, 2.0]),
                                                       "value": jnp.stack([e[name], e[name]])}
            for e in ents]  # fmt: skip
    nk = max(int(np.shape(t["energy"])[0]) for t in tabs)
    pt = [pad_table(t["energy"], t["value"], nk) for t in tabs]
    return {"energy": jnp.stack([a for a, _ in pt]), "value": jnp.stack([b for _, b in pt])}


def pack(optics: Optics, named: dict) -> dict:
    """Return pack_params from named params (``named_params``' structure; values may be traced, gradients flow)."""
    import jax
    import jax.numpy as jnp

    def f32(t):
        return jax.tree_util.tree_map(lambda v: jnp.asarray(v, jnp.float32), t)

    kernel = {}
    for kind in optics.kinds:
        ents = [named["surfaces" if o.split(":", 1)[0] == "surface" else "sensors"][o.split(":", 1)[1]]
                if o is not None else None for o in optics.rows[kind.name]]  # fmt: skip
        kp = {}
        for name in kind.params:
            if (kind.name, name) in optics.kernel_laws:  # spectral: (energy, value) tables per row
                kp[name] = _packed_table(name, ents)
            else:
                kp[name] = jnp.stack([jnp.asarray(e[name], jnp.float32) for e in ents])
        kernel[kind.name] = kp
    return {"kernel": kernel, "media": {n: f32(named["media"][n]) for n in optics.media_names}}


# ------------------------------------------------------------------------------------------------- the compiler
def _class_name(sig):
    med, skin, sensor, roles = sig
    s = med
    if skin is not None:
        s += f"|skin:{skin}"
    if sensor is not None:
        s += f"|sensor:{sensor}"
    for partner, d, surf in roles:
        s += f"|border:{surf}{'->' if d == 'out' else '<-'}[{partner}]"
    return s


def _kind_order(k):
    return (ORDERED_TYPES.index(k), "") if k in ORDERED_TYPES else (len(ORDERED_TYPES), k.name)


def _names(pvs, n=NAMES_SHOWN):
    return ", ".join(pvs[:n]) + (f", ... (+{len(pvs) - n})" if len(pvs) > n else "")


def _classify(order, sig_of):
    index, sigs, class_of = {}, [], {}
    for n in order:
        c = index.setdefault(sig_of[n], len(sigs))
        if c == len(sigs):
            sigs.append(sig_of[n])
        class_of[n] = c
    return sigs, class_of


def _classes(tree, d):
    """Return (signatures, class_of, sig_of) by colour refinement of the border roles to a fixed point.

    A border surface binds a PV pair, so a bordered PV's signature holds its border roles; the partner enters by its
    CLASS, refined to a fixed point, not by its PV name (2000 identical PMTs, each glass bordering its own interior,
    are two classes, not 4000). Once stable the numbering repeats, so the roles name the final classes.
    """
    roles = {n: [] for n in tree.order}
    for (pre, post), s in d.borders.items():
        roles[pre].append((post, "out", s))
        roles[post].append((pre, "in", s))
    base = {n: (d.medium[n], d.skin[n], d.sensor[n]) for n in tree.order}
    sigs, class_of = _classify(tree.order, {n: base[n] + ((),) for n in tree.order})
    while True:
        sig_of = {n: base[n] + (tuple(sorted({(class_of[p], r, s) for p, r, s in roles[n]})),) for n in tree.order}
        prev, (sigs, class_of) = len(sigs), _classify(tree.order, sig_of)
        if len(sigs) == prev:
            return sigs, class_of, sig_of


class _Regions:
    """The class-pair surfaces of one compile.

    ``found``: class pair -> [type, origin, reasons, examples, count, side]; ``rels``: class pair -> relations of
    post to pre; ``legs``: every adjacency, from an opaque class too (a leg of a coincident face).
    """

    def __init__(self, library, tree, d, opaque, classes):
        self.library, self.tree, self.d, self.opaque = library, tree, d, opaque
        self.sigs, self.class_of, self.sig_of, self.class_names = classes
        self.found, self.rels, self.legs = {}, {}, {}

    def _surface_kernel(self, pre, post, rel):
        lib, d = self.library, self.d
        surf, reason = d.borders.get((pre, post)), "border"
        if surf is None:
            order = (("post", post), ("pre", pre)) if rel == "daughter" else (("pre", pre), ("post", post))
            for side, n in order:
                if d.skin[n] is not None:
                    surf, reason = d.skin[n], f"skin({side})"
                    break
        if surf is not None:
            e = lib.surfaces[surf]
            same = self.tree.by_name[pre].material == self.tree.by_name[post].material
            if e.ignored_between_same_material and same:
                return legacy.PASS, None, f"{reason} {surf}: same-material"
            return e.kind, f"surface:{surf}", reason
        if d.medium[pre] == d.medium[post]:
            return legacy.PASS, None, "default"
        if not self.opaque[d.medium[post]]:
            return legacy.DIELECTRIC, None, "default"
        return legacy.ABSORBER, None, "default: NoRINDEX"

    def kernel_of(self, pre, post, rel):
        """Return (type, origin, reason, side credited) of the crossing pre -> post (Geant4's precedence)."""
        lib, d = self.library, self.d
        if d.enters_sensor(pre, post):
            kind, origin, reason = lib.sensors[d.sensor[post]].kind, f"sensor:{d.sensor[post]}", "sensor"
        elif d.borders.get((pre, post)) in lib.sensors:  # resolve: pre is a sensor PV
            s = d.borders[(pre, post)]
            return lib.sensors[s].kind, f"sensor:{s}", f"border sensor {s} (credited to pre)", "pre"
        else:
            kind, origin, reason = self._surface_kernel(pre, post, rel)
        # a stacked type (coated) never ends in the post medium: its STRAIGHT is the film crossing onto the backing
        # at the same point, and coated refuses backings that transmit
        if self.opaque[d.medium[post]] and not kind.stack and any(FAR_SIDE[o] for o in kind.outcomes):
            if kind is not legacy.DIELECTRIC:
                raise OpticsError(f"{pre!r} -> {post!r}: kind {kind.name} ({reason}) would put the photon into the "
                                  f"opaque medium {d.medium[post]!r}")  # fmt: skip
            kind, origin, reason = legacy.ABSORBER, None, f"{reason}: NoRINDEX"
        if not kind.params:
            origin = None
        return kind, origin, reason, "post"

    def visit(self, pre, post, rel, count):
        """Resolve one adjacent PV pair (``count`` PV pairs like it) into its class pair."""
        c = self.class_of
        if c[pre] != c[post]:
            self.legs.setdefault((c[pre], c[post]), [set(), (pre, post)])[0].add(rel)
        if self.opaque[self.d.medium[pre]]:
            return
        kind, origin, reason, side = self.kernel_of(pre, post, rel)
        key = (c[pre], c[post])
        if key[0] == key[1]:
            if kind is not legacy.PASS:
                raise OpticsError(
                    f"class {self.class_names[key[0]]!r} touches itself across a {kind.name} boundary ({reason}; "
                    f"e.g. {pre!r} -> {post!r}): the crossing would be invisible. Declare the placements disjoint "
                    "(bindings.disjoint) if they do not touch, or give them distinct signatures"
                )
            return
        self.rels.setdefault(key, set()).add(rel)
        f = self.found.get(key)
        if f is None:
            self.found[key] = [kind, origin, [reason], [(pre, post)], count, side]
            return
        if (f[0], f[1], f[5]) != (kind, origin, side):
            a, b = f[3][0]
            raise OpticsError(
                f"class pair {self.class_names[key[0]]!r} -> {self.class_names[key[1]]!r} resolves to two kernels: "
                f"{a!r} -> {b!r}: {f[0].name} {f[1] or ''} ({f[2][0]}), {pre!r} -> {post!r}: {kind.name} "
                f"{origin or ''} ({reason}). The signature missed a distinction"
            )
        if reason not in f[2]:
            f[2].append(reason)
        if len(f[3]) < MAX_EXAMPLES:
            f[3].append((pre, post))
        f[4] += count

    def visit_tree(self):
        """Visit every mother <-> daughter pair, then the siblings."""
        tree, d = self.tree, self.d
        for n in tree.order:
            m = tree.by_name[n].mother
            if m is not None:
                self.visit(m, n, "daughter", 1)
                self.visit(n, m, "mother", 1)
        # A border binds a PV pair, not a group pair: siblings group by signature; each (group pair, surface) is
        # visited by one of its bordered pairs, and the rest of the group pair (no border) by a representative.
        sib_borders = {}
        for (a, b), s in d.borders.items():
            ma = tree.by_name[a].mother
            if ma is not None and ma == tree.by_name[b].mother and not d.is_disjoint(a, b):
                sib_borders.setdefault(ma, []).append((a, b, s))
        env = getattr(tree, "groups", {}) or {}  # dissolved envelopes: siblings touch only within one instance
        for m in tree.order:
            if len(tree.children[m]) >= 2:
                self._visit_siblings(m, sib_borders.get(m, ()), env)

    def _sibling_groups(self, m, env):
        gkey, parts = {}, {}
        for n in self.tree.children[m]:
            p = self.tree.by_name[n]
            gkey[n] = (self.sig_of[n], p.material, p.lv)
            parts.setdefault(env.get(n, ""), {}).setdefault(gkey[n], []).append(n)
        nd = {}
        for pair in self.d.disjoint:
            a, b = sorted(pair)
            if self.tree.by_name[a].mother == m and env.get(a, "") == env.get(b, ""):
                for x, y in ((a, b), (b, a)):
                    nd[gkey[x], gkey[y]] = nd.get((gkey[x], gkey[y]), 0) + 1
        return gkey, parts, nd

    def _representative(self, ga, gb, simple):
        """Return the first PV pair of two groups that may be crossed without a border, or None."""
        if simple:  # no borders, no disjoint pairs: the first distinct pair (the general search finds the same)
            if ga is not gb:
                return ga[0], gb[0]
            return (ga[0], ga[1]) if len(ga) > 1 else None
        d = self.d
        return next(((a, b) for a in ga for b in gb
                     if a != b and (a, b) not in d.borders and not d.is_disjoint(a, b)), None)  # fmt: skip

    def _group_pairs(self, parts, gkey):
        by_name, d = self.tree.by_name, self.d
        simple = not d.borders and not d.disjoint
        total, rep = {}, {}  # per group pair: the touching pairs over every envelope instance, one representative
        for groups in parts.values():
            glist = list(groups.values())
            heads = [(by_name[g[0]].lv, gkey[g[0]], len(g)) for g in glist]
            for ga, (lva, ka, na) in zip(glist, heads, strict=False):
                for gb, (lvb, kb, nb) in zip(glist, heads, strict=False):
                    if lva == lvb and lva in d.disjoint_lvs:
                        continue
                    kk = (ka, kb)
                    total[kk] = total.get(kk, 0) + na * (nb - (ga is gb))
                    if kk not in rep:
                        r = self._representative(ga, gb, simple)
                        if r is not None:
                            rep[kk] = r
        return total, rep

    def _visit_siblings(self, m, sib_borders, env):
        gkey, parts, nd = self._sibling_groups(m, env)
        nb, brep = {}, {}
        for a, b, s in sib_borders:
            kk = (gkey[a], gkey[b])
            nb[kk] = nb.get(kk, 0) + 1
            brep.setdefault((kk, s), [a, b, 0])[2] += 1
        for a, b, c in brep.values():
            self.visit(a, b, "sibling", c)
        total, rep = self._group_pairs(parts, gkey)
        for kk, tot in total.items():
            live = tot - nd.get(kk, 0) - nb.get(kk, 0)
            if live > 0:
                self.visit(rep[kk][0], rep[kk][1], "sibling", live)


# A daughter's face may lie ON its mother's boundary (SBND's GAr ullage under the cryostat roof): the geometry then
# reports a crossing from the daughter's class straight into what lies beyond the mother, a class pair no adjacency
# produces. Such pairs are derived at class level: p -> q via b, with p inside b and q beyond it (outward), or p
# beyond b and q inside it (inward). Geant4 locates the photon directly in q: the crossing is ONE interface p -> q,
# resolved like any other (kernel_of on the PV pair); b has no part in it -- for an EXACTLY coincident face. Except
# where a leg is a PASS (b one medium with p, or with q, no surface between): b is then optically p (q), and the
# crossing is the other leg's interface, its border and sensor included (Geant4 steps into any b thicker than its
# 1e-12 m tolerance, and a float32 geometry merges a b thinner than its band). Nested coincident faces compose (to a
# fixed point). A class pair also adjacent directly keeps its direct surface (the report lists any that would
# differ). A candidate with no valid surface is left unresolved and listed: crossed, the photon is reported lost.
OUTWARD = ("mother", "sibling", "outward")
INWARD = ("daughter", "sibling", "inward")


class _Coincident:
    """The coincident-face pairs of one compile (see the comment above)."""

    def __init__(self, regions: _Regions):
        self.r = regions
        t, c = regions.tree, regions.class_of
        self.inside = {(c[n], c[t.by_name[n].mother]) for n in t.order if t.by_name[n].mother is not None}
        self.via, self.shadowed, self.unresolved, self.prenote = {}, [], {}, []

    def _relation(self, p, b, q, rp, rq):
        if (p, b) in self.inside and rq & set(OUTWARD) and rp & {"mother", "outward"}:
            return "outward"
        if (q, b) in self.inside and rp & set(INWARD) and rq & {"daughter", "inward"}:
            return "inward"
        return None

    def _through_pass(self, p, b, q, fb, fq):
        names = self.r.class_names
        k_ = fq if fb[0] is legacy.PASS else fb
        kind, origin, side = k_[0], k_[1], k_[5]
        leg = (b, q) if fb[0] is legacy.PASS else (p, b)
        reason = f"pass via {names[b]}: as {names[leg[0]]} -> {names[leg[1]]} ({k_[2][0]})"
        if side == "pre":
            self.prenote.append(f"  {p} -> {q} via {b}: {kind.name} {origin or ''} is credited to the region "
                                f"before, {names[p]} where b is merged (a float32-thin b), not {b}")  # fmt: skip
        return kind, origin, reason, side

    def _candidate(self, p, b, q, pvs, rel):
        """Resolve one candidate p -> q via b; return whether a new class pair was added."""
        r = self.r
        if (p, q) in self.unresolved:
            return False
        pre_pv, post_pv = pvs
        fb, fq = r.found.get((p, b)), r.found.get((b, q))
        if fb is not None and fq is not None and (fb[0] is legacy.PASS or fq[0] is legacy.PASS):
            kind, origin, reason, side = self._through_pass(p, b, q, fb, fq)
        else:
            try:
                kind, origin, reason, side = r.kernel_of(pre_pv, post_pv, rel)
            except OpticsError as err:
                if (p, q) not in r.found:
                    self.unresolved[(p, q)] = f"via {r.class_names[b]}: {err}"
                return False
            reason = f"coincident via {r.class_names[b]}: {reason}"
        if (p, q) in r.found:
            f = r.found[(p, q)]
            if (p, q) not in self.via and (f[0], f[1], f[5]) != (kind, origin, side):
                self.shadowed.append(f"  {p} -> {q} via {b}: direct {f[0].name} kept (the coincident face would be "
                                     f"{kind.name} {origin or ''})")  # fmt: skip
            r.rels[(p, q)].add(rel)
            r.legs[(p, q)][0].add(rel)
            return False
        r.found[(p, q)] = [kind, origin, [reason], [(pre_pv, post_pv)], 0, side]
        r.rels[(p, q)] = {rel}
        r.legs.setdefault((p, q), [set(), (pre_pv, post_pv)])[0].add(rel)
        self.via[(p, q)] = b
        return True

    def run(self):
        """Add the coincident-face pairs to the regions' ``found``, to a fixed point."""
        r = self.r
        changed = True
        while changed:
            changed = False
            for (b, q), (rq, eq) in sorted(r.legs.items(), key=lambda kv: kv[0]):
                for (p, b2), (rp, ep) in sorted(r.legs.items(), key=lambda kv: kv[0]):
                    if b2 != b or p == q or r.opaque[r.sigs[p][0]]:
                        continue
                    rel = self._relation(p, b, q, rp, rq)
                    if rel is not None and self._candidate(p, b, q, (ep[0], eq[1]), rel):
                        changed = True
        return self


def _tables(found, n_classes):
    kinds = tuple(sorted({f[0] for f in found.values()}, key=_kind_order))
    if len({k.name for k in kinds}) != len(kinds):
        raise OpticsError(f"two different kinds share a name: {[k.name for k in kinds]}")
    rows = {k.name: tuple(sorted({f[1] for f in found.values() if f[0] is k}, key=lambda o: o or "")) for k in kinds}
    kind_of = np.full((n_classes, n_classes), -1, np.int32)
    row_of = np.full((n_classes, n_classes), -1, np.int32)
    deposit_pre = np.zeros((n_classes, n_classes), bool)
    pairs = {}
    for a, b in sorted(found):
        kind, origin, reasons, examples, count, side = found[(a, b)]
        r = rows[kind.name].index(origin)
        kind_of[a, b], row_of[a, b], deposit_pre[a, b] = kinds.index(kind), r, side == "pre"
        pairs[(a, b)] = Pair(kind, origin, r, tuple(reasons), tuple(examples), count)
    return kinds, rows, kind_of, row_of, deposit_pre, pairs


def _report(head, class_names, class_pvs, pairs, coincident, tags):
    lines = [head, "classes:"]
    for c in range(len(class_names)):
        lines.append(f"  [{c}] {class_names[c]}  ({len(class_pvs[c])} PVs: {_names(class_pvs[c])})")
    lines.append("crossed class pairs (pre -> post: kernel [row] origin; why; PV pairs):")
    for (a, b), p in pairs.items():
        ex = ", ".join(f"{x}->{y}" for x, y in p.examples)
        lines.append(f"  {a} -> {b}: {p.kind.name} [{p.row}] {p.origin or '-'}; {', '.join(p.reasons)}; "
                     f"{p.n_pv_pairs} (e.g. {ex})")  # fmt: skip
    if coincident.via:
        lines.append(f"coincident faces (via a zero-thickness class): {len(coincident.via)} pairs; "
                     + ", ".join(f"{a} -> {b} via {v}" for (a, b), v in sorted(coincident.via.items())))  # fmt: skip
    lines += coincident.shadowed
    lines += coincident.prenote
    for (a, b), why in sorted(coincident.unresolved.items()):
        lines.append(f"  unresolved coincident face {a} -> {b} ({why}): never crossed; if the geometry crosses it the "
                     "photon is reported lost")  # fmt: skip
    sensor_pvs, tag_of, sensor_of = tags
    for s in sorted({sensor_of[n] for n in sensor_pvs}):
        t = [tag_of[n] for n in sensor_pvs if sensor_of[n] == s]
        lines.append(f"sensor {s}: {len(t)} instances, tags {t[0]}..{t[-1]}")
    return "\n".join(lines)


def _media(library, d, lam_domain):
    lo, hi = (float(x) for x in lam_domain)
    if not (np.isfinite(lo) and np.isfinite(hi) and 0.0 < lo < hi):
        raise OpticsError(f"wavelength domain must be finite with 0 < lam_min < lam_max, got {lam_domain}")
    media_names = tuple(sorted(set(d.medium.values())))
    for name in media_names:
        e = library.media[name]
        try:
            bulk.validate(e.medium, e.params, lo, hi)
        except KeyError as err:
            raise OpticsError(f"medium {name!r}: missing params {err}") from None
    return lo, hi, media_names, {name: library.media[name].medium.opaque for name in media_names}


def compile_classes(library: Library, tree, bindings: Bindings, lam_domain) -> Optics:
    """Return the Optics of a library on a tree (the class compiler, scalar params only; see ``compile``).

    Notation: C the number of classes.
    """
    d = resolve(library, tree, bindings)
    lo, hi, media_names, opaque = _media(library, d, lam_domain)
    sigs, class_of, sig_of = _classes(tree, d)
    C = len(sigs)
    members = [[] for _ in range(C)]
    for n in tree.order:
        members[class_of[n]].append(n)
    class_pvs = tuple(tuple(m) for m in members)
    class_names = tuple(_class_name(s) for s in sigs)
    regions = _Regions(library, tree, d, opaque, (sigs, class_of, sig_of, class_names))
    regions.visit_tree()
    coincident = _Coincident(regions).run()
    kinds, rows, kind_of, row_of, deposit_pre, pairs = _tables(regions.found, C)
    media_of = np.array([media_names.index(s[0]) for s in sigs], np.int32)
    by_name = tree.by_name
    sensor_pvs = sorted((n for n in tree.order if d.sensor[n] is not None),
                        key=lambda n: (d.sensor[n], by_name[n].copy is None, by_name[n].copy or 0, n))  # fmt: skip
    tag_of = {n: i for i, n in enumerate(sensor_pvs)}
    head = (f"optics: {C} classes, {len(pairs)} crossed class pairs, kinds {[k.name for k in kinds]}, "
            f"{len(sensor_pvs)} sensors, lambda [{lo:g}, {hi:g}] nm")  # fmt: skip
    report = _report(head, class_names, class_pvs, pairs, coincident, (sensor_pvs, tag_of, d.sensor))
    optics = Optics(C, class_names, tuple(sigs), class_of, class_pvs, kinds, kind_of, row_of, media_of,
                    tuple(library.media[n].medium for n in media_names), media_names, rows, pairs, tag_of,
                    tuple(sensor_pvs), tuple(d.sensor[n] for n in sensor_pvs), (lo, hi), {}, report,
                    deposit_pre if deposit_pre.any() else None)  # fmt: skip
    object.__setattr__(optics, "params", pack_params(optics, library))
    return optics


# ------------------------------------------------------------------------------------------ compile with spectra
def pad(e, v, n):
    """Return an (energy, value) table padded to n knots past its end (flat; float64)."""
    e, v = np.asarray(e, np.float64), np.asarray(v, np.float64)
    k = n - len(e)
    if k <= 0:
        return e, v
    step = max(PAD_STEP * abs(e[-1]), PAD_STEP)
    return np.concatenate([e, e[-1] + step * np.arange(1, k + 1)]), np.concatenate([v, np.full(k, v[-1])])


def _scalar_library(library):
    """Return the library with every spectral param replaced by 0.0, and the spectral params by entry."""
    spectral = {}

    def plain(what, entries, make):
        out = {}
        for n_, e in entries.items():
            sp = {k: v for k, v in e.params.items() if isinstance(v, tuple)}
            if sp:
                spectral[f"{what}:{n_}"] = sp
                e = make(e, {k: (0.0 if k in sp else v) for k, v in e.params.items()})
            out[n_] = e
        return out

    lib = Library(library.media,
                  plain("surface", library.surfaces,
                        lambda e, p: SurfaceEntry(e.kind, p, e.ignored_between_same_material)),
                  plain("sensor", library.sensors, lambda e, p: SensorEntry(e.kind, p)))  # fmt: skip
    return lib, spectral


def _spectral_rows(library, rows, spectral, kind, pname, lam):
    vals = [spectral.get(o, {}).get(pname) for o in rows]
    if all(v is None for v in vals):
        return None, None
    used = {id(v[0]): v[0] for v in vals if v is not None}
    if len(used) != 1:
        raise OpticsError(f"{kind.name}.{pname}: rows use different Laws")
    law = next(iter(used.values()))
    if law.params != ("energy", "value"):
        raise OpticsError(f"{kind.name}.{pname}: only (energy, value) tables are packed")
    lps = []
    for o, v in zip(rows, vals, strict=False):
        if v is None:
            what, nm = o.split(":", 1)
            s = float((library.surfaces if what == "surface" else library.sensors)[nm].params[pname])
            lp = {"energy": np.array([1.0, 2.0]), "value": np.array([s, s])}
        else:
            lp = v[1]
        laws.validate(law, lp, lam[0], lam[1], name=f"{o}.{pname}")
        lps.append(lp)
    nk = max(len(lp["energy"]) for lp in lps)
    padded = [pad(lp["energy"], lp["value"], nk) for lp in lps]
    return law, {"energy": np.stack([a for a, _ in padded]).astype(np.float32),
                 "value": np.stack([b for _, b in padded]).astype(np.float32)}  # fmt: skip


def compile(library: Library, tree, bindings: Bindings, lam_domain) -> Optics:
    """Return the Optics of a library whose surface / sensor params may be spectral ((Law, params) pairs).

    ``compile_classes`` on the library with its spectral params set aside, then the spectral columns packed: the rows
    of one (type, param) share one Law, scalar rows become flat tables, knots are padded to a common length with
    clamped tails, and each row's law is validated over the wavelength domain.
    """
    lib, spectral = _scalar_library(library)
    optics = compile_classes(lib, tree, bindings, lam_domain)
    kernel = {k: dict(v) for k, v in optics.params["kernel"].items()}
    found = {}
    for kind in optics.kinds:
        for pname in kind.params:
            law, packed = _spectral_rows(library, optics.rows[kind.name], spectral, kind, pname, optics.lam_domain)
            if law is not None:
                kernel[kind.name][pname] = packed
                found[(kind.name, pname)] = law
    object.__setattr__(optics, "kernel_laws", found)
    object.__setattr__(optics, "params", dict(optics.params, kernel=kernel))
    return optics


# ---------------------------------------------------------------------------------------- the GDML's borders
@dataclass(frozen=True)
class BorderSensor:
    """How border_library reads the borders into a sensor volume.

    ``model`` names the sensor model (None: the default, stock Geant4's reading); ``eff_in`` replaces the detection
    efficiency on entry (a sensitive detector that counts every absorption: WCSim's, 1); ``prefix`` names the
    sensors ("<prefix>:<surface>").
    """

    model: str | None = None
    eff_in: float | None = None
    prefix: str = "sensor"


def _segments(tree):
    seg = {}
    for pv in tree.order:
        parts = pv.split("/")
        if len(parts) >= 2:  # a physvol placed several times in one mother: '<name>#<index>'
            seg.setdefault(re.sub(r"#\d+$", "", parts[-2]), []).append(pv)
    return seg


def _aliased(tree):
    al = getattr(tree, "aliases", {}) or {}

    def alias(n):  # a dissolved envelope's name -> the placement holding its region
        seen = set()
        while n in al and n not in seen:
            seen.add(n)
            n = al[n]
        return n

    return alias


def _border_pairs(tree, seg, a, b, inside):
    """Return the adjacent tree PV pairs of one border a -> b (GDML physvol names)."""
    from lumen.build.tree import OUTSIDE_PV

    grp = getattr(tree, "groups", {}) or {}  # dissolved envelopes: siblings touch within one envelope instance
    bs = set(seg.get(b, ()))
    b_in, b_sib = {}, {}  # b's instances by mother (x's daughters), and by (mother, envelope instance) (siblings)
    for y in bs:
        b_in.setdefault(tree.by_name[y].mother, []).append(y)
        b_sib.setdefault((tree.by_name[y].mother, grp.get(y, "")), []).append(y)
    pairs = []
    for x in seg.get(a, ()):  # x's neighbours only (daughters, mother, siblings): linear, not |a| |b|
        m = tree.by_name[x].mother
        sib = b_sib.get((m, grp.get(x, "")), ()) if m is not None else ()
        near = set(b_in.get(x, ())) | ({m} & bs) | set(sib)
        pairs += [(x, y) for y in sorted(near) if y != x and relation(tree, x, y) is not None]
        # the kept detector against its GDML mother, which the tree holds as the outside: a border from the
        # detector volume into the physvol that contains it
        if m == OUTSIDE_PV and not bs and inside(a, b):
            pairs.append((x, OUTSIDE_PV))
    if not seg.get(a) and seg.get(b) and inside(b, a):  # ... and from that mother into the detector
        pairs += [(OUTSIDE_PV, y) for y in seg[b] if tree.by_name[y].mother == OUTSIDE_PV]
    return pairs


def border_bindings(df, tree) -> dict:
    """Return the GDML border surfaces on a tree: {(pre tree PV, post tree PV): surface name}.

    A GDML physvol name matches every world instance whose placement segment (the path segment before the member
    volume) is that name (or that name + '#k': one physvol placed several times in one mother); a border applies to
    each adjacent (mother/daughter or sibling) instance pair. Fail closed when a border matches nothing; a border
    whose two volumes both lie outside the kept detector is dropped with a warning.
    """
    seg = _segments(tree)
    lv_of = {p.name: p.volume for p in df.placements}  # physvol -> its LV

    def inside(dname, mname):  # is physvol dname a daughter of physvol mname's LV
        return mname in lv_of and any(df.placements[i].name == dname for i in df.volumes[lv_of[mname]].daughters)

    alias = _aliased(tree)
    out, dropped = {}, []
    for border in df.borders:
        a, b = alias(border.pv1), alias(border.pv2)
        pairs = _border_pairs(tree, seg, a, b, inside)
        if not pairs and not seg.get(a) and not seg.get(b):
            dropped.append(border.name)  # both volumes lie outside the kept detector
            continue
        if not pairs:
            raise OpticsError(f"bordersurface {border.name!r} ({a!r} -> {b!r}): no adjacent world instances")
        for pr in pairs:
            out[pr] = border.surface
    if dropped:
        warnings.warn(f"{len(dropped)} border surfaces lie wholly outside the kept detector and are dropped (e.g. "
                      f"{dropped[:3]})", stacklevel=2)  # fmt: skip
    return out


def border_library(df, tree, sensor_pvs, lam_domain=None, how: BorderSensor | None = None):
    """Return the GDML's border surfaces with its sensors: (sensors, surfaces, borders, sensor bindings).

    Exactly in the directions the GDML defines them (G4LogicalBorderSurface: ordered pv1 -> pv2). A border INTO a
    sensor volume is that sensor (the photon absorbed-and-detected there by the surface's sensor model,
    ``lumen.rules.optics.sensor``); a border OUT OF one whose surface detects is a sensor too, its hit credited to the
    volume left (Geant4's Detection leaving a sensitive volume, e.g. WCSim's ProcessHits_boundary); any other border
    is its surface. A sensor volume detects only across its borders (SensorBinding entry "border"). ``how``: the
    sensor model, the efficiency on entry and the sensors' name prefix (BorderSensor).

    Returns sensors {name: SensorEntry}, surfaces {name: SurfaceEntry}, borders {(pre, post): name} and the sensor
    bindings.
    """
    from lumen.rules import optics as rules

    how = BorderSensor() if how is None else how
    n_lv = Counter(tree.by_name[p].lv for p in sensor_pvs)
    sensors, surfaces, borders, sbind, bound = {}, {}, {}, [], set()
    for (pre, post), s in sorted(border_bindings(df, tree).items()):
        rec = df.surfaces[s]
        if post in sensor_pvs:
            k = f"{how.prefix}:{s}"
            if k not in sensors:
                sensors[k] = rules.sensor(rec, lam_domain, how.model, eff=how.eff_in)
            lv = tree.by_name[post].lv
            if (k, lv) not in bound:
                bound.add((k, lv))
                sbind.append(SensorBinding(k, n_lv[lv], lv=lv, entry="border"))
            borders[(pre, post)] = k
        elif pre in sensor_pvs and rules.nonzero(rec.properties, "EFFICIENCY"):
            k = f"{how.prefix}:{s}|out"
            if k not in sensors:
                sensors[k] = rules.sensor(rec, lam_domain, how.model)
            borders[(pre, post)] = k
        else:
            if s not in surfaces:
                surfaces[s] = rules.surface(rec)
            borders[(pre, post)] = s
    return sensors, surfaces, borders, tuple(sbind)


def channel_of(df, mother_lv: str, child_lv: str) -> int:
    """Return the copy number of the one placement of ``child_lv`` inside ``mother_lv`` (LArSoft OpDet channels)."""
    hits = [df.placements[i].copy for i in df.volumes[mother_lv].daughters if df.placements[i].volume == child_lv]
    if len(hits) != 1:
        raise OpticsError(f"{mother_lv!r}: {len(hits)} placements of {child_lv!r} (want 1)")
    return 0 if hits[0] is None else int(hits[0])  # Geant4's default copy number is 0
