"""The surface lookup: Geant4's boundary rule over every crossing the navigator reports, as the tables the step reads.

    optics = compile(library, tree, bindings, lam_domain)
    optics.kernel(pre, post)          # the Pair of the crossing pre -> post: surface type, row, why

Three inputs, all plain Python (``lumen.build.library`` makes them from a detector file):

    Library    named media (bulk.Medium + params), surfaces (a surface type + params) and sensor types (a detecting
               type + params)
    Tree       the physical volumes as world instances (``lumen.build.tree``): each PV names its logical volume,
               material and mother PV
    Bindings   attach a library to one tree by EXACT names: materials -> media, skin surfaces (LV -> surface), border
               surfaces (ordered PV pair -> surface), sensors (LV or PV -> sensor type, expected count) and sibling
               adjacency facts

``resolve`` checks all three against each other and fails closed. Each adjacent ordered PV pair (pre, post) --
mother/daughter always, siblings unless declared disjoint -- and each coincident face p -> q via a zero-thickness b
resolves to a surface type by ``lumen.rules.boundary`` (Geant4's precedence). Pairs leaving an opaque medium are never
crossed.

Classes are an internal compression, never part of the rule: the PVs with the same symbolic signature (medium, skin
surface, sensor type, border roles = {(partner class, 'out' | 'in', surface)}, refined to a fixed point) share every
crossing, so the step's tables are (class, class) arrays and the navigator carries a class per volume. They never
depend on parameter values. Guards (fail closed): two PV pairs of one class pair resolving to different surfaces (the
signature missed a distinction); a class adjacent to itself across a non-PASS surface; laws invalid over the
wavelength domain. A DETECT deposit goes to the tag of the PV entered, except across an exit sensor (``deposit_pre``).

``compile`` also packs spectral surface and sensor params ((Law, params) pairs of the shared clamped tables) as
``Optics.kernel_laws`` (one Law per (type, param), knots padded to a common length with clamped tails).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from lumen.media import bulk, laws
from lumen.rules import boundary
from lumen.surfaces import _legacy as legacy
from lumen.surfaces.outcomes import DETECT

# the surface-type order of the packed tables; every other type sorts after these, by name
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


def _entry(library, origin):
    what, name = origin.split(":", 1)
    return (library.surfaces if what == "surface" else library.sensors)[name]


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
    """Return the step's params (``kernel`` rows by type, ``media``) from named params (``named_params``' structure).

    Values may be traced: gradients flow to the named params.
    """
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

    # the PV-level view lumen.rules.boundary reads
    borders = property(lambda self: self.d.borders)
    skin = property(lambda self: self.d.skin)
    sensor = property(lambda self: self.d.sensor)
    medium = property(lambda self: self.d.medium)

    def enters_sensor(self, pre, post):
        """Return whether the crossing pre -> post meets post's sensor type."""
        return self.d.enters_sensor(pre, post)

    def material(self, pv):
        """Return a PV's tree material."""
        return self.tree.by_name[pv].material

    def kernel_of(self, pre, post, rel):
        """Return (type, origin, reason, side credited) of the crossing pre -> post (``boundary.crossing``)."""
        try:
            return boundary.crossing(self, pre, post, rel)
        except boundary.BoundaryError as err:
            raise OpticsError(str(err)) from None

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


class _Coincident:
    """The coincident-face pairs of one compile (see the comment above)."""

    def __init__(self, regions: _Regions):
        self.r = regions
        t, c = regions.tree, regions.class_of
        self.inside = {(c[n], c[t.by_name[n].mother]) for n in t.order if t.by_name[n].mother is not None}
        self.via, self.shadowed, self.unresolved, self.prenote = {}, [], {}, []

    def _relation(self, p, b, q, rp, rq):
        return boundary.coincident((p, b) in self.inside, (q, b) in self.inside, rp, rq)

    def _through_pass(self, p, b, q, fb, fq):
        names = self.r.class_names
        leg = (b, q) if fb[0] is legacy.PASS else (p, b)
        kind, origin, reason, side = boundary.through(fb, fq, names[b], (names[leg[0]], names[leg[1]]))
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


def _resolve_all(library: Library, tree, bindings: Bindings, lam_domain) -> Optics:
    """Return the Optics of a library on a tree without its packed params (``compile`` packs them).

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
    return optics


# ------------------------------------------------------------------------------------------ compile with spectra
def _spectral_laws(library, optics):
    """Return {(type name, param): Law} of the spectral columns, each row's law validated over the wavelength domain.

    The rows of one (type, param) share one Law, an (energy, value) table; a scalar row in a spectral column is a flat
    table.
    """
    lo, hi = optics.lam_domain
    found = {}
    for kind in optics.kinds:
        for pname in kind.params:
            vals = [(o, _entry(library, o).params[pname]) for o in optics.rows[kind.name]]
            used = {id(v[0]): v[0] for _, v in vals if isinstance(v, tuple)}
            if not used:
                continue
            if len(used) != 1:
                raise OpticsError(f"{kind.name}.{pname}: rows use different Laws")
            law = next(iter(used.values()))
            if law.params != ("energy", "value"):
                raise OpticsError(f"{kind.name}.{pname}: only (energy, value) tables are packed")
            for o, v in vals:
                lp = v[1] if isinstance(v, tuple) else {"energy": np.array([1.0, 2.0]), "value": np.array([v, v])}
                laws.validate(law, lp, lo, hi, name=f"{o}.{pname}")
            found[(kind.name, pname)] = law
    return found


def compile(library: Library, tree, bindings: Bindings, lam_domain) -> Optics:
    """Return the Optics of a library on a tree, its surface and sensor params possibly spectral ((Law, params)).

    The crossings resolved (``_resolve_all``), then the params packed by ``pack``: scalar rows stacked, spectral
    columns as (energy, value) tables padded to a common length with clamped tails.
    """
    optics = _resolve_all(library, tree, bindings, lam_domain)
    optics = dataclasses.replace(optics, kernel_laws=_spectral_laws(library, optics))
    named = named_params(optics, library)
    return dataclasses.replace(optics, params=_numpy(pack(optics, named), {"kernel": None, "media": named["media"]}))


def _numpy(tree, order):
    """Return a params tree as numpy arrays, its dicts in ``order``'s key order (the library's; a pytree map sorts)."""
    if not isinstance(tree, Mapping):
        return np.asarray(tree)
    keys = list(order) + [k for k in tree if k not in order] if isinstance(order, Mapping) else list(tree)
    return {k: _numpy(tree[k], order.get(k) if isinstance(order, Mapping) else None) for k in keys if k in tree}
