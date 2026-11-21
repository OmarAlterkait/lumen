"""The surface lookup: Geant4's boundary rule over every crossing the navigator reports, as the tables the step reads.

    optics = compile(library, tree, bindings)
    optics.pair(pre, post)          # the Pair of the crossing pre -> post: surface type, row, why

Three inputs, all plain Python (``lumen.build.library`` makes them from a detector file):

    Library    named media (bulk.Medium + params), surfaces (a surface type + params) and sensor types (a detecting
               type + params)
    Tree       the physical volumes as world instances (``lumen.build.tree``): each PV names its logical volume,
               material and mother PV
    Bindings   attach a library to one tree by EXACT names: materials -> media, skin surfaces (LV -> surface), border
               surfaces (ordered physvol pair -> surface: every touching instance pair of the two, as Geant4's
               G4LogicalBorderSurface; never expanded into instance pairs) and sensors (LV or PV -> sensor type,
               expected count)

``resolve`` checks all three against each other and fails closed. Each ordered PV pair (pre, post) that may touch --
mother/daughter and siblings -- and each coincident face p -> q via a zero-thickness b resolves to a surface type by
``lumen.rules.boundary`` (Geant4's precedence). Pairs leaving an opaque medium are never crossed. A pair that never
touches is never reported by the navigator, so taking siblings to touch costs nothing but the guard below.

Classes are an internal compression, never part of the rule: the PVs with the same symbolic signature (medium, skin
surface, sensor type, border roles = {(partner class, 'out' | 'in', surface)}, refined to a fixed point) share every
crossing, so the step's tables are (class, class) arrays and the navigator carries a class per volume. They never
depend on parameter values. A crossing between two PVs of one class (touching siblings of one LV with a skin or an
any-entry sensor, as Geant4 crosses them) is the class table's diagonal: the navigator reports a crossing into another
volume of the same class wherever that diagonal has a surface (``lumen.build.tracked.vertices``); a diagonal that
resolves to a pass is no crossing and is left out. Guards (fail closed): two PV pairs of one class pair resolving to
different surfaces (the signature missed a distinction), the diagonal included; laws invalid at their own knots
(``lumen.media.laws.validate``). A detect deposit goes to the tag of the PV entered (its placement: the navigator's
volume), except across an exit sensor (``deposit_pre``).

Every crossed (surface type, origin) is a row of one parameter table (``SurfaceTable``: a column per parameter of
``lumen.surfaces.types.COLUMNS``, the rows' constants and the per-row structure the kernel reads), which ``compile``
packs: a spectral column ((Law, params) pairs of the shared clamped tables) as (energy, value) tables padded to a
common knot count with clamped tails, each row's Law named per row.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from lumen.media import bulk, laws
from lumen.rules import boundary
from lumen.rules.choices import PAD_STEP, PER_NORMAL_RTOL  # stated there
from lumen.rules.entries import DescriptionError, MediumEntry, SensorEntry, SurfaceEntry
from lumen.surfaces import facets, types
from lumen.surfaces.outcomes import N_OUTCOMES

# the surface-type order of the packed tables; every other type sorts after these, by name
ORDERED_TYPES = (
    types.PASS,
    types.ABSORBER,
    types.DIELECTRIC,
    types.DIELECTRIC_RT,
    types.REFLECTOR,
    types.PHOTOCATHODE,
    types.COUNTING,
    types.UNIFIED_METAL,
)
MAX_EXAMPLES = 3  # example PV pairs kept per crossed class pair
NAMES_SHOWN = 6  # PV names shown per class in the report
PER_NORMAL_POINTS = 2001  # wavelengths between the knots where detection_margins samples A_n (see its docstring)


class OpticsError(DescriptionError):
    """A description the class compiler cannot turn into tables."""


# ---------------------------------------------------------------------------------------------------------- library
@dataclass(frozen=True, eq=False)
class Library:
    """Named media, surfaces and sensor types."""

    media: Mapping[str, MediumEntry] = field(default_factory=dict)
    surfaces: Mapping[str, SurfaceEntry] = field(default_factory=dict)
    sensors: Mapping[str, SensorEntry] = field(default_factory=dict)
    domain: tuple | None = None  # the light's photon energies (eV intervals; None: every one): lumen.rules.optics


def _check_params(what, stype, params):
    have, want = set(params), set(stype.params)
    if have != want:
        raise DescriptionError(f"{what}: surface type {stype.name} takes parameters {sorted(want)}, got {sorted(have)} "
                               f"(missing {sorted(want - have)}, unknown {sorted(have - want)})")  # fmt: skip


def check_library(lib: Library):
    """Check the structure of every entry (the laws are validated by the compiler, for the media used)."""
    for name, e in lib.media.items():
        if not isinstance(e, MediumEntry) or not isinstance(e.medium, bulk.Medium):
            raise DescriptionError(f"medium {name!r}: expected a MediumEntry(bulk.Medium, params)")
    for name, e in lib.surfaces.items():
        if not isinstance(e, SurfaceEntry) or not isinstance(e.surface_type, types.SurfaceType):
            raise DescriptionError(f"surface {name!r}: expected a SurfaceEntry(SurfaceType, params)")
        if e.surface_type.detects:
            raise DescriptionError(f"surface {name!r}: surface type {e.surface_type.name} detects; detection needs a "
                                   "sensor binding (library.sensors), not a surface")  # fmt: skip
        _check_params(f"surface {name!r}", e.surface_type, e.params)
    for name, e in lib.sensors.items():
        if not isinstance(e, SensorEntry) or not isinstance(e.surface_type, types.SurfaceType):
            raise DescriptionError(f"sensor {name!r}: expected a SensorEntry(SurfaceType, params)")
        if not e.surface_type.detects:
            raise DescriptionError(f"sensor {name!r}: surface type {e.surface_type.name} cannot detect")
        _check_params(f"sensor {name!r}", e.surface_type, e.params)


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


def _touching(tree, xs, ys):
    """Return the PVs of xs with a neighbour (``relation``) among ys: one pass over each, no pair enumerated."""
    yset = set(ys)
    by_mother: dict = {}
    for y in ys:
        m = tree.by_name[y].mother
        by_mother[m] = by_mother.get(m, 0) + 1
    out = []
    for x in xs:
        m = tree.by_name[x].mother
        sib = 0 if m is None else by_mother.get(m, 0) - (x in yset)
        if sib > 0 or by_mother.get(x, 0) > 0 or m in yset:
            out.append(x)
    return tuple(out)


def border_instances(tree, a: str, b: str):
    """Return the world instances of a border's physvols a -> b that touch: (a's with a neighbour among b's, b's with
    one among a's). Linear in the instances: a border names a physvol pair, never its instance pairs."""  # noqa: D205, D209
    xs, ys = tree.instances_of.get(a, ()), tree.instances_of.get(b, ())
    return _touching(tree, xs, ys), _touching(tree, ys, xs)


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
    # (pre physvol, post physvol) -> surface | sensor: every touching instance pair of the two (``Tree.physvol``)
    borders: Mapping[tuple[str, str], str] = field(default_factory=dict)
    sensors: tuple[SensorBinding, ...] = ()


@dataclass(frozen=True, eq=False)
class Description:
    """The checked description: per-PV medium / skin / sensor names and the borders."""

    library: Library
    tree: object
    bindings: Bindings
    medium: Mapping[str, str]  # PV -> library medium
    skin: Mapping[str, str | None]  # PV -> surface of its LV skin (None)
    sensor: Mapping[str, str | None]  # PV -> sensor type (None)
    borders: Mapping[tuple[str, str], str]
    sensor_entry: Mapping[str, str] = field(default_factory=dict)  # sensor PV -> "any" | "border"

    def border(self, pre, post):
        """Return the border surface (or sensor) of the crossing between PVs pre -> post, None: none (``Bindings``)."""
        s = self.borders.get((self.tree.physvol(pre), self.tree.physvol(post)))
        return None if s is None or relation(self.tree, pre, post) is None else s

    def enters_sensor(self, pre, post):
        """Return whether the crossing pre -> post meets post's sensor type."""
        if self.sensor[post] is None:
            return False
        return self.sensor_entry.get(post, "any") == "any" or self.border(pre, post) == self.sensor[post]


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


def _check_borders(library, tree, b):
    for (pre, post), s in sorted(b.borders.items()):
        for n in (pre, post):
            if n not in tree.instances_of:
                raise DescriptionError(f"border surface {s!r} ({pre!r} -> {post!r}): unknown physvol {n!r}")
        if (s in library.surfaces) == (s in library.sensors):
            what = "both a library surface and a library sensor" if s in library.surfaces else \
                "not a library surface or sensor"  # fmt: skip
            raise DescriptionError(f"border ({pre!r} -> {post!r}): {s!r} is {what}")
        if not border_instances(tree, pre, post)[0]:
            raise DescriptionError(f"border surface {s!r} ({pre!r} -> {post!r}): the volumes never touch "
                                   "(not mother/daughter, not siblings)")  # fmt: skip


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


def _check_sensor_borders(library, tree, b, sensor, entry, skin):
    for n in sorted(sensor):
        if skin[n] is not None:
            raise DescriptionError(f"sensor PV {n!r} also has a skin surface {skin[n]!r} on its LV: ambiguous "
                                   "(the sensor's surface type is the boundary of a sensor)")  # fmt: skip
    entered = set()
    for (a, z), s in sorted(b.borders.items()):
        xs, ys = border_instances(tree, a, z)
        for post in ys:
            if post in sensor and not (entry[post] == "border" and s == sensor[post]):
                raise DescriptionError(f"border surface {s!r} ({a!r} -> {z!r}) into sensor PV {post!r}: ambiguous "
                                       "(the sensor's surface type is the boundary of a sensor; a border may name "
                                       "the sensor itself when its binding has entry='border')")  # fmt: skip
            if post in sensor:  # the entry border of an entry='border' sensor (above)
                entered.add(post)
        if s in library.sensors and not all(y in sensor for y in ys):  # an exit sensor: credited to the PV left
            for pre in xs:
                if pre not in sensor:
                    raise DescriptionError(f"border sensor {s!r} ({a!r} -> {z!r}): {pre!r} is not a sensor PV, so "
                                           "a detection leaving it has no sensor index to go to")  # fmt: skip
    for n in sorted(n for n in sensor if entry[n] == "border" and n not in entered):
        raise DescriptionError(f"sensor PV {n!r} has entry='border' but no border bound to {sensor[n]!r} leads "
                               "into it: it would never detect")  # fmt: skip


def resolve(library: Library, tree, bindings: Bindings) -> Description:
    """Check a library, a tree and bindings against each other (fail closed) and return the Description."""
    check_library(library)
    b = bindings
    _check_materials(library, tree, b)
    _check_borders(library, tree, b)
    sensor, entry = _bind_sensors(library, tree, b)
    skin = {n: b.skins.get(tree.by_name[n].lv) for n in tree.order}
    _check_sensor_borders(library, tree, b, sensor, entry, skin)
    medium = {n: b.materials[tree.by_name[n].material] for n in tree.order}
    return Description(library, tree, b, medium, skin, {n: sensor.get(n) for n in tree.order}, dict(b.borders), entry)


# ------------------------------------------------------------------------------------------------- the compiled optics
@dataclass(frozen=True, eq=False)
class Pair:
    """The surface of one crossed class pair and why.

    ``reasons``: "sensor", "border", "skin(post)", "skin(pre)", "default", "same-material", "NoRINDEX", ...;
    ``examples``: PV pairs; ``n_pv_pairs``: the number of adjacent PV pairs.
    """

    surface_type: types.SurfaceType
    origin: str | None  # "surface:<name>" / "sensor:<name>" for types with params, None otherwise
    row: int  # the surface table's row
    reasons: tuple[str, ...]
    examples: tuple[tuple[str, str], ...]
    n_pv_pairs: int


class Row(NamedTuple):
    """A row of the surface table: its surface type and origin ("surface:<name>" / "sensor:<name>", None without
    params)."""  # noqa: D205, D209

    surface_type: types.SurfaceType
    origin: str | None


@dataclass(frozen=True, eq=False)
class SurfaceTable:
    """The surface rows' structure: one parameter table read by one kernel (``lumen.surfaces.types.kernel``).

    ``params``: the columns some row reads as a parameter, or whose constants differ across rows
    (``Optics.params["surfaces"]``: (n,) values, or (energy, value) tables of (n, K) for a spectral column, with its
    constant rows' values (n,) as "const"); the others are one constant of the code; ``reads``: per row {column:
    parameter name}; ``consts``: per column the rows' constants (n,), each row's value where it does
    not read the column as a parameter (a spectral column's constant rows read it, not a law); ``laws``: per spectral
    column the Laws of the rows that read it as a parameter and ``law_of`` per row the index of its own (a scalar
    parameter's: a flat table under a tabulated Law). Per row (n,): ``model`` its Interaction's index in ``models``,
    the flags of ``types.Flags``, ``support`` (n, 10) its outcomes and ``lobe_of`` its GLOSSY facet path in
    ``lobes``. Structure only: which values are parameters (keys
    or the build's) and which are constants of the build, never a value at run time.
    """

    rows: tuple[Row, ...]
    params: tuple[str, ...]
    reads: tuple[Mapping[str, str], ...]
    consts: Mapping[str, np.ndarray]
    laws: Mapping[str, tuple]
    law_of: Mapping[str, np.ndarray]
    models: tuple[types.Interaction, ...]
    flags: types.Flags
    support: np.ndarray
    lobes: tuple[str, ...]
    lobe_of: np.ndarray

    @property
    def any_per_normal(self) -> bool:
        """Whether a row is read for a sensor model's QE."""
        return bool(np.any(self.flags.per_normal))


@dataclass(frozen=True, eq=False)
class Optics:
    """The compiled optics: the classes, the surface rows, the sensor tags, the packed params and the report."""

    n_classes: int
    class_names: tuple[str, ...]
    signatures: tuple[tuple, ...]
    class_of: Mapping[str, int]  # PV -> class
    class_pvs: tuple[tuple[str, ...], ...]
    rows: tuple[Row, ...]  # the surface table's rows, by surface type (``ORDERED_TYPES``, then by name), then origin
    row_of: np.ndarray  # (C, C) int32 row of the surface table, -1: never crossed (on the diagonal: a pass)
    medium_of: np.ndarray  # (C,) int32 index into media
    media: tuple[bulk.Medium, ...]
    media_names: tuple[str, ...]  # library names, keys of params["media"]
    pairs: Mapping[tuple[int, int], Pair]
    sensor_index: Mapping[str, int]  # sensor PV -> tag 0..N-1
    sensor_pvs: tuple[str, ...]
    sensor_entries: tuple[str, ...]
    params: dict
    report: str
    deposit_pre: np.ndarray | None = None  # (C, C) bool: a detect deposit goes to the tag before; None: none do
    table: SurfaceTable | None = None  # the rows' structure (``compile``)
    # (C,) int32: per class of a sensor counting a photon born inside (SensorEntry.born), a class b whose crossing
    # (b, c) is its entry, whose detection the photon's emission there deposits; -1 none; None: no such sensor
    born_pair: np.ndarray | None = None

    def pair(self, pre, post):
        """Return the Pair crossed from PV pre into PV post (None: never crossed, or a pass inside one class)."""
        return self.pairs.get((self.class_of[pre], self.class_of[post]))


def _entry(library, origin):
    what, name = origin.split(":", 1)
    return (library.surfaces if what == "surface" else library.sensors)[name]


def named_params(optics: Optics, library: Library) -> dict:
    """Return the params of the library entries the optics uses, by NAME: {"media", "surfaces", "sensors"}.

    The calibration space, stable across recompiles (which renumber classes and rows); ``pack`` turns it into the
    engine's rows. A spectral param (Law, params) is its params: the Law is static (``SurfaceTable.laws``).
    """

    def values(p):
        return {k: (v[1] if isinstance(v, tuple) else v) for k, v in p.items()}

    out = {"media": {n: library.media[n].params for n in optics.media_names}, "surfaces": {}, "sensors": {}}
    for _, origin in optics.rows:
        if origin is not None:
            what, name = origin.split(":", 1)
            out["surfaces" if what == "surface" else "sensors"][name] = values(_entry(library, origin).params)
    return out


def pad_table(e, v, n):
    """Return an (energy, value) table padded to n knots past its end (flat, clamped), as float32.

    The energies are the build's (concrete, float64 with numpy: bit-identical to the compiled tables); a concrete value
    column likewise, a traced one with jnp.
    """
    import jax
    import jax.numpy as jnp

    e64 = np.asarray(e, np.float64)
    k = n - e64.shape[0]
    if k > 0:
        step = max(PAD_STEP * abs(e64[-1]), PAD_STEP)
        e64 = np.concatenate([e64, e64[-1] + step * np.arange(1, k + 1)])
    e_out = e64.astype(np.float32)
    if not isinstance(v, jax.core.Tracer):
        v64 = np.asarray(v, np.float64)
        v_out = np.concatenate([v64, np.full(max(k, 0), v64[-1])]).astype(np.float32)
    else:
        v32 = jnp.asarray(v, jnp.float32)
        v_out = jnp.concatenate([v32, jnp.repeat(v32[-1:], max(k, 0))])
    return e_out, v_out


def _host(x) -> bool:
    """Return whether a value is on the host (numpy or Python): packed with numpy, moved to the device once."""
    import jax

    return not any(isinstance(a, (jax.Array, jax.core.Tracer)) for a in jax.tree_util.tree_leaves(x))


def _stack32(xs):
    """Return the values stacked as float32: with numpy when every one is on the host (one array, no device op per
    value), else with jnp (a traced value keeps its derivative)."""  # noqa: D205, D209
    import jax.numpy as jnp

    if all(_host(x) for x in xs):
        return np.stack([np.asarray(x, np.float32) for x in xs])
    return jnp.stack([jnp.asarray(x, jnp.float32) for x in xs])


def _packed_table(vals):
    import jax.numpy as jnp

    def two(v):
        return np.stack([v, v]) if _host(v) else jnp.stack([v, v])

    tabs = [v if hasattr(v, "items") else {"energy": np.array([1.0, 2.0]), "value": two(v)} for v in vals]
    nk = max(int(np.shape(t["energy"])[0]) for t in tabs)
    pt = [pad_table(t["energy"], t["value"], nk) for t in tabs]
    return {"energy": _stack32([a for a, _ in pt]), "value": _stack32([b for _, b in pt])}


def pack(optics: Optics, named: dict) -> dict:
    """Return the step's params (``surfaces``: the surface table's parameter columns, ``media``) from named params
    (``named_params``).

    A column holds each row's parameter where the row reads it and the row's constant elsewhere (``SurfaceTable``); a
    spectral column (energy, value) tables, a constant row's flat. Values may be traced: gradients flow to the named
    params. A host value is packed as a numpy float32 array (the same bits as on the device), so that the caller moves
    the packed tree to the device in one transfer.
    """  # noqa: D205
    import jax
    import jax.numpy as jnp

    def f32(t):
        return jax.tree_util.tree_map(lambda v: np.asarray(v, np.float32) if _host(v) else jnp.asarray(v, jnp.float32),
                                      t)  # fmt: skip

    tab = optics.table
    ents = [named["surfaces" if o.split(":", 1)[0] == "surface" else "sensors"][o.split(":", 1)[1]]
            if o is not None else None for _, o in optics.rows]  # fmt: skip
    cols = {}
    for c in tab.params:
        vals = [e[tab.reads[i][c]] if c in tab.reads[i] else tab.consts[c][i] for i, e in enumerate(ents)]
        cols[c] = _packed_table(vals) if c in tab.laws else _stack32(vals)
        if c in tab.laws and not all(c in rd for rd in tab.reads):  # its constant rows' values, read as they are
            cols[c]["const"] = np.asarray(tab.consts[c], np.float32)
    return {"surfaces": cols, "media": {n: f32(named["media"][n]) for n in optics.media_names}}


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


def _partner_classes(tree, xs, ys, class_of):
    """Return per PV of xs the classes of its neighbours among ys (``relation``), by class counts per mother: no
    instance pair enumerated."""  # noqa: D205, D209
    yset, under = set(ys), {}
    for y in ys:
        m = tree.by_name[y].mother
        under.setdefault(m, {})
        under[m][class_of[y]] = under[m].get(class_of[y], 0) + 1
    out = {}
    for x in xs:
        m = tree.by_name[x].mother
        got = set(under.get(x, ()))  # its daughters among ys
        if m is not None:  # its siblings among ys, itself not
            got |= {c for c, k in under.get(m, {}).items() if k - (x in yset and class_of[x] == c) > 0}
        if m in yset:
            got.add(class_of[m])
        out[x] = got
    return out


def _classes(tree, d):
    """Return (signatures, class_of, sig_of) by colour refinement of the border roles to a fixed point.

    A border surface binds a physvol pair, so a bordered PV's signature holds its border roles; the partner enters by
    its CLASS, refined to a fixed point, not by its PV name (2000 identical PMTs, each glass bordering its own
    interior, are two classes, not 4000). Once stable the numbering repeats, so the roles name the final classes.
    """
    bord = [(border_instances(tree, a, b), (a, b), s) for (a, b), s in d.borders.items()]
    base = {n: (d.medium[n], d.skin[n], d.sensor[n]) for n in tree.order}
    sigs, class_of = _classify(tree.order, {n: base[n] + ((),) for n in tree.order})
    while True:
        roles = {n: set() for n in tree.order}
        for (xs, ys), _, s in bord:
            for x, cs in _partner_classes(tree, xs, ys, class_of).items():
                roles[x] |= {(c, "out", s) for c in cs}
            for y, cs in _partner_classes(tree, ys, xs, class_of).items():
                roles[y] |= {(c, "in", s) for c in cs}
        sig_of = {n: base[n] + (tuple(sorted(roles[n])),) for n in tree.order}
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
    skin = property(lambda self: self.d.skin)
    sensor = property(lambda self: self.d.sensor)
    medium = property(lambda self: self.d.medium)

    def enters_sensor(self, pre, post):
        """Return whether the crossing pre -> post meets post's sensor type."""
        return self.d.enters_sensor(pre, post)

    def border(self, pre, post):
        """Return the border surface (or sensor) of the crossing pre -> post (``Description.border``)."""
        return self.d.border(pre, post)

    def material(self, pv):
        """Return a PV's tree material."""
        return self.tree.by_name[pv].material

    def surface_of(self, pre, post, rel):
        """Return (type, origin, reason, side credited) of the crossing pre -> post (``boundary.crossing``)."""
        try:
            return boundary.crossing(self, pre, post, rel)
        except boundary.BoundaryError as err:
            raise OpticsError(str(err)) from None

    def visit(self, pre, post, rel, count):
        """Resolve one adjacent PV pair (``count`` PV pairs like it) into its class pair, the diagonal included."""
        c = self.class_of
        if c[pre] != c[post]:
            self.legs.setdefault((c[pre], c[post]), [set(), (pre, post)])[0].add(rel)
        if self.opaque[self.d.medium[pre]]:
            return
        stype, origin, reason, side = self.surface_of(pre, post, rel)
        key = (c[pre], c[post])
        self.rels.setdefault(key, set()).add(rel)
        f = self.found.get(key)
        if f is None:
            self.found[key] = [stype, origin, [reason], [(pre, post)], count, side]
            return
        if (f[0], f[1], f[5]) != (stype, origin, side):
            a, b = f[3][0]
            raise OpticsError(
                f"volume groups {self.class_names[key[0]]!r} -> {self.class_names[key[1]]!r} resolve to two surface "
                "types: "
                f"{a!r} -> {b!r}: {f[0].name} {f[1] or ''} ({f[2][0]}), {pre!r} -> {post!r}: {stype.name} "
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
        # siblings by class and material (a surface ignored between one material reads it): each (group pair, border
        # surface) is visited by one of its bordered pairs, the rest of the group pair (no border) by one
        # representative; counted from the instances per (group, physvol), never enumerated (a border names a
        # physvol pair: |a| |b| instance pairs under one mother)
        bout: dict = {}
        for (a, b), s in d.borders.items():
            bout.setdefault(a, []).append((b, s))
        for m in tree.order:
            if len(tree.children[m]) >= 2:
                self._visit_siblings(m, bout)

    def _visit_siblings(self, m, bout):
        c, phys, pv = self.class_of, self.tree.physvol, self.tree.by_name
        by: dict = {}  # (class, material) -> physvol -> its instances among m's daughters
        for n in self.tree.children[m]:
            by.setdefault((c[n], pv[n].material), {}).setdefault(phys(n), []).append(n)
        nb, brep = self._bordered(by, bout)
        for a, b, k in brep.values():
            self.visit(a, b, "sibling", k)
        cnt = {k: sum(map(len, g.values())) for k, g in by.items()}
        for ka in by:
            for kb in by:
                live = cnt[ka] * (cnt[kb] - (ka == kb)) - nb.get((ka, kb), 0)
                if live > 0:
                    a, b = self._representative(by[ka], by[kb], bout)
                    self.visit(a, b, "sibling", live)

    @staticmethod
    def _bordered(by, bout):
        """Return (per group pair its bordered sibling pairs, per (group pair, surface) [a pair, their count]) of one
        mother's daughters ``by`` (group -> physvol -> PVs)."""  # noqa: D205, D209
        where: dict = {}  # physvol -> the classes holding it here
        for k, g in by.items():
            for p in g:
                where.setdefault(p, []).append(k)
        nb, brep = {}, {}
        for pa, kas in where.items():
            for pb, s in bout.get(pa, ()):
                for ka in kas:
                    for kb in where.get(pb, ()):
                        xs, ys = by[ka][pa], by[kb][pb]
                        same = xs is ys
                        k = len(xs) * (len(ys) - same)
                        if k:
                            nb[(ka, kb)] = nb.get((ka, kb), 0) + k
                            brep.setdefault((ka, kb, s), [xs[0], ys[1] if same else ys[0], 0])[2] += k
        return nb, brep

    @staticmethod
    def _representative(ga, gb, bout):
        """Return a pair of distinct PVs of two groups' physvols (``ga``, ``gb``: physvol -> PVs) with no border
        between them: per physvol of ga, the physvols of gb it borders are skipped (at most its borders)."""  # noqa: D205, D209
        for pa, xs in ga.items():
            skip = {pb for pb, _ in bout.get(pa, ())}
            for pb, ys in gb.items():
                if pb in skip:
                    continue
                if xs is not ys:
                    return xs[0], ys[0]
                if len(xs) > 1:
                    return xs[0], xs[1]
        raise OpticsError("internal: a group pair counted as touching without a border has no such pair")


# A daughter's face may lie ON its mother's boundary (a gas layer under a vessel's roof): the geometry then
# reports a crossing from the daughter's class straight into what lies beyond the mother, a class pair no adjacency
# produces. Such pairs are derived at class level: p -> q via b, with p inside b and q beyond it (outward), or p
# beyond b and q inside it (inward). Geant4 locates the photon directly in q: the crossing is ONE interface p -> q,
# resolved like any other (surface_of on the PV pair); b has no part in it -- for an EXACTLY coincident face. Except
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
        leg = (b, q) if fb[0] == types.PASS else (p, b)
        stype, origin, reason, side = boundary.through(fb, fq, names[b], (names[leg[0]], names[leg[1]]))
        if side == "pre":
            self.prenote.append(f"  {p} -> {q} via {b}: {stype.name} {origin or ''} is credited to the region "
                                f"before, {names[p]} where b is merged (a float32-thin b), not {b}")  # fmt: skip
        return stype, origin, reason, side

    def _candidate(self, p, b, q, pvs, rel):
        """Resolve one candidate p -> q via b; return whether a new class pair was added."""
        r = self.r
        if (p, q) in self.unresolved:
            return False
        pre_pv, post_pv = pvs
        fb, fq = r.found.get((p, b)), r.found.get((b, q))
        if fb is not None and fq is not None and (fb[0] == types.PASS or fq[0] == types.PASS):
            stype, origin, reason, side = self._through_pass(p, b, q, fb, fq)
        else:
            try:
                stype, origin, reason, side = r.surface_of(pre_pv, post_pv, rel)
            except OpticsError as err:
                if (p, q) not in r.found:
                    self.unresolved[(p, q)] = f"via {r.class_names[b]}: {err}"
                return False
            reason = f"coincident via {r.class_names[b]}: {reason}"
        if (p, q) in r.found:
            f = r.found[(p, q)]
            if (p, q) not in self.via and (f[0], f[1], f[5]) != (stype, origin, side):
                self.shadowed.append(f"  {p} -> {q} via {b}: direct {f[0].name} kept (the coincident face would be "
                                     f"{stype.name} {origin or ''})")  # fmt: skip
            r.rels[(p, q)].add(rel)
            r.legs[(p, q)][0].add(rel)
            return False
        r.found[(p, q)] = [stype, origin, [reason], [(pre_pv, post_pv)], 0, side]
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
    """Return (the surface table's rows, row_of, deposit_pre, pairs): one row per crossed (surface type, origin),
    ordered by type (``ORDERED_TYPES``, then by name), then by origin."""  # noqa: D205, D209
    surface_types = sorted({f[0] for f in found.values()}, key=_kind_order)
    if len({k.name for k in surface_types}) != len(surface_types):
        raise OpticsError(f"two different surface types share a name: {[k.name for k in surface_types]}")
    rows = tuple(Row(k, o) for k in surface_types
                 for o in sorted({f[1] for f in found.values() if f[0] == k}, key=lambda o: o or ""))  # fmt: skip
    row_of = np.full((n_classes, n_classes), -1, np.int32)
    deposit_pre = np.zeros((n_classes, n_classes), bool)
    pairs = {}
    for a, b in sorted(found):
        stype, origin, reasons, examples, count, side = found[(a, b)]
        r = rows.index(Row(stype, origin))
        row_of[a, b], deposit_pre[a, b] = r, side == "pre"
        pairs[(a, b)] = Pair(stype, origin, r, tuple(reasons), tuple(examples), count)
    return rows, row_of, deposit_pre, pairs


def _report(head, class_names, class_pvs, pairs, coincident, tags):
    lines = [head, "classes:"]
    for c in range(len(class_names)):
        lines.append(f"  [{c}] {class_names[c]}  ({len(class_pvs[c])} PVs: {_names(class_pvs[c])})")
    lines.append("crossed class pairs (pre -> post: surface type [row] origin; why; PV pairs):")
    for (a, b), p in pairs.items():
        ex = ", ".join(f"{x}->{y}" for x, y in p.examples)
        lines.append(f"  {a} -> {b}: {p.surface_type.name} [{p.row}] {p.origin or '-'}; {', '.join(p.reasons)}; "
                     f"{p.n_pv_pairs} (e.g. {ex})")  # fmt: skip
    if coincident.via:
        lines.append(f"coincident faces (via a zero-thickness class): {len(coincident.via)} pairs; "
                     + ", ".join(f"{a} -> {b} via {v}" for (a, b), v in sorted(coincident.via.items())))  # fmt: skip
    lines += coincident.shadowed
    lines += coincident.prenote
    for (a, b), why in sorted(coincident.unresolved.items()):
        lines.append(f"  unresolved coincident face {a} -> {b} ({why}): never crossed; if the geometry crosses it the "
                     "photon is reported lost")  # fmt: skip
    sensor_pvs, sensor_index, sensor_of = tags
    for s in sorted({sensor_of[n] for n in sensor_pvs}):
        t = [sensor_index[n] for n in sensor_pvs if sensor_of[n] == s]
        lines.append(f"sensor {s}: {len(t)} instances, tags {t[0]}..{t[-1]}")
    return "\n".join(lines)


def _media(library, d):
    media_names = tuple(sorted(set(d.medium.values())))
    for name in media_names:
        e = library.media[name]
        try:
            bulk.validate(e.medium, e.params, library.domain)
        except KeyError as err:
            raise OpticsError(f"medium {name!r}: missing params {err}") from None
    return media_names, {name: library.media[name].medium.opaque for name in media_names}


def _resolve_all(library: Library, tree, bindings: Bindings) -> Optics:
    """Return the Optics of a library on a tree without its packed params (``compile`` packs them).

    Notation: C the number of classes.
    """
    d = resolve(library, tree, bindings)
    media_names, opaque = _media(library, d)
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
    # a pass between two PVs of one class is no crossing: left out of the tables, the navigator does not stop there
    found = {k: f for k, f in regions.found.items() if k[0] != k[1] or f[0] != types.PASS}
    rows, row_of, deposit_pre, pairs = _tables(found, C)
    media_of = np.array([media_names.index(s[0]) for s in sigs], np.int32)
    by_name = tree.by_name
    sensor_pvs = sorted((n for n in tree.order if d.sensor[n] is not None),
                        key=lambda n: (d.sensor[n], by_name[n].copy is None, by_name[n].copy or 0, n))  # fmt: skip
    sensor_index = {n: i for i, n in enumerate(sensor_pvs)}
    names = list({r.surface_type.name: None for r in rows})
    head = f"optics: {C} classes, {len(pairs)} crossed class pairs, surface types {names}, {len(sensor_pvs)} sensors"
    report = _report(head, class_names, class_pvs, pairs, coincident, (sensor_pvs, sensor_index, d.sensor))
    born = np.full(C, -1, np.int32)
    born_cls = {class_of[n] for n in sensor_pvs if library.sensors[d.sensor[n]].born}
    for a, b in sorted(pairs):
        if b in born_cls and a != b and born[b] < 0:  # the entry of the sensor's class (any_entry: the same from any)
            born[b] = a
    return Optics(n_classes=C, class_names=class_names, signatures=tuple(sigs), class_of=class_of,
                  class_pvs=class_pvs, rows=rows, row_of=row_of, medium_of=media_of,
                  media=tuple(library.media[n].medium for n in media_names), media_names=media_names, pairs=pairs,
                  sensor_index=sensor_index, sensor_pvs=tuple(sensor_pvs),
                  sensor_entries=tuple(d.sensor[n] for n in sensor_pvs), params={}, report=report,
                  deposit_pre=deposit_pre if deposit_pre.any() else None,
                  born_pair=born if (born >= 0).any() else None)  # fmt: skip


# ------------------------------------------------------------------------------------------ the surface table
def _surface_table(library, rows) -> SurfaceTable:
    """Return the SurfaceTable of the rows: per column which rows read it as a parameter (structure: the entry's
    params, never a value), the rows' constants, the spectral columns' Laws (each row's validated at its own knots)
    and the per-row structure."""  # noqa: D205, D209
    n = len(rows)
    reads = tuple({st.column(q): q for q in st.params} for st, _ in rows)
    cols = [st.row() for st, _ in rows]
    consts = {c: np.asarray([r[c] for r in cols], np.float32) for c in types.COLUMNS}
    # a column is an operand where a row reads it as a parameter or where the rows' constants differ (the photon loop
    # closes over no array); else it is one constant of the code
    params = tuple(c for c in types.COLUMNS if any(c in rd for rd in reads) or np.any(consts[c] != consts[c][:1]))
    laws_, law_of = {}, {}
    for c in params:
        vals = [
            (o, _entry(library, o).params[rd[c]] if c in rd else None) for (_, o), rd in zip(rows, reads, strict=True)
        ]
        if not any(isinstance(v, tuple) for _, v in vals):
            continue
        forms = list({id(v[0]): v[0] for _, v in vals if isinstance(v, tuple)}.values())
        for law in forms:
            if law.params != ("energy", "value"):
                raise OpticsError(f"column {c}: only (energy, value) tables are packed, not {law}")
        flat = next((law for law in forms if law.form == "tabulated"), None)
        if flat is None and any(v is not None and not isinstance(v, tuple) for _, v in vals):
            flat = laws.tabulated("fraction")  # a scalar parameter in a spectral column: its flat table, as itself
            forms.append(flat)
        idx = np.zeros(n, np.int32)
        for i, (o, v) in enumerate(vals):
            if v is None:  # a constant row: its constant, never a law's (``lumen.transport.surface_event``)
                continue
            law, lp = (v[0], v[1]) if isinstance(v, tuple) else (flat, {"energy": np.array([1.0, 2.0]),
                                                                        "value": np.array([v, v])})  # fmt: skip
            idx[i] = forms.index(law)
            laws.validate(law, lp, name=f"{o}.{reads[i][c]}", domain=library.domain)
        laws_[c], law_of[c] = tuple(forms), idx
    present = {st.model for st, _ in rows}
    models = tuple(m for m in types.MODELS if m in present) or (types.G4_REFLECT,)
    fl = [st.flags() for st, _ in rows]
    flags = types.Flags(np.asarray([models.index(st.model) for st, _ in rows], np.int32),
                        *(np.asarray([getattr(f, k) for f in fl], bool) for k in types.Flags._fields[1:]))  # fmt: skip
    support = np.zeros((n, N_OUTCOMES), bool)
    for i, (st, _) in enumerate(rows):
        support[i, list(st.outcomes)] = True
    lobes = tuple(dict.fromkeys(facets.lobe_name(st) for st, _ in rows)) or ("unified",)
    return SurfaceTable(rows=rows, params=params, reads=reads, consts=consts, laws=laws_, law_of=law_of,
                        models=models, flags=flags, support=support, lobes=lobes,
                        lobe_of=np.asarray([lobes.index(facets.lobe_name(st)) for st, _ in rows],
                                           np.int32))  # fmt: skip


def _knots(v):
    """Return the knots (nm) of a packed param or index: a table's ((Law, params) or its params), else none."""
    p = v[1] if isinstance(v, tuple) else v
    return laws.knots(p) if isinstance(p, Mapping) and "energy" in p else np.zeros(0)


def _pair_values(library, optics, pair, origin, named=None):
    """Return (the indices of the two sides' media (None: no index), the entry's params) of a crossing, at the
    build's values or at ``named`` (``lumen.build.pack.named``: a run's values)."""  # noqa: D205, D209
    sides = [(optics.media[optics.medium_of[c]], optics.media_names[optics.medium_of[c]]) for c in pair]
    media = {nm: library.media[nm].params for _, nm in sides} if named is None else named["media"]
    index = [None if m.index is None else media[nm]["index"] for m, nm in sides]
    params = _entry(library, origin).params
    if named is not None:  # the run's values, each table under the entry's own law
        what, name = origin.split(":", 1)
        v = named["surfaces" if what == "surface" else "sensors"][name]
        params = {k: (law[0], v[k]) if isinstance(law, tuple) else v[k] for k, law in params.items()}
    return index, params


def _normal_context(optics, pair, values, lam):
    """Return (params, context) of a crossing's type at normal incidence at the wavelengths ``lam`` (nm)."""
    index, params = values
    lj = jnp.asarray(lam, jnp.float32)
    media = [optics.media[optics.medium_of[c]] for c in pair]
    n = [jnp.ones_like(lj) if i is None else m.index(i, lj) for m, i in zip(media, index, strict=True)]
    p = {k: (v[0](v[1], lj) if isinstance(v, tuple) else jnp.asarray(v, jnp.float32)) for k, v in params.items()}
    ctx = dict(lam=lj, wi=jnp.broadcast_to(jnp.asarray([0.0, 0.0, -1.0]), (lj.shape[0], 3)),
               n=jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0]), (lj.shape[0], 3)), n1=n[0], n2=n[1])  # fmt: skip
    return p, ctx


def normal_absorptance(library, optics, lam) -> np.ndarray:
    """Return each sensor's (the photon loop's order) A_n at the wavelengths ``lam`` (nm): the normal-incidence
    absorptance its deposit is divided by (a type read for a sensor model's QE, ``types.per_normal``), else 1.

    A_n depends on the indices on both sides, so each crossing into the sensor's entry is evaluated; crossings that
    give different A_n are refused (the sensor has no one A_n).
    """  # noqa: D205
    lam = np.atleast_1d(np.asarray(lam, np.float64))
    by_entry: dict = {}
    for (a, b), stype, origin, name in _per_normal_crossings(optics):
        p, ctx = _normal_context(optics, (a, b), _pair_values(library, optics, (a, b), origin), lam)
        a_n = np.asarray(types.normal_absorptance(stype, p, ctx), np.float64)
        if name in by_entry and not np.allclose(by_entry[name], a_n, rtol=PER_NORMAL_RTOL, atol=0.0):
            raise OpticsError(f"sensor entry {name!r}: crossings into it have different normal-incidence absorptance "
                              f"({by_entry[name]} and {a_n} at {lam} nm)")  # fmt: skip
        by_entry[name] = a_n
    return np.stack([by_entry.get(e, np.ones_like(lam)) for e in optics.sensor_entries])


def _per_normal_crossings(optics):
    """Yield ((a, b), surface type, origin, sensor entry name) of each crossing into a sensor whose type is read for
    a sensor model's QE (``types.per_normal``)."""  # noqa: D205, D209
    for ab, pair in sorted(optics.pairs.items()):
        origin = pair.origin
        if origin is not None and origin.startswith("sensor:") and pair.surface_type.per_normal:
            yield ab, pair.surface_type, origin, origin.split(":", 1)[1]


def detection_margins(library, optics, readout, named=None):
    """Yield (origin, (a, b), lam, D, A_n) of each crossing into a sensor whose type is read for a sensor model's QE
    (``types.per_normal``): its model's detection at normal incidence D and the cathode's absorptance A_n there, at the
    build's values or at ``named`` (``lumen.build.pack.named``: a run's, possibly traced; ``readout``: {sensor entry
    name: (``lam -> D``, the model's own wavelength knots or band edges, nm)}). The invariant D <= A_n is checked by
    the run's value checks (``lumen.build.values``): a QE above A_n has no detection probability.

    A_n depends on the indices on both sides, so each crossing is evaluated. Every table involved is held at its end
    values beyond its knots (structure: the build's), so ``lam`` spans the union of their knot spans: every knot, both
    sides of every band edge, and ``PER_NORMAL_POINTS`` wavelengths between. The knots alone do not bound it, as they
    bound a table (``laws.validate``): A_n is a Fresnel function of the indices, not linear in their tables, so D - A_n
    can peak between two knots. The points sample it there; they are a resolution, not a value fitted to anything.
    """  # noqa: D205
    for (a, b), stype, origin, name in _per_normal_crossings(optics):
        if name not in readout:
            continue
        index, params = _pair_values(library, optics, (a, b), origin, named)
        detect, edges = readout[name][0], np.asarray(readout[name][1], np.float64)
        known = np.concatenate([*map(_knots, index), *map(_knots, params.values()), edges])
        if known.size:
            between = np.linspace(known.min(), known.max(), PER_NORMAL_POINTS)
            lam = np.unique(np.concatenate([between, known, np.nextafter(edges, 0.0)]))
        else:  # no table: one wavelength
            lam = np.ones(1)
        p, ctx = _normal_context(optics, (a, b), (index, params), lam)
        a_n = types.normal_absorptance(stype, p, ctx)
        yield origin, (a, b), lam, jnp.broadcast_to(detect(lam), a_n.shape), a_n


def detection_refusal(optics, margins) -> str | None:
    """Return the refusal of the first concrete crossing of ``detection_margins`` whose D exceeds A_n, else None."""
    for origin, (a, b), lam, d, a_n in margins:
        d, a_n = np.asarray(d, np.float64), np.asarray(a_n, np.float64)
        bad = d > a_n * (1.0 + PER_NORMAL_RTOL)
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            return (f"{origin} ({optics.class_names[a]} -> {optics.class_names[b]}): its sensor model's QE "
                    f"{d[i]:.4g} at normal incidence at {lam[i]:.1f} nm exceeds the cathode's absorptance there "
                    f"{a_n[i]:.4g}: no detection probability gives it")  # fmt: skip
    return None


def compile(library: Library, tree, bindings: Bindings) -> Optics:
    """Return the Optics of a library on a tree, its surface and sensor params possibly spectral ((Law, params)).

    The crossings resolved (``_resolve_all``), the surface table's structure (``SurfaceTable``), then the params
    packed by ``pack``: scalar columns stacked, spectral columns as (energy, value) tables padded to a common length
    with clamped tails.
    """
    optics = _resolve_all(library, tree, bindings)
    optics = dataclasses.replace(optics, table=_surface_table(library, optics.rows))
    named = named_params(optics, library)
    order = {"surfaces": None, "media": named["media"]}
    return dataclasses.replace(optics, params=_numpy(pack(optics, named), order))


def _numpy(tree, order):
    """Return a params tree as numpy arrays, its dicts in ``order``'s key order (the library's; a pytree map sorts)."""
    if not isinstance(tree, Mapping):
        return np.asarray(tree)
    keys = list(order) + [k for k in tree if k not in order] if isinstance(order, Mapping) else list(tree)
    return {k: _numpy(tree[k], order.get(k) if isinstance(order, Mapping) else None) for k in keys if k in tree}
