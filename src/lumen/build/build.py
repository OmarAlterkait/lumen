"""Build a detector from its file: the geometry lowered to its scene, the volume tree, the surfaces and media.

    parts = build(detector_file, sensor_set)       # Parts: scene, tree, optics, library, ...
    params = parameters(detector_file, sensor_set, parts, calibration)

Nothing here reads a recipe: every choice is in the detector file and its sensor set.
"""

from __future__ import annotations

from dataclasses import dataclass

from lumen.build import library as optics_library
from lumen.build import lookup, lowering, pack, spec
from lumen.build import tree as volume_tree
from lumen.geometry import scene as sc
from lumen.models import model_table
from lumen.sensors import group_model


@dataclass(frozen=True, eq=False)
class Parts:
    """The built detector: the lowered geometry, the scene the navigator uses, the trees and the optics.

    ``key_rows`` maps each packed leaf (section, entry, *leaf) to the ``library.Source`` it reads; ``named`` is the
    surface lookup's named parameters as built, ``reads`` the leaves read from a key [(path, key, column, derive)]
    and ``knots`` the knot energies the build packs per key (``lumen.build.pack``, ``lumen.build.values``).
    """

    scene: object
    nav_scene: object
    tree: object
    instances: tuple
    library: object
    optics: object
    key_rows: dict
    named: dict
    reads: tuple
    knots: dict
    models: dict  # the medium models passed to the build, by name (lumen.models)
    nav_instances: object = None  # the navigator scene's ``scene.instances``, when the build listed them


def build(df, sensor_set, models=()) -> Parts:
    """Return the Parts of a detector file and sensor set; ``models``: the medium models passed to the build (values,
    ``lumen.models.MediumModel``), the build's model table with lumen's own."""  # noqa: D205, D209
    table = model_table(models)
    loaded = lowering.lower(df, detector=df.detector_volume, sensors=sensor_set.is_sensor)
    nav_scene = lowering.split_unions(loaded.scene)
    T = loaded.instances if loaded.instances is not None else sc.instances(loaded.scene)
    tree, insts = volume_tree.tree_from_geometry(loaded.scene, loaded.media, T)
    library, bindings, key_rows = optics_library.library_and_bindings(df, tree, sensor_set, table)
    optics = lookup.compile(library, tree, bindings)
    named, reads, knots = pack.leaf_map(optics, library, key_rows)
    return Parts(loaded.scene, nav_scene, tree, tuple(insts), library, optics, key_rows, named, reads, knots, table,
                 T if nav_scene is loaded.scene else None)  # fmt: skip


def group_settings(g) -> dict:
    """Return ``sensors/<group>/<setting>`` -> (value, Param) of a group's settings that are values."""
    return {f"sensors/{g.name}/{field}": (float(v), spec.info(field)) for field, v in g.settings.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}  # fmt: skip


def sensor_defaults(sensor_set, parts) -> dict:
    """Return ``sensors/<group>/<field>`` -> (value, Param): each group's settings that are values, then the
    default values of its after-tracking sensor model (``lumen.sensors.group_model``, its value set;
    per-sensor fields over the group's sensors in ``parts``). Every sensor is in exactly one group (the groups that
    share a volume partition its sensors), refused otherwise."""  # noqa: D205, D209
    out, owner = {}, {}
    for g in sensor_set.groups:
        out.update(group_settings(g))
        tags = group_tags(g, parts, sensor_set)
        for t in tags:
            if t in owner:
                raise ValueError(f"sensor {parts.optics.sensor_pvs[t]!r} is in the sensor groups {owner[t]!r} and "
                                 f"{g.name!r}: a sensor is in one group")  # fmt: skip
            owner[t] = g.name
        model = group_model(g)
        if model is not None:
            info = model.param_info(g.name)
            out.update({k: (v, info[k]) for k, v in model.defaults(g.name, len(tags)).items()})
    free = [pv for t, pv in enumerate(parts.optics.sensor_pvs) if t not in owner]  # a volume's groups cover it
    if free:
        i = sensor_set._pos.get(free[0])  # type: ignore[attr-defined]
        cols = {c: v[i] for c, v in sensor_set.columns.items()} if i is not None else {}
        raise ValueError(f"{len(free)} sensors are in no sensor group, e.g. {free[0]!r} (id-map columns {cols}): the "
                         "groups that select a volume select each of its sensors, in exactly one group")  # fmt: skip
    return out


def group_tags(group, parts, sensor_set) -> list[int]:
    """Return the sensor tags (the photon loop's sensor indices) of a sensor group, ascending."""
    pvs = parts.optics.sensor_pvs
    return [i for i, pv in enumerate(pvs) if sensor_set.in_group(group, pv, parts.tree.by_name[pv].lv)]


def parameters(df, sensor_set, parts, calibration=None, responses=()):
    """Return the Params of a built detector: its file's values, the sensor defaults, the responses' parameters at
    their defaults (``lumen.response``), then the calibration's values (None: nominal)."""  # noqa: D205, D209
    params = spec.param_spec_and_values(df, sensor_defaults(sensor_set, parts), parts.key_rows, parts.models)
    return spec.resolve(add_responses(params, sensor_set, responses), calibration)


def add_responses(params, sensor_set, responses):
    """Return ``params`` with each response's keys ``sensors/<group>/<name>`` added at their defaults."""
    groups = {g.name for g in sensor_set.groups}
    for r in responses:
        if r.group not in groups:
            raise ValueError(f"load(responses=): {r.name!r} reads group {r.group!r}; the groups are {sorted(groups)}")
        taken = [r.prefix + k for k, _ in r.declared if r.prefix + k in params
                 and params.explain(r.prefix + k) != f"response {r.name}"]  # fmt: skip
        if taken:
            raise ValueError(f"load(responses=): {r.name!r} declares {taken}, keys the detector has already "
                             "(rename them)")  # fmt: skip
        params = r.parameters(params)
    return params
