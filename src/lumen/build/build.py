"""Build a detector from its file: the geometry lowered to its scene, the volume tree, the surfaces and media.

    parts = build(detector_file, sensor_set)       # Parts: scene, tree, optics, library, ...
    params = parameters(detector_file, sensor_set, parts, calibrations)

Nothing here reads a recipe: every choice is in the detector file and its sensor set.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from lumen.build import library as optics_library
from lumen.build import lowering, spec, surfaces
from lumen.build import tree as volume_tree
from lumen.params import ParamInfo


@dataclass(frozen=True, eq=False)
class Parts:
    """The built detector: the lowered geometry, the scene the navigator uses, the trees and the optics."""

    loaded: object
    scene: object
    nav_scene: object
    base_tree: object
    tree: object
    instances: tuple
    library: object
    bindings: object
    optics: object
    sources: dict


def build(df, sensor_set) -> Parts:
    """Return the Parts of a detector file and sensor set."""
    loaded = lowering.lower(df, detector=df.detector_volume, sensors=sensor_set.is_sensor)
    nav_scene = lowering.split_unions(loaded.scene)
    base = volume_tree.tree_from_geometry(loaded)
    insts = tuple(volume_tree.instances(loaded))
    tree, library, bindings, sources = optics_library.library_and_bindings(bound(df), base, insts, sensor_set)
    optics = surfaces.compile(library, tree, bindings, df.wavelength_range)
    return Parts(loaded, loaded.scene, nav_scene, base, tree, insts, library, bindings, optics, sources)


def bound(df):
    """Return the detector file with only the borders the simulation binds (``Border.bound``)."""
    if all(b.bound for b in df.borders):
        return df
    return dataclasses.replace(df, borders=tuple(b for b in df.borders if b.bound))


def sensor_defaults(sensor_set, parts=None) -> dict:
    """Return ``sensors/<group>/<field>`` -> (value, ParamInfo): each group's settings that are values, then the
    default values of its after-tracking sensor model (``lumen.plugins.sensor_models``, the value set ``defaults``;
    per-sensor fields over the group's sensors in ``parts``)."""  # noqa: D205, D209
    from lumen import plugins

    out = {}
    for g in sensor_set.groups:
        for field, v in g.settings.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and field != "name_prefix":
                key = f"sensors/{g.name}/{field}"
                out[key] = (float(v), ParamInfo(key, "", (0.0, 1.0), "reweight"))
        if g.model in plugins.sensor_models.names() and parts is not None:
            if g.defaults is None:
                raise ValueError(f"sensor group {g.name!r}: model {g.model!r} needs its value set (defaults)")
            model = plugins.sensor_models.get(g.model)(g.defaults)
            info = model.param_info(g.name)
            n = len(group_tags(g, parts))
            out.update({k: (v, info[k]) for k, v in model.defaults(g.name, n).items()})
    return out


def group_tags(group, parts) -> list[int]:
    """Return the sensor tags (the photon loop's sensor indices) of a sensor group, ascending."""
    return [i for i, pv in enumerate(parts.optics.sensor_pvs) if group.selects(parts.tree.by_name[pv].lv)]


def parameters(df, sensor_set, parts, calibrations=()):
    """Return the Params of a built detector: its file's values, the sensor defaults, then the calibrations."""
    return spec.resolve(spec.param_spec_and_values(df, sensor_defaults(sensor_set, parts)), calibrations)
