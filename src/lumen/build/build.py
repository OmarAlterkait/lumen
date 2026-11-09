"""Build a detector from its file: the geometry lowered to its scene, the volume tree, the surfaces and media.

    parts = build(detector_file, sensor_set)       # Parts: scene, tree, optics, library, ...
    params = parameters(detector_file, sensor_set, parts, calibrations)

Nothing here reads a recipe: every choice is in the detector file and its sensor set.
"""

from __future__ import annotations

from dataclasses import dataclass

from lumen.build import _legacy_regions, legacy_library, lowering, spec
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
    sv = sensor_set.volumes
    loaded = lowering.lower(df, detector=df.detector_volume, sensors=lowering.Sensors(aux=sv.aux, names=sv.names))
    n = df.build_options.get("split_unions")
    nav_scene = lowering.split_unions(loaded.scene, n) if n else loaded.scene
    base = volume_tree.tree_from_geometry(loaded)
    insts = tuple(volume_tree.instances(loaded))
    tree, library, bindings, sources = legacy_library.library_and_bindings(df, base, insts, sensor_set)
    optics = _legacy_regions.compile(library, tree, bindings, df.wavelength_range)
    return Parts(loaded, loaded.scene, nav_scene, base, tree, insts, library, bindings, optics, sources)


def sensor_defaults(sensor_set) -> dict:
    """Return ``sensors/<group>/<field>`` -> (value, ParamInfo) of each group's model settings that are values."""
    out = {}
    for g in sensor_set.groups:
        for field, v in g.settings.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and field != "name_prefix":
                key = f"sensors/{g.name}/{field}"
                out[key] = (float(v), ParamInfo(key, "", (0.0, 1.0), "reweight"))
    return out


def parameters(df, sensor_set, parts, calibrations=()):
    """Return the Params of a built detector: its file's values, the sensor defaults, then the calibration chain."""
    return spec.resolve(spec.param_spec_and_values(df, sensor_defaults(sensor_set)), calibrations)
