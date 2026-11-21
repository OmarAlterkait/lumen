"""Sensor models after tracking: ``lumen:sensor`` (acceptance tables over features of the deposit, a delay per feature
cell, time spread, SPE charge, threshold, pulse, dark rate, front end; ``lumen.sensors.features`` the features) and its
value sets (a path, or ``<package>:<set>``); see ``lumen.sensors.value_sets``. ``group_model(group)`` is a sensor
group's model, built from its value set (``SensorGroup.values``): a group with a value set is modelled.
"""  # noqa: D205

from lumen.sensors.value_sets import (
    SensorValuesError,
    SensorValuesModel,
    load_model,
    model_from_dict,
    model_of,
    read_values,
    write_defaults,
)


def group_model(group):
    """Return a sensor group's sensor model, built from its value set, or None when the group has none.

    The value set is the group's ``values``: its content (a converted detector's copy), or the reference a recipe
    gives (a group not yet converted), read now.
    """
    if group.values is None:
        return None
    return model_of(read_values(group.values) if isinstance(group.values, str) else group.values)


def group_models(sensor_set) -> dict:
    """Return {group name: SensorValuesModel} of every group of a sensor set that has a value set, in its order."""
    return {g.name: m for g in sensor_set.groups if (m := group_model(g)) is not None}


def normal_detection(sensor_set, parts, params) -> dict:
    """Return {sensor entry name: (lam -> D, wavelength knots or band edges (nm))} of the groups whose sensor model
    applies its QE after tracking on a cathode read for it (``types.per_normal``): D the model's detection at normal
    incidence at the values ``params`` (possibly traced), the largest over the group's sensors (the run's check of
    D <= A_n, ``lumen.build.values``, ``lumen.build.lookup.detection_margins``)."""  # noqa: D205, D209
    import jax
    import jax.numpy as jnp
    import numpy as np

    from lumen._core.safe import traced
    from lumen.build.build import group_tags
    from lumen.rules import sensors as sensor_rules

    optics, out = parts.optics, {}
    for g in sensor_set.groups:
        model = group_model(g)
        if model is None or not sensor_rules.after_tracking(g):
            continue
        tags = group_tags(g, parts, sensor_set)
        if not tags:
            continue
        v = model.split(params, g.name)
        wl = model.axis("wavelength")  # its wavelength knots or the edges between its bands
        lam_points = [] if wl is None else list(wl.knots if wl.knots is not None else wl.edges[1:-1])
        sensors = np.arange(len(tags))
        leaves = [] if traced(v) else jax.tree_util.tree_leaves(v)  # a traced value is never read on the host
        per = [np.asarray(x) for x in leaves if np.shape(x)[:1] == (len(tags),)]
        if per:  # concrete values: sensors of equal values detect alike, one of each (SK: one of 11146)
            sensors = np.sort(np.unique(np.stack([x.reshape(len(tags), -1) for x in per], 1).reshape(len(tags), -1),
                                        axis=0, return_index=True)[1])  # fmt: skip

        def detect(lam, model=model, v=v, sensors=sensors):
            rows = model.normal_rows(np.broadcast_to(np.asarray(lam)[None, :], (len(sensors), np.size(lam))))
            return jnp.max(jnp.asarray(model.detection(v, sensors[:, None], rows)), axis=0)

        for name, e in parts.library.sensors.items():  # the group's sensors: the borders into its volumes, and out
            if e.group == g.name or name in {optics.sensor_entries[t] for t in tags}:
                out[name] = (detect, lam_points)
    return out


__all__ = [
    "SensorValuesError", "SensorValuesModel", "group_model", "group_models",
    "load_model", "model_from_dict", "read_values", "write_defaults",
]  # fmt: skip
