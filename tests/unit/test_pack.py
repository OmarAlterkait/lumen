"""``pack`` is a gather (lumen.build.pack): each packed value read from a key is that key's value, the rest the build's.
Its arrays are, bit for bit, those of the library rebuilt from the values applied to the detector file (the oracle
here: what pack did before), at the detector's values, at a new value of any key, for a calibrated detector, and under
``jax.jit``."""

import numpy as np
from test_yaml import tank  # noqa: F401 -- the tutorial tank, converted once

import lumen
from lumen.build import library, lookup, pack
from lumen.format.apply import apply, sensor_set
from lumen.format.calibrations import Table
from lumen.rules.optics import VALUE_RULES


def _bits(tree):
    import jax

    return [np.asarray(x).tobytes() for x in jax.tree_util.tree_leaves(tree)]


def rebuilt(det, params):
    """The oracle: the library rebuilt on the host from the values applied to the detector file, packed."""
    df, sset = apply(det.file, params, VALUE_RULES), sensor_set(det.sensor_set, params)
    parts = det.parts
    lib, _, _ = library.library_and_bindings(df, parts.tree, sset)
    return lookup.pack(parts.optics, lookup.named_params(parts.optics, lib))


def _moved(v):
    if isinstance(v, Table):
        return Table(np.asarray(v.energy), np.asarray(v.value) * 1.01)
    a = np.asarray(v, np.float64)
    return a * 0.99 if np.ndim(a) else (float(a) * 0.99 if float(a) else 0.01)


def test_each_key_packs_as_the_rebuilt_library_concrete_and_jitted(tank):  # noqa: F811
    import jax

    det = lumen.load(tank)
    p0 = det.params()
    assert _bits(pack.pack(det, p0)) == _bits(rebuilt(det, p0)) == _bits(det.parts.optics.params)
    jitted = jax.jit(lambda q: pack.pack(det, q))
    moved = 0
    for k in p0:
        try:
            p = p0.set(k, _moved(p0[k]))
        except ValueError:  # outside the key's range: another key
            continue
        want = _bits(rebuilt(det, p))
        assert _bits(pack.pack(det, p)) == want, k
        assert _bits(jitted(p)) == want, k
        moved += 1
    assert moved


def test_a_calibrated_detector_packs_its_calibration_s_values(tank):  # noqa: F811
    """A calibration's values are gathered as any values are; nothing is rebuilt."""
    det = lumen.load(tank)
    p0 = det.params()
    k = sorted(src.key for src in det.parts.key_rows.values() if src.key and isinstance(p0[src.key], Table))[0]
    det.save_calibration(p0.set(k, _moved(p0[k])), "pack_moved", tank / "calibrations")
    cdet = lumen.load(tank, calibration="pack_moved")
    q = cdet.params()
    assert _bits(pack.pack(cdet, q)) == _bits(rebuilt(cdet, q))
    assert _bits(pack.pack(cdet, q)) != _bits(pack.pack(det, p0))


def test_a_table_s_knot_energies_are_the_detector_s(tank):  # noqa: F811
    """The knot energies the build packs are structure (as the knot count): new ones are refused (InvalidValue), and
    counted under jax.jit, never packed silently with the old ones."""
    import jax
    import pytest

    from lumen.build import values

    det = lumen.load(tank)
    p0 = det.params()
    k = sorted(src.key for src in det.parts.key_rows.values() if src.key and src.column == "energy")[0]
    v = p0[k]
    moved = p0.set(k, Table(np.asarray(v.energy) * 1.001, np.asarray(v.value)))
    with pytest.raises(values.InvalidValue, match="knot energies"):
        pack.checked(det, moved)
    assert int(jax.jit(lambda q: values.check(det, q)[2])(moved)) == len(v.energy)
    assert int(jax.jit(lambda q: values.check(det, q)[2])(p0)) == 0
