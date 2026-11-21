"""lumen:sensor value sets: the value-set rules fail closed; a set is a path or <package>:<set>; a group with a value
set is modelled by it (nothing registered). (The experiments' sets and their tests are lumen-wc's and lumen-tpc's.)"""

import json

import numpy as np
import pytest
import readout_toys as toys

from lumen.sensors import load_model, value_sets, write_defaults


@pytest.mark.parametrize("change,match", [
    (dict(tts=dict(form="lorentz", sigma_ns=1.0)), "tts form 'lorentz'"),
    (dict(ranges={}), "declare the ranges"),
    (dict(provenance=dict(source="", status="measured")), "provenance needs a source"),
    (dict(qe=dict(wavelength_nm=[300.0, 400.0], values=[0.1])), r"qe has values of shape \(1,\) for its axes"),
    (dict(charge=dict(form="gauss_exp", sigma_pe=0.3)), "needs the fields"),
    (dict(model="lumen:pmt"), "for model 'lumen:pmt'; this model is 'lumen:sensor'"),  # the old name: refused
    (dict(ranges={**toys.toy_set()["ranges"], "pulse/t_peek": [4.0, 8.0]}), "a range for 'pulse/t_peek', which is no"),
])  # fmt: skip
def test_value_set_rules_fail_closed(change, match):
    with pytest.raises(value_sets.SensorValuesError, match=match):
        toys.model(**change)


def test_a_component_field_has_its_form_s_range():
    """A field a value set need not declare a range for has its component form's (``Form.RANGES``): a charge
    fraction above 1 or a negative width is refused where values enter, as every range is."""
    from lumen.params import Params

    info = toys.model().param_info("g")
    assert info["sensors/g/charge/fraction"].range == (0.0, 1.0)
    assert info["sensors/g/charge/sigma"].range == info["sensors/g/charge/tau"].range == (0.0, np.inf)
    p = Params(dict.fromkeys(info, 0.5), info, {})
    for key, bad in (("sensors/g/charge/fraction", 1.5), ("sensors/g/charge/sigma", -0.1)):
        with pytest.raises(ValueError, match="outside"):
            p.set(key, bad)


def test_unknown_set_and_registry(tmp_path):
    with pytest.raises(value_sets.SensorValuesError, match="no sensor value set 'lumen:nonexistent'"):
        load_model("lumen:nonexistent")  # lumen ships none
    with pytest.raises(value_sets.SensorValuesError, match="no package 'no_such_package'"):
        load_model("no_such_package:x")
    pkg = tmp_path / "mypkg"
    (pkg / "sensor_values").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "sensor_values" / "toy.json").write_text(json.dumps(toys.toy_set()))
    import sys

    sys.path.insert(0, str(tmp_path))
    try:
        assert load_model("mypkg:toy").set_name == "toy"  # <package>:<set>: the package's sensor_values/<set>.json
        with pytest.raises(value_sets.SensorValuesError, match="no sensor value set 'mypkg:toy@2'"):
            load_model("mypkg:toy@2")  # no version pin: a set is its name
    finally:
        sys.path.remove(str(tmp_path))
    out = write_defaults(toys.TOY_SET, tmp_path / "mine.json")
    assert json.loads(out.read_text())["name"] == "toy"


def test_the_time_walk_is_read_like_every_component():
    """Its unit suffixes are checked and an extra field refused, as for every other component (it bypassed them)."""
    tw = {"charge_pe": [0.5, 1.0], "shift_ns": [0.0, 0.1], "sigma_ns": [1.0, 0.5]}
    m = toys.model(time_walk=tw)
    assert m.values["time_walk/sigma"].dtype == np.float64 and m.time_walk.knots == (0.5, 1.0)
    with pytest.raises(value_sets.SensorValuesError, match="is in pe, the field is in ns"):
        toys.model(time_walk={"charge_pe": tw["charge_pe"], "shift_ns": tw["shift_ns"], "sigma_pe": tw["sigma_ns"]})
    with pytest.raises(value_sets.SensorValuesError, match="needs the fields"):
        toys.model(time_walk={**tw, "extra_ns": [0.0, 0.0]})


def test_the_sensor_model_is_the_value_set_model():
    """Nothing is registered: a group with a value set is modelled, its model built from the value set directly (its
    content, or a reference read now); a group without one has no model; a schema other than lumen-sensor-values/1 is
    refused naming it."""
    from lumen.format.sensorsets import SensorGroup
    from lumen.sensors import SensorValuesModel, group_model

    g = SensorGroup("pmt", rule="any_entry", lv="S", values=toys.toy_set())
    assert isinstance(group_model(g), SensorValuesModel) and group_model(g) is group_model(g)
    assert group_model(SensorGroup("pmt", rule="any_entry", lv="S", values=toys.TOY_SET)) is group_model(g)
    assert group_model(SensorGroup("x", rule="any_entry", lv="S")) is None
    with pytest.raises(ValueError, match="lumen-sensor-values/1"):
        value_sets.model_from_dict({**toys.toy_set(), "schema": "lumen-pmt-values/3"})


def test_the_value_set_schema_reserves_package_names_and_refuses_other_records():
    """A form or axis named "package:Name" is reserved (refused by this lumen); a record outside the schema (the 0.1
    hit_axes) and the second wavelength-band spelling are refused."""
    base = toys.toy_set()
    with pytest.raises(value_sets.SensorValuesError, match="'package:Name'"):
        value_sets.model_from_dict({**base, "tts": {"form": "mypkg:Lognormal", "sigma_ns": 1.0}})
    with pytest.raises(value_sets.SensorValuesError, match="'package:Name'"):
        value_sets.model_from_dict({**base, "axes": {"mypkg:radius_m": [0.0, 1.0]}})
    with pytest.raises(value_sets.SensorValuesError, match=r"unknown records \['hit_axes'\]"):
        value_sets.model_from_dict({**base, "hit_axes": {"x_m": [0.0, 1.0]}})
    with pytest.raises(value_sets.SensorValuesError, match="band_edges_nm"):
        value_sets.model_from_dict({**base, "axes": {"wavelength_edges_nm": [400.0]}})


def test_the_efficiency_on_entry_is_a_setting_of_a_group_without_a_value_set():
    """sensors/<group>/efficiency is the value set's per-sensor efficiency after tracking; the efficiency on entry of a
    group without a value set is its setting entry_efficiency (sensors/<group>/entry_efficiency): one key, one meaning.
    A group with a value set counts every absorption entering, so the setting is refused there, as is an unknown one."""
    from lumen.build import build
    from lumen.format.sensorsets import SensorGroup, SensorSet

    with pytest.raises(ValueError, match="drop the setting entry_efficiency"):
        SensorSet((SensorGroup("pmt", rule="any_entry", lv="P", values=toys.TOY_SET,
                               settings={"entry_efficiency": 1.0}),))  # fmt: skip
    with pytest.raises(ValueError, match=r"unknown settings \['efficiency'\]"):
        SensorSet((SensorGroup("x", rule="any_entry", lv="X", settings={"efficiency": 0.5}),))
    assert set(build.group_settings(SensorGroup("x", rule="any_entry", lv="X", settings={"entry_efficiency": 0.5}))) \
        == {"sensors/x/entry_efficiency"}  # fmt: skip
