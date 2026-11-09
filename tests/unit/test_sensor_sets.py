"""The shipped lumen:pmt value sets: each loads with its provenance and reproduces its source's own numbers; the
value-set rules fail closed; the registry resolves the model."""

import json
import warnings
from importlib import resources
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import readout_toys as toys

from lumen import plugins
from lumen import readout as rd
from lumen.readout import hits as hm
from lumen.sensors import load_model, pmt, write_defaults

SETS = ("sk_r3600", "wcte_r14374", "wcsim_annie_R7081", "wcsim_annie_D784KFLB", "wcsim_annie_R5912HQE",
        "wcsim_annie_R7081HQE", "sbnd_pmt", "sbnd_pmt_uncoated", "microboone_qe")  # fmt: skip


def _load(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_model(f"lumen:{name}@1")


@pytest.mark.parametrize("name", SETS)
def test_shipped_set_loads_with_provenance(name):
    m = _load(name)
    assert m.provenance["source"] and m.provenance["status"] in pmt.STATUSES
    assert m.readout in pmt.READOUTS
    g = rd.make_grid(m, (-400.0, 1100.0))
    assert g.n_lam == len(m.values["qe"])


def test_unverified_timing_warns():
    with pytest.warns(UserWarning, match="timing unverified"):
        pmt.model_from_dict(
            json.loads((Path(str(resources.files("lumen.sensors"))) / "data/sk_r3600.json").read_text())
        )


def test_sk_spe_and_threshold_reproduce_wcsim():
    """The SK set's SPE table is WCSim's rn1pe (mean 0.9993 pe) and its threshold gives WCSim's A_1 = 0.7390."""
    m = _load("sk_r3600")
    mean, _ = m.charge.moments({"pdf": jnp.asarray(m.values["charge/pdf"])})
    np.testing.assert_allclose(float(mean), 0.99935, atol=2e-5)  # refs_new/sk_consistency_reference.json spe_mean
    v = m.split(m.defaults("pmt", 1), "pmt")
    v = {
        k: ({a: jnp.asarray(b, jnp.float32) for a, b in x.items()} if isinstance(x, dict) else x) for k, x in v.items()
    }
    n, n_neg, q = hm.charge_grid(m)
    dq = m.charge_step
    spe = np.asarray(m.charge.pmf(v["charge"], np.r_[q - dq / 2, q[-1] + dq / 2]))
    acc = np.asarray(m.threshold.accept(v["threshold"], jnp.asarray(q, jnp.float32)))
    np.testing.assert_allclose(np.sum(spe * acc), 0.73902, atol=1e-3)  # the reference's closed form


def test_sbnd_pulse_is_the_ideal_spe():
    """sbnd_pmt's pulse: the fcl's 10-90 % rise and fall times about the 55.1 ns transit time."""
    m = _load("sbnd_pmt")
    t = jnp.linspace(40.0, 100.0, 60001)
    h = np.asarray(m.pulse.shape({k: jnp.asarray(m.values[f"pulse/{k}"]) for k in m.pulse.fields()}, t))
    tt = np.asarray(t)
    rise = tt[(h >= 0.9) & (tt < 55.1)].min() - tt[(h >= 0.1) & (tt < 55.1)].min()
    fall = tt[(h >= 0.1) & (tt > 55.1)].max() - tt[(h >= 0.9) & (tt > 55.1)].max()
    np.testing.assert_allclose([rise, fall], [3.8, 13.7], atol=2e-3)


@pytest.mark.parametrize("change,match", [
    (dict(tts=dict(form="lorentz", sigma_ns=1.0)), "tts form 'lorentz'"),
    (dict(ranges={}), "declare the ranges"),
    (dict(provenance=dict(source="", status="measured")), "provenance needs a source"),
    (dict(qe=dict(wavelength_nm=[300.0, 400.0], values=[0.1])), "qe has 1 values for 2"),
    (dict(charge=dict(form="gauss_exp", sigma_pe=0.3)), "needs the fields"),
])  # fmt: skip
def test_value_set_rules_fail_closed(change, match):
    with pytest.raises(pmt.SensorValuesError, match=match):
        toys.model(**change)


def test_unknown_set_and_registry(tmp_path):
    with pytest.raises(pmt.SensorValuesError, match="no shipped sensor value set"):
        load_model("lumen:nonexistent@1")
    assert plugins.sensor_models.get("lumen:pmt") is load_model
    out = write_defaults("lumen:wcte_r14374@1", tmp_path / "mine.json")
    assert json.loads(out.read_text())["name"] == "wcte_r14374"
