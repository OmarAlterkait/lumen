"""The validation machinery without a photon loop: manifests and the store, records, the report, the statistics, the
charge check's reference reading and sensor matching, and the gradient check's parameter families."""

import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lumen import plugins
from lumen.surfaces.types import SurfaceType
from lumen.validate import charge, gradients, records, references, report, stats


def product(store, pid, files):
    d = store / pid
    d.mkdir(parents=True)
    for name, text in files.items():
        (d / name).write_bytes(text)
    return {n: "sha256:" + hashlib.sha256(t).hexdigest() for n, t in files.items()}


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    """A detector package's own manifest and store (another detector, no code change)."""
    store = tmp_path / "store"
    hits = tmp_path / "hits.npz"
    np.savez(hits, src=np.array([0, 0, 0], np.int32), sensor=np.array([0, 1, 1], np.int32),
             t=np.array([1.0, 2.0, 2.5], np.float32), E=np.array([3.0, 3.0, 3.0], np.float32))  # fmt: skip
    files = {"hits.npz": hits.read_bytes(), "launched.json": b"[100]", "sources.cfg": b"0 0 0 50 3.0\n",
             "sensors.csv": b"sensor,path,lv,cx_mm,cy_mm,cz_mm\n0,a,pmt,0,0,1000\n1,b,pmt,0,1000,0\n"}  # fmt: skip
    charge_entry = dict(name="toydet", detector="toydet", reference="toyref", dt=1.0, nt=10)
    doc = dict(schema=references.SCHEMA, store=str(store), checks={"charge": [charge_entry]},
               products={"toyref": dict(files=product(store, "toyref", files))})  # fmt: skip
    path = tmp_path / "references.json"
    path.write_text(json.dumps(doc))
    pkg = plugins.DetectorPackage("toydet", None, "gdml/toy.gdml", references=str(path))
    monkeypatch.setitem(plugins.detectors._entries, "toydet", pkg)
    return store


def test_a_detector_package_brings_its_own_references(manifest):
    spec = references.spec("charge", "toydet")
    assert spec["reference"] == "toyref" and "toydet" in references.names("charge")
    ref = charge.read_reference(references.product("toyref", spec))
    assert ref["launched"].tolist() == [100.0] and ref["centres"].shape == (2, 3) and ref["n_cfg"].tolist() == [50.0]
    assert "sk" in references.names("charge")  # lumen's own entries stay listed


def test_a_product_whose_file_changed_is_refused(manifest):
    spec = references.spec("charge", "toydet")
    f = manifest / "toyref" / "launched.json"
    f.write_bytes(b"[101]")
    references._VERIFIED.clear()
    with pytest.raises(references.ReferenceStoreError, match="sha256"):
        references.product("toyref", spec)


def test_records_are_never_written_into_a_store(manifest, tmp_path):
    rows = [records.row("charge", "toydet", "ratio", 0.999, 0.001, z=-1.0, passed=True, reference="toyref")]
    with pytest.raises(PermissionError, match="reference store"):
        records.write(rows, manifest / "out", "charge_toydet")
    path = records.write(rows, tmp_path / "out", "charge_toydet")
    back = records.read([tmp_path / "out"])
    assert path.exists() and set(records.FIELDS) <= set(back[0]) and back[0]["value"] == 0.999
    md = report.markdown(back)
    assert "| toydet | ratio | 0.999 ± 0.001 |" in md and "## Not validated" in md


def test_the_pass_rules_and_the_statistics():
    assert stats.chi2_passes(1.04, 11096) and not stats.chi2_passes(1.06, 11096)
    assert stats.chi2_passes(1.7, 30) and not stats.chi2_passes(1.8, 30)
    assert stats.t_null(32) == pytest.approx(31 / 29)
    h, late, early = stats.node_split(np.array([0.5, 9.75, -0.5]), 1.0, 10)
    assert h.sum() + late + early == pytest.approx(3.0) and h[0] == pytest.approx(1.0) and late == pytest.approx(0.75)
    assert stats.richardson(lambda x: np.exp(2 * x), 1 / 16) == pytest.approx(2.0, rel=1e-5)
    assert stats.richardson(lambda x: x**2 + x, 1 / 64, one_sided=True) == pytest.approx(1.0, abs=1e-12)


def test_coincident_sensors_count_as_one():
    ref = dict(centres=np.array([[0.0, 0.0, 1000.0], [0.0, 1000.0, 0.0]]))
    lumen = np.array([[0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0], [0.0, 0.0, 1000.0]])  # sensors 1 and 2 coincide
    of_ref, can = charge.match(ref, lumen, 5.0)
    assert of_ref.tolist() == [1, 0] and can.tolist() == [0, 1, 1]
    with pytest.raises(ValueError, match="farther"):
        charge.match(ref, lumen + 10.0, 5.0)


class _Optics:
    rows = {"painted": ["surface:A", "surface:B"]}
    surface_types = (SurfaceType("painted", ("R", "sigma_alpha"), (), lambda p, ctx: None),)  # its parameters' domains


def test_the_gradient_families_and_their_values():
    p0 = {"surface_types": {"painted": {"R": np.array([0.3, 1.0], np.float32), "sigma_alpha": np.array([0.1, 0.0])}},
          "media": {"Water0x1": {"absorption": {"energy": np.ones(2), "value": np.array([20.0, 30.0])},
                                 "index": {"energy": np.ones(2), "value": np.array([1.33, 1.34])}}}}  # fmt: skip
    p0 = jax.tree_util.tree_map(jnp.asarray, p0)
    fams = gradients.families(_Optics(), p0)
    names = {f[0]: f[3] for f in fams}
    assert names == {"painted.R[surface:A]": "add", "painted.R[surface:B]": "edge",
                     "painted.sigma_alpha[surface:A]": "mul", "Water:absorption": "mul"}  # fmt: skip
    p = gradients.apply(p0, fams, np.array([0.1 if f[0] == "painted.R[surface:A]" else 0.0 for f in fams]))
    assert float(p["surface_types"]["painted"]["R"][0]) == pytest.approx(0.3 + 0.1 * 0.3)


def test_a_sharded_gradient_check_merges_into_one_summary():
    """Each shard holds families j % n == i; the merge needs every family exactly once, from one sharding."""
    from lumen.validate import gradients

    def fam(q, ok, n=2):
        return records.row("gradients", "toy", f"dQ {q}", 1.0, passed=ok, seconds=1.0, shard_of=n, families_total=3,
                           family="abc".index(q), settings={"photons": 16})  # fmt: skip

    a, b = [fam("a", True), fam("c", True)], [fam("b", None)]
    rows = gradients.merge("toy", [a, b])
    s = rows[-1]
    assert s["quantity"] == "families passing" and s["pass"] is True and s["value"] == 2
    assert s["extra"]["gated"] == 2 and s["extra"]["families"] == 3
    with pytest.raises(ValueError, match="two shards"):
        gradients.merge("toy", [a, a])
    with pytest.raises(ValueError, match="2 of 3"):
        gradients.merge("toy", [a, []])
