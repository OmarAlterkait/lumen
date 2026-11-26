"""The validation machinery without a photon loop: manifests and the store, records, the report, the statistics, the
charge check's reference reading and sensor matching, and the gradient check's parameter families."""

import hashlib
import json
import types

import numpy as np
import pytest

from lumen.build import lookup
from lumen.compare import charge, gradients, records, references, report
from lumen.surfaces.types import G4_REFLECT, SurfaceType
from lumen.tools import stats


def product(store, pid, files):
    d = store / pid
    d.mkdir(parents=True)
    for name, text in files.items():
        (d / name).write_bytes(text)
    return {n: "sha256:" + hashlib.sha256(t).hexdigest() for n, t in files.items()}


@pytest.fixture
def manifest(tmp_path):
    """Another detector's own manifest and store (no code change): (its store, its manifest's path)."""
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
    return store, path


def test_a_detector_brings_its_own_references(manifest):
    _, path = manifest
    spec = references.spec("charge", "toydet", (path,))
    assert spec["reference"] == "toyref" and "toydet" in references.names("charge", (path,))
    ref = charge.read_reference(references.product("toyref", spec, (path,)))
    assert ref["launched"].tolist() == [100.0] and ref["centres"].shape == (2, 3) and ref["n_cfg"].tolist() == [50.0]
    assert "coinc_sib" in references.names("overlaps", (path,))  # lumen's own entries stay listed


def test_a_check_name_is_one_entry_across_the_manifests(manifest, tmp_path):
    """A manifest naming a check another manifest names is refused, never silently preferred."""
    store, path = manifest
    again = tmp_path / "again.json"
    again.write_text(path.read_text())
    with pytest.raises(references.ReferenceStoreError, match="2 manifests name a charge check 'toydet'"):
        references.spec("charge", "toydet", [path, again])


def test_a_product_whose_file_changed_is_refused(manifest):
    store, path = manifest
    spec = references.spec("charge", "toydet", (path,))
    f = store / "toyref" / "launched.json"
    f.write_bytes(b"[101]")
    references._VERIFIED.clear()
    with pytest.raises(references.ReferenceStoreError, match="sha256"):
        references.product("toyref", spec, (path,))


def test_a_missing_product_is_fetched_from_its_release_asset_and_checked(tmp_path, monkeypatch):
    """A product the store lacks comes from its entry's url, or from the same file name under the LUMEN_ASSETS mirror;
    the archive's sha256 is checked and the product placed whole."""
    import tarfile

    files = {"a.txt": b"alpha", "b.npz": b"beta"}
    digests = product(tmp_path / "src", "p", files)
    asset = tmp_path / "assets" / "p.tar.gz"
    asset.parent.mkdir()
    with tarfile.open(asset, "w:gz") as t:
        for name in files:
            t.add(tmp_path / "src" / "p" / name, arcname=name)
    entry = dict(files=digests, url=asset.as_uri(), sha256="sha256:" + hashlib.sha256(asset.read_bytes()).hexdigest())
    store = tmp_path / "store"
    path = tmp_path / "references.json"
    path.write_text(json.dumps(dict(schema=references.SCHEMA, store=str(store), checks={}, products={"p": entry})))
    d = references.product("p", {"_store": str(store)}, (path,))
    assert d == store / "p" and (d / "b.npz").read_bytes() == b"beta"
    assert sorted(q.name for q in store.iterdir()) == ["p"]  # placed whole, nothing left beside it
    bad = dict(entry, sha256="sha256:" + "0" * 64)
    path.write_text(json.dumps(dict(schema=references.SCHEMA, checks={}, products={"p": bad})))
    with pytest.raises(references.ReferenceStoreError, match="sha256 differs"):
        references.product("p", {"_store": str(tmp_path / "s2")}, (path,))
    assert not (tmp_path / "s2" / "p").exists()
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "p.tar.gz").write_bytes(asset.read_bytes())
    asset.unlink()  # the entry's own url is gone: only the mirror has it
    monkeypatch.setenv("LUMEN_ASSETS", mirror.as_uri())
    path.write_text(json.dumps(dict(schema=references.SCHEMA, checks={}, products={"p": entry})))
    d = references.product("p", {"_store": str(tmp_path / "s3")}, (path,))
    assert (d / "a.txt").read_bytes() == b"alpha"


def test_geant4_on_the_overlap_free_export_gates_and_on_the_original_gdml_is_reported(tmp_path):
    """One rule for every detector: g4ref on the export (its producer's ``export`` names the run on the original GDML)
    gates; g4ref on the original GDML is reported; another simulation gates only where its detector has no export
    reference (an original-GDML g4ref run alone does not demote it)."""
    g4 = dict(binary=references.REFERENCE_ENGINE)
    products = {"g4-a": dict(files={}, producer=g4), "g4-a-owned": dict(files={}, producer=dict(g4, export=dict(
                tool="lumen.compare.ownership", sha256="sha256:0", original="g4-a"))),
                "sim-a": dict(files={}, producer=dict(tool="WCSim")), "g4-b": dict(files={}, producer=g4),
                "sim-b": dict(files={}, producer=dict(tool="WCSim"))}  # fmt: skip
    named = (("a_g4", "a", "g4-a-owned"), ("a_g4_original", "a", "g4-a"), ("a", "a", "sim-a"), ("b_g4", "b", "g4-b"),
             ("b", "b", "sim-b"))  # fmt: skip
    entries = [dict(name=n, detector=d, reference=r) for n, d, r in named]
    path = tmp_path / "references.json"
    path.write_text(json.dumps(dict(schema=references.SCHEMA, checks={"charge": entries}, products=products)))
    m = (path,)
    gate = {e["name"]: references.gated("charge", references.spec("charge", e["name"], m), m) for e in entries}
    assert gate == {"a_g4": True, "a_g4_original": False, "a": False, "b_g4": False, "b": True}
    assert references.original("g4-a-owned", m) == "g4-a" and references.original("g4-a", m) is None
    assert references.original("sim-a", m) is None


def test_geant4_against_geant4_and_the_reported_original_rows():
    """G4(export) / G4(original): photons normalised, Poisson errors, chi2 over the sensors with enough hits; the
    original GDML's rows never carry a verdict."""
    a, b = np.array([100.0, 400.0, 2.0]), np.array([50.0, 200.0, 1.0])
    g = charge.geant4_ratio(a, b, 2000, 1000)  # twice the photons, twice the hits
    assert g["ratio"] == pytest.approx(1.0) and g["chi2"] == 0.0 and g["ndf"] == 2 and g["k"] == 2.0
    assert g["err"] == pytest.approx(np.sqrt(1 / 502 + 1 / 251))
    assert charge.geant4_ratio(a * 0.9, b, 1000, 1000)["ratio"] == pytest.approx(1.8)
    out = [dict(src=0, band=0, geant4=charge.geant4_ratio(a, b, 1000, 1000),
                original=dict(n_ref=251.0, mu=255.0, mc_var=4.0))]  # fmt: skip
    rows = charge._original_rows("toy", out, "g4-orig")
    assert [r["quantity"] for r in rows] == ["Geant4 export / original source 0", "Geant4 export / original",
                                             "ratio to the original GDML's reference"]  # fmt: skip
    assert all(r["pass"] is None and r["reference"] == "g4-orig" for r in rows)
    assert rows[1]["value"] == pytest.approx(2.0) and rows[2]["value"] == pytest.approx(255 / 251)


def test_records_are_never_written_into_a_store(manifest, tmp_path):
    store, path = manifest
    rows = [records.row("charge", "toydet", "ratio", 0.999, 0.001, z=-1.0, passed=True, reference="toyref")]
    with pytest.raises(PermissionError, match="reference store"):
        records.write(rows, store / "out", "charge_toydet", (path,))
    path = records.write(rows, tmp_path / "out", "charge_toydet")
    back = records.read([tmp_path / "out"])
    assert path.exists() and set(records.FIELDS) <= set(back[0]) and back[0]["value"] == 0.999
    md = report.markdown(back)
    assert "| toydet | ratio | 0.999 ± 0.001 |" in md and "## Not validated" in md


def test_the_report_prints_every_gated_charge_row(tmp_path):
    names = ("ratio source 0", "sensors agree source 0", "ratio", "truncated weight")
    rows = [records.row("charge", "toydet", q, 0.5, 0.1, passed=True, reference="toyref") for q in names]
    rows.append(records.row("charge", "toydet", "mean time lumen - reference source 0 (ns)", 0.01, 0.02))
    records.write(rows, tmp_path / "out", "charge_toydet")
    md = report.markdown(records.read([tmp_path / "out"]))
    assert "| charge | toydet | 4 of 4 |" in md and "4 of 4 gated rows pass." in md
    assert all(f"| toydet | {r['quantity']} |" in md for r in rows)


def test_the_pass_rules_and_the_statistics():
    # three standard deviations above the null, no floor: Pearson terms (1, 2), or the null of the terms given
    assert stats.chi2_passes(1.04, 11096) and not stats.chi2_passes(1.041, 11096)
    assert stats.chi2_passes(1.7, 30) and not stats.chi2_passes(1.8, 30)
    assert stats.chi2_passes(1.07, 11096, 1.04, 2.2) and not stats.chi2_passes(1.09, 11096, 1.04, 2.2)
    assert stats.t_null(32) == pytest.approx(31 / 29)
    nu = 31
    assert stats.t_var(32) == pytest.approx(2 * nu * nu * (nu - 1) / ((nu - 2) ** 2 * (nu - 4)))


def test_the_per_sensor_terms_with_lumen_s_moment_variance_are_pearson_s():
    """The charge check's per-sensor terms, (mu - n)^2 / ((mu + n) / 2 + var_mc) with var_mc from lumen's photons'
    second moment (here unit-weight photons: sum x^2 = the count), have Pearson's null, 1 and 2 (the moment's own
    noise, over N photons, is negligible), and ``reference_test`` holds its level on such draws."""
    rg = np.random.default_rng(0)
    f, lam, s = 2.0, 50.0, 11_000
    vals, passed = [], []
    for _ in range(40):
        n = rg.poisson(lam, s).astype(float)
        q = rg.poisson(lam * f, s).astype(float)  # f times the reference's photons, unit weights
        mu, v = q / f, q / f**2  # sum x^2 = q: variance n/(n-1) (q - q^2/N) ~ q, in the reference's photons
        a = 0.5 * (mu + n)
        vals.append(np.mean((mu - n) ** 2 / (a + v)))
        passed.append(stats.reference_test(mu, v, q, n, a, n)["passed"])
    assert abs(np.mean(vals) - 1.0) < 3 * np.sqrt(2.0 / s) / np.sqrt(len(vals))
    assert np.std(vals) == pytest.approx(np.sqrt(2.0 / s), rel=0.4)
    assert sum(passed) >= 38  # level 0.0027 per draw


def test_the_node_split_and_richardson():
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
    _painted = SurfaceType("painted", ("R", "sigma_alpha"), (), model=G4_REFLECT)
    rows = (lookup.Row(_painted, "surface:A"), lookup.Row(_painted, "surface:B"))


class _Params(dict):
    """The keys families() reads: values, and each key's gradient method (RINDEX pending)."""

    def info(self, key):
        pending = key.endswith("RINDEX")
        return types.SimpleNamespace(
            available=not pending, method="moving_paths+reweight" if pending else "reweight", range=(-np.inf, np.inf)
        )

    def set(self, key, value):
        return _Params(self, **{key: value})


def test_the_gradient_families_and_their_values():
    from lumen.format.calibrations import Table

    src = lambda k, c="value": types.SimpleNamespace(key=k, column=c)  # noqa: E731
    rows = {("surfaces", "A", "R"): src("surfaces/A/REFLECTIVITY"),
            ("surfaces", "B", "R"): src("surfaces/B/REFLECTIVITY"),
            ("surfaces", "A", "sigma_alpha"): src("surfaces/A/SIGMAALPHA"),
            ("media", "Water", "absorption"): src("materials/Water/ABSLENGTH"),
            ("media", "Water", "index"): src("materials/Water/RINDEX"),
            ("media", "Water", "energy"): src("materials/Water/ABSLENGTH", "energy")}  # fmt: skip
    det = types.SimpleNamespace(parts=types.SimpleNamespace(optics=_Optics(), key_rows=rows))
    p = _Params({"surfaces/A/REFLECTIVITY": np.float32(0.3), "surfaces/B/REFLECTIVITY": np.float32(1.0),
                 "surfaces/A/SIGMAALPHA": np.float32(0.1),
                 "materials/Water/ABSLENGTH": Table(np.ones(2), np.array([20.0, 30.0], np.float32)),
                 "materials/Water/RINDEX": Table(np.ones(2), np.array([1.33, 1.34], np.float32))})  # fmt: skip
    fams = gradients.families(det, p)
    want = {"surfaces/A/REFLECTIVITY": "add", "surfaces/B/REFLECTIVITY": "edge", "surfaces/A/SIGMAALPHA": "mul",
            "materials/Water/ABSLENGTH": "mul"}  # fmt: skip
    assert {f[0]: f[1] for f in fams} == want
    a = next(f for f in fams if f[0] == "surfaces/A/REFLECTIVITY")
    assert float(gradients.at(p, a, 0.1)["surfaces/A/REFLECTIVITY"]) == pytest.approx(0.3 + 0.1 * 0.3)
    b = next(f for f in fams if f[0] == "surfaces/B/REFLECTIVITY")
    assert float(gradients.at(p, b, 0.1)["surfaces/B/REFLECTIVITY"]) == pytest.approx(0.9)  # inward from the edge


class _Runs:
    """A fake check: S sensors in G groups, N photons; AD (per photon) and values from functions of the seed."""

    def __init__(self, ad_tiny=0.0, production=8, k=4, smooth=True, wrong_ad=False):
        self.o, self.p0 = dict(seeds=k, groups=2, photons=50, production=production), None
        self.ad_tiny, self.smooth, self.seen = ad_tiny, smooth, {"tiny": set(), "ladder": 0, "value": set()}
        self.wrong = wrong_ad  # the ladder's slope 10 % off AD: a realisation the FD cannot follow to its derivative

    def _pp(self, seed, shift, tiny=False):
        c = np.random.default_rng(seed).random((50, 2, 2)) + shift
        return c + 1e-3 * (np.random.default_rng(3000 + seed).random((50, 2, 2)) - 0.5) if tiny else c

    def jvp(self, fp, seed, eps=None, per_photon=False):
        if eps is not None:
            self.seen["tiny"].add(seed)
        c = self._pp(seed, self.ad_tiny if eps is not None else 0.0, tiny=eps is not None)
        s = np.concatenate([c.sum(0).T.repeat(2, 1) / 2], 0)  # (2, S = 4): each group's light split over 2 sensors
        return np.full((2, 4), 10.0), s, c

    def value(self, p, seed, reference=None, eps=None, per_photon=False):
        if per_photon:  # the value test's runs (the ladder's are not per photon)
            self.seen["value"].add(eps)
        c = np.random.default_rng(1000 + seed).random((50, 2, 2)) + 1.0
        if reference is None:  # its own values: the same seed's light, another realisation of the last per mille
            c = c + 1e-3 * (np.random.default_rng(2000 + seed).random((50, 2, 2)) - 0.5)
        v = c.sum(0).T.repeat(2, 1) / 2
        return (v, c) if per_photon else v

    def ladder(self, f, one_sided):
        self.seen["ladder"] += 1
        ad = lambda s: np.random.default_rng(s).random((50, 2, 2)).sum(0).T.repeat(2, 1) / 2  # noqa: E731
        hs = [gradients.H / 2**k for k in range(4)]
        seed = self.seen["ladder"] - 1
        dev = (lambda h: 0.0) if self.smooth else (lambda h: 50.0 * np.sqrt(h))
        m = 1.1 if self.wrong else 1.0
        return None, {x: 10.0 + m * x * ad(seed) + np.sign(x) * dev(abs(x)) for x in hs + [-h for h in hs]}


def test_the_gradient_family_tests(monkeypatch):
    """The three tests from the recorded statistics: a clean family passes; the roulette moving the derivative fails the
    production test; a realisation not smooth at the ladder's steps is not tested for exactness (its order reported)
    and is gated on the other two; the production test runs on ``production`` seeds and the ladder on K; the value
    test's two sides run at the TINY roulette."""
    monkeypatch.setattr(gradients, "at", lambda p, fam, t: t)
    clean = _Runs()
    r = gradients.family(clean, ("k", "mul", None, True))
    assert r["passed"] is True and r["exact"]["applies"] and r["exact"]["ratio"] <= 1.0, r["note"]
    assert clean.seen["tiny"] == set(range(8)) and clean.seen["ladder"] == 4
    assert clean.seen["value"] == {gradients.TINY}  # both sides at the TINY roulette: they share every decision
    assert len(r["seeds"]["ad"]) == 8 and len(r["seeds"]["v_ref"]) == 4  # the per-seed statistics are kept
    off = gradients.family(_Runs(ad_tiny=0.05), ("k", "mul", None, True))
    assert off["passed"] is False and "production off" in off["note"]
    edge = gradients.family(_Runs(wrong_ad=True), ("k", "mul", None, False))  # moves a support edge (structure)
    assert edge["passed"] is True and "moves a support edge" in edge["note"] and edge["exact"]["ratio"] > 1.0
    rough = gradients.family(_Runs(smooth=False), ("k", "mul", None, False))
    assert not rough["exact"]["applies"] and "exactness reported" in rough["note"] and rough["passed"] is True
    assert gradients.DEFAULTS["production"] == 64


def test_a_sharded_gradient_check_merges_into_one_summary(tmp_path):
    """Each shard holds families j % n == i; the merge needs every family exactly once, from one sharding."""
    from lumen.compare import gradients

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
    shard = records.read([records.write(a, tmp_path, "s0")])
    shard[0]["gpu"] = "a GPU"  # as a shard run on one stamps it
    out = records.read([records.write(gradients.merge("toy", [shard, b]), tmp_path, "merged")])
    assert out[0]["gpu"] == "a GPU" and out[-1]["gpu"] == records.environment()["gpu"]  # the merge keeps each row's


def test_a_manifest_s_reverse_flag_is_honoured(monkeypatch):
    """The reverse row runs when the manifest says ``"reverse": true``; the command's flag only overrides it."""
    from lumen.compare import cli

    monkeypatch.setattr(gradients.lumen, "load", lambda path, **kw: None)
    monkeypatch.setattr(gradients, "_Runs", lambda det, spec, o: types.SimpleNamespace(det=det, p0=None))
    monkeypatch.setattr(gradients, "families", lambda det, p: [])
    monkeypatch.setattr(gradients, "_reverse", lambda name, runs, fams: records.row("gradients", name, "reverse", 0.0))

    def ran(manifest, flag):
        monkeypatch.setattr(references, "spec", lambda check, name, extra: manifest)
        return any(r["quantity"] == "reverse" for r in gradients.check("toy", detector="x", reverse=flag))

    assert ran({"reverse": True}, None) and not ran({}, None)
    assert not ran({"reverse": True}, False) and ran({}, True)
    flags = ([], ["--reverse"], ["--no-reverse"])
    assert [cli._parser().parse_args(["gradients", *f]).reverse for f in flags] == [None, True, False]


def test_the_mc_gate_compares_bit_for_bit():
    """lumen.compare.mc: identical records pass; one cell's last bit, or another key, fails and is counted."""
    from lumen.compare import mc

    rng = np.random.default_rng(0)
    ref = dict(keys=np.array(["a", "b"]), names=np.array(["s0", "s1", "s2"]), primal=rng.random((4, 3)),
               tangent=rng.random((2, 4, 3)))  # fmt: skip
    same, lines = mc.compare(ref, {k: np.copy(v) for k, v in ref.items()})
    assert same and len(lines) == 3 and all("bitwise identical" in x for x in lines)
    new = {k: np.copy(v) for k, v in ref.items()}
    new["tangent"][1, 2, 0] = np.nextafter(new["tangent"][1, 2, 0], 2.0)
    same, lines = mc.compare(ref, new)
    assert not same and "DIFFERS (1 of 12 cells)" in lines[2] and "bitwise identical" in lines[1]
    assert mc.compare(ref, dict(new, keys=np.array(["a", "c"])))[0] is False


def test_the_one_compare_command_runs_the_mc_records_and_reports_the_environment(tmp_path):
    """``python -m lumen.compare mc compare`` (the one command) exits 0 on identical records and names the environment
    entries that differ first; a run's environment records jax, jaxlib, tallymap, CUDA, NVRTC and the device."""
    import json

    from lumen.compare import cli, mc, records

    env = records.environment()
    assert {"lumen", "tallymap", "jax", "jaxlib", "cuda", "nvrtc"} <= set(env["commits"]) and env["gpu"]
    rng = np.random.default_rng(1)
    ref = dict(keys=np.array(["a"]), names=np.array(["s0", "s1"]), primal=rng.random((2, 2)),
               tangent=rng.random((1, 2, 2)), environment=json.dumps(env))  # fmt: skip
    other = dict(ref, environment=json.dumps({**env, "commits": {**env["commits"], "jax": "0.0.1"}}))
    np.savez(tmp_path / "a.npz", **ref)
    np.savez(tmp_path / "b.npz", **other)
    assert cli.main(["mc", "compare", str(tmp_path / "a.npz"), str(tmp_path / "b.npz")]) == 0
    assert mc.environment_changes(dict(np.load(tmp_path / "a.npz")), dict(np.load(tmp_path / "b.npz"))) == [
        f"environment jax: {env['commits']['jax']} -> 0.0.1"
    ]


def test_the_mc_agreement_is_the_one_test():
    """lumen.compare.mc.agree: two records of the same expectation on other seeds agree; a shifted family does not."""
    from lumen.compare import mc

    def rec(seed0, shift=0.0):
        r = np.random.default_rng(seed0)
        return dict(keys=np.array(["a"]), names=np.array([f"s{i}" for i in range(40)]), seed0=seed0,
                    primal=r.poisson(50.0, (16, 40)).astype(float),
                    tangent=(r.normal(1.0, 0.3, (1, 16, 40)) + shift))  # fmt: skip

    ok, lines = mc.agree(rec(0), rec(16))
    assert ok and all("agrees" in x and "independent" in x for x in lines), lines
    ok, lines = mc.agree(rec(0), rec(16, shift=0.3))
    assert not ok and "DIFFERS" in lines[1] and "agrees" in lines[0]
    r = rec(0)  # paired: a deterministic offset within float32 rounding is no failure, one past it is
    near = dict(r, tangent=r["tangent"] * (1.0 + 2e-7))
    assert mc.agree(r, near)[0]
    assert not mc.agree(r, dict(r, tangent=r["tangent"] * (1.0 + 1e-4)))[0]


def _coupled(rng, k, n, delta, eps=0.01, w=0.005):
    """Return per-seed outputs (k, 1) of two sides sharing every uniform, and each seed's photons (n, 1, 2) per side.

    A photon at reference weight w plays the roulette at eps; on one side its weight is w (1 + delta) (the live values
    reweighted, the ratio carried by a survivor), on the other the roulette reads w (1 + delta) itself: an unbiased
    pair whose difference is the survivors' offset and its compensation, the rare photon kept on one side only.
    """
    a, b, ca, cb = [], [], [], []
    for _ in range(k):
        u = rng.random(n)
        pa = np.where(u < w / eps, eps * (1 + delta), 0.0)  # kept at wr / eps, then carries eps w / wr
        pb = np.where(u < w * (1 + delta) / eps, eps, 0.0)  # kept at its own weight / eps, then carries eps
        a.append([pa.sum()]), b.append([pb.sum()])
        ca.append(np.stack([pa, pa], -1)[:, None]), cb.append(np.stack([pb, pb], -1)[:, None])
    return np.array(a), np.array(b), ca, cb


def test_the_value_test_floor_is_the_unsampled_flips_variance():
    """Two sides on the same uniforms, unbiased, whose decisions differ rarely (0.01 per seed): on 16 seeds the paired
    spread shows the survivors' offset without its compensation and the test rejects nearly always; with the floor
    from the offsets (``gradients.disagreement``) it rejects at about its level, and still finds a bias of a few
    floors."""
    rng = np.random.default_rng(12)
    level, runs, k, n, delta = 0.05, 300, 16, 2000, 1e-5
    rej = {"none": 0, "floor": 0, "bias": 0}
    for _ in range(runs):
        a, b, ca, cb = _coupled(rng, k, n, delta)
        floor = np.sqrt(sum(gradients.disagreement(x, y) for x, y in zip(ca, cb, strict=True))[0, :1]) / k
        rej["none"] += stats.mean_test(a, b, paired=True, level=level)["passed"] is False
        rej["floor"] += stats.mean_test(a, b, paired=True, level=level, floor=floor)["passed"] is False
        rej["bias"] += stats.mean_test(a + 5 * floor, b, paired=True, level=level, floor=floor)["passed"] is False
    sd = np.sqrt(level * (1 - level) / runs)
    assert rej["none"] / runs > 0.5, rej
    assert rej["floor"] / runs <= level + 3 * sd, rej
    assert rej["bias"] / runs >= 0.8, rej


def _no_power(*args, **kw):
    """A statistical test whose outputs all lack power (``stats.mean_test``'s result then)."""
    return dict(z=np.zeros(6), df=np.zeros(6), power=np.zeros(6, bool), stat=0.0, p=1.0, n=0, level=1e-3, passed=None)


def test_a_gate_never_passes_on_nothing():
    """The one rule (``stats.verdict``, ``stats.gate``): a failure fails; no result passing, an empty set included, is
    no verdict and fails a gate; a summary with no family gated fails."""
    assert stats.verdict([]) is None and stats.gate([]) is False
    assert stats.verdict([None, None]) is None and stats.gate([None]) is False
    assert stats.verdict([True, None]) is True and stats.gate([True, None]) is True
    assert stats.verdict([True, False, None]) is False and stats.gate([False]) is False
    reported = [records.row("gradients", "toy", "dQ a", 0.0, passed=None)]
    assert gradients.summary("toy", reported, 1, 0.0, {})["pass"] is False
    assert gradients.summary("toy", [], 0, 0.0, {})["pass"] is False


def test_no_power_is_reported_never_agreement(monkeypatch):
    """Records with too few seeds for any difference to show have no verdict, and the gate does not pass on them; a
    gradient family whose statistical tests lack power keeps a failed exactness test (deterministic: it needs no
    power), passes on a passed one, and is reported with neither."""
    from lumen.compare import mc

    def rec(seed0):
        r = np.random.default_rng(seed0)
        return dict(keys=np.array(["a"]), names=np.array(["s0", "s1"]), seed0=seed0,
                    primal=r.poisson(50.0, (3, 2)).astype(float), tangent=r.normal(1.0, 0.3, (1, 3, 2)))  # fmt: skip

    ok, lines = mc.agree(rec(0), rec(3))  # 3 seeds: below ln(tests / ALPHA)
    assert ok is False and all("no power" in x for x in lines), lines
    monkeypatch.setattr(stats, "mean_test", _no_power)
    monkeypatch.setattr(gradients, "at", lambda p, fam, t: t)
    wrong = gradients.family(_Runs(wrong_ad=True), ("k", "mul", None, True))
    assert wrong["passed"] is False and "not exact" in wrong["note"] and "without power" in wrong["note"]
    assert gradients.family(_Runs(), ("k", "mul", None, True))["passed"] is True  # exact, and nothing failed
    rough = gradients.family(_Runs(smooth=False), ("k", "mul", None, False))
    assert rough["passed"] is None and "no test has power" in rough["note"]


def test_a_failing_gradient_row_names_its_test_and_statistic(monkeypatch):
    """The note of a failed statistical test is what was measured (the test, its score and p against its level), not
    an explanation."""
    monkeypatch.setattr(gradients, "at", lambda p, fam, t: t)
    off = gradients.family(_Runs(ad_tiny=0.05), ("k", "mul", None, True))
    assert off["passed"] is False and "production off (z " in off["note"] and "roulette" not in off["note"]


def test_a_charge_row_where_lumen_detects_nothing_fails():
    """lumen at 0 against a reference count has an error (the two samples' mean as Poisson variance) and a score,
    never z = 0; a reference count too small to reject at the level is reported (no power)."""
    t = stats.ratio_test(0.0, 0.0, 50.0)
    assert t["passed"] is False and t["err"] > 0 and t["z"] == pytest.approx(-10.0) and t["ratio"] == 0.0
    assert stats.ratio_test(0.0, 0.0, 3.0)["passed"] is None and stats.ratio_test(52.0, 4.0, 50.0)["passed"] is True
    s, nt = 2, 10  # sensors, time nodes
    r = dict(charge=np.zeros((s, 1)), charge_var=np.zeros((s, 1)), Q2=np.zeros((s, 1)), total_var=np.zeros(1),
             photons=1000, H=np.zeros((1, 1, nt)))  # fmt: skip
    ctx = dict(can=np.arange(s), types=np.array(["pmt", "pmt"]), dist=np.array([1.0, 2.0]), launched=500.0,
               distance_groups=1, p_group=np.ones(1), t_ref=np.full(60, 3.0), dt=1.0, tests=1)  # fmt: skip
    row = charge.compare_band(r, 0, np.array([30.0, 30.0]), np.ones(s), ctx)
    assert row["mu"] == 0.0 and row["ratio_passed"] is False and row["z"] < -3.0
    health = dict(truncated=0.0, lost=0.0, abandoned=0.0, escaped=0.0, photons=1000, launched=500.0)
    rows = charge._rows("toy", {"reference": "ref"}, [dict(row, src=0, band=0)], health, {"gated": True})
    assert [x["pass"] for x in rows if x["quantity"].startswith("ratio")] == [False, False]
    # a navigation failure gates on every detector: lost or abandoned weight that could move the total beyond its
    # Monte Carlo error (here 0) fails the check
    assert {x["quantity"]: x["pass"] for x in rows}["lost and abandoned weight"] is True
    rows = charge._rows(
        "toy", {"reference": "ref"}, [dict(row, src=0, band=0)], dict(health, lost=2.0), {"gated": True}
    )
    assert {x["quantity"]: x for x in rows}["lost and abandoned weight"]["pass"] is False


def test_gpu_sh_refuses_a_cpu_fallback(tmp_path, monkeypatch):
    """tools/ci/gpu.sh stops before the physics tier when JAX does not run on a GPU (the tier would skip itself)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "tools" / "ci" / "gpu.sh"
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("OUT", str(tmp_path))
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent), prepend=os.pathsep)  # this python first
    p = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=300)
    assert p.returncode != 0 and "JAX runs on cpu, not a GPU" in p.stderr, (p.returncode, p.stdout, p.stderr)
    assert "passed" not in p.stdout
