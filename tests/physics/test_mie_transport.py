"""Mie scattering through the photon step (labels_and_mie.md M-T6, M-T7, M-T8): the physics tier, on a GPU.

- M-T6, the isotropic-source sphere: a point source at the centre of a sphere of radius R = 1 m in a Mie-only medium
  (MIEHG 20 m, no absorption), the wall a detecting counting sensor. (i) The charge is N (each photon's expected
  deposits sum to 1 in expectation: checked against the replicas' error); (ii) the direct light, the first step's
  deposits, is N exp(-R / L) to float32 accuracy (each photon deposits exp(-mu R) there); (iii) the single-scatter
  light (the second step's deposits) per path-length bin on (R, 3R) against the float64 quadrature of
  f1(l) = int da / L e^(-a / L) int dmu p(mu) e^(-b / L) delta(l - a - b), b = -a mu + sqrt(R^2 - a^2 (1 - mu^2)).
- M-T7, its derivatives: d/dL of the direct light against N R / L^2 e^(-R / L) (exact), d/d(g_f, g_b, r) of the
  single-scatter bins against the quadrature's derivatives (paired z), forward against reverse mode, and the forward
  ratio at its edge r = 1 (both lobes defined: WITHIN), with the edge rule's moved reference.
- M-T8, the g4ref ring toy: a pencil beam at the centre of a sphere cut into 36 polar rings of 5 degrees, one counting
  sensor per ring; the per-ring charge and mean time against Geant4 10.7.4 and 11.2.2 (tests/data/g4ref_mie, reduced
  from g4ref's one-shell runs by tools/port/mie_rings_reduce.py), cases A (0.6, 0.3, 0.7) and B (0, 0.8, 0.5).
"""

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import lumen
from lumen import run as runner
from lumen._core.safe import literal
from lumen.format.sensorsets import SensorGroup
from lumen.geometry import tracked
from lumen.recipe import Recipe
from lumen.transport import estimator, item, state
from lumen.transport import step as steps
from lumen.units import C_VACUUM, HC

pytestmark = pytest.mark.skipif(jax.default_backend() != "gpu", reason="the physics tier runs on a GPU")
DATA = Path(__file__).resolve().parents[1] / "data" / "g4ref_mie"
R, L_MIE, N_IDX, E_EV, R_OUT = 1.0, 20.0, 1.33, 3.0, 1.01
LAM = HC / E_EV
CASES = {"A": (0.6, 0.3, 0.7), "B": (0.0, 0.8, 0.5), "water": (0.95, 0.3, 1.0)}
N_ISO, N_RING, REPLICAS = 1 << 20, 4_000_000, 16
ELL = np.linspace(R, 3 * R, 21)  # single-scatter path-length bins (m)
Z_MAX, P_MIN = 4.0, 1e-3
Z_T = 5.0  # the largest |t| over the bins with 16 groups (t with 15 degrees of freedom: 3e-3 over 20 bins)

GDML = """<?xml version="1.0" encoding="UTF-8"?>
<gdml>
<define>
  <matrix name="RIDX" coldim="2" values="1.0*eV 1.33 12.0*eV 1.33"/>
  <matrix name="MIEL" coldim="2" values="1.0*eV {L}*m 12.0*eV {L}*m"/>
  <matrix name="MIEF" coldim="1" values="{gf}"/>
  <matrix name="MIEB" coldim="1" values="{gb}"/>
  <matrix name="MIER" coldim="1" values="{r}"/>
</define>
<materials>
  <element name="Hyd" formula="H" Z="1"><atom value="1.008"/></element>
  <material name="MieLike" state="liquid"><D value="1.0"/><fraction n="1.0" ref="Hyd"/>
    <property name="RINDEX" ref="RIDX"/><property name="MIEHG" ref="MIEL"/>
    <property name="MIEHG_FORWARD" ref="MIEF"/><property name="MIEHG_BACKWARD" ref="MIEB"/>
    <property name="MIEHG_FORWARD_RATIO" ref="MIER"/></material>
  <material name="SensLike" state="liquid"><D value="1.0"/><fraction n="1.0" ref="Hyd"/>
    <property name="RINDEX" ref="RIDX"/></material>
</materials>
<solids>
  <orb name="WorldOrb" r="1.2" lunit="m"/>
{solids}
</solids>
<structure>
{volumes}
  <volume name="volWorld"><materialref ref="MieLike"/><solidref ref="WorldOrb"/>
{placements}
  </volume>
</structure>
<setup name="Default" version="1.0"><world ref="volWorld"/></setup>
</gdml>
"""


def write_gdml(path, case, rings):
    gf, gb, r = CASES[case]
    solids, volumes, placements = [], [], []
    for i in range(rings):
        d = 180.0 / rings
        solids.append(f'  <sphere name="S{i}" rmin="{R}" rmax="{R_OUT}" startphi="0" deltaphi="360" '
                      f'starttheta="{i * d}" deltatheta="{d}" aunit="deg" lunit="m"/>')  # fmt: skip
        volumes.append(f'  <volume name="volRing{i}"><materialref ref="SensLike"/><solidref ref="S{i}"/></volume>')
        placements.append(f'    <physvol name="pvRing{i}"><volumeref ref="volRing{i}"/></physvol>')
    text = GDML.format(L=L_MIE, gf=gf, gb=gb, r=r, solids="\n".join(solids), volumes="\n".join(volumes),
                       placements="\n".join(placements))  # fmt: skip
    path.write_text(text)
    return path


@pytest.fixture(scope="module")
def detectors(tmp_path_factory):
    out = {}

    def get(case, rings):
        if (case, rings) not in out:
            d = tmp_path_factory.mktemp(f"mie_{case}_{rings}")
            # one group per ring: adjacent rings of one sensor type would be one region, crossed invisibly
            groups = tuple(SensorGroup(f"ring{i}", "any_entry", "lumen:counting", lv=f"volRing{i}",
                                       settings={"eff": 1.0}) for i in range(rings))  # fmt: skip
            recipe = Recipe(f"mie_{case}", "volWorld", (300.0, 600.0), groups)
            out[(case, rings)] = lumen.load(lumen.convert(write_gdml(d / "mie.gdml", case, rings), recipe,
                                                          d / "mie.lumen"))  # fmt: skip
        return out[(case, rings)]

    return get


def medium_key(params):
    return next(k for k in params["media"] if "MieLike" in k)


def emission(n, seed, iso=True):
    rng = np.random.default_rng(seed)
    if iso:
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
    else:
        d = np.tile([0.0, 0.0, 1.0], (n, 1))
    z = jnp.zeros(n, jnp.float32)
    return state.Emission(jnp.zeros((n, 3), jnp.float32), jnp.asarray(d, jnp.float32), z, z + LAM, z + 1.0,
                          jnp.zeros((n, 2, 3), jnp.float32), jnp.full(n, iso))  # fmt: skip


class SingleScatter:
    """A diagnostic: the first step's deposits (direct light) and the second step's per path-length bin."""

    def outputs(self):
        return {"direct": (1, True), "f1": (len(ELL) - 1, True), "f1sq": (len(ELL) - 1, True)}

    def init(self, ph):
        return jnp.int32(0)

    def update(self, mem, ph, ph2, dep, pid):
        ell = dep.time * C_VACUUM / N_IDX
        b = jnp.searchsorted(literal(ELL), ell, side="right") - 1  # literals: a loop closes over no float array
        ok = (dep.sensor >= 0) & (ph.vertex == 1) & (b >= 0) & (b < len(ELL) - 1)
        cell = jnp.where(ok, b, -1)
        direct = jnp.where((dep.sensor >= 0) & (ph.vertex == 0), 0, -1)
        return mem, [("direct", direct, dep.q), ("f1", cell, dep.q), ("f1sq", cell, dep.q * dep.q)]


MODELS = {}  # one Model per detector and step config, one PhotonLoop per shape: new ones would recompile
SPECS = {}  # one PhotonLoop per shape: a new one would be a new static argument of the loop


SINGLE = SingleScatter()


def simulate(det, params, rows, *, reference=None, cfg=None, replicas=REPLICAS, seed=7):
    cfg = cfg or steps.StepConfig()
    if (id(det), cfg) not in MODELS:
        MODELS[(id(det), cfg)] = steps.Model(tracked.from_parts(det.parts), det.parts.optics, cfg)
    model = MODELS[(id(det), cfg)]
    tally = item.HistogramShape(len(det.parts.optics.sensor_pvs), (300.0, 600.0))
    spec = SPECS.setdefault(tally, item.photons(tally))
    ops = steps.operands(model, params, reference, steps.Controls(10.0, float("inf"), 0.0), item.grid(tally))
    key = jax.random.PRNGKey(seed)
    t = runner.run(spec, model, ops, rows, key, max_vertices=20000, replicas=replicas, diagnostics=(SINGLE,))
    return t.outputs


# ------------------------------------------------------------------------------------------ the float64 quadrature
def hg_mu(mu, g):
    """The Henyey-Greenstein density in mu (integrates to 1 on [-1, 1])."""
    return 0.5 * (1 - g * g) / (1 + g * g - 2 * g * mu) ** 1.5


def phase_mu(mu, gf, gb, r):
    return r * hg_mu(mu, gf) + (1 - r) * hg_mu(mu, -gb)


def hg_cdf(mu, g):
    """The Henyey-Greenstein CDF in mu, F(-1) = 0, F(1) = 1."""
    if g == 0.0:
        return 0.5 * (mu + 1.0)
    return (1 - g * g) / (2 * g) * (1 / np.sqrt(1 + g * g - 2 * g * mu) - 1 / (1 + g))


def hg_cdf_inverse(u, g):
    if g == 0.0:
        return 2.0 * u - 1.0
    x = 2 * g * u / (1 - g * g) + 1 / (1 + g)  # 1 / sqrt(1 + g^2 - 2 g mu)
    return np.clip((1 + g * g - 1 / (x * x)) / (2 * g), -1.0, 1.0)


def f1_bins(gf, gb, r, length=L_MIE, n_a=400, n_u=64):
    """The single-scatter light per path-length bin (per photon), by float64 Gauss-Legendre quadrature.

    For a scatter at a, l = a + b(mu) runs monotonically from R (mu = 1) to R + 2a (mu = -1), with
    mu(l) = (R^2 - a^2 - (l - a)^2) / (2 a (l - a)); each bin is integrated over each lobe's own CDF between its
    edges' mu (a peaked lobe is then smooth), the forward lobe HG(g_f) with weight r, the backward HG(-g_b) 1 - r.
    """
    xa, wa = np.polynomial.legendre.leggauss(n_a)
    a = 0.5 * R * (xa + 1)
    wa = 0.5 * R * wa
    xu, wu = np.polynomial.legendre.leggauss(n_u)
    out = np.zeros(len(ELL) - 1)
    for ai, wai in zip(a, wa, strict=True):
        b_edge = np.clip(ELL - ai, 1e-300, None)
        mu_edge = np.clip((R * R - ai * ai - b_edge * b_edge) / (2 * ai * b_edge), -1.0, 1.0)
        for share, g in ((r, gf), (1 - r, -gb)):
            if share == 0.0:
                continue
            f_hi, f_lo = hg_cdf(mu_edge[:-1], g), hg_cdf(mu_edge[1:], g)
            u = 0.5 * (f_hi - f_lo)[:, None] * xu + 0.5 * (f_hi + f_lo)[:, None]
            mu = hg_cdf_inverse(u, g)
            b = -ai * mu + np.sqrt(R * R - ai * ai * (1 - mu * mu))
            out += share * wai / length * math.exp(-ai / length) * np.sum(0.5 * (f_hi - f_lo)[:, None] * wu
                                                                           * np.exp(-b / length), 1)  # fmt: skip
    return out


# --------------------------------------------------------------------------------------------------- M-T6, M-T7
@pytest.mark.parametrize("case", ["A", "B", "water"])
def test_isotropic_sphere_charge_direct_light_and_single_scatters(detectors, case):
    det = detectors(case, 1)
    o = simulate(det, det.pack(), emission(N_ISO, seed=1))
    q = np.asarray(o["Q"], np.float64).reshape(REPLICAS, -1).sum(1) / (N_ISO / REPLICAS)
    z_q = (q.mean() - 1.0) / (q.std(ddof=1) / math.sqrt(REPLICAS))
    assert abs(z_q) < Z_MAX, (case, q.mean(), z_q)
    direct = float(o["direct"][0]) / N_ISO
    assert direct == pytest.approx(math.exp(-R / L_MIE), rel=2e-5), case
    f1 = np.asarray(o["f1"], np.float64) / N_ISO
    want = f1_bins(*CASES[case])
    sq = np.asarray(o["f1sq"], np.float64)
    err = np.where(sq > 0, np.sqrt(sq) / N_ISO, np.sqrt(want / N_ISO))  # an empty bin: the bound of its error (q <= 1)
    z = (f1 - want) / err
    c2 = float(np.sum(z * z))
    assert stats.chi2.sf(c2, len(z)) > P_MIN and np.max(np.abs(z)) < Z_MAX, (case, c2, z.round(2).tolist())


def _scaled(p, m, which, th):
    lobe = dict(p["media"][m]["mie"]["lobe"])
    mie = dict(p["media"][m]["mie"])
    if which == "length":
        mie["length"] = dict(mie["length"], value=mie["length"]["value"] * (1.0 + th))
    else:
        lobe[which] = lobe[which] + th
        mie["lobe"] = lobe
    return {**p, "media": {**p["media"], m: {**p["media"][m], "mie": mie}}}


def _jvp(det, which, rows, cfg=None, seed=7):
    p0 = jax.tree_util.tree_map(jnp.asarray, det.pack())
    m = medium_key(p0)

    def obs(th):
        o = simulate(det, _scaled(p0, m, which, th), rows, reference=_scaled(p0, m, which, jax.lax.stop_gradient(th)),
                     cfg=cfg, replicas=1, seed=seed)  # fmt: skip
        return o["direct"][0], o["f1"]

    return jax.jit(lambda: jax.jvp(obs, (jnp.float32(0.0),), (jnp.float32(1.0),)))()


def test_the_direct_light_derivative_in_the_mie_length_is_exact(detectors):
    det = detectors("A", 1)
    (_, _), (d_direct, _) = _jvp(det, "length", emission(1 << 18, seed=2))
    n = 1 << 18
    want = n * (R / L_MIE) * math.exp(-R / L_MIE)  # d/dth of exp(-R / (L (1 + th))) at th = 0
    assert float(d_direct) == pytest.approx(want, rel=1e-4)


def _quad_derivative(case, which, h=1e-4):
    gf, gb, r = CASES[case]
    at = dict(g_forward=gf, g_backward=gb, forward_ratio=r)
    if which == "forward_ratio" and r >= 1.0:  # one-sided, inward; f1 is linear in r: exact
        return f1_bins(gf, gb, 1.0) - f1_bins(gf, gb, 0.0)
    lo, hi = dict(at), dict(at)
    lo[which], hi[which] = at[which] - h, at[which] + h
    return (f1_bins(lo["g_forward"], lo["g_backward"], lo["forward_ratio"])
            - f1_bins(hi["g_forward"], hi["g_backward"], hi["forward_ratio"])) / (-2 * h)  # fmt: skip


@pytest.mark.parametrize("which", ["g_forward", "g_backward", "forward_ratio"])
def test_single_scatter_derivatives_in_the_lobe_constants(detectors, which):
    det = detectors("A", 1)
    n, k = 1 << 21, 16
    ad = []
    for s in range(k):  # k independent keys and photons: the AD estimate's error
        rows = emission(n // k, seed=10 + s)
        ad.append(np.asarray(_jvp(det, which, rows, seed=10 + s)[1][1], np.float64) / (n // k))
    ad = np.array(ad)
    want = _quad_derivative("A", which)
    z = (ad.mean(0) - want) / np.maximum(ad.std(0, ddof=1) / math.sqrt(k), 1e-12)
    assert np.max(np.abs(z)) < Z_T, (which, z.round(2).tolist())


def test_the_forward_ratio_at_its_edge_keeps_its_one_sided_derivative(detectors):
    det = detectors("water", 1)  # r = 1 with both lobes defined: WITHIN
    p0 = det.pack()
    m = medium_key(p0)
    rows = estimator.edge_rows(det.parts.optics, p0, [("media", m, "mie", "lobe", "forward_ratio")])
    assert rows == ((m, "mie", "forward_ratio"),)
    cfg = steps.StepConfig(edge_rows=rows)
    n, k = 1 << 21, 16
    runs = [_jvp(det, "forward_ratio", emission(n // k, seed=30 + s), cfg, seed=30 + s)[1][1] for s in range(k)]
    ad = np.array([np.asarray(r, np.float64) / (n // k) for r in runs])
    want = _quad_derivative("water", "forward_ratio")
    z = (ad.mean(0) - want) / np.maximum(ad.std(0, ddof=1) / math.sqrt(k), 1e-12)
    assert np.max(np.abs(z)) < Z_T, z.round(2).tolist()
    # the value with the moved reference is unchanged in expectation
    o = simulate(det, p0, emission(N_ISO, seed=3), cfg=cfg)
    f1 = np.asarray(o["f1"], np.float64) / N_ISO
    want = f1_bins(*CASES["water"])
    sq = np.asarray(o["f1sq"], np.float64)
    zz = (f1 - want) / np.where(sq > 0, np.sqrt(sq) / N_ISO, np.sqrt(want / N_ISO))
    assert stats.chi2.sf(float(np.sum(zz * zz)), len(zz)) > P_MIN


def test_forward_and_reverse_mode_agree(detectors):
    det = detectors("A", 1)
    p0 = jax.tree_util.tree_map(jnp.asarray, det.pack())
    m = medium_key(p0)
    rows = emission(1 << 16, seed=4)
    c = jnp.asarray(np.random.default_rng(0).normal(size=len(ELL) - 1), jnp.float32)

    def scalar(th):
        o = simulate(det, _scaled(p0, m, "g_forward", th), rows,
                     reference=_scaled(p0, m, "g_forward", jax.lax.stop_gradient(th)), replicas=1)  # fmt: skip
        return jnp.sum(c * o["f1"])

    fwd = jax.jit(lambda: jax.jvp(scalar, (jnp.float32(0.0),), (jnp.float32(1.0),))[1])()
    rev = jax.jit(jax.grad(scalar))(jnp.float32(0.0))
    assert float(rev) == pytest.approx(float(fwd), rel=1e-4, abs=1e-3)


# ----------------------------------------------------------------------------------------------------------- M-T8
@pytest.mark.parametrize("geant4", ["10.7.4", "11.2.2"])
@pytest.mark.parametrize("case", ["A", "B"])
def test_ring_toy_against_geant4(detectors, case, geant4):
    man = json.loads((DATA / "rings.json").read_text())
    name = f"g4_{geant4}_mie{case}_rings.npz"
    with np.load(DATA / name) as g:
        counts, t_sum, t2_sum, launched = g["counts"], g["t_sum"], g["t2_sum"], int(g["launched"])
    assert man[name]["hits"] == int(counts.sum())
    det = detectors(case, 36)
    o = simulate(det, det.pack(), emission(N_RING, seed=5, iso=False))
    ring = np.array([int(pv.rsplit("pvRing", 1)[1].split("/")[0]) for pv in det.parts.optics.sensor_pvs])
    q = np.zeros((REPLICAS, 36))
    tw = np.zeros((REPLICAS, 36))
    q[:, ring] = np.asarray(o["Q"], np.float64).reshape(REPLICAS, -1)
    tw[:, ring] = np.asarray(o["Tw"], np.float64).reshape(REPLICAS, -1)
    per = N_RING / REPLICAS
    ours, ours_err = q.mean(0) / per, q.std(0, ddof=1) / per / math.sqrt(REPLICAS)
    ref, ref_err = counts / launched, np.sqrt(counts) / launched
    z = (ours - ref) / np.sqrt(ours_err**2 + ref_err**2)
    c2 = float(np.sum(z * z))
    assert stats.chi2.sf(c2, 36) > P_MIN and np.max(np.abs(z)) < Z_MAX, (case, geant4, z.round(2).tolist())
    # the mean arrival time per ring, where both have light
    use = counts > 1000
    t_ref = t_sum[use] / counts[use]
    t_ref_err = np.sqrt(np.maximum(t2_sum[use] / counts[use] - t_ref**2, 0.0) / counts[use])
    t_ours = tw.sum(0)[use] / q.sum(0)[use]
    t_rep = tw[:, use] / np.maximum(q[:, use], 1e-30)
    t_err = t_rep.std(0, ddof=1) / math.sqrt(REPLICAS)
    zt = (t_ours - t_ref) / np.sqrt(t_err**2 + t_ref_err**2)
    assert np.max(np.abs(zt)) < Z_MAX, (case, geant4, zt.round(2).tolist())
