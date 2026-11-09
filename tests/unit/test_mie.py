"""Mie scattering, Geant4's G4OpMieHG: the lobe, its density, the channel and the reading of the material tables.

The gate (labels_and_mie.md section 2.7, M-T1 to M-T3 and M-T5, plus the single-scatter distributions of the g4ref toy):

- the sampler is Geant4's inverse CDF, uniform for uniform, against a float64 literal port of PostStepDoIt;
- the density is the two-lobe mixture, normalised, and the density of the sampler (chi2 in bins of exact mass);
- the score identity per parameter, finite derivatives in forward and reverse mode;
- the Henyey-Greenstein moments <P_l(mu)> = r g_f^l + (1 - r) (-g_b)^l, l = 1..4;
- the single-scatter angle distribution of Geant4 10.7.4 and 11.2.2 (tests/data/g4ref_mie);
- the reading of MIEHG and its three constants: refusals, inert constants, the fixed ratio, the detector file round
  trip.
"""

import dataclasses
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from lobe_contract import check_lobe
from optics_toy import read_toy
from scipy import stats
from test_importer import GDML, write

import lumen
from lumen._core import rng
from lumen.build import spec
from lumen.format import schema
from lumen.gdml import exporter
from lumen.media import bulk, phase
from lumen.rules import optics as rules

F32 = jnp.float32
DATA = Path(__file__).resolve().parents[1] / "data" / "g4ref_mie"
Z_MAX = 4.5
P_MIN = 1e-3  # the smallest chi2 p-value a distribution test accepts
G = (0.0, 0.3, 0.9, 0.99, 0.999)
CASES = [(gf, gb, r) for gf in G for gb in G for r in (0.0, 0.3, 1.0)]
CASES += [(-0.5, 0.4, 0.6), (0.7, -0.6, 0.2), (-0.999, -0.999, 0.5), (0.6, 0.3, 0.7), (0.0, 0.8, 0.5)]
WI = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])


def params(gf, gb, r):
    return {"g_forward": F32(gf), "g_backward": F32(gb), "forward_ratio": F32(r)}


def uniforms(n, seed=0):
    keys = jax.vmap(lambda i: rng.photon_key(jax.random.PRNGKey(seed), i))(jnp.arange(n))
    return jax.vmap(lambda k: rng.uniforms(k, 3))(keys)


@jax.jit
def _sample(p, wi, u):
    return jax.vmap(lambda uu: phase.MIE_HG.sample(p, wi, wi, uu))(u)


def sample_mu(case, n, seed=0, wi=WI):
    """Return (directions, mu) of n draws at a case, float64."""
    wi = jnp.asarray(wi, F32)
    wo = np.asarray(_sample(params(*case), wi, uniforms(n, seed)), np.float64)
    return wo, wo @ np.asarray(wi, np.float64)


def angle(wo, wi):
    """The angle between unit vectors (float64), accurate at both poles: 2 atan2(|wo - wi|, |wo + wi|)."""
    wo, wi = np.asarray(wo, np.float64), np.asarray(wi, np.float64)
    return 2.0 * np.arctan2(np.linalg.norm(wo - wi, axis=-1), np.linalg.norm(wo + wi, axis=-1))


# ----------------------------------------------------------------------------------- float64 references
def theta_grid(n=6000):
    """Scattering angles on [0, pi], dense at both poles, where a lobe of width 1 - g = 1e-3 peaks."""
    t = np.geomspace(1e-8, 0.5 * np.pi, n)
    return np.unique(np.concatenate([[0.0], t, np.pi - t, [np.pi]]))


def hg_q(theta, g):
    """1 + g^2 - 2 g cos(theta) (float64) from the half angle, positive terms only for either sign of g."""
    s2, c2 = np.sin(0.5 * theta) ** 2, np.cos(0.5 * theta) ** 2
    return (1.0 - g) ** 2 + 4.0 * g * s2 if g >= 0.0 else (1.0 + g) ** 2 - 4.0 * g * c2


def mie_density(theta, gf, gb, r):
    """Geant4's mixture r HG(g_f) + (1 - r) HG(-g_b) per steradian at the scattering angle theta (float64)."""
    return sum(w * (1.0 - g * g) / (4.0 * np.pi * hg_q(theta, g) ** 1.5) for w, g in ((r, gf), (1.0 - r, -gb)))


def hg_angle_cdf(theta, g):
    """P(angle <= theta) of HG(g) (float64, exact; absolute accuracy ~1e-16)."""
    if g == 0.0:
        return np.sin(0.5 * theta) ** 2
    return (1.0 - g * g) / (2.0 * g) * (1.0 / (1.0 - g) - 1.0 / np.sqrt(hg_q(theta, g)))


def mie_angle_cdf(theta, gf, gb, r):
    """P(angle <= theta) of Geant4's mixture (float64)."""
    return r * hg_angle_cdf(theta, gf) + (1.0 - r) * hg_angle_cdf(theta, -gb)


def g4_mie_hg(u0, u1, gf, gb, r):
    """G4OpMieHG::PostStepDoIt (10.7.4 = 11.2.2), literally, in float64: (forward lobe?, theta) per uniform pair."""
    forward = u0 <= r
    g = np.where(forward, gf, gb)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = 2.0 * u1 * (1.0 + g) ** 2 * (1.0 - g + g * u1) / (1.0 - g + 2.0 * g * u1) ** 2 - 1.0
    theta = np.where(g != 0.0, np.arccos(np.clip(x, -1.0, 1.0)), np.arccos(2.0 * u1 - 1.0))
    return forward, np.where(forward, theta, np.pi - theta)


# ------------------------------------------------------------------------------------------- the lobe
def test_the_sampler_is_geant4s_inverse_cdf_uniform_for_uniform():
    """the same uniforms give Geant4's lobe and its angle: 1 -/+ cos(theta) to float32 rounding, never cancelling."""
    u = np.asarray(uniforms(20_000, seed=3), np.float64)
    for case in CASES:
        gf, gb, r = (float(np.float32(x)) for x in case)  # the float32 values lumen holds
        fwd_g4, theta = g4_mie_hg(u[:, 0], u[:, 1], gf, gb, r)
        fwd = np.asarray(u[:, 0] <= np.float32(r))
        assert np.array_equal(fwd, fwd_g4), (gf, gb, r)
        g = np.where(fwd, np.float32(gf), np.float32(gb)).astype(np.float32)
        cm, cp = (np.asarray(c, np.float64) for c in phase.mie_hg_cos(jnp.asarray(g), jnp.asarray(u[:, 1], F32)))
        cm, cp = np.where(fwd, cm, cp), np.where(fwd, cp, cm)
        # 1 -/+ cos(theta) of Geant4's float64 angle; atol: its own cancellation in acos(x) at x = -/+1
        assert np.allclose(cm, 2.0 * np.sin(theta / 2) ** 2, rtol=3e-5, atol=1e-15), (gf, gb, r)
        assert np.allclose(cp, 2.0 * np.cos(theta / 2) ** 2, rtol=3e-5, atol=1e-15), (gf, gb, r)
    # g = 0 is exactly 1 - mu = 2 (1 - u), with no branch
    cm, _ = phase.mie_hg_cos(F32(0.0), jnp.asarray(u[:100, 1], F32))
    np.testing.assert_array_equal(np.asarray(cm), np.asarray(2.0 * (1.0 - jnp.asarray(u[:100, 1], F32))))


def test_the_density_is_the_two_lobe_mixture_and_normalised():
    """logpdf = log(r HG(g_f) + (1 - r) HG(-g_b)) per sr (float64) to 2e-4 at both poles, and integrates to 1."""
    th = theta_grid()
    e1 = np.cross(WI, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    wi32 = np.asarray(WI, np.float32)
    wo32 = (np.cos(th)[:, None] * WI + np.sin(th)[:, None] * e1).astype(np.float32)
    th32 = angle(wo32, wi32)  # the angle of the float32 vectors the density sees
    lp = jax.jit(jax.vmap(lambda p, w: phase.MIE_HG.logpdf(p, jnp.asarray(wi32), None, w), (None, 0)))
    for gf, gb, r in CASES:
        v = np.asarray(lp(params(gf, gb, r), jnp.asarray(wo32)), np.float64)
        truth = mie_density(th32, *(float(np.float32(x)) for x in (gf, gb, r)))
        assert np.allclose(np.exp(v), truth, rtol=2e-4), (gf, gb, r, np.max(np.abs(np.exp(v) / truth - 1)))
        f = np.exp(v) * np.sin(th)
        integral = np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(th)) * 2 * np.pi
        assert abs(integral - 1.0) < 2e-4, (gf, gb, r, integral)


@pytest.mark.parametrize("case", [(0.2, 0.0, 0.5), (0.6, 0.3, 0.7), (0.9, 0.3, 1.0), (0.3, 0.9, 0.0), (-0.5, 0.4, 0.6)])
def test_the_lobe_contract_on_the_direction_grid(case):
    """the repository's lobe contract (unit vectors, normalisation, chi2, score identity, finite derivatives)."""
    check_lobe(phase.MIE_HG, params(*case), WI, WI, label=str(case))


def test_the_sampler_follows_the_density_in_bins_of_exact_mass():
    """chi2 of the drawn scattering angle in 40 equal-mass bins of the exact CDF, every case, both poles included.

    In the angle, not mu: a lobe 1e-3 rad wide at a pole is 5e-7 wide in mu, the float32 resolution of mu there.
    """
    n = 200_000
    grid = theta_grid(20_000)
    for case in CASES:
        wo, _ = sample_mu(case, n, seed=11)
        th = angle(wo, np.asarray(WI, np.float32))
        c32 = tuple(float(np.float32(x)) for x in case)
        edges = np.interp(np.linspace(0, 1, 41), mie_angle_cdf(grid, *c32), grid)
        edges[0], edges[-1] = 0.0, np.pi
        obs, _ = np.histogram(th, edges)
        exp = np.diff(mie_angle_cdf(edges, *c32)) * n
        ok = exp > 20
        c2 = float(np.sum((obs[ok] - exp[ok]) ** 2 / exp[ok]))
        assert stats.chi2.sf(c2, int(ok.sum()) - 1) > P_MIN, (case, c2, int(ok.sum()))


def test_the_score_identity_and_finite_derivatives_forward_and_reverse():
    """E_ref[d logpdf / d theta] = 0 per parameter; jvp and grad finite in the parameters, wi and wo, at the edges."""
    n = 100_000
    wi = jnp.asarray(WI, F32)

    def lp(p, a, w):
        return phase.MIE_HG.logpdf(p, a, None, w)

    grad = jax.jit(jax.vmap(jax.grad(lp, argnums=(0, 1, 2)), (None, None, 0)))
    for case in CASES:
        p = params(*case)
        wo = jnp.asarray(sample_mu(case, n, seed=5)[0], F32)
        g_p, g_wi, g_wo = grad(p, wi, wo)
        for leaf in (g_wi, g_wo, *g_p.values()):
            assert np.all(np.isfinite(np.asarray(leaf))), case
        for name, g in g_p.items():
            g = np.asarray(g, np.float64)
            # the ratio's score at a ratio of 0 or 1 is unbounded where the absent lobe peaks (mean 0, labels_and_mie
            # 2.5); every other score is bounded, and its mean is zero
            if name == "forward_ratio" and case[2] in (0.0, 1.0):
                continue
            if name == "forward_ratio" and case[0] == -case[1]:  # two equal lobes: the ratio's score is 0
                assert np.max(np.abs(g)) < 1e-5, case
                continue
            z = g.mean() / max(g.std(ddof=1) / math.sqrt(n), 1e-30)
            assert abs(z) < Z_MAX, (case, name, z)
        tangent = {k: jnp.ones_like(v) for k, v in p.items()}
        _, t = jax.jvp(lambda q: jax.vmap(lambda w: lp(q, wi, w))(wo[:1000]), (p,), (tangent,))
        assert np.all(np.isfinite(np.asarray(t))), case


@pytest.mark.parametrize("case", [(0.0, 0.0, 0.5), (0.6, 0.3, 0.7), (0.0, 0.8, 0.5), (0.95, 0.0, 1.0),
                                  (0.0, 0.9, 0.0), (0.999, 0.999, 0.3), (-0.5, 0.4, 0.6)])  # fmt: skip
def test_henyey_greenstein_moments(case):
    """<P_l(mu)> = r g_f^l + (1 - r) (-g_b)^l for l = 1..4 from 1e6 draws, |z| <= 4.5."""
    gf, gb, r = case
    _, mu = sample_mu(case, 1_000_000, seed=17)
    for order in range(1, 5):
        pl = np.polynomial.legendre.legval(mu, [0] * order + [1])
        truth = r * gf**order + (1 - r) * (-gb) ** order
        z = (pl.mean() - truth) / (pl.std(ddof=1) / math.sqrt(len(pl)))
        assert abs(z) < Z_MAX, (case, order, pl.mean(), truth, z)


@pytest.mark.parametrize("geant4", ["10.7.4", "11.2.2"])
@pytest.mark.parametrize("name", ["mieA", "mieB"])
def test_single_scatter_angles_match_geant4(geant4, name):
    """the drawn cos(theta) histogram against Geant4's single scatters in the g4ref toy (40 bins, both errors)."""
    man = json.loads((DATA / "MANIFEST.json").read_text())
    case = tuple(man["cases"][name][k] for k in ("g_forward", "g_backward", "forward_ratio"))
    with np.load(DATA / f"g4_{geant4}_{name}_hist.npz") as z:
        edges, obs, err, model = z["edges"], z["obs"], z["err"], z["A_mirror"]
    n = 4_000_000
    _, mu = sample_mu(case, n, seed=23)
    h, _ = np.histogram(mu, edges)
    ours, ours_err = h / n, np.sqrt(h) / n
    c2 = float(np.sum((obs - ours) ** 2 / (err**2 + ours_err**2)))
    assert stats.chi2.sf(c2, len(obs) - 1) > P_MIN, (geant4, name, c2)
    # and lumen against the exact bin masses of Geant4's reading
    c2_exact = float(np.sum((ours - model) ** 2 / np.maximum(model / n, 1e-30)))
    assert stats.chi2.sf(c2_exact, len(obs) - 1) > P_MIN, (geant4, name, c2_exact)


# --------------------------------------------------------------------------------------- reading the tables
def _mie_props(**consts):
    out = {"MIEHG": schema.Property("MIEHG", np.array([20.0, 5.0]), np.array([2.0, 4.5]))}
    out.update({k: schema.Property(k, np.asarray(v, np.float64)) for k, v in consts.items()})
    return out


@pytest.fixture(scope="module")
def liq(tmp_path_factory):
    return read_toy(tmp_path_factory).materials["Liq"]


def _with(material, props):
    return dataclasses.replace(material, properties={**material.properties, **props})


def test_the_mie_channel_comes_between_rayleigh_and_wls(liq):
    m = _with(liq, _mie_props(MIEHG_FORWARD=0.6, MIEHG_BACKWARD=0.3, MIEHG_FORWARD_RATIO=0.7))
    e = rules.medium(m)
    assert [c.name for c in e.medium.channels] == ["rayleigh", "mie"]
    assert e.medium.channels[1].lobe is phase.MIE_HG and bulk.CHANNEL_ORDER == ("rayleigh", "mie", "wls")
    assert e.params["mie"]["lobe"] == {"g_forward": 0.6, "g_backward": 0.3, "forward_ratio": 0.7}
    ev = bulk.evaluate(e.medium, jax.tree_util.tree_map(lambda x: jnp.asarray(x, F32), e.params), F32(1239.841984 / 4))
    assert abs(float(ev["mu"][1]) - 1 / np.interp(4.0, [2.0, 4.5], [20.0, 5.0])) < 1e-6
    bulk.validate(e.medium, e.params, 300.0, 600.0)
    # a medium without MIEHG is unchanged: no channel, no parameters
    assert [c.name for c in rules.medium(liq).medium.channels] == ["rayleigh"]


def test_mie_constants_read_as_geant4_reads_them(liq):
    def refused(match, **c):
        with pytest.raises(rules.OpticsRefused, match=match):
            rules.medium(_with(liq, _mie_props(**c)))

    refused("without MIEHG_FORWARD_RATIO.*mat202", MIEHG_FORWARD=0.6, MIEHG_BACKWARD=0.3)
    refused("without MIEHG_BACKWARD.*backward", MIEHG_FORWARD=0.6, MIEHG_FORWARD_RATIO=0.7)
    refused("without MIEHG_FORWARD.*forward", MIEHG_BACKWARD=0.3, MIEHG_FORWARD_RATIO=0.7)
    refused("outside", MIEHG_FORWARD=0.9995, MIEHG_BACKWARD=0.3, MIEHG_FORWARD_RATIO=0.7)
    refused("outside", MIEHG_FORWARD=0.5, MIEHG_BACKWARD=0.3, MIEHG_FORWARD_RATIO=1.2)
    with pytest.raises(rules.OpticsRefused, match="MIEHG must be a table"):
        rules.medium(_with(liq, {"MIEHG": schema.Property("MIEHG", np.asarray(20.0))}))
    tab = schema.Property("MIEHG_FORWARD", np.array([0.5, 0.6]), np.array([2.0, 4.5]))
    with pytest.raises(rules.OpticsRefused, match="must be a constant"):
        rules.medium(_with(liq, {**_mie_props(MIEHG_BACKWARD=0.3, MIEHG_FORWARD_RATIO=0.7), "MIEHG_FORWARD": tab}))
    # negative g is Geant4's flipped lobe; a constant never read may be absent (ratio 1: the backward g)
    e = rules.medium(_with(liq, _mie_props(MIEHG_FORWARD=-0.5, MIEHG_FORWARD_RATIO=1.0)))
    assert e.params["mie"]["lobe"] == {"g_forward": -0.5, "g_backward": 0.0, "forward_ratio": 1.0}
    assert rules.mie_constants(_with(liq, _mie_props(MIEHG_FORWARD=-0.5, MIEHG_FORWARD_RATIO=1.0)).properties,
                               "x")[1] == {"MIEHG_FORWARD_RATIO"}  # fmt: skip
    # without MIEHG the constants are never read: no channel, and not parameters
    inert = _with(liq, {k: schema.Property(k, np.asarray(0.5)) for k in rules.MIE_KEYS})
    assert [c.name for c in rules.medium(inert).medium.channels] == ["rayleigh"]
    assert spec.not_parameters(inert) == set(rules.MIE_KEYS)
    assert not any("MIEHG" in line for line in rules.key_report({"inert": inert}))  # Geant4's keys, not user keys


MIE_GDML = GDML.replace(
    '  <matrix name="GV"',
    '  <matrix name="MIEL" coldim="2" values="2.0*eV 20*m 4.5*eV 5*m"/>\n'
    '  <matrix name="MIEF" coldim="1" values="0.6"/>\n  <matrix name="MIEB" coldim="1" values="0.3"/>\n'
    '  <matrix name="MIER" coldim="1" values="0.7"/>\n  <matrix name="GV"',
).replace(
    '    <property name="ABSLENGTH" ref="ABS0x2"/>\n',
    '    <property name="ABSLENGTH" ref="ABS0x2"/>\n    <property name="MIEHG" ref="MIEL"/>\n'
    '    <property name="MIEHG_FORWARD" ref="MIEF"/>\n    <property name="MIEHG_BACKWARD" ref="MIEB"/>\n'
    '    <property name="MIEHG_FORWARD_RATIO" ref="MIER"/>\n',
)


def test_a_mie_detector_converts_loads_calibrates_and_exports(tmp_path):
    from test_api import RECIPE

    start = MIE_GDML.index('  <bordersurface name="B1"')
    end = MIE_GDML.index("</bordersurface>") + len("</bordersurface>\n")
    path = lumen.convert(write(tmp_path, MIE_GDML[:start] + MIE_GDML[end:]), RECIPE, tmp_path / "mie.lumen")
    det = lumen.load(path)
    p = det.params()
    for key, (unit, rng_) in {"MIEHG": ("m", (0.0, math.inf)), "MIEHG_FORWARD": ("", (-0.999, 0.999)),
                              "MIEHG_BACKWARD": ("", (-0.999, 0.999)),
                              "MIEHG_FORWARD_RATIO": ("", (0.0, 1.0))}.items():  # fmt: skip
        info = p.info(f"materials/Water/{key}")
        assert (info.unit, info.range, info.method, info.available) == (unit, rng_, "reweight", True), key
    with pytest.raises(ValueError, match="outside"):
        p.set("materials/Water/MIEHG_FORWARD", 1.0)
    water = next(k for k in det.pack()["media"] if "Water" in k)
    assert np.asarray(det.pack()["media"][water]["mie"]["lobe"]["g_forward"]).item() == pytest.approx(0.6)
    packed = det.pack(p.set("materials/Water/MIEHG_FORWARD", 0.8))
    assert np.asarray(packed["media"][water]["mie"]["lobe"]["g_forward"]).item() == pytest.approx(0.8)
    # GDML export writes the three constants back (coldim 1) and the read-back file is the same
    exporter.export(det.file, tmp_path / "x" / "mie.gdml", scope="full")
    back = exporter.read_export(tmp_path / "x" / "mie.gdml").detector_file
    for key in ("MIEHG", *rules.MIE_KEYS):
        a, b = det.file.materials["Water"].properties[key], back.materials["Water"].properties[key]
        np.testing.assert_array_equal(a.value, b.value)
