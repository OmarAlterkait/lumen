"""GLISUR facet refraction (glisur ground dielectric_dielectric): the path's weights, as the photon step forms them.

The path is drawn at the reference (polish, the two indices, the photon's state) and weighted by its live / reference
density, facet by facet (``lumen.surfaces.facets.dielectric_event_logpdf``; ``lumen.transport.surface_event``,
``_dielectric_path_branch``). What is checked:

- reweighted to other values (a smaller facet ball, either index moved), the paths drawn at the reference are the
  paths drawn there: the last direction's histogram and the share that ends beyond the surface;
- the derivatives in the polish and in either index are those of the draw itself (float32 forward mode against
  central differences of the analog draw with common random numbers);
- between matched indices a facet refraction keeps the direction, with weight 1.

Geant4's own paths: ``tests/unit/test_g4_boundary.py``.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from lumen.surfaces import facets
from lumen.tools.stats import ALPHA, SUITE_TESTS, z_limit
from lumen.transport import surface_event as se

F32 = jnp.float32
UP = jnp.array([0.0, 0.0, 1.0], F32)
P_MIN = ALPHA / SUITE_TESTS
N = 1 << 18
R0, IDX0, THETA = 0.7, (1.0, 1.67), 1.05  # polish 0.3, gas into TPB, 60 degrees
EDGES_Z, EDGES_PHI = np.linspace(-1.0, 1.0, 21), np.linspace(0.0, math.pi, 7)


def _uniforms(seed, n):
    u = jax.random.uniform(jax.random.PRNGKey(seed), (n, 3), F32, 1e-7, 1 - 1e-7)
    return ((u * 2.0**23).astype(jnp.int32).astype(F32) + 0.5) * 2.0**-23


def _wi(theta=THETA):
    return jnp.array([math.sin(theta), 0.0, -math.cos(theta)], F32)


def _states(seed, n, wi):
    """Return random linear states perpendicular to wi (unpolarised light, filled as the step fills it)."""
    s = np.array([0.0, 1.0, 0.0])
    p = np.cross(np.asarray(wi, np.float64), s)
    phi = np.random.default_rng(seed).uniform(0.0, 2.0 * math.pi, n)[:, None]
    return jnp.asarray(np.cos(phi) * p + np.sin(phi) * s, F32), jnp.zeros((n, 3), F32)


def _weighted(lobe_name, live, ref, shared=False, theta=THETA):
    """Return f(u, er, ei, r, n1, n2) -> (direction, weight, far) of the step's dielectric branch, drawn at ``ref``."""
    lobe = facets.FACET_LOBES[lobe_name]
    wi = _wi(theta)

    def one(u, er, ei, r, n1, n2):
        r_ref, idx_ref = (jax.lax.stop_gradient(r), (jax.lax.stop_gradient(n1), jax.lax.stop_gradient(n2))) if shared \
            else (F32(ref[0]), (F32(ref[1]), F32(ref[2])))  # fmt: skip
        lb = se.Lobes(wi=wi, n=UP, er=er, ei=ei, width={"sigma_alpha": r}, width_ref={"sigma_alpha": r_ref}, rand=u,
                      shared=shared, idx=(n1, n2), idx_ref=idx_ref)  # fmt: skip
        out = se._dielectric_path_branch(lobe, lb)(None)
        return out.d, out.w, out.far

    return one, live


def _analog(r, n1, n2, u, er, ei, theta=THETA):
    """Return (last directions, far) of paths drawn at (r, n1, n2) by the lobe's own sampler (float64 out)."""
    lobe, wi = facets.FACET_LOBES["glisur_dielectric"], _wi(theta)

    def one(uu, a, b):
        p = {"sigma_alpha": F32(r), "idx": (F32(n1), F32(n2)), "er": a, "ei": b}
        _, wo, far, _ = lobe.path(p, wi, UP, uu, lambda c, ev: c, ())
        return wo, far

    wo, far = jax.jit(jax.vmap(one))(u, er, ei)
    return np.asarray(wo, np.float64), np.asarray(far, np.float64)


def _hist(d, w=None):
    return np.histogram2d(d[:, 2], np.abs(np.arctan2(d[:, 1], d[:, 0])), [EDGES_Z, EDGES_PHI], weights=w)[0].ravel()


@pytest.mark.parametrize("lobe_name", ["glisur_dielectric", "glisur_dielectric+cover"])
@pytest.mark.parametrize("live", [(0.67, *IDX0), (R0, 1.0, 1.60), (R0, 1.0, 1.74), (R0, 1.03, 1.67)],
                         ids=["narrower-ball", "n2-down", "n2-up", "n1-up"])  # fmt: skip
def test_a_path_reweighted_to_other_values_is_the_path_drawn_there(lobe_name, live):
    """Reference paths weighted live / reference against paths drawn at the live values: the histogram (weighted
    chi2, both samples' variances) and the share beyond the surface."""
    one, _ = _weighted(lobe_name, live, (R0, *IDX0))
    u = _uniforms(1, N)
    er, ei = _states(2, N, _wi())
    d, w, far = jax.jit(jax.vmap(one, (0, 0, 0, None, None, None)))(u, er, ei, *(F32(x) for x in live))
    d, w, far = np.asarray(d, np.float64), np.asarray(w, np.float64), np.asarray(far, np.float64)
    hw, vw = _hist(d, w), _hist(d, w * w)
    d2, far2 = _analog(*live, _uniforms(3, N), *_states(4, N, _wi()))
    hd = _hist(d2)
    use = (vw + hd) > 0
    c2 = float(np.sum((hw[use] - hd[use]) ** 2 / (vw[use] + hd[use])))
    assert stats.chi2.sf(c2, int(use.sum())) > P_MIN, (c2, int(use.sum()))
    m_w, s_w = float(np.mean(w * far)), float(np.std(w * far)) / math.sqrt(N)
    m_d, s_d = float(np.mean(far2)), float(np.std(far2)) / math.sqrt(N)
    assert abs(m_w - m_d) < z_limit() * math.hypot(s_w, s_d), (m_w, m_d)


@pytest.mark.parametrize("which", [0, 1, 2], ids=["width", "n1", "n2"])
def test_the_derivatives_are_those_of_the_draw(which):
    """d/dx E[far] and E[dz] from the step's weights (forward mode, the covered lobe, the reference the live values)
    against central differences of the analog draw at x +- h with common random numbers, Richardson extrapolated."""
    x0 = [R0, *IDX0]
    one, _ = _weighted("glisur_dielectric+cover", x0, x0, shared=True)
    u = _uniforms(5, N)
    er, ei = _states(6, N, _wi())

    def f(u_, a, b):
        def g(t):
            x = [F32(v) for v in x0]
            x[which] = x[which] + t
            d, w, far = one(u_, a, b, *x)
            return jnp.stack([w * far.astype(F32), w * d[2]])

        return jax.jvp(g, (F32(0.0),), (F32(1.0),))[1]

    ad = np.asarray(jax.jit(jax.vmap(f))(u, er, ei), np.float64)
    uu, (ea, eb) = _uniforms(7, 4 * N), _states(8, 4 * N, _wi())

    def obs(t):
        x = list(x0)
        x[which] += t
        d, far = _analog(*x, uu, ea, eb)
        return np.stack([far, d[:, 2]], 1)

    fd = [(obs(h) - obs(-h)) / (2 * h) for h in (0.02, 0.01)]
    rich = (4 * fd[1] - fd[0]) / 3
    for k in range(2):
        err = math.hypot(ad[:, k].std() / math.sqrt(N), rich[:, k].std() / math.sqrt(len(rich)))
        assert abs(ad[:, k].mean() - rich[:, k].mean()) < z_limit(2) * err, (k, ad[:, k].mean(), rich[:, k].mean())


def test_between_matched_indices_a_facet_refraction_passes_straight_with_weight_one():
    """Equal indices: one facet, refracted, the direction and the state kept, weight 1 (live and reference)."""
    lobe, wi = facets.FACET_LOBES["glisur_dielectric+cover"], _wi()
    u = _uniforms(9, 4096)
    er, ei = _states(10, 4096, wi)

    def one(uu, a, b):
        n = F32(1.4)
        lb = se.Lobes(wi=wi, n=UP, er=a, ei=b, width={"sigma_alpha": F32(R0)}, width_ref={"sigma_alpha": F32(R0)},
                      rand=uu, shared=False, idx=(n, n), idx_ref=(n, n))  # fmt: skip
        out = se._dielectric_path_branch(lobe, lb)(None)
        return out.d, out.w, out.far, out.er

    d, w, far, er1 = (np.asarray(x) for x in jax.jit(jax.vmap(one))(u, er, ei))
    np.testing.assert_allclose(d, np.tile(np.asarray(wi), (4096, 1)), atol=1e-6)
    np.testing.assert_allclose(er1, np.asarray(er), atol=1e-6)
    assert np.all(far) and np.allclose(w, 1.0)


# ------------------------------------------------------------------------------------------------- through simulate
TOY_POLISH, TOY_N, TOY_THETA, TOY_CASE = 0.6, (1.0, 1.5), 50.0, "glisur_n1_1.5_p0.6_th50"  # tests/data/g4_boundary
TOY = f"""<?xml version="1.0" encoding="UTF-8" ?>
<gdml>
<define>
  <matrix name="RA" coldim="2" values="2.0*eV {TOY_N[0]} 4.5*eV {TOY_N[0]}"/>
  <matrix name="RB" coldim="2" values="2.0*eV {TOY_N[1]} 4.5*eV {TOY_N[1]}"/>
</define>
<materials>
  <element name="O" formula="O" Z="8"> <atom value="15.999"/> </element>
  <material name="GasA" state="gas"> <property name="RINDEX" ref="RA"/> <D value="0.001"/> <fraction n="1.0" ref="O"/>
  </material>
  <material name="SolidB" state="solid"> <property name="RINDEX" ref="RB"/> <D value="1.0"/> <fraction n="1.0" ref="O"/>
  </material>
  <material name="Black" state="solid"> <D value="1.0"/> <fraction n="1.0" ref="O"/> </material>
</materials>
<solids>
  <box name="World" x="300" y="300" z="1" lunit="m"/>
  <box name="Stack" x="200" y="200" z="0.022" lunit="m"/>
  <box name="Slab" x="200" y="200" z="0.010" lunit="m"/>
  <box name="Sense" x="200" y="200" z="0.001" lunit="m"/>
  <opticalsurface name="TPB" model="glisur" finish="ground" type="dielectric_dielectric" value="{TOY_POLISH}"/>
</solids>
<structure>
  <volume name="SenseA"> <materialref ref="GasA"/> <solidref ref="Sense"/> </volume>
  <volume name="SenseB"> <materialref ref="SolidB"/> <solidref ref="Sense"/> </volume>
  <volume name="A"> <materialref ref="GasA"/> <solidref ref="Slab"/> </volume>
  <volume name="B"> <materialref ref="SolidB"/> <solidref ref="Slab"/> </volume>
  <volume name="StackLV"> <materialref ref="GasA"/> <solidref ref="Stack"/>
    <physvol name="a_pv"> <volumeref ref="A"/> <position name="pa" unit="m" x="0" y="0" z="0.005"/> </physvol>
    <physvol name="b_pv"> <volumeref ref="B"/> <position name="pb" unit="m" x="0" y="0" z="-0.005"/> </physvol>
    <physvol name="sa_pv"> <volumeref ref="SenseA"/> <position name="psa" unit="m" x="0" y="0" z="0.0105"/> </physvol>
    <physvol name="sb_pv"> <volumeref ref="SenseB"/> <position name="psb" unit="m" x="0" y="0" z="-0.0105"/> </physvol>
  </volume>
  <volume name="WorldLV"> <materialref ref="Black"/> <solidref ref="World"/>
    <physvol name="stack"> <volumeref ref="StackLV"/> </physvol>
  </volume>
  <bordersurface name="AB" surfaceproperty="TPB"> <physvolref ref="a_pv"/> <physvolref ref="b_pv"/> </bordersurface>
</structure>
<setup name="Default" version="1.0"> <world ref="WorldLV"/> </setup>
</gdml>
"""
TOY_N_PHOTONS = 1 << 16


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """A gas slab over a solid one, 200 m wide and 1 cm thick, the border between them GLISUR ground (polish 0.6),
    each counted beyond its far face by a sensor slab of its own material; a pencil beam from the gas at 50 degrees.
    Every photon meets the border once and ends in one of the two sensors."""
    import lumen
    from lumen.format.sensorsets import SensorGroup
    from lumen.recipe import Recipe
    from lumen.sources import FromArrays

    tmp = tmp_path_factory.mktemp("facet_toy")
    (tmp / "facets.gdml").write_text(TOY)
    groups = tuple(SensorGroup(g, rule="any_entry", lv=lv, settings={"entry_efficiency": 1.0})
                   for g, lv in (("above", "SenseA"), ("below", "SenseB")))  # fmt: skip
    det = lumen.load(lumen.convert(tmp / "facets.gdml", Recipe("facets", "StackLV", groups=groups), tmp / "f.lumen"))
    n, th = TOY_N_PHOTONS, math.radians(TOY_THETA)
    src = FromArrays(np.tile([0.0, 0.0, 0.002], (n, 1)), np.tile([math.sin(th), 0.0, -math.cos(th)], (n, 1)),
                     np.zeros(n), np.full(n, 400.0), np.ones(n))  # fmt: skip
    below = [i for i, name in enumerate(det.sensor_table().name) if name.endswith("SenseB")][0]
    return det, src, below


def _g4_case():
    import json
    from pathlib import Path

    data = Path(__file__).resolve().parents[1] / "data" / "g4_boundary"
    case = json.loads((data / "MANIFEST.json").read_text())["cases"][TOY_CASE]
    with np.load(data / case["file"]) as z:
        return case, z["hist"][z["edges_z"][:-1] < 0].sum() / z["hist"].sum(), int(z["launched"])


@pytest.mark.slow  # converts a toy and compiles the photon loop: about a minute on 4 CPUs
def test_through_simulate_the_share_beyond_the_border_is_geant4s(toy):
    import lumen
    from lumen.rules.choices import GLISUR_R_MIN

    det, src, below = toy
    case, g4, n_g4 = _g4_case()
    assert (case["n1"], case["n2"], case["value"], case["theta"]) == (*TOY_N, TOY_POLISH, TOY_THETA)
    assert det.params().info("surfaces/TPB/polish").range == (0.0, 1.0 - GLISUR_R_MIN)
    light = lumen.simulate(det, src, seed=1)
    a, e = np.asarray(light.arrivals, np.float64), np.asarray(light.arrivals_error, np.float64)
    share, err = a[below] / TOY_N_PHOTONS, e[below] / TOY_N_PHOTONS
    assert abs(share - g4) < z_limit() * math.hypot(err, math.sqrt(g4 * (1 - g4) / n_g4)), (share, err, g4)


@pytest.mark.slow  # compiles the photon loop's tangent: minutes on 4 CPUs
def test_through_simulate_the_polish_derivative_is_the_draws(toy):
    """d/d polish of the share below the border, from simulate's tangent (8 seeds), against central differences of
    the analog facet path at the toy's values (d/d polish = -d/dr). An index's derivative is pending in this version
    (RINDEX's method, ``lumen.params.PENDING``): a held reference reaches other indices instead (the next test)."""
    import lumen

    det, src, below = toy
    p0 = det.params()

    def share(t, seed):
        return lumen.simulate(det, src, p0.set("surfaces/TPB/polish", TOY_POLISH + t), seed=seed).arrivals[below]

    ad = np.array([float(jax.jvp(lambda t, s=s: share(t, s), (F32(0.0),), (F32(1.0),))[1]) for s in range(1, 9)])
    ad = ad / TOY_N_PHOTONS
    x0, th = [1.0 - TOY_POLISH, *TOY_N], math.radians(TOY_THETA)
    uu, (ea, eb) = _uniforms(11, 4 * N), _states(12, 4 * N, _wi(th))

    def far(t):
        return _analog(x0[0] + t, *x0[1:], uu, ea, eb, theta=th)[1]

    fd = [(far(h) - far(-h)) / (2 * h) for h in (0.02, 0.01)]
    rich = -(4 * fd[1] - fd[0]) / 3
    err = math.hypot(ad.std(ddof=1) / math.sqrt(len(ad)), rich.std() / math.sqrt(len(rich)))
    assert abs(ad.mean() - rich.mean()) < z_limit() * err, (ad.mean(), rich.mean(), err)


@pytest.mark.slow  # compiles the photon loop: about a minute on 4 CPUs
@pytest.mark.parametrize(("key", "factor"), [("materials/SolidB/RINDEX", 1.005), ("materials/GasA/RINDEX", 1.02),
                                             ("surfaces/TPB/polish", 1.1)])  # fmt: skip
def test_through_simulate_a_held_reference_reaches_other_values(toy, key, factor):
    """simulate(params, reference=default): the paths drawn at the default values and weighted to ``params`` (an index
    or the polish moved; the facet ball narrower) give the light drawn at ``params`` (other seeds). A move of the far
    medium's index leaves the reference's facet ball (docs/explanation/known-limits.md): a small one here."""
    import lumen

    det, src, below = toy
    p0 = det.params()
    p = p0.scale(key, factor)
    held = [lumen.simulate(det, src, p, reference=p0, seed=s) for s in (1, 2)]
    drawn = [lumen.simulate(det, src, p, seed=s) for s in (3, 4)]
    a = [np.asarray(x.arrivals[below], np.float64) / TOY_N_PHOTONS for x in (*held, *drawn)]
    e = [np.asarray(x.arrivals_error[below], np.float64) / TOY_N_PHOTONS for x in (*held, *drawn)]
    m_h, m_d = (a[0] + a[1]) / 2, (a[2] + a[3]) / 2
    err = 0.5 * math.sqrt(e[0] ** 2 + e[1] ** 2 + e[2] ** 2 + e[3] ** 2)
    assert abs(m_h - m_d) < z_limit() * err, (key, m_h, m_d, err)
