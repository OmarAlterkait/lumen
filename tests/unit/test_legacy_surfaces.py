"""The legacy surface types: the contract (probabilities on the declared outcomes, summing to 1, finite derivatives
everywhere incl. the critical angle and grazing incidence), Geant4's chained cuts, and the Fresnel physics against
float64."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lumen.media import phase, spectra
from lumen.surfaces import _legacy as legacy
from lumen.surfaces import fresnel
from lumen.surfaces.outcomes import (
    ABSORB,
    BACKSCATTER,
    DETECT,
    DIFFUSE,
    GLOSSY,
    N_OUTCOMES,
    REEMIT,
    REFRACT,
    SPECULAR,
    STRAIGHT,
)

F32 = jnp.float32
NZ = jnp.asarray([0.0, 0.0, 1.0], F32)
Z_MAX = 4.5


def ctx_of(cos_i, n1, n2, lam=400.0):
    s = math.sqrt(max(1.0 - cos_i * cos_i, 0.0))
    return dict(lam=F32(lam), wi=jnp.asarray([s, 0.0, -cos_i], F32), n=NZ, n1=F32(n1), n2=F32(n2))


def np_fresnel(c, n1, n2):
    s2 = (n1 / n2) ** 2 * (1 - c * c)
    if s2 >= 1:
        return 1.0
    ct = math.sqrt(1 - s2)
    rs = ((n1 * c - n2 * ct) / (n1 * c + n2 * ct)) ** 2
    rp = ((n2 * c - n1 * ct) / (n2 * c + n1 * ct)) ** 2
    return 0.5 * (rs + rp)


SETS = {
    "pass": [{}],
    "absorber": [{}],
    "dielectric": [{}],
    "reflector": [
        {"R": 0.9, "f_spec": 0.3, "f_glossy": 0.5, "sigma_alpha": 0.1},
        {"R": 0.0, "f_spec": 0.0, "f_glossy": 0.0, "sigma_alpha": 0.005},
        {"R": 1.0, "f_spec": 1.0, "f_glossy": 0.0, "sigma_alpha": 1.0},
        {"R": 1.0, "f_spec": 0.0, "f_glossy": 1.0, "sigma_alpha": 1.0},
    ],
    "photocathode": [
        {"nr": 2.7, "nk": 1.5, "qe": 0.25, "f_spec": 0.5, "f_glossy": 0.3, "sigma_alpha": 0.05},
        {"nr": 1.6, "nk": 1e-4, "qe": 0.0, "f_spec": 1.0, "f_glossy": 0.0, "sigma_alpha": 0.01},
        {"nr": 3.5, "nk": 2.5, "qe": 1.0, "f_spec": 0.0, "f_glossy": 1.0, "sigma_alpha": 0.5},
    ],
    "counting": [{"eff": 0.03}, {"eff": 0.0}, {"eff": 1.0}],
    "sensor": [
        {"R": 0.3, "f_spec": 0.6, "f_glossy": 0.2, "sigma_alpha": 0.05, "qe": 0.3},
        {"R": 0.0, "f_spec": 0.0, "f_glossy": 0.0, "sigma_alpha": 0.005, "qe": 1.0},
        {"R": 1.0, "f_spec": 1.0, "f_glossy": 0.0, "sigma_alpha": 1.0, "qe": 0.0},
    ],
    "dielectric_metal": [
        {"R": 0.9, "p_spike": 0.2, "p_lobe": 0.5, "p_back": 0.1, "sigma_alpha": 0.1, "eff": 0.2},
        {"R": 0.0, "p_spike": 0.0, "p_lobe": 0.0, "p_back": 0.0, "sigma_alpha": 0.005, "eff": 1.0},
        {"R": 1.2, "p_spike": 0.7, "p_lobe": 0.6, "p_back": 0.3, "sigma_alpha": 0.3, "eff": 0.5},
    ],
    "dielectric_rt": [{"R": 0.7, "T": 0.1}, {"R": 0.0, "T": 0.0}, {"R": 1.0, "T": 0.0}, {"R": 0.8, "T": 0.4}],
    "dielectric_rt_sensor": [
        {"R": 0.7, "T": 0.1, "eff": 0.4},
        {"R": 0.0, "T": 1.0, "eff": 0.5},
        {"R": 0.8, "T": 0.4, "eff": 0.5},
    ],
    "glisur_metal": [{"R": 0.8, "polish": 0.7}, {"R": 1.3, "polish": 0.0}],
    "glisur_metal_eff": [{"R": 0.8, "polish": 0.7, "eff": 0.3}, {"R": 0.0, "polish": 0.99, "eff": 1.0}],
    "specular_metal": [{"R": 0.9}, {"R": 1.5}, {"R": -0.2}],
    "unified_metal": [
        {"R": 0.8, "p_spike": 0.1, "p_lobe": 0.5, "p_back": 0.1, "sigma_alpha": 0.2},
        {"R": 0.5, "p_spike": 0.7, "p_lobe": 0.6, "p_back": 0.3, "sigma_alpha": 0.5},
    ],
    "painted": [{"R": 0.6, "f_spec": 1.0}, {"R": 0.6, "f_spec": 0.0}],
    "painted_t": [{"R": 0.5, "T": 0.3, "f_spec": 0.0}, {"R": 0.7, "T": 0.6, "f_spec": 1.0}],
}
COS = [1.0, 0.9, 0.5, 0.2, 0.05, 1e-3, 0.0]
PAIRS = [(1.0, 1.33), (1.33, 1.0), (1.33, 1.5), (1.5, 1.0), (1.36, 1.36), (1.0, 1.6)]


def test_every_type_has_parameter_sets():
    assert len(legacy.TYPES) == 16
    assert sorted(t.name for t in legacy.TYPES) == sorted(SETS)


def contract_rows(sets):
    """Every (parameter set, index pair, incidence) combination as stacked rows, the critical angles included."""
    rows = []
    for p0 in sets:
        for n1, n2 in PAIRS:
            crit = [math.sqrt(1 - (n2 / n1) ** 2)] if n1 > n2 else []
            for c in COS + crit:
                rows.append((p0, n1, n2, c))
    s = np.array([math.sqrt(max(1.0 - c * c, 0.0)) for _, _, _, c in rows])
    c = np.array([r[3] for r in rows])
    wi = np.stack([s, np.zeros_like(s), -c], -1).astype(np.float32)
    er = np.stack([np.full_like(s, 0.6), 0.8 * c, 0.8 * s], -1).astype(np.float32)
    params = {k: np.array([r[0][k] for r in rows], np.float32) for k in sets[0]}
    return params, wi, er, np.array([r[1] for r in rows], np.float32), np.array([r[2] for r in rows], np.float32)


@pytest.mark.parametrize("kind", legacy.TYPES, ids=lambda k: k.name)
@pytest.mark.parametrize("polarised", [False, True])
def test_type_contract(kind, polarised):
    params, wi, er, n1, n2 = contract_rows(SETS[kind.name])

    def probs(p, a, b, w, e):
        ctx = dict(lam=F32(400.0), wi=w, n=NZ, n1=a, n2=b)
        if polarised:
            ctx.update(er=e, ei=jnp.zeros(3, F32))
        return kind.probs(p, ctx)

    def weighted(p, a, b, w, e):
        return jnp.sum(probs(p, a, b, w, e) * jnp.arange(1.0, N_OUTCOMES + 1))

    pr = np.asarray(jax.jit(jax.vmap(probs))(params, n1, n2, wi, er), np.float64)
    grads = jax.jit(jax.vmap(jax.grad(weighted, argnums=(0, 1, 2, 3))))(params, n1, n2, wi, er)
    assert pr.shape == (len(wi), N_OUTCOMES)
    assert np.all(pr >= -1e-7) and np.all(pr <= 1 + 1e-6), (kind.name, pr.min(), pr.max())
    assert np.max(np.abs(pr.sum(1) - 1.0)) < 2e-6, kind.name
    off = [i for i in range(N_OUTCOMES) if i not in kind.outcomes]
    assert np.all(pr[:, off] == 0.0), (kind.name, "outcome outside the declared support")
    for leaf in jax.tree_util.tree_leaves(grads):
        bad = ~np.isfinite(np.asarray(leaf))
        assert not bad.any(), (kind.name, "non-finite derivative", np.flatnonzero(bad.reshape(len(wi), -1).any(1)))


def test_fresnel_against_float64_limits_and_reciprocity():
    refl = jax.jit(jax.vmap(fresnel.unpolarised_reflectance))
    for n1, n2 in PAIRS:
        c = np.linspace(1e-3, 1.0, 400)
        got = np.asarray(refl(jnp.asarray(c, F32), jnp.full(400, n1, F32), jnp.full(400, n2, F32)), np.float64)
        ref = np.array([np_fresnel(x, n1, n2) for x in c])
        tol = np.where((c > 1e-2) | (n1 != n2), 5e-5, 2e-3)  # float32: 1 - sin^2 loses digits at grazing, n1 = n2
        assert np.all(np.abs(got - ref) < tol), (n1, n2, c[np.argmax(np.abs(got - ref) - tol)])
        normal = float(fresnel.unpolarised_reflectance(F32(1.0), F32(n1), F32(n2)))
        assert abs(normal - ((n1 - n2) / (n1 + n2)) ** 2) < 1e-6
        if n1 < n2:  # reciprocity: R(theta_i, n1 -> n2) = R(theta_t, n2 -> n1)
            for c in (0.9, 0.5, 0.1):
                ct = math.sqrt(1 - (n1 / n2) ** 2 * (1 - c * c))
                a = float(fresnel.unpolarised_reflectance(F32(c), F32(n1), F32(n2)))
                b = float(fresnel.unpolarised_reflectance(F32(ct), F32(n2), F32(n1)))
                assert abs(a - b) < 2e-6
    assert float(fresnel.unpolarised_reflectance(F32(0.5), F32(1.36), F32(1.36))) < 1e-7


def test_absorbing_fresnel_tends_to_dielectric_and_photocathode_shares():
    for n1 in (1.33, 1.5):
        for nr in (2.0, 2.7, 3.5):
            for c in (1.0, 0.7, 0.3, 0.05):
                a = float(fresnel.unpolarised_absorbing_reflectance(F32(c), F32(n1), F32(nr), F32(1e-5)))
                assert abs(a - np_fresnel(c, n1, nr)) < 2e-5, (n1, nr, c, a)
    p = {
        "nr": F32(2.7),
        "nk": F32(1.5),
        "qe": F32(0.25),
        "f_spec": F32(1.0),
        "f_glossy": F32(0.0),
        "sigma_alpha": F32(0.1),
    }
    pr = np.asarray(legacy.PHOTOCATHODE.probs(p, ctx_of(0.8, 1.5, 1.5)))
    R = float(fresnel.unpolarised_absorbing_reflectance(F32(0.8), F32(1.5), F32(2.7), F32(1.5)))
    np.testing.assert_allclose(pr[[SPECULAR, DETECT, ABSORB]], [R, (1 - R) * 0.25, (1 - R) * 0.75], atol=1e-6)


def test_dielectric_rt_is_geant4s_absorption_before_fresnel():
    """rand > R + T absorbed (detected with EFFICIENCY), R < rand <= R + T straight, rand <= R Fresnel."""
    p = {"R": F32(0.6), "T": F32(0.15), "eff": F32(0.3)}
    for c, n1, n2 in ((0.8, 1.5, 1.0), (0.3, 1.5, 1.0), (0.9, 1.0, 1.5)):
        pr = np.asarray(legacy.DIELECTRIC_RT_SENSOR.probs(p, ctx_of(c, n1, n2)), np.float64)
        F = np_fresnel(c, n1, n2)
        np.testing.assert_allclose(
            pr[[REFRACT, SPECULAR, STRAIGHT, DETECT, ABSORB]],
            [0.6 * (1 - F), 0.6 * F, 0.15, 0.25 * 0.3, 0.25 * 0.7],
            atol=2e-6,
        )
        q = np.asarray(legacy.DIELECTRIC_RT.probs({"R": p["R"], "T": p["T"]}, ctx_of(c, n1, n2)), np.float64)
        np.testing.assert_allclose(q[[DETECT, ABSORB]], [0.0, 0.25], atol=2e-6)  # as a surface: absorbed


def test_painted_and_unified_types_are_geant4s_cuts():
    """PAINTED_T: rand > R + T absorbed, rand > R transmitted, else reflected (values summing past 1 included)."""
    N = 1_000_000
    rg = np.random.default_rng(2)
    ctx = dict(lam=F32(400.0), wi=jnp.asarray([0.0, 0.6, -0.8], F32), n=NZ, n1=F32(1.3), n2=F32(1.0))
    for R, T in ((0.5, 0.3), (0.7, 0.6), (0.0, 0.2)):
        pr = np.asarray(legacy.PAINTED_T.probs({"R": F32(R), "T": F32(T), "f_spec": F32(0.0)}, ctx))
        u = rg.random(N)
        freq = np.array([np.mean(~(u > R)), np.mean((u > R) & ~(u > R + T)), np.mean(u > R + T)])
        want = pr[[DIFFUSE, STRAIGHT, ABSORB]]
        z = (freq - want) / np.sqrt(np.maximum(want * (1 - want), 1e-12) / N)
        assert np.max(np.abs(z)) < Z_MAX and abs(pr.sum() - 1) < 1e-6
    pu = np.asarray(
        legacy.UNIFIED_METAL.probs(
            {"R": F32(0.8), "p_spike": F32(0.1), "p_lobe": F32(0.5), "p_back": F32(0.1), "sigma_alpha": F32(0.2)}, ctx
        )
    )
    assert np.allclose(pu[[SPECULAR, GLOSSY, BACKSCATTER, DIFFUSE, ABSORB]], [0.08, 0.4, 0.08, 0.24, 0.2], atol=1e-6)
    assert pu[DETECT] == 0


def test_clip01_cuts_keep_the_linear_derivative_on_the_simplex_faces():
    for r in (0.0, 1.0):
        g = jax.jacfwd(lambda x: jnp.stack(legacy.chained_cuts(x)))(F32(r))
        np.testing.assert_array_equal(np.asarray(g), [1.0, -1.0])


def test_glisur_types_name_their_facet_path_lobe():
    assert legacy.GLISUR_METAL.glossy == legacy.GLISUR_METAL_EFF.glossy == "glisur"
    assert legacy.UNIFIED_METAL.glossy == legacy.REFLECTOR.glossy == "unified"
    w = legacy.GLISUR_METAL.lobe_params({"R": F32(0.8), "polish": F32(0.7)}, {})
    assert abs(float(w["sigma_alpha"]) - 0.3) < 1e-7


TPB = spectra.wavelength_spectrum("tpb_like", [380, 400, 420, 440, 470, 520], [0.0, 0.6, 1.0, 0.8, 0.3, 0.0])


def test_coated_film_probabilities_and_faces():
    kind = legacy.coated("tpb_on_reflector", legacy.REFLECTOR, TPB)
    p = {
        "L": F32(2e-6),
        "d": F32(1e-6),
        "eta": F32(1.2),
        "tau": F32(1.5),
        "axis": jnp.asarray([0.0, 0.0, 1.0], F32),
        "b:R": F32(0.9),
        "b:f_spec": F32(0.3),
        "b:f_glossy": F32(0.5),
        "b:sigma_alpha": F32(0.1),
    }
    c = 0.6
    film = np.asarray(kind.probs(p, ctx_of(c, 1.0, 1.0)), np.float64)  # the coated face (n . axis = 1)
    A = 1 - math.exp(-1e-6 / (2e-6 * c))
    np.testing.assert_allclose(film[[REEMIT, ABSORB, STRAIGHT]], [A * 1.2, 0.0, 1 - A], rtol=1e-5)
    bare_ctx = dict(ctx_of(c, 1.0, 1.0), n=-NZ, wi=-ctx_of(c, 1.0, 1.0)["wi"])  # the other face: the bare backing
    bare = np.asarray(kind.probs(p, bare_ctx))
    want = np.asarray(legacy.REFLECTOR.probs({k[2:]: v for k, v in p.items() if k.startswith("b:")}, bare_ctx))
    np.testing.assert_array_equal(bare, want)
    backing_layer = np.asarray(kind.probs(p, dict(ctx_of(c, 1.0, 1.0), layer=2)))
    np.testing.assert_array_equal(backing_layer, want)
    lp = kind.lobe_params(p, ctx_of(c, 1.0, 1.0))
    assert bool(lp["film"]) and bool(lp["coated"]) and kind.reemit_lobe is phase.ISOTROPIC and kind.stack


def test_coated_refuses_a_transmitting_backing():
    with pytest.raises(ValueError, match="only an opaque backing"):
        legacy.coated("bad", legacy.DIELECTRIC, TPB)
