"""The Geant4 surface types: the contract (probabilities on the declared outcomes, summing to 1, finite derivatives
everywhere incl. the critical angle and grazing incidence), Geant4's chained cuts, and the Fresnel physics against
float64."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from surface_reference import ABSORB, DETECT, outcome_vector
from surface_reference import N_FULL as N_OUTCOMES
from surface_reference import full as _full8
from test_surface_table import table_of

from lumen.rules.entries import SensorEntry
from lumen.surfaces import fresnel, types
from lumen.surfaces.outcomes import (
    BACKSCATTER,
    DIFFUSE,
    GLOSSY,
    REFRACT,
    SPECULAR,
    STRAIGHT,
)
from lumen.tools.stats import z_limit

F32 = jnp.float32
NZ = jnp.asarray([0.0, 0.0, 1.0], F32)


def full(stype, p, ctx):
    """Return a row's six continuing shares, then the detected share (DETECT) and the loss (ABSORB), as the per-type
    code gave them (``surface_reference.full``)."""
    return _full8(stype.evaluate(p, ctx))


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
    "photocathode": [
        {"nr": 2.7, "nk": 1.5, "eff": 0.25},
        {"nr": 1.6, "nk": 1e-4, "eff": 0.0},
        {"nr": 3.5, "nk": 2.5, "eff": 1.0},
    ],
    "counting": [{"eff": 0.03}, {"eff": 0.0}, {"eff": 1.0}],
    "dielectric_rt": [
        {"R": 0.7, "T": 0.1, "eff": 0.0},
        {"R": 0.0, "T": 0.0, "eff": 0.0},
        {"R": 1.0, "T": 0.0, "eff": 0.0},
        {"R": 0.7, "T": 0.1, "eff": 0.4},
        {"R": 0.0, "T": 1.0, "eff": 0.5},
        {"R": 0.8, "T": 0.4, "eff": 0.5},
    ],
    "glisur_metal": [
        {"R": 0.8, "polish": 0.7, "eff": 0.3},
        {"R": 1.3, "polish": 0.0, "eff": 0.0},
        {"R": 0.0, "polish": 0.99, "eff": 1.0},
    ],
    "unified_metal": [
        {"R": 0.8, "p_spike": 0.1, "p_lobe": 0.5, "p_back": 0.1, "sigma_alpha": 0.2, "eff": 0.0},
        {"R": 0.5, "p_spike": 0.7, "p_lobe": 0.6, "p_back": 0.3, "sigma_alpha": 0.5, "eff": 0.0},
        {"R": 0.9, "p_spike": 0.2, "p_lobe": 0.5, "p_back": 0.1, "sigma_alpha": 0.1, "eff": 0.2},
        {"R": 0.0, "p_spike": 0.0, "p_lobe": 0.0, "p_back": 0.0, "sigma_alpha": 0.005, "eff": 1.0},
        {"R": 1.2, "p_spike": 0.7, "p_lobe": 0.6, "p_back": 0.3, "sigma_alpha": 0.3, "eff": 0.5},
    ],
    "unified_metal_normal": [
        {"R": 0.8, "p_spike": 0.1, "p_lobe": 0.5, "p_back": 0.1, "eff": 0.0},
        {"R": 0.5, "p_spike": 0.7, "p_lobe": 0.6, "p_back": 0.3, "eff": 0.4},
    ],
    "painted_t": [{"R": 0.5, "T": 0.3, "f_spec": 0.0, "eff": 0.0}, {"R": 0.7, "T": 0.6, "f_spec": 1.0, "eff": 0.5}],
    "glisur_dielectric": [
        {"R": 1.0, "T": 0.0, "polish": 0.01, "eff": 0.0},
        {"R": 0.7, "T": 0.2, "polish": 0.5, "eff": 0.4},
        {"R": 0.8, "T": 0.4, "polish": 0.99, "eff": 1.0},
    ],
}
COS = [1.0, 0.9, 0.5, 0.2, 0.05, 1e-3, 0.0]
PAIRS = [(1.0, 1.33), (1.33, 1.0), (1.33, 1.5), (1.5, 1.0), (1.36, 1.36), (1.0, 1.6)]


def test_every_type_has_parameter_sets():
    assert len(types.TYPES) == 11
    assert sorted(t.name for t in types.TYPES) == sorted(SETS)


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


@pytest.mark.parametrize("kind", types.TYPES, ids=lambda k: k.name)
@pytest.mark.parametrize("polarised", [False, True])
def test_type_contract(kind, polarised):
    params, wi, er, n1, n2 = contract_rows(SETS[kind.name])

    def probs(p, a, b, w, e):
        ctx = dict(lam=F32(400.0), wi=w, n=NZ, n1=a, n2=b)
        if polarised:
            ctx.update(er=e, ei=jnp.zeros(3, F32))
        return full(kind, p, ctx)

    def weighted(p, a, b, w, e):
        return jnp.sum(probs(p, a, b, w, e) * jnp.arange(1.0, N_OUTCOMES + 1))

    pr = np.asarray(jax.jit(jax.vmap(probs))(params, n1, n2, wi, er), np.float64)
    grads = jax.jit(jax.vmap(jax.grad(weighted, argnums=(0, 1, 2, 3))))(params, n1, n2, wi, er)
    assert pr.shape == (len(wi), N_OUTCOMES)
    assert np.all(pr >= -1e-7) and np.all(pr <= 1 + 1e-6), (kind.name, pr.min(), pr.max())
    assert np.max(np.abs(pr.sum(1) - 1.0)) < 2e-6, kind.name
    off = [i for i in range(N_OUTCOMES) if i not in (*kind.outcomes, DETECT, ABSORB)]
    if not kind.detects:
        off.append(DETECT)
    assert np.all(pr[:, off] == 0.0), (kind.name, "outcome outside the declared support")
    for leaf in jax.tree_util.tree_leaves(grads):
        bad = ~np.isfinite(np.asarray(leaf))
        assert not bad.any(), (kind.name, "non-finite derivative", np.flatnonzero(bad.reshape(len(wi), -1).any(1)))


def _unpolarised(cos_i, n1, n2):
    """Return the dielectric's unpolarised reflectance as the surface types take it: the amplitudes'."""
    rs, rp, _, _ = fresnel.amplitudes(cos_i, n1, n2)
    return jnp.clip(fresnel.reflectance(0.5, 0.5, rs, rp), 0.0, 1.0)


def test_fresnel_against_float64_limits_and_reciprocity():
    refl = jax.jit(jax.vmap(_unpolarised))
    for n1, n2 in PAIRS:
        c = np.linspace(1e-3, 1.0, 400)
        got = np.asarray(refl(jnp.asarray(c, F32), jnp.full(400, n1, F32), jnp.full(400, n2, F32)), np.float64)
        ref = np.array([np_fresnel(x, n1, n2) for x in c])
        tol = np.where((c > 1e-2) | (n1 != n2), 5e-5, 2e-3)  # float32: 1 - sin^2 loses digits at grazing, n1 = n2
        assert np.all(np.abs(got - ref) < tol), (n1, n2, c[np.argmax(np.abs(got - ref) - tol)])
        normal = float(_unpolarised(F32(1.0), F32(n1), F32(n2)))
        assert abs(normal - ((n1 - n2) / (n1 + n2)) ** 2) < 1e-6
        if n1 < n2:  # reciprocity: R(theta_i, n1 -> n2) = R(theta_t, n2 -> n1)
            for c in (0.9, 0.5, 0.1):
                ct = math.sqrt(1 - (n1 / n2) ** 2 * (1 - c * c))
                a = float(_unpolarised(F32(c), F32(n1), F32(n2)))
                b = float(_unpolarised(F32(ct), F32(n2), F32(n1)))
                assert abs(a - b) < 2e-6
    assert float(_unpolarised(F32(0.5), F32(1.36), F32(1.36))) < 1e-7


def test_absorbing_fresnel_tends_to_dielectric_and_photocathode_shares():
    for n1 in (1.33, 1.5):
        for nr in (2.0, 2.7, 3.5):
            for c in (1.0, 0.7, 0.3, 0.05):
                a = float(fresnel.unpolarised_absorbing_reflectance(F32(c), F32(n1), F32(nr), F32(1e-5)))
                assert abs(a - np_fresnel(c, n1, nr)) < 2e-5, (n1, nr, c, a)
    p = {
        "nr": F32(2.7),
        "nk": F32(1.5),
        "eff": F32(0.25),
    }
    pr = np.asarray(full(types.PHOTOCATHODE, p, ctx_of(0.8, 1.5, 1.5)))
    R = float(fresnel.unpolarised_absorbing_reflectance(F32(0.8), F32(1.5), F32(2.7), F32(1.5)))
    np.testing.assert_allclose(pr[[SPECULAR, DETECT, ABSORB]], [R, (1 - R) * 0.25, (1 - R) * 0.75], atol=1e-6)


def test_dielectric_rt_is_geant4s_absorption_before_fresnel():
    """rand > R + T absorbed (detected with EFFICIENCY), R < rand <= R + T straight, rand <= R Fresnel."""
    p = {"R": F32(0.6), "T": F32(0.15), "eff": F32(0.3)}
    for c, n1, n2 in ((0.8, 1.5, 1.0), (0.3, 1.5, 1.0), (0.9, 1.0, 1.5)):
        pr = np.asarray(full(types.DIELECTRIC_RT, p, ctx_of(c, n1, n2)), np.float64)
        F = np_fresnel(c, n1, n2)
        np.testing.assert_allclose(
            pr[[REFRACT, SPECULAR, STRAIGHT, DETECT, ABSORB]],
            [0.6 * (1 - F), 0.6 * F, 0.15, 0.25 * 0.3, 0.25 * 0.7],
            atol=2e-6,
        )
        surface = types.no_detection(types.DIELECTRIC_RT)
        q = np.asarray(full(surface, {"R": p["R"], "T": p["T"]}, ctx_of(c, n1, n2)), np.float64)
        np.testing.assert_allclose(q[[DETECT, ABSORB]], [0.0, 0.25], atol=2e-6)  # as a surface: absorbed
        assert not surface.detects and surface.params == ("R", "T")


def test_painted_and_unified_types_are_geant4s_cuts():
    """PAINTED_T: rand > R + T absorbed, rand > R transmitted, else reflected (values summing past 1 included)."""
    N = 1_000_000
    rg = np.random.default_rng(2)
    ctx = dict(lam=F32(400.0), wi=jnp.asarray([0.0, 0.6, -0.8], F32), n=NZ, n1=F32(1.3), n2=F32(1.0))
    for R, T in ((0.5, 0.3), (0.7, 0.6), (0.0, 0.2)):
        pr = np.asarray(full(types.PAINTED_T, {"R": F32(R), "T": F32(T), "f_spec": F32(0.0), "eff": F32(0.0)}, ctx))
        u = rg.random(N)
        freq = np.array([np.mean(~(u > R)), np.mean((u > R) & ~(u > R + T)), np.mean(u > R + T)])
        want = pr[[DIFFUSE, STRAIGHT, ABSORB]]
        z = (freq - want) / np.sqrt(np.maximum(want * (1 - want), 1e-12) / N)
        assert np.max(np.abs(z)) < z_limit(9) and abs(pr.sum() - 1) < 1e-6  # 3 outcomes x 3 settings
    pu = np.asarray(
        full(
            types.UNIFIED_METAL,
            {
                "R": F32(0.8),
                "p_spike": F32(0.1),
                "p_lobe": F32(0.5),
                "p_back": F32(0.1),
                "sigma_alpha": F32(0.2),
                "eff": F32(0.0),
            },
            ctx,
        )
    )
    assert np.allclose(pu[[SPECULAR, GLOSSY, BACKSCATTER, DIFFUSE, ABSORB]], [0.08, 0.4, 0.08, 0.24, 0.2], atol=1e-6)
    assert pu[DETECT] == 0


def test_sigma_alpha_zero_folds_the_lobe_into_the_spike_and_a_metal_transmits_as_geant4():
    """sigma_alpha 0: GetFacetNormal returns the normal, the lobe reflects as the spike (the rest unchanged). A
    TRANSMITTANCE on a metal: rand <= R reflects as the type does, R < rand <= R + T straight, the rest absorbed
    (DoAbsorption, detected with eff); values summing past 1 cut as Geant4's chain."""
    ctx = dict(lam=F32(400.0), wi=jnp.asarray([0.0, 0.6, -0.8], F32), n=NZ, n1=F32(1.3), n2=F32(1.0))
    p = {"R": F32(0.8), "p_spike": F32(0.1), "p_lobe": F32(0.5), "p_back": F32(0.1), "eff": F32(0.25)}
    flat = np.asarray(full(types.UNIFIED_METAL_NORMAL, p, ctx))
    lobe = np.asarray(full(types.UNIFIED_METAL, p | {"sigma_alpha": F32(0.2)}, ctx))
    np.testing.assert_allclose(flat[SPECULAR], lobe[SPECULAR] + lobe[GLOSSY], atol=1e-7)
    np.testing.assert_allclose(
        flat[[BACKSCATTER, DIFFUSE, DETECT, ABSORB]], lobe[[BACKSCATTER, DIFFUSE, DETECT, ABSORB]]
    )
    assert flat[GLOSSY] == 0
    for R, T in ((0.5, 0.3), (0.7, 0.6), (0.0, 0.2)):
        pr = np.asarray(
            full(types.transmitting(types.SPECULAR_METAL), {"R": F32(R), "T": F32(T), "eff": F32(0.4)}, ctx)
        )
        t = min(T, 1.0 - R)
        np.testing.assert_allclose(pr[[SPECULAR, STRAIGHT, DETECT, ABSORB]],
                                   [R, t, 0.4 * (1 - R - t), 0.6 * (1 - R - t)], atol=1e-6)  # fmt: skip
    assert types.transmitting(types.UNIFIED_METAL) == types.transmitting(types.UNIFIED_METAL)  # equal: one type


def test_clip01_cuts_keep_the_linear_derivative_on_the_simplex_faces():
    for r in (0.0, 1.0):
        g = jax.jacfwd(lambda x: jnp.stack(types.chained_cuts(x)))(F32(r))
        np.testing.assert_array_equal(np.asarray(g), [1.0, -1.0])


def test_glisur_types_name_their_facet_path_lobe():
    assert types.GLISUR_METAL.glossy == "glisur"
    assert types.UNIFIED_METAL.glossy == types.REFLECTOR.glossy == "unified"
    w = types.GLISUR_METAL.evaluate({"R": F32(0.8), "polish": F32(0.7), "eff": F32(0.0)}, ctx_of(0.8, 1.33, 1.0))
    assert abs(float(w.sigma_alpha) - 0.3) < 1e-7
    # the dielectric's facet path refracts too: its GLOSSY may end beyond the surface, at width 1 - polish
    d = types.GLISUR_DIELECTRIC
    assert d.glossy == "glisur_dielectric" and d.refracts and not types.GLISUR_METAL.refracts
    w = d.evaluate({"R": F32(0.8), "T": F32(0.1), "polish": F32(0.7), "eff": F32(0.0)}, ctx_of(0.8, 1.33, 1.0))
    assert abs(float(w.sigma_alpha) - 0.3) < 1e-7


def test_fixed_keeps_a_cut_chains_later_shares():
    """A fixed type's support survives Geant4's chained cuts: at p_spike = p_lobe = 0.5 the backscatter and Lambertian
    shares are 0, yet UNIFIED_METAL with eff fixed still backscatters and reflects diffusely."""
    from lumen.surfaces.outcomes import BACKSCATTER, DIFFUSE

    f = types.fixed(types.UNIFIED_METAL, {"eff": 0.0})
    assert BACKSCATTER in f.outcomes and DIFFUSE in f.outcomes
    assert f.outcomes == types.UNIFIED_METAL.outcomes


def test_a_fixed_types_support_is_its_structure_never_a_probe():
    """A fixed type keeps the outcomes its own rule leaves possible (types.SurfaceType.support): a split reflection
    loses the shares a fixed split makes 0, and a type that declares no rule keeps every outcome (nothing is
    evaluated to decide)."""
    assert types.fixed(types.PAINTED, {"f_spec": 1.0}).outcomes == (SPECULAR,)
    assert types.fixed(types.PAINTED, {"f_spec": 0.0}).outcomes == (DIFFUSE,)
    assert types.PAINTED.detects and not types.REFLECTOR.detects
    ref = types.fixed(types.REFLECTOR, {"R": 0.5})
    assert ref.outcomes == (SPECULAR, DIFFUSE)
    assert types.fixed(ref, {"f_spec": 1.0}).outcomes == (SPECULAR,)  # fixed again: one rule
    raw = types.SurfaceType("raw", ("R",), (SPECULAR,))
    assert types.fixed(raw, {"R": 0.5}).outcomes == (SPECULAR,)


def test_a_type_read_for_a_sensor_model_deposits_over_its_normal_absorptance():
    """A detecting type read for a sensor model's QE (types.per_normal) keeps its shares and deposits its detected
    share over A_n (its absorptance at normal incidence, float64 closed form), so the model's QE is the detection at
    normal incidence; a type without eff or DETECT is refused."""
    from lumen.transport import surface_event

    p = {"nr": F32(2.7), "nk": F32(1.5), "eff": F32(1.0)}
    nc = complex(2.7, 1.5)
    a_n = 1 - abs((1.5 - nc) / (1.5 + nc)) ** 2
    pn = types.per_normal(types.PHOTOCATHODE)
    assert pn.per_normal and not types.PHOTOCATHODE.per_normal and pn == types.per_normal(types.PHOTOCATHODE)
    assert float(types.normal_absorptance(types.PHOTOCATHODE, p, ctx_of(0.6, 1.5, 1.0))) == pytest.approx(a_n, 1e-5)
    for c in (1.0, 0.6):
        a = float(types.PHOTOCATHODE.evaluate(p, ctx_of(c, 1.5, 1.0)).detect)
        tables, pk = table_of({"s": SensorEntry(pn, p)})
        ev = surface_event.type_probs(tables, pk, jnp.int32(0), ctx_of(c, 1.5, 1.0))
        assert float(ev.detect * ev.inv_an) == pytest.approx(a / a_n, rel=1e-5)
        np.testing.assert_array_equal(
            np.asarray(ev.probs), np.asarray(types.PHOTOCATHODE.evaluate(p, ctx_of(c, 1.5, 1.0)).probs)
        )
    for bad in (types.PASS, types.no_detection(types.PHOTOCATHODE)):
        with pytest.raises(ValueError, match="detecting type"):
            types.per_normal(bad)


def test_the_surface_types_domains_are_the_declared_ranges_of_the_keys_they_read():
    """The surfaces declare their ranges and defensive roles themselves (they import nothing of the rules); each range
    equals the one the rules declare for the Geant4 key it reads; the shares that can zero an outcome have role
    "zero", GLISUR's polish (it moves the facet cone's edge) "support"."""
    from lumen.rules import keys
    from lumen.surfaces import types

    model = types.G4_REFLECT
    assert {p: model.params[p].range for p in keys.SURFACE_KEYS} == {p: keys.PARAMS[k].range for p, k in
                                                                   keys.SURFACE_KEYS.items()}  # fmt: skip
    assert types.PAINTED_T.range("f_spec") == keys.FRACTION.range and types.PAINTED_T.role("f_spec") == "zero"
    zero = {p for t in types.TYPES for p in t.params if t.role(p) == "zero"}
    assert zero == {"R", "T", "p_spike", "p_lobe", "p_back", "eff", "f_spec"}
    assert {p for t in types.TYPES for p in t.params if t.role(p) == "support"} == {"polish"}


def test_the_types_with_constants_fixed_give_the_probabilities_they_replace():
    """PAINTED, REFLECTOR and SPECULAR_METAL are PAINTED_T and UNIFIED_METAL_NORMAL with constants fixed: bitwise the
    probabilities (float32, the deleted types' own code, below) and the outcomes of the types they replaced."""
    from lumen.surfaces.outcomes import DIFFUSE

    def reflector(p, ctx):  # the deleted REFLECTOR
        R, fs = p["R"], p["f_spec"]
        return outcome_vector(specular=R * fs, diffuse=R * (1.0 - fs), absorb=1.0 - R)

    def painted(p, ctx):  # the deleted PAINTED
        r, a = types.chained_cuts(p["R"])
        return outcome_vector(specular=r * p["f_spec"], diffuse=r * (1.0 - p["f_spec"]), detect=a * p["eff"],
                              absorb=a * (1.0 - p["eff"]))  # fmt: skip

    def specular_metal(p, ctx):  # the deleted SPECULAR_METAL
        r, a = types.chained_cuts(p["R"])
        return outcome_vector(specular=r, detect=a * p["eff"], absorb=a * (1.0 - p["eff"]))

    cases = (
        (types.REFLECTOR, (SPECULAR, DIFFUSE), [{"R": 0.9, "f_spec": 0.3}, {"R": 0.0, "f_spec": 0.0},
                                                         {"R": 1.0, "f_spec": 1.0}, {"R": 0.37, "f_spec": 0.0}],
         reflector),
        (types.PAINTED, (SPECULAR, DIFFUSE), [{"R": 0.6, "f_spec": 1.0, "eff": 0.0},
                                                               {"R": 0.6, "f_spec": 0.0, "eff": 0.3}], painted),
        (types.SPECULAR_METAL, (SPECULAR,), [{"R": 0.9, "eff": 0.0}, {"R": 1.5, "eff": 0.5},
                                                             {"R": -0.2, "eff": 1.0}], specular_metal),
    )  # fmt: skip
    ctx = ctx_of(0.8, 1.33, 1.0)
    for kind, outs, sets, old in cases:
        assert kind.outcomes == outs and kind not in types.TYPES
        for p0 in sets:
            p = {k: F32(v) for k, v in p0.items()}
            ev, want = kind.evaluate(p, ctx), np.asarray(old(p, ctx))
            got = np.asarray(ev.probs)
            assert np.array_equal(got, want[:DETECT])
            assert np.array_equal(np.asarray(ev.detect), want[DETECT]), (kind.name, p0, got, want)
