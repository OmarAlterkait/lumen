"""Closed-form physics through the navigator and the step (the physics tier), on the toys of ``toys.py``.

- a point source at the centre of a specular sensor sphere: Q/N = (1 - R) e^(-a/L) / (1 - R e^(-2a/L)), each photon
  deterministic, so value and derivatives (R, L) are exact to float32;
- the same sphere Lambertian: Q/N = (1 - R) e^(-a/L) / (1 - R E), E = 2 (1 - (1 + b) e^(-b)) / b^2, b = 2a/L;
- an enclosing sensor with Rayleigh scattering and no absorption: Q/N = 1 for any R and scattering length, so both
  derivatives are 0 (in expectation: the detection is expected over the free path, so each photon's charge varies);
- an opaque cube before the sensor, as a box and as a tessellated mesh: the same charge, the shadow's solid angle;
- border directions: a detecting border A -> B counts light from A into B, none from B into A;
- a skin between two touching placements of one LV (one class: the class table's diagonal) acts as in Geant4 10.7.4;
- g4ref's overlap toys (coincident and partly overlapping siblings of one solid, owned by the first placed)
  against Geant4 10.7.4's detected fraction and mean arrival time on their overlap-free GDML;
Statistical comparisons over K independent seeds (``toys.simulate(..., seeds=K)``) use the suites' threshold for a
Student score with K - 1 degrees of freedom (``stats.z_limit``): the mean over the seeds against its standard error
from them.
"""

import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import toys
from toys import DATA, EXACT, A, simulate

from lumen.compare import photons
from lumen.rules.navigation import BAND as BAND_REL
from lumen.tools import stats
from lumen.transport import state
from lumen.transport import step as steps

CONTROLS = steps.Controls(max_distance=10.0, roulette=EXACT)  # the overlap toys' 10 m reach

F32 = jnp.float32
LAM = 400.0


def rows_at(n, x=(0.0, 0.0, 0.0), seed=0):
    return photons.isotropic(jnp.tile(jnp.asarray(x, F32), (n, 1)), jnp.full(n, LAM, F32), jax.random.PRNGKey(seed))


def _packed_families(optics, p0) -> list:
    """Return [(name, keys, row, mode, scale)] over the packed parameters: each medium's lengths and re-emission
    constants (multiplied), each surface row's parameters (a probability inside its domain added at
    min(v - lo, hi - v); at an edge one-sided, inward; others multiplied). The toys' low-level runs perturb the packed
    parameters directly (``apply``)."""
    out, surf = [], []
    for path, v in jax.tree_util.tree_flatten_with_path(p0)[0]:
        keys = tuple(getattr(q, "key", getattr(q, "idx", None)) for q in path)
        a = np.asarray(v, np.float64)
        if keys[-1] in ("energy", "const") or not np.issubdtype(a.dtype, np.floating):
            continue
        if keys[0] == "media":
            fin = np.isfinite(a) & (a > 0)
            if "index" not in keys and any(s in keys for s in ("absorption", "length", "yield", "tau", "eta")) and \
                    fin.any() and not np.all(a[fin] >= 1e6):  # fmt: skip
                out.append((keys[1].split("0x")[0] + ":" + "/".join(str(k) for k in keys[2:] if k != "value"), keys,
                            None, "mul", None))  # fmt: skip
        elif keys[0] == "surfaces":  # each row that reads the column as a parameter, by (type, parameter, row)
            for r, ((st, origin), reads) in enumerate(zip(optics.rows, optics.table.reads, strict=True)):
                if keys[1] not in reads:
                    continue
                q = reads[keys[1]]
                fam = _row_family(
                    f"{st.name}.{q}[{str(origin)[:40]}]", keys, r, a[r], st.range(q) if st.role(q) == "zero" else None
                )
                if fam is not None:
                    surf.append(((st.name, q, r), fam))
    return out + [f for _, f in sorted(surf, key=lambda x: x[0])]


def _row_family(nm, keys, r, ar, dom):
    """Return a surface row's family: inside its domain added at min(v - lo, hi - v), at an edge one-sided inward,
    without a domain multiplied; None where it has none (a probability with no room, a zero value)."""
    if dom is None:
        return (nm, keys, r, "mul", None) if np.any(ar != 0) else None
    lo, hi = dom
    c = np.minimum(ar - lo, hi - ar)
    if np.all(c > 0) or not (np.all(ar <= lo) or np.all(ar >= hi)):
        return (nm, keys, r, "add", np.clip(c, 0.0, None)) if np.any(c > 0) else None
    return (nm, keys, r, "edge", np.ones_like(ar) if np.all(ar <= lo) else -np.ones_like(ar))


def _get(tree, keys):
    for k in keys:
        tree = tree[k]
    return tree


def _set(tree, keys, v):
    return v if not keys else {**tree, keys[0]: _set(tree[keys[0]], keys[1:], v)}


def apply(p0, fams, th):
    """Return the packed parameters at th (one value per family, traced)."""
    p = p0
    for j, (_, keys, row, mode, scale) in enumerate(fams):
        v = _get(p, keys)
        dv = v * th[j] if mode == "mul" else jnp.asarray(scale, v.dtype) * th[j]
        p = _set(p, keys, v + dv if row is None else v.at[row].add(dv if mode != "mul" else dv[row]))
    return p


def families(s, *names):
    """The packed parameters and the families whose names contain ``names`` (in that order)."""
    p0 = photons.packed(s.det)
    fams = _packed_families(s.det.parts.optics, p0)
    return p0, [next(f for f in fams if nm in f[0]) for nm in names]


def charge(s, p0, fams, rows, seeds=1, **kw):
    """Return th -> per-seed charge per photon (the families' th as in the gradient check; the rows split over the
    seeds)."""
    n = rows.x.shape[0] / seeds

    def q(th):
        p, pr = apply(p0, fams, th), apply(p0, fams, jax.lax.stop_gradient(th))
        return simulate(s, p, rows, reference=pr, seeds=seeds, **kw)["charge:0"].reshape(seeds, -1).sum(1) / n

    return q


def specular(r, length):
    return (1 - r) * jnp.exp(-A / length) / (1 - r * jnp.exp(-2 * A / length))


def lambert(r, length):
    b = 2 * A / length
    return (1 - r) * jnp.exp(-A / length) / (1 - r * 2 * (1 - (1 + b) * jnp.exp(-b)) / b**2)


def test_specular_sphere_closed_form_and_exact_derivatives(toy):
    s = toy("sphere", r=0.3, absorption=2.0)
    p0, fams = families(s, ".R[", "Liquid:absorption")
    q = charge(s, p0, fams, rows_at(1024))
    th = jnp.zeros(2, F32)
    v = float(q(th)[0])
    d = [float(jax.jvp(lambda t: q(t)[0], (th,), (th.at[j].set(1.0),))[1]) for j in range(2)]
    dr, dl = (float(x) for x in jax.grad(specular, argnums=(0, 1))(0.3, 2.0))
    assert v == pytest.approx(float(specular(0.3, 2.0)), rel=3e-5)
    assert d[0] == pytest.approx(0.3 * dr, rel=1e-4)  # the family adds min(R, 1 - R) th to R
    assert d[1] == pytest.approx(2.0 * dl, rel=1e-4)  # ... and multiplies L by 1 + th


def test_lambertian_sphere_closed_form_and_derivatives(toy):
    s = toy("sphere", r=0.5, finish="ground", sigma_alpha=0.1, absorption=2.0)
    p0, fams = families(s, ".R[", "Liquid:absorption")
    k = 32
    q = charge(s, p0, fams, rows_at(k * 4096), seeds=k)
    th = jnp.zeros(2, F32)
    v = np.asarray(q(th), np.float64)
    dr, dl = (float(x) for x in jax.grad(lambert, argnums=(0, 1))(0.5, 2.0))
    assert abs((v.mean() - float(lambert(0.5, 2.0))) / stats.sem(v)) < stats.z_limit(df=k - 1)
    for j, want in ((0, 0.5 * dr), (1, 2.0 * dl)):
        dv = np.asarray(jax.jvp(q, (th,), (th.at[j].set(1.0),))[1], np.float64)
        assert abs((dv.mean() - want) / stats.sem(dv)) < stats.z_limit(2, k - 1), (j, dv.mean(), want)


def test_an_enclosing_sensor_detects_every_photon_whatever_the_scattering(toy):
    s = toy("sphere", r=0.5, absorption=1e9, rayleigh=0.5)
    p0, fams = families(s, ".R[", "Liquid:rayleigh")
    k = 16
    q = charge(s, p0, fams, rows_at(k * 4096), seeds=k)
    th = jnp.zeros(2, F32)
    v = np.asarray(q(th), np.float64)  # expected detection over the free path: unbiased, not one per photon
    assert abs((v.mean() - 1.0) / stats.sem(v)) < stats.z_limit(df=k - 1)
    for j in range(2):
        dv = np.asarray(jax.jvp(q, (th,), (th.at[j].set(1.0),))[1], np.float64)
        assert abs(dv.mean() / stats.sem(dv)) < stats.z_limit(2, k - 1), (j, dv.mean())


def test_a_user_groupvel_times_the_photons(toy):
    """Geant4 times a photon by its material's GROUPVEL table when the user gives one (G4Track): from the centre of
    the sphere every photon arrives at A / v_g, and the table is a parameter (``materials/Liquid/GROUPVEL``)."""
    v = 150.0  # mm / ns, not RINDEX's 225
    s = toy("sphere", r=0.0, absorption=1e9, groupvel=v)
    assert "materials/Liquid/GROUPVEL" in s.det.params()
    out = simulate(s, photons.packed(s.det), rows_at(4096))
    assert float(out["Tw"].sum() / out["charge:0"].sum()) == pytest.approx(A * 1e3 / v, rel=1e-5)


def test_a_zero_length_step_out_of_a_medium_without_rindex_goes_on(toy):
    """A photon left at the face of a medium without RINDEX (a reflection relocated it there) and heading out: its
    zero-length step is Geant4's StepTooSmall, which G4OpBoundaryProcess returns before its NoRINDEX test, so it goes
    on into the glass beyond, never lost. One step from a photon placed in the steel, on the face it shares with the
    glass, heading into the glass."""
    s = toy("junction")
    nav = steps.geometry(s.model, photons.operands(s, photons.packed(s.det))).nav
    v_steel, v_glass = (int(nav.locate(jnp.asarray(x, F32))) for x in ((0.11, 0.0, 0.0), (0.05, 0.0, 0.0)))
    cls = nav.dev()["cls"]
    row = toys.pencil((0.1, 0.01, 0.02), (-1.0, 0.0, 0.0), LAM, 1)
    row = jax.tree_util.tree_map(lambda a: a[0] if a is not None else None, row)
    ph = state.start(row, v_steel, cls[v_steel])
    ops = photons.operands(s, photons.packed(s.det))
    ph2, dep = jax.jit(lambda p: steps.step(s.model, ops, p, jax.random.PRNGKey(1)))(ph)
    assert bool(ph2.alive) and float(dep.lost) == 0.0
    assert int(ph2.cls) == int(cls[v_glass]) and int(ph2.vol) == v_glass
    assert float(jnp.linalg.norm(ph2.x - ph.x)) < 1e-6 and float(ph2.d[0]) == -1.0


def test_an_opaque_mesh_casts_the_shadow_of_the_same_box(toy):
    half, z = 0.1, 0.5
    omega = 4 * math.asin((2 * half) ** 2 / ((2 * half) ** 2 + 4 * (z - half) ** 2))
    want = math.exp(-A / 1e3) * (1 - omega / (4 * math.pi))
    k, out = 16, {}
    for kind in ("box", "mesh"):
        s = toy("sphere", r=0.0, absorption=1e3, occluder=(kind, half, z))
        out[kind] = np.asarray(charge(s, *families(s, "Liquid:absorption"), rows_at(k * 16384), seeds=k)(
            jnp.zeros(1, F32)), np.float64)  # fmt: skip
    assert np.max(np.abs(out["box"] - out["mesh"])) < 1e-4  # the same photons meet the same face
    assert abs((out["box"].mean() - want) / stats.sem(out["box"])) < stats.z_limit(df=k - 1)


def test_a_border_surface_counts_in_its_own_direction_only(toy):
    s = toy("border_pair")
    p0 = photons.packed(s.det)
    into_b = toys.pencil((-0.25, 0.0, 0.0), (1.0, 0.0, 0.0), LAM, 256)
    into_a = toys.pencil((0.25, 0.0, 0.0), (-1.0, 0.0, 0.0), LAM, 256)
    assert float(simulate(s, p0, into_b)["charge:0"].sum()) / 256 == pytest.approx(math.exp(-0.25 / 1e3), rel=1e-5)
    assert float(simulate(s, p0, into_a)["charge:0"].sum()) == 0.0


def self_pair_fraction(r, t, length):
    """Q/N of ``toys.self_pair``'s pencil from A's centre along +x: a chain over the three faces x = -0.5, 0, 0.5 m
    (each transmits t, reflects r, absorbs the rest), the states (A, +x at 0), (A, -x at -0.5), (B, +x at 0.5),
    (B, -x at 0), 0.5 m of absorption length ``length`` between faces. Stepping over x = 0 would give
    e^(-0.75/L) t / (1 - r^2 e^(-2/L))."""
    e = math.exp(-0.5 / length)
    m = np.array([[1, -r * e, -t * e, 0], [-r * e, 1, 0, 0], [0, 0, 1, -r * e], [0, -t * e, -r * e, 1]])
    return math.exp(-0.25 / length) * float(np.linalg.solve(m, [0.0, 0.0, t, 0.0])[0])


def test_a_skin_between_two_volumes_of_one_class_acts_as_in_geant4(toy):
    """Two touching placements A | B of one LV with a frontpainted skin (R 0.3, T 0.5) are one class: leaving A into B
    Geant4 applies the skin (pre's). The crossing is the class table's diagonal and the navigator stops there.
    Geant4 10.7.4 (g4ref, the same GDML, 1e6 photons): 0.309385 +- 0.00046; the chain: 0.31001; stepping over the
    shared face: 0.549."""
    s = toy("self_pair")
    c = s.det.parts.optics.class_of
    assert c["WorldLV/pvA/BoxLV"] == c["WorldLV/pvB/BoxLV"]
    k, n = 16, 1 << 16
    rows = toys.pencil((-0.25, 0.0, 0.0), (1.0, 0.0, 0.0), LAM, n)
    out = simulate(s, photons.packed(s.det), rows, seeds=k, controls=steps.Controls(max_distance=10.0, roulette=EXACT))
    f = np.asarray(out["charge:0"], np.float64).reshape(k, -1).sum(1) / (n / k)
    want = self_pair_fraction(0.3, 0.5, 1e3)
    assert abs((f.mean() - want) / stats.sem(f)) < stats.z_limit(df=k - 1), (f.mean(), want)
    assert abs((f.mean() - 0.309385) / np.hypot(stats.sem(f), 0.00046)) < stats.z_limit(df=k - 1), f.mean()


@pytest.mark.parametrize("name", ["coinT_AB", "partT_BA", "vx3_ABC"])
def test_an_overlap_toy_through_the_step_matches_geant4(toy, name):
    g4 = json.loads((DATA / "geant4.json").read_text())[name]
    s = toy("overlap", name)
    k, n = 16, 1 << 18
    rows = toys.pencil((-0.8, 0.0, 0.0), (1.0, 0.0, 0.0), g4["wavelength_nm"], n)
    out = simulate(s, photons.packed(s.det), rows, seeds=k, controls=steps.Controls(max_distance=10.0, roulette=EXACT))
    q = np.asarray(out["charge:0"], np.float64).reshape(k, -1).sum(1)
    tw = np.asarray(out["Tw"], np.float64).reshape(k, -1).sum(1)
    f = q / (n / k)
    t = tw / q

    def agree(v, ref, err):  # no Monte Carlo error on either side (every photon the same path): float32's resolution
        return abs(v - ref) <= BAND_REL * abs(ref) if err == 0 else abs(v - ref) / err < stats.z_limit(2, k - 1)

    assert agree(f.mean(), g4["fraction"], np.hypot(stats.sem(f), g4["fraction_error"]))
    assert agree(tw.sum() / q.sum(), g4["mean_time_ns"], np.hypot(stats.sem(t), g4["mean_time_error_ns"]))
