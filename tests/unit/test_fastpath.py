"""The diffusion jump's data and step code (``StepConfig.jump``): the clearance bound with a mesh in a jump medium,
the walk inside the clearance ball, and the tangents it carries."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from nav_toys import defn, make_scene, scene_mesh

from lumen.geometry import clearance, navigate, navigator_build, solids
from lumen.media import phase
from lumen.transport import fastpath

LAM = 0.01  # the jump medium's shortest Rayleigh length (m): h = 2 x 4 x LAM / sqrt(3) = 4.6 cm


def _point_triangle(p, tri):
    """Return the distance from points p (N, 3) to a triangle (3, 3) (Ericson, Real-Time Collision Detection 5.1.5)."""
    a, b, c = tri
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ap @ ab, ap @ ac
    bp, cp = p - b, p - c
    d3, d4, d5, d6 = bp @ ab, bp @ ac, cp @ ab, cp @ ac
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    den = va + vb + vc
    v, w = vb / den, vc / den
    q = a + v[:, None] * ab + w[:, None] * ac  # the face region
    for cond, qq in (
        ((d1 <= 0) & (d2 <= 0), np.broadcast_to(a, p.shape)),
        ((d3 >= 0) & (d4 <= d3), np.broadcast_to(b, p.shape)),
        ((d6 >= 0) & (d5 <= d6), np.broadcast_to(c, p.shape)),
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + (d1 / (d1 - d3))[:, None] * ab),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + (d2 / (d2 - d6))[:, None] * ac),
        ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), b + ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None] * (c - b)),
    ):
        q = np.where(cond[:, None], qq, q)
    return np.linalg.norm(p - q, axis=1)


def _box_surface(p, lo, hi):
    """Return the distance from points p to the surface of the box [lo, hi]."""
    out = np.maximum(np.maximum(lo - p, p - hi), 0.0)
    dout = np.linalg.norm(out, axis=1)
    din = np.min(np.minimum(p - lo, hi - p), 1)
    return np.where(dout > 0, dout, din)


def _toy():
    """scene_mesh (a 2 m world box, medium 1, holding an opaque tetrahedron mesh at (0.3, 0.1, 0) and a 0.2 m box at
    (-0.4, 0, 0)) with medium 1 a Rayleigh-only medium (the jump's), and the surfaces' true distance."""
    sc = scene_mesh()
    nav = navigate.Navigator.from_scene(sc)
    ray = SimpleNamespace(channels=[SimpleNamespace(name="rayleigh", emission=None, lobe=phase.RAYLEIGH)])
    other = SimpleNamespace(channels=[])
    optics = SimpleNamespace(medium_of=np.arange(8), media=[ray if m == 1 else other for m in range(8)],
                             media_names=[f"m{m}" for m in range(8)])  # fmt: skip
    library = SimpleNamespace(media={"m1": SimpleNamespace(params={"rayleigh": {"length": {"value": LAM}}})})
    tris = np.asarray(sc.definitions["tet"].member.solid.tris, np.float64) + np.array([0.3, 0.1, 0.0])

    def dist(p):
        d = np.min([_point_triangle(p, t) for t in tris], 0)
        d = np.minimum(d, _box_surface(p, np.array([-0.5, -0.1, -0.1]), np.array([-0.3, 0.1, 0.1])))
        return np.minimum(d, _box_surface(p, -np.ones(3), np.ones(3)))

    return nav, (optics, library), tris, dist


def test_the_clearance_bound_counts_a_mesh_in_a_jump_medium():
    """Every mesh's world box enters the bound (0 inside it, the box distance outside): at every node of the jump
    medium's grid and at random points through the device lookup the bound is below the true distance to every
    surface, the mesh's triangles included; nodes in the mesh's box get 0; and the mesh is what bounds the nodes
    nearest to it (their bound is below their distance to every other surface)."""
    nav, (optics, library), tris, dist = _toy()
    lo, hi = clearance.mesh_boxes(nav)
    assert (
        lo.shape == (1, 3)
        and np.all(lo[0] <= tris.reshape(-1, 3).min(0))
        and np.all(hi[0] >= tris.reshape(-1, 3).max(0))
    )
    jump = fastpath.prepare(SimpleNamespace(nav=nav), optics, library)
    assert jump.media == (1,), jump.info
    G = jump.grids[0]
    axes = [G["lo"][a] + G["h"] * np.arange(G["n"][a]) for a in range(3)]
    X = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    clr = np.asarray(G["clear"], np.float64).reshape(-1)
    d = dist(X)
    assert np.all(clr <= d), float(np.max(clr - d))
    in_mesh = np.all((X >= lo[0]) & (X <= hi[0]), 1)
    assert in_mesh.any() and np.all(clr[in_mesh] == 0.0)
    near = np.min([_point_triangle(X, t) for t in tris], 0)
    rest = np.minimum(_box_surface(X, np.array([-0.5, -0.1, -0.1]), np.array([-0.3, 0.1, 0.1])),
                      _box_surface(X, -np.ones(3), np.ones(3)))  # fmt: skip
    by_mesh = ~in_mesh & (near < rest - G["h"])
    assert by_mesh.sum() > 10 and np.all(clr[by_mesh] <= near[by_mesh])
    # the device's lookup at random points (the nearest node's bound less the distance to it)
    rng = np.random.default_rng(3)
    P = rng.uniform(-0.98, 0.98, (4096, 3)).astype(np.float32)
    Gd = fastpath.operands(jump)["grids"][0]
    ins, c = jax.vmap(lambda x: fastpath._clear_at(Gd, x))(jnp.asarray(P))
    c = np.where(np.asarray(ins), np.asarray(c, np.float64), -np.inf)
    assert np.all(c <= dist(P.astype(np.float64)))


def _diffuse_case(clear=5.0):
    """A walk that acts: a 21^3 grid of clearance ``clear`` m, mu_ref = 10 / m (a ball of 50 scattering lengths), a
    linear state."""
    G = dict(lo=jnp.full(3, -10.0, jnp.float32), h=jnp.float32(1.0), n=jnp.full(3, 21, jnp.int32),
             clear=jnp.full((21, 21, 21), clear, jnp.float32))  # fmt: skip
    model = SimpleNamespace(jump=SimpleNamespace(media=(1,)))
    ops = SimpleNamespace(tables={"diff": dict(grids=(G,))})
    ph = SimpleNamespace(alive=jnp.bool_(True), vertex=jnp.int32(3))
    sc = SimpleNamespace(medium_idx=jnp.int32(1), mu_r=jnp.float32(10.0), mu=jnp.float32(10.0),
                         mu_abs=jnp.float32(0.01), v_g=jnp.float32(0.2),
                         rand={"jkey": jax.random.fold_in(jax.random.key_data(jax.random.PRNGKey(5)), 3)})  # fmt: skip
    b = SimpleNamespace(ei=jnp.zeros(3, jnp.float32), er=jnp.array([1.0, 0, 0], jnp.float32),
                        wo=jnp.array([0, 0, 1.0], jnp.float32))  # fmt: skip
    return model, ops, ph, sc, b


def test_the_walk_leaves_the_ball_and_carries_the_position_tangent():
    """The walk ends on the ball's sphere (or inside after its budget), with a unit direction perpendicular to its
    linear state; x' = x + stop_gradient(offset): a tangent of the photon's position passes through unchanged, the
    weight's tangent is the score (n / mu - L) dmu; a non-linear state or no clearance gives the identity."""
    model, ops, ph, sc, b = _diffuse_case()
    x = jnp.array([0.1, -0.2, 0.3], jnp.float32)
    dx = jnp.array([1.0, 2.0, -3.0], jnp.float32)
    (xn, dn, ern, _, wj, dt, _), (tx, _, _, _, tw, tt, _) = jax.jvp(
        lambda x_: fastpath.diffuse(model, ops, ph, sc, x_, b), (x,), (dx,)
    )
    r = float(jnp.linalg.norm(xn - x))
    assert 1e-3 < r <= 5.0 * (1 + 1e-5)
    assert abs(float(jnp.linalg.norm(dn)) - 1.0) < 1e-5 and abs(float(jnp.sum(dn * ern))) < 1e-5
    np.testing.assert_array_equal(np.asarray(tx), np.asarray(dx))
    assert float(tw) == 0.0 and float(tt) == 0.0 and float(dt) > 0.0
    L = float(dt) * 0.2

    def w(mu):
        return fastpath.diffuse(model, ops, ph, sc._replace(mu=mu) if hasattr(sc, "_replace") else
                                SimpleNamespace(**{**vars(sc), "mu": mu}), x, b)[4]  # fmt: skip

    _, tw_mu = jax.jvp(w, (jnp.float32(10.0),), (jnp.float32(1.0),))
    n_minus = float(tw_mu) / float(wj) + L  # n / mu
    assert abs(n_minus * 10.0 - round(n_minus * 10.0)) < 1e-2  # n is a whole number of scatterings
    b2 = SimpleNamespace(ei=jnp.array([0, 1e-7, 0], jnp.float32), er=b.er, wo=b.wo)  # not linear: no walk
    xn2, *_ = fastpath.diffuse(model, ops, ph, sc, x, b2)
    np.testing.assert_array_equal(np.asarray(xn2), np.asarray(x))
    model, ops, ph, sc, b = _diffuse_case(clear=0.0)  # no clearance: no walk
    xn3, _, _, _, wj3, dt3, _ = fastpath.diffuse(model, ops, ph, sc, x, b)
    np.testing.assert_array_equal(np.asarray(xn3), np.asarray(x))
    assert float(wj3) == 1.0 and float(dt3) == 0.0


def test_the_walk_carries_the_start_state_s_tangent_as_the_plain_walk():
    """A tangent of the start state (er, ei) reaches the exit state and the weight as through the plain walk with the
    same draws: each scattering drawn at the held state, its density log p(k | e) and the projection taken at the live
    state (exact's lp and _state_after); the walk's carry gives the same tangents (a replay of its uniforms here)."""
    from lumen._core import rng
    from lumen.surfaces import polar

    model, ops, ph, sc, b = _diffuse_case(clear=0.6)  # a short walk: a few scatterings
    x = jnp.array([0.1, -0.2, 0.3], jnp.float32)
    der = jnp.array([0.0, 0.3, -0.2], jnp.float32)
    dei = jnp.array([0.0, 0.5, 0.1], jnp.float32)

    def walk(er, ei):
        bb = SimpleNamespace(ei=ei, er=er, wo=b.wo)
        return fastpath.diffuse(model, ops, ph, sc, x, bb)

    (xn, dn, ern, ein, wj, dt, _), (_, _, t_er, t_ei, t_w, _, _) = jax.jvp(walk, (b.er, b.ei), (der, dei))
    z = jnp.zeros(3, jnp.float32)
    sg = jax.lax.stop_gradient

    def plain(er, ei):  # the replay: the walk's uniforms, its exit test, the live state's densities and projections
        y, d, e, f, j = z, b.wo, er, ei, 0
        c, mu = 0.6 - float(jnp.linalg.norm(x - jnp.round(x))), 10.0
        logw = 0.0
        for _ in range(int(np.ceil((c * mu) ** 2))):  # the walk's budget of scatterings
            u = rng.uniforms(jax.random.fold_in(sc.rand["jkey"], j), 4)
            s = -jnp.log(u[0]) / mu
            by = jnp.sum(y * d)
            t = -by + jnp.sqrt(jnp.maximum(by * by - (jnp.sum(y * y) - c * c), 0.0))
            if bool(s >= t):
                y = y + t * d
                break
            k = sg(phase.rayleigh_sample(sg(e), z, u[1:4]))
            logw = logw + phase.rayleigh_logpdf(e, f, k)
            e, f = polar.project(e, f, k)
            y, d, j = y + s * d, k, j + 1
        return e, f, logw

    (e_p, f_p, lw), (te_p, tf_p, tlw) = jax.jvp(plain, (b.er, b.ei), (der, dei))
    np.testing.assert_allclose(np.asarray(ern), np.asarray(e_p), atol=1e-5)
    np.testing.assert_allclose(np.asarray(t_er), np.asarray(te_p), atol=2e-4)
    np.testing.assert_allclose(np.asarray(t_ei), np.asarray(tf_p), atol=2e-4)
    np.testing.assert_allclose(float(t_w) / float(wj), float(tlw), atol=2e-4)
    assert float(jnp.abs(tlw)) > 1e-3  # the case has a score to carry


def _transparent_toy():
    """A 2 m world box (medium 1) holding C, a 0.4 m box of medium 1 at the centre that holds W (a 0.1 m box of
    medium 4 at (0.1, 0, 0)), and P (a 0.2 m box of medium 1 at (0.6, 0, 0), kept: a sensor's) holding Q (0.1 m,
    medium 5). Given the classes, C is transparent (dissolved); P is not."""
    box = solids.g4_box
    sc = make_scene({
        "world": defn("world", box(2.0, 2.0, 2.0), 1, [("C", np.eye(3), (0.0, 0, 0)), ("P", np.eye(3), (0.6, 0, 0))]),
        "C": defn("C", box(0.4, 0.4, 0.4), 1, [("W", np.eye(3), (0.1, 0, 0))]),
        "W": defn("W", box(0.1, 0.1, 0.1), 4),
        "P": defn("P", box(0.2, 0.2, 0.2), 1, [("Q", np.eye(3), (0.0, 0, 0))]),
        "Q": defn("Q", box(0.1, 0.1, 0.1), 5),
    })  # fmt: skip
    plain = navigate.Navigator.from_scene(sc)
    cls = np.asarray(plain.host["cls"])
    p = plain.defs.index("P")
    cl = navigator_build.Classes(tuple(cls[: plain.n].tolist()), (), int(cls[plain.n]), (p,), tuple(range(8)))
    arr = navigator_build.build(sc, classes=cl)
    return plain, navigate.Navigator(arr), arr


def test_the_clearance_bound_counts_only_the_volumes_the_navigator_can_be_in():
    """A transparent volume (dissolved by the navigator: a photon is never in one, and crossing its faces changes
    neither its volume nor its class) has no term in the bound: with the classes the bound is below the true distance
    to every other surface (the world, W inside C, P and Q) and above the distance to C's faces where those are
    nearest; without them (nothing dissolved) it stays below the distance to C's faces too."""
    plain, clear, arr = _transparent_toy()
    c = clear.defs.index("C")
    assert arr.stats["transparent"] == 1 and c not in navigator_build.navigable(arr)
    assert len(navigator_build.navigable(plain.arrays)) == plain.n and len(navigator_build.navigable(arr)) == arr.n - 1
    ray = SimpleNamespace(channels=[SimpleNamespace(name="rayleigh", emission=None, lobe=phase.RAYLEIGH)])
    other = SimpleNamespace(channels=[])
    optics = SimpleNamespace(medium_of=np.arange(8), media=[ray if m == 1 else other for m in range(8)],
                             media_names=[f"m{m}" for m in range(8)])  # fmt: skip
    library = SimpleNamespace(media={"m1": SimpleNamespace(params={"rayleigh": {"length": {"value": LAM}}})})
    boxes = {"world": (-np.ones(3), np.ones(3)), "W": (np.array([0.05, -0.05, -0.05]), np.array([0.15, 0.05, 0.05])),
             "P": (np.array([0.5, -0.1, -0.1]), np.array([0.7, 0.1, 0.1])),
             "Q": (np.array([0.55, -0.05, -0.05]), np.array([0.65, 0.05, 0.05])),
             "C": (np.full(3, -0.2), np.full(3, 0.2))}  # fmt: skip
    for nav, skipped in ((clear, True), (plain, False)):
        jump = fastpath.prepare(SimpleNamespace(nav=nav), optics, library)
        assert jump.media == (1,), jump.info
        G = jump.grids[0]
        axes = [G["lo"][a] + G["h"] * np.arange(G["n"][a]) for a in range(3)]
        X = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        clr = np.asarray(G["clear"], np.float64).reshape(-1)
        d = {k: _box_surface(X, lo, hi) for k, (lo, hi) in boxes.items()}
        rest = np.min([d[k] for k in ("world", "W", "P", "Q")], 0)
        assert np.all(clr <= rest), float(np.max(clr - rest))
        by_c = d["C"] < rest - 2.0 * G["h"]  # C's faces are the nearest by more than the lookup's slack
        assert by_c.sum() > 10
        if skipped:
            assert np.any(clr[by_c] > d["C"][by_c] + G["h"])
        else:
            assert np.all(clr <= d["C"])
    eager = fastpath.prepare(SimpleNamespace(nav=clear), optics, library).grids[0]["clear"]
    traced = jax.jit(lambda: fastpath.prepare(SimpleNamespace(nav=clear), optics, library).grids[0]["clear"])()
    np.testing.assert_array_equal(np.asarray(traced), np.asarray(eager))  # a first fast run may be under jax.jit


def test_the_jump_is_on_by_default_and_a_probe_run_compiles_without_it():
    """A run takes the diffusion jump by default (truth rows included: the walk is the photon's own path) and records
    ``("speed", "fast")``; a probe diagnostic sees every step, so it compiles the plain walk and records
    "exact (diagnostics)". There is no public speed argument: the private ``_jump`` serves lumen's own validation."""
    import inspect

    import lumen
    from lumen.transport.step import StepConfig

    sig = inspect.signature(lumen.simulate).parameters
    assert "speed" not in sig and sig["_jump"].default is True
    assert StepConfig().jump is False  # the reference code, the plain Rayleigh walk


def test_the_grid_reads_the_rayleigh_length_as_built():
    """A medium's shortest Rayleigh length is its built channel's (the library's): a table derived from RINDEX
    (G4OpRayleigh's water, no RAYLEIGH key) as a GDML's RAYLEIGH table, so the fast path acts in both."""
    table = {"energy": np.array([2.0, 3.0]), "value": np.array([60.0, 20.0])}  # RINDEX's knots, derived values
    optics = SimpleNamespace(media=[SimpleNamespace(channels=[SimpleNamespace(name="rayleigh")])], media_names=["W"])
    library = SimpleNamespace(media={"W": SimpleNamespace(params={"rayleigh": {"length": table}})})
    assert fastpath._shortest_rayleigh(optics, library, 0) == 20.0
