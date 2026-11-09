"""Small scenes for the navigator tests, built with the scene builder (metres), and the batched-call helpers.

Ported from the baseline's tests/test_navigator.py (its scenes, there in Geant4's mm); the walk against the float64
oracle is the baseline's ``_walk_check``.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from lumen.build import scene, solids
from lumen.geometry import navigate
from lumen.validate import oracle

F32, I32 = jnp.float32, jnp.int32
T_ORC = 1e-9  # the oracle's start (its merge length is 1e-10)
BAND = 8 * 2.0**-23  # the float32 band per metre of coordinate


def rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def rx(deg):
    a = np.radians(deg)
    return np.array([[1.0, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def g4(op, a, b, R=None, t=(0.0, 0.0, 0.0)):
    """A GDML-level boolean (``g4bool``: split_unions may decompose it)."""
    n = solids.Op(op, a, solids.Xf(np.eye(3) if R is None else R, np.asarray(t, float), b))
    n.g4bool = True
    return n


def make_scene(defs, root="world", outside=0, hole_fillers=()):
    return scene.Scene(dict(defs), root, scene.Label(outside, -1, "outside"), [], list(hole_fillers))


def defn(name, solid, med, pls=(), sensor=-1):
    """A definition of one solid (or none) placing (definition, R, t) daughters named '<definition>_<k>'."""
    members = [scene.Member(solid, scene.Label(med, sensor, name), 0)] if solid is not None else []
    return scene.Definition(name, members, [scene.Placement(p[0], p[1], np.asarray(p[2], float), name=f"{p[0]}_{k}")
                                            for k, p in enumerate(pls)])  # fmt: skip


def mesh_def(name, tris, med):
    return scene.Definition(name, [scene.Member(scene.Mesh(tris), scene.Label(med, -1, name), 0)], [])


def scene_nest():
    """boxes and spheres nested 3 deep, off centre, rotated; a daughter flush with its mother's face; siblings sharing
    a face (coincident faces)"""
    return make_scene({
        "world": defn("world", solids.g4_box(2.0, 2.0, 2.0), 1, [("tank", rz(20), (0.1, 0, 0)),
                                                              ("left", np.eye(3), (-0.7, 0, 0)),
                                                              ("right", np.eye(3), (-0.5, 0, 0))]),
        "tank": defn("tank", solids.g4_box(0.8, 0.9, 1.0), 2, [("ball", np.eye(3), (-0.05, 0.05, 0)),
                                                            ("flush", np.eye(3), (0.3, 0.0, 0.0))]),
        "ball": defn("ball", solids.g4_orb(0.2), 3, [("core", np.eye(3), (0.03, 0, 0))]),
        "core": defn("core", solids.g4_orb(0.1), 4),
        "flush": defn("flush", solids.g4_box(0.2, 0.1, 0.1), 5),  # x in [0.2, 0.4]: flush with tank's +x face
        "left": defn("left", solids.g4_box(0.2, 0.2, 0.2), 6),  # x in [-0.8, -0.6]
        "right": defn("right", solids.g4_box(0.2, 0.2, 0.2), 7),  # x in [-0.6, -0.4]: shares a face with left
    })  # fmt: skip


def scene_bool():
    """a box minus a sphere, a union of two boxes (one solid), and an assembly placing two cubes"""
    sub = g4("sub", solids.g4_box(0.4, 0.4, 0.4), solids.g4_orb(0.15), t=(0.1, 0, 0))
    uni = g4("union", solids.g4_box(0.3, 0.1, 0.1), solids.g4_box(0.1, 0.3, 0.1), R=rz(30), t=(0.05, 0, 0))
    return make_scene({
        "world": defn("world", solids.g4_box(2.0, 2.0, 2.0), 1, [("sub", np.eye(3), (-0.4, 0, 0)),
                                                              ("uni", rx(10), (0.4, 0, 0)),
                                                              ("asm", rz(15), (0, 0.5, 0))]),
        "sub": defn("sub", sub, 2), "uni": defn("uni", uni, 3),
        "asm": defn("asm", None, 0, [("cube", np.eye(3), (-0.1, 0, 0)), ("cube", np.eye(3), (0.1, 0, 0))]),
        "cube": defn("cube", solids.g4_box(0.1, 0.1, 0.1), 4),
    })  # fmt: skip


def scene_wires(n=40):
    """a plane volume of n thin wires (r 0.5 mm, 600 mm long) all at 60 deg to the plane's z axis: a frame group"""
    pls = [("wire", rx(-60), (0.0, 0.0, -0.3 + 0.6 * (k + 0.5) / n)) for k in range(n)]
    return make_scene({
        "world": defn("world", solids.g4_box(2.0, 2.0, 2.0), 1, [("plane", np.eye(3), (0, 0, 0))]),
        "plane": defn("plane", solids.g4_box(0.02, 1.6, 1.6), 1, pls),
        "wire": defn("wire", solids.g4_tube(0.0, 0.0005, 0.6), 2),
    })  # fmt: skip


def scene_split():
    """a union of 20 small boxes (a frame) minus a slab: the solid split_unions places as its pieces"""
    s = solids.g4_box(0.04, 0.04, 0.04)
    for k in range(1, 20):
        s = g4("union", s, solids.g4_box(0.04, 0.04, 0.04), t=(0.03 * k, 0.01 * (k % 3), 0.0))
    s = g4("sub", s, solids.g4_box(2.0, 2.0, 0.01), t=(0, 0, 0.012))
    return make_scene({
        "world": defn("world", solids.g4_box(2.0, 2.0, 2.0), 1, [("frame", np.eye(3), (-0.3, 0.0, 0.0))]),
        "frame": defn("frame", s, 2),
    })  # fmt: skip


def scene_overlap():
    """two siblings overlapping by 50 mm, and a daughter protruding 20 mm from its mother"""
    return make_scene({
        "world": defn("world", solids.g4_box(2.0, 2.0, 2.0), 1, [("a", np.eye(3), (-0.15, 0, 0)),
                                                              ("b", np.eye(3), (0.1, 0, 0)),
                                                              ("m", np.eye(3), (0, 0.6, 0))]),
        "a": defn("a", solids.g4_box(0.4, 0.4, 0.4), 2), "b": defn("b", solids.g4_box(0.2, 0.2, 0.2), 3),
        "m": defn("m", solids.g4_box(0.2, 0.2, 0.2), 4, [("p", np.eye(3), (0.11, 0, 0))]),
        "p": defn("p", solids.g4_box(0.04, 0.04, 0.04), 5),
    })  # fmt: skip


def tet_tris():
    v = np.array([[0, 0, 0], [0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]]) - 0.075
    return v[np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])]


def scene_mesh():
    """an opaque tetrahedron mesh beside a box"""
    return make_scene({
        "world": defn("world", solids.g4_box(2.0, 2.0, 2.0), 1, [("tet", np.eye(3), (0.3, 0.1, 0)),
                                                              ("box", np.eye(3), (-0.4, 0, 0))]),
        "tet": mesh_def("tet", tet_tris(), 6),
        "box": defn("box", solids.g4_box(0.2, 0.2, 0.2), 2),
    })  # fmt: skip


def nav_of(sc):
    return navigate.Navigator.from_scene(sc)


SCENES = {"nest": scene_nest, "bool": scene_bool, "wires": scene_wires, "mesh": scene_mesh, "split": scene_split,
          "overlap": scene_overlap}  # fmt: skip


@functools.cache
def toy(name):
    """Return (scene, navigator) of a named toy, built once per session (read only: never change the scene)."""
    sc = SCENES[name]()
    return sc, nav_of(sc)


# ------------------------------------------------------------------------------------------------ batched calls
# A test's cost is tracing, lowering and compiling its programs, not running them, and the persistent cache stores
# only the compiled code. A jax.vmap of the query's nested loops takes ~5 s to trace; the one-ray program traces in
# under 1 s and compiles in 1-2 s with the navigator's arrays as constants (3 s as operands). So a batch runs one
# compiled one-ray program per row from the host (~0.1 ms a row), and each program is made once per session per
# navigator and function, shared by every test of that toy. ``by_layout``: the navigator's arrays enter as operands
# instead, so one program serves every navigator of the same layout (the GDML cube toys).
_PROGRAMS = {}


def layout(nav):
    """Return what a traced navigator program depends on besides its operands' values."""
    leaves, tree = jax.tree.flatten(nav.operands())
    return (nav.n, nav.n_mesh, nav.chunk, int(nav.CH0), tuple(nav.SRF), bool(nav.E.has_torus), int(nav.E.empty_leaf),
            tree, tuple((a.shape, str(a.dtype)) for a in leaves))  # fmt: skip


def _program(fn, nav, by_layout):
    prec = jax.config.jax_default_matmul_precision
    key = (fn, layout(nav), prec) if by_layout else (fn, id(nav), prec)
    if key not in _PROGRAMS:
        if by_layout:
            prog = jax.jit(lambda ops, *a, tmpl=nav: fn(tmpl.bind(ops), *a))
        else:
            prog = jax.jit(functools.partial(fn, nav))
        _PROGRAMS[key] = (nav, prog)  # the navigator held: its id stays its own
    prog = _PROGRAMS[key][1]
    return functools.partial(prog, nav.operands()) if by_layout else prog


def _host(a):
    a = np.asarray(a)
    return a.astype(np.float32 if a.dtype.kind == "f" else np.int32 if a.dtype.kind in "iu" else a.dtype)


def per_ray(fn, nav, *rays, shared=(), by_layout=False):
    """Return fn(nav, *row, *shared) for each row of the arrays in ``rays`` (leading axis), stacked as numpy arrays.

    ``fn``: a module-level function of a navigator (its program is cached on fn and the navigator); ``shared``:
    arrays passed to every row (t_max, a pass table).
    """
    prog = _program(fn, nav, by_layout)
    rows = [_host(r) for r in rays]
    sh = tuple(jnp.asarray(_host(s)) for s in shared)
    out = jax.device_get([prog(*(r[i] for r in rows), *sh) for i in range(len(rows[0]))])
    return jax.tree.map(lambda *a: np.stack(a), *out)


def _locate(nv, x):
    return nv.locate(x)


def _query(nv, o, d, v, last, t_max):
    return nv.query(o, d, v, last=last, t_max=t_max)


def _query_sn(nv, o, d, v, last, sn, t_max):
    return nv.query(o, d, v, last=last, sn=sn, t_max=t_max)


def _query_pass(nv, o, d, v, last, t_max, pass_tab):
    return nv.query(o, d, v, last=last, t_max=t_max, pass_tab=pass_tab)


def locate(nav, X, by_layout=False):
    """Return the volume holding each point of X (n, 3): ``nav.locate`` per point."""
    return per_ray(_locate, nav, np.asarray(X, np.float32).reshape(-1, 3), by_layout=by_layout)


def query(nav, X, D, vol, last=-1, t_max=1e3, sn=None, pass_tab=None, by_layout=False):
    """Return ``nav.query`` per ray (a dict of arrays): X, D (n, 3) or (3,); vol, last (n,) or one; sn (n, 3), (3,)."""
    X, D = np.asarray(X, np.float32).reshape(-1, 3), np.asarray(D, np.float32).reshape(-1, 3)
    n = max(len(X), len(D), np.size(vol), np.size(last), 0 if sn is None else np.size(sn) // 3)
    rows = [np.broadcast_to(X, (n, 3)), np.broadcast_to(D, (n, 3)), np.broadcast_to(np.asarray(vol, np.int32), (n,)),
            np.broadcast_to(np.asarray(last, np.int32), (n,))]  # fmt: skip
    tm = np.float32(t_max)
    if pass_tab is not None:
        assert sn is None
        return per_ray(_query_pass, nav, *rows, shared=(tm, np.asarray(pass_tab, bool)), by_layout=by_layout)
    if sn is not None:
        sn = np.broadcast_to(np.asarray(sn, np.float32).reshape(-1, 3), (n, 3))
        return per_ray(_query_sn, nav, *rows, sn, shared=(tm,), by_layout=by_layout)
    return per_ray(_query, nav, *rows, shared=(tm,), by_layout=by_layout)


def one(r):
    """Return row 0 of a per-ray result (a single ray's)."""
    return jax.tree.map(lambda a: a[0], r)


def rays(n, seed, half=0.95):
    rng = np.random.default_rng(seed)
    O_ = rng.uniform(-half, half, (n, 3))
    Dr = rng.normal(size=(n, 3))
    return O_, Dr / np.linalg.norm(Dr, axis=1, keepdims=True)


def _agree(orc, r, i, o32, d, t0, tol):
    ro = orc.query(o32.astype(float), d, t0, 10.0, push=0.0, medium=int(r["cls_before"][i]))
    ok = (ro["hit"] == bool(r["hit"][i])) and (not ro["hit"] or (
        abs(ro["t"] - r["t"][i]) <= tol * (1 + abs(ro["t"])) and ro["after"][0] == r["cls_after"][i]))  # fmt: skip
    return ok or ro["start"] != "ok", ro


def walk_check(sc, nav=None, n=1500, steps=4, seed=0, orc_scene=None, tol=2e-5):
    """Random starts; each step the navigator's first visible change against the oracle's from the same point.

    Continued from the hit point by identity, transmitted (the volume after) or reflected (the volume before) at
    random. Returns (agree, total, bad).
    """
    nav = nav_of(sc) if nav is None else nav
    orc = oracle.Oracle(orc_scene or sc)
    O_, Dr = rays(n, seed)
    vol = locate(nav, O_)
    last = np.full(n, -1)
    rng = np.random.default_rng(seed + 1)
    O32 = O_.astype(np.float32)
    agree = total = 0
    bad = []
    alive = np.ones(n, bool)
    for s in range(steps):
        r = query(nav, O32, Dr, vol, last, t_max=10.0)
        for i in np.flatnonzero(alive):
            # a continued walk starts on a surface: the navigator takes everything within the band of the start as
            # the start (Geant4's kCarTolerance), so the oracle starts past that band too
            t0 = T_ORC if s == 0 else 2.0 * (BAND * float(np.max(np.abs(O32[i]))) + 1e-9)
            ok, ro = _agree(orc, r, i, O32[i], Dr[i], t0, tol)
            total += 1
            agree += ok
            if not ok:
                tn = float(r["t"][i]) if r["hit"][i] else 0.0
                to = float(ro["t"]) if ro["hit"] else 0.0
                bad.append((s, i, O32[i].tolist(), Dr[i].tolist(), int(vol[i]), ro["t"], ro["after"][0],
                            float(r["t"][i]), int(r["cls_after"][i]), bool(r["zero"][i]),
                            (O32[i] + tn * Dr[i]).tolist(), (O32[i] + to * Dr[i]).tolist()))  # fmt: skip
        # a walk ends where it leaves the world or enters an opaque part (a mesh: it holds no point)
        alive &= r["hit"].astype(bool) & (r["cls_after"] != 0) & (r["vol_after"] <= nav.n)
        nrm = r["normal"].astype(float)  # faces against d
        Dn = rng.normal(size=Dr.shape)
        Dn /= np.linalg.norm(Dn, axis=1, keepdims=True)
        trans = rng.random(n) < 0.5
        s_ = np.sign(np.sum(Dn * nrm, 1))
        Dn = np.where(((trans & (s_ > 0)) | (~trans & (s_ < 0)))[:, None], -Dn, Dn)  # transmit: along -nrm
        O32 = np.where(alive[:, None], (O32 + r["t"][:, None].astype(np.float32) * Dr.astype(np.float32)), O32)
        vol = np.where(alive & trans, r["vol_after"], vol)
        last = np.where(alive, r["item"], last)
        Dr = np.where(alive[:, None], Dn, Dr)
    return agree, total, bad
