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


def make_scene(defs, root="world", outside=0, overlays=()):
    return scene.Scene(dict(defs), root, scene.Label(outside, -1, "outside"), [], list(overlays))


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
def vm(fn):
    """jit(vmap(fn))."""
    return jax.jit(jax.vmap(fn))


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
    q = vm(lambda o, d, v, l_: nav.query(o, d, v, last=l_, t_max=10.0))
    O_, Dr = rays(n, seed)
    vol = np.asarray(vm(nav.locate)(jnp.asarray(O_, F32)))
    last = np.full(n, -1)
    rng = np.random.default_rng(seed + 1)
    O32 = O_.astype(np.float32)
    agree = total = 0
    bad = []
    alive = np.ones(n, bool)
    for s in range(steps):
        r = {k: np.asarray(v) for k, v in q(jnp.asarray(O32), jnp.asarray(Dr, F32), jnp.asarray(vol, I32),
                                            jnp.asarray(last, I32)).items()}  # fmt: skip
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
