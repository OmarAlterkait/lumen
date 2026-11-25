"""A float32-thin same-medium shell keeps its cathode, at the navigator: ~20 m from the origin (bands ~19 um) a glass
orb holds a cavity inside a 1 um shell of the glass's own medium, the cathode border on shell -> cavity (or the cavity
in the glass directly, the border on glass -> cavity). The float32 geometry merges glass -> shell -> cavity into one
crossing (glass -> cavity); glass -> shell is a pass pair, so the class pair's kernel is the cathode's, as with the
shell absent (g4ref: a 1 um shell detects as a 0.1 mm one).

The surface lookup's side of coincident faces is in test_surfaces.py."""

import numpy as np
from nav_toys import defn, make_scene, one, per_ray, query, toy
from test_surfaces import MATS, library, tree

from lumen.build import lookup as regions
from lumen.build import tracked
from lumen.build.tree import OUTSIDE_PV
from lumen.geometry import solids
from lumen.surfaces import types

GLASS = "glass_0/glass"


def _inner(shell):
    return "glass_0/shell_0/inner_0/inner" if shell else "glass_0/inner_0/inner"


def shell_scene(shell):
    """the glass orb (r 0.2 m) at (20, 0, 0) in a 60 m world, its cavity (r 0.1 m) in a 1 um shell or not"""
    defs = {"world": defn("world", solids.g4_box(60.0, 60.0, 60.0), 0, [("glass", np.eye(3), (20.0, 0.0, 0.0))]),
            "glass": defn("glass", solids.g4_orb(0.2), 1, [("shell" if shell else "inner", np.eye(3), (0, 0, 0))]),
            "inner": defn("inner", solids.g4_orb(0.1), 2)}  # fmt: skip
    if shell:
        defs["shell"] = defn("shell", solids.g4_orb(0.100001), 1, [("inner", np.eye(3), (0.0, 0.0, 0.0))])
    return make_scene(defs, outside=3)


def shell_optics(shell):
    """the optics of the shell scene: water world, acrylic glass and shell, LAr cavity, the cathode border"""
    i_ = _inner(shell)
    pvs = [(OUTSIDE_PV, "OutLV", "Steel", None), ("world", "WorldLV", "Water", OUTSIDE_PV),
           (GLASS, "GlassLV", "Acrylic", "world")]  # fmt: skip
    if shell:
        pvs.append(("glass_0/shell_0/shell", "ShellLV", "Acrylic", GLASS))
    pvs.append((i_, "InnerLV", "LAr", pvs[-1][0]))
    b_pre = "glass_0/shell_0/shell" if shell else GLASS
    bind = regions.Bindings({m: MATS[m] for m in {p[2] for p in pvs}},
                            sensors=(regions.SensorBinding("pmt", 1, pv=i_, entry="border"),),
                            borders={(b_pre, i_): "pmt"})  # fmt: skip
    return regions.compile(library(), tree(*pvs), bind)


def test_a_float32_thin_same_medium_shell_keeps_its_cathode():
    rng = np.random.default_rng(7)
    N = 512
    u = np.c_[-np.ones(N), rng.normal(0, 0.15, (N, 2))]
    Dr = (u / np.linalg.norm(u, axis=1, keepdims=True)).astype(np.float32)
    x0 = np.asarray([20.15, 0.0, 0.0], np.float32)
    kinds = {}
    for shell in (True, False):
        o = shell_optics(shell)
        geom = tracked.from_scene(shell_scene(shell), o)
        assert (geom.vertex_tab is not None) == shell  # glass -> shell: one medium, no surface: a pass pair
        r = per_ray(lambda nav, d, geom=geom: geom.query(None, x0, d, 0, 10.0, vol=-1), geom.nav, Dr)
        inner = o.class_of[_inner(shell)]
        ok = r["hit"] & (r["cls_after"] == inner) & (r["vol_after"] == geom.nav.n - 1)
        assert ok.all(), (shell, int((~ok).sum()), r["cls_before"][:4], r["cls_after"][:4], inner)
        np.testing.assert_allclose(np.linalg.norm(x0 + r["t"][:, None] * Dr - np.array([20.0, 0, 0]), axis=1), 0.1,
                                   atol=1e-4)  # fmt: skip
        # the pair the navigator reports (glass -> cavity: the 1 um shell is merged into the crossing) resolves to
        # the cathode: the compiler takes a coincident crossing past a pass pair to the border
        kinds[shell] = {o.rows[int(k)].surface_type for k in o.row_of[r["cls_before"], r["cls_after"]]}
    assert kinds[True] == kinds[False] == {types.PHOTOCATHODE}, kinds


def test_a_start_within_the_band_of_coincident_faces_leaves_every_one_of_them():
    """A start inside the innermost of three volumes whose +z faces coincide, within the on-face band of that face
    and heading out, ends beyond all three: the on-face rule's zero-length exit is relocated past the face, not into
    a mother whose own face the photon has not passed (a bulk event in a flush film: its light is the mother's
    otherwise, read with the mother's surface).
    """
    from lumen.rules import navigation

    _, nav = toy("coincident")
    inner, world = nav.names.index("env_0/glass_0/inner_0"), nav.names.index("")
    band = navigation.band(0.3)  # inner, glass and env are flush at z = 0.3
    for depth in (0.5 * band, 3.0 * band):  # within K_ON bands of the face: the start is on it
        for d in ([0.0, 0.0, 1.0], [0.99, 0.0, 0.141]):  # straight out, and grazing (the on-face rule's own window)
            r = one(query(nav, [0.0, 0.0, 0.3 - depth], d, inner, t_max=10.0))
            assert bool(r["hit"]) and bool(r["zero"]) and float(r["t"]) == 0.0, (depth, d)
            assert int(r["vol_after"]) == world, (depth, d, nav.names[int(r["vol_after"])])
            assert float(r["normal"][2]) < -0.99, (depth, d, r["normal"])  # the face's normal, against the ray
