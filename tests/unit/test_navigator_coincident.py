"""A float32-thin same-medium shell keeps its cathode, at the navigator: ~20 m from the origin (bands ~19 um) a glass
orb holds a cavity inside a 1 um shell of the glass's own medium, the cathode border on shell -> cavity (or the cavity
in the glass directly, the border on glass -> cavity). The float32 geometry merges glass -> shell -> cavity into one
crossing (glass -> cavity); glass -> shell is a pass pair, so the class pair's kernel is the cathode's, as with the
shell dissolved (g4ref: a 1 um shell detects as a 0.1 mm one).

Ported from the navigator part of the baseline's tests/test_coincident.py; its transport toys (the charge with and
without the shell, the reflecting coincident face) come with the transport. Its class-compiler tests are in
test_legacy_regions.py."""

import numpy as np
from nav_toys import defn, make_scene, per_ray
from test_legacy_regions import LAM, MATS, library, tree

from lumen.build import _legacy_regions as regions
from lumen.build import solids
from lumen.build.tree import OUTSIDE_PV
from lumen.geometry import _legacy_tracked as tracked
from lumen.surfaces import _legacy as legacy

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
    """the legacy optics of the shell scene: water world, acrylic glass and shell, LAr cavity, the cathode border"""
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
    return regions.compile(library(), tree(*pvs), bind, LAM)


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
        assert (geom.pass_tab is not None) == shell  # glass -> shell: one medium, no surface: a pass pair
        r = per_ray(lambda nav, d, geom=geom: geom.query(None, x0, d, 0, 10.0, vol=-1), geom.nav, Dr)
        inner = o.class_of[_inner(shell)]
        ok = r["hit"] & (r["cls_after"] == inner) & (r["vol_after"] == geom.nav.n - 1)
        assert ok.all(), (shell, int((~ok).sum()), r["cls_before"][:4], r["cls_after"][:4], inner)
        np.testing.assert_allclose(np.linalg.norm(x0 + r["t"][:, None] * Dr - np.array([20.0, 0, 0]), axis=1), 0.1,
                                   atol=1e-4)  # fmt: skip
        # the pair the navigator reports (glass -> cavity: the 1 um shell is merged into the crossing) resolves to
        # the cathode: the compiler takes a coincident crossing past a pass pair to the border
        kinds[shell] = {o.kinds[int(k)] for k in o.kind_of[r["cls_before"], r["cls_after"]]}
    assert kinds[True] == kinds[False] == {legacy.PHOTOCATHODE}, kinds
