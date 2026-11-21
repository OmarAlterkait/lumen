"""The display layout is read from the sensors' positions alone: planes, an unrolled cylinder, or a sphere."""

import numpy as np
import pytest

from lumen.tools import layout


def _tank(radius=2.0, half=3.0, rings=12, per_ring=36, cap_rows=7, axis=2):
    """A water Cherenkov tank: rings on the barrel and a grid on each cap, about ``axis``."""
    phi = np.linspace(0, 2 * np.pi, per_ring, endpoint=False) + 0.1
    z = np.linspace(-half + 0.2, half - 0.2, rings)
    barrel = np.array([[radius * np.cos(f), radius * np.sin(f), h] for h in z for f in phi])
    g = np.linspace(-radius + 0.2, radius - 0.2, cap_rows)
    disc = np.array([[x, y] for x in g for y in g if np.hypot(x, y) < radius - 0.15])
    caps = np.vstack([np.column_stack([disc, np.full(len(disc), s * half)]) for s in (1, -1)])
    p = np.vstack([barrel, caps])
    return np.roll(p, axis - 2, axis=1), len(barrel)


def _planes(xs=(-2.0, 2.0), ny=4, nz=9, jitter=0.0):
    """Light detectors on planes of constant x (a TPC's), each a grid in (y, z)."""
    rng = np.random.default_rng(1)
    y, z = np.meshgrid(np.linspace(-1.5, 1.5, ny), np.linspace(0.2, 4.8, nz))
    return np.vstack([np.column_stack([np.full(y.size, x) + jitter * rng.uniform(-1, 1, y.size), y.ravel(), z.ravel()])
                      for x in xs])  # fmt: skip


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_tank_unrolls_about_its_axis_with_caps_above_and_below(axis):
    p, nb = _tank(axis=axis)
    lay = layout.of(p)
    assert lay.kind == "cylinder" and f"about {'xyz'[axis]}" in lay.note
    y = lay.xy[:, 1]
    assert np.all(np.abs(y[:nb]) <= 3.0 + 1e-9)  # the barrel between the caps' discs
    assert np.all(np.abs(y[nb:]) > 3.0)  # each cap outside it
    assert np.ptp(lay.xy[:nb, 0]) == pytest.approx(2 * np.pi * 2.0 * (35 / 36), rel=1e-6)  # R phi, seam in a gap
    assert len(np.unique(lay.xy.round(9), axis=0)) == len(p)  # no two sensors drawn on top of each other


def test_planes_facing_each_other_are_planes_not_a_cylinder_and_rings_are_not_planes():
    for xs in ((-2.0, 2.0), (0.0,), (-4.0, -1.0, 1.0, 4.0)):  # SBND, MicroBooNE, ICARUS-like
        lay = layout.of(_planes(xs, jitter=0.004))
        assert lay.kind == "planes" and lay.note.startswith(f"{len(xs)} plane")
        assert len(lay.labels) == len(xs)
    lay = layout.of(_planes((-2.0, 2.0)))
    top = lay.xy[: lay.xy.shape[0] // 2, 1]  # x = -2 is drawn below x = +2
    assert top.max() < lay.xy[lay.xy.shape[0] // 2 :, 1].min()
    assert layout.of(_tank()[0]).kind == "cylinder"  # its barrel's rings are slabs, but not filled


def test_anything_else_is_a_sphere_and_a_kind_can_be_forced_or_is_refused():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(400, 3))
    sphere = 3.0 * v / np.linalg.norm(v, axis=1, keepdims=True)
    lay = layout.of(sphere)
    assert lay.kind == "sphere" and np.all(np.abs(lay.xy[:, 0]) <= 2 * np.sqrt(2) + 1e-9)
    assert layout.of(_tank()[0], "sphere").kind == "sphere"
    with pytest.raises(ValueError, match="do not lie in planes"):
        layout.of(sphere, "planes")
    with pytest.raises(ValueError, match=r"\(S, 3\)"):
        layout.of(np.zeros((3, 2)))


def test_the_displays_draw_on_every_layout(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")
    from lumen.tools import plot

    for i, p in enumerate((_tank()[0], _planes(), _planes((0.0,)))):
        q = 1.0 / np.sum((p - p.mean(0) - 0.3) ** 2, axis=1)
        q[::7] = 0.0
        fig = plot.event(p, q, np.linalg.norm(p, axis=1) / 0.22)
        fig.savefig(tmp_path / f"e{i}.png")
        ax = plot.display(p, q - q.mean(), diverging=True)
        ax.figure.savefig(tmp_path / f"d{i}.png")
    with pytest.raises(ValueError, match="values for"):
        plot.display(_planes(), np.ones(3))


def test_every_sensor_is_drawn_inside_its_surface_and_the_surfaces_do_not_overlap():
    for p in (_tank()[0], _planes(), _planes((-4.0, -1.0, 1.0, 4.0))):
        lay = layout.of(p)
        rects = [o for o in lay.outlines if o[0] == "rect"]
        circles = [o for o in lay.outlines if o[0] == "circle"]

        def inside(q):
            in_rect = any(np.all((q >= r[1]) & (q <= (r[1][0] + r[2], r[1][1] + r[3]))) for r in rects)
            in_circle = any(np.all(np.linalg.norm(q - c[1], axis=1) <= c[2] + 1e-9) for c in circles)
            return in_rect or in_circle

        assert all(inside(q) for q in lay.shapes)
        spans = sorted([(r[1][1], r[1][1] + r[3]) for r in rects] + [(c[1][1] - c[2], c[1][1] + c[2]) for c in circles])
        assert all(a[1] < b[0] for a, b in zip(spans, spans[1:], strict=False))  # stacked, none over another
