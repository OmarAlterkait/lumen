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
        boxes = [(r[1][0], r[1][1], r[1][0] + r[2], r[1][1] + r[3]) for r in rects]
        boxes += [(c[1][0] - c[2], c[1][1] - c[2], c[1][0] + c[2], c[1][1] + c[2]) for c in circles]
        for i, a in enumerate(boxes):  # no surface over another (a cap's square around its disc included)
            for b in boxes[i + 1 :]:
                assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1], (a, b)


def test_a_box_with_no_clearly_thinnest_axis_faces_no_axis():
    assert layout._thinnest([0.3, 0.012, 0.3]) == 1  # an X-ARAPUCA window: it faces along its thin axis
    assert layout._thinnest([0.265, 0.265, 0.265]) is None  # a PMT's cube-like box (Super-K's): its x is arbitrary
    assert layout._thinnest([0.13, 0.12, 0.10]) is None


def test_a_face_in_far_pieces_joins_the_walls_its_sensors_sit_on_and_a_grid_of_two_rows_is_one_face():
    walls = _planes(xs=(-2.0, 2.0), ny=4, nz=9)  # two walls of constant x, sensors facing x
    side = np.array([[x, y, 2.5] for x in (-2.0, 2.0) for y in (-1.0, 0.0, 1.0)])  # on the walls, facing z
    p = np.vstack([walls, side])
    n = np.vstack([np.tile([1.0, 0.0, 0.0], (len(walls), 1)), np.tile([0.0, 0.0, 1.0], (len(side), 1))])
    face, info, keep = layout._faces(p, n)
    assert [k for k, _ in info] == [0, 0] and keep.all()  # the two columns 4 m apart are no face of constant z
    assert np.all(np.abs(p[np.arange(len(p)), 0] - np.array([info[f][1] for f in face])) < 1e-9)  # each on its wall
    rows = np.array([[0.3 * i, 0.3 * j, 7.0] for i in range(10) for j in range(2)])  # ND-LAr's tiles: 2 x 10
    face, info, keep = layout._faces(rows, np.tile([0.0, 0.0, 1.0], (len(rows), 1)))
    assert info == [(2, 7.0)] and np.all(face == 0) and keep.all()


def test_the_planes_are_along_one_axis_the_one_holding_most_sensors_then_fewest_planes():
    g = np.array([[a, b] for a in np.linspace(-3, 3, 8) for b in np.linspace(0, 10, 20)])  # 160 sensors a wall
    cathode = np.column_stack([np.zeros(len(g)), g])  # FD2: one plane of constant x ...
    walls = np.vstack([np.column_stack([g[:80, 0], np.full(80, s * 7.0), g[:80, 1]]) for s in (1, -1)])  # ... 2 of y
    p = np.vstack([cathode, walls])
    n = np.vstack([np.tile([1.0, 0, 0], (len(cathode), 1)), np.tile([0, 1.0, 0], (len(walls), 1))])
    face, info, keep = layout._faces(p, n)
    assert info == [(0, 0.0)] and keep.sum() == len(cathode) and np.all(face[keep] == 0)  # a tie: the one plane
    lay_note = layout._faces_map(p[keep], face[keep], info)[4]
    assert "constant x" in lay_note and "constant y" not in lay_note
