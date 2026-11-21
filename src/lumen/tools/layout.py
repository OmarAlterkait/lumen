"""Where each sensor goes on a flat event display, and its outline there, from the detector's geometry.

    lay = layout.of(det)                    # or of positions (S, 3) m; kind="planes" | "cylinder" | "sphere" to force
    lay.kind, lay.xy, lay.shapes            # the layout, each sensor's display position (m), its outline there

Three layouts, tried in this order; the only scale is the sensors' own pitch (the median distance to the nearest
sensor), so nothing is tuned per detector:

- **planes** (a TPC's light detectors, a box): along one axis the sensors fall in slabs, each thinner than the pitch
  within it and each filling an area (a sensor more than half a pitch inside the slab's outline: a cylinder's ring of
  sensors is a slab too, but every sensor of it is on its outline). Each plane is drawn in true scale, its longer
  in-plane axis across, all seen from the same side and stacked by their coordinate (largest on top).
- **cylinder** (a water Cherenkov tank): about one axis every sensor lies within a pitch of the sensors' bounding
  cylinder (a multi-PMT module's small PMTs too), and some on its barrel. The barrel is unrolled to (R phi, height)
  with its seam in the widest gap between sensors in azimuth; each cap is a disc above or below it, touching the
  barrel where the azimuths meet. A sensor belongs to the surface it is nearer to.
- **sphere** (anything else): an equal-area (Hammer) projection of each sensor's direction from the sensors' centre.

Given a detector, each sensor is drawn with its own outline: its solid's box in its own frame (a round solid, a tube,
cone, sphere or ellipsoid, by the circles of its ends) placed as the sensor is, carried through the same map as the
positions, and its convex hull taken there; a PMT is a disc of its radius, a bar or an X-ARAPUCA its rectangle, each
turned as it is placed. Given positions alone (or on a sphere), each sensor is a disc of 0.9 pitch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

KINDS = ("planes", "cylinder", "sphere")
AXES = "xyz"
ROUND = {"tube", "cutTube", "cone", "polycone", "sphere", "orb", "ellipsoid", "eltube", "elcone", "paraboloid",
         "hype", "torus", "genericPolycone"}  # fmt: skip  # solids whose cross-section across their z is round
CIRCLE = 32  # points on each end circle of a round solid's outline


@dataclass(frozen=True)
class Layout:
    """A display layout: ``xy`` (S, 2) m, each sensor's ``shapes`` (its outline polygon (k, 2), m), the outlines of the
    surfaces (``("rect", (x0, y0), w, h)``, ``("circle", (x, y), r)``, ``("ellipse", a, b)``), labels ``(x, y,
    text)`` and a one-line description."""  # noqa: D205, D209

    kind: str
    xy: np.ndarray
    shapes: list
    outlines: tuple = field(default_factory=tuple)
    labels: tuple = field(default_factory=tuple)
    note: str = ""

    @property
    def size(self) -> float:
        """The typical sensor's width on the display (m)."""
        return float(np.median([np.ptp(s, axis=0).max() for s in self.shapes]))


def pitch(p) -> float:
    """Return the median distance from each point to its nearest neighbour (0 for fewer than two points)."""
    p = np.asarray(p, np.float64)
    if len(p) < 2:
        return 0.0
    d, _ = cKDTree(p).query(p, k=2)
    return float(np.median(d[:, 1]))


def of(det_or_positions, kind: str = "auto") -> Layout:
    """Return the layout of a detector's sensors (with their outlines), or of positions (S, 3) m (discs)."""
    det = det_or_positions if hasattr(det_or_positions, "sensor_table") else None
    p = np.asarray(det.sensor_table().position if det is not None else det_or_positions, np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0:
        raise ValueError(f"layout: positions must be (S, 3), got {p.shape}")
    if kind not in ("auto", *KINDS):
        raise ValueError(f"layout: kind {kind!r} is not 'auto' or one of {KINDS}")
    centres = _module_centres(det, p) if det is not None else p
    made = None
    if kind in ("auto", "planes"):
        pl = _planes(p)
        if pl is not None:
            made = _planes_map(p, *pl)
        elif kind == "planes":
            raise ValueError("layout: the sensors do not lie in planes")
    if made is None and kind in ("auto", "cylinder"):
        cy = _cylinder(p)
        if cy is not None:
            made = _cylinder_map(p, *cy, centres)
        elif kind == "cylinder":
            raise ValueError("layout: the sensors do not lie on a cylinder")
    if made is None:
        made = _sphere_map(p)
    name, to_xy, part, texts, note = made
    xy = to_xy(p, np.arange(len(p)))
    pts = _outline_points(det) if det is not None and name != "sphere" else None
    if pts is None:
        d = 0.45 * (pitch(xy) or pitch(p) or 1.0)
        a = np.linspace(0, 2 * np.pi, CIRCLE, endpoint=False)
        disc = d * np.column_stack([np.cos(a), np.sin(a)])
        shapes = [c + disc for c in xy]
    else:
        m = pts.shape[1]
        flat = to_xy(pts.reshape(-1, 3), np.repeat(np.arange(len(p)), m)).reshape(len(p), m, 2)
        shapes = [_hull(q) for q in flat]
    return _place(name, xy, shapes, part, texts, note)


def _place(name, xy, shapes, part, texts, note) -> Layout:
    """Return the layout with its surfaces placed from what is drawn.

    Every sensor's outline lies inside its surface's: the planes stacked one under the other, the caps clear of the
    barrel; each surface's outline is a margin around its sensors.
    """
    size = float(np.median([np.ptp(q, axis=0).max() for q in shapes]))
    margin = 0.5 * size
    shift = np.zeros((int(part.max()) + 1, 2))
    outlines, labels = [], []
    box = [np.vstack([shapes[i] for i in np.flatnonzero(part == j)]) for j in range(len(shift))]
    if name == "planes":
        y = 0.0
        for j, b in enumerate(box):  # part j is the j-th plane from the top
            lo, hi = b.min(0) - margin, b.max(0) + margin
            shift[j, 1] = y - hi[1]
            outlines.append(("rect", (lo[0], lo[1] + shift[j, 1]), hi[0] - lo[0], hi[1] - lo[1]))
            labels.append((lo[0], y + 0.4 * size, texts[j]))
            y -= hi[1] - lo[1] + 3.0 * size  # room for the next plane's label
    elif name == "cylinder":
        lo, hi = box[0].min(0) - margin, box[0].max(0) + margin  # part 0: the barrel
        outlines.append(("rect", (lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1]))
        for j, sgn in ((1, 1.0), (2, -1.0)):  # the top and the bottom cap, each a disc around its sensors
            if j >= len(box):
                continue
            centre = 0.5 * (box[j].min(0) + box[j].max(0))
            centre[0] = 0.0
            r = float(np.linalg.norm(box[j] - centre, axis=1).max()) + margin
            edge = hi[1] if sgn > 0 else lo[1]  # clear of the barrel by a margin
            want = edge + sgn * (r + margin)
            shift[j, 1] = want - centre[1]
            outlines.append(("circle", (0.0, want), r))
    else:
        outlines.append(("ellipse", 2 * np.sqrt(2), np.sqrt(2)))
    xy = xy + shift[part]
    shapes = [q + shift[j] for q, j in zip(shapes, part, strict=True)]
    return Layout(name, xy, shapes, tuple(outlines), tuple(labels), note)


# ----------------------------------------------------------------------------------------------------- recognition
def _slabs(v, gap):
    """Label each value by its slab: sorted values split where consecutive ones are more than ``gap`` apart."""
    o = np.argsort(v, kind="stable")
    lab = np.empty(len(v), int)
    lab[o] = np.concatenate([[0], np.cumsum(np.diff(v[o]) > gap)])
    return lab


def _filled(q) -> bool:
    """Whether 2D points fill an area: some point more than half a pitch inside their convex hull."""
    q = np.unique(q, axis=0)
    if len(q) < 4:
        return True
    try:
        h = ConvexHull(q)
    except QhullError:  # collinear: a row of sensors is a degenerate plane
        return True
    inside = -(q @ h.equations[:, :2].T + h.equations[:, 2]).max(axis=1)  # each point's distance to the outline
    return bool(inside.max() > 0.5 * pitch(q))


def _planes(p):
    best = None
    gap = pitch(p)
    for k in range(3):
        lab = _slabs(p[:, k], gap)
        n = int(lab.max()) + 1
        slabs = [lab == i for i in range(n)]
        if n > len(p) / 4 or min(int(m.sum()) for m in slabs) < 2:
            continue
        flat = all(np.ptp(p[m, k]) < pitch(np.delete(p[m], k, axis=1)) for m in slabs)
        if flat and all(_filled(np.delete(p[m], k, axis=1)) for m in slabs) and (best is None or n < best[1]):
            best = (k, n, lab)
    return best


def _cylinder(p):
    c = 0.5 * (p.min(0) + p.max(0))
    s = pitch(p)
    fits = []
    for k in range(3):
        a, b = (k + 1) % 3, (k + 2) % 3
        h = p[:, k] - c[k]
        r = np.hypot(p[:, a] - c[a], p[:, b] - c[b])
        big_r, big_h = r.max(), np.abs(h).max()
        if not ((big_h - np.abs(h)) >= (big_r - r)).any():  # no barrel: planes facing each other are not a cylinder
            continue
        fits.append((float(np.max(np.minimum(big_r - r, big_h - np.abs(h)))), k))
    if not fits or min(fits)[0] >= s:
        return None
    return (min(fits)[1], c)


# ---------------------------------------------------------------------------------- the maps from world to display
def _planes_map(p, k, n, lab):
    o = [i for i in range(3) if i != k]
    ext = np.ptp(p[:, o], axis=0)
    u, v = (o[0], o[1]) if ext[0] >= ext[1] else (o[1], o[0])  # the longer in-plane axis across
    order = np.argsort([-p[lab == i, k].mean() for i in range(n)])  # the largest coordinate on top
    rows = np.empty(n, int)
    rows[order] = np.arange(n)

    def to_xy(x, owner):
        return np.column_stack([x[:, u], x[:, v]])

    texts = [f"{AXES[k]} = {p[lab == i, k].mean():+.2f} m" for i in order]
    side = "+" if np.cross(np.eye(3)[u], np.eye(3)[v])[k] > 0 else "-"  # across x up points at the viewer
    note = (
        f"{n} plane{'s' if n > 1 else ''} of constant {AXES[k]}, seen from {side}{AXES[k]}: {AXES[u]} across, "
        f"{AXES[v]} up"
    )
    return "planes", to_xy, rows[lab], texts, note


def _cylinder_map(p, k, c, centres):
    a, b = (k + 1) % 3, (k + 2) % 3

    def cyl(x):
        return x[:, k] - c[k], np.hypot(x[:, a] - c[a], x[:, b] - c[b]), np.arctan2(x[:, b] - c[b], x[:, a] - c[a])

    h, r, phi = cyl(p)
    big_r, big_h = r.max(), np.abs(h).max()
    hm, rm, _ = cyl(centres)  # a module's sensors go with it: to the surface its centre is nearer to
    cap = (big_h - np.abs(hm)) < (big_r - rm)
    seam = _widest_gap(phi[~cap])
    side = np.where(cap, np.sign(hm), 0.0)  # +1 top cap, -1 bottom cap, 0 barrel
    ref = np.mod(phi - seam, 2 * np.pi) - np.pi  # each sensor's own azimuth from the barrel's middle
    gap = pitch(p)
    yc = big_h + gap + big_r  # each cap's centre: it touches the barrel at phi = 0

    def to_xy(x, owner):
        hh, rr, ff = cyl(x)
        f = ref[owner] + np.mod(ff - seam - np.pi - ref[owner] + np.pi, 2 * np.pi) - np.pi  # near the sensor's own
        sd = side[owner]
        out = np.column_stack([big_r * f, hh])
        capped = sd != 0
        out[capped, 0] = rr[capped] * np.sin(f[capped])
        out[capped, 1] = sd[capped] * (yc - rr[capped] * np.cos(f[capped]))
        return out

    part = np.where(side > 0, 1, np.where(side < 0, 2, 0))  # 0 the barrel, 1 the top cap, 2 the bottom cap
    note = (f"cylinder about {AXES[k]}, R {big_r:.2f} m, height {2 * big_h:.2f} m: the barrel unrolled (R phi "
            f"across, {AXES[k]} up), the caps above and below")  # fmt: skip
    return "cylinder", to_xy, part, (), note


def _sphere_map(p):
    c = p.mean(0)

    def to_xy(x, owner):
        d = x - c
        n = np.linalg.norm(d, axis=1)
        lon = np.arctan2(d[:, 1], d[:, 0])
        lat = np.arcsin(np.clip(d[:, 2] / np.where(n > 0, n, 1.0), -1.0, 1.0))
        den = np.sqrt(1 + np.cos(lat) * np.cos(lon / 2))
        return np.column_stack([2 * np.sqrt(2) * np.cos(lat) * np.sin(lon / 2) / den, np.sqrt(2) * np.sin(lat) / den])

    return ("sphere", to_xy, np.zeros(len(p), int), (),
            "equal-area (Hammer) projection of each sensor's direction from the centre, z up")  # fmt: skip


def _module_centres(det, p):
    """Return each sensor's module centre, or its own position.

    A module is the lowest volume holding several sensors, where each such volume stands apart from the next (its
    radius under half the distance between their centres): a multi-PMT module, not a cap assembly holding a cap and
    the barrel's border rows.
    """
    mother = {x.pv: x.mother for x in det.parts.instances}
    names = list(det.sensor_table().name)
    under: dict = {}
    for i, pv in enumerate(names):  # every sensor under each of its ancestors
        m = mother.get(pv)
        while m:
            under.setdefault(m, []).append(i)
            m = mother.get(m)
    lowest = {}
    for anc, idx in sorted(under.items(), key=lambda kv: len(kv[1])):  # the smallest first
        if len(idx) > 1:
            for i in idx:
                lowest.setdefault(i, anc)
    groups: dict = {}
    for i, anc in lowest.items():
        groups.setdefault(anc, []).append(i)
    groups = {g: np.array(v) for g, v in groups.items()}
    if len(groups) < 2:
        return p
    cen = np.array([p[v].mean(0) for v in groups.values()])
    span = np.array([np.linalg.norm(p[v] - p[v].mean(0), axis=1).max() for v in groups.values()])  # its radius
    near, _ = cKDTree(cen).query(cen, k=2)
    out = p.copy()
    for (_g, v), c, sp, d in zip(groups.items(), cen, span, near[:, 1], strict=True):
        if 2 * sp < d:  # a module: apart from the next one (two modules' radii fit between their centres)
            out[v] = c
    return out


# ---------------------------------------------------------------------------------------------- the sensors' outline
def _outline_points(det):
    """Each sensor's outline points in the world (S, M, 3): its solid's box (round: its end circles), placed."""
    from lumen.build.lowering import SolidMapper, solid_box

    f = det.file
    pose = {x.pv: x for x in det.parts.instances}
    mapper, boxes, out = SolidMapper(f), {}, []
    a = np.linspace(0, 2 * np.pi, CIRCLE, endpoint=False)
    for pv in det.sensor_table().name:
        x = pose[pv]
        name = f.volumes[x.lv].solid
        if name not in boxes:
            lo, hi = (np.asarray(v, np.float64) for v in solid_box(mapper.solid(name)))
            c, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
            if _round(f, name):
                ring = np.column_stack([half[0] * np.cos(a), half[1] * np.sin(a)])
                local = np.vstack([np.column_stack([ring, np.full(CIRCLE, z)]) for z in (-half[2], half[2])]) + c
            else:
                local = c + half * np.array([[i, j, kk] for i in (-1, 1) for j in (-1, 1) for kk in (-1, 1)], float)
                local = np.vstack([local] * (2 * CIRCLE // 8))  # every sensor the same number of points
            boxes[name] = local
        out.append(boxes[name] @ np.asarray(x.R, np.float64).T + np.asarray(x.t, np.float64))
    return np.stack(out)


def _round(f, name) -> bool:
    """Whether a solid's cross-section across its z is round (a boolean: its first solid's)."""
    s = f.solids[name]
    while s.refs and s.type not in ROUND:
        s = f.solids[s.refs[0]]
    return s.type in ROUND


def _hull(q):
    try:
        return q[ConvexHull(q).vertices]
    except QhullError:  # seen edge on: a sliver of its extent
        lo, hi = q.min(0), q.max(0)
        return np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]])


def _widest_gap(phi) -> float:
    """Return the azimuth in the middle of the widest gap between the given azimuths (rad)."""
    if len(phi) < 2:
        return np.pi
    s = np.sort(phi)
    gaps = np.diff(np.concatenate([s, [s[0] + 2 * np.pi]]))
    i = int(np.argmax(gaps))
    return float(s[i] + 0.5 * gaps[i])
