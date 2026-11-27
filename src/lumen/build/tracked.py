"""The navigator with the surface lookup's classes: the geometry the step queries.

The step's photon carries its volume and the surface it sits on; the surface lookup (``lumen.build.lookup``) gives
each placed volume a class, each sensor volume its sensor index, and which class pairs a crossing is a vertex at (one
table, ``vertices``: every pair but a pass pair, and the classes with a surface between two of their volumes on its
diagonal). This module joins the two:
the class of every navigator volume by its placed-volume name, and the query in the form the step reads.

    geom = from_parts(parts)                   # parts: lumen.build.build.build(...); or from_scene(scene, optics)
    q = geom.query(x, d, t_max, vol=vol, last=srf, sn=sn)                 # dict, the step's contract
    t, n = geom.surface(q, x, d)               # the winner's (t, normal), live in (x, d)
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from lumen.build import tree as volume_tree
from lumen.geometry import navigate, navigator_build
from lumen.geometry import scene as sc
from lumen.surfaces import types


class GeometryError(ValueError):
    """The navigator's volumes and the optics' placed volumes do not match."""


SHOWN = 3  # error messages show at most this many names


def vertices(optics):
    """Return (C, C) bool: whether a crossing from a volume of class a into one of class b is a vertex.

    At a vertex the step applies a surface; the navigator steps over every other crossing.
    Off the diagonal every class pair is one but a pass pair: the same medium, the PASS kernel, neither class a
    sensor's nor opaque (two classes of one medium split only by a role elsewhere: a detector's inner water and the
    water envelopes inside it). On the diagonal, a crossing into another volume of the same class is one where the
    class table's diagonal has a surface (touching siblings of one LV with a skin or an any-entry sensor), as Geant4
    stops at every volume boundary.
    """
    n = optics.n_classes
    med, ko = np.asarray(optics.medium_of), np.asarray(optics.row_of)
    passk = {i for i, r in enumerate(optics.rows) if r.surface_type == types.PASS}
    tagged = {int(optics.class_of[pv]) for pv in optics.sensor_pvs}
    opq = np.array([m.opaque for m in optics.media], bool)
    ok = ~opq[med.astype(int)] & ~np.isin(np.arange(n), list(tagged))
    passes = ok[:, None] & ok[None, :] & (med[:, None] == med[None, :]) & np.isin(ko, list(passk))
    out = ~passes
    out[np.diag_indices(n)] = np.diag(ko) >= 0
    return out


def interchangeable(optics, vertex):
    """Return per class the least class it is interchangeable with: a pair crossed without a vertex both ways
    (``vertices``) whose two classes cross every other class alike (surface row, both ways, the deposit's
    side), neither a sensor's -- a photon in one may be carried as the other, exactly."""  # noqa: D205, D209
    n = optics.n_classes
    pass_tab = ~vertex & ~np.eye(n, dtype=bool)
    R = np.asarray(optics.row_of)
    dp = np.zeros((n, n), bool) if optics.deposit_pre is None else np.asarray(optics.deposit_pre, bool)
    canon = np.arange(n)
    for a, b in zip(*np.nonzero(np.triu(pass_tab & pass_tab.T, 1)), strict=True):
        rest = np.ones(n, bool)
        rest[[a, b]] = False
        if all(np.array_equal(M[a, rest], M[b, rest]) and np.array_equal(M[rest, a], M[rest, b]) for M in (R, dp)):
            canon[canon == canon[b]] = canon[a]
    return tuple(int(c) for c in canon)


class TrackedGeometry:
    """The navigator with a class per volume, a sensor index per volume (-1 none) and the vertex table (``vertices``;
    None: a crossing is a vertex where the class changes)."""  # noqa: D205, D209

    def __init__(self, nav, n_classes, sensor_of_vol, vertex_tab=None):
        self.nav = nav
        self.n_classes = int(n_classes)
        self.sensor_of_vol = np.asarray(sensor_of_vol, np.int32)  # (n volumes + 1 + n meshes,)
        plain = vertex_tab is None or np.array_equal(vertex_tab, ~np.eye(len(vertex_tab), dtype=bool))
        self.vertex_tab = None if plain else np.asarray(vertex_tab, bool)

    def __hash__(self):
        """Hash by identity (a static argument of the step)."""
        return id(self)

    def query(self, x, d, t_max, vol=-1, last=-1, sn=None, s_at=None, s_free=None):  # noqa: PLR0913 -- the step's contract
        """Return the next visible crossing in the step's form (dict).

        ``s_free``, the step's free path, is not read here: ``lumen.geometry.reach`` is the geometry that reads it.

        ``vol`` < 0: not known (a source's photon), located here, once; ``last``: the srf the photon sits on;
        ``sn``: the normal of the event that left it; ``s_at``: a bulk event's distance (``vol_at``, the volume the ray
        is in there, and ``cls_at`` its class). ``sensor``: the sensor index of the volume entered (-1 none);
        ``item``: the srf met; ``win``: the winner (item, leaf, face, zero) for ``surface``.
        """
        nav = self.nav
        vol = jnp.asarray(vol, jnp.int32)
        v0 = nav.locate(x, start=vol)  # the descent only where vol is not known
        r = nav.query(x, d, v0, last=jnp.where(vol < 0, -1, last), t_max=t_max, vertex_tab=self.vertex_tab, sn=sn,
                      s_at=s_at)  # fmt: skip
        sensors = jnp.asarray(self.sensor_of_vol)
        return dict(t=r["t"], hit=r["hit"], cls_after=r["cls_after"],
                    sensor=jnp.where(r["hit"], sensors[r["vol_after"]], -1), normal=r["normal"],
                    cls_before=r["cls_before"], cls_start=r["cls_start"], t_pass0=r["t_pass0"],
                    t_pass1=r["t_pass1"], vol_after=r["vol_after"], vol_before=r["vol_before"], vol_start=v0,
                    item=r["srf"], cont=r["cont"], vol_at=r["vol_at"],
                    cls_at=nav.dev()["cls"][r["vol_at"]],
                    win=jnp.stack([r["item"], r["leaf"], r["face"], r["zero"].astype(jnp.int32)]))  # fmt: skip

    def bind(self, ops):
        """Return this geometry with its navigator reading the operands ``ops`` (``Navigator.operands``)."""
        return TrackedGeometry(self.nav.bind(ops), self.n_classes, self.sensor_of_vol, self.vertex_tab)

    def surface(self, q, x, d):
        """Return (t, normal) of the surface query q met, live in the ray (x, d) (``navigate.surface_at``)."""
        return navigate.surface_at(self.nav, q, x, d)


def classes(nav, scene, optics, T=None):  # noqa: N803 -- T: the scene's instances
    """Return (class per volume, per mesh, the outside's, per volume its PV) of the optics on the scene's volumes.

    ``nav``: the navigator whose volumes must be these, in this order, by instance path (None: not checked). ``T``:
    the scene's ``instances``, when known.

    A split union's piece (``Definition.piece_of``) takes its original placement's class: the PV of the union's
    instance (the piece's parent in ``scene.instances``). Fails closed when the scene's PVs are not exactly the
    optics' (another loader's naming, a dropped volume).
    """
    T, pvs = sc.instances(scene) if T is None else T, []
    for i, dname in enumerate(T.defn):
        d = scene.definitions[dname]
        if d.member is None:
            continue
        j = int(T.parent[i]) if d.piece_of else i
        seg = sc.member_segment(d, d.piece_of or dname)
        pvs.append((f"{T.path[j]}/{seg}" if T.path[j] else seg, isinstance(d.member.solid, sc.Mesh), T.path[i]))
    have = {p for p, *_ in pvs} | {volume_tree.OUTSIDE_PV}
    want = set(optics.class_of)
    if have != want:
        raise GeometryError(f"the scene's PVs differ from the optics': {len(have - want)} not in the optics (e.g. "
                            f"{sorted(have - want)[:SHOWN]}), {len(want - have)} missing (e.g. "
                            f"{sorted(want - have)[:SHOWN]})")  # fmt: skip
    vols = [p for p, m, _ in pvs if not m]
    meshes = [p for p, m, _ in pvs if m]
    if nav is not None and (len(vols) != nav.n or len(meshes) != nav.n_mesh):
        raise GeometryError(f"navigator volumes {nav.n} / meshes {nav.n_mesh} vs PVs {len(vols)} / {len(meshes)}")
    paths = [path for _, m, path in pvs if not m]  # each volume's instance path, the navigator's name for it
    if nav is not None and list(nav.names) != paths:  # volume i of the optics is volume i of the navigator
        i = next(i for i, (a, b) in enumerate(zip(nav.names, paths, strict=True)) if a != b)
        raise GeometryError(f"navigator volume {i} is {nav.names[i]!r}, the optics' {paths[i]!r}: the class and "
                            "sensor tables would attach to other volumes")  # fmt: skip
    c = optics.class_of
    return [c[p] for p in vols], [c[p] for p in meshes], c[volume_tree.OUTSIDE_PV], vols


def from_scene(scene, optics, cache=False, precision="float32"):
    """Return the TrackedGeometry of a scene and its compiled optics, its navigator in ``precision``.

    Its navigator knows the classes: a volume of its listers' class (or one interchangeable with it,
    ``interchangeable``), no sensor and not of a self pair's class, bounds nothing a photon sees and is no candidate
    (``lumen.geometry.navigator_build``, transparent volumes).
    """
    T = sc.instances(scene)
    vol_cls, mesh_cls, out_cls, vols = classes(None, scene, optics, T)
    sensor_index = dict(optics.sensor_index)
    sens = [i for i, p in enumerate(vols) if p in sensor_index]
    vtx = vertices(optics)
    # never transparent: a sensor, and a volume of a class whose crossing into another of its volumes is a vertex
    is_sensor = set(sens)
    kept = [i for i, c in enumerate(vol_cls) if i in is_sensor or vtx[int(c), int(c)]]
    cl = navigator_build.Classes(tuple(int(c) for c in vol_cls), tuple(int(c) for c in mesh_cls), int(out_cls),
                                 tuple(kept), interchangeable(optics, vtx))  # fmt: skip
    nav = navigate.Navigator.from_scene(scene, cache=cache, classes=cl, precision=precision)
    classes(nav, scene, optics, T)
    sensor = np.full(nav.n + 1 + nav.n_mesh, -1, np.int32)
    for i in sens:
        sensor[i] = sensor_index[vols[i]]
    return TrackedGeometry(nav, optics.n_classes, sensor, vtx)


def from_parts(parts, cache=False, precision="float32"):
    """Return the TrackedGeometry of a built detector (``lumen.build.build.Parts``): its navigator scene and optics,
    the navigator in ``precision``."""  # noqa: D205, D209
    return from_scene(parts.nav_scene, parts.optics, cache=cache, precision=precision)
