"""The navigator with the surface lookup's classes: the geometry the step queries.

The step's photon carries its volume and the surface it sits on; the surface lookup (``lumen.build.surfaces``)
gives each placed volume a class, each sensor volume a tag, and the class pairs a photon crosses without a vertex (pass
pairs). This module joins the two: the class of every navigator volume by its placed-volume name, and the query in
the form the baseline's step reads.

    geom = from_parts(parts)                   # parts: lumen.build.build.build(...); or from_scene(scene, optics)
    q = geom.query(None, x, d, cls, t_max, vol=vol, last=srf, sn=sn)       # dict, the baseline's step contract
    t, n = geom.surface(None, q, x, d)         # the winner's (t, normal), live in (x, d)

Notes
-----
Ported from the baseline's lumen.geometry.tracked (from_scene, TrackedGeometry) and lumen.adapters.geometry
(pass_pairs). Not ported: the push contract (``eps_push``, ``t_min``), the overlap links (``overlaps=``,
``link_overlaps``), the pose match (``pose_alias``, ``instances=``) and the report (``lumen.validate.overlaps``).
"""

from __future__ import annotations

import re

import jax.numpy as jnp
import numpy as np

from lumen.build import tree as volume_tree
from lumen.geometry import navigate
from lumen.surfaces import _legacy as legacy


class GeometryError(ValueError):
    """The navigator's volumes and the optics' placed volumes do not match."""


SHOWN = 3  # error messages show at most this many names


def pass_pairs(optics):
    """Return (C, C) bool: the class pairs crossed without a vertex.

    Same medium, the PASS kernel, neither class a sensor's nor opaque (two classes of one medium split only by a role
    elsewhere: a detector's inner water and the water envelopes inside it). The navigator steps over them.
    """
    n = optics.n_classes
    med, ko = np.asarray(optics.medium_of), np.asarray(optics.kind_of)
    passk = {i for i, k in enumerate(optics.kinds) if k is legacy.PASS}
    tagged = {int(optics.class_of[pv]) for pv in optics.tag_pvs}
    opq = np.array([m.opaque for m in optics.media], bool)
    ok = ~opq[med.astype(int)] & ~np.isin(np.arange(n), list(tagged))
    return (
        ok[:, None] & ok[None, :] & (med[:, None] == med[None, :]) & np.isin(ko, list(passk)) & ~np.eye(n, dtype=bool)
    )


def piece_alias(pv):
    """Return a split union piece's ('<path>/pc<k>/<volume>') original placement's PV."""
    return re.sub(r"/pc\d+/(?=[^/]+$)", "/", pv)


class TrackedGeometry:
    """The navigator with a class per volume, a tag per volume (-1 none) and the pass pairs."""

    tracks_volume = True

    def __init__(self, nav, n_classes, tag_of_vol, pass_tab=None):
        self.nav = nav
        self.n_classes = int(n_classes)
        self.tag_of_vol = np.asarray(tag_of_vol, np.int32)  # (n volumes + 1 + n meshes,)
        self.pass_tab = None if pass_tab is None or not np.any(pass_tab) else np.asarray(pass_tab, bool)

    def __hash__(self):
        """Hash by identity (a static argument of the step)."""
        return id(self)

    def query(self, gp, x, d, cls, t_max, vol=-1, last=-1, sn=None):  # noqa: PLR0913 -- the step's contract
        """Return the next visible crossing in the baseline's step form (dict; ``gp`` and ``cls`` are not read).

        ``vol`` < 0: not known (a source's photon), located here, once; ``last``: the srf the photon sits on;
        ``sn``: the normal of the event that left it. ``tag``: the optics tag of the volume entered (-1 none);
        ``item``: the srf met; ``win``: the winner (item, leaf, face, zero) for ``surface``.
        """
        nav = self.nav
        vol = jnp.asarray(vol, jnp.int32)
        v0 = nav.locate(x, start=vol)  # the descent only where vol is not known
        r = nav.query(x, d, v0, last=jnp.where(vol < 0, -1, last), t_max=t_max, pass_tab=self.pass_tab, sn=sn)
        tags = jnp.asarray(self.tag_of_vol)
        return dict(t=r["t"], hit=r["hit"], cls_after=r["cls_after"], tag=jnp.where(r["hit"], tags[r["vol_after"]], -1),
                    normal=r["normal"], cls_before=r["cls_before"], cls_start=r["cls_start"], t_pass0=r["t_pass0"],
                    t_pass1=r["t_pass1"], vol_after=r["vol_after"], vol_before=r["vol_before"], vol_start=v0,
                    vol_reflect=r["vol_reflect"], item=r["srf"], cont=r["cont"],
                    win=jnp.stack([r["item"], r["leaf"], r["face"], r["zero"].astype(jnp.int32)]))  # fmt: skip

    def bind(self, ops):
        """Return this geometry with its navigator reading the operands ``ops`` (``Navigator.operands``)."""
        return TrackedGeometry(self.nav.bind(ops), self.n_classes, self.tag_of_vol, self.pass_tab)

    def surface(self, gp, q, x, d):
        """Return (t, normal) of the surface query q met, live in the ray (x, d) (``navigate.surface_at``)."""
        return navigate.surface_at(self.nav, q, x, d)


def classes(nav, scene, optics):
    """Return (class per volume, per mesh, the outside's, per volume its PV) of the optics on the navigator's volumes.

    A split union's piece takes its original placement's class. Fails closed when the scene's PVs are not exactly the
    optics' (another loader's naming, a dropped volume).
    """
    raw = [(x.pv, x.mesh) for x in volume_tree.instances(scene)]
    pvs = [(p if p in optics.class_of else piece_alias(p), m) for p, m in raw]
    have = {p for p, _ in pvs} | {volume_tree.OUTSIDE_PV}
    want = set(optics.class_of)
    if have != want:
        raise GeometryError(f"the scene's PVs differ from the optics': {len(have - want)} not in the optics (e.g. "
                            f"{sorted(have - want)[:SHOWN]}), {len(want - have)} missing (e.g. "
                            f"{sorted(want - have)[:SHOWN]})")  # fmt: skip
    vols = [p for p, m in pvs if not m]
    meshes = [p for p, m in pvs if m]
    if len(vols) != nav.n or len(meshes) != nav.n_mesh:
        raise GeometryError(f"navigator volumes {nav.n} / meshes {nav.n_mesh} vs PVs {len(vols)} / {len(meshes)}")
    c = optics.class_of
    return [c[p] for p in vols], [c[p] for p in meshes], c[volume_tree.OUTSIDE_PV], vols


def _check_coincident(nav, optics, vols):
    """Fail closed on coincident siblings of different classes with no interface between them.

    The navigator fills their region with the lowest index; leaving it the photon crosses into the highest, then out
    of that one by Geant4's zero-length step: the pair needs an interface (an opaque first one is never left).
    """
    ko = np.asarray(optics.kind_of)
    for grp in nav.coincident:
        ca, cb = optics.class_of[vols[grp[0]]], optics.class_of[vols[grp[-1]]]
        if ca != cb and (ko[ca] >= 0).any() and int(ko[ca, cb]) < 0:
            raise GeometryError(f"coincident siblings {[vols[i] for i in grp]}: no interface between them (classes "
                                f"{ca} -> {cb}); Geant4 crosses one leaving the first")  # fmt: skip


def from_scene(scene, optics, nav=None, cache=False):
    """Return the TrackedGeometry of a scene and its compiled optics (``nav``: its navigator, if already built)."""
    nav = navigate.Navigator.from_scene(scene, cache=cache) if nav is None else nav
    vol_cls, mesh_cls, out_cls, vols = classes(nav, scene, optics)
    nav = nav.with_classes(vol_cls, mesh_cls, out_cls)
    _check_coincident(nav, optics, vols)
    tag = np.full(nav.n + 1 + nav.n_mesh, -1, np.int32)
    tag_of = dict(optics.tag_of)
    for i, p in enumerate(vols):
        if p in tag_of:
            tag[i] = tag_of[p]
    return TrackedGeometry(nav, optics.n_classes, tag, pass_pairs(optics))


def from_parts(parts, cache=False):
    """Return the TrackedGeometry of a built detector (``lumen.build.build.Parts``): its navigator scene and optics."""
    return from_scene(parts.nav_scene, parts.optics, cache=cache)
