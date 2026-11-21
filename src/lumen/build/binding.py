"""Geant4's binding rules for skin and border surfaces: which skin applies, and the outside's surface.

The rules a structure must meet (one skin per logical volume, a border's physvol names unique, one border per ordered
pair) are checked with the structure (``lumen.format.check.check_bindings``); the exporter and the library call them
too, with their own error class. A recipe's or import rule's skin replaces the source's on its volume
(``effective_skins``).

The outside's surface (``outside_volume``, ``outside_skin``): leaving the detector volume, a photon meets the border
into the volume it enters, then the detector volume's skin, then that volume's skin (G4OpBoundaryProcess); that volume
is a sibling holding the detector volume in a hole of its solid, otherwise the mother.
"""

from __future__ import annotations

import numpy as np

from lumen._core.errors import LibraryError
from lumen.build import lowering
from lumen.format import schema
from lumen.format.check import check_bindings
from lumen.geometry import solids as so
from lumen.rules.tolerances import CONTACT, POSE_ATOL


def effective_skins(df) -> dict:
    """Return {LV: surface}: the GDML's skins, a recipe's or import rule's skin replacing the GDML's on its volume
    (two of a recipe's on one volume raise ``LibraryError``)."""  # noqa: D205, D209
    recipe = [k for k in df.skins if k.origin != "source"]
    check_bindings([(k.surface, k.volume) for k in recipe], (), (), LibraryError)
    out = {k.volume: k.surface for k in df.skins if k.origin == "source"}
    out.update({k.volume: k.surface for k in recipe})
    return out


def _encloses(df, sib, d) -> bool:
    """Return whether sibling ``sib`` encloses placement ``d``: its solid is a subtraction one of whose holes is ``d``'s
    solid at ``d``'s pose, the hole strictly inside the minuend (every face of ``d`` touches ``sib``).
    """  # noqa: D205
    mapper = lowering.SolidMapper(df)
    s_name = df.volumes[sib.volume].solid
    if s_name is None or df.solids[s_name].type != "subtraction":
        return False
    (ls, ts), (ld, td) = lowering.placement_pose(sib), lowering.placement_pose(d)
    d_solid = df.solids[df.volumes[d.volume].solid]

    def same(ref, lh, th):
        r = df.solids[ref]
        shape = ref == d_solid.name or (r.type == d_solid.type and not r.refs and not d_solid.refs and set(r.params)
                                        == set(d_solid.params) and all(abs(r.params[k] - d_solid.params[k]) <= CONTACT
                                                                       for k in r.params))  # fmt: skip
        return shape and np.allclose(ls @ lh, ld, atol=POSE_ATOL) and np.allclose(ls @ th + ts, td, atol=POSE_ATOL)

    holes = [h for h in mapper.holes(s_name, same) if h[4]]
    if len(holes) != 1:
        return False
    _, ref, lh, th, _ = holes[0]
    s = df.solids[s_name]
    m1, t1 = lowering.boolean_pose(s.transforms[0])
    a = so.node_bbox(so.Xf(m1, t1, mapper.solid(s.refs[0])))
    b = so.node_bbox(so.Xf(lh, th, mapper.solid(ref)))
    return a is not None and b is not None and bool(np.all(a[0] < b[0] - CONTACT) and np.all(b[1] < a[1] - CONTACT))


def outside_volume(df) -> str | None:
    """Return the logical volume a photon leaving the detector volume enters (Geant4's post-step volume), or None.

    Leaving the detector placement D, Geant4 (G4OpBoundaryProcess::PostStepDoIt) looks up the border (D, X), then the
    skin of D, then the skin of X, where X is the volume the photon enters: a sibling of D whose solid holds D in a
    hole (``_encloses``: every face of D touches it), otherwise D's mother. None when the detector volume is the world
    (the detector volume selects one placement: ``lumen.build.lowering.find_detector``).
    """
    if df.detector_volume == df.world:
        return None
    chains = schema.placements_of(df, df.detector_volume)
    assert len(chains) == 1, (df.detector_volume, len(chains))  # find_detector refuses any other count
    i = chains[0][-1]
    d = df.placements[i]
    sibs = [df.placements[j] for j in df.volumes[d.mother].daughters if j != i]
    enclosing = [q for q in sibs if _encloses(df, q, d)]
    return enclosing[0].volume if len(enclosing) == 1 else d.mother


def outside_skin(df) -> str | None:
    """Return the surface of the outside region: the skin of ``outside_volume`` (a recipe's replacing the GDML's)."""
    x = outside_volume(df)
    return None if x is None else effective_skins(df).get(x)
