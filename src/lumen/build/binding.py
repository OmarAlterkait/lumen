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

import dataclasses

import numpy as np

from lumen._core.errors import LibraryError
from lumen.build import lowering
from lumen.format import schema
from lumen.format.check import check_bindings
from lumen.geometry import solids as so
from lumen.rules.tolerances import BAND_FLOOR, CONTACT, POSE_ATOL


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


def _outside_chain(df) -> tuple[tuple[int, ...], str]:
    """Return (the placement chain, its LV) of the volume a photon leaving the detector volume enters
    (``outside_volume``); the world's chain is (). An assembly is no volume to Geant4: its mother's is entered."""  # noqa: D205, D209
    chain = schema.placements_of(df, df.detector_volume)[0]
    i, d = chain[-1], df.placements[chain[-1]]
    sibs = [j for j in df.volumes[d.mother].daughters if j != i]
    enclosing = [j for j in sibs if _encloses(df, df.placements[j], d)]
    out = chain[:-1] + (enclosing[0],) if len(enclosing) == 1 else chain[:-1]
    while out and df.volumes[df.placements[out[-1]].volume].assembly:
        out = out[:-1]
    return out, (df.placements[out[-1]].volume if out else df.world)


def transporting(df, lv) -> bool:
    """Return whether Geant4 transports an optical photon in volume ``lv``: its material has RINDEX (without, a photon
    entering it is killed, NoRINDEX: G4OpBoundaryProcess 10.7.4)."""  # noqa: D205, D209
    m = df.volumes[lv].material
    return m is not None and "RINDEX" in df.materials[m].properties


ABSORBED_AT_ENTRY = BAND_FLOOR  # m: an ABSLENGTH below the smallest band absorbs a photon on its first step in


def absorbed_at_entry(df, lv) -> bool:
    """Return whether volume ``lv``'s material absorbs a photon as it enters: its ABSLENGTH is below
    ``ABSORBED_AT_ENTRY`` at every knot (ANNIE's StainlessSteel and Air, 1e-11 m); none crosses it."""  # noqa: D205, D209
    p = df.materials[df.volumes[lv].material].properties.get("ABSLENGTH")
    return p is not None and float(np.max(np.asarray(p.value, np.float64))) < ABSORBED_AT_ENTRY


def detector_volume_closed(df) -> tuple[str, tuple[str, ...]]:
    """Return (the detector volume, the report) closed as Geant4 transports light: while a photon leaving the
    detector volume enters a volume Geant4 transports it in (``transporting``), the detector volume grows to that
    volume. lumen absorbs a photon leaving the detector volume (after the outside's surface, ``outside_skin``); Geant4
    absorbs it there only where it enters a volume without RINDEX, so a smaller detector volume would drop light
    Geant4 brings back (Super-K's WCBarrel in WC's air: reflected at the water's outer face). A volume that absorbs
    a photon as it enters (``absorbed_at_entry``) is kept, for the reflection at its face, and closes the detector: no
    light crosses it. The world is closed."""  # noqa: D205, D209
    report = []
    while df.detector_volume != df.world:
        chain, lv = _outside_chain(df)
        if not transporting(df, lv):
            break
        spec = df.world if not chain else lv if len(schema.placements_of(df, lv)) == 1 else \
            schema.placement_path(df, chain)  # fmt: skip
        stop = absorbed_at_entry(df, lv)
        report.append(f"detector volume {df.detector_volume} -> {spec}: a photon leaving it enters {lv} "
                      f"({df.volumes[lv].material}, with RINDEX), where Geant4 transports it on"
                      + (f"; {df.volumes[lv].material} absorbs it at entry (ABSLENGTH < {ABSORBED_AT_ENTRY} m): "
                         "none crosses, the detector is closed" if stop else ""))  # fmt: skip
        df = dataclasses.replace(df, detector_volume=spec)
        if stop:
            break
    return df.detector_volume, tuple(report)


def outside_skin(df) -> str | None:
    """Return the surface of the outside region: the skin of ``outside_volume`` (a recipe's replacing the GDML's)."""
    x = outside_volume(df)
    return None if x is None else effective_skins(df).get(x)
