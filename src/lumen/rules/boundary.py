"""Geant4's boundary rule: the surface a photon meets crossing from one placed volume into the next.

    stype, origin, reason, side = crossing(v, pre, post, rel)
    stype, origin, reason, side = through(leg_pb, leg_bq, names)     # a coincident face across a pass

``v`` (``lumen.build.surfaces.Volumes``) gives per placed volume (PV) its medium, material, skin surface and sensor,
the border surfaces by ordered PV pair, the library entries and which media are opaque. For an ordered crossing
(pre, post), in Geant4's order:

1. entering a sensor PV whose rule is "any entry" (or across the border bound to its sensor): that sensor's type;
   a border sensor out of pre (an exit sensor: WCSim's Detection leaving the tube) is credited to pre;
2. the border surface of the ordered pair (pre, post): directional, as the GDML defines it;
3. the skins: post's then pre's when post is a daughter of pre (Geant4's enteredDaughter), otherwise pre's then
   post's;
4. the material default: the same medium is no boundary (Pass); post without RINDEX absorbs (NoRINDEX); otherwise
   the Fresnel interface of the two indices.

A polished dielectric_dielectric surface (found at 2 or 3) between two PVs of one material is skipped (SameMaterial).
A surface whose far-side outcome would put the photon into a medium without RINDEX: the Fresnel default absorbs
(NoRINDEX); a frontpainted transmission is absorbed where it lands (Geant4 transmits, then kills the photon at its
next step: ``legacy.into_opaque``); a stacked (coated) type never ends beyond its backing; any other is refused.

A daughter's face may lie ON its mother's boundary (SBND's ullage under the cryostat roof; a float32-thin shell): the
navigator then reports p -> q with the zero-thickness b between them skipped. Geant4 locates the photon directly in
q: one interface p -> q resolved by ``crossing``, except where a leg through b is a Pass (b one medium with p, or
with q, no surface between), which makes b optically p (or q): the crossing is the other leg's interface, its border
and sensor included (``through``). ``coincident`` says whether p -> q via b is such a face.

Notes
-----
Geant4 10.7 / 11.2: G4OpBoundaryProcess::PostStepDoIt (G4LogicalBorderSurface::GetSurface(pre, post) first, then
G4LogicalSkinSurface::GetSurface of post's logical volume when post's mother is pre, else pre's; the default
dielectric_dielectric polished; SameMaterial and Material2 without RINDEX for the polished finish only). The sensor
entry and the exit sensors are lumen's reading of a sensitive detector (WCSim's ProcessHits_boundary). Geant4 steps
into any b thicker than its 1e-12 m tolerance; a float32 geometry merges a b thinner than its band.
"""

from __future__ import annotations

from lumen.surfaces import types as legacy
from lumen.surfaces.outcomes import FAR_SIDE


class BoundaryError(ValueError):
    """A crossing Geant4's rule cannot give a surface for (a type that would end in an opaque medium)."""


OUTWARD = ("mother", "sibling", "outward")  # p inside b, q beyond it
INWARD = ("daughter", "sibling", "inward")  # p beyond b, q inside it


def _surface(v, pre, post, rel):
    """Return (type, origin, reason) of steps 2 to 4 (no sensor)."""
    surf, reason = v.borders.get((pre, post)), "border"
    if surf is None:
        order = (("post", post), ("pre", pre)) if rel == "daughter" else (("pre", pre), ("post", post))
        for side, n in order:
            if v.skin[n] is not None:
                surf, reason = v.skin[n], f"skin({side})"
                break
    if surf is not None:
        e = v.library.surfaces[surf]
        if e.ignored_between_same_material and v.material(pre) == v.material(post):
            return legacy.PASS, None, f"{reason} {surf}: same-material"
        return e.surface_type, f"surface:{surf}", reason
    if v.medium[pre] == v.medium[post]:
        return legacy.PASS, None, "default"
    if not v.opaque[v.medium[post]]:
        return legacy.DIELECTRIC, None, "default"
    return legacy.ABSORBER, None, "default: NoRINDEX"


def crossing(v, pre, post, rel):
    """Return (type, origin, reason, side credited) of the crossing pre -> post (``rel``: post relative to pre)."""
    lib = v.library
    if v.enters_sensor(pre, post):
        stype, origin, reason = lib.sensors[v.sensor[post]].surface_type, f"sensor:{v.sensor[post]}", "sensor"
    elif v.borders.get((pre, post)) in lib.sensors:  # an exit sensor: pre is a sensor PV
        s = v.borders[(pre, post)]
        return lib.sensors[s].surface_type, f"sensor:{s}", f"border sensor {s} (credited to pre)", "pre"
    else:
        stype, origin, reason = _surface(v, pre, post, rel)
    if v.opaque[v.medium[post]] and not stype.stack and any(FAR_SIDE[o] for o in stype.outcomes):
        if stype is legacy.DIELECTRIC:
            stype, origin, reason = legacy.ABSORBER, None, f"{reason}: NoRINDEX"
        elif legacy.is_painted(stype):
            stype, reason = legacy.into_opaque(stype), f"{reason}: transmitted into {v.medium[post]!r}, absorbed"
        else:
            raise BoundaryError(f"{pre!r} -> {post!r}: type {stype.name} ({reason}) would put the photon into the "
                                f"opaque medium {v.medium[post]!r}")  # fmt: skip
    if not stype.params:
        origin = None
    return stype, origin, reason, "post"


def coincident(inside_pb, inside_qb, rp, rq):
    """Return "outward" / "inward" when p -> q via b is a coincident face, else None.

    ``inside_pb``: p is a daughter of b; ``inside_qb``: q is; ``rp`` the relations of b to p met so far, ``rq`` those
    of q to b.
    """
    if inside_pb and rq & set(OUTWARD) and rp & {"mother", "outward"}:
        return "outward"
    if inside_qb and rp & set(INWARD) and rq & {"daughter", "inward"}:
        return "inward"
    return None


def through(leg_pb, leg_bq, b_name, names):
    """Return (type, origin, reason, side) of p -> q via b when one leg is a Pass: the other leg's interface.

    ``leg_pb`` / ``leg_bq``: the resolved legs (type, origin, reasons, ..., side), ``names`` the two class names of
    the leg used.
    """
    use = leg_bq if leg_pb[0] is legacy.PASS else leg_pb
    reason = f"pass via {b_name}: as {names[0]} -> {names[1]} ({use[2][0]})"
    return use[0], use[1], reason, use[5]
