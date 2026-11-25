"""Geant4's boundary rule: the surface a photon meets crossing from one placed volume into the next.

    stype, origin, reason, side = crossing(v, pre, post, rel)
    stype, origin, reason, side = through(leg_pb, leg_bq, b_name, names)  # a coincident face across a pass

``v`` (``lumen.build.lookup.Description``) gives per placed volume (PV) its medium, material, skin surface and
sensor, the border surface of an ordered PV pair (``border(pre, post)``: its physvols' border, where they touch, or
for physvols that meet only across coincident faces between their nearest instances), the
library entries and which media are opaque. For an ordered crossing (pre, post), in Geant4's order:

1. entering a sensor PV whose rule is "any entry" (or across the border bound to its sensor, or where Geant4 picks
   its own detecting skin): that sensor's type; a border sensor out of pre (an exit sensor: WCSim's Detection leaving
   the tube) is credited to pre. Leaving a skin sensor, its skin is the surface and detects nothing
   (G4OpBoundaryProcess::InvokeSD calls the sensitive detector of the volume entered);
2. the border surface of the ordered pair (pre, post): directional, as the GDML defines it;
3. the skins: post's then pre's when post is a daughter of pre (Geant4's enteredDaughter), otherwise pre's then
   post's;
4. the material default: the same medium is no boundary (Pass); post without RINDEX absorbs (NoRINDEX); otherwise
   the Fresnel interface of the two indices.

A polished dielectric_dielectric surface (found at 1, 2 or 3) between two PVs of one material is skipped (SameMaterial).
A surface whose far-side outcome would put the photon into a medium without RINDEX follows Geant4's rule for its
(type, finish), which the reader records on the entry (``far_side``): the default and a polished or ground
dielectric_dielectric surface, detecting ones included (an exit sensor's too), kill the photon (NoRINDEX, before the
R / T draw: ``FAR_KILL``); a frontpainted transmission is absorbed where it lands (Geant4 transmits, then kills the
photon at its next step: ``FAR_ABSORB``, ``lumen.surfaces.types.into_opaque``); an entry with no Geant4 reading (a
registered model's) is refused.

A daughter's face may lie ON its mother's boundary (a gas layer under a vessel's roof; a float32-thin shell): the
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

from lumen.surfaces import types
from lumen.surfaces.outcomes import FAR_SIDE


class BoundaryError(ValueError):
    """A crossing Geant4's rule cannot give a surface for (a type that would end in an opaque medium)."""


FAR_KILL, FAR_ABSORB = "kill", "absorb"  # a surface's far side into a medium without RINDEX (SurfaceEntry.far_side)
OUTWARD = ("mother", "sibling", "outward")  # p inside b, q beyond it
INWARD = ("daughter", "sibling", "inward")  # p beyond b, q inside it


def _surface(v, pre, post, rel):
    """Return (type, origin, reason, far side) of steps 2 to 4 (no sensor)."""
    surf, reason = v.border(pre, post), "border"
    if surf is None:
        order = (("post", post), ("pre", pre)) if rel == "daughter" else (("pre", pre), ("post", post))
        for side, n in order:
            if v.skin[n] is not None:
                surf, reason = v.skin[n], f"skin({side})"
                break
    if surf is not None:
        e = v.library.surfaces[surf]
        if e.ignored_between_same_material and v.material(pre) == v.material(post):
            return types.PASS, None, f"{reason} {surf}: same-material", None
        return e.surface_type, f"surface:{surf}", reason, e.far_side
    if v.medium[pre] == v.medium[post]:
        return types.PASS, None, "default", None
    if not v.opaque[v.medium[post]]:
        return types.DIELECTRIC, None, "default", FAR_KILL
    return types.ABSORBER, None, "default: NoRINDEX", None


def crossing(v, pre, post, rel):
    """Return (type, origin, reason, side credited) of the crossing pre -> post (``rel``: post relative to pre).

    A sensor's surface is read as any surface is: skipped between two volumes of one material where its reading says
    so (SameMaterial, before any detection), and its far side into an opaque medium as its reading records, an exit
    sensor's too.
    """
    lib, side, name = v.library, "post", None
    if v.enters_sensor(pre, post):
        name, reason = v.sensor[post], "sensor"
    elif v.border(pre, post) in lib.sensors:  # an exit sensor: pre is a sensor PV
        name = v.border(pre, post)
        side, reason = "pre", f"border sensor {name} (credited to pre)"
    if name is None:
        stype, origin, reason, far = _surface(v, pre, post, rel)
    elif lib.sensors[name].ignored_between_same_material and v.material(pre) == v.material(post):
        return types.PASS, None, f"{reason}: same-material", "post"
    else:
        stype, origin, far = lib.sensors[name].surface_type, f"sensor:{name}", lib.sensors[name].far_side
    if v.opaque[v.medium[post]] and (any(FAR_SIDE[o] for o in stype.outcomes) or stype.refracts):
        if far == FAR_KILL:
            stype, origin, reason = types.ABSORBER, None, f"{reason}: NoRINDEX"
        elif far == FAR_ABSORB:
            stype, reason = types.into_opaque(stype), f"{reason}: transmitted into {v.medium[post]!r}, absorbed"
        else:
            raise BoundaryError(f"{pre!r} -> {post!r}: type {stype.name} ({reason}) would put the photon into the "
                                f"opaque medium {v.medium[post]!r}")  # fmt: skip
    if not stype.params:
        origin = None
    return stype, origin, reason, side


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
    use = leg_bq if leg_pb[0] is types.PASS else leg_pb
    reason = f"pass via {b_name}: as {names[0]} -> {names[1]} ({use[2][0]})"
    return use[0], use[1], reason, use[5]
