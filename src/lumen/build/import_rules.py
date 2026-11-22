"""The general import rules: rewritten once at conversion, for every detector, whatever its source.

    df = import_rules.apply(df)          # after the recipe's decisions (``lumen.recipe.import_structure``)

- **Geant4-10 scintillation keys are translated** (``geant4_10``): a material's FASTCOMPONENT, SLOWCOMPONENT,
  FAST/SLOWTIMECONSTANT, FAST/SLOWSCINTILLATIONRISETIME and YIELDRATIO as Geant4 11's SCINTILLATIONCOMPONENTi,
  SCINTILLATIONTIMECONSTANTi, SCINTILLATIONRISETIMEi and SCINTILLATIONYIELDi (``lumen.rules.keys.
  geant4_11_scintillation``): the light Geant4 10.7.4's G4Scintillation emits from them (``geant4_10_light``), which
  Geant4 11 itself does not reproduce (its GDML reader adds the old names as new user keys, and G4Scintillation,
  finding no SCINTILLATIONCOMPONENT1, emits nothing). The translated record is read back by lumen's own reader and
  compared with a reference implementation of how Geant4 10.7 reads the original (``check_scintillation``); a
  translation that would change the light is never written (AssertionError). Each is a ``report`` line
  ("translated: ...") and an entry of ``provenance["translations"]`` (neither is in the content digest; the
  translated content is). Keys with no single reading are refused (``TranslationRefused``), naming what to add or
  delete. A recipe's decision sees the source's keys as they are (it runs before the rules).

The two rules below are what float32 tracking cannot resolve; each choice is recorded with ``origin="import"`` and a
note.

- **The band where a volume is.** Both rules compare a length with the navigator's band at the volume's own
  coordinate scale: the largest |coordinate| of its solid's box over every world placement of it (``_scales``), as the
  navigator takes its band at a crossing's coordinates.
- **Absorbed at entry.** A mesh volume's material whose ABSLENGTH is below the band of every mesh volume of it at
  every knot is bound opaque: Geant4 refracts a photon in and absorbs it
  within that length, and lumen's navigator does not track inside a mesh (the facet rule). Its surfaces still apply.
  Elsewhere such a medium is tracked as Geant4 tracks it (the photon refracts in and is absorbed on its first step).
- **Thin WLS boxes are refused** (docs/dev/decisions.md, 9). A box of a WLS material (WLSABSLENGTH) thinner than the
  band where it is placed is not resolved by the navigator, and lumen lowers no film onto a surface: the conversion
  is refused, and a recipe makes the box one the navigator resolves (a coated foil: split into a coating half and a
  substrate half, each thicker than the band).

The union split by leaf count is the navigator build's rule (``lumen.build.lowering.split_unions``).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from lumen.build import lowering
from lumen.format import schema
from lumen.geometry import solids as so
from lumen.rules import keys
from lumen.rules.navigation import band

ORIGIN = "import"


class ImportRuleError(ValueError):
    """A WLS box thinner than the band where it is placed."""


def apply(df: schema.DetectorFile) -> schema.DetectorFile:
    """Return the detector file with the general import rules applied (see the module docstring)."""
    return thin_wls(absorbed_at_entry(geant4_10(df)))


def _detector_lvs(df):
    lvs, stack = set(), [schema.detector_lv(df)]
    while stack:
        lv = stack.pop()
        if lv not in lvs:
            lvs.add(lv)
            stack += [df.placements[i].volume for i in df.volumes[lv].daughters]
    return lvs


def _scales(df, lvs) -> dict:
    """Return {lv: the largest |coordinate| (m) of its solid's box over every world placement of it} of ``lvs``."""
    mapper = lowering.SolidMapper(df)
    radius = {}
    for lv in lvs:
        bb = lowering.solid_box(mapper.solid(df.volumes[lv].solid)) if df.volumes[lv].solid else None
        radius[lv] = float(np.max(np.linalg.norm(so.corners(*bb), axis=1))) if bb is not None else 0.0
    holds: dict = {}  # lv -> whether its subtree places one of lvs (the walk descends only there)

    def has(lv):
        if lv not in holds:
            holds[lv] = False
            holds[lv] = lv in radius or any(has(df.placements[i].volume) for i in df.volumes[lv].daughters)
        return holds[lv]

    out = dict.fromkeys(lvs, 0.0)

    def walk(lv, lin, t):
        if lv in out:
            out[lv] = max(out[lv], float(np.max(np.abs(t))) + radius[lv])
        for i in df.volumes[lv].daughters:
            pl = df.placements[i]
            if has(pl.volume):
                lc, tc = lowering.placement_pose(pl)
                walk(pl.volume, lin @ lc, lin @ tc + t)

    walk(df.world, np.eye(3), np.zeros(3))
    return out


def absorbed_at_entry(df: schema.DetectorFile) -> schema.DetectorFile:
    """Bind opaque every mesh volume's material that absorbs within the band where its meshes are, at every knot."""
    lvs = _detector_lvs(df)
    mesh_lvs = [
        lv for lv in sorted(lvs) if df.volumes[lv].solid and df.solids[df.volumes[lv].solid].type == "tessellated"
    ]
    scale = _scales(df, mesh_lvs) if mesh_lvs else {}
    meshes = {df.volumes[lv].material for lv in mesh_lvs}
    mats, rep = dict(df.materials), list(df.report)
    for name in sorted(m for m in meshes if m in mats):
        a, o = mats[name].properties.get("ABSLENGTH"), mats[name].optics
        b = min(band(scale[lv]) for lv in mesh_lvs if df.volumes[lv].material == name)  # the finest band it is at
        if a is None or o.name != "geant4" or float(np.max(a.value)) >= b:
            continue
        note = (
            f"mesh material with ABSLENGTH < {b:.3g} m (the band where its meshes are) at every knot: absorbed at entry"
        )
        mats[name] = dataclasses.replace(
            mats[name], optics=schema.OpticsModel("lumen:opaque", settings={}, origin=ORIGIN, note=note)
        )
        rep.append(f"import: material {name} bound opaque ({note})")
    return dataclasses.replace(df, materials=mats, report=tuple(rep))


def _box(df, lv):
    s = df.solids.get(df.volumes[lv].solid or "")
    return None if s is None or s.type != "box" else np.array([s.params[k] for k in "xyz"], np.float64)


def thin_wls(df: schema.DetectorFile) -> schema.DetectorFile:
    """Refuse a box of a WLS material thinner than the band where it is placed (see the module docstring)."""
    cand = [lv for lv in sorted(_detector_lvs(df)) if _box(df, lv) is not None
            and "WLSABSLENGTH" in df.materials[df.volumes[lv].material].properties]  # fmt: skip
    scale = _scales(df, cand) if cand else {}
    thin = [lv for lv in cand if _box(df, lv).min() < band(scale[lv])]
    if thin:
        raise ImportRuleError(f"WLS boxes {thin} are thinner than the band where they are placed: the navigator does "
                              "not resolve them, and lumen lowers no film; thicken them in the recipe (a coated "
                              "foil: split it into a coating half and a substrate half)")  # fmt: skip
    return df


# ------------------------------------------------------------------------------- Geant4-10 scintillation keys
class TranslationRefused(ValueError):  # noqa: N818 (named like ImportRefused, OpticsRefused)
    """A record in a form with no single meaning to translate: refused, naming what to add or delete."""


G4_11_SCINTILLATION = frozenset(k for v in keys.SCINTILLATION_KEYS.values() for k in v)


def refuse_geant4_10(props: dict, what: str) -> None:
    """Refuse (TranslationRefused) Geant4-10 scintillation keys with no single reading, naming what to add or delete.

    Geant4 10.7.4's G4Scintillation::PostStepDoIt (default optics parameters: enhanced time constants off,
    excitation ratio 1): no FASTCOMPONENT and no SLOWCOMPONENT emits nothing; a component reads its
    FAST/SLOWTIMECONSTANT (GetConstProperty: a missing key aborts); two components read YIELDRATIO (aborts without it)
    and give the fast one G4int(min(YIELDRATIO, 1) N) photons, the slow one the rest (a negative ratio gives the slow
    one more than the yield: no physical reading). Keys of both releases have two readings (Geant4 10.7 reads the 10
    keys, Geant4 11 the 11 keys).
    """
    old = sorted(k for k in props if k in keys.G4_10_KEYS)
    if not old:
        return
    new = sorted(k for k in props if k in G4_11_SCINTILLATION)
    if new:
        raise TranslationRefused(f"{what} has the Geant4-10 scintillation keys {old} and the Geant4 11 keys {new}: "
                                 "Geant4 10.7 reads only the first, Geant4 11 only the second; delete one of the two "
                                 "sets")  # fmt: skip
    present = [c for c in ("FAST", "SLOW") if f"{c}COMPONENT" in props]
    if not present:
        raise TranslationRefused(f"{what} has the Geant4-10 scintillation keys {old} without a FASTCOMPONENT or "
                                 "SLOWCOMPONENT spectrum (Geant4 10.7 emits no light from them): add the spectrum, or "
                                 "delete the keys")  # fmt: skip
    need = [f"{c}TIMECONSTANT" for c in present if f"{c}TIMECONSTANT" not in props]
    if len(present) == 2 and "YIELDRATIO" not in props:
        need.append("YIELDRATIO")
    if need:
        raise TranslationRefused(f"{what} has the Geant4-10 scintillation keys {old} without {need} (Geant4 10.7 "
                                 f"aborts at its first scintillating step): add {need}")  # fmt: skip
    if len(present) == 2:
        r = np.asarray(props["YIELDRATIO"].value, np.float64)
        if r.ndim or not np.isfinite(r) or r < 0.0:
            raise TranslationRefused(f"{what} has YIELDRATIO {r.tolist()}: Geant4 10.7 gives the fast component "
                                     "min(YIELDRATIO, 1) of the light; give a finite constant >= 0 (the fast "
                                     "share)")  # fmt: skip


def geant4_10_light(props: dict) -> list:
    """Return the light Geant4 10.7.4's G4Scintillation emits from a material's Geant4-10 keys: per component,
    (spectrum Property, decay time ns, rise time ns or None, share of the yield). The reference reading the
    translation is checked against, written from G4Scintillation::PostStepDoIt independently of the translation.

    With FASTCOMPONENT and SLOWCOMPONENT (N_timeconstants = 2): the fast one first, with FASTTIMECONSTANT (and
    FASTSCINTILLATIONRISETIME, read under the finite-rise-time switch), numPhot = G4int(min(yieldRatio, 1) *
    fNumPhotons); then the slow one with SLOWTIMECONSTANT, fNumPhotons - numPhot. With one of them: that one with its
    own time constants, all the photons. The yield (SCINTILLATIONYIELD) is the same key in 10.7 and 11.
    """  # noqa: D205
    fast, slow = props.get("FASTCOMPONENT"), props.get("SLOWCOMPONENT")
    if fast is not None and slow is not None:
        r = min(float(props["YIELDRATIO"].value), 1.0)
        parts = [("FAST", fast, r), ("SLOW", slow, 1.0 - r)]
    elif fast is not None:
        parts = [("FAST", fast, 1.0)]
    else:
        parts = [("SLOW", slow, 1.0)]
    rise = lambda c: float(props[f"{c}SCINTILLATIONRISETIME"].value) if f"{c}SCINTILLATIONRISETIME" in props else None  # noqa: E731
    return [(spec, float(props[f"{c}TIMECONSTANT"].value), rise(c), share) for c, spec, share in parts]


def check_scintillation(old: dict, new: dict, what: str) -> None:
    """Check that lumen's scintillation source reads the translated keys ``new`` as Geant4 10.7 reads ``old``
    (``geant4_10_light``): the same components in order, each spectrum's energies and values bit for bit, its decay
    and rise times exactly, its share to float32 rounding (lumen computes it in float32), and the same
    SCINTILLATIONYIELD. Raises AssertionError naming the difference."""  # noqa: D205, D209
    from lumen.sources import scintillation as sc  # lazily: the sources import the rules

    k = keys.SCINTILLATION_KEYS
    ref = geant4_10_light(old)
    comps = keys.components(lambda key: key in new)
    if len(comps) != len(ref):
        raise AssertionError(f"{what}: Geant4 10.7's {len(ref)} components read as {len(comps)}")
    f = np.asarray(sc.shares(lambda key, d: float(new[key].value) if key in new else d, comps), np.float64)
    for i, (spec, tau, rise, share) in zip(comps, ref, strict=True):
        s = new[k["spectra"][i]]
        if not (np.array_equal(s.value, spec.value) and np.array_equal(s.energy, spec.energy)):
            raise AssertionError(f"{what}: component {i + 1}'s spectrum differs from Geant4 10.7's")
        if float(new[k["times"][i]].value) != tau:
            raise AssertionError(f"{what}: component {i + 1}'s decay time {float(new[k['times'][i]].value)} != {tau}")
        got = float(new[k["rises"][i]].value) if k["rises"][i] in new else None
        if got != rise:
            raise AssertionError(f"{what}: component {i + 1}'s rise time {got} != {rise}")
        if abs(f[i] - share) > 1e-6:
            raise AssertionError(f"{what}: component {i + 1}'s share {f[i]} != Geant4 10.7's {share}")
    y_old, y_new = old.get("SCINTILLATIONYIELD"), new.get("SCINTILLATIONYIELD")
    if (y_old is None) != (y_new is None) or (y_old is not None and not np.array_equal(y_old.value, y_new.value)):
        raise AssertionError(f"{what}: SCINTILLATIONYIELD changed")
    if any(key in new for key in keys.G4_10_KEYS):
        raise AssertionError(f"{what}: Geant4-10 keys left after the translation")


def _same_light_both(props: dict, what: str):
    """Return, for a material with both releases' scintillation keys that give the same light (lumen's export writes
    both: ``lumen.rules.keys.geant4_10_scintillation``), the rule dropping the Geant4-10 keys; else None (one set, or
    two readings: ``refuse_geant4_10`` refuses it)."""  # noqa: D205, D209
    old = sorted(k for k in props if k in keys.G4_10_KEYS)
    if not old or not any(k in G4_11_SCINTILLATION for k in props):
        return None
    ten = {k: v for k, v in props.items() if k in keys.G4_10_KEYS or k == "SCINTILLATIONYIELD"}
    eleven = {k: v for k, v in props.items() if k not in keys.G4_10_KEYS}
    try:
        refuse_geant4_10(ten, what)
        check_scintillation(ten, eleven, what)
    except (TranslationRefused, AssertionError):
        return None

    def drop(m):
        rec = dict(translation="geant4_10_scintillation", material=m.name, old=old, new=[], unread_by_geant4_10=[],
                   reference="Geant4 10.7.4 G4Scintillation::PostStepDoIt (lumen.build.import_rules.geant4_10_light)",
                   check="the Geant4-10 keys give the Geant4-11 keys' light: dropped")  # fmt: skip
        return (dataclasses.replace(m, properties=eleven), rec,
                f"{what}: Geant4-10 scintillation keys {old} beside Geant4 11's, the same light; dropped")

    return drop


def geant4_10_scintillation(name: str, m: schema.Material):
    """Return (material, translation record or None, report line or None) for a material's Geant4-10 keys."""
    what = f"material {name!r}"
    both = _same_light_both(m.properties, what)
    if both is not None:
        return both(m)
    refuse_geant4_10(m.properties, what)
    props, note = keys.geant4_11_scintillation(m.properties)
    if note is None:
        return m, None, None
    check_scintillation(m.properties, props, what)
    old = sorted(k for k in m.properties if k in keys.G4_10_KEYS)
    new = sorted(k for k in props if k not in m.properties)
    read = {"FASTCOMPONENT", "FASTTIMECONSTANT", "FASTSCINTILLATIONRISETIME"} if "FASTCOMPONENT" in m.properties \
        else set()  # fmt: skip
    if "SLOWCOMPONENT" in m.properties:
        read |= {"SLOWCOMPONENT", "SLOWTIMECONSTANT", "SLOWSCINTILLATIONRISETIME"}
    if {"FASTCOMPONENT", "SLOWCOMPONENT"} <= set(m.properties):
        read.add("YIELDRATIO")
    unread = [k for k in old if k not in read]  # keys Geant4 10.7 reads in no case of this material (dropped)
    rec = dict(translation="geant4_10_scintillation", material=name, old=old, new=new, unread_by_geant4_10=unread,
               reference="Geant4 10.7.4 G4Scintillation::PostStepDoIt (lumen.build.import_rules.geant4_10_light)",
               check="components, spectra, decay and rise times, shares and yield equal to the reference")  # fmt: skip
    line = f"{what}: {note}; checked against Geant4 10.7's reading"
    if unread:
        line += f"; {unread} read by Geant4 10.7 in no case here, dropped"
    return dataclasses.replace(m, properties=props), rec, line


def geant4_10(df: schema.DetectorFile) -> schema.DetectorFile:
    """Return the detector with every material's Geant4-10 scintillation keys translated, each reported and recorded
    (see the module docstring); keys with no single reading are refused (TranslationRefused, naming what to change).
    A detector with nothing to translate is returned as it is."""  # noqa: D205, D209
    mats, recs, rep = dict(df.materials), [], list(df.report)
    for name in df.materials:
        m, rec, line = geant4_10_scintillation(name, mats[name])
        mats[name] = m
        if rec:
            recs.append(rec)
            rep.append(f"translated: {line}")
    if not recs:
        return df
    prov = dict(df.provenance, translations=[*df.provenance.get("translations", []), *recs])
    return dataclasses.replace(df, materials=mats, provenance=prov, report=tuple(rep))
