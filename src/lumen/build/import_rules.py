"""The general import rules: what float32 tracking cannot resolve, rewritten once at conversion, for every detector.

    df = import_rules.apply(df)          # after the recipe's decisions (``lumen.Recipe.apply``)

Each choice is recorded with ``origin="import"`` and a note.

- **Absorbed at entry.** A mesh volume's material whose ABSLENGTH is below the band floor (``BAND_FLOOR``, the
  smallest length the navigator resolves) at every knot is bound opaque: Geant4 refracts a photon in and absorbs it
  within that length, and lumen's navigator does not track inside a mesh (the facet rule). Its surfaces still apply.
  Elsewhere such a medium is tracked as Geant4 tracks it (the photon refracts in and is absorbed on its first step).
- **One-sided WLS films** (docs/dev/decisions.md, 9). A box of a WLS material (WLSABSLENGTH) thinner than the band
  at its first placement, with no surface of its own, is a film on the face it touches: per placement, its sibling boxes
  in its mother (turned by signed permutations, as is the film) are searched for a contact; the contact set must be
  exactly one complete flat face of one backing, else the conversion is refused. Every placement of the backing's volume
  must carry a film on the same face of its own frame (a film on both faces is refused). The film material is bound to
  its mother's material (index-matched: checked when the film is built), and the backing's skin becomes a
  ``lumen:wls_film`` surface named after the film volume, with that face's normal as ``local_axis`` (the build turns it
  by each placement's world rotation: ``lumen.build.library.split_films``). Geant4 never attaches a film to a face: its
  effect follows from the face the film volume touches.

The union split by leaf count is the navigator build's rule (``lumen.build.lowering.split_unions``).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from lumen.build import lowering
from lumen.format import schema
from lumen.rules.navigation import BAND_FLOOR, BAND_ULPS, FLOAT32_EPS

ORIGIN = "import"
CONTACT = 1e-12  # m: two GDML lengths that agree to this are equal (1e-9 mm)


class ImportRuleError(ValueError):
    """A WLS film whose contact is not exactly one complete flat face per placement."""


def apply(df: schema.DetectorFile) -> schema.DetectorFile:
    """Return the detector file with the general import rules applied (see the module docstring)."""
    return wls_films(absorbed_at_entry(df))


def _detector_lvs(df):
    lvs, stack = set(), [df.detector_volume]
    while stack:
        lv = stack.pop()
        if lv not in lvs:
            lvs.add(lv)
            stack += [df.placements[i].volume for i in df.volumes[lv].daughters]
    return lvs


def absorbed_at_entry(df: schema.DetectorFile) -> schema.DetectorFile:
    """Bind opaque every mesh volume's material that absorbs within the band floor at every knot."""
    lvs = _detector_lvs(df)
    meshes = {df.volumes[lv].material for lv in lvs
              if df.volumes[lv].solid and df.solids[df.volumes[lv].solid].type == "tessellated"}  # fmt: skip
    mats, rep = dict(df.materials), list(df.report)
    for name in sorted(m for m in meshes if m in mats):
        a, o = mats[name].properties.get("ABSLENGTH"), mats[name].optics
        if a is None or o.name != "geant4" or float(np.max(a.value)) >= BAND_FLOOR:
            continue
        note = f"mesh material with ABSLENGTH < {BAND_FLOOR} m at every knot: absorbed at entry"
        mats[name] = dataclasses.replace(mats[name], optics=schema.OpticsModel("lumen:opaque", 1, {"medium": name},
                                                                               ORIGIN, note))  # fmt: skip
        rep.append(f"import: material {name} bound opaque ({note})")
    return dataclasses.replace(df, materials=mats, report=tuple(rep))


def _box(df, lv):
    s = df.solids.get(df.volumes[lv].solid or "")
    return None if s is None or s.type != "box" else np.array([s.params[k] for k in "xyz"], np.float64)


def _pose(df, i):
    """Return (half extents in the mother's frame, centre, matrix, exact) of a box placement, or None (not a box).

    ``exact``: the placement turns the box by a signed permutation, so its extents in the mother's frame are its faces.
    """
    p, b = df.placements[i], _box(df, df.placements[i].volume)
    if b is None:
        return None
    m, t = lowering.placement_pose(p)
    r = np.round(m)
    exact = bool(np.allclose(m, r, rtol=0.0, atol=CONTACT)) and np.allclose(np.abs(r).sum(0), 1.0)
    return np.abs(r if exact else m) @ (b / 2), t, r, exact


def _contacts(df, i):
    """Return [(sibling placement, the face's normal in the sibling's frame)]: film placement i's large faces on a
    sibling's faces."""  # noqa: D205, D209
    p = df.placements[i]
    hf, tf, _, exact = _pose(df, i)
    if not exact:
        raise ImportRuleError(f"WLS film {p.name!r} is turned off the axes of its mother: not a flat face contact")
    k, rest, out = int(np.argmin(hf)), [a for a in range(3) if a != int(np.argmin(hf))], []
    for j in df.volumes[p.mother].daughters:
        q = _pose(df, j) if j != i else None
        if q is None:
            continue
        hb, tb, rb, ok = q
        for s in (1.0, -1.0):  # the film's face -s on the sibling's face +s (the film on its +s side)
            if abs(tf[k] - s * hf[k] - (tb[k] + s * hb[k])) >= CONTACT:
                continue
            lo, hi = np.maximum(tf - hf, tb - hb)[rest], np.minimum(tf + hf, tb + hb)[rest]
            if np.all(hi > lo + CONTACT):  # the faces overlap: they must be one face
                if not (ok and np.all(np.abs((tf - tb)[rest]) < CONTACT) and np.all(np.abs((hf - hb)[rest]) < CONTACT)):
                    raise ImportRuleError(f"WLS film {p.name!r} covers part of a face of {df.placements[j].name!r}"
                                          " (or a turned face): not one complete flat face")  # fmt: skip
                out.append((j, tuple(float(x) + 0.0 for x in rb.T @ (s * np.eye(3)[k]))))
    return out


def _film_lvs(df):
    paths, out = lowering._first_paths(df), []
    for lv in sorted(_detector_lvs(df)):
        v, b = df.volumes[lv], _box(df, lv)
        if b is None or "WLSABSLENGTH" not in df.materials[v.material].properties:
            continue
        t = paths[lv][-1][2]
        if b.min() < BAND_ULPS * FLOAT32_EPS * float(np.max(np.abs(t))) + BAND_FLOOR:
            out.append(lv)
    return out


def _backings(df, films):
    """Return {backing LV: [(film LV, film placement, backing placement, local axis)]}, every contact checked."""
    out = {}
    for f in films:
        pvs = [i for i, p in enumerate(df.placements) if p.volume == f]
        if any(k.volume == f for k in df.skins) or any(df.placements[i].name in (b.pv1, b.pv2)
                                                        for b in df.borders for i in pvs):  # fmt: skip
            raise ImportRuleError(f"WLS film {f!r} carries an optical surface of its own")
        for i in pvs:
            c = _contacts(df, i)
            if len(c) != 1:
                raise ImportRuleError(f"WLS film placement {df.placements[i].name!r} touches {len(c)} faces "
                                      "(want exactly one complete flat face)")  # fmt: skip
            j, axis = c[0]
            out.setdefault(df.placements[j].volume, []).append((f, i, j, axis))
    for b, rows in out.items():
        placed = [j for j, p in enumerate(df.placements) if p.volume == b]
        if len({r[3] for r in rows}) != 1 or len({r[0] for r in rows}) != 1 or sorted(r[2] for r in rows) != placed:
            raise ImportRuleError(f"backing {b!r}: its placements do not all carry one film on one face of its "
                                  f"frame ({sorted({r[3] for r in rows})}; {len(rows)} films, {len(placed)} "
                                  "placements): a film on both faces, or on some placements only")  # fmt: skip
    return out


def wls_films(df: schema.DetectorFile) -> schema.DetectorFile:
    """Lower every one-sided WLS film onto its backing's face (see the module docstring)."""
    found = _backings(df, _film_lvs(df))
    mats, surfs, skins, rep = dict(df.materials), dict(df.surfaces), list(df.skins), list(df.report)
    for b, rows in found.items():
        f, i, _, axis = rows[0]
        front = df.volumes[df.placements[i].mother].material
        backing = next((k.surface for k in df.skins if k.volume == b), None)
        if backing is None:
            raise ImportRuleError(f"WLS film {f!r}: its backing {b!r} has no skin surface")
        m = df.volumes[f].material
        note = f"the WLS film {f!r} on the face {axis} of {b!r}"
        bind = schema.OpticsModel("lumen:same_as", 1, {"material": front}, ORIGIN, f"{note}: index-matched to {front}")
        mats[m] = dataclasses.replace(mats[m], optics=bind)
        settings = dict(film_material=m, front_material=front, backing=backing,
                        thickness_m=float(_box(df, f).min()), local_axis=list(axis))  # fmt: skip
        surfs[f] = schema.Surface(f, "", "", "", 0.0, {}, schema.OpticsModel("lumen:wls_film", 1, settings, ORIGIN,
                                                                             note))  # fmt: skip
        skins.append(schema.Skin(f, f, b, ORIGIN, f"replaces {backing!r} on {b!r}"))
        rep.append(f"import: {note}: {m} bound to {front}; {b!r}'s skin is a film over {backing}")
    return dataclasses.replace(df, materials=mats, surfaces=surfs, skins=tuple(skins), report=tuple(rep))
