"""Reading and writing a detector file: a ``<name>.lumen/`` directory of canonical JSON and one ``.npy`` per array.

Canonical JSON: sorted keys, one-space indent, floats as the shortest repr that reads back exactly, and each record
of a list on one line, so two detector files compare with ``diff``. Every content file's sha256 is in the manifest
of ``detector.json``; the content digest is the sha256 of that manifest. A save goes to a temporary directory moved
into place, so a reader never sees half a detector file, and an existing detector file is never overwritten unless
asked. Reading refuses a newer major schema version, unknown required fields and a file whose sha256 differs from
the manifest.
"""

from __future__ import annotations

import hashlib
import io as _io
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from lumen.format import schema

SAFE = re.compile(r"^[A-Za-z0-9_.+-]+$")
CONTENT = (
    "materials.json",
    "elements.json",
    "solids.json",
    "volumes.json",
    "placements.json",
    "surfaces.json",
    "userinfo.json",
)


class FormatError(ValueError):
    """A detector file that cannot be read: a wrong schema, a missing or changed file, a broken reference."""


# ------------------------------------------------------------------------------------------------ canonical JSON
def _number(x: float) -> str:
    if not np.isfinite(x):
        raise FormatError(f"non-finite number {x!r}: the detector file stores finite numbers only")
    return repr(float(x))


def _flat(v) -> str:
    """One value on one line."""
    if isinstance(v, bool) or v is None:
        return json.dumps(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return _number(float(v))
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, dict):
        return "{" + ", ".join(f"{json.dumps(str(k), ensure_ascii=False)}: {_flat(v[k])}" for k in sorted(v)) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_flat(x) for x in v) + "]"
    raise FormatError(f"cannot store {type(v).__name__} in JSON")


def dumps(obj: dict) -> str:
    """Canonical JSON of a dict: top-level keys sorted; a dict value one entry per line, a list one item per line."""
    lines = ["{"]
    keys = sorted(obj)
    for i, k in enumerate(keys):
        v, sep = obj[k], "," if i < len(keys) - 1 else ""
        if isinstance(v, dict) and v:
            inner = sorted(v)
            lines.append(f" {json.dumps(k)}: {{")
            lines += [
                f"  {json.dumps(str(j), ensure_ascii=False)}: {_flat(v[j])}{',' if n < len(inner) - 1 else ''}"
                for n, j in enumerate(inner)
            ]
            lines.append(f" }}{sep}")
        elif isinstance(v, (list, tuple)) and v:
            lines.append(f" {json.dumps(k)}: [")
            lines += [f"  {_flat(x)}{',' if n < len(v) - 1 else ''}" for n, x in enumerate(v)]
            lines.append(f" ]{sep}")
        else:
            lines.append(f" {json.dumps(k)}: {_flat(v)}{sep}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def sha256_bytes(data: bytes) -> str:
    """Return 'sha256:' and the hex digest of data."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def npy_bytes(a: np.ndarray) -> bytes:
    """Return the .npy bytes of an array (deterministic: no pickles, C order)."""
    buf = _io.BytesIO()
    np.save(buf, np.ascontiguousarray(a), allow_pickle=False)
    return buf.getvalue()


# --------------------------------------------------------------------------------------------- records <-> JSON
class _Writer:
    """Collects the files of one detector file: JSON documents and arrays by relative path."""

    def __init__(self):
        self.files: dict[str, bytes] = {}

    def array(self, rel: str, a) -> str:
        if rel in self.files:
            raise FormatError(f"two arrays at {rel}")
        self.files[rel] = npy_bytes(np.asarray(a))
        return rel


def _array_dir(kind: str, name: str, used: dict) -> str:
    base = name if SAFE.match(name) else "x" + hashlib.sha256(name.encode()).hexdigest()[:16]
    while used.setdefault(f"{kind}/{base}", name) != name:
        base += "_"
    return f"arrays/{kind}/{base}"


def _aux(a: schema.Aux) -> list:
    return [a.type, a.value, a.unit, [_aux(c) for c in a.children]]


def _aux_back(v) -> schema.Aux:
    return schema.Aux(v[0], v[1], v[2], tuple(_aux_back(c) for c in v[3]))


def _props(w: _Writer, d: str, props: dict) -> dict:
    out = {}
    for key, p in props.items():
        rec = dict(
            matrix=p.matrix,
            origin=p.origin,
            note=p.note,
            value=w.array(f"{d}/{key}.value.npy", np.asarray(p.value, np.float64)),
            energy=None if p.energy is None else w.array(f"{d}/{key}.energy_eV.npy", np.asarray(p.energy, np.float64)),
        )
        out[key] = rec
    return out


def _optics(o: schema.OpticsModel) -> dict:
    return dict(name=o.name, version=o.version, settings=o.settings, origin=o.origin, note=o.note)


def to_files(df: schema.DetectorFile) -> dict[str, bytes]:
    """Return every file of the detector file by relative path (detector.json last, with the manifest)."""
    w, used = _Writer(), {}
    mats = {}
    for name, m in df.materials.items():
        c = m.composition
        mats[name] = dict(
            state=m.state,
            temperature_K=m.temperature,
            pressure_Pa=m.pressure,
            composition=dict(
                density_g_cm3=c.density_g_cm3,
                z=c.z,
                molar_mass_g_mol=c.molar_mass_g_mol,
                components=[[x.ref, x.fraction, x.atoms] for x in c.components],
            ),
            optics=_optics(m.optics),
            properties=_props(w, _array_dir("materials", name, used), m.properties),
        )
    elements = {
        n: dict(
            formula=e.formula,
            z=e.z,
            molar_mass_g_mol=e.molar_mass_g_mol,
            isotopes=[[x.ref, x.fraction, x.atoms] for x in e.isotopes],
        )
        for n, e in df.elements.items()
    }
    isotopes = {n: dict(z=i.z, n=i.n, molar_mass_g_mol=i.molar_mass_g_mol) for n, i in df.isotopes.items()}
    solids = {}
    for name, s in df.solids.items():
        rec = dict(type=s.type, params=s.params)
        if s.refs:
            rec["refs"] = list(s.refs)
            rec["transforms"] = [[list(t.rotation), list(t.position)] for t in s.transforms]
        for k in ("planes", "vertices", "sections"):
            if getattr(s, k):
                rec[k] = [list(x) for x in getattr(s, k)]
        if s.scale is not None:
            rec["scale"] = list(s.scale)
        if s.mesh is not None:
            rec["mesh"] = w.array(f"{_array_dir('meshes', s.mesh, used)}.npy", df.meshes[s.mesh])
        solids[name] = rec
    volumes = {
        n: dict(
            solid=v.solid,
            material=v.material,
            assembly=v.assembly,
            daughters=list(v.daughters),
            aux=[_aux(a) for a in v.aux],
        )
        for n, v in df.volumes.items()
    }
    pls = df.placements
    w.array("arrays/placements.position_m.npy", np.array([p.position for p in pls], np.float64).reshape(-1, 3))
    w.array("arrays/placements.rotation_rad.npy", np.array([p.rotation for p in pls], np.float64).reshape(-1, 3))
    w.array(
        "arrays/placements.scale.npy",
        np.array([p.scale if p.scale is not None else (0.0, 0.0, 0.0) for p in pls], np.float64).reshape(-1, 3),
    )
    surfaces = {
        n: dict(
            type=s.type,
            model=s.model,
            finish=s.finish,
            value=s.value,
            optics=_optics(s.optics),
            inert=list(s.inert),
            properties=_props(w, _array_dir("surfaces", n, used), s.properties),
        )
        for n, s in df.surfaces.items()
    }
    docs = {
        "materials.json": dict(materials=mats),
        "elements.json": dict(elements=elements, isotopes=isotopes),
        "solids.json": dict(solids=solids),
        "volumes.json": dict(world=df.world, volumes=volumes),
        "placements.json": dict(
            placements=[[p.name, p.volume, p.mother, p.copy] for p in pls], columns=["name", "volume", "mother", "copy"]
        ),
        "surfaces.json": dict(
            surfaces=surfaces,
            skins=[[k.name, k.surface, k.volume, k.origin, k.note] for k in df.skins],
            borders=[[b.name, b.surface, b.pv1, b.pv2, b.origin] for b in df.borders],
        ),
        "userinfo.json": dict(userinfo=[_aux(a) for a in df.userinfo]),
    }
    files = dict(w.files)
    files.update({k: dumps(v).encode() for k, v in docs.items()})
    manifest = {k: sha256_bytes(files[k]) for k in sorted(files)}
    head = dict(
        schema=df.schema,
        name=df.name,
        world=df.world,
        detector_volume=df.detector_volume,
        wavelength_range_nm=list(df.wavelength_range),
        outside=df.outside,
        outside_surface=df.outside_surface,
        provenance=df.provenance,
        report=list(df.report),
        manifest=manifest,
    )
    files["detector.json"] = dumps(head).encode()
    return files


def content_digest(files: dict[str, bytes]) -> str:
    """Return the content digest: sha256 of the canonical manifest (content files only)."""
    head = json.loads(files["detector.json"])
    return sha256_bytes(dumps(dict(manifest=head["manifest"])).encode())


def save(df: schema.DetectorFile, path, overwrite: bool = False) -> Path:
    """Write the detector file at ``path`` (a ``.lumen`` directory) atomically; refuse to overwrite unless asked."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FormatError(f"{path} exists: a detector file is never overwritten (pass overwrite=True)")
    files = to_files(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"))
    try:
        for rel, data in files.items():
            f = tmp / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(data)
        if path.exists():
            shutil.rmtree(path)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return path


# ------------------------------------------------------------------------------------------------------- reading
def _load_npy(root: Path, rel: str, manifest: dict) -> np.ndarray:
    data = (root / rel).read_bytes()
    if manifest.get(rel) != sha256_bytes(data):
        raise FormatError(f"{root}: {rel} does not match the manifest (changed since it was written)")
    return np.load(_io.BytesIO(data), allow_pickle=False)


def _props_back(root, d: dict, manifest) -> dict:
    return {
        k: schema.Property(
            k,
            _load_npy(root, r["value"], manifest),
            None if r["energy"] is None else _load_npy(root, r["energy"], manifest),
            r["matrix"],
            r["origin"],
            r["note"],
        )
        for k, r in d.items()
    }


def _optics_back(d: dict) -> schema.OpticsModel:
    return schema.OpticsModel(d["name"], d["version"], d["settings"], d["origin"], d["note"])


def _component(v) -> schema.Component:
    return schema.Component(v[0], v[1], v[2])


def read(path) -> schema.DetectorFile:
    """Read a detector file, checking its schema and every file against the manifest."""
    root = Path(path)
    head = json.loads((root / "detector.json").read_text())
    found = head.get("schema", "")
    major = found.rsplit("/", 1)[-1].split(".")[0] if "/" in found else ""
    if not found.startswith("lumen-detector/") or major != schema.SCHEMA.rsplit("/", 1)[-1]:
        raise FormatError(
            f"{root} is schema {found!r}; this lumen reads {schema.SCHEMA}. Upgrade lumen, or open it "
            "with the version that wrote it."
        )
    man = head["manifest"]

    def doc(name):
        data = (root / name).read_bytes()
        if man.get(name) != sha256_bytes(data):
            raise FormatError(f"{root}: {name} does not match the manifest (changed since it was written)")
        return json.loads(data)

    d = {name: doc(name) for name in CONTENT}
    materials = {}
    for n, m in d["materials.json"]["materials"].items():
        c = m["composition"]
        comp = schema.Composition(
            c["density_g_cm3"], tuple(_component(x) for x in c["components"]), c["z"], c["molar_mass_g_mol"]
        )
        materials[n] = schema.Material(
            n,
            m["state"],
            comp,
            m["temperature_K"],
            m["pressure_Pa"],
            _props_back(root, m["properties"], man),
            _optics_back(m["optics"]),
        )
    el = d["elements.json"]
    elements = {
        n: schema.Element(n, e["formula"], e["z"], e["molar_mass_g_mol"], tuple(_component(x) for x in e["isotopes"]))
        for n, e in el["elements"].items()
    }
    isotopes = {n: schema.Isotope(n, i["z"], i["n"], i["molar_mass_g_mol"]) for n, i in el["isotopes"].items()}
    solids, meshes = {}, {}
    for n, s in d["solids.json"]["solids"].items():
        mesh = None
        if "mesh" in s:
            mesh = n
            meshes[n] = _load_npy(root, s["mesh"], man)
        solids[n] = schema.Solid(
            n,
            s["type"],
            dict(s["params"]),
            tuple(s.get("refs", ())),
            tuple(schema.Transform(tuple(r), tuple(t)) for r, t in s.get("transforms", ())),
            None if "scale" not in s else tuple(s["scale"]),
            tuple(tuple(x) for x in s.get("planes", ())),
            tuple(tuple(x) for x in s.get("vertices", ())),
            tuple(tuple(x) for x in s.get("sections", ())),
            mesh,
        )
    vols = d["volumes.json"]
    volumes = {
        n: schema.Volume(
            n, v["solid"], v["material"], tuple(v["daughters"]), v["assembly"], tuple(_aux_back(a) for a in v["aux"])
        )
        for n, v in vols["volumes"].items()
    }
    pos = _load_npy(root, "arrays/placements.position_m.npy", man)
    rot = _load_npy(root, "arrays/placements.rotation_rad.npy", man)
    scl = _load_npy(root, "arrays/placements.scale.npy", man)
    placements = tuple(
        schema.Placement(
            nm,
            v,
            mo,
            cp,
            tuple(pos[i].tolist()),
            tuple(rot[i].tolist()),
            None if not np.any(scl[i]) else tuple(scl[i].tolist()),
        )
        for i, (nm, v, mo, cp) in enumerate(d["placements.json"]["placements"])
    )
    sf = d["surfaces.json"]
    surfaces = {
        n: schema.Surface(
            n,
            s["type"],
            s["model"],
            s["finish"],
            s["value"],
            _props_back(root, s["properties"], man),
            _optics_back(s["optics"]),
            tuple(s["inert"]),
        )
        for n, s in sf["surfaces"].items()
    }
    return schema.DetectorFile(
        name=head["name"],
        world=head["world"],
        detector_volume=head["detector_volume"],
        wavelength_range=tuple(head["wavelength_range_nm"]),
        solids=solids,
        volumes=volumes,
        placements=placements,
        materials=materials,
        elements=elements,
        isotopes=isotopes,
        surfaces=surfaces,
        skins=tuple(schema.Skin(*k) for k in sf["skins"]),
        borders=tuple(schema.Border(*b) for b in sf["borders"]),
        meshes=meshes,
        outside=head["outside"],
        outside_surface=head["outside_surface"],
        userinfo=tuple(_aux_back(a) for a in d["userinfo.json"]["userinfo"]),
        provenance=head["provenance"],
        report=tuple(head["report"]),
        schema=found,
    )


def equal(a: schema.DetectorFile, b: schema.DetectorFile) -> bool:
    """Whether two detector files are the same value (every file byte-identical)."""
    return to_files(a) == to_files(b)


__all__ = ["FormatError", "content_digest", "dumps", "equal", "read", "save", "to_files"]
