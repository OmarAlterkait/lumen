"""Source tokens: each length and energy exactly as the GDML wrote it, so an export makes Geant4 read the same doubles.

lumen keeps the exact decimal value of a literal (``lumen.gdml.units``: "2.5*eV" is 2.5 eV), while Geant4 multiplies
by the unit (fl(2.5) * fl(1e-6) = 2.4999999999999998e-06 MeV): one ulp apart for some literals. The detector file
therefore also keeps the source token of a value, and the exporter writes the token back while the stored value still
decodes from it bit for bit (not calibrated, not overlaid); a changed value is written canonically.

Sparse: a token is kept only where the canonical writer would write something else. A length element (a position,
or a solid's length attributes, planes and extrusion points, all under one unit) is kept when its unit is not mm or
one of its literals is not the decimal value the canonical writer gives; a matrix energy (or a length-key matrix
value) when it is not a bare literal of the canonical decimal value. This is a comparison of decimal strings, with no
emulation of Geant4's evaluator. Angles, unitless values and the other matrix values are not kept: lumen reads them as
Geant4 computes them, and the canonical literal (the shortest repr) reads back to the same double in Geant4.

An expression is kept as written when it evaluates without the file's defines (self-contained); one that names a
define is kept as its value before the unit, in parentheses (``"(12.700000000000001)"``), which the reader and Geant4
both evaluate to the same double (``fallbacks`` counts them). Not reproducible: a RELATIVE tessellated facet (Geant4
sums its vertices in mm, lumen in m; the export writes it ABSOLUTE), counted in ``relative_facets``.

    tokens = collect(doc, solids, placements, materials, surfaces)    # the detector file's ``tokens``
    {"solids": {S: {"unit", "params", "planes", "vertices", "sections"} | tet: {"unit", "points"}},
     "transforms": {S: {"<k>": [unit, [x, y, z]]}}, "placements": {"<i>": [unit, [x, y, z]]},
     "meshes": {S: [[triangle, vertex, unit, [x, y, z]], ...]},
     "materials" | "surfaces": {name: {KEY: {"energy": [...] | None, "value": [...] | None, "matrix_unit": ...}}}}
"""

from __future__ import annotations

import math
import re
from decimal import Decimal

import numpy as np

from lumen.gdml import reader
from lumen.gdml.units import ENERGY_EV, LENGTH_M, LITERAL, MM_PER_M, PRODUCT, exact, is_literal, shift

MM = "mm"
TINY = reader.TINY
DBL_DIG = 15  # decimal digits a double always round-trips (float.h)
_MANTISSA = re.compile(r"[^0-9]")


def evaluate(text: str):
    """Return the value of an expression with no defines (CLHEP units and functions only), or None."""
    try:
        return reader.Evaluator().eval(text)
    except (reader.GDMLError, ValueError, ZeroDivisionError, OverflowError):
        return None


def _plus_zero(values) -> bool:
    return all(x == 0.0 and math.copysign(1.0, x) > 0 for x in values)


def _normal(v) -> bool:
    return v is not None and math.isfinite(v) and (v == 0.0 or abs(v) >= TINY)


def _canonical(text: str, v: float, places: int) -> bool:
    """Whether ``text`` is a bare literal of the decimal value the canonical writer gives ``v``.

    A literal of at most ``DBL_DIG`` significant digits always is: ``v`` is its value shifted, rounded once, and the
    shortest repr of that double is that decimal (two decimals of 15 digits never round to one double), so only the
    longer ones are compared.
    """
    if not LITERAL.match(text) or not math.isfinite(v):
        return False
    digits = _MANTISSA.sub("", text.strip().split("e")[0].split("E")[0]).lstrip("0").rstrip("0")
    if len(digits) <= DBL_DIG:
        return True
    return Decimal(text.strip()) == Decimal(shift(repr(float(v)), places))


class Collector:
    """Collects the tokens of one document, and counts what cannot be kept as written."""

    def __init__(self, doc: reader.GDMLDocument):
        self.doc = doc
        self.out: dict = {}
        self.fallbacks = 0
        self.relative_facets = 0

    def _text(self, text: str, what: str):
        """Return a token text: as written if self-contained, else its value in parentheses (None: not storable)."""
        if is_literal(text) or PRODUCT.match(text):
            return text
        v = self.doc.num(text, what)
        fresh = evaluate(text)
        if fresh is not None and fresh == v and _normal(v):
            return text
        if not _normal(v):
            return None
        self.fallbacks += 1
        return f"({v!r})"

    def element(self, unit, texts, values) -> list | None:
        """Return the token [unit, texts] of lengths under one unit (values in m), or None when canonical."""
        unit = (unit or MM).strip()
        if unit == MM and all(_canonical(t, v, 3) for t, v in zip(texts, values, strict=True)):
            return None
        out = [self._text(t, "length") for t in texts]
        return None if any(t is None for t in out) else [unit, out]

    def table(self, toks: list, internal, stored, places) -> list | None:
        """Return the tokens of a matrix column (None entries: canonical), or None when every entry is canonical."""
        if len(toks) != len(stored) or all(_canonical(t, v, places) for t, v in zip(toks, stored, strict=True)):
            return None
        out: list = []
        for t, g, v in zip(toks, internal, stored, strict=True):
            if _canonical(t, v, places) or not _normal(g):
                out.append(None)
            elif LITERAL.match(t) or PRODUCT.match(t) or (evaluate(t) is not None and evaluate(t) == g):
                out.append(t)
            else:
                self.fallbacks += 1
                out.append(f"({float(g)!r})")
        return out

    def put(self, kind: str, name, key, value) -> None:
        """Record a token (None: nothing to keep)."""
        if value is not None:
            self.out.setdefault(kind, {}).setdefault(name, {})[str(key)] = value


def _tet(c: Collector, s: reader.Solid) -> None:
    """Record a tet's token: its lunit and its four positions as written (Geant4 scales them by the lunit again)."""
    lunit = (s.texts.get("lunit") or MM).strip()
    pts = [((c.doc.position_text[r][0] or MM).strip(), list(c.doc.position_text[r][1]), c.doc.positions[r])
           for r in s.tet[0]]  # fmt: skip
    if lunit == MM and all(u == MM and all(_canonical(t, v, 3) for t, v in zip(tx, vals, strict=True))
                           for u, tx, vals in pts):  # fmt: skip
        return
    toks = [[u, [c._text(t, "tet vertex") for t in tx]] for u, tx, _ in pts]
    if all(t is not None for _, tx in toks for t in tx):
        c.out.setdefault("solids", {})[s.name] = dict(unit=lunit, points=toks)


def _solid(c: Collector, s: reader.Solid, rec) -> None:
    """Record a solid's token: its lunit and every length it holds (attributes, planes, extrusion points)."""
    t = s.texts
    names = list(t.get("params", {}))
    texts = [t["params"][k] for k in names]
    values = [rec.params[k] for k in names]
    for key, stored in (("planes", rec.planes), ("vertices", rec.vertices), ("sections", rec.sections)):
        for row, vals in zip(t.get(key, []), stored, strict=True):
            texts += list(row)
            values += list(vals[: len(row)])
    tok = c.element(t.get("unit"), texts, values)
    if tok is None:
        return
    it = iter(tok[1])
    out: dict = dict(unit=tok[0], params={k: next(it) for k in names})
    for key in ("planes", "vertices", "sections"):
        if t.get(key):
            out[key] = [[next(it) for _ in row] for row in t[key]]
    c.out.setdefault("solids", {})[s.name] = out


def _mesh(c: Collector, s: reader.Solid, mesh) -> None:
    pos, done, rows = c.doc.position_text, {}, []
    tri = 0
    for refs, ftype in s.facets:
        n = 2 if len(refs) == 4 else 1
        if ftype == "RELATIVE":
            c.relative_facets += 1
            tri += n
            continue
        for corner in ([0, 1, 2], [0, 2, 3])[:n]:
            for v, i in enumerate(corner):
                r = refs[i]
                if r not in done:
                    p = pos.get(r)
                    done[r] = None if p is None else c.element(p[0], p[1], mesh[tri, v].tolist())
                tok = done[r]
                if tok is not None:
                    rows.append([tri, v, tok[0], tok[1]])
            tri += 1
    if rows:
        c.out.setdefault("meshes", {})[s.name] = rows


def _property(c: Collector, kind, owner, key, ref, p) -> None:
    from lumen.gdml.importer import LENGTH_KEYS

    toks = c.doc.matrix_text.get(ref)
    if not toks:
        return
    m = c.doc.matrix(ref)
    rec: dict = {"energy": None, "value": None, "matrix_unit": c.doc.matrix_unit.get(ref)}
    if p.energy is None:
        if key in LENGTH_KEYS:
            rec["value"] = c.table(toks[:1], [float(m[0, 0])], [float(np.asarray(p.value))], 3)
    else:
        rec["energy"] = c.table(toks[0::2], m[:, 0].tolist(), np.asarray(p.energy).tolist(), -6)
        if key in LENGTH_KEYS:
            rec["value"] = c.table(toks[1::2], m[:, 1].tolist(), np.asarray(p.value).tolist(), 3)
    if any(v is not None for v in rec.values()):
        c.put(kind, owner, key, rec)


def collect(doc: reader.GDMLDocument, solids: dict, meshes: dict, placements: tuple, materials: dict,
            surfaces: dict) -> tuple[dict, list]:  # fmt: skip
    """Return (the tokens of the module docstring, report lines) of a document and the records read from it."""
    c = Collector(doc)
    _geometry(c, solids, meshes, placements)
    for kind, recs, src in (("materials", materials, doc.materials), ("surfaces", surfaces, doc.surfaces)):
        for name, r in src.items():
            for prop, ref in r.props if name in recs else ():
                key = prop.split("0x")[0]
                if key in recs[name].properties:
                    _property(c, kind, name, key, ref, recs[name].properties[key])
    rep = []
    if c.fallbacks:
        rep.append(f"tokens: {c.fallbacks} value(s) name a define; kept as their value in parentheses")
    if c.relative_facets:
        rep.append(f"tokens: {c.relative_facets} RELATIVE facet(s): an export writes them ABSOLUTE, so Geant4's mm "
                   "sums are not reproduced")  # fmt: skip
    return c.out, rep


def _geometry(c: Collector, solids: dict, meshes: dict, placements: tuple) -> None:
    """Record the tokens of the solids, their transforms and the placements."""
    doc = c.doc
    for name, s in doc.solids.items():
        if s.type == "tessellated":
            _mesh(c, s, meshes[name])
        elif s.type == "tet":
            _tet(c, s)
        elif s.texts:
            _solid(c, s, solids[name])
        for k, tr in enumerate(s.transforms):
            if tr.position_text is not None and not _plus_zero(tr.position):
                c.put("transforms", name, k, c.element(tr.position_text[0], tr.position_text[1], tr.position))
    i = 0
    for v in doc.volumes.values():
        for pv in v.placements:
            t = pv.transform.position_text
            if t is not None and not _plus_zero(placements[i].position):  # an export writes no +0 position
                tok = c.element(t[0], t[1], placements[i].position)
                if tok is not None:
                    c.out.setdefault("placements", {})[str(i)] = tok
            i += 1


# -------------------------------------------------------------------------------------------- decoding (export)
def length(text: str, unit: str) -> float | None:
    """Return a length token's value in m as the reader reads it, or None if it does not read."""
    try:
        if is_literal(text) and unit in LENGTH_M:
            return exact(text, LENGTH_M[unit])
        e = reader.Evaluator()
        return e.eval(text) * e.eval(unit) / MM_PER_M
    except (reader.GDMLError, ValueError, ZeroDivisionError, ArithmeticError):
        return None


def energy(text: str) -> float | None:
    """Return a matrix energy token's value in eV as the importer reads it, or None."""
    m = PRODUCT.match(text)
    if LITERAL.match(text):
        return float(Decimal(text.strip()) * Decimal("1000000"))
    if m and m.group(2) in ENERGY_EV:
        return float(Decimal(m.group(1)) * Decimal(ENERGY_EV[m.group(2)]))
    g = evaluate(text)
    return None if g is None else g * 1.0e6


def length_value(text: str) -> float | None:
    """Return a length-key matrix value token in m as the importer reads it, or None."""
    m = PRODUCT.match(text)
    if LITERAL.match(text):
        return float(Decimal(text.strip()) * Decimal(LENGTH_M[MM]))
    if m and m.group(2) in LENGTH_M:
        return float(Decimal(m.group(1)) * Decimal(LENGTH_M[m.group(2)]))
    g = evaluate(text)
    return None if g is None else g / MM_PER_M


def same(a, b) -> bool:
    """Whether two doubles are the same bits (a None never is)."""
    return a is not None and b is not None and a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
