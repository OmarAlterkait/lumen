"""The reference side of the GDML repairs: a Geant4-loadable copy of a GDML file, the repairs as text edits.

    text, lines = repair(src_text)            # every repair; or: python -m lumen.gdml.repair SRC DST [REPAIR ...]

A Geant4 reference input made from a GDML gets exactly the repairs lumen's reader applies, always, to the defects it
finds (``lumen.gdml.reader.REPAIRS``; stock Geant4 10.7.4 and 11.2.2 refuse each defect, tests/data/gdml_repairs).
Each repair edits only its own defect, so applying them all to a clean file changes nothing. Written apart from
lumen's reader (the standard library only, no other lumen import) and checked against it: lumen reads SRC and DST to
the same detector file (tests/unit/test_gdml_repairs.py). Every byte outside the edited elements is kept (geometry
untouched). It lives in the package so that ``lumen.validate.geometry`` can make its Geant4 input;
``python -m lumen.gdml.repair`` is its command line.

    denormal                each nonzero number literal below the smallest normal double (strtod's ERANGE), in an
                            attribute Geant4 evaluates or an <expression>, is written 0
    matrix_tail             a <matrix> whose value count, as Geant4 counts it (values that end in whitespace or are
                            empty have one more value, 0), is not a multiple of its coldim loses the unpaired tail
    non_finite              an inf / -inf <matrix> value is written 1e30 / -1e30
    identical_redefinition  a constant, variable, quantity, expression or matrix defined again identically: the later
                            definitions are deleted (the first kept)
    constant_property       a <property> whose ref names a define that is not a <matrix> is deleted
    undefined_matrix_property  a <property> whose ref names no define is deleted
    constant_surface_table  an opticalsurface <property> of a key G4OpBoundaryProcess reads only as a table
                            (REFLECTIVITY, TRANSMITTANCE, EFFICIENCY, the unified constants) naming a coldim 1 matrix is
                            deleted: Geant4 stores it as a constant property it never reads, so its default applies

Prints one line per edit.
"""

import re
import sys
import xml.etree.ElementTree as ET
from decimal import Decimal

REPAIRS = (
    "denormal",
    "matrix_tail",
    "non_finite",
    "identical_redefinition",
    "constant_property",
    "undefined_matrix_property",
    "constant_surface_table",
)
TABLE_ONLY_SURFACE_KEYS = ("REFLECTIVITY", "TRANSMITTANCE", "EFFICIENCY", "SPECULARLOBECONSTANT",
                           "SPECULARSPIKECONSTANT", "BACKSCATTERCONSTANT")  # fmt: skip
TINY = sys.float_info.min
SKIP = {"name", "ref", "formula", "state", "surfaceproperty", "auxtype", "auxvalue", "auxunit", "version"}
NUMBER = re.compile(r"(?<![\w.])(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])")
BIG_EXPONENT = re.compile(r"\d[eE][-+]?\d{3}")  # an out-of-range literal has one (or > 300 digits)
COMMENT = re.compile(r"<!--.*?-->|<!\[CDATA\[.*?\]\]>|<\?.*?\?>", re.S)
START = re.compile(r"<([A-Za-z_][\w:.-]*)([^<>]*)>")
ATTR = re.compile(r"""([\w:.-]+)(\s*=\s*)(["'])(.*?)\3""", re.S)
EXPRESSION = re.compile(r"(<expression\b[^<>]*>)(.*?)(</expression\s*>)", re.S)
ELEMENT = re.compile(
    r"<(constant|variable|quantity|expression|matrix|property)\b(?:[^<>]*?/>|[^<>]*>.*?</\1\s*>)", re.S
)
DEFINES = ("constant", "variable", "quantity", "expression", "matrix", "position", "rotation", "scale")


def _tag(el):
    return el.tag.split("}", 1)[-1]


def _mask(text):
    """Return the text with comments, CDATA and processing instructions blanked (offsets kept)."""
    return COMMENT.sub(lambda m: " " * len(m.group(0)), text)


def _edit(text, edits):
    out, at = [], 0
    for a, b, new in sorted(edits):
        out += [text[at:a], new]
        at = b
    return "".join(out + [text[at:]])


def _denormals(text, out):
    """Write each out-of-range-small literal 0; refuse an overflowing one (Geant4 refuses it, nothing to repair)."""
    n: dict = {}

    def zero(m, where):
        v = abs(float(m.group(0)))
        if v == float("inf"):
            raise ValueError(f"{where}: {m.group(0)} overflows a double")
        if v < TINY and Decimal(m.group(0)) != 0:
            n[where] = n.get(where, 0) + 1
            return "0"
        return m.group(0)

    def start(m):
        name = re.search(r"""\bname\s*=\s*(["'])(.*?)\1""", m.group(2))
        where = f"{m.group(1)} {name.group(2) if name else None!r}"

        def attr(a):
            k, v = a.group(1), a.group(4)
            if k in SKIP or ":" in k or not (len(v) > 300 or BIG_EXPONENT.search(v)):
                return a.group(0)
            return k + a.group(2) + a.group(3) + NUMBER.sub(lambda x: zero(x, where), v) + a.group(3)

        return "<" + m.group(1) + ATTR.sub(attr, m.group(2)) + ">"

    masked = _mask(text)
    edits = [(m.start(), m.end(), start(m)) for m in START.finditer(masked)]
    text = _edit(text, [(a, b, new) for a, b, new in edits if new != text[a:b]])
    masked = _mask(text)
    edits = [(m.start(2), m.end(2), NUMBER.sub(lambda x: zero(x, "expression"), text[m.start(2):m.end(2)]))
             for m in EXPRESSION.finditer(masked)]  # fmt: skip
    out += [f"{w}: {k} denormal literal(s) written 0" for w, k in n.items()]
    return _edit(text, [(a, b, new) for a, b, new in edits if new != text[a:b]])


def _matrix_values(el, piece, repairs, out):
    """Return the matrix element's text with its values attribute repaired (or unchanged)."""
    raw, nm = el.get("values") or "", el.get("name")
    real, phantom = raw.split(), not raw or raw[-1].isspace()
    cd, n, before = int(round(float(el.get("coldim", "1")))), len(real) + phantom, (real, phantom)
    toks = real
    if "matrix_tail" in repairs and n % cd:
        toks, phantom = (real + ["0"] * phantom)[: n - n % cd], False
        out.append(f"matrix {nm!r}: {n} values for coldim {cd}, the unpaired tail ({n % cd}) dropped")
    inf = [k for k, t in enumerate(toks) if t.lower().lstrip("+-") in ("inf", "infinity")]
    if "non_finite" in repairs and inf:
        toks = [("-1e30" if t[0] == "-" else "1e30") if k in inf else t for k, t in enumerate(toks)]
        out.append(f"matrix {nm!r}: {len(inf)} non-finite value(s) written +-1e30")
    if (toks, phantom) == before:
        return piece
    new = " ".join(toks) + (" " if phantom else "")
    return re.sub(r"""(\bvalues\s*=\s*)(["']).*?\2""", lambda m: m.group(1) + m.group(2) + new + m.group(2), piece,
                  count=1, flags=re.S)  # fmt: skip


def _property_repair(el, kind, coldim, on_surface, repairs):
    """Return the line of a <property> the repairs delete (its ref names no matrix, or a constant on a surface)."""
    nm, ref = el.get("name"), el.get("ref")
    if kind != "matrix" and ("undefined_matrix_property" if kind is None else "constant_property") in repairs:
        return f"property {nm} -> {kind or 'nothing'} {ref!r} deleted"
    if (kind == "matrix" and on_surface and "constant_surface_table" in repairs and coldim.get(ref) == 1
            and nm.split("0x")[0] in TABLE_ONLY_SURFACE_KEYS):  # fmt: skip
        return f"surface property {nm} -> the constant matrix {ref!r} deleted"
    return None


def _facts(root):
    """Return (each define's first kind, each matrix's first coldim, the ids of the opticalsurfaces' properties)."""
    kinds: dict = {}
    coldim: dict = {}
    for el in root.iter():
        if _tag(el) in DEFINES:
            kinds.setdefault(el.get("name"), _tag(el))
        if _tag(el) == "matrix":
            coldim.setdefault(el.get("name"), int(round(float(el.get("coldim", "1")))))
    on_surface = {id(p) for s in root.iter() if _tag(s) == "opticalsurface" for p in s if _tag(p) == "property"}
    return kinds, coldim, on_surface


def repair(text, repairs=REPAIRS):
    """Return (the repaired text, one line per edit); ``repairs``: the kinds to apply (default every one)."""
    bad = set(repairs) - set(REPAIRS)
    if bad:
        raise ValueError(f"unknown repairs {sorted(bad)}; known: {list(REPAIRS)}")
    out: list = []
    if "denormal" in repairs:
        text = _denormals(text, out)
    root = ET.fromstring(text.encode("utf-8"))
    kinds, coldim, on_surface = _facts(root)
    els = [el for el in root.iter() if _tag(el) in ("constant", "variable", "quantity", "expression", "matrix",
                                                    "property")]  # fmt: skip
    matches = list(ELEMENT.finditer(_mask(text)))
    if [_tag(e) for e in els] != [m.group(1) for m in matches]:
        raise ValueError("the elements found in the text are not the parsed ones (an unusual layout): not repaired")
    seen: dict = {}
    edits: list = []
    for el, m in zip(els, matches, strict=True):
        piece, tg, nm = text[m.start() : m.end()], _tag(el), el.get("name")
        if tg == "property":
            line = _property_repair(el, kinds.get(el.get("ref")), coldim, id(el) in on_surface, repairs)
            if line:
                edits.append((m.start(), m.end(), ""))
                out.append(line)
            continue
        key = (tg, tuple(sorted(el.attrib.items())), (el.text or "").strip())
        if nm in seen and seen[nm] == key and "identical_redefinition" in repairs:
            edits.append((m.start(), m.end(), ""))
            out.append(f"identical re-definition of {tg} {nm!r} deleted (the first kept)")
            continue
        seen.setdefault(nm, key)
        if tg == "matrix":
            new = _matrix_values(el, piece, repairs, out)
            if new != piece:
                edits.append((m.start(), m.end(), new))
    return _edit(text, edits), out


def main(argv):
    """Repair SRC into DST (see the module docstring)."""
    if len(argv) < 2 or argv[0] == argv[1]:
        sys.exit(__doc__)
    with open(argv[0], encoding="utf-8") as f:
        text, out = repair(f.read(), argv[2:] or REPAIRS)
    with open(argv[1], "w", encoding="utf-8") as f:
        f.write(text)
    ET.parse(argv[1])  # still well-formed
    sys.stdout.write("\n".join(out or ["no edits"]) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
