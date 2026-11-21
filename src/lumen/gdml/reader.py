"""The GDML reader: one parser, one CLHEP evaluator, lengths converted to metres by exact decimal scaling.

The file is untrusted input: no ``eval``; expressions go through a small recursive-descent parser with CLHEP units.

    doc = read(path)                            # GDMLDocument: defines, materials, solids, volumes, surfaces

Values: lengths are in m, a plain literal in a decimal length unit converted exactly (``12.5`` mm is exactly 0.0125,
rounded once) and anything else evaluated with CLHEP's units (``lumen.gdml.units``) and divided by 1000 once; angles
in rad; matrices (property tables) in Geant4's internal units (MeV, mm, ns), with the source text of every value kept
so the importer can convert plain literals exactly too. CLHEP's units are those of Geant4's C++ (SystemOfUnits.h,
1 eV = 1e-6 MeV); Geant4's GDML evaluator derives its energy and mass units from the 1993 electron charge instead
(G4GDMLEvaluator, Geant4 10.7.4, measured: 1*eV is 1.0000004344090314e-06 MeV), so a value written with an energy or
mass unit reads 4.3e-7 (relative) larger in Geant4 than here. Bare literals in internal units read the same in both
(the GDML exporter writes only those).

Follows Geant4 (10.7.4 and 11.2.2, measured: tests/data/gdml_repairs): what Geant4's reader refuses is a GDMLError
here too, and nothing is repaired on the way -- a denormal or out-of-range number literal, an inf or a NaN in a
matrix, a matrix whose value count is not a multiple of its coldim (counted as Geant4 counts: a values attribute that
ends in whitespace, or is empty, has one more value, 0, as ``while (!eof) stream >> token`` reads it), a constant,
variable, quantity, expression or matrix defined twice, and a ``<property>`` whose ref names no matrix. Each such
error names the repair ``python -m lumen.gdml.repair SRC DST`` writes (``lumen.gdml.repair``): convert the repaired
copy, whose sha256 the detector file then records as its source. Attribute names are matched exactly, as Geant4
matches them (``<tube RMAX=...>`` is refused: Geant4 would ignore it and read rmax as 0); a ``<property>`` keeps
its matrix's shape, as G4GDMLReadMaterials / Solids::PropertyRead do: coldim 1 is a constant property, anything wider
an (energy, value) table, on materials and surfaces alike.

Fail closed: an expression that does not evaluate, a reference to a missing position, rotation, solid or volume, an
attribute Geant4 does not read on a solid, and every construct lumen does not support -- ``<loop>``, replicavol,
divisionvol, paramvol, a physvol ``<file>`` -- raise GDMLError.

Beyond the geometry the reader keeps what the importer needs: material compositions, elements, isotopes, temperatures
and pressures, and the setup named ``Default``.
"""

from __future__ import annotations

import collections
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np

from lumen.format import solids as catalogue
from lumen.gdml.lexical import BIG_EXPONENT, LONG_LITERAL, NUMBER, SKIP_LITERALS, TINY
from lumen.gdml.units import UNITS, is_literal
from lumen.units import ENERGY_EV, LENGTH_M, MM_PER_M, exact


class GDMLError(ValueError):
    """A GDML file lumen cannot read as Geant4 would: the message names the element and the reason."""


def refused(kind: str) -> str:
    """Return the tail of a GDMLError for what Geant4 refuses: the repair (``lumen.gdml.repair.REPAIRS``) or none."""
    from lumen.gdml.repair import REPAIRS

    if kind not in REPAIRS:
        return "Geant4 refuses it too"
    return (f"Geant4 refuses it too; the repair {kind!r} ({REPAIRS[kind]}) writes a copy Geant4 and lumen both read: "
            "python -m lumen.gdml.repair SRC DST, then convert DST")  # fmt: skip


FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "atan2": math.atan2, "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh, "exp": math.exp,
    "log": math.log, "log10": math.log10, "sqrt": math.sqrt, "pow": math.pow, "abs": abs, "fabs": abs,
    "min": min, "max": max, "floor": math.floor, "ceil": math.ceil, "int": lambda x: float(int(x)),
}  # fmt: skip
_TOK = re.compile(
    r"\s*(?:(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|(\*\*|[-+*/^(),\[\]]))"
)
UNIT_ATTRS = {"name", "lunit", "aunit"}  # beside a solid's own attributes (lumen.format.solids.ATTRIBUTES)
BOOLEAN = ("union", "subtraction", "intersection")
UNSUPPORTED = ("replicavol", "divisionvol", "paramvol", "loop")
PASCAL_PA = {"pascal": "1", "Pa": "1"}  # the pressure units that are exact multiples of a pascal


def _out_of_range(lit: str) -> str | None:
    """Return "denormal" or "overflow" for a number literal Geant4's strtod reports out of range (ERANGE)."""
    v = abs(float(lit))
    return "overflow" if math.isinf(v) else "denormal" if v < TINY and Decimal(lit) != 0 else None


def tag(el) -> str:
    """Return an element's tag without its namespace."""
    t = el.tag
    return t.split("}", 1)[1] if "}" in t else t


def g4_strip(name):
    """Return Geant4's stripped name (G4GDMLRead::StripName): everything from the first '0x' on dropped."""
    if name is None:
        return None
    k = name.find("0x")
    return name if k < 0 else name[:k]


class Evaluator:
    """GDML arithmetic.

    Numbers, + - * / ^ **, parentheses, CLHEP units, defines (resolved lazily), matrix elements m[i] and m[i, j]
    (1-based), and the functions of FUNCS.
    """

    def __init__(self):
        self.raw = {}  # name -> (expression, unit expression or None)
        self.cache = {}
        self.matrices = {}  # name -> (coldim, [floats]) normalised
        self.busy = set()

    def define(self, name, expr, unit=None):
        """Record a define (evaluated when first used)."""
        self.raw[name] = (expr, unit)

    def lookup(self, name):
        """Return the value of a name: a define, a CLHEP unit or a one-value matrix."""
        if name in self.cache:
            return self.cache[name]
        if name in self.raw:
            if name in self.busy:
                raise GDMLError("cyclic definition of " + name)
            self.busy.add(name)
            try:
                expr, unit = self.raw[name]
                v = self.eval(expr)
                if unit:
                    v *= self.eval(unit)
            finally:
                self.busy.discard(name)
            self.cache[name] = v
            return v
        if name in UNITS:
            return UNITS[name]
        if name in self.matrices and len(self.matrices[name][1]) == 1:
            return self.matrices[name][1][0]
        raise GDMLError("unknown identifier " + name)

    def eval(self, s):
        """Return the value of an expression, in Geant4's internal units."""
        if s is None:
            raise GDMLError("missing value")
        s = s.strip()
        if is_literal(s):
            return float(s)
        toks: list[tuple[str, object]] = []
        pos = 0
        while pos < len(s):
            mm = _TOK.match(s, pos)
            if not mm or mm.end() == pos:
                if s[pos:].strip() == "":
                    break
                raise GDMLError(f"bad token in {s!r} at {pos}")
            pos = mm.end()
            if mm.group(1) is not None:
                toks.append(("n", float(mm.group(1))))
            elif mm.group(2) is not None:
                toks.append(("i", mm.group(2)))
            else:
                toks.append(("o", mm.group(3)))
        saved: tuple = (getattr(self, "_t", None), getattr(self, "_p", 0))
        self._t, self._p = toks, 0
        try:
            v = self._expr()
            if self._p != len(toks):
                raise GDMLError(f"trailing tokens in {s!r}")
        finally:
            self._t, self._p = saved
        return v

    def _peek(self):
        return self._t[self._p] if self._p < len(self._t) else (None, None)

    def _next(self):
        t = self._peek()
        self._p += 1
        return t

    def _expr(self):
        v = self._term()
        while True:
            k, t = self._peek()
            if k == "o" and t in ("+", "-"):
                self._next()
                r = self._term()
                v = v + r if t == "+" else v - r
            else:
                return v

    def _term(self):
        v = self._unary()
        while True:
            k, t = self._peek()
            if k == "o" and t in ("*", "/"):
                self._next()
                r = self._unary()
                v = v * r if t == "*" else v / r
            else:
                return v

    def _unary(self):
        k, t = self._peek()
        if k == "o" and t in ("+", "-"):
            self._next()
            v = self._unary()
            return -v if t == "-" else v
        return self._power()

    def _power(self):
        v = self._atom()
        k, t = self._peek()
        if k == "o" and t in ("^", "**"):
            self._next()
            v = math.pow(v, self._unary())
        return v

    def _call(self, name):
        args = []
        k3, t3 = self._peek()
        if not (k3 == "o" and t3 == ")"):
            while True:
                args.append(self._expr())
                k3, t3 = self._next()
                if k3 == "o" and t3 == ")":
                    break
                if not (k3 == "o" and t3 == ","):
                    raise GDMLError("bad call of " + name)
        else:
            self._next()
        f = FUNCS.get(name)
        if f is None:
            raise GDMLError("unknown function " + name)
        return float(f(*args))  # type: ignore[operator]

    def _element(self, name):
        idx = [self._expr()]
        k3, t3 = self._next()
        while k3 == "o" and t3 == ",":
            idx.append(self._expr())
            k3, t3 = self._next()
        if name not in self.matrices:
            raise GDMLError("unknown matrix " + name)
        cd, vals = self.matrices[name]
        j = int(idx[0]) - 1 if len(idx) == 1 else (int(idx[0]) - 1) * cd + int(idx[1]) - 1
        if not 0 <= j < len(vals):
            raise GDMLError(f"index out of range: {name}{idx}")
        return vals[j]

    def _atom(self):
        k, t = self._next()
        if k == "n":
            return t
        if k == "i":
            k2, t2 = self._peek()
            if k2 == "o" and t2 == "(":
                self._next()
                return self._call(t)
            if k2 == "o" and t2 == "[":
                self._next()
                return self._element(t)
            return self.lookup(t)
        if k == "o" and t == "(":
            v = self._expr()
            k2, t2 = self._next()
            if not (k2 == "o" and t2 == ")"):
                raise GDMLError("missing )")
            return v
        raise GDMLError(f"unexpected token {t!r}")


# ------------------------------------------------------------------------------------------------- the document
@dataclass
class Transform:
    """A position (m), GDML rotation angles (rad) and an optional scale."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: tuple[float, float, float] | None = None


@dataclass
class Solid:
    """A solid as read: numeric attributes (m, rad, unitless), children, transforms, planes and facets."""

    name: str
    type: str
    params: dict = field(default_factory=dict)
    refs: list = field(default_factory=list)
    transforms: list = field(default_factory=list)  # Transform per ref (booleans: first, second)
    planes: list = field(default_factory=list)  # polycone / polyhedra (rmin, rmax, z); generic (r, z)
    vertices: list = field(default_factory=list)  # xtru (x, y)
    sections: list = field(default_factory=list)  # xtru (z, x offset, y offset, scaling factor)
    facets: list = field(default_factory=list)  # tessellated: (position refs, "ABSOLUTE" | "RELATIVE")
    tet: tuple = ()  # tet: (position refs, the lunit in mm)


@dataclass
class Physvol:
    """A placement as read.

    The child volume, its name (Geant4's ``<lv>_PV`` when the GDML gives none), copy number and transform.
    """

    child: str
    name: str
    copy: int | None
    transform: Transform


@dataclass
class Volume:
    """A logical volume or an assembly as read."""

    name: str
    material: str | None = None
    solid: str | None = None
    placements: list = field(default_factory=list)
    assembly: bool = False


@dataclass
class Material:
    """A material as read: state, property references, conditions and composition (raw numbers with units)."""

    name: str
    state: str | None = None
    props: list = field(default_factory=list)  # (property name, matrix ref)
    density_g_cm3: float | None = None
    temperature: float | None = None  # K
    pressure: float | None = None  # Pa
    mee_ev: float | None = None  # the mean excitation energy, eV
    z: float | None = None
    molar_mass_g_mol: float | None = None
    fractions: list = field(default_factory=list)  # (ref, mass fraction)
    composites: list = field(default_factory=list)  # (ref, number of atoms)


@dataclass
class Element:
    """An element as read."""

    name: str
    formula: str
    z: float | None
    molar_mass_g_mol: float | None = None
    fractions: list = field(default_factory=list)  # (isotope ref, abundance)


@dataclass
class Isotope:
    """An isotope as read."""

    name: str
    z: int
    n: int
    molar_mass_g_mol: float


@dataclass
class OpticalSurface:
    """An opticalsurface as read: the four attributes as written (None when absent) and its property references."""

    name: str
    model: str | None
    finish: str | None
    type: str | None
    value: str | None
    props: list = field(default_factory=list)


class GDMLDocument:
    """A parsed GDML file. See the module docstring for units; ``report`` lists every normalisation."""

    def __init__(self, path):
        self.path = str(path)
        self.ev = Evaluator()
        self.positions, self.rotations, self.scales = {}, {}, {}
        self.solids = collections.OrderedDict()
        self.volumes = collections.OrderedDict()
        self.materials = collections.OrderedDict()
        self.elements = collections.OrderedDict()
        self.isotopes = collections.OrderedDict()
        self.surfaces = collections.OrderedDict()
        self.borders, self.skins = [], []  # (name, surface, [pv, pv]) and (name, surface, [lv])
        self.matrix_text = {}  # name -> the source tokens of its values
        self.world = None
        self.report = []
        self._defs = {}

    def refuse(self, kind, where):
        """Raise the GDMLError of what Geant4 refuses (``kind``: a repair of ``lumen.gdml.repair``, or another)."""
        raise GDMLError(f"gdml {self.path}: {where}: {refused(kind)}")

    # ---- numbers
    def num(self, s, what):
        """Return the value of an expression in Geant4's internal units; the error names ``what``."""
        try:
            return self.ev.eval(s)
        except GDMLError as e:
            raise GDMLError(f"{what}: {e}") from None

    def integer(self, s, what):
        """Return an integer expression's value, refusing one with a fractional part, as Geant4 does
        (G4GDMLEvaluator::EvaluateInteger, a FatalException: matrix coldim, composite n, isotope Z and N)."""  # noqa: D205, D209
        v = self.num(s, what)
        if v != int(v):
            raise GDMLError(f"{what}: {s!r} is expected to have an integer value (G4GDMLEvaluator::EvaluateInteger)")
        return int(v)

    def length(self, text, unit, what):
        """Return a length in m: exact for a plain literal in a decimal unit, else Geant4's value / 1000 once."""
        unit = "mm" if unit is None else unit.strip()
        if is_literal(text) and unit in LENGTH_M:
            return exact(text, LENGTH_M[unit])
        return self.num(text, what) * self.num(unit, what) / MM_PER_M

    def angle(self, text, unit, what):
        """Return an angle in rad, as Geant4 computes it."""
        return self.num(text, what) * self.num("rad" if unit is None else unit, what)

    def vec(self, el, kind):
        """Return a position (m), rotation (rad) or scale element's three values."""
        what = f"{tag(el)} {el.get('name')!r}"
        if kind == "scale":
            return tuple(self.num(el.get(a, "1"), what) for a in ("x", "y", "z"))
        if kind == "rotation":
            return tuple(self.angle(el.get(a, "0"), el.get("unit"), what) for a in ("x", "y", "z"))
        return tuple(self.length(el.get(a, "0"), el.get("unit"), what) for a in ("x", "y", "z"))

    def transform(self, parent, pos=("position", "positionref"), rot=("rotation", "rotationref"),
                  scale=("scale", "scaleref")):  # fmt: skip
        """Return the Transform of an element's position, rotation and scale children (inline or by reference)."""
        t, r, sc = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None
        where = f"{tag(parent)} {parent.get('name')!r}"
        tables = {pos[1]: (self.positions, "positionref"), rot[1]: (self.rotations, "rotationref")}
        if scale:
            tables[scale[1]] = (self.scales, "scaleref")
        for c in parent:
            tg = tag(c)
            if tg == pos[0]:
                t = self.vec(c, "position")
            elif tg == rot[0]:
                r = self.vec(c, "rotation")
            elif scale and tg == scale[0]:
                sc = self.vec(c, "scale")
            elif tg in tables:
                table, label = tables[tg]
                ref = c.get("ref")
                if ref not in table:
                    raise GDMLError(f"{where}: {label} {ref!r} is not defined")
                v = table[ref]
                if tg == pos[1]:
                    t = v
                elif tg == rot[1]:
                    r = v
                else:
                    sc = v
        return Transform(t, r, sc)

    def matrix(self, name):
        """Return a matrix as (rows, coldim) float64 in Geant4's internal units; raise if undefined."""
        if name not in self.ev.matrices:
            raise GDMLError(f"matrix {name!r} is not defined")
        cd, vals = self.ev.matrices[name]
        return np.asarray(vals, float).reshape(-1, cd)

    # ---- parse
    def parse(self):
        """Read the file; return self."""
        try:
            root = ET.parse(self.path).getroot()
        except ET.ParseError as e:
            raise GDMLError(f"gdml {self.path}: XML error (external entity includes are not supported): {e}") from None
        for el in root.iter():
            if tag(el) in UNSUPPORTED:
                raise GDMLError(f"gdml {self.path}: <{tag(el)}> {el.get('name')!r} is not supported")
        self._literals(root)
        sections = collections.defaultdict(list)
        for sec in root:
            sections[tag(sec)].append(sec)
        for sec in sections["define"]:
            self.read_define(sec)
        for sec in sections["materials"]:
            self.read_materials(sec)
        for sec in sections["solids"]:
            self.read_solids(sec)
        for sec in sections["structure"]:
            self.read_structure(sec)
        self.world = self._world(sections["setup"])
        if self.world not in self.volumes:
            raise GDMLError(f"gdml {self.path}: world volume {self.world!r} is not defined")
        self._property_refs()
        return self

    def _property_refs(self):
        """Refuse a property whose ref names a define that is not a matrix, or nothing (G4GDMLReadDefine::GetMatrix)."""
        owners = [(f"material {n!r}", m.props) for n, m in self.materials.items()]
        for owner, props in owners + [(f"opticalsurface {n!r}", s.props) for n, s in self.surfaces.items()]:
            for name, ref in props:
                if ref not in self._defs:
                    self.refuse("undefined_matrix_property", f"{owner}: property {name} names {ref!r}, not defined")
                if self._defs[ref][0] != "matrix":
                    self.refuse("constant_property", f"{owner}: property {name} names {self._defs[ref][0]} {ref!r}")

    def _literals(self, root):
        """Refuse an out-of-range (denormal or overflowing) number literal Geant4 evaluates (strtod sets ERANGE)."""
        for el in root.iter():
            texts = [(k, v) for k, v in el.attrib.items() if k not in SKIP_LITERALS and "}" not in k]
            if tag(el) == "expression" and el.text:
                texts.append(("text", el.text))
            for k, v in texts:
                if not (len(v) > LONG_LITERAL or BIG_EXPONENT.search(v)):
                    continue
                for m in NUMBER.finditer(v):
                    bad = _out_of_range(m.group(0))
                    if bad:
                        self.refuse("denormal" if bad == "denormal" else "overflow",
                                    f"{tag(el)} {el.get('name')!r} {k}: {m.group(0)} is {bad} as a double")  # fmt: skip

    def _world(self, setups):
        named = [s for s in setups if s.get("name") == "Default"]
        for s in named or setups[:1]:
            for c in s:
                if tag(c) == "world":
                    return c.get("ref")
        return None

    def _redefined(self, el):
        """Whether a define was seen before; a conflicting one, or an evaluator define given twice, is a GDMLError.

        Geant4 refuses an evaluator define's re-definition (G4GDMLEvaluator::DefineConstant / DefineVariable); a
        position, rotation or scale it re-defines (its map keeps the last), so an identical one is read once and
        reported.
        """
        nm = el.get("name")
        key = (tag(el), tuple(sorted(el.attrib.items())), (el.text or "").strip())
        if nm not in self._defs:
            self._defs[nm] = key
            return False
        if self._defs[nm] != key:
            raise GDMLError(f"gdml {self.path}: conflicting re-definition of {tag(el)} {nm!r}")
        if tag(el) not in ("position", "rotation", "scale"):
            self.refuse("identical_redefinition", f"{tag(el)} {nm!r} defined again, identically")
        self.report.append(f"identical re-definition of {tag(el)} {nm!r} read once (Geant4 keeps the last)")
        return True

    def read_define(self, sec):
        """Read a <define> section."""
        vecs = []
        for el in sec:
            tg, nm = tag(el), el.get("name")
            kinds = ("constant", "variable", "quantity", "expression", "matrix", "position", "rotation", "scale")
            if tg in kinds and self._redefined(el):
                continue
            if tg in ("constant", "variable"):
                self.ev.define(nm, el.get("value"))
            elif tg == "quantity":
                self.ev.define(nm, el.get("value"), el.get("unit"))
            elif tg == "expression":
                self.ev.define(nm, el.text or "0")
            elif tg == "matrix":
                self.read_matrix(el)
            elif tg in ("position", "rotation", "scale"):
                vecs.append(el)
        for el in vecs:  # after every constant is known
            table = {"position": self.positions, "rotation": self.rotations, "scale": self.scales}[tag(el)]
            table[el.get("name")] = self.vec(el, tag(el))

    def read_matrix(self, el):
        """Read a <matrix>: values evaluated in internal units, the source tokens kept (see the module docstring)."""
        nm, raw = el.get("name"), el.get("values") or ""
        toks = raw.split() + (["0"] if not raw or raw[-1].isspace() else [])  # Geant4's count: a trailing 0
        try:
            cd = self.integer(el.get("coldim", "1"), "coldim")
            if cd < 1:
                raise GDMLError(f"coldim {cd}")
        except GDMLError as e:
            raise GDMLError(f"gdml {self.path}: matrix {nm!r}: {e}") from None
        if len(toks) % cd:
            self.refuse("matrix_tail", f"matrix {nm!r}: {len(toks)} values (as Geant4 counts them) for coldim {cd}")
        if not raw or raw[-1].isspace():
            self.report.append(f"matrix {nm!r}: its values end in whitespace; read as Geant4 reads them, with one "
                               "more value, 0")  # fmt: skip
        if any(t.lower().lstrip("+-") in ("inf", "infinity") for t in toks):
            self.refuse("non_finite", f"matrix {nm!r}: an inf value")
        try:  # Geant4 evaluates each value as it reads the define (used or not) and refuses a bad one
            vals = [self.ev.eval(t) for t in toks]
        except GDMLError as e:
            raise GDMLError(f"gdml {self.path}: matrix {nm!r}: {e}") from None
        if not all(math.isfinite(v) for v in vals):
            self.refuse(
                "not_finite", f"matrix {nm!r}: a value evaluates to {next(v for v in vals if not math.isfinite(v))}"
            )
        self.ev.matrices[nm] = (cd, vals)
        self.matrix_text[nm] = toks

    def read_materials(self, sec):
        """Read a <materials> section: isotopes, elements, materials with their compositions and properties."""
        for el in sec:
            tg, nm = tag(el), el.get("name")
            if tg == "isotope":
                atom = el.find("atom")
                z, n = (self.integer(el.get(k), f"isotope {nm!r} {k}") for k in ("Z", "N"))
                self.isotopes[nm] = Isotope(nm, z, n, self._atom(atom))
            elif tg == "element":
                atom = el.find("atom")
                fr = [(c.get("ref"), self.num(c.get("n"), f"element {nm!r}")) for c in el if tag(c) == "fraction"]
                self.elements[nm] = Element(nm, el.get("formula", ""), None if el.get("Z") is None else
                                            self.num(el.get("Z"), f"element {nm!r}"),
                                            None if atom is None else self._atom(atom), fr)  # fmt: skip
            elif tg == "material":
                self.materials[nm] = self._material(el)

    def _atom(self, atom):
        """Return the molar mass of an <atom> in g/mol: a literal in g/mole exactly, else Geant4's value / (g/mole)."""
        if atom is None:
            return None
        unit = atom.get("unit", "g/mole").replace(" ", "")
        if unit in ("g/mole", "g/mol") and is_literal(atom.get("value")):
            return float(atom.get("value"))
        g_mol = UNITS["g"] / UNITS["mole"]
        return self.num(atom.get("value"), "atom") * self.num(unit, "atom unit") / g_mol

    def _material(self, el):
        nm = el.get("name")
        what = f"material {nm!r}"
        m = Material(nm, el.get("state"))
        for c in el:
            ct = tag(c)
            if ct == "property":
                m.props.append((c.get("name"), c.get("ref")))
            elif ct in ("D", "Dref"):
                m.density_g_cm3 = self._density(c, what)
            elif ct == "T":
                m.temperature = self.num(c.get("value"), what) * self.num(c.get("unit", "K"), what)
            elif ct == "P":
                m.pressure = self._quantity(c, "pascal", PASCAL_PA, what)
            elif ct == "MEE":
                m.mee_ev = self._quantity(c, "eV", ENERGY_EV, what)
            elif ct == "atom":
                m.molar_mass_g_mol = self._atom(c)
            elif ct == "fraction":
                m.fractions.append((c.get("ref"), self.num(c.get("n"), what)))
            elif ct == "composite":
                m.composites.append((c.get("ref"), self.integer(c.get("n"), what)))
        if el.get("Z") is not None:
            m.z = self.num(el.get("Z"), what)
        return m

    def _quantity(self, c, to, factors, what):
        """Return a <P> or <MEE> value in the unit ``to`` (its default unit in Geant4).

        ``factors`` gives the units that are exact decimal multiples of ``to``: a plain literal in one of them is
        scaled exactly (one rounding); anything else is Geant4's value (G4GDMLReadMaterials::PRead, MEERead: the value
        times its unit) divided by ``to`` once.
        """
        unit, text = c.get("unit", to).strip(), c.get("value")
        if unit in factors and is_literal(text):
            return exact(text, factors[unit])
        return self.num(text, what) * self.num(unit, what) / UNITS[to]

    def _density(self, c, what):
        if tag(c) == "Dref":
            return self.num(c.get("ref"), what) / (UNITS["g"] / UNITS["cm3"])
        unit = c.get("unit", "g/cm3")
        if unit.replace(" ", "") == "g/cm3" and is_literal(c.get("value")):
            return float(c.get("value"))
        return self.num(c.get("value"), what) * self.num(unit, what) / (UNITS["g"] / UNITS["cm3"])

    def read_solids(self, sec):
        """Read a <solids> section (optical surfaces live there too)."""
        for el in sec:
            tg, nm = tag(el), el.get("name")
            if tg == "opticalsurface":
                self.surfaces[nm] = OpticalSurface(
                    nm,
                    el.get("model"),
                    el.get("finish"),
                    el.get("type"),
                    el.get("value"),
                    [(c.get("name"), c.get("ref")) for c in el if tag(c) == "property"],
                )
                continue
            self.solids[nm] = self._solid(el)
        for s in self.solids.values():
            for r in s.refs:
                if r not in self.solids:
                    raise GDMLError(f"{s.type} {s.name!r}: solid {r!r} is not defined")

    def _solid(self, el):
        tg, nm = tag(el), el.get("name")
        what = f"{tg} {nm!r}"
        s = Solid(nm, tg)
        lu = el.get("lunit")
        self._attributes(el, s, what)
        if tg in BOOLEAN:
            self._boolean(el, s)
        elif tg == "multiUnion":
            for c in el:
                if tag(c) == "multiUnionNode":
                    s.refs.append(next((cc.get("ref") for cc in c if tag(cc) == "solid"), None))
                    s.transforms.append(self.transform(c, scale=None))
        elif tg == "scaledSolid":
            self._scaled(el, s, what)
        elif tg in ("tessellated", "tet"):
            self._facets(el, s, what)
        else:
            self._solid_children(el, s, lu, what)
        return s

    def _attributes(self, el, s, what):
        """Read a solid's own attributes, each by Geant4's exact name (``lumen.format.solids.ATTRIBUTES``)."""
        tg, lu, au = s.type, el.get("lunit"), el.get("aunit")
        if tg not in catalogue.ATTRIBUTES:  # the lowering refuses it where it is used (or keeps an assembly)
            self.report.append(f"{what}: solid type {tg!r} is not one lumen supports: its attributes are not read")
            return
        dims = dict(catalogue.ATTRIBUTES[tg])
        extra = {f"vertex{i}" for i in range(1, 5)} if tg == "tet" else set()
        for a, v in el.attrib.items():
            if a in UNIT_ATTRS or a in extra or "}" in a:
                continue
            if a not in dims:
                raise GDMLError(f"gdml {self.path}: {what}: attribute {a!r} is not one Geant4 reads for {tg} (names "
                                f"are case-sensitive; {tg} has {sorted(dims)}): Geant4 ignores it")  # fmt: skip
            if dims[a] == catalogue.A:
                s.params[a] = self.angle(v, au, f"{what} attribute {a}")
            elif dims[a] == catalogue.U:
                s.params[a] = self.num(v, f"{what} attribute {a}")
            else:
                s.params[a] = self.length(v, lu, f"{what} attribute {a}")

    def _boolean(self, el, s):
        first = next((c.get("ref") for c in el if tag(c) == "first"), None)
        second = next((c.get("ref") for c in el if tag(c) == "second"), None)
        second_t = self.transform(el, scale=None)
        first_t = self.transform(el, pos=("firstposition", "firstpositionref"),
                                 rot=("firstrotation", "firstrotationref"), scale=None)  # fmt: skip
        s.refs, s.transforms = [first, second], [first_t, second_t]

    def _scaled(self, el, s, what):
        sc = (1.0, 1.0, 1.0)
        for c in el:
            if tag(c) == "solidref":
                s.refs = [c.get("ref")]
            elif tag(c) == "scale":
                sc = self.vec(c, "scale")
            elif tag(c) == "scaleref":
                if c.get("ref") not in self.scales:
                    raise GDMLError(f"{what}: scaleref {c.get('ref')!r} is not defined")
                sc = self.scales[c.get("ref")]
        s.transforms = [Transform(scale=sc)]

    def _facets(self, el, s, what):
        if s.type == "tet":
            refs = [el.get(f"vertex{i}") for i in range(1, 5)]
        else:
            refs = []
        for r in refs:
            if r not in self.positions:
                raise GDMLError(f"{what}: vertex {r!r} is not defined")
        if s.type == "tet":
            s.tet = (refs, self.num(el.get("lunit", "mm"), f"{what} lunit"))
            return
        for c in el:
            ct = tag(c)
            if ct in ("triangular", "quadrangular"):
                frefs = [c.get(f"vertex{i}") for i in range(1, 5 if ct == "quadrangular" else 4)]
                for r in frefs:
                    if r not in self.positions:
                        raise GDMLError(f"{what}: facet vertex {r!r} is not defined")
                s.facets.append((frefs, "RELATIVE" if c.get("type") == "RELATIVE" else "ABSOLUTE"))  # exact, as Geant4

    def _solid_children(self, el, s, lu, what):
        tg = s.type
        if tg in ("polycone", "polyhedra", "genericPolycone", "genericPolyhedra"):
            kind, keys = (
                ("zplane", ("rmin", "rmax", "z")) if tg in ("polycone", "polyhedra") else ("rzpoint", ("r", "z"))
            )
            rows = [[c.get(k, "0") for k in keys] for c in el if tag(c) == kind]
            s.planes = [tuple(self.length(t, lu, f"{what} {kind}") for t in row) for row in rows]
        elif tg == "xtru":
            for c in el:
                if tag(c) == "twoDimVertex":
                    s.vertices.append((self.length(c.get("x"), lu, what), self.length(c.get("y"), lu, what)))
                elif tag(c) == "section":
                    s.sections.append((self.length(c.get("zPosition"), lu, what),
                                       self.length(c.get("xOffset"), lu, what),
                                       self.length(c.get("yOffset"), lu, what),
                                       self.num(c.get("scalingFactor", "1"), what)))  # fmt: skip

    def read_structure(self, sec):
        """Read a <structure> section: volumes, assemblies, border and skin surfaces."""
        for el in sec:
            tg = tag(el)
            if tg == "bordersurface":
                self.borders.append((el.get("name"), el.get("surfaceproperty"),
                                     [c.get("ref") for c in el if tag(c) == "physvolref"]))  # fmt: skip
                continue
            if tg == "skinsurface":
                self.skins.append((el.get("name"), el.get("surfaceproperty"),
                                   [c.get("ref") for c in el if tag(c) == "volumeref"]))  # fmt: skip
                continue
            if tg not in ("volume", "assembly"):
                raise GDMLError(f"gdml {self.path}: structure element <{tg}> is not supported")
            v = Volume(el.get("name"), assembly=tg == "assembly")
            for c in el:
                self._volume_child(v, c)
            self.volumes[v.name] = v

    def _volume_child(self, v, c):
        ct = tag(c)
        if ct == "materialref":
            v.material = c.get("ref")
        elif ct == "solidref":
            v.solid = c.get("ref")
            if v.solid not in self.solids:
                raise GDMLError(f"volume {v.name!r}: solid {v.solid!r} is not defined")
        elif ct == "physvol":
            child = None
            for cc in c:
                if tag(cc) == "volumeref":
                    child = cc.get("ref")
                elif tag(cc) == "file":
                    raise GDMLError(f"volume {v.name!r}: a physvol <file> ({cc.get('name')!r}) is not supported")
            if child not in self.volumes:
                raise GDMLError(f"volume {v.name!r}: physvol {c.get('name')!r} places {child!r}, not defined before it")
            copy = c.get("copynumber")
            v.placements.append(Physvol(str(child), c.get("name") or f"{child}_PV",
                                        None if copy is None else int(round(self.num(copy, "copynumber"))),
                                        self.transform(c)))  # fmt: skip


def read(path) -> GDMLDocument:
    """Read a GDML file as Geant4 reads it (see the module docstring); what Geant4 refuses is a GDMLError."""
    return GDMLDocument(path).parse()
