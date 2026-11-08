"""THE GDML reader: one parser, one CLHEP evaluator, lengths converted to metres by exact decimal scaling.

The file is untrusted input: no ``eval``; expressions go through a small recursive-descent parser with CLHEP units.

    doc = read(path)            # GDMLDocument: defines, materials, elements, isotopes, solids, volumes, surfaces
    write_normalised(src, dst)  # a copy Geant4's own reader accepts (the exporter defects below fixed)

Values: lengths are in m, a plain literal in a decimal length unit converted exactly (``12.5`` mm is exactly 0.0125,
rounded once) and anything else evaluated as Geant4 does and divided by 1000 once; angles in rad as Geant4 computes
them; matrices (property tables) in Geant4's internal units (MeV, mm, ns), with the source text of every value kept
so the importer can convert those exactly too.

Fail closed: an expression that does not evaluate, a reference to a missing position, rotation, solid or volume, and
every construct lumen does not support -- ``<loop>``, replicavol, divisionvol, paramvol, a physvol ``<file>`` -- raise
GDMLError.

Exporter defects (WCSim and experiment exports; Geant4's reader refuses each) are normalised and listed in
``doc.report``, never silently: a denormal number is 0; an identical re-definition of a define is one definition (a
conflicting one is an error); a matrix whose value count is not a multiple of its coldim loses the unpaired tail;
a non-finite matrix value is read as is and reported (``write_normalised`` writes +-1e30). A matrix that does not
evaluate raises only where it is used (``doc.matrix(name)``).

Notes
-----
Ported from the reader of lumen.scene.gdml (rebuild/next), itself from LUCiD's gdml_v2 reader, and extended with
what the importer needs: material compositions, elements, isotopes, temperatures and pressures, nested auxiliary
tags, the top-level userinfo, and the setup named ``Default``.
"""

from __future__ import annotations

import collections
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from lumen.gdml.units import LENGTH_M, MM_PER_M, UNITS, exact, is_literal


class GDMLError(ValueError):
    """A GDML file lumen cannot read as Geant4 would: the message names the element and the reason."""


FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "atan2": math.atan2, "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh, "exp": math.exp,
    "log": math.log, "log10": math.log10, "sqrt": math.sqrt, "pow": math.pow, "abs": abs, "fabs": abs,
    "min": min, "max": max, "floor": math.floor, "ceil": math.ceil, "int": lambda x: float(int(x)),
}  # fmt: skip
TINY = sys.float_info.min  # the smallest normal double: a nonzero number below it is a denormal
_TOK = re.compile(
    r"\s*(?:(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|(\*\*|[-+*/^(),\[\]]))"
)
ANGLE_ATTRS = {"startphi", "deltaphi", "starttheta", "deltatheta", "theta", "phi", "alpha", "alpha1", "alpha2",
               "phitwist", "twistedangle", "inst", "outst", "alph"}  # fmt: skip
UNITLESS_ATTRS = {"numsides", "lowx", "lowy", "lowz", "highx", "highy", "highz", "scalingfactor", "zorder", "sx",
                  "sy", "sz", "nseg", "nsides"}  # fmt: skip
SKIP_ATTRS = {"name", "lunit", "aunit", "unit", "ref", "type"}
BOOLEAN = ("union", "subtraction", "intersection")
UNSUPPORTED = ("replicavol", "divisionvol", "paramvol", "loop")


def _denormal(v: float) -> bool:
    return v != 0.0 and abs(v) < TINY


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
        self.n_denormal = 0

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
        try:
            v = float(s)
            if _denormal(v):
                self.n_denormal += 1
                return 0.0
            return v
        except ValueError:
            pass
        toks, pos = [], 0
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
        saved = (getattr(self, "_t", None), getattr(self, "_p", 0))
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
        return float(f(*args))

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
class Aux:
    """An auxiliary tag and its nested tags."""

    type: str
    value: str
    unit: str = ""
    children: list = field(default_factory=list)


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
    aux: list = field(default_factory=list)
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
        self.userinfo = []
        self.matrix_text = {}  # name -> the source tokens of its values
        self.world = None
        self.report = []
        self._matrix_err = {}
        self._defs = {}

    # ---- numbers
    def num(self, s, what):
        """Return the value of an expression in Geant4's internal units; the error names ``what``."""
        try:
            return self.ev.eval(s)
        except GDMLError as e:
            raise GDMLError(f"{what}: {e}") from None

    def length(self, text, unit, what):
        """Return a length in m: exact for a plain literal in a decimal unit, else Geant4's value / 1000 once."""
        unit = "mm" if unit is None else unit.strip()
        if is_literal(text) and unit in LENGTH_M:
            if _denormal(float(text)):
                self.ev.n_denormal += 1
                return 0.0
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
        """Return a matrix as (rows, coldim) float64 in Geant4's internal units; raise if undefined or unreadable."""
        if name in self._matrix_err:
            raise GDMLError(f"matrix {name!r}: {self._matrix_err[name]}")
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
        for sec in sections["userinfo"]:
            self.userinfo += [self._aux(c) for c in sec if tag(c) == "auxiliary"]
        self.world = self._world(sections["setup"])
        if self.world not in self.volumes:
            raise GDMLError(f"gdml {self.path}: world volume {self.world!r} is not defined")
        if self.ev.n_denormal:
            self.report.append(f"normalised: {self.ev.n_denormal} denormal literal(s) outside matrices read as 0")
        return self

    def _world(self, setups):
        named = [s for s in setups if s.get("name") == "Default"]
        for s in named or setups[:1]:
            for c in s:
                if tag(c) == "world":
                    return c.get("ref")
        return None

    def _redefined(self, el):
        """Whether a define was seen before: identical -> True (read once, reported); conflicting -> GDMLError."""
        nm = el.get("name")
        key = (tag(el), tuple(sorted(el.attrib.items())), (el.text or "").strip())
        if nm not in self._defs:
            self._defs[nm] = key
            return False
        if self._defs[nm] != key:
            raise GDMLError(f"gdml {self.path}: conflicting re-definition of {tag(el)} {nm!r}")
        self.report.append(f"normalised: identical re-definition of {tag(el)} {nm!r} read once")
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
        """Read a <matrix>: values evaluated in internal units, the source tokens kept."""
        nm = el.get("name")
        toks = (el.get("values") or "").split()
        try:
            cd = int(round(self.ev.eval(el.get("coldim", "1"))))
            if cd < 1:
                raise GDMLError(f"coldim {cd}")
            vals, nd = [], 0
            for tok in toks:
                try:
                    v = float(tok)
                except ValueError:
                    v = self.ev.eval(tok)
                if _denormal(v):
                    v, nd = 0.0, nd + 1
                vals.append(v)
            if nd:
                self.report.append(f"normalised: matrix {nm!r}: {nd} denormal value(s) read as 0")
            nf = sum(1 for v in vals if not math.isfinite(v))
            if nf:
                self.report.append(f"normalised: matrix {nm!r}: {nf} non-finite value(s) (kept; +-1e30 in "
                                   "write_normalised)")  # fmt: skip
            if len(vals) % cd:
                self.report.append(f"normalised: matrix {nm!r}: {len(vals)} values for coldim {cd}, the unpaired tail "
                                   f"({len(vals) % cd}) dropped")  # fmt: skip
                vals = vals[: len(vals) - len(vals) % cd]
                toks = toks[: len(vals)]
            self.ev.matrices[nm] = (cd, vals)
            self.matrix_text[nm] = toks
        except GDMLError as e:
            self._matrix_err[nm] = str(e)

    def read_materials(self, sec):
        """Read a <materials> section: isotopes, elements, materials with their compositions and properties."""
        for el in sec:
            tg, nm = tag(el), el.get("name")
            if tg == "isotope":
                atom = el.find("atom")
                self.isotopes[nm] = Isotope(nm, int(float(el.get("Z"))), int(float(el.get("N"))), self._atom(atom))
            elif tg == "element":
                atom = el.find("atom")
                fr = [(c.get("ref"), self.num(c.get("n"), f"element {nm!r}")) for c in el if tag(c) == "fraction"]
                self.elements[nm] = Element(nm, el.get("formula", ""), None if el.get("Z") is None else
                                            self.num(el.get("Z"), f"element {nm!r}"),
                                            None if atom is None else self._atom(atom), fr)  # fmt: skip
            elif tg == "material":
                self.materials[nm] = self._material(el)

    def _atom(self, atom):
        """Return the molar mass of an <atom> in g/mol."""
        if atom is None:
            return None
        g_mol = UNITS["g"] / UNITS["mole"]
        return self.num(atom.get("value"), "atom") * self.num(atom.get("unit", "g/mole"), "atom unit") / g_mol

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
                m.pressure = self.num(c.get("value"), what) * self.num(c.get("unit", "pascal"), what) / \
                    UNITS["pascal"]  # fmt: skip
            elif ct == "atom":
                m.molar_mass_g_mol = self._atom(c)
            elif ct == "fraction":
                m.fractions.append((c.get("ref"), self.num(c.get("n"), what)))
            elif ct == "composite":
                m.composites.append((c.get("ref"), int(round(self.num(c.get("n"), what)))))
        if el.get("Z") is not None:
            m.z = self.num(el.get("Z"), what)
        return m

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
        lu, au = el.get("lunit"), el.get("aunit")
        for a, v in el.attrib.items():
            al = a.lower()
            if al in SKIP_ATTRS or al.startswith("vertex"):
                continue
            if al in ANGLE_ATTRS:
                s.params[al] = self.angle(v, au, f"{what} attribute {a}")
            elif al in UNITLESS_ATTRS or (tg == "elcone" and al in ("dx", "dy")):
                s.params[al] = self.num(v, f"{what} attribute {a}")
            else:
                s.params[al] = self.length(v, lu, f"{what} attribute {a}")
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
                s.facets.append((frefs, (c.get("type") or "ABSOLUTE").upper()))

    def _solid_children(self, el, s, lu, what):
        tg = s.type
        if tg in ("polycone", "polyhedra"):
            s.planes = [tuple(self.length(c.get(k, "0"), lu, f"{what} zplane") for k in ("rmin", "rmax", "z"))
                        for c in el if tag(c) == "zplane"]  # fmt: skip
        elif tg in ("genericPolycone", "genericPolyhedra"):
            s.planes = [tuple(self.length(c.get(k, "0"), lu, f"{what} rzpoint") for k in ("r", "z"))
                        for c in el if tag(c) == "rzpoint"]  # fmt: skip
        elif tg == "xtru":
            for c in el:
                if tag(c) == "twoDimVertex":
                    s.vertices.append((self.length(c.get("x"), lu, what), self.length(c.get("y"), lu, what)))
                elif tag(c) == "section":
                    s.sections.append((self.length(c.get("zPosition"), lu, what),
                                       self.length(c.get("xOffset"), lu, what),
                                       self.length(c.get("yOffset"), lu, what),
                                       self.num(c.get("scalingFactor", "1"), what)))  # fmt: skip

    def _aux(self, c):
        return Aux(c.get("auxtype", ""), c.get("auxvalue", ""), c.get("auxunit", ""),
                   [self._aux(cc) for cc in c if tag(cc) == "auxiliary"])  # fmt: skip

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
        elif ct == "auxiliary":
            v.aux.append(self._aux(c))
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
            v.placements.append(Physvol(child, c.get("name") or child + "_PV",
                                        None if copy is None else int(round(self.num(copy, "copynumber"))),
                                        self.transform(c)))  # fmt: skip


def read(path) -> GDMLDocument:
    """Read a GDML file; see the module docstring."""
    return GDMLDocument(path).parse()


# ---------------------------------------------------------------------------------------- a Geant4-readable copy
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+")


def write_normalised(src, dst):
    """Write a copy of a GDML file with the exporter defects fixed as ``read`` fixes them; return the report lines.

    Denormal literals become 0, identical re-definitions are dropped (a conflicting one is an error), infinite matrix
    values become +-1e30 (a NaN is an error) and matrices are trimmed to a multiple of their coldim.
    """
    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    tree = ET.parse(src)
    root = tree.getroot()
    rep, nden = [], [0]

    def fix(text):
        def f(mo):
            if _denormal(float(mo.group(0))):
                nden[0] += 1
                return "0"
            return mo.group(0)

        return _NUM.sub(f, text)

    for el in root.iter():
        for k, v in list(el.attrib.items()):
            nv = fix(v)
            if nv != v:
                el.set(k, nv)
    seen = {}
    for sec in root:
        if tag(sec) == "define":
            for el in list(sec):
                _normalise_define(src, sec, el, seen, rep)
    if nden[0]:
        rep.append(f"{nden[0]} denormal value(s) written as 0")
    tree.write(dst, xml_declaration=True, encoding="UTF-8")
    return rep


def _normalise_define(src, sec, el, seen, rep):
    nm = el.get("name")
    if nm is None:
        return
    key = (tag(el), tuple(sorted(el.attrib.items())), (el.text or "").strip())
    if nm in seen:
        if seen[nm] != key:
            raise GDMLError(f"gdml {src}: conflicting re-definition of {tag(el)} {nm!r}")
        sec.remove(el)
        rep.append(f"identical re-definition of {tag(el)} {nm!r} dropped")
        return
    seen[nm] = key
    if tag(el) != "matrix":
        return
    vals = (el.get("values") or "").split()
    nonf = [k for k, v in enumerate(vals) if v.lower().lstrip("+-") in ("inf", "infinity", "nan")]
    for k in nonf:
        if "nan" in vals[k].lower():
            raise GDMLError(f"gdml {src}: matrix {nm!r} has a nan value")
        vals[k] = "-1e30" if vals[k].startswith("-") else "1e30"
    if nonf:
        el.set("values", " ".join(vals))
        rep.append(f"matrix {nm!r}: {len(nonf)} infinite value(s) written as +-1e30")
    cd = int(round(float(el.get("coldim", "1"))))
    if len(vals) % cd:
        el.set("values", " ".join(vals[: len(vals) - len(vals) % cd]))
        rep.append(f"matrix {nm!r}: {len(vals)} values for coldim {cd}, tail dropped")
