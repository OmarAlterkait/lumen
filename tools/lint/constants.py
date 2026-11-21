"""The constants lint: no bare number in src/lumen outside the modules that state numbers.

    python tools/lint/constants.py            # check: every hit is on the allowlist, every entry still matches
    python tools/lint/constants.py --list     # print every hit (file:line, enclosing name, literal)

A number of lumen's optics, transport, readout or checks lives in ``lumen.rules.choices`` (a choice, with what it
trades), ``lumen.rules.tolerances`` (the geometry's) or ``lumen.units`` (physical constants and units), named and with
its reason; every other module imports it. This check finds numeric literals anywhere else in src/lumen.

Exempt, as trivially structural (the whole list):

1. the values 0, 1, 2, -1 and 0.5 (and their float spellings);
2. a literal inside a subscript (an index or a slice);
3. an int argument of a shape or index function (``SHAPE_CALLS``) or an int keyword argument naming a shape, an axis
   or a count (``SHAPE_KEYWORDS``), an int compared with ``len(...)``, ``.ndim``, ``.shape`` or ``.size``, and an int
   in a tuple or list of int literals only (a shape, an index table: ``(n, 3)``, ``(0, 1, 3, 2)``);
4. an int operand of a bit shift, an int exponent (``x ** 3``), and an int beside ``.itemsize`` (a dtype's size);
5. an int in an assignment to a name containing "version" (a format's or a schema's version);
6. the imaginary unit ``1j``;
7. a number given to a plot's styling keyword (``STYLE_KEYWORDS``: sizes, widths, alphas), which is presentation.

A literal derived in place from named values passes when the literal itself is exempt (``2 * TWO_PI``); a strings'
content, a docstring and a bool are never numbers here. Every other hit must be on the allowlist
(``constants_allow.txt``), keyed by file, enclosing qualified name and literal (not by line, so an edit elsewhere does
not move it), with its count (``*``: every literal of that name, a published table) and its class: ``derivable`` (to
be written from named values), ``published`` (a Geant4 or published value, cited), ``choice`` (to move to
``lumen.rules.choices`` with its reason) or ``structural`` (a false positive of the rules above). The check fails
on a hit the list does not cover and on an entry that matches fewer hits than it states, so the list cannot go
stale.
"""

from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "lumen"
ALLOW = Path(__file__).with_name("constants_allow.txt")
# the modules that state numbers (relative to src/lumen)
STATED = {"rules/choices.py", "rules/tolerances.py", "units.py"}
TRIVIAL = {0, 1, 2, -1, 0.5}
SHAPE_CALLS = {
    "reshape", "zeros", "ones", "full", "empty", "zeros_like", "ones_like", "range", "arange", "tile", "repeat",
    "broadcast_to", "expand_dims", "squeeze", "one_hot", "eye", "identity", "split", "array_split", "moveaxis",
    "swapaxes", "transpose", "take", "take_along_axis", "roll", "fold_in", "split_keys", "islice", "enumerate",
    "itemgetter", "partition", "pad", "concatenate", "stack", "rjust", "ljust", "zfill", "group",
}  # fmt: skip
SHAPE_KEYWORDS = {"axis", "axes", "shape", "size", "ndim", "ndmin", "num", "n", "maxsize", "maxlen", "stacklevel",
                  "maxsplit", "start", "length", "count", "indent", "width", "depth", "dims", "keepdims"}  # fmt: skip
SHAPE_ATTRS = {"ndim", "shape", "size"}
STYLE_KEYWORDS = {"figsize", "lw", "linewidth", "ms", "markersize", "alpha", "fontsize", "shrink", "density",
                  "arrowsize", "capsize", "s", "mew", "elinewidth"}  # fmt: skip
CLASSES = ("derivable", "published", "choice", "structural")


def _value(node):
    """Return the number a node spells (a constant or a negated one), or None."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _value(node.operand)
        return None if v is None else (-v if isinstance(node.op, ast.USub) else v)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float, complex))
        and not isinstance(node.value, bool)
    ):
        return node.value
    return None


def _shape_like(node) -> bool:
    """Return whether ``node`` reads a shape: ``len(...)``, ``.ndim``, ``.shape`` (subscripted or not), ``.size``."""
    if isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len":
        return True
    return isinstance(node, ast.Attribute) and node.attr in SHAPE_ATTRS


def _call_name(func) -> str:
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""


class _Finder(ast.NodeVisitor):
    def __init__(self, path: str):
        self.path, self.scope, self.hits = path, [], []
        self.parents: dict = {}

    def visit_FunctionDef(self, node):  # noqa: N802
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815
    visit_ClassDef = visit_FunctionDef  # noqa: N815

    def generic_visit(self, node):
        for child in ast.iter_child_nodes(node):
            self.parents[child] = node
        if (v := _value(node)) is not None:
            if not self._exempt(node, v):
                self.hits.append((self.path, node.lineno, ".".join(self.scope) or "<module>", _spell(v)))
            return  # a negated constant is one literal
        super().generic_visit(node)

    def _exempt(self, node, v) -> bool:
        if v in TRIVIAL or v == 1j:
            return True
        integer = isinstance(v, int)
        child, parent = node, self.parents.get(node)
        while parent is not None:  # rule 2: anywhere inside a subscript's slice
            if isinstance(parent, ast.Subscript) and child is parent.slice:
                return True
            if isinstance(parent, (ast.stmt, ast.Call, ast.Lambda)):
                break
            child, parent = parent, self.parents.get(parent)
        parent = self.parents.get(node)
        if (
            integer
            and isinstance(parent, (ast.Tuple, ast.List))
            and all(isinstance(_value(e), int) for e in parent.elts)
        ):
            return True
        style = node
        while style is not None and not isinstance(style, (ast.stmt, ast.Call)):
            if isinstance(style, ast.keyword) and style.arg in STYLE_KEYWORDS:
                return True
            style = self.parents.get(style)
        while isinstance(parent, (ast.Tuple, ast.List)):  # a shape tuple: its call or keyword decides
            node, parent = parent, self.parents.get(parent)
        if integer and isinstance(parent, ast.Call) and node in parent.args:
            return _call_name(parent.func) in SHAPE_CALLS
        if integer and isinstance(parent, ast.keyword):
            return parent.arg in SHAPE_KEYWORDS
        if integer and isinstance(parent, ast.Compare):
            return any(_shape_like(x) for x in [parent.left, *parent.comparators])
        if integer and isinstance(parent, ast.BinOp):
            if isinstance(parent.op, (ast.LShift, ast.RShift)):
                return True
            if isinstance(parent.op, ast.Pow) and node is parent.right:
                return True
            other = parent.left if node is parent.right else parent.right
            if any(isinstance(x, ast.Attribute) and x.attr == "itemsize" for x in ast.walk(other)):
                return True
        if integer and isinstance(parent, (ast.Assign, ast.AnnAssign)):
            targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
            return any(isinstance(t, ast.Name) and "version" in t.id.lower() for t in targets)
        return False


def _spell(v) -> str:
    return repr(v)


def hits(src: Path = SRC) -> list:
    """Return every hit (path relative to src/lumen, line, enclosing qualified name, literal) under ``src``."""
    out = []
    for f in sorted(src.rglob("*.py")):
        rel = f.relative_to(src).as_posix()
        if rel in STATED:
            continue
        finder = _Finder(rel)
        finder.visit(ast.parse(f.read_text(), str(f)))
        out += finder.hits
    return out


def allowlist(path: Path = ALLOW) -> dict:
    """Return {(path, qualname, literal): (count, class, note)} of the allowlist (``path | name | literal | count |
    class | note`` per line; # comments)."""
    out = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5 or parts[4] not in CLASSES:
            raise ValueError(f"{path.name}:{n}: want 'path | name | literal | count | class | note', got {line!r}")
        key = (parts[0], parts[1], parts[2])
        if key in out:
            raise ValueError(f"{path.name}:{n}: {key} listed twice")
        out[key] = (int(parts[3]), parts[4], parts[5] if len(parts) > 5 else "")
    return out


def check(src: Path = SRC, allow: Path = ALLOW) -> list:
    """Return the problems: hits the allowlist does not cover and entries that match fewer hits than they state."""
    listed = allowlist(allow)
    # a hit counts against its literal's entry, else its name's '*' entry (a published table's every literal)
    found = [(p, line, q, lit if (p, q, lit) in listed or (p, q, "*") not in listed else "*")
             for p, line, q, lit in hits(src)]  # fmt: skip
    seen = Counter((p, q, lit) for p, _, q, lit in found)
    problems = []
    for p, line, q, lit in found:
        if seen[(p, q, lit)] > listed.get((p, q, lit), (0,))[0]:
            problems.append(
                f"src/lumen/{p}:{line}: {lit} in {q}: a number outside rules/choices.py, "
                "rules/tolerances.py and units.py (name it there, or derive it from named values)"
            )
    for key, (count, _, _) in listed.items():
        if seen[key] < count:
            problems.append(f"{allow.name}: {' | '.join(key)} lists {count}, matches {seen[key]}: update it")
    return sorted(set(problems))


def main(argv) -> int:
    """Run the check (or ``--list`` the hits); return the exit status."""
    if "--list" in argv:
        for p, line, q, lit in hits():
            print(f"src/lumen/{p}:{line}\t{q}\t{lit}")
        return 0
    problems = check()
    for p in problems:
        print(p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
