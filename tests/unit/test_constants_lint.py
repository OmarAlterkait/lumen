"""The constants lint (tools/lint/constants.py): no bare number in src/lumen outside the modules that state numbers."""

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("constants_lint", ROOT / "tools" / "lint" / "constants.py")
lint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lint)


def _hits(code: str) -> list:
    finder = lint._Finder("x.py")
    finder.visit(ast.parse(code))
    return [lit for *_, lit in finder.hits]


def test_every_number_in_src_is_stated_or_listed():
    problems = lint.check()
    assert not problems, "\n".join(problems)


def test_the_rules_flag_a_bare_number_and_pass_structure():
    assert _hits("x = 0.37 * y") == ["0.37"]
    assert _hits("def f(h=1e-6):\n    return h") == ["1e-06"]
    assert _hits("x = -3.5") == ["-3.5"]
    # structural: trivial values, indices, shapes, shifts, int exponents, dtype sizes, the imaginary unit, styling
    assert _hits("y = 2 * TWO_PI * x[3] + 0.5 * z ** 3 - 1") == []
    assert _hits("a = jnp.zeros((n, 3)); b = a.reshape(-1, 4); c = 1 << 20; d = k * x.itemsize * 4") == []
    assert _hits("e = len(v) == 3; f = 1j * g; ax.plot(x, y, lw=0.8, figsize=(6, 4))") == []
    assert _hits("FORMAT_VERSION = 4\nfaces = [(0, 1, 3, 2), (4, 5, 7, 6)]") == []


def test_a_stale_entry_fails(tmp_path):
    allow = tmp_path / "allow.txt"
    allow.write_text(lint.ALLOW.read_text() + "nowhere.py | f | 0.37 | 1 | choice | none\n")
    assert any("nowhere.py" in p for p in lint.check(allow=allow))
