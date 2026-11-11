"""Style rule 10 (docs/dev/style.md): text a user reads uses plain words, never internal terms or plan codes.

Checked: the string literals (messages, docstrings) of the Python files given, and the prose, code comments and
string literals of the Markdown files given (fenced Python code is tokenized, so the keyword ``class`` is fine but a
message saying "optical class" is not). Inline code spans in Markdown prose (`...`) are not checked: they quote
identifiers, or messages whose source is checked where it is written. Prints file:line and the word; exits 1 on any
hit.

    jargon.py FILE...
"""

from __future__ import annotations

import io
import re
import sys
import tokenize
from pathlib import Path

WORDS = (
    r"charts?",
    r"hold",
    r"held",
    r"slots?",
    r"lanes?",
    r"kinds?",
    r"class(es)?",
    r"authority",
    r"descriptions?",
    r"overlays?",
    r"fold(s|ed|ing)?",
    r"lockstep",
    r"phase [ABC]",
    r"decisions? \d+",
    r"edge rows?",
    r"the baseline'?s?",
    r"the port'?s?",
    r"port steps?",
    r"job \d{7,}",
    r"[a-z_]+\.md section",
    r"naming pass",
)
CODES = (
    r"[EDGSR][1-9]",  # check codes: E1, D1, G1, G2, S1, R1-R3 (not R0, a reflectance at normal incidence)
    r"T[1-9]",  # readout toys T1-T9
    r"B0",
    r"[AB]\d{1,2}b?",  # the baseline and the port steps (A0-A9, A6b, B1-B13)
    r"D\d{1,2}",  # decision numbers
    r"OUT",  # the edge rule's internal labels (a value Geant4 never reads; a parameter at its range edge)
    r"WITHIN",
    r"FDX",  # the fixed-reference finite difference
)
# codes are case-sensitive (an "a1" in a variable is not a plan code); words are not
CODE_RE = re.compile(r"\b(" + "|".join(CODES) + r")\b")
WORD_RE = re.compile(r"\b(" + "|".join(WORDS) + r")\b", re.IGNORECASE)


def hits(text: str) -> list[str]:
    return [m.group(0) for m in WORD_RE.finditer(text)] + [m.group(0) for m in CODE_RE.finditer(text)]


def python_texts(source: str, first_line: int = 1) -> list[tuple[int, str]]:
    """(line, text) of every string literal and comment of a Python source."""
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type in (tokenize.STRING, tokenize.COMMENT):
                out.append((first_line + tok.start[0] - 1, tok.string))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        out = [(first_line + i, line) for i, line in enumerate(source.splitlines())]
    return out


def markdown_texts(source: str) -> list[tuple[int, str]]:
    out, fence, block, start = [], None, [], 0
    for i, line in enumerate(source.splitlines(), 1):
        m = re.match(r"^\s*(```|~~~)(\w*)", line)
        if fence is None and m:
            fence, block, start = m.group(2).lower(), [], i + 1
        elif fence is not None and m:
            code = "\n".join(block)
            out += (
                python_texts(code, start)
                if fence in ("python", "py", "")
                else [(start + k, x) for k, x in enumerate(block)]
            )
            fence = None
        elif fence is not None:
            block.append(line)
        else:
            out.append((i, re.sub(r"`[^`]*`", "``", line)))
    return out


def main(paths: list[str]) -> int:
    bad = 0
    for p in paths:
        src = Path(p).read_text(encoding="utf-8")
        texts = python_texts(src) if p.endswith(".py") else markdown_texts(src)
        for line, text in texts:
            for word in hits(text):
                print(f"{p}:{line}: '{word}' is an internal term or code; use the user words of the glossary")
                bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
