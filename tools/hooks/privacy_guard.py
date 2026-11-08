"""Refuse a commit whose staged content, paths or message match a private pattern.

The patterns live outside the repository, in the file named by ``LUMEN_PRIVATE_PATTERNS``: one pattern per line,
read as a case-insensitive Python regular expression (a line that is not a valid expression is matched literally);
blank lines and lines starting with ``#`` are ignored. The guard never prints a pattern or the text it matched, only
where the match is, so its own output cannot leak what it protects.

    privacy_guard.py staged          pre-commit: every staged (added, copied, modified, renamed) file and its path
    privacy_guard.py message FILE    commit-msg: the commit message

With ``LUMEN_PRIVATE_PATTERNS`` unset the guard does nothing and says so. Set but unreadable, it refuses the commit.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

VARIABLE = "LUMEN_PRIVATE_PATTERNS"


def load_patterns(path: Path) -> list[re.Pattern[bytes]]:
    out = []
    for raw in path.read_bytes().splitlines():
        line = raw.strip()
        if not line or line.startswith(b"#"):
            continue
        try:
            out.append(re.compile(line, re.IGNORECASE))
        except re.error:
            out.append(re.compile(re.escape(line), re.IGNORECASE))
    return out


def first_match(patterns: list[re.Pattern[bytes]], data: bytes) -> tuple[int, int] | None:
    """(pattern number from 1, line number from 1) of the first match in data, or None."""
    for k, pat in enumerate(patterns, 1):
        m = pat.search(data)
        if m:
            return k, data.count(b"\n", 0, m.start()) + 1
    return None


def staged_paths() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"], check=True, capture_output=True
    ).stdout
    return [p.decode() for p in out.split(b"\0") if p]


def check_staged(patterns: list[re.Pattern[bytes]]) -> int:
    bad = 0
    for i, path in enumerate(staged_paths(), 1):
        hit = first_match(patterns, path.encode())
        if hit:
            print(f"privacy guard: the path of staged file #{i} matches pattern {hit[0]}", file=sys.stderr)
            bad += 1
            continue
        blob = subprocess.run(["git", "show", f":{path}"], check=True, capture_output=True).stdout
        hit = first_match(patterns, blob)
        if hit:
            print(f"privacy guard: {path} line {hit[1]} matches pattern {hit[0]}", file=sys.stderr)
            bad += 1
    return bad


def check_message(path: Path, patterns: list[re.Pattern[bytes]]) -> int:
    hit = first_match(patterns, path.read_bytes())
    if hit:
        print(f"privacy guard: the commit message, line {hit[1]}, matches pattern {hit[0]}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str]) -> int:
    name = os.environ.get(VARIABLE)
    if not name:
        print(f"privacy guard: {VARIABLE} is not set; nothing checked", file=sys.stderr)
        return 0
    try:
        patterns = load_patterns(Path(name))
    except OSError as e:
        print(f"privacy guard: cannot read the file named by {VARIABLE} ({e.strerror}); refusing", file=sys.stderr)
        return 1
    if argv[:1] == ["staged"]:
        bad = check_staged(patterns)
    elif argv[:1] == ["message"] and len(argv) == 2:
        bad = check_message(Path(argv[1]), patterns)
    else:
        print("usage: privacy_guard.py staged | message FILE", file=sys.stderr)
        return 2
    if bad:
        print(f"privacy guard: {bad} match(es); nothing private may enter the repository", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
