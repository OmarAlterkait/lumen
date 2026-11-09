"""Refuse a commit whose staged content, paths or message match a private pattern.

The patterns live outside the repository, in the file named by ``LUMEN_PRIVATE_PATTERNS``: one pattern per line,
read as a case-insensitive Python regular expression (a line that is not a valid expression is matched literally);
blank lines and lines starting with ``#`` are ignored. The guard never prints a pattern or the text it matched, only
where the match is, so its own output cannot leak what it protects.

    privacy_guard.py staged          pre-commit: every staged (added, copied, modified, renamed) file and its path
    privacy_guard.py message FILE    commit-msg: the commit message
    privacy_guard.py push REMOTE URL pre-push: every outgoing commit (refs on stdin, as git gives them): each
                                     commit's raw object (author, committer, message), the paths and blobs it adds or
                                     changes, and the local and remote ref names (branches and tags)

A blob that is a zip archive (``.npz`` included) or gzip stream is also checked member by member, decompressed, with
member names; an archive member is checked a second time with its zero bytes removed, which reads the UTF-32 and
UTF-16 text of a NumPy string array as plain text.

With ``LUMEN_PRIVATE_PATTERNS`` unset the guard does nothing and says so (outside contributors). Set but unreadable,
it refuses (fails closed).
"""

from __future__ import annotations

import gzip
import io
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

VARIABLE = "LUMEN_PRIVATE_PATTERNS"
ZERO = "0" * 40  # git's id of a ref that does not exist (a new remote ref, or a deleted local one)
ZIP_MAGIC, GZIP_MAGIC = b"PK\x03\x04", b"\x1f\x8b"


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


def views(data: bytes):
    """Yield ``(where, bytes)`` to check for one blob: itself and, for an archive, each decompressed member."""
    yield "", data
    try:
        if data.startswith(ZIP_MAGIC):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for k, info in enumerate(z.infolist(), 1):
                    member = z.read(info)
                    yield f" (archive member #{k}, its name)", info.filename.encode()
                    yield f" (archive member #{k})", member
                    yield f" (archive member #{k}, zero bytes removed)", member.replace(b"\0", b"")
        elif data.startswith(GZIP_MAGIC):
            yield " (decompressed)", gzip.decompress(data)
    except (zipfile.BadZipFile, OSError, EOFError) as e:  # an unreadable archive cannot be cleared: refuse it
        raise ValueError(f"cannot decompress ({type(e).__name__})") from None


def blob_hit(patterns: list[re.Pattern[bytes]], data: bytes) -> str | None:
    """Return where in a blob (``line N``, with the archive member) the first match is, or None."""
    try:
        for where, view in views(data):
            hit = first_match(patterns, view)
            if hit:
                return f"line {hit[1]}{where} matches pattern {hit[0]}"
    except ValueError as e:
        return str(e)
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
        hit = blob_hit(patterns, blob)
        if hit:
            print(f"privacy guard: {path} {hit}", file=sys.stderr)
            bad += 1
    return bad


def git_out(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def outgoing(lines: list[str]) -> tuple[list[str], list[str]]:
    """Return ``(ref names, commit ids)`` of a push, from pre-push's stdin lines."""
    refs, commits = [], []
    for line in lines:
        parts = line.split()
        if len(parts) != 4:
            continue
        local_ref, local_sha, remote_ref, remote_sha = parts
        refs += [local_ref, remote_ref]
        if local_sha == ZERO:  # a deletion sends no commit
            continue
        span = [local_sha, "--not", "--remotes"] if remote_sha == ZERO else [f"{remote_sha}..{local_sha}"]
        commits += [c for c in git_out("rev-list", *span).decode().split() if c not in commits]
    return refs, commits


def check_commit(patterns: list[re.Pattern[bytes]], sha: str, n: int) -> int:
    """Check one outgoing commit: its raw object, and every path and blob it adds or changes."""
    bad = 0
    hit = first_match(patterns, git_out("cat-file", "commit", sha))
    if hit:
        print(f"privacy guard: outgoing commit #{n} (author, committer or message) matches pattern {hit[0]}",
              file=sys.stderr)  # fmt: skip
        bad += 1
    raw = git_out("diff-tree", "-r", "--root", "--no-commit-id", "--raw", "-z", "--diff-filter=ACMR", sha).split(b"\0")
    for head, path in zip(raw[0::2], raw[1::2], strict=False):  # the -z output ends in a separator
        blob_sha = head.split()[3].decode()
        hit = first_match(patterns, path)
        where = "a path" if hit else blob_hit(patterns, git_out("cat-file", "blob", blob_sha))
        if hit or where:
            shown = f"file #{bad + 1}" if hit else path.decode(errors="replace")
            print(f"privacy guard: outgoing commit #{n}: {shown} {'matches' if hit else where}", file=sys.stderr)
            bad += 1
    return bad


def check_push(patterns: list[re.Pattern[bytes]], lines: list[str]) -> int:
    refs, commits = outgoing(lines)
    bad = sum(check_commit(patterns, sha, n) for n, sha in enumerate(commits, 1))
    for k, ref in enumerate(refs, 1):
        if first_match(patterns, ref.encode()):
            print(f"privacy guard: ref name #{k} of the push matches a pattern", file=sys.stderr)
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
    elif argv[:1] == ["push"]:
        bad = check_push(patterns, sys.stdin.read().splitlines())
    else:
        print("usage: privacy_guard.py staged | message FILE | push [REMOTE URL]", file=sys.stderr)
        return 2
    if bad:
        print(f"privacy guard: {bad} match(es); nothing private may enter the repository", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
