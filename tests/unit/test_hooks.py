"""The commit hooks: the privacy guard and the plain-words check."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / "tools" / "hooks" / "privacy_guard.py"
JARGON = ROOT / "tools" / "hooks" / "jargon.py"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def guard(repo, patterns, *args):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(repo)}
    if patterns is not None:
        env["LUMEN_PRIVATE_PATTERNS"] = str(patterns)
    return subprocess.run([sys.executable, str(GUARD), *args], cwd=repo, env=env, capture_output=True, text=True)


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    return repo


def test_privacy_guard_refuses_content_paths_and_messages(tmp_path):
    repo = make_repo(tmp_path)
    pats = tmp_path / "patterns.txt"
    pats.write_text("# a comment\n\nforbiddenword\n")
    (repo / "ok.py").write_text("x = 1\n")
    git(repo, "add", "ok.py")
    assert guard(repo, pats, "staged").returncode == 0
    (repo / "bad.py").write_text("a = 1\n# mentions ForbiddenWord here\n")
    git(repo, "add", "bad.py")
    r = guard(repo, pats, "staged")
    assert r.returncode == 1
    assert "bad.py line 2 matches pattern 1" in r.stderr
    assert "forbidden" not in r.stderr.lower()  # never prints what it protects
    git(repo, "rm", "-q", "--cached", "bad.py")
    (repo / "forbiddenword_notes.txt").write_text("harmless\n")
    git(repo, "add", "forbiddenword_notes.txt")
    r = guard(repo, pats, "staged")
    assert r.returncode == 1 and "path of staged file" in r.stderr and "forbidden" not in r.stderr.lower()
    msg = tmp_path / "msg"
    msg.write_text("Fix the forbiddenWord reader\n")
    assert guard(repo, pats, "message", str(msg)).returncode == 1
    msg.write_text("Fix the reader\n")
    assert guard(repo, pats, "message", str(msg)).returncode == 0


def test_privacy_guard_unset_is_a_noop_and_unreadable_refuses(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "a.txt").write_text("forbiddenword\n")
    git(repo, "add", "a.txt")
    r = guard(repo, None, "staged")
    assert r.returncode == 0 and "not set" in r.stderr
    assert guard(repo, tmp_path / "missing.txt", "staged").returncode == 1


def push_guard(repo, patterns, ref="refs/heads/main"):
    """Run the guard's pre-push check on the push of HEAD to a remote that has none of it."""
    sha = git(repo, "rev-parse", "HEAD").stdout.strip()
    env = {"PATH": "/usr/bin:/bin", "HOME": str(repo), "LUMEN_PRIVATE_PATTERNS": str(patterns)}
    line = f"{ref} {sha} {ref} {'0' * 40}\n"
    return subprocess.run([sys.executable, str(GUARD), "push", "origin", "url"], cwd=repo, env=env, input=line,
                          capture_output=True, text=True)  # fmt: skip


def commit(repo, name, data, author="A U Thor <a@example.org>"):
    (repo / name).write_bytes(data)
    git(repo, "add", name)
    git(repo, "-c", "user.name=c", "-c", "user.email=c@example.org", "commit", "-q", "--author", author, "-m", "add")


def test_privacy_guard_pre_push_reads_commits_archives_authors_and_refs(tmp_path):
    import io

    import numpy as np

    repo = make_repo(tmp_path)
    pats = tmp_path / "patterns.txt"
    pats.write_text("forbiddenword\n")
    commit(repo, "ok.txt", b"fine\n")
    assert push_guard(repo, pats).returncode == 0
    buf = io.BytesIO()
    np.savez_compressed(buf, names=np.array(["a", "the_ForbiddenWord_pmt"]))  # UTF-32 inside a deflated member
    commit(repo, "names.npz", buf.getvalue())
    r = push_guard(repo, pats)
    assert r.returncode == 1 and "archive member" in r.stderr and "forbidden" not in r.stderr.lower()
    git(repo, "reset", "-q", "--hard", "HEAD~1")
    commit(repo, "two.txt", b"fine\n", author="forbiddenword <x@example.org>")
    assert push_guard(repo, pats).returncode == 1
    git(repo, "reset", "-q", "--hard", "HEAD~1")
    r = push_guard(repo, pats, ref="refs/tags/forbiddenword-1")
    assert r.returncode == 1 and "ref name" in r.stderr


def test_jargon_hook(tmp_path):
    good = tmp_path / "errors.py"
    good.write_text('class LumenError(Exception):\n    """Raised when a photon is lost."""\n')
    bad = tmp_path / "api.py"
    bad.write_text('def f():\n    raise ValueError("the optical class has no chart (see E1)")\n')
    doc = tmp_path / "tut.md"
    doc.write_text("Plain prose.\n\n```python\nclass R9000:  # a model\n    pass\n```\n\nThe overlay is applied.\n")
    run = lambda *p: subprocess.run([sys.executable, str(JARGON), *map(str, p)], capture_output=True, text=True)  # noqa: E731
    assert run(good).returncode == 0
    r = run(bad, doc)
    assert r.returncode == 1
    for word in ("'class'", "'chart'", "'E1'", "'overlay'"):
        assert word in r.stdout
    assert "tut.md:4" not in r.stdout  # the keyword `class` in code is fine
