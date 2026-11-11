"""The reference pages stay true to the code: the API page is what tools/docs/api_reference.py writes from the
docstrings, and the value-set page's code runs."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import doc_pages

ROOT = Path(__file__).resolve().parents[2]


def test_api_page_is_generated_from_the_docstrings():
    r = subprocess.run([sys.executable, str(ROOT / "tools/docs/api_reference.py"), "--check"], capture_output=True,
                       text=True, cwd=ROOT)  # fmt: skip
    assert r.returncode == 0, r.stdout + r.stderr


def test_value_sets_page_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for i, code in enumerate(doc_pages.blocks("docs/reference/value-sets.md")):
        exec(compile(code, f"value-sets.md[{i}]", "exec"), {})
    assert (tmp_path / "my_pmt.json").exists()
