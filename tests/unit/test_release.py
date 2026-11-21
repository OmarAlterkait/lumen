"""The release's version strings agree: the README's status, CITATION.cff and the newest CHANGELOG heading."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_version_strings_agree():
    readme = re.search(r"\*\*Status: (\d+\.\d+\.\d+)\.\*\*", (ROOT / "README.md").read_text()).group(1)
    citation = re.search(r"^version: (\S+)$", (ROOT / "CITATION.cff").read_text(), re.M).group(1)
    changelog = re.search(r"^## (\S+) \(", (ROOT / "CHANGELOG.md").read_text(), re.M).group(1)
    install = set(re.findall(r"lumen\.git@v(\d+\.\d+\.\d+)", (ROOT / "docs/howto/install.md").read_text()))
    assert readme == citation == changelog and install == {readme}
