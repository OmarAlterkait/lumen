"""The how-to pages' code runs as written (the physics tier: it simulates on the GPU)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import doc_pages
from lumen import _env


def test_sources_page_runs(tmp_path, monkeypatch):
    ns = doc_pages.run("docs/howto/sources.md", tmp_path, monkeypatch)
    assert float(ns["light"].arrivals.sum()) > 0


@pytest.mark.skipif(_env.data_dir() is None, reason="needs LUMEN_DATA (the shipped detectors' GDML)")
def test_shipped_detectors_page_runs(tmp_path, monkeypatch):
    ns = doc_pages.run("docs/howto/shipped-detectors.md", tmp_path, monkeypatch)
    assert len(ns["det"].sensor_table()["name"]) > 0
