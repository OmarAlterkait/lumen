"""The how-to pages' code runs as written (the physics tier: it simulates on the GPU)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import doc_pages


def test_sources_page_runs(tmp_path, monkeypatch):
    ns = doc_pages.run("docs/howto/sources.md", tmp_path, monkeypatch)
    assert float(ns["light"].arrivals.sum()) > 0


def test_yaml_detector_page_runs(tmp_path, monkeypatch):
    """the hand-written YAML toy converts, loads and simulates"""
    ns = doc_pages.run("docs/howto/yaml-detector.md", tmp_path, monkeypatch)
    assert len(ns["det"].sensor_set.ids) == 6 and float(ns["light"].arrivals.sum()) > 0


def test_readme_quick_start_runs(tmp_path, monkeypatch):
    """the README's quick start, from a clone, on the toy detector shipped in examples/ (nothing downloaded)"""
    ns = doc_pages.run("README.md", tmp_path, monkeypatch)
    assert ns["light"].arrivals.shape == (8,) and float(ns["light"].arrivals.sum()) > 0
