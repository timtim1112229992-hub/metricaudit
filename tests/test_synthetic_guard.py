# -*- coding: utf-8 -*-
"""Tests that a synthetic run cannot be mistaken for, or overwrite, a real one.

The failure being prevented is quiet. Someone runs the package without setting
the session variable, the outputs are replaced by numbers from a generator, and
the next person to read them has no way to tell. These tests make that failure
loud.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metricaudit.config import SYNTHETIC_DIR  # noqa: E402
from metricaudit.ingest import ENV_VAR, data_dir, load  # noqa: E402
from metricaudit.pipeline import _refuse_synthetic_overwrite, run  # noqa: E402

SYNTHETIC_PRESENT = (SYNTHETIC_DIR / "08_group_progress.csv").exists()


def test_absent_environment_variable_selects_the_synthetic_session(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    if not SYNTHETIC_PRESENT:
        pytest.skip("synthetic session has not been generated")
    folder, source = data_dir()
    assert source == "synthetic"
    assert folder == SYNTHETIC_DIR


def test_a_bad_environment_variable_is_refused_rather_than_ignored(monkeypatch):
    """Falling back to synthetic on a typo would be the worst of both."""
    monkeypatch.setenv(ENV_VAR, str(ROOT / "no" / "such" / "place"))
    with pytest.raises(FileNotFoundError, match="not a directory"):
        data_dir()


def test_synthetic_run_refuses_to_overwrite_restricted_results(tmp_path,
                                                               monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    if not SYNTHETIC_PRESENT:
        pytest.skip("synthetic session has not been generated")
    (tmp_path / "results.json").write_text(
        json.dumps({"corpus": {"source": "restricted"}}), encoding="utf-8")
    corpus = load()
    with pytest.raises(RuntimeError, match="will not overwrite"):
        _refuse_synthetic_overwrite(corpus, tmp_path)


def test_synthetic_run_may_overwrite_its_own_results(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    if not SYNTHETIC_PRESENT:
        pytest.skip("synthetic session has not been generated")
    (tmp_path / "results.json").write_text(
        json.dumps({"corpus": {"source": "synthetic"}}), encoding="utf-8")
    _refuse_synthetic_overwrite(load(), tmp_path)


@pytest.mark.skipif(not SYNTHETIC_PRESENT, reason="no synthetic session")
def test_synthetic_run_publishes_no_provenance(tmp_path, monkeypatch):
    """A provenance package describes records. Synthetic records are not records."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    before = {p.name: p.read_bytes() for p in ROOT.joinpath("provenance").glob("*")} \
        if ROOT.joinpath("provenance").is_dir() else {}
    run(load(), output_dir=tmp_path)
    after = {p.name: p.read_bytes() for p in ROOT.joinpath("provenance").glob("*")} \
        if ROOT.joinpath("provenance").is_dir() else {}
    assert before == after, "a synthetic run altered the release package"


@pytest.mark.skipif(not SYNTHETIC_PRESENT, reason="no synthetic session")
def test_synthetic_output_is_stamped_as_synthetic(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    results = run(load(), output_dir=tmp_path)
    assert results["corpus"]["source"] == "synthetic"
    written = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert written["corpus"]["source"] == "synthetic"


@pytest.mark.skipif(not SYNTHETIC_PRESENT, reason="no synthetic session")
def test_the_synthetic_session_contains_a_fault_to_find(tmp_path, monkeypatch):
    """A stand-in in which everything reconciles would test nothing.

    The generator plants a gauge-summation fault. If the audit stops detecting
    it, either the generator or the detector has broken, and the package would
    otherwise pass its own suite while finding nothing.
    """
    monkeypatch.delenv(ENV_VAR, raising=False)
    results = run(load(), output_dir=tmp_path)
    by_indicator = {row["indicator"]: row["classification"]
                    for row in results["reconciliation_by_indicator"]
                    .to_dict(orient="records")}
    assert by_indicator["help_click_count"] == "divergent"
    assert by_indicator["interaction_count"] == "reconciled"

    mechanisms = " ".join(results["attribution"]["mechanism"].astype(str))
    assert "gauge_summation" in mechanisms
