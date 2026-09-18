#!/usr/bin/env python
"""Render the tables and figures from a completed run.

Reads outputs/results.json and writes into outputs/, which the repository
excludes. A figure is a finding, and findings travel with the paper.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metricaudit import figures, tables  # noqa: E402

OUTPUTS = ROOT / "outputs"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    results = OUTPUTS / "results.json"
    if not results.exists():
        print(f"no results at {results}; run scripts/run_pipeline.py first")
        return 1

    figures.render_all(results, OUTPUTS / "figures")
    print("rendering tables")
    tables.render_csv(results, OUTPUTS / "tables")
    try:
        tables.render_docx(results, OUTPUTS / "Tables.docx")
    except ImportError:
        print("  python-docx is not installed, so only the CSV tables were written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
