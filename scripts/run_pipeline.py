#!/usr/bin/env python
"""Run the audit and print the gate.

With no METRICAUDIT_DATA_DIR set this reads the synthetic session, stamps the
output accordingly, and publishes no provenance.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metricaudit.ingest import load  # noqa: E402
from metricaudit.pipeline import gate_report, run  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    corpus = load()
    print(f"source: {corpus.source}")
    for name, frame in sorted(corpus.frames.items()):
        mapping = corpus.column_map[name]
        parsed = mapping["parsed_for_numeric_keys_only"]
        note = (f", {len(parsed)} parsed for declared numeric keys" if parsed else "")
        print(f"  {name:<14} {len(frame):>6} rows, "
              f"{len(mapping['read']):>2} columns read{note}")

    results = run(corpus)
    print()
    print(gate_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
