#!/usr/bin/env python
"""Print the audit's findings in readable form, for checking a run by eye.

Writes nothing. Everything it shows comes from outputs/results.json, so it is
safe to run against a completed audit without re-executing it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def rule(title: str) -> None:
    print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")


def frame(records) -> pd.DataFrame:
    return pd.DataFrame(records) if records else pd.DataFrame()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = ROOT / "outputs" / "results.json"
    if not path.exists():
        print(f"no results at {path}; run scripts/run_pipeline.py first")
        return 1
    r = json.loads(path.read_text(encoding="utf-8"))

    rule(f"corpus, source = {r['corpus']['source']}")
    print(f"  groups: {r['corpus']['n_groups']}   "
          f"ruleset: {r['corpus']['ruleset_version']}")

    rule("census and parity")
    c = r["census"]
    print(f"  operational {c['n_operational']}   reporting {c['n_reporting']}   "
          f"raw difference {c['raw_difference']}")
    print(f"  exclusive to reporting: {c['exclusive_to_reporting']}")
    print(f"  exclusive to operational: {c['exclusive_to_operational']}")
    print(f"  difference explained by exclusive categories: "
          f"{c['difference_explained_by_exclusive_categories']}")
    print(f"  residual difference: {c['residual_difference']}")

    print("\n  identifier parity (M3):")
    print(frame(r["parity_identifiers"]).to_string(index=False))
    print("\n  composite-key parity within the write window:")
    print(frame(r["parity_composite"]).to_string(index=False))
    print("\n  duplication within each store:")
    print(frame(r["internal_duplication"]).to_string(index=False))

    rule("reconciliation by indicator (M2)")
    print(frame(r["reconciliation_by_indicator"]).to_string(index=False))

    rule("reconciliation by group, divergent indicators only")
    g = frame(r["reconciliation_by_group"])
    if not g.empty:
        bad = g[~g["matches"].astype(bool)]
        cols = [c for c in ("indicator", "group_number", "reported", "recomputed",
                            "absolute_difference", "ratio", "direction")
                if c in bad.columns]
        print(bad[cols].to_string(index=False))

    rule("divergence summary, extreme case separated (M12)")
    print(frame(r["divergence"]).to_string(index=False))

    rule("mechanism attribution (M6, M7)")
    print(frame(r["attribution"]).to_string(index=False))

    rule("candidate predictions per group")
    m = frame(r["prediction_matrix"])
    if not m.empty:
        print(m.to_string(index=False))

    rule("completeness (M8)")
    print(frame(r["completeness"]).to_string(index=False))
    print(f"\n  display-name drift: {r['name_drift']}")
    print("\n  unnamed by category:")
    print(frame(r["unnamed_by_category"]).to_string(index=False))

    rule("archive collections and redundancy (D6)")
    print(f"  collections: {r['archive_collections']}")
    print(frame(r["archive_redundancy"]).to_string(index=False))

    rule("downstream consequence (M9, M10, M4, M5)")
    for indicator, block in r["consequence"].items():
        print(f"\n  --- {indicator} ---")
        imp = block["impact"]
        for label in ("reported", "reconciled", "reported_without_extreme",
                      "reconciled_without_extreme"):
            d = imp[label]
            print(f"    {label:<28} n={d['n']:<3} mean={d['mean']:>10.3f} "
                  f"sd={d['sd']:>10.3f} median={d['median']:>8.2f} "
                  f"min={d['min']:>6.1f} max={d['max']:>8.1f}")
        print(f"    mean inflation factor: {imp['mean_inflation_factor']:.3f}")
        print(f"    sd inflation factor:   {imp['sd_inflation_factor']:.3f}")
        print(f"    extreme group: {imp['extreme_group']}")
        print(f"    rank agreement: {imp['rank_agreement']}")
        print(f"    divergence interval: {block['divergence_interval']}")
        print(f"    agreement (log scale): {block['agreement']}")
        a = block["association"]
        if a:
            print(f"    association with {a['outcome']}:")
            print(f"      from reported:   {a['from_reported']}")
            print(f"      from reconciled: {a['from_reconciled']}")

    rule("assumption sweep (M11)")
    print(frame(r["sweep"]).to_string(index=False))
    print("\n  dependence:")
    print(frame(r["assumption_dependence"]).to_string(index=False))
    print(f"\n  aggregation order: {r['aggregation_order']}")

    rule("gates")
    for name, state in r["gates"].items():
        print(f"  {str(state.get('pass')):<6} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
