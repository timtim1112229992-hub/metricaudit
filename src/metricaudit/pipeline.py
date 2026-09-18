"""P1 to P10 in order, and the gate that decides whether drafting may begin.

The run order is fixed and the gate is not advisory. The outline commits to not
writing the Results section until every criterion is met, and a gate that
reports its own failure and continues would make that commitment decorative. So
the gate is computed, recorded, and returned as a pass or a refusal.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import attribute, audit, index, parity, provenance, reconcile, simulate, sweep
from .config import OUTPUT_DIR, SETTINGS
from .ingest import Corpus, load
from .rules import RULES, RULESET_VERSION


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _refuse_synthetic_overwrite(corpus: Corpus, output_dir: Path) -> None:
    """A synthetic run must not replace figures computed from the session.

    The failure this prevents is quiet and expensive: someone runs the package
    without setting the corpus variable, the outputs are replaced by numbers
    from a generator, and the next person to read them has no way to tell.
    """
    existing = output_dir / "results.json"
    if corpus.source != "synthetic" or not existing.exists():
        return
    try:
        held = json.loads(existing.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if held.get("corpus", {}).get("source") == "restricted":
        raise RuntimeError(
            f"{existing} holds results computed from the restricted session. A "
            f"synthetic run will not overwrite them. Set METRICAUDIT_DATA_DIR, "
            f"or write elsewhere.")


def run(corpus: Corpus | None = None, output_dir: Path | None = None,
        publish_provenance: bool | None = None) -> dict:
    """Execute the audit end to end and return everything it computed."""
    corpus = corpus or load()
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _refuse_synthetic_overwrite(corpus, output_dir)

    # P2: identifier sets and category maps.
    indexes = index.build(corpus)
    coverage = index.category_coverage(indexes)
    integrity = index.integrity(corpus)

    # P3: parity, both ways.
    identifier_parity = parity.identifier_parity(indexes)
    composite_parity = parity.composite_parity(corpus.frames)
    duplication = parity.internal_duplication(corpus.frames)

    # P4: reconciliation.
    per_group, per_column = reconcile.apply_rules(corpus)
    divergence = reconcile.divergence_summary(per_group)
    temporal = reconcile.temporal_summary(per_group)

    # P5: mechanism attribution, and a description of the field it implicates.
    attribution, prediction_matrix = attribute.attribute(corpus, per_group, per_column)
    gauge_profile = attribute.gauge_profile(corpus)
    # The trace of the group whose divergence is largest, which is the one the
    # mechanism has to be legible on.
    if not gauge_profile.empty:
        worst = int(gauge_profile.loc[gauge_profile["summed"].idxmax(),
                                      "group_number"])
        gauge_trace = attribute.gauge_trace(corpus, worst)
    else:
        gauge_trace = pd.DataFrame()
    # A timestamp column is attributed to a store, and a store holds more than
    # one kind of activity, so the category of the record it was stamped from is
    # what decides whose activity the column is reporting.
    temporal_trace = attribute.temporal_trace(corpus, per_group)

    # P6: completeness.
    completeness = audit.completeness(corpus)
    drift = audit.name_drift(corpus)
    unnamed = audit.unnamed_by_category(corpus)
    redundancy = audit.archive_redundancy(corpus)

    # P7: downstream consequence.
    consequence = simulate.simulate(corpus, per_group, per_column)

    # P8: assumption sweep.
    sweeps = sweep.run(corpus, per_column)

    results = {
        "corpus": {"source": corpus.source,
                   "n_groups": int(len(corpus["view"])),
                   "ruleset_version": RULESET_VERSION,
                   "counts": corpus.counts},
        "census": coverage,
        "integrity": integrity,
        "parity_identifiers": identifier_parity,
        "parity_composite": composite_parity,
        "internal_duplication": duplication,
        "reconciliation_by_indicator": per_column,
        "reconciliation_by_group": per_group,
        "divergence": divergence,
        "temporal_divergence": temporal,
        "attribution": attribution,
        "prediction_matrix": prediction_matrix,
        "gauge_profile": gauge_profile,
        "gauge_trace": gauge_trace,
        "temporal_trace": temporal_trace,
        "completeness": completeness,
        "name_drift": drift,
        "unnamed_by_category": unnamed,
        "archive_redundancy": redundancy,
        "consequence": consequence,
        "sweep": sweeps["table"],
        "assumption_dependence": sweeps["dependence"],
        "aggregation_order": sweeps["aggregation_order"],
        "archive_collections": (corpus.arrays or {}).get("collections", {}),
    }
    results["gates"] = gates(corpus, results)

    if publish_provenance is None:
        publish_provenance = corpus.source == "restricted"
    if publish_provenance:
        manifest = provenance.build_manifest(
            {"source": corpus.source, "column_map": corpus.column_map,
             "tables": corpus.digests, "counts": corpus.counts},
            notes="Indicator-level reconciliation of a reporting view against "
                  "its source event stores.")
        provenance.write_manifest(manifest)

    (output_dir / "results.json").write_text(
        json.dumps(_jsonable(results), indent=1, sort_keys=True, default=str),
        encoding="utf-8")
    return results


def _result_digest(results: dict) -> str:
    """A digest over everything reported, for the determinism check."""
    trimmed = {k: v for k, v in results.items() if k != "gates"}
    blob = json.dumps(_jsonable(trimmed), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def gates(corpus: Corpus, results: dict) -> dict:
    """G1 to G7, each answered from the run rather than from recollection."""
    view_columns = set(corpus["view"].columns)
    classified = set(results["reconciliation_by_indicator"]["indicator"])
    unexamined = sorted(view_columns - classified)

    attribution = results["attribution"]
    if attribution.empty:
        g3 = {"pass": True, "n_divergent_groups": 0,
              "note": "no divergence arose, so none required attribution"}
    else:
        divergent_rows = attribution[attribution["mechanism"] != "no divergence"]
        labelled = divergent_rows["mechanism"].notna() & \
            (divergent_rows["mechanism"].astype(str) != "")
        g3 = {"pass": bool(labelled.all()),
              "n_divergent_groups": int(len(divergent_rows)),
              "n_unattributed": int((divergent_rows["mechanism"]
                                     == attribute.UNATTRIBUTED).sum()),
              "note": "unattributed is a permitted outcome; unlabelled is not"}

    dependence = results["assumption_dependence"]
    sweep_table = results["sweep"]
    g4 = {"pass": not sweep_table.empty and not dependence.empty,
          "n_settings_run": int(len(sweep_table)),
          "n_assumption_dependent": int(dependence["assumption_dependent"].sum())
          if not dependence.empty else 0}

    consequence = results["consequence"]
    by_group = results["reconciliation_by_group"]
    divergent_numeric = set(
        results["reconciliation_by_indicator"]
        .loc[results["reconciliation_by_indicator"]["classification"] == "divergent",
             "indicator"]) & set(by_group.loc[by_group["numeric"].astype(bool),
                                              "indicator"])
    g5 = {"pass": set(consequence) >= divergent_numeric,
          "n_simulated": len(consequence),
          "n_expected": len(divergent_numeric)}

    ordering = provenance.rules_committed_before_code()
    g2 = {"pass": bool(ordering.get("rules_precede_applier")),
          **ordering, "ruleset_version": RULESET_VERSION,
          "n_rules": len(RULES)}

    released = sorted(p.name for p in provenance.PROVENANCE_DIR.glob("*")) \
        if provenance.PROVENANCE_DIR.is_dir() else []
    permitted = {"column_map.json", "record_digests.json", "run_manifest.json",
                 "counting_rules.json"}
    g6 = {"pass": set(released) <= permitted,
          "released": released,
          "unexpected": sorted(set(released) - permitted),
          "note": "no record, estimate, table or figure may appear here"}
    if corpus.source != "restricted":
        # A synthetic run writes no provenance, so anything in that directory is
        # left over from a restricted run. Reporting it as this run's release
        # would credit the synthetic run with a package it did not produce.
        g6 = {"pass": True, "released": [],
              "unexpected": [],
              "note": f"a {corpus.source} run publishes no provenance; any "
                      f"files in the directory belong to an earlier run"}

    return {
        "G1_every_column_classified": {
            "pass": not unexamined, "n_columns": len(view_columns),
            "n_classified": len(classified & view_columns),
            "unexamined": unexamined},
        "G2_rules_committed_before_recomputation": g2,
        "G3_every_divergence_attributed_or_declared_unattributed": g3,
        "G4_assumption_sweep_complete": g4,
        "G5_impact_quantified": g5,
        "G6_release_package_is_provenance_only": g6,
        "G7_clean_re_execution_reproduces": {
            "pass": None,
            "note": "answered by scripts/check_determinism.py, which re-executes "
                    "from the fixed snapshot in a separate process and compares "
                    "the result digest",
            "result_digest": _result_digest(results)},
    }


def gate_report(results: dict) -> str:
    lines = ["Completion gate", "=" * 60]
    every = True
    for name, state in results["gates"].items():
        verdict = state.get("pass")
        mark = "pass" if verdict else ("deferred" if verdict is None else "FAIL")
        if verdict is False:
            every = False
        lines.append(f"  {mark:<9} {name}")
        for key, value in state.items():
            if key == "pass":
                continue
            lines.append(f"            {key}: {value}")
    lines.append("=" * 60)
    lines.append("drafting may begin" if every
                 else "drafting is blocked until the failures above are cleared")
    return "\n".join(lines)
