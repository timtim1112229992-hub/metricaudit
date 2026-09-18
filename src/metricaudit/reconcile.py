"""P4 and M2: apply each declared rule and classify every reported column.

The comparison is exact. A tolerance would be a kindness to the platform and a
disservice to the reader, since the faults worth finding include rounding, and a
tolerance wide enough to absorb a rounding fault is wide enough to hide one.

Three outcomes are possible and all three are reported. Reconciled means the
recomputation matched for every group. Divergent means it failed for at least
one. Unverifiable means no path from source events to the column exists, which
is a property of the pipeline rather than a failure of the audit, and is
recorded as such.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SETTINGS
from .ingest import Corpus
from .rules import RULES, CountingRule

RECONCILED = "reconciled"
DIVERGENT = "divergent"
UNVERIFIABLE = "unverifiable"


def _align(rule: CountingRule, recomputed: pd.Series, index: pd.Index) -> pd.Series:
    """Put a recomputation on the view's index, filling absences honestly.

    A group with no help clicks produces no row in a grouped count, and the
    absence means zero. A group missing from a derived timestamp means no
    activity was recorded, which is not zero and is left missing.
    """
    aligned = recomputed.reindex(index)
    if rule.kind == "count":
        aligned = aligned.fillna(0)
    return aligned


def _both_numeric(reported: pd.Series, recomputed: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(reported)
                and pd.api.types.is_numeric_dtype(recomputed))


def _equal(reported: pd.Series, recomputed: pd.Series, kind: str) -> pd.Series:
    # Timezone-aware timestamps are a pandas extension dtype rather than a numpy
    # one, so the check has to go through pandas or it raises on the very column
    # it was written for.
    if pd.api.types.is_datetime64_any_dtype(recomputed):
        left = pd.to_datetime(reported, format="ISO8601", utc=True)
        return left.eq(recomputed)
    if _both_numeric(reported, recomputed):
        return (reported.astype(float) - recomputed.astype(float)).abs() \
            <= SETTINGS.tolerance
    return reported.astype(str).eq(recomputed.astype(str))


def apply_rules(corpus: Corpus) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute every column and return the per-group and per-column results."""
    view = corpus["view"].set_index("group_id")
    per_group: list[dict] = []
    per_column: list[dict] = []

    for rule in RULES:
        if rule.column not in corpus["view"].columns:
            per_column.append({
                "indicator": rule.column, "kind": rule.kind,
                "classification": UNVERIFIABLE, "n_groups": 0, "n_matching": 0,
                "n_diverging": 0, "reason": "column absent from the reporting view",
                "ruleset_source": rule.source})
            continue

        if not rule.verifiable:
            per_column.append({
                "indicator": rule.column, "kind": rule.kind,
                "classification": UNVERIFIABLE,
                "n_groups": int(len(view)), "n_matching": 0, "n_diverging": 0,
                "reason": "no declared source path exists for this column",
                "ruleset_source": rule.source})
            continue

        reported = view[rule.column]
        recomputed = _align(rule, rule.recompute(corpus.frames), view.index)
        matches = _equal(reported, recomputed, rule.kind)
        numeric = _both_numeric(reported, recomputed)
        temporal = pd.api.types.is_datetime64_any_dtype(recomputed)
        reported_time = (pd.to_datetime(reported, format="ISO8601", utc=True)
                         if temporal else None)

        for group_id in view.index:
            rep = reported.loc[group_id]
            rec = recomputed.loc[group_id]
            row = {
                "indicator": rule.column,
                "kind": rule.kind,
                "group_id": group_id,
                "group_number": int(view.loc[group_id, "group_number"]),
                "reported": rep,
                "recomputed": rec,
                "matches": bool(matches.loc[group_id]),
                # Carried per row because the per-group frame stacks indicators
                # of different types, so the stacked column's dtype says nothing
                # about any single indicator in it.
                "numeric": bool(numeric),
                "temporal": bool(temporal),
            }
            if numeric:
                rep_f, rec_f = float(rep), float(rec)
                row["absolute_difference"] = rep_f - rec_f
                row["ratio"] = rep_f / rec_f if rec_f else np.nan
                row["direction"] = ("equal" if rep_f == rec_f
                                    else "reported higher" if rep_f > rec_f
                                    else "reported lower")
            elif temporal:
                # A timestamp that disagrees by a fraction of a second and one
                # that disagrees by twenty are both "divergent" under an exact
                # test, and reporting only the verdict would make them look
                # alike. The offset is what tells them apart.
                offset = (reported_time.loc[group_id] - rec).total_seconds()
                row["offset_s"] = float(offset)
                row["direction"] = ("equal" if offset == 0
                                    else "reported later" if offset > 0
                                    else "reported earlier")
            per_group.append(row)

        n_match = int(matches.sum())
        per_column.append({
            "indicator": rule.column,
            "kind": rule.kind,
            "classification": RECONCILED if n_match == len(view) else DIVERGENT,
            "n_groups": int(len(view)),
            "n_matching": n_match,
            "n_diverging": int(len(view) - n_match),
            "reason": "",
            "ruleset_source": rule.source,
        })

    groups = pd.DataFrame(per_group)
    columns = pd.DataFrame(per_column)
    order = {"count": 0, "aggregate": 1, "derived": 2, "key": 3}
    columns = columns.sort_values(
        ["kind", "indicator"], key=lambda s: s.map(order).fillna(9)
        if s.name == "kind" else s).reset_index(drop=True)
    return groups, columns


def divergence_summary(per_group: pd.DataFrame) -> pd.DataFrame:
    """Per-indicator distribution of the divergence, with the extreme separated.

    M12 is enforced here rather than left to the writing. The pooled figures
    exclude the single largest divergence, because a counter that accumulates
    without bound sets any mean it enters and the mean then describes the
    counter rather than the session.
    """
    rows = []
    numeric = per_group[per_group["numeric"].astype(bool)]
    for indicator, block in numeric.groupby("indicator"):
        diverging = block[~block["matches"]]
        if diverging.empty:
            rows.append({"indicator": indicator, "n_diverging": 0,
                         "extreme_group": None, "extreme_ratio": np.nan,
                         "pooled_max_ratio": np.nan, "pooled_median_ratio": np.nan,
                         "pooled_max_absolute": np.nan, "all_upward": None})
            continue
        extreme = diverging.loc[diverging["absolute_difference"].abs().idxmax()]
        pooled = diverging.drop(index=extreme.name)
        rows.append({
            "indicator": indicator,
            "n_diverging": int(len(diverging)),
            "extreme_group": int(extreme["group_number"]),
            "extreme_ratio": float(extreme["ratio"]),
            "extreme_absolute": float(extreme["absolute_difference"]),
            "pooled_max_ratio": float(pooled["ratio"].max()) if len(pooled) else np.nan,
            "pooled_median_ratio": float(pooled["ratio"].median()) if len(pooled) else np.nan,
            "pooled_max_absolute": float(pooled["absolute_difference"].max())
            if len(pooled) else np.nan,
            "all_upward": bool((diverging["absolute_difference"] > 0).all()),
        })
    return pd.DataFrame(rows).sort_values("indicator").reset_index(drop=True)


def temporal_summary(per_group: pd.DataFrame) -> pd.DataFrame:
    """The size of a timestamp divergence, which the verdict alone conceals."""
    rows = []
    temporal = per_group[per_group.get("temporal", False).astype(bool)] \
        if "temporal" in per_group.columns else per_group.iloc[0:0]
    for indicator, block in temporal.groupby("indicator"):
        offsets = block["offset_s"].astype(float)
        rows.append({
            "indicator": indicator,
            "n_groups": int(len(block)),
            "n_diverging": int((~block["matches"]).sum()),
            "median_offset_s": float(offsets.median()),
            "min_offset_s": float(offsets.min()),
            "max_offset_s": float(offsets.max()),
            "n_within_one_second": int((offsets.abs() <= 1).sum()),
            "all_later": bool((offsets > 0).all()),
        })
    return pd.DataFrame(rows)
