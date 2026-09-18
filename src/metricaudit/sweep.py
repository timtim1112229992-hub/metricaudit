"""P8 and M11: which conclusions survive a different analyst.

Every recomputation rests on choices the platform does not document: whether a
repeat within a store is one event or two, and how a percentage was rounded. An
audit that made those choices once and reported the result would be offering its
own assumptions as findings.

So the reconciliation is re-run under each alternative, and each indicator is
labelled by whether its outcome changed. An indicator that reconciles under
every assumption is robust. One whose classification flips is assumption
dependent, and any claim about it has to be stated with the assumption attached.
"""
from __future__ import annotations

import pandas as pd

from .config import SETTINGS
from .ingest import Corpus
from .parity import _composite_key
from .rules import BY_COLUMN, round_mode


def _dedup(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    """One store under a stated view of what counts as a distinct event."""
    if mode == "as_stored":
        return frame
    working = frame.copy()
    working["_key"] = _composite_key(working)
    if mode == "distinct_composite":
        return working.drop_duplicates("_key")
    if mode == "distinct_within_window":
        # Two records sharing a composite key are one event where they also sit
        # within the write window of each other. Beyond it they are a genuine
        # repeat: a pupil who clicks for help twice in a lesson has asked twice.
        working["_t"] = pd.to_datetime(working["created_at"], format="ISO8601",
                                       utc=True)
        working = working.sort_values(["_key", "_t"])
        gap = working.groupby("_key")["_t"].diff().dt.total_seconds()
        keep = gap.isna() | (gap > SETTINGS.parity_window_s)
        return working[keep]
    raise ValueError(f"unknown deduplication mode {mode!r}")


def dedup_sweep(corpus: Corpus, per_column: pd.DataFrame) -> pd.DataFrame:
    """Recount every counted indicator under each deduplication assumption."""
    view = corpus["view"].set_index("group_id")
    counted = per_column[per_column["kind"] == "count"]["indicator"]

    rows = []
    for mode in SETTINGS.dedup_modes:
        frames = dict(corpus.frames)
        frames["operational"] = _dedup(corpus["operational"], mode)
        for indicator in counted:
            rule = BY_COLUMN.get(indicator)
            if rule is None or not rule.verifiable:
                continue
            recomputed = rule.recompute(frames).reindex(view.index).fillna(0)
            reported = view[indicator].astype(float)
            matches = (reported - recomputed.astype(float)).abs() <= SETTINGS.tolerance
            rows.append({
                "assumption": "deduplication",
                "setting": mode,
                "indicator": indicator,
                "n_groups": int(len(view)),
                "n_matching": int(matches.sum()),
                "classification": "reconciled" if matches.all() else "divergent",
                "total_recomputed": float(recomputed.sum()),
            })
    return pd.DataFrame(rows)


def rounding_sweep(corpus: Corpus, per_column: pd.DataFrame) -> pd.DataFrame:
    """Recompute every aggregate under each rounding convention.

    Included because the reported percentage column is an integer and nothing
    states how it was produced from a mean. Where more than one convention
    reproduces the column, the data cannot identify which was used, and that is
    reported rather than resolved by preference.
    """
    view = corpus["view"].set_index("group_id")
    subs = corpus["submissions"]
    aggregates = per_column[per_column["kind"] == "aggregate"]["indicator"]

    rows = []
    for indicator in aggregates:
        if indicator != "avg_completion_pct":
            continue
        exact = subs.groupby("group_id")["completion_rate"].mean() * 100
        reported = view[indicator].astype(float)
        for mode in SETTINGS.rounding_modes:
            recomputed = exact.map(lambda v, m=mode: round_mode(v, m)) \
                .reindex(view.index)
            matches = (reported - recomputed.astype(float)).abs() <= SETTINGS.tolerance
            rows.append({
                "assumption": "rounding",
                "setting": mode,
                "indicator": indicator,
                "n_groups": int(len(view)),
                "n_matching": int(matches.sum()),
                "classification": "reconciled" if matches.all() else "divergent",
                "total_recomputed": float(recomputed.sum()),
            })
    return pd.DataFrame(rows)


def aggregation_order(corpus: Corpus) -> dict:
    """What rounding per group costs a reader who then averages the column.

    Not an assumption the audit makes but one a reader of the view inevitably
    does. The column is rounded once per group; anyone computing a class figure
    from it averages twelve already-rounded values, and the result differs from
    the same figure computed from the unrounded rates. The gap is small and it
    is the kind of small that gets reported to four decimal places.
    """
    subs = corpus["submissions"]
    view = corpus["view"]
    exact = float((subs.groupby("group_id")["completion_rate"].mean() * 100).mean())
    from_column = float(view["avg_completion_pct"].astype(float).mean())
    pooled = float(subs["completion_rate"].mean() * 100)
    return {
        "mean_of_reported_column": from_column,
        "mean_of_unrounded_group_means": exact,
        "pooled_over_all_submissions": pooled,
        "difference_from_rounding": from_column - exact,
        "difference_from_unequal_group_sizes": exact - pooled,
    }


def run(corpus: Corpus, per_column: pd.DataFrame) -> dict:
    dedup = dedup_sweep(corpus, per_column)
    rounding = rounding_sweep(corpus, per_column)
    table = pd.concat([dedup, rounding], ignore_index=True)

    dependence = []
    for indicator, block in table.groupby("indicator"):
        outcomes = set(block["classification"])
        dependence.append({
            "indicator": indicator,
            "n_settings": int(len(block)),
            "classifications_seen": ", ".join(sorted(outcomes)),
            "assumption_dependent": len(outcomes) > 1,
            "settings_reconciling": ", ".join(
                sorted(block.loc[block["classification"] == "reconciled", "setting"]))
            or "none",
        })

    return {"table": table,
            "dependence": pd.DataFrame(dependence).sort_values("indicator")
            .reset_index(drop=True),
            "aggregation_order": aggregation_order(corpus)}
