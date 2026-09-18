"""P7, M9 and M10: what the fault costs an analyst who never noticed it.

A divergence is only interesting if it changes something. This step recomputes
the statistics an analyst would actually publish, once from the reported values
and once from the reconciled ones, and reports the pair. Central tendency,
dispersion and rank order are all included because they fail differently: a mean
moves, a standard deviation moves further, and a rank order can invert entirely
while both summaries look merely large.

The representative association is estimated twice for the same reason. A
coefficient is the form in which an indicator usually enters a published
argument, so the distance between the two estimates is the distance between a
conclusion and the conclusion that the same analysis would have reached from
correct arithmetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SETTINGS
from .ingest import Corpus
from .models import bca_interval, kendall_tau, limits_of_agreement, spearman


def descriptives(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {k: float("nan") for k in
                ("n", "mean", "sd", "median", "iqr", "min", "max", "cv")}
    sd = float(values.std(ddof=1)) if values.size > 1 else float("nan")
    mean = float(values.mean())
    return {"n": int(values.size), "mean": mean, "sd": sd,
            "median": float(np.median(values)),
            "iqr": float(np.subtract(*np.percentile(values, [75, 25]))),
            "min": float(values.min()), "max": float(values.max()),
            "cv": sd / mean if mean else float("nan")}


def impact(per_group: pd.DataFrame, indicator: str) -> dict:
    """M9: the same group-level summary under each value set.

    Computed both with and without the extreme case. Including it shows what an
    analyst would have published; excluding it shows whether the rest of the
    fault would have been noticed, and the answer to the second question is
    usually no.
    """
    block = per_group[per_group["indicator"] == indicator]
    if block.empty:
        return {}
    reported = block["reported"].astype(float).to_numpy()
    reconciled = block["recomputed"].astype(float).to_numpy()

    diverging = block[~block["matches"]]
    extreme_idx = (diverging["absolute_difference"].abs().idxmax()
                   if not diverging.empty else None)
    keep = block.index != extreme_idx if extreme_idx is not None else slice(None)

    out = {
        "indicator": indicator,
        "reported": descriptives(reported),
        "reconciled": descriptives(reconciled),
        "rank_agreement": kendall_tau(reported, reconciled),
        "extreme_group": (int(block.loc[extreme_idx, "group_number"])
                          if extreme_idx is not None else None),
    }
    out["reported_without_extreme"] = descriptives(block.loc[keep, "reported"]
                                                   .astype(float).to_numpy())
    out["reconciled_without_extreme"] = descriptives(block.loc[keep, "recomputed"]
                                                     .astype(float).to_numpy())

    rep_mean = out["reported"]["mean"]
    rec_mean = out["reconciled"]["mean"]
    out["mean_inflation_factor"] = rep_mean / rec_mean if rec_mean else float("nan")
    rep_sd, rec_sd = out["reported"]["sd"], out["reconciled"]["sd"]
    out["sd_inflation_factor"] = rep_sd / rec_sd if rec_sd else float("nan")
    return out


def agreement(per_group: pd.DataFrame, indicator: str) -> dict:
    """M5: the difference-against-mean summary, on the log scale.

    On the raw scale one unbounded counter sets the limits of agreement by
    itself. The log scale turns the differences into ratios, which is the scale
    on which a doubling and a twentyfold inflation are commensurable.
    """
    block = per_group[per_group["indicator"] == indicator]
    rep = block["reported"].astype(float).to_numpy()
    rec = block["recomputed"].astype(float).to_numpy()
    keep = (rep > 0) & (rec > 0)
    if keep.sum() < 2:
        return {"n_usable": int(keep.sum()), "estimable": False}
    log_diff = np.log10(rep[keep]) - np.log10(rec[keep])
    log_mean = (np.log10(rep[keep]) + np.log10(rec[keep])) / 2
    out = limits_of_agreement(log_diff, SETTINGS.confidence)
    out.update({"n_usable": int(keep.sum()), "estimable": True,
                "n_dropped_for_zero": int((~keep).sum()),
                "mean_log_magnitude": float(log_mean.mean())})
    return out


def divergence_interval(per_group: pd.DataFrame, indicator: str) -> dict:
    """M4: a bootstrap interval on the divergence ratio, over groups.

    Over groups rather than over records, because groups are the independent
    unit and a record-level resample would treat one group's two hundred events
    as two hundred pieces of evidence about the pipeline.
    """
    block = per_group[per_group["indicator"] == indicator]
    ratios = block["ratio"].astype(float).to_numpy()
    ratios = ratios[np.isfinite(ratios)]
    return {
        "indicator": indicator,
        "median_ratio": bca_interval(ratios, np.median,
                                     SETTINGS.bootstrap_replicates, SETTINGS.seed,
                                     SETTINGS.confidence),
        "mean_ratio": bca_interval(ratios, np.mean,
                                   SETTINGS.bootstrap_replicates, SETTINGS.seed + 1,
                                   SETTINGS.confidence),
        "n_groups": int(len(ratios)),
    }


def association(corpus: Corpus, per_group: pd.DataFrame, indicator: str,
                against: str = "avg_completion_pct") -> dict:
    """M10: one association a reader might plausibly have drawn, estimated twice.

    The pairing is chosen to be the kind of question the reporting view invites
    rather than one selected for effect: whether groups that asked for more help
    completed less of the work. It is reported as a demonstration of
    sensitivity, not as a finding about the lesson, and the code says so because
    the distinction will otherwise be lost the moment the number is quoted.
    """
    block = per_group[per_group["indicator"] == indicator].set_index("group_id")
    if block.empty:
        return {}
    view = corpus["view"].set_index("group_id")
    if against not in view.columns:
        return {}
    outcome = view[against].reindex(block.index).astype(float).to_numpy()

    return {
        "indicator": indicator,
        "outcome": against,
        "interpretation": "reported as a sensitivity demonstration, not as a "
                          "substantive claim about the lesson",
        "from_reported": spearman(block["reported"].astype(float).to_numpy(), outcome),
        "from_reconciled": spearman(block["recomputed"].astype(float).to_numpy(),
                                    outcome),
    }


def simulate(corpus: Corpus, per_group: pd.DataFrame,
             per_column: pd.DataFrame) -> dict:
    """Every divergent indicator put through the downstream consequence."""
    divergent = list(per_column.loc[per_column["classification"] == "divergent",
                                    "indicator"])
    numeric = set(per_group.loc[per_group["numeric"].astype(bool), "indicator"])
    out = {}
    for indicator in divergent:
        if indicator not in numeric:
            continue
        out[indicator] = {
            "impact": impact(per_group, indicator),
            "agreement": agreement(per_group, indicator),
            "divergence_interval": divergence_interval(per_group, indicator),
            "association": association(corpus, per_group, indicator),
        }
    return out
