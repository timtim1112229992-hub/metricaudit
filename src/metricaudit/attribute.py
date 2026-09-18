"""P5, M6 and M7: what produced the divergence, stated as predictions.

A mechanism named after the fact is a label. A mechanism registered as a
function from source events to a predicted value is a claim that can fail, and
the difference is the whole of the method here.

Each candidate below predicts what the reporting view would hold if that
mechanism, and no other, were operating. The observation is then compared with
every prediction and assigned to the one that reproduces it exactly. Where no
candidate reproduces it, the group is reported unattributed. Tolerance is zero:
a mechanism that predicts an observation to within a few units has been asserted
rather than demonstrated, and the residual is reported so a reader can see how
far the nearest candidate fell short.

The candidate set is deliberately wider than the mechanisms expected to fire.
Registering only the mechanism one expects to find turns attribution into
confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .config import SETTINGS
from .ingest import Corpus

UNATTRIBUTED = "unattributed"

# The source event category each counted indicator claims to count, which is
# what the store-based candidates need in order to form a prediction.
EVENT_FOR_INDICATOR = {
    "help_click_count": "agent.help_click",
    "quiz_wrong_count": "quiz.wrong_answer",
    "interaction_count": None,
}

# Indicators with a running client-side counter behind them, and the field that
# carries it in the event payload and in the agent's own snapshot.
GAUGE_FOR_INDICATOR = {
    "help_click_count": {"payload": "payload.help_count", "snapshot": "snapshot.helpClicks"},
}


@dataclass(frozen=True)
class Candidate:
    """One mechanism, expressed as a prediction it cannot escape."""

    name: str
    description: str
    predict: Callable[[Corpus, str, pd.Index], pd.Series | None]


def _counts(frame: pd.DataFrame, category: str | None, index: pd.Index) -> pd.Series:
    sub = frame if category is None else frame[frame["event_type"] == category]
    return sub.groupby("group_id").size().reindex(index).fillna(0).astype(float)


def _single_store(corpus: Corpus, indicator: str, index: pd.Index) -> pd.Series | None:
    if indicator not in EVENT_FOR_INDICATOR:
        return None
    return _counts(corpus["operational"], EVENT_FOR_INDICATOR[indicator], index)


def _cross_store_sum(corpus: Corpus, indicator: str, index: pd.Index) -> pd.Series | None:
    """Both stores receive the client event and the aggregate sums across both.

    The mechanism the architecture most obviously invites, and the reason the
    audit was specified before the data were seen.
    """
    if indicator not in EVENT_FOR_INDICATOR:
        return None
    category = EVENT_FOR_INDICATOR[indicator]
    return (_counts(corpus["operational"], category, index)
            + _counts(corpus["reporting"], category, index))


def _repeated_client_emission(corpus: Corpus, indicator: str,
                              index: pd.Index) -> pd.Series | None:
    """The client sent the same event more than once and nothing collapsed it.

    Predicts the as-stored count in the reporting store, which is where a
    redelivered event lands without being reconciled against what is already
    there.
    """
    if indicator not in EVENT_FOR_INDICATOR:
        return None
    return _counts(corpus["reporting"], EVENT_FOR_INDICATOR[indicator], index)


def _absent_idempotency(corpus: Corpus, indicator: str,
                        index: pd.Index) -> pd.Series | None:
    """No key by which a repeat could be recognised, so every arrival counts.

    Predicts every record either store holds for the category, duplicates
    included, which is the upper bound on what a counter with no idempotency key
    can report from the stored events alone.
    """
    if indicator not in EVENT_FOR_INDICATOR:
        return None
    category = EVENT_FOR_INDICATOR[indicator]
    ops = corpus["operational"]
    rep = corpus["reporting"]
    ops_n = _counts(ops, category, index)
    rep_n = _counts(rep, category, index)
    return ops_n + rep_n


def _gauge_summation(corpus: Corpus, indicator: str, index: pd.Index) -> pd.Series | None:
    """A cumulative counter is summed instead of being read once.

    The client maintains a running total and reports its current value on every
    message. An aggregate that sums that field across messages is adding a gauge
    to itself, and the result grows with the square of the activity rather than
    with the activity. This is the candidate that distinguishes a counter that
    is wrong by a factor of two from one that is wrong without bound.
    """
    gauge = GAUGE_FOR_INDICATOR.get(indicator)
    if gauge is None or "decisions" not in corpus:
        return None
    dec = corpus["decisions"]
    column = gauge["snapshot"]
    if column not in dec.columns:
        return None
    return dec.groupby("group_id")[column].sum().reindex(index).fillna(0).astype(float)


def _gauge_final(corpus: Corpus, indicator: str, index: pd.Index) -> pd.Series | None:
    """The cumulative counter read once, at its final value.

    What the reporting column would hold if the gauge were used as a gauge. Kept
    in the candidate set because it is the correct use of the same field, so its
    distance from the observation measures the cost of the error rather than
    merely its presence.
    """
    gauge = GAUGE_FOR_INDICATOR.get(indicator)
    if gauge is None:
        return None
    ops = corpus["operational"]
    column = gauge["payload"]
    if column not in ops.columns:
        return None
    category = EVENT_FOR_INDICATOR.get(indicator)
    sub = ops if category is None else ops[ops["event_type"] == category]
    return sub.groupby("group_id")[column].max().reindex(index).fillna(0).astype(float)


def _decision_count(corpus: Corpus, indicator: str, index: pd.Index) -> pd.Series | None:
    """The counter records agent turns rather than pupil actions.

    A plausible confusion whenever an agent both responds to an event and logs
    one, and worth excluding explicitly rather than by assumption.
    """
    if "decisions" not in corpus:
        return None
    return corpus["decisions"].groupby("group_id").size() \
        .reindex(index).fillna(0).astype(float)


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("single_store", "counted once from the operational store", _single_store),
    Candidate("cross_store_summation", "summed across two stores that both "
              "receive the same client event", _cross_store_sum),
    Candidate("repeated_client_emission", "the client emitted the event more "
              "than once and nothing collapsed the repeats", _repeated_client_emission),
    Candidate("absent_idempotency", "every arrival counted, there being no key "
              "by which a repeat could be recognised", _absent_idempotency),
    Candidate("gauge_summation", "a running cumulative counter summed across "
              "every message that carried it, rather than read once",
              _gauge_summation),
    Candidate("gauge_final_value", "the running counter read once at its final "
              "value, which is its correct use", _gauge_final),
    Candidate("decision_count", "agent turns counted in place of pupil actions",
              _decision_count),
)


def predictions(corpus: Corpus, indicator: str, index: pd.Index) -> pd.DataFrame:
    """Every candidate's prediction for one indicator, per group."""
    out = {}
    for candidate in CANDIDATES:
        predicted = candidate.predict(corpus, indicator, index)
        if predicted is not None:
            out[candidate.name] = predicted
    return pd.DataFrame(out, index=index)


def attribute(corpus: Corpus, per_group: pd.DataFrame,
              per_column: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """M6 and M7 over every divergent indicator.

    Returns the per-group attribution and the matrix of candidate predictions,
    the latter so that a reader can see what each mechanism would have implied
    rather than only which one won.
    """
    divergent = per_column.loc[per_column["classification"] == "divergent", "indicator"]
    rows: list[dict] = []
    matrices: list[pd.DataFrame] = []

    view = corpus["view"].set_index("group_id")
    numbers = view["group_number"]

    for indicator in divergent:
        block = per_group[per_group["indicator"] == indicator].set_index("group_id")
        if "reported" not in block.columns or block.empty:
            continue
        index = block.index

        # A divergence in a timestamp or a label is real and is reported, but
        # the candidate mechanisms are all arithmetic and have nothing to say
        # about it. Saying so beats coercing the column into a number. The test
        # reads the per-row flag rather than the column's dtype, because the
        # frame stacks indicators of several types and the stacked dtype is
        # object whatever any single indicator holds.
        if not bool(block["numeric"].iloc[0]):
            for group_id in index:
                if bool(block.loc[group_id, "matches"]):
                    continue
                rows.append({"indicator": indicator, "group_id": group_id,
                             "group_number": int(numbers.loc[group_id]),
                             "reported": np.nan, "mechanism": UNATTRIBUTED,
                             "predicted": np.nan, "residual": np.nan,
                             "reason": "the column is not numeric, so no "
                                       "arithmetic mechanism applies to it"})
            continue

        matrix = predictions(corpus, indicator, index)
        if matrix.empty:
            for group_id in index:
                rows.append({"indicator": indicator, "group_id": group_id,
                             "group_number": int(numbers.loc[group_id]),
                             "reported": block.loc[group_id, "reported"],
                             "mechanism": UNATTRIBUTED, "predicted": np.nan,
                             "residual": np.nan,
                             "reason": "no candidate mechanism is applicable"})
            continue

        tidy = matrix.copy()
        tidy.insert(0, "reported", block["reported"].astype(float))
        tidy.insert(0, "indicator", indicator)
        tidy.insert(1, "group_number", numbers.reindex(index).astype(int))
        matrices.append(tidy.reset_index())

        for group_id in index:
            observed = float(block.loc[group_id, "reported"])
            matched = block.loc[group_id, "matches"]
            preds = matrix.loc[group_id]
            residuals = (preds - observed).abs()
            nearest = residuals.idxmin()
            exact = residuals[residuals <= SETTINGS.mechanism_tolerance]

            if bool(matched):
                mechanism, predicted, residual, reason = (
                    "no divergence", float(preds.get("single_store", np.nan)), 0.0,
                    "the declared rule reproduced the reported value")
            elif len(exact) == 1:
                mechanism = exact.index[0]
                predicted, residual, reason = float(preds[mechanism]), 0.0, ""
            elif len(exact) > 1:
                # Two candidates predicting the same value are not two
                # explanations, they are one prediction the design cannot
                # separate. Saying so is more use than picking the first.
                mechanism = " or ".join(sorted(exact.index))
                predicted, residual = float(preds[exact.index[0]]), 0.0
                reason = "candidates are not separable on this observation"
            else:
                mechanism, predicted = UNATTRIBUTED, float(preds[nearest])
                residual = float(residuals[nearest])
                reason = (f"no candidate reproduced the value; nearest was "
                          f"{nearest} at a residual of {residual:.0f}")

            rows.append({"indicator": indicator, "group_id": group_id,
                         "group_number": int(numbers.loc[group_id]),
                         "reported": observed, "mechanism": mechanism,
                         "predicted": predicted, "residual": residual,
                         "reason": reason})

    attribution = pd.DataFrame(rows)
    if not attribution.empty:
        attribution = attribution.sort_values(["indicator", "group_number"]) \
            .reset_index(drop=True)
    matrix = pd.concat(matrices, ignore_index=True) if matrices else pd.DataFrame()
    return attribution, matrix
