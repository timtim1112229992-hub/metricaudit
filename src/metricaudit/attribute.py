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
    """A running counter is summed instead of being read once.

    The client maintains a counter and reports its current value on every
    message it sends. Summing that field across messages adds the gauge to
    itself: each increment is counted once for every message that follows it
    while the counter stands. The total therefore grows with the product of the
    increments and the messages carrying them, not with the increments alone,
    which is what separates a counter wrong by a bounded factor from one wrong
    without bound.

    The counter in this corpus resets periodically rather than accumulating
    across the session, so the re-counting is bounded by the length of a run
    rather than by the length of the session. That makes the inflation smaller
    than an accumulating counter would produce and much harder to recognise,
    since neither the reported total nor the stored field carries any trace of
    where the resets fell.
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
    """The running counter read once, at its highest observed value.

    What the reporting column would hold if the gauge were used as a gauge
    rather than as an increment. Kept in the candidate set because it is the
    disciplined use of the same field, and its distance from the truth is
    therefore the more interesting measurement: because the counter resets, its
    highest value is the highest within one run and not the session total, so
    reading it correctly still does not recover the quantity the column names.
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
    Candidate("gauge_summation", "a running counter summed across every message "
              "that carried it, rather than read once", _gauge_summation),
    Candidate("gauge_final_value", "the running counter read once at its highest "
              "observed value, which is the disciplined use of the field",
              _gauge_final),
    Candidate("decision_count", "agent turns counted in place of pupil actions",
              _decision_count),
)


# Tables whose timestamps could plausibly stand behind a "last active" column.
# Registered so that a temporal divergence is tested against alternative sources
# rather than reported as a mystery.
TEMPORAL_SOURCES = {
    "operational_store_max": "operational",
    "reporting_store_max": "reporting",
    "agent_decision_max": "decisions",
    "learner_question_max": "qa",
}


def _latest(frame: pd.DataFrame, index: pd.Index) -> pd.Series:
    stamps = pd.to_datetime(frame["created_at"], format="ISO8601", utc=True)
    return stamps.groupby(frame["group_id"]).max().reindex(index)


def temporal_predictions(corpus: Corpus, index: pd.Index) -> pd.DataFrame:
    """What each candidate source would put in a "last active" column.

    A column of this kind can be stamped from any table the platform happens to
    touch last, and which one it is decides whether the column means "the group
    last did something" or "the agent last did something to the group". Those
    are different claims about a class of children and the view does not say
    which it is making.
    """
    out = {}
    for name, table in TEMPORAL_SOURCES.items():
        if table in corpus:
            out[name] = _latest(corpus[table], index)
    if out:
        combined = pd.concat(out.values(), axis=1).max(axis=1)
        out["any_table_max"] = combined
    return pd.DataFrame(out, index=index)


# The field naming what kind of record each store holds. A timestamp column is
# attributed to a store, and a store is not a single kind of activity, so the
# store alone does not settle whose activity the column is timing.
TEMPORAL_CATEGORY = {
    "operational": "event_type",
    "reporting": "event_type",
    "decisions": "action",
}


def temporal_trace(corpus: Corpus, per_group: pd.DataFrame,
                   indicator: str = "last_active_at") -> pd.DataFrame:
    """For each group, what the latest record in each candidate store was.

    Attribution names the store a reported timestamp came from. That is not yet
    a statement about meaning, because a store holding both a pupil's action and
    the agent's response to it will answer "last active" with whichever came
    last. The category of the latest record is what turns the attribution into a
    claim, and it is carried here so the claim can be checked rather than taken
    on the store's name.

    Holds a group number, a store name, an offset in seconds and a category
    label. No identifier, no timestamp and no text.
    """
    if indicator not in set(per_group.get("indicator", [])):
        return pd.DataFrame()
    block = per_group[per_group["indicator"] == indicator]
    view = corpus["view"].set_index("group_id")
    reported = pd.to_datetime(view[indicator], format="ISO8601", utc=True)

    rows = []
    for group_id in block["group_id"]:
        when = reported.get(group_id)
        if pd.isna(when):
            continue
        for store, field in TEMPORAL_CATEGORY.items():
            if store not in corpus:
                continue
            frame = corpus[store]
            part = frame[frame["group_id"] == group_id]
            if part.empty:
                continue
            stamps = pd.to_datetime(part["created_at"], format="ISO8601", utc=True)
            position = stamps.idxmax()
            rows.append({
                "indicator": indicator,
                "group_number": int(view.loc[group_id, "group_number"]),
                "store": store,
                "offset_s": float((when - stamps.loc[position]).total_seconds()),
                "latest_record_category": (str(part.loc[position, field])
                                           if field in part.columns else ""),
            })
    return pd.DataFrame(rows).sort_values(["group_number", "store"]) \
        .reset_index(drop=True)


def gauge_profile(corpus: Corpus, indicator: str = "help_click_count") -> pd.DataFrame:
    """Describe the counter the attributed mechanism sums.

    Attribution establishes that summing the field reproduces the reported
    column. It does not establish what the field is, and the two are different
    claims. A counter that accumulates across a session and one that resets
    periodically both reproduce an inflated total when summed, but they imply
    different magnitudes, different diagnostics and different remedies, so the
    description has to be measured rather than assumed.

    A run is a maximal stretch of a group's snapshots, in time order, over which
    the counter does not fall. Reporting the number of runs alongside the
    highest value reached and the value left at the end is what distinguishes
    the two shapes: an accumulating counter has one run per group and ends at
    its maximum, and this one does neither.
    """
    gauge = GAUGE_FOR_INDICATOR.get(indicator)
    if gauge is None or "decisions" not in corpus:
        return pd.DataFrame()
    dec = corpus["decisions"]
    column = gauge["snapshot"]
    if column not in dec.columns:
        return pd.DataFrame()

    view = corpus["view"].set_index("group_id")
    numbers = view["group_number"]
    stamps = pd.to_datetime(dec["created_at"], format="ISO8601", utc=True)

    rows = []
    for group_id, block in dec.assign(_t=stamps).groupby("group_id"):
        block = block.sort_values("_t")
        values = block[column].astype(float).to_numpy()
        if not len(values):
            continue
        breaks = [0] + [i for i in range(1, len(values)) if values[i] < values[i - 1]]
        breaks.append(len(values))
        runs = [(breaks[k], breaks[k + 1]) for k in range(len(breaks) - 1)]
        stages = block["stage"].to_numpy() if "stage" in block.columns else None
        at_stage_change = 0 if stages is None else sum(
            1 for i in breaks[1:-1] if stages[i] != stages[i - 1])

        rows.append({
            "indicator": indicator,
            "group_id": group_id,
            "group_number": int(numbers.get(group_id, -1)),
            "n_snapshots": len(values),
            "n_runs": len(runs),
            "n_resets": len(runs) - 1,
            "resets_at_a_stage_change": at_stage_change,
            "longest_run": max(e - s for s, e in runs),
            "highest_value": float(values.max()),
            "final_value": float(values[-1]),
            "summed": float(values.sum()),
            "accumulates_across_the_session": len(runs) == 1,
        })
    return pd.DataFrame(rows).sort_values("group_number").reset_index(drop=True)


def gauge_trace(corpus: Corpus, group_number: int,
                indicator: str = "help_click_count") -> pd.DataFrame:
    """The counter's ordered values for one group, with its stage alongside.

    The profile counts the runs. This carries the sequence they were counted
    from, so that a reader can see the resets rather than take the count on
    trust. It holds the counter, the stage index and the position in the
    sequence, and no timestamp, identifier or text.
    """
    gauge = GAUGE_FOR_INDICATOR.get(indicator)
    if gauge is None or "decisions" not in corpus:
        return pd.DataFrame()
    dec = corpus["decisions"]
    column = gauge["snapshot"]
    if column not in dec.columns:
        return pd.DataFrame()

    view = corpus["view"].set_index("group_id")
    matches = view.index[view["group_number"] == group_number]
    if not len(matches):
        return pd.DataFrame()

    block = dec[dec["group_id"] == matches[0]].copy()
    block["_t"] = pd.to_datetime(block["created_at"], format="ISO8601", utc=True)
    block = block.sort_values("_t").reset_index(drop=True)
    values = block[column].astype(float)
    return pd.DataFrame({
        "group_number": group_number,
        "position": range(1, len(block) + 1),
        "stage": block["stage"] if "stage" in block.columns else pd.NA,
        "counter": values,
        "reset_here": [False] + [bool(values[i] < values[i - 1])
                                 for i in range(1, len(values))],
    })


def predictions(corpus: Corpus, indicator: str, index: pd.Index) -> pd.DataFrame:
    """Every candidate's prediction for one indicator, per group."""
    out = {}
    for candidate in CANDIDATES:
        predicted = candidate.predict(corpus, indicator, index)
        if predicted is not None:
            out[candidate.name] = predicted
    return pd.DataFrame(out, index=index)


def _attribute_temporal(corpus: Corpus, indicator: str, block: pd.DataFrame,
                        numbers: pd.Series) -> list[dict]:
    """Match a reported timestamp to the table that most likely produced it.

    Attribution here is within a declared tolerance rather than exact, because
    two writes of one event land microseconds apart and calling that a
    divergence would describe the clock rather than the pipeline. The tolerance
    is stated in the settings and is justified by the write lag the parity
    analysis measures, so it is an empirical quantity and not a concession.
    """
    index = block.index
    matrix = temporal_predictions(corpus, index)
    reported = pd.to_datetime(block["reported"], format="ISO8601", utc=True)
    tolerance = SETTINGS.temporal_tolerance_s

    rows = []
    for group_id in index:
        if bool(block.loc[group_id, "matches"]):
            rows.append({"indicator": indicator, "group_id": group_id,
                         "group_number": int(numbers.loc[group_id]),
                         "reported": np.nan, "mechanism": "no divergence",
                         "predicted": np.nan, "residual": 0.0,
                         "reason": "the declared rule reproduced the reported value"})
            continue

        offsets = (matrix.loc[group_id] - reported.loc[group_id]) \
            .map(lambda d: abs(d.total_seconds()) if pd.notna(d) else np.nan)
        offsets = offsets.dropna()
        if offsets.empty:
            rows.append({"indicator": indicator, "group_id": group_id,
                         "group_number": int(numbers.loc[group_id]),
                         "reported": np.nan, "mechanism": UNATTRIBUTED,
                         "predicted": np.nan, "residual": np.nan,
                         "reason": "no candidate source carries a timestamp for "
                                   "this group"})
            continue

        within = offsets[offsets <= tolerance]
        nearest = offsets.idxmin()
        if len(within) == 1:
            mechanism, reason = within.index[0], ""
        elif len(within) > 1:
            mechanism = " or ".join(sorted(within.index))
            reason = "candidate sources are not separable within the write lag"
        else:
            mechanism = UNATTRIBUTED
            reason = (f"no candidate source is within {tolerance:g} s; nearest "
                      f"was {nearest} at {offsets[nearest]:.3f} s")

        rows.append({"indicator": indicator, "group_id": group_id,
                     "group_number": int(numbers.loc[group_id]),
                     "reported": np.nan, "mechanism": mechanism,
                     "predicted": np.nan, "residual": float(offsets[nearest]),
                     "reason": reason})
    return rows


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
            if bool(block.get("temporal", pd.Series([False])).iloc[0]):
                rows.extend(_attribute_temporal(corpus, indicator, block, numbers))
            else:
                for group_id in index:
                    if bool(block.loc[group_id, "matches"]):
                        continue
                    rows.append({"indicator": indicator, "group_id": group_id,
                                 "group_number": int(numbers.loc[group_id]),
                                 "reported": np.nan, "mechanism": UNATTRIBUTED,
                                 "predicted": np.nan, "residual": np.nan,
                                 "reason": "the column is neither numeric nor a "
                                           "timestamp, so no registered "
                                           "mechanism applies to it"})
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
