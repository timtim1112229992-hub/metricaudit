"""M1: the declared counting rules.

This file exists to be committed before the recomputation that uses it. The
audit's claim is that a reported figure differs from what its own source events
support, and that claim is worth nothing if the rule defining "what the source
supports" was chosen after seeing how well it fitted. So each rule is written
down here first, in the form a reader of the reporting view would most naturally
assume, and the recomputation is then run against it without adjustment.

The emphasis on the natural reading matters. The question is not whether some
rule can be found that reproduces a reported column, since with enough freedom
one nearly always can. The question is whether the column means what its name
says. A rule is therefore derived from the column's name and the documented
semantics of the store, never from inspection of how close it lands.

Where no path from source events to a reported column exists, the rule records
that fact and the indicator is classified unverifiable rather than forced.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from typing import Callable, Mapping

import pandas as pd

# Bumped whenever any rule below changes. The manifest records it, so a reader
# comparing two runs can tell whether a difference is data or definition.
RULESET_VERSION = "2026.09.18.1"

Frames = Mapping[str, pd.DataFrame]


def round_mode(value: float, mode: str) -> int:
    """Rounding, stated explicitly because the reported column depends on it.

    Python's built-in round is half-to-even, which is a defensible convention
    and not the one most people picture. Since one of the audit's findings turns
    on a percentage column, the convention is named rather than inherited.
    """
    modes = {"half_up": ROUND_HALF_UP, "half_even": ROUND_HALF_EVEN,
             "floor": ROUND_FLOOR, "ceil": ROUND_CEILING}
    if mode not in modes:
        raise ValueError(f"unknown rounding mode {mode!r}")
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=modes[mode]))


def payload_value(blob: object, key: str) -> float | None:
    """One numeric field from a serialised payload, or nothing.

    Deliberately narrow. The payloads also hold pupil writing, and a function
    that returned the whole object would put that writing within reach of every
    caller in the package.
    """
    if not isinstance(blob, str):
        return None
    try:
        obj = json.loads(blob)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


@dataclass(frozen=True)
class CountingRule:
    """One declared recomputation of one reported column.

    Attributes:
        column: the reporting-view column this rule recomputes.
        kind: count, aggregate, derived, or key. Determines how the comparison
            is made and keeps identity columns out of the count of reconciled
            measurements, where they would flatter the result.
        statement: the rule in words, as it would be read by someone deciding
            whether the column means what they think.
        source: the table the recomputation draws on.
        dedup: the deduplication assumption in force.
        window: the inclusion window.
        missing: how records lacking a group identifier are treated.
        recompute: the rule itself, or None where no source path exists.
    """

    column: str
    kind: str
    statement: str
    source: str
    dedup: str
    window: str
    missing: str
    recompute: Callable[[Frames], pd.Series] | None

    @property
    def verifiable(self) -> bool:
        return self.recompute is not None


def _by_group(frame: pd.DataFrame, mask: pd.Series | None = None) -> pd.Series:
    sub = frame if mask is None else frame[mask]
    return sub.groupby("group_id").size()


def _ops_event(name: str) -> Callable[[Frames], pd.Series]:
    def rule(frames: Frames) -> pd.Series:
        ops = frames["operational"]
        return _by_group(ops, ops["event_type"] == name)
    return rule


def _completed_stages(frames: Frames) -> pd.Series:
    subs = frames["submissions"]
    return _by_group(subs, subs["is_completed"].astype(bool))


def _mean_completion(frames: Frames) -> pd.Series:
    subs = frames["submissions"]
    mean = subs.groupby("group_id")["completion_rate"].mean() * 100
    return mean.map(lambda v: round_mode(v, "half_up"))


def _interactions(frames: Frames) -> pd.Series:
    return _by_group(frames["operational"])


def _current_stage(frames: Frames) -> pd.Series:
    return frames["submissions"].groupby("group_id")["stage"].max()


def _last_active(frames: Frames) -> pd.Series:
    ops = frames["operational"]
    return pd.to_datetime(ops["created_at"], format="ISO8601", utc=True) \
        .groupby(ops["group_id"]).max()


def _group_attr(name: str) -> Callable[[Frames], pd.Series]:
    def rule(frames: Frames) -> pd.Series:
        groups = frames["groups"]
        return groups.set_index("id")[name]
    return rule


RULES: tuple[CountingRule, ...] = (
    CountingRule(
        column="interaction_count",
        kind="count",
        statement="The number of records the operational store holds for the "
                  "group. The column's name promises a count of that group's "
                  "interactions, and the operational store is the log of them.",
        source="operational",
        dedup="none; every stored record counts once, including any the client "
              "sent more than once, because the column claims to describe what "
              "the store holds",
        window="the whole session",
        missing="records without a group identifier belong to no group and are "
                "excluded",
        recompute=_interactions,
    ),
    CountingRule(
        column="help_click_count",
        kind="count",
        statement="The number of times the group asked the agent for help, "
                  "taken as the number of agent.help_click records in the "
                  "operational store. One click by a pupil produces one record, "
                  "so one record is one request.",
        source="operational",
        dedup="none; each stored event counts once",
        window="the whole session",
        missing="records without a group identifier are excluded",
        recompute=_ops_event("agent.help_click"),
    ),
    CountingRule(
        column="quiz_wrong_count",
        kind="count",
        statement="The number of incorrect quiz answers the group submitted, "
                  "taken as the number of quiz.wrong_answer records in the "
                  "operational store.",
        source="operational",
        dedup="none; each stored event counts once",
        window="the whole session",
        missing="records without a group identifier are excluded",
        recompute=_ops_event("quiz.wrong_answer"),
    ),
    CountingRule(
        column="completed_stage_count",
        kind="count",
        statement="The number of the lesson's stages the group completed, taken "
                  "as the number of that group's stage submissions whose "
                  "completion flag is set.",
        source="submissions",
        dedup="none; one submission row per group and stage",
        window="the whole session",
        missing="submissions carry a group identifier in every case",
        recompute=_completed_stages,
    ),
    CountingRule(
        column="avg_completion_pct",
        kind="aggregate",
        statement="The group's mean stage completion rate as a whole "
                  "percentage, taken as the mean of the completion rate across "
                  "that group's submissions, multiplied by one hundred and "
                  "rounded. Half is rounded upward, which is the convention a "
                  "reader is most likely to assume; the alternatives are tested "
                  "in the assumption sweep.",
        source="submissions",
        dedup="none",
        window="the whole session",
        missing="submissions carry a group identifier in every case",
        recompute=_mean_completion,
    ),
    CountingRule(
        column="current_stage",
        kind="derived",
        statement="The furthest stage the group reached, taken as the highest "
                  "stage index among that group's submissions.",
        source="submissions",
        dedup="none",
        window="the whole session",
        missing="submissions carry a group identifier in every case",
        recompute=_current_stage,
    ),
    CountingRule(
        column="last_active_at",
        kind="derived",
        statement="The time of the group's most recent activity, taken as the "
                  "latest creation timestamp among that group's operational "
                  "records.",
        source="operational",
        dedup="none",
        window="the whole session",
        missing="records without a group identifier are excluded",
        recompute=_last_active,
    ),
    CountingRule(
        column="group_number",
        kind="key",
        statement="The group's number, as held on the group record.",
        source="groups",
        dedup="not applicable",
        window="not applicable",
        missing="not applicable",
        recompute=_group_attr("group_number"),
    ),
    CountingRule(
        column="group_name",
        kind="key",
        statement="The group's display name, as held on the group record. The "
                  "view stores it alongside the identifier rather than joining "
                  "for it, so the two can drift apart.",
        source="groups",
        dedup="not applicable",
        window="not applicable",
        missing="not applicable",
        recompute=_group_attr("name"),
    ),
    CountingRule(
        column="session_id",
        kind="key",
        statement="The session the group belongs to, as held on the group "
                  "record.",
        source="groups",
        dedup="not applicable",
        window="not applicable",
        missing="not applicable",
        recompute=_group_attr("session_id"),
    ),
    CountingRule(
        column="group_id",
        kind="key",
        statement="The group's identifier, which is the view's own key and is "
                  "checked for membership against the group table rather than "
                  "recomputed.",
        source="groups",
        dedup="not applicable",
        window="not applicable",
        missing="not applicable",
        recompute=None,
    ),
    CountingRule(
        column="group_status",
        kind="derived",
        statement="The group's help state at the close of the session. No "
                  "source path exists: the state is written by the agent as it "
                  "runs and is not derivable from any stored event, since "
                  "nothing records when a group left the state as well as when "
                  "it entered one. Declared unverifiable before execution "
                  "rather than after a recomputation failed.",
        source="none",
        dedup="not applicable",
        window="not applicable",
        missing="not applicable",
        recompute=None,
    ),
)

BY_COLUMN = {rule.column: rule for rule in RULES}


def covered_columns() -> tuple[str, ...]:
    """Every reporting-view column the ruleset accounts for.

    G1 requires that no column is left unexamined, and the gate checks this
    against the view's actual columns rather than against a list someone
    remembered to update.
    """
    return tuple(rule.column for rule in RULES)
