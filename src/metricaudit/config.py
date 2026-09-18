"""What the audit is allowed to read, and the settings that fix a run.

The column map is a disclosure as much as a convenience. A reader who wants to
know whether a reported figure could have been reached from the data has to know
which columns the analysis touched, and listing them here means the answer is
part of the release rather than part of the source code's folklore.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_DIR = PACKAGE_ROOT / "provenance"
OUTPUT_DIR = PACKAGE_ROOT / "outputs"
SYNTHETIC_DIR = PACKAGE_ROOT / "synthetic"

# The three stores the audit reconciles, plus the tables that supply the source
# events for the recomputations. The reporting view is the object of study; the
# rest are the evidence against which it is checked.
TABLES = {
    "groups": "02_groups",
    "submissions": "03_stage_submissions",
    "decisions": "04_agent_decisions",
    "qa": "05_agent_qa_messages",
    "operational": "06_interaction_logs",
    "reporting": "07_analytics_events",
    "view": "08_group_progress",
}

# Tables without which no research question can be answered. The agent tables
# are needed for mechanism attribution but not for reconciliation itself, so a
# run against a corpus that lacks them still produces a classification.
REQUIRED_TABLES = ("groups", "submissions", "operational", "reporting", "view")

# Every column the analysis is permitted to read, by table. Anything absent from
# this map is not read, and the ingest step enforces that rather than trusting
# the rest of the package to be well behaved.
COLUMN_MAP = {
    "groups": ("id", "session_id", "group_number", "name", "current_stage",
               "status", "last_active_at", "created_at"),
    "submissions": ("id", "group_id", "stage", "stage_key", "is_completed",
                    "completion_rate", "completed_at", "updated_at"),
    "decisions": ("id", "session_id", "group_id", "stage", "action",
                  "trigger_source", "metrics_snapshot_json", "created_at"),
    "qa": ("id", "session_id", "group_id", "stage", "created_at"),
    "operational": ("id", "session_id", "group_id", "stage", "event_type",
                    "target", "payload_json", "client_ts", "created_at"),
    "reporting": ("id", "session_id", "group_id", "group_name", "event_type",
                  "payload_json", "created_at"),
    "view": ("group_id", "session_id", "group_number", "group_name",
             "current_stage", "group_status", "last_active_at",
             "completed_stage_count", "avg_completion_pct", "help_click_count",
             "quiz_wrong_count", "interaction_count"),
}

# The only keys lifted out of a serialised payload, and all of them numeric or
# boolean. The payloads also carry pupil writing and a teacher's class name, and
# the audit has no business reading either, so the extractor takes these by name
# and ignores the rest of the object.
PAYLOAD_KEYS = {
    "operational": ("help_count", "completion_rate", "is_completed", "stage"),
    "reporting": ("help_count", "completion_rate", "is_completed", "stage",
                  "help_clicks"),
}

# Keys lifted from the agent's metrics snapshot, which is where the running
# help-click gauge is recorded.
SNAPSHOT_KEYS = ("helpClicks", "coreFilled", "coreTotal")

# Columns that may hold text a child or a teacher wrote. Two of them have to be
# read, because the gauge behind the disputed indicator lives inside a
# serialised payload and there is no other route to it. They are read only
# through the narrow extractor, which takes declared numeric keys by name, and
# the release writer refuses any column map that touches one of them without
# declaring which keys it took.
TEXT_BEARING_COLUMNS = ("payload_json", "metrics_snapshot_json",
                        "form_snapshot_json", "form_data_json", "question",
                        "answer", "message", "hint", "scaffolds", "class_name")

# Columns that must never appear in a written artefact, whatever the reason for
# having read them.
FORBIDDEN_COLUMNS = TEXT_BEARING_COLUMNS + ("access_token", "teacher_pin",
                                            "join_code")

# The identity columns of the reporting view. These are keys rather than
# measurements, and they are classified separately so that the count of
# reconciled indicators is not inflated by columns that could not diverge.
VIEW_KEY_COLUMNS = ("group_id", "session_id")


@dataclass(frozen=True)
class Settings:
    """Everything that could change a reported number, in one place."""

    seed: int = 20260918
    confidence: float = 0.95
    bootstrap_replicates: int = 10000

    # Two figures are treated as reconciled only when they are equal. A
    # tolerance would let a rounding fault pass as agreement, and rounding
    # faults are among the things being looked for.
    tolerance: int = 0

    # How close a candidate mechanism must come before its prediction counts as
    # an explanation. Exact, for the same reason: a mechanism that predicts the
    # observation to within a few units has not been demonstrated, it has been
    # asserted.
    mechanism_tolerance: int = 0

    # Seconds either side of an operational record within which a reporting
    # record carrying the same group, category and payload is treated as the
    # second write of one event. The two stores are written in sequence rather
    # than atomically, so exact timestamp equality would find nothing.
    parity_window_s: float = 2.0

    # Alternative rounding conventions tested in the assumption sweep, since the
    # percentage column depends on one and no documentation states which.
    rounding_modes: tuple[str, ...] = ("half_up", "half_even", "floor", "ceil")

    # Alternative deduplication assumptions tested in the sweep.
    dedup_modes: tuple[str, ...] = ("as_stored", "distinct_composite",
                                    "distinct_within_window")


SETTINGS = Settings()
