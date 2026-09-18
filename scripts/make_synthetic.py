#!/usr/bin/env python
"""Generate the stand-in session that makes this package runnable by a stranger.

A reconciliation tool that cannot be run without the restricted data is not much
of a release, so this writes a session with the same schema and shape and
nothing else in common. It is built to contain a reconciliation fault, because a
stand-in in which everything reconciles would demonstrate only that the code
runs, not that it detects anything.

The fault planted here is gauge summation: a running counter is snapshotted on
every agent turn and the reporting view sums the snapshots. The magnitudes are
chosen to differ from the observed ones so that nobody can mistake output from
this corpus for a replication.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "synthetic"

SEED = 4242
N_GROUPS = 8          # the session has twelve; this deliberately does not
N_STAGES = 5          # the session has seven
SESSION_ID = "s0000000-0000-4000-8000-000000000001"
START = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)

STAGE_KEYS = ("warmup", "observe", "predict", "test", "review")
WRITE_LAG_S = 0.09    # the second store's write, so parity has something to find


def uid(prefix: str, n: int) -> str:
    return f"{prefix}{n:08d}-0000-4000-8000-000000000000"


def stamp(t: datetime) -> str:
    return t.isoformat()


def write(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.csv"
    if not rows:
        path.write_text("no_records\n", encoding="utf-8", newline="")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path.name:<28} {len(rows):>5} rows")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rng = np.random.default_rng(SEED)

    groups, submissions, decisions, qa = [], [], [], []
    operational, reporting, view = [], [], []

    reporting.append({
        "id": uid("a", 900001), "session_id": SESSION_ID, "group_id": "",
        "group_name": "", "event_type": "session.created",
        "payload_json": json.dumps({"group_count": N_GROUPS}),
        "created_at": stamp(START)})

    counter = 0
    for g in range(1, N_GROUPS + 1):
        gid = uid("g", g)
        base = START + timedelta(minutes=2 * g)
        groups.append({
            "id": gid, "session_id": SESSION_ID, "group_number": g,
            "name": f"Team {g}", "access_token": uid("k", g),
            "current_stage": N_STAGES - 1, "status": "active" if g % 2 else "needs_help",
            "members": "[]", "last_active_at": "", "created_at": stamp(START)})

        # Stage submissions, with a completion rate that is not a round number
        # so that the rounding sweep has something to separate.
        rates = []
        for stage in range(N_STAGES):
            rate = float(np.round(rng.uniform(0.55, 1.0), 4))
            rates.append(rate)
            t = base + timedelta(minutes=3 * stage)
            submissions.append({
                "id": uid("s", g * 100 + stage), "group_id": gid, "stage": stage,
                "stage_key": STAGE_KEYS[stage], "is_completed": True,
                "completion_rate": rate, "completed_at": stamp(t),
                "updated_at": stamp(t),
                "form_data_json": json.dumps({"note": "synthetic placeholder"})})

        # Events. The help counter is a running gauge, as in the real client.
        n_help = int(rng.integers(1, 7)) if g != 3 else 41   # one loud group
        n_wrong = int(rng.integers(0, 4))
        n_idle = int(rng.integers(2, 9))

        gauge = 0
        help_times = []
        for i in range(n_help):
            gauge += 1
            t = base + timedelta(seconds=40 * i + 11)
            help_times.append((t, gauge))
            payload = json.dumps({"help_count": gauge})
            operational.append({
                "id": uid("o", g * 10000 + i), "session_id": SESSION_ID,
                "group_id": gid, "stage": i % N_STAGES,
                "event_type": "agent.help_click", "target": "help-button",
                "payload_json": payload, "client_ts": stamp(t), "created_at": stamp(t)})
            reporting.append({
                "id": uid("a", g * 10000 + i), "session_id": SESSION_ID,
                "group_id": gid, "group_name": f"Team {g}",
                "event_type": "agent.help_click", "payload_json": payload,
                "created_at": stamp(t + timedelta(seconds=WRITE_LAG_S))})

        for i in range(n_wrong):
            t = base + timedelta(seconds=95 * i + 27)
            payload = json.dumps({"item": i, "is_correct": False})
            operational.append({
                "id": uid("o", g * 10000 + 2000 + i), "session_id": SESSION_ID,
                "group_id": gid, "stage": i % N_STAGES,
                "event_type": "quiz.wrong_answer", "target": f"stage-{i % N_STAGES}",
                "payload_json": payload, "client_ts": stamp(t), "created_at": stamp(t)})
            reporting.append({
                "id": uid("a", g * 10000 + 2000 + i), "session_id": SESSION_ID,
                "group_id": gid, "group_name": f"Team {g}",
                "event_type": "quiz.wrong_answer", "payload_json": payload,
                "created_at": stamp(t + timedelta(seconds=WRITE_LAG_S))})

        for i in range(n_idle):
            t = base + timedelta(seconds=70 * i + 5)
            payload = json.dumps({"stage": i % N_STAGES})
            operational.append({
                "id": uid("o", g * 10000 + 4000 + i), "session_id": SESSION_ID,
                "group_id": gid, "stage": i % N_STAGES,
                "event_type": "agent.idle_trigger", "target": f"stage-{i % N_STAGES}",
                "payload_json": payload, "client_ts": stamp(t), "created_at": stamp(t)})
            reporting.append({
                "id": uid("a", g * 10000 + 4000 + i), "session_id": SESSION_ID,
                "group_id": gid, "group_name": f"Team {g}",
                "event_type": "agent.idle_trigger", "payload_json": payload,
                "created_at": stamp(t + timedelta(seconds=WRITE_LAG_S))})

        for stage in range(N_STAGES):
            t = base + timedelta(minutes=3 * stage, seconds=2)
            payload = json.dumps({"stage_key": STAGE_KEYS[stage],
                                  "is_completed": True,
                                  "completion_rate": rates[stage]})
            operational.append({
                "id": uid("o", g * 10000 + 6000 + stage), "session_id": SESSION_ID,
                "group_id": gid, "stage": stage,
                "event_type": "stage.submission_save", "target": f"stage-{stage}",
                "payload_json": payload, "client_ts": stamp(t), "created_at": stamp(t)})
            # One group's last save never reaches the reporting store, so the
            # parity check has a real gap to find rather than a perfect mirror.
            if not (g == 5 and stage == N_STAGES - 1):
                reporting.append({
                    "id": uid("a", g * 10000 + 6000 + stage), "session_id": SESSION_ID,
                    "group_id": gid, "group_name": f"Team {g}",
                    "event_type": "stage.submission_save", "payload_json": payload,
                    "created_at": stamp(t + timedelta(seconds=WRITE_LAG_S))})

        # Agent turns, each snapshotting the gauge as it then stood. Summing
        # these is the planted fault.
        snapshots = []
        for i, (t, value) in enumerate(help_times):
            snapshots.append((t + timedelta(seconds=1), value, "help_click"))
        for i in range(n_idle):
            t = base + timedelta(seconds=70 * i + 6)
            reached = sum(1 for ht, _ in help_times if ht <= t)
            snapshots.append((t, reached, "idle"))

        for i, (t, value, trigger) in enumerate(sorted(snapshots)):
            did = uid("d", g * 10000 + i)
            snapshot = json.dumps({"helpClicks": value, "coreFilled": i % 4,
                                   "coreTotal": 4})
            decisions.append({
                "id": did, "session_id": SESSION_ID, "group_id": gid,
                "stage": i % N_STAGES, "action": "scaffold" if i % 3 else "probe",
                "trigger_source": trigger, "message": "synthetic placeholder",
                "hint": "synthetic placeholder", "scaffolds": "[]",
                "metrics_snapshot_json": snapshot,
                "form_snapshot_json": json.dumps({"note": "synthetic placeholder"}),
                "created_at": stamp(t)})
            reporting.append({
                "id": uid("a", g * 10000 + 8000 + i), "session_id": SESSION_ID,
                "group_id": gid, "group_name": "",
                "event_type": "agent.decision",
                "payload_json": json.dumps({"stage": i % N_STAGES,
                                            "action": "scaffold"}),
                "created_at": stamp(t)})

        n_qa = int(rng.integers(0, 4))
        for i in range(n_qa):
            t = base + timedelta(seconds=120 * i + 44)
            qa.append({"id": uid("q", g * 100 + i), "session_id": SESSION_ID,
                       "group_id": gid, "stage": i % N_STAGES,
                       "question": "synthetic placeholder",
                       "answer": "synthetic placeholder", "on_topic": True,
                       "source": "llm", "created_at": stamp(t)})
            reporting.append({
                "id": uid("a", g * 10000 + 9000 + i), "session_id": SESSION_ID,
                "group_id": gid, "group_name": "",
                "event_type": "agent.student_question",
                "payload_json": json.dumps({"stage": i % N_STAGES}),
                "created_at": stamp(t)})

        counter += 1

    # The reporting view, built the way the platform builds it: correct for
    # three columns and summing a gauge for the fourth.
    ops_by_group: dict[str, list[dict]] = {}
    for row in operational:
        ops_by_group.setdefault(row["group_id"], []).append(row)
    dec_by_group: dict[str, list[dict]] = {}
    for row in decisions:
        dec_by_group.setdefault(row["group_id"], []).append(row)

    for group in groups:
        gid = group["id"]
        mine = ops_by_group.get(gid, [])
        rates = [r["completion_rate"] for r in submissions if r["group_id"] == gid]
        gauge_sum = sum(json.loads(d["metrics_snapshot_json"])["helpClicks"]
                        for d in dec_by_group.get(gid, []))
        last = max(r["created_at"] for r in mine)
        # Stamped a moment after the event, as a write completion would be.
        group["last_active_at"] = stamp(
            datetime.fromisoformat(last) + timedelta(seconds=0.05))

        view.append({
            "group_id": gid, "session_id": SESSION_ID,
            "group_number": group["group_number"], "group_name": group["name"],
            "current_stage": N_STAGES - 1, "group_status": group["status"],
            "last_active_at": group["last_active_at"],
            "completed_stage_count": len(rates),
            "avg_completion_pct": int(np.floor(float(np.mean(rates)) * 100 + 0.5)),
            "help_click_count": gauge_sum,
            "quiz_wrong_count": sum(1 for r in mine
                                    if r["event_type"] == "quiz.wrong_answer"),
            "interaction_count": len(mine)})

    print("writing the synthetic session")
    write("02_groups", groups)
    write("03_stage_submissions", submissions)
    write("04_agent_decisions", decisions)
    write("05_agent_qa_messages", qa)
    write("06_interaction_logs", operational)
    write("07_analytics_events", reporting)
    write("08_group_progress", view)
    print(f"\nplanted fault: help_click_count sums a running gauge across agent "
          f"turns\nseed: {SEED}, groups: {N_GROUPS}, stages: {N_STAGES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
