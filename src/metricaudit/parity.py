"""P3 and M3: is the dual write faithful?

The architecture writes each client event to two stores. If that write is
faithful then the two hold the same events, and an aggregate that reads from
both is at worst redundant. If it is not faithful, then which store an indicator
draws on changes the indicator's value, and no reader of the reporting layer can
tell which it drew on.

Parity is tested two ways because the first way turns out to be uninformative,
and that is itself worth reporting. Comparing primary keys is the natural test
and the one a reader would assume had been done. It answers nothing when the two
stores mint their own keys, and a Jaccard similarity of zero across every shared
category does not mean the write lost everything. It means the write left
nothing to join on.

So a second test compares a composite natural key within a tolerance window.
That is weaker evidence than a shared key would be, and the weakness is the
finding: absent an idempotency key, neither the platform nor an auditor can
distinguish a redelivered event from a genuine repeat.
"""
from __future__ import annotations

import pandas as pd

from .config import SETTINGS
from .index import StoreIndex

COMPOSITE = ("session_id", "group_id", "event_type", "payload_json")


def jaccard(left: frozenset, right: frozenset) -> float:
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def identifier_parity(indexes: dict[str, StoreIndex]) -> pd.DataFrame:
    """M3 as specified: Jaccard and set difference on identifiers per category."""
    ops, rep = indexes["operational"], indexes["reporting"]
    shared = sorted(set(ops.categories) & set(rep.categories))

    rows = []
    for category in shared:
        left = ops.by_category[category]
        right = rep.by_category[category]
        rows.append({
            "event_category": category,
            "n_operational": len(left),
            "n_reporting": len(right),
            "count_difference": len(right) - len(left),
            "n_shared_identifiers": len(left & right),
            "jaccard": jaccard(left, right),
            "operational_only": len(left - right),
            "reporting_only": len(right - left),
        })
    return pd.DataFrame(rows).sort_values("event_category").reset_index(drop=True)


def _composite_key(frame: pd.DataFrame) -> pd.Series:
    parts = [frame[c].astype(str) for c in COMPOSITE if c in frame.columns]
    return parts[0].str.cat(parts[1:], sep="\u001f")


def composite_parity(corpus_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Parity on a natural key within a window, since the keys do not join.

    Matching is greedy and one-to-one within each composite group: each
    operational record claims the nearest unclaimed reporting record inside the
    window. Greedy nearest-neighbour matching can in principle be beaten by an
    optimal assignment, but only where events of one category collide inside the
    window, and a pairing that is ambiguous at that resolution is not evidence
    of a faithful write in the first place.
    """
    ops = corpus_frames["operational"].copy()
    rep = corpus_frames["reporting"].copy()
    ops["_key"] = _composite_key(ops)
    rep["_key"] = _composite_key(rep)
    ops["_t"] = pd.to_datetime(ops["created_at"], format="ISO8601", utc=True)
    rep["_t"] = pd.to_datetime(rep["created_at"], format="ISO8601", utc=True)

    window = pd.Timedelta(seconds=SETTINGS.parity_window_s)
    rows = []
    shared = sorted(set(ops["event_type"]) & set(rep["event_type"]))

    for category in shared:
        o = ops[ops["event_type"] == category]
        r = rep[rep["event_type"] == category]
        matched = 0
        offsets: list[float] = []
        claimed: set[int] = set()

        for key, left in o.groupby("_key", sort=False):
            right = r[r["_key"] == key]
            if right.empty:
                continue
            for _, lrow in left.iterrows():
                free = right[~right.index.isin(claimed)]
                if free.empty:
                    break
                delta = (free["_t"] - lrow["_t"]).abs()
                nearest = delta.idxmin()
                if delta[nearest] <= window:
                    claimed.add(nearest)
                    matched += 1
                    offsets.append(
                        (free.loc[nearest, "_t"] - lrow["_t"]).total_seconds())

        rows.append({
            "event_category": category,
            "n_operational": len(o),
            "n_reporting": len(r),
            "n_matched_pairs": matched,
            "operational_unmatched": len(o) - matched,
            "reporting_unmatched": len(r) - matched,
            "median_write_offset_s": float(pd.Series(offsets).median()) if offsets else None,
            "max_write_offset_s": float(pd.Series(offsets).abs().max()) if offsets else None,
        })
    return pd.DataFrame(rows).sort_values("event_category").reset_index(drop=True)


def internal_duplication(corpus_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Records a store holds more than once, separated from records that repeat.

    Distinct from parity. This asks whether one store received the same event
    twice, which is what repeated client emission looks like from inside a
    single store, and it is a precondition for attributing a divergence to that
    mechanism rather than to the cross-store write.

    The separation matters more than the count. A group that saves the same
    stage twice an hour apart has produced two identical payloads and two
    genuine events; only a repeat arriving inside the write window is a
    candidate duplicate. Counting bare payload repeats would have called a third
    of the operational store duplicated, which would have been a finding about
    how children work rather than about the pipeline.
    """
    rows = []
    for name in ("operational", "reporting"):
        frame = corpus_frames[name].copy()
        frame["_key"] = _composite_key(frame)
        frame["_t"] = pd.to_datetime(frame["created_at"], format="ISO8601", utc=True)
        frame = frame.sort_values(["_key", "_t"])
        gap = frame.groupby("_key")["_t"].diff().dt.total_seconds()
        within = gap.notna() & (gap <= SETTINGS.parity_window_s)
        rows.append({
            "store": name,
            "n_records": len(frame),
            "n_distinct_payload_repeats": int(frame["_key"].nunique()),
            "n_repeats_beyond_window": int(
                (gap.notna() & (gap > SETTINGS.parity_window_s)).sum()),
            "n_repeats_within_window": int(within.sum()),
            "duplication_rate_within_window": float(within.sum() / len(frame))
            if len(frame) else float("nan"),
        })
    return pd.DataFrame(rows)
