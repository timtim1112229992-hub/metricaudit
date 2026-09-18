"""P1 and D1 to D3, D7: read the stores once, fix them by digest, count the trail.

Two things happen here that the rest of the package depends on. The analysed set
is fixed by digest before any transformation, so that what is reported can be
tied to a specific set of records without those records being disclosed. And the
read is restricted to the declared column map, so that a later step cannot reach
for a field the release says was never touched.

The count trail exists because an audit that silently loses rows would attribute
its own losses to the platform. Every step records what it received and what it
returned, and the pipeline refuses to continue where those disagree.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import (COLUMN_MAP, PAYLOAD_KEYS, REQUIRED_TABLES, SNAPSHOT_KEYS,
                     SYNTHETIC_DIR, TABLES, TEXT_BEARING_COLUMNS)
from .rules import payload_value

ENV_VAR = "METRICAUDIT_DATA_DIR"


@dataclass
class Corpus:
    """The fixed, column-restricted analysed set plus its custody record."""

    frames: dict[str, pd.DataFrame]
    source: str
    digests: dict[str, dict] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    arrays: dict[str, dict] = field(default_factory=dict)
    column_map: dict[str, dict] = field(default_factory=dict)

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.frames[name]

    def __contains__(self, name: str) -> bool:
        return name in self.frames


def data_dir() -> tuple[Path, str]:
    """Where to read from, and whether it is the real session.

    A run against synthetic data that does not say so is worse than no run at
    all, because its output looks like a result. The label travels with the
    corpus from here and is stamped on everything written.
    """
    configured = os.environ.get(ENV_VAR)
    if configured:
        path = Path(configured).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"{ENV_VAR} points at {path}, which is not a directory")
        return path, "restricted"
    if SYNTHETIC_DIR.is_dir():
        return SYNTHETIC_DIR, "synthetic"
    raise FileNotFoundError(
        f"no corpus: set {ENV_VAR}, or generate one with scripts/make_synthetic.py")


def _locate(folder: Path, stem: str) -> Path | None:
    for suffix in (".xlsx", ".csv"):
        candidate = folder / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path)


def _record_digest(frame: pd.DataFrame) -> list[str]:
    """One digest per record, over the permitted columns in declared order.

    D2. A reader given these can confirm that the set analysed here is the set
    described, without being given a single record. That is the whole of the
    confidentiality bargain: the claim becomes checkable and the children's
    writing stays where it is.
    """
    ordered = frame.reindex(columns=sorted(frame.columns))
    return [hashlib.sha256(
        "\u001f".join("" if pd.isna(v) else str(v) for v in row).encode("utf-8")
    ).hexdigest() for row in ordered.itertuples(index=False, name=None)]


def _aggregate_digest(records: list[str]) -> str:
    joined = "\n".join(records).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _restrict(name: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep only the columns the map permits, and record how each was read.

    The map is a disclosure, and enforcing it here rather than documenting it
    elsewhere makes the disclosure true by construction. Columns that may carry
    written text are recorded separately, together with the exact numeric keys
    lifted out of them, because "we read the payload column" and "we read two
    integers from inside the payload column" are different undertakings and only
    the second one is being given.
    """
    permitted = [c for c in COLUMN_MAP[name] if c in frame.columns]
    parsed = {}
    for column in permitted:
        if column not in TEXT_BEARING_COLUMNS:
            continue
        if column == "metrics_snapshot_json":
            parsed[column] = sorted(SNAPSHOT_KEYS)
        else:
            parsed[column] = sorted(PAYLOAD_KEYS.get(name, ()))
    return frame[permitted].copy(), {"read": permitted,
                                     "parsed_for_numeric_keys_only": parsed}


def _typed_payloads(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """D3: lift the declared numeric payload fields into columns of their own.

    The original serialised string is retained on the frame, because a derived
    field whose source cannot be re-read is not auditable. It is retained in
    memory only; the release writer refuses to emit it.
    """
    keys = PAYLOAD_KEYS.get(name, ())
    if not keys or "payload_json" not in frame.columns:
        return frame
    for key in keys:
        frame[f"payload.{key}"] = frame["payload_json"].map(
            lambda blob, k=key: payload_value(blob, k))
    return frame


def _typed_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    if "metrics_snapshot_json" not in frame.columns:
        return frame
    for key in SNAPSHOT_KEYS:
        frame[f"snapshot.{key}"] = frame["metrics_snapshot_json"].map(
            lambda blob, k=key: payload_value(blob, k))
    return frame


def _scan_arrays(path: Path) -> dict[str, dict]:
    """D6: every stored collection in the archive, and the subset relations.

    An export that carries two arrays over the same records, one of them
    truncated, offers a reader a silent route to an incorrect result. Finding
    that requires comparing identifier sets across every pair of collections,
    which is cheap and is never done.
    """
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(obj, dict):
        return {}

    collections = {}
    for key, value in obj.items():
        if isinstance(value, list) and value and isinstance(value[0], dict) \
                and "id" in value[0]:
            collections[key] = {r["id"] for r in value}

    relations = []
    names = sorted(collections)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            left, right = collections[a], collections[b]
            if left < right:
                relations.append({"subset": a, "superset": b, "n_subset": len(left),
                                  "n_superset": len(right), "strict": True})
            elif right < left:
                relations.append({"subset": b, "superset": a, "n_subset": len(right),
                                  "n_superset": len(left), "strict": True})
    return {"collections": {k: len(v) for k, v in collections.items()},
            "subset_relations": relations}


def load(folder: Path | None = None) -> Corpus:
    """Read every store once, restricted to the column map and fixed by digest."""
    if folder is None:
        folder, source = data_dir()
    else:
        source = "restricted" if folder != SYNTHETIC_DIR else "synthetic"

    frames: dict[str, pd.DataFrame] = {}
    digests: dict[str, dict] = {}
    counts: dict[str, int] = {}
    column_map: dict[str, list[str]] = {}

    for name, stem in TABLES.items():
        path = _locate(folder, stem)
        if path is None:
            if name in REQUIRED_TABLES:
                raise FileNotFoundError(f"{stem} is required and was not found in {folder}")
            continue
        raw = _read(path)
        counts[f"{name}.rows_in_file"] = len(raw)

        restricted, mapping = _restrict(name, raw)
        column_map[name] = mapping
        counts[f"{name}.columns_read"] = len(mapping["read"])
        counts[f"{name}.columns_ignored"] = len(raw.columns) - len(mapping["read"])

        records = _record_digest(restricted)
        digests[name] = {"n_records": len(records),
                         "aggregate_sha256": _aggregate_digest(records),
                         "record_sha256": records}

        restricted = _typed_payloads(name, restricted)
        if name == "decisions":
            restricted = _typed_snapshots(restricted)

        if len(restricted) != counts[f"{name}.rows_in_file"]:
            raise RuntimeError(f"{name} changed row count during ingest")
        frames[name] = restricted
        counts[f"{name}.rows_analysed"] = len(restricted)

    arrays = {}
    exports = sorted(p for p in folder.glob("*.json"))
    if exports:
        arrays = _scan_arrays(exports[0])

    return Corpus(frames=frames, source=source, digests=digests, counts=counts,
                  arrays=arrays, column_map=column_map)
