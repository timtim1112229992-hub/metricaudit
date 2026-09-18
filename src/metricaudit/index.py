"""P2 and D4, D5: identifier sets and category maps per store.

Comparing two stores by record count answers a weaker question than comparing
them by membership. Two stores can hold the same number of records and not the
same records, and a pipeline that loses one event while duplicating another
looks perfectly healthy under a count. The sets are built here so that the
comparison in P3 can be set-theoretic.

D5 marks the categories present in one store only, because those explain
legitimate differences in volume and must be separated from the differences that
do not have an explanation.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .ingest import Corpus

STORES = ("operational", "reporting")


@dataclass(frozen=True)
class StoreIndex:
    """One store's identifiers, categories and per-category membership."""

    name: str
    identifiers: frozenset
    categories: dict[str, int]
    by_category: dict[str, frozenset]
    n_records: int

    @property
    def identifier_cardinality(self) -> int:
        return len(self.identifiers)


def build(corpus: Corpus) -> dict[str, StoreIndex]:
    indexes = {}
    for name in STORES:
        frame = corpus[name]
        categories = frame["event_type"].value_counts().to_dict()
        by_category = {
            category: frozenset(frame.loc[frame["event_type"] == category, "id"])
            for category in categories
        }
        indexes[name] = StoreIndex(
            name=name,
            identifiers=frozenset(frame["id"]),
            categories={k: int(v) for k, v in categories.items()},
            by_category=by_category,
            n_records=len(frame),
        )
    return indexes


def category_coverage(indexes: dict[str, StoreIndex]) -> dict:
    """D5: which categories each store holds, and which are exclusive to one.

    The exclusive categories are the honest part of a volume difference. Naming
    them separates the difference that has an explanation from the remainder,
    and it is the remainder that the audit is about.
    """
    ops, rep = indexes["operational"], indexes["reporting"]
    ops_cats, rep_cats = set(ops.categories), set(rep.categories)
    shared = sorted(ops_cats & rep_cats)

    exclusive_reporting = sorted(rep_cats - ops_cats)
    exclusive_operational = sorted(ops_cats - rep_cats)
    explained = sum(rep.categories[c] for c in exclusive_reporting) \
        - sum(ops.categories[c] for c in exclusive_operational)

    return {
        "shared_categories": shared,
        "exclusive_to_reporting": {c: rep.categories[c] for c in exclusive_reporting},
        "exclusive_to_operational": {c: ops.categories[c] for c in exclusive_operational},
        "n_operational": ops.n_records,
        "n_reporting": rep.n_records,
        "raw_difference": rep.n_records - ops.n_records,
        "difference_explained_by_exclusive_categories": int(explained),
        "residual_difference": int(rep.n_records - ops.n_records - explained),
    }


def integrity(corpus: Corpus) -> pd.DataFrame:
    """D7: cardinality at the point of ingest, to be compared after each step."""
    rows = []
    for name, frame in corpus.frames.items():
        key = "id" if "id" in frame.columns else frame.columns[0]
        rows.append({
            "table": name,
            "n_rows": len(frame),
            "n_distinct_keys": int(frame[key].nunique()),
            "key_column": key,
            "duplicate_keys": int(len(frame) - frame[key].nunique()),
        })
    return pd.DataFrame(rows).sort_values("table").reset_index(drop=True)
