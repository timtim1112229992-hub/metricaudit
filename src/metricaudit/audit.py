"""P6 and M8: completeness and referential integrity across both stores.

Reconciliation asks whether a figure follows from the records. This asks whether
the records are in a state to support any figure at all. A store holding events
that belong to no group, or carrying a display name on some rows and not others,
will still produce a tidy reporting view, and the tidiness will not mention it.

Every proportion carries an exact interval, and a proportion observed as zero
carries a one-sided bound instead of a bare zero, so that an absence is reported
with a stated precision rather than as a fact.
"""
from __future__ import annotations

import pandas as pd

from .config import SETTINGS
from .ingest import Corpus
from .models import clopper_pearson, exact_upper_bound


def _proportion(label: str, store: str, successes: int, trials: int) -> dict:
    rate = successes / trials if trials else float("nan")
    lower, upper = clopper_pearson(successes, trials, SETTINGS.confidence)
    row = {"store": store, "check": label, "n_affected": int(successes),
           "n_records": int(trials), "rate": rate,
           "ci_lower": lower, "ci_upper": upper, "zero_bound": None}
    if successes == 0 and trials:
        row["zero_bound"] = exact_upper_bound(trials, SETTINGS.confidence)
    return row


def completeness(corpus: Corpus) -> pd.DataFrame:
    """Missing identifiers, absent display names, unclassified stage references."""
    rows = []
    known_groups = set(corpus["groups"]["id"])

    ops = corpus["operational"]
    rows.append(_proportion("records without a group identifier", "operational",
                            int(ops["group_id"].isna().sum()), len(ops)))
    rows.append(_proportion("records without a stage reference", "operational",
                            int(ops["stage"].isna().sum()), len(ops)))
    rows.append(_proportion("records whose group identifier is not a known group",
                            "operational",
                            int((~ops["group_id"].isin(known_groups)
                                 & ops["group_id"].notna()).sum()), len(ops)))
    if "client_ts" in ops.columns:
        rows.append(_proportion("records without a client timestamp", "operational",
                                int(ops["client_ts"].isna().sum()), len(ops)))

    rep = corpus["reporting"]
    rows.append(_proportion("records without a group identifier", "reporting",
                            int(rep["group_id"].isna().sum()), len(rep)))
    if "group_name" in rep.columns:
        rows.append(_proportion("records without a display name", "reporting",
                                int(rep["group_name"].isna().sum()), len(rep)))
    rows.append(_proportion("records whose group identifier is not a known group",
                            "reporting",
                            int((~rep["group_id"].isin(known_groups)
                                 & rep["group_id"].notna()).sum()), len(rep)))

    view = corpus["view"]
    rows.append(_proportion("view rows whose group is not a known group", "view",
                            int((~view["group_id"].isin(known_groups)).sum()),
                            len(view)))
    return pd.DataFrame(rows)


def name_drift(corpus: Corpus) -> dict:
    """Does the denormalised display name still agree with the group record?

    A view that stores a name beside an identifier has taken on an obligation to
    keep the two in step, and nothing in the architecture discharges it. Worth
    checking even where it passes, because passing is not the same as being
    guaranteed, and the distinction is the point.
    """
    groups = corpus["groups"].set_index("id")["name"]
    out = {}
    for store in ("reporting", "view"):
        frame = corpus[store]
        column = "group_name"
        if column not in frame.columns:
            continue
        present = frame[frame[column].notna() & frame["group_id"].notna()]
        expected = present["group_id"].map(groups)
        disagree = int((present[column].astype(str) != expected.astype(str)).sum())
        out[store] = {"n_compared": int(len(present)), "n_disagreeing": disagree,
                      "n_unnamed": int(frame[column].isna().sum())}
    return out


def unnamed_by_category(corpus: Corpus) -> pd.DataFrame:
    """Where the missing display names sit.

    A name absent at random is untidiness. A name absent on exactly the
    categories one store holds and the other does not is a structural fact about
    how the row was written, and points at which write path omitted the join.
    """
    rep = corpus["reporting"]
    if "group_name" not in rep.columns:
        return pd.DataFrame()
    table = rep.assign(unnamed=rep["group_name"].isna()) \
        .groupby("event_type")["unnamed"].agg(["sum", "size"]).reset_index()
    table.columns = ["event_category", "n_unnamed", "n_records"]
    table["n_unnamed"] = table["n_unnamed"].astype(int)
    table["all_unnamed"] = table["n_unnamed"] == table["n_records"]
    return table.sort_values("event_category").reset_index(drop=True)


def archive_redundancy(corpus: Corpus) -> pd.DataFrame:
    """D6 as a reportable table: collections that duplicate one another.

    A reader of the export picks an array by name. Where one array's identifiers
    are a strict subset of another's, the choice silently determines how much of
    the session is analysed, and nothing in either array says so.
    """
    relations = (corpus.arrays or {}).get("subset_relations", [])
    if not relations:
        return pd.DataFrame()
    frame = pd.DataFrame(relations)
    frame["proportion_covered"] = frame["n_subset"] / frame["n_superset"]
    return frame.sort_values("n_subset", ascending=False).reset_index(drop=True)
