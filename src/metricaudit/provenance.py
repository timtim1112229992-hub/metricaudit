"""P10 and D8: the release package.

The bargain this package is built around is that a reader should be able to
confirm the audit without receiving the records. What leaves the machine is the
column map saying which fields were read, a digest per analysed record, and a
manifest recording the code state, the environment and every parameter that
could move a number.

No record, no estimate, no table and no figure is included. That exclusion is
the one the outline states most firmly, and it is enforced here by a writer that
refuses rather than by a convention someone has to remember.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import FORBIDDEN_COLUMNS, PROVENANCE_DIR, SETTINGS, TEXT_BEARING_COLUMNS
from .rules import RULESET_VERSION, RULES

PACKAGES = ("numpy", "pandas", "scipy", "matplotlib", "openpyxl")

ALLOWED_KEYS = {"schema", "generated_utc", "source", "code", "environment",
                "settings", "column_map", "tables", "counts", "counting_rules",
                "notes"}


def _git(*args: str) -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(("git", *args), cwd=root, capture_output=True,
                             text=True, timeout=15)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _versions() -> dict:
    found = {}
    for name in PACKAGES:
        try:
            found[name] = __import__(name).__version__
        except Exception:
            found[name] = None
    return found


def rule_digest() -> str:
    """A digest over the declared rules themselves.

    G2 asks whether the rules predate the recomputation. The commit history
    answers that; this answers the narrower question of whether the rules that
    ran are the rules that were committed, which a reader cannot check from a
    commit identifier alone once the file has been edited.
    """
    blob = "\n".join(
        f"{r.column}|{r.kind}|{r.source}|{r.dedup}|{r.window}|{r.missing}|"
        f"{r.statement}" for r in RULES)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def rules_committed_before_code() -> dict:
    """Evidence for G2, read from the repository rather than asserted.

    Compares the commit that last added the rules file against the commit that
    first added the module which applies it. If the second is not strictly later
    the gate has not been met, and the audit's central claim loses the thing
    that made it more than a curve fit.
    """
    def first_commit(path: str) -> str | None:
        return _git("log", "--diff-filter=A", "--format=%H", "--", path)

    def commit_time(commit: str) -> int | None:
        stamp = _git("show", "-s", "--format=%ct", commit)
        return int(stamp) if stamp and stamp.isdigit() else None

    rules_commit = first_commit("src/metricaudit/rules.py")
    applier_commit = first_commit("src/metricaudit/reconcile.py")
    if not rules_commit or not applier_commit:
        return {"checked": False, "reason": "commit history unavailable"}

    rules_commit = rules_commit.splitlines()[-1]
    applier_commit = applier_commit.splitlines()[-1]
    t_rules, t_applier = commit_time(rules_commit), commit_time(applier_commit)
    if t_rules is None or t_applier is None:
        return {"checked": False, "reason": "commit timestamps unavailable"}

    return {"checked": True,
            "rules_first_committed": rules_commit,
            "applier_first_committed": applier_commit,
            "rules_precede_applier": t_rules < t_applier,
            "seconds_between": t_applier - t_rules}


def counting_rule_register() -> list[dict]:
    """The rules as released: their statements, assumptions and version.

    Released in full because a reader cannot judge a reconciliation without
    knowing what was reconciled against, and because a rule stated in a paper's
    methods section and a rule executed in code have a way of drifting apart.
    """
    return [{"column": r.column, "kind": r.kind, "source": r.source,
             "statement": r.statement, "deduplication": r.dedup,
             "window": r.window, "missing_identifiers": r.missing,
             "verifiable": r.verifiable} for r in RULES]


def build_manifest(corpus_meta: dict, notes: str = "") -> dict:
    return {
        "schema": "metricaudit/provenance/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": corpus_meta.get("source"),
        "code": {"commit": _git("rev-parse", "HEAD"),
                 "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                 "dirty": bool(_git("status", "--porcelain")),
                 "ruleset_version": RULESET_VERSION,
                 "ruleset_sha256": rule_digest(),
                 "rule_ordering": rules_committed_before_code()},
        "environment": {"python": sys.version.split()[0],
                        "implementation": platform.python_implementation(),
                        "packages": _versions()},
        "settings": {"seed": SETTINGS.seed,
                     "confidence": SETTINGS.confidence,
                     "bootstrap_replicates": SETTINGS.bootstrap_replicates,
                     "tolerance": SETTINGS.tolerance,
                     "mechanism_tolerance": SETTINGS.mechanism_tolerance,
                     "parity_window_s": SETTINGS.parity_window_s,
                     "rounding_modes": list(SETTINGS.rounding_modes),
                     "dedup_modes": list(SETTINGS.dedup_modes)},
        "column_map": corpus_meta.get("column_map"),
        "tables": corpus_meta.get("tables"),
        "counts": corpus_meta.get("counts"),
        "counting_rules": counting_rule_register(),
        "notes": notes,
    }


def _assert_releasable(manifest: dict) -> None:
    extra = set(manifest) - ALLOWED_KEYS
    if extra:
        raise ValueError(
            f"manifest carries keys outside the release policy: {sorted(extra)}")

    for name, entry in (manifest.get("tables") or {}).items():
        unexpected = set(entry) - {"n_records", "aggregate_sha256", "record_sha256"}
        if unexpected:
            raise ValueError(f"digest block for {name} carries {sorted(unexpected)}")
        for digest in entry["record_sha256"]:
            if not (len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)):
                raise ValueError(
                    f"digest block for {name} contains a non-digest value")

    for table, mapping in (manifest.get("column_map") or {}).items():
        if not isinstance(mapping, dict) or "read" not in mapping:
            raise ValueError(f"column map for {table} is not in the released shape")
        declared = mapping.get("parsed_for_numeric_keys_only") or {}
        for column in mapping["read"]:
            if column not in TEXT_BEARING_COLUMNS:
                continue
            # Reading a column that can hold written text is permitted only
            # through the narrow extractor, and only against keys named here.
            # An undeclared one means some other code path reached the raw
            # string, which is the failure this whole arrangement exists to
            # prevent.
            if not declared.get(column):
                raise ValueError(
                    f"column map for {table} reads {column}, which can carry "
                    f"written text, without declaring the numeric keys taken "
                    f"from it")


def write_manifest(manifest: dict, directory: Path | None = None) -> Path:
    _assert_releasable(manifest)
    directory = directory or PROVENANCE_DIR
    directory.mkdir(parents=True, exist_ok=True)

    tables = manifest.pop("tables")
    digests = {"schema": "metricaudit/digests/1",
               "tables": {k: {"n_records": v["n_records"],
                              "aggregate_sha256": v["aggregate_sha256"],
                              "record_sha256": v["record_sha256"]}
                          for k, v in tables.items()}}
    (directory / "record_digests.json").write_text(
        json.dumps(digests, indent=1, sort_keys=True), encoding="utf-8")

    (directory / "column_map.json").write_text(
        json.dumps({"schema": "metricaudit/columnmap/1",
                    "columns": manifest["column_map"]}, indent=1, sort_keys=True),
        encoding="utf-8")

    (directory / "counting_rules.json").write_text(
        json.dumps({"schema": "metricaudit/rules/1",
                    "ruleset_version": RULESET_VERSION,
                    "ruleset_sha256": rule_digest(),
                    "rules": manifest["counting_rules"]}, indent=1, sort_keys=True),
        encoding="utf-8")

    manifest["tables"] = {k: {"n_records": v["n_records"],
                              "aggregate_sha256": v["aggregate_sha256"]}
                          for k, v in tables.items()}
    path = directory / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return path


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
