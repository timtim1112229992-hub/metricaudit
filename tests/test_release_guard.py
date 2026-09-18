# -*- coding: utf-8 -*-
"""Tests that stop the session's records leaving this machine.

The custody undertaking is that only the column map, the record digests, the
counting rules and the run manifest are releasable. These tests fail if any
released artefact carries participant text, if the manifest writer is handed
something outside the policy, or if a file the policy excludes has been staged
into the repository. They fail rather than warn, because a warning in a test
suite is a leak with a note attached.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metricaudit.config import TEXT_BEARING_COLUMNS  # noqa: E402
from metricaudit.provenance import (ALLOWED_KEYS, _assert_releasable)  # noqa: E402

OUTPUTS = ROOT / "outputs"
PROVENANCE = ROOT / "provenance"

# Any run of Han characters long enough to be something a person wrote. The
# session is a Chinese primary class, so this is the shape a leak would take.
HAN_RUN = re.compile(r"[\u4e00-\u9fff]{2,}")

RELEASABLE = {"column_map.json", "record_digests.json", "run_manifest.json",
              "counting_rules.json"}


def test_manifest_writer_refuses_an_unexpected_key():
    with pytest.raises(ValueError, match="outside the release policy"):
        _assert_releasable({"schema": "x", "entries": ["a pupil's sentence"]})


def test_manifest_writer_refuses_a_digest_block_that_is_not_digests():
    bad = {"schema": "x",
           "tables": {"operational": {"n_records": 1, "aggregate_sha256": "a" * 64,
                                      "record_sha256": ["not a digest"]}}}
    with pytest.raises(ValueError, match="non-digest"):
        _assert_releasable(bad)


def test_manifest_writer_refuses_an_undeclared_read_of_a_text_column():
    """The one concession the design makes, held to its stated width.

    Reading a payload column is permitted because the gauge lives inside one.
    It is permitted only through the narrow extractor, and only against keys
    named in the map. A map that reads such a column without naming the keys
    means some other code path reached the raw string.
    """
    bad = {"schema": "x", "tables": {},
           "column_map": {"operational": {"read": ["id", "payload_json"],
                                          "parsed_for_numeric_keys_only": {}}}}
    with pytest.raises(ValueError, match="without declaring the numeric keys"):
        _assert_releasable(bad)


def test_manifest_writer_accepts_a_declared_read():
    ok = {"schema": "x", "tables": {},
          "column_map": {"operational": {
              "read": ["id", "payload_json"],
              "parsed_for_numeric_keys_only": {"payload_json": ["help_count"]}}}}
    _assert_releasable(ok)


def test_every_text_bearing_column_is_named_in_the_policy():
    """A column that can hold writing must be known to the policy that guards it."""
    assert "payload_json" in TEXT_BEARING_COLUMNS
    assert "metrics_snapshot_json" in TEXT_BEARING_COLUMNS
    assert "form_data_json" in TEXT_BEARING_COLUMNS


@pytest.mark.skipif(not (PROVENANCE / "run_manifest.json").exists(),
                    reason="pipeline has not been run")
def test_manifest_holds_only_permitted_keys():
    manifest = json.loads((PROVENANCE / "run_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) <= ALLOWED_KEYS
    blob = json.dumps(manifest, ensure_ascii=False)
    assert not HAN_RUN.findall(blob), "the manifest carries written text"
    # The path to the restricted session is not a disclosure the manifest makes.
    assert "original_data" not in blob


@pytest.mark.skipif(not (PROVENANCE / "record_digests.json").exists(),
                    reason="pipeline has not been run")
def test_digest_register_is_digests_and_nothing_else():
    register = json.loads((PROVENANCE / "record_digests.json").read_text(encoding="utf-8"))
    for name, entry in register["tables"].items():
        assert set(entry) == {"n_records", "aggregate_sha256", "record_sha256"}
        assert len(entry["record_sha256"]) == entry["n_records"]
        for digest in entry["record_sha256"]:
            assert re.fullmatch(r"[0-9a-f]{64}", digest), f"{name} holds a non-digest"


@pytest.mark.skipif(not PROVENANCE.is_dir(), reason="nothing published yet")
def test_only_the_release_package_appears_in_provenance():
    """G6, checked as a property of the directory rather than of intentions."""
    present = {p.name for p in PROVENANCE.glob("*") if p.is_file()}
    assert present <= RELEASABLE, f"unexpected files in provenance: {present - RELEASABLE}"


@pytest.mark.skipif(not PROVENANCE.is_dir(), reason="nothing published yet")
def test_no_released_file_carries_participant_text():
    for path in PROVENANCE.glob("*.json"):
        blob = path.read_text(encoding="utf-8")
        assert not HAN_RUN.findall(blob), f"{path.name} carries written text"


def _tracked() -> list[str] | None:
    """Paths git is tracking, or None where this is not a working repository."""
    try:
        out = subprocess.run(("git", "ls-files"), cwd=ROOT, capture_output=True,
                             text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.split() if out.returncode == 0 else None


def test_no_excluded_file_is_tracked():
    """The ignore rules are a policy; this is the check that they were obeyed.

    A file already tracked before its pattern was added to .gitignore stays
    tracked and is pushed on the next commit without complaint, which is the
    ordinary way data reaches a public repository. Reading the index directly
    catches that, where reading the ignore file cannot.
    """
    tracked = _tracked()
    if tracked is None:
        pytest.skip("not a git working tree")

    forbidden_dirs = ("outputs/", "data/", "raw/", "raw_data/", "original_data/",
                      "results/", "figures/", "artefacts/", "artifacts/", "logs/")
    forbidden_types = (".xlsx", ".xls", ".png", ".eps", ".pdf", ".docx", ".pptx",
                       ".parquet", ".pkl", ".sqlite", ".npy", ".h5")

    offenders = []
    for path in tracked:
        if path.startswith(forbidden_dirs):
            offenders.append(path)
        elif path.endswith(forbidden_types):
            offenders.append(path)
        elif path.endswith(".csv") and not path.startswith("synthetic/"):
            offenders.append(path)
        elif path.endswith(".json") and not path.startswith("provenance/"):
            offenders.append(path)
    assert not offenders, f"excluded files are tracked: {offenders}"


def test_only_csv_is_tracked_under_the_synthetic_session():
    """The one carve-out in the ignore rules, held to its stated width."""
    tracked = _tracked()
    if tracked is None:
        pytest.skip("not a git working tree")
    stray = [p for p in tracked
             if p.startswith("synthetic/") and not p.endswith(".csv")]
    assert not stray, f"the synthetic exception is being used for {stray}"


def test_no_tracked_file_carries_participant_text():
    tracked = _tracked()
    if tracked is None:
        pytest.skip("not a git working tree")
    for name in tracked:
        path = ROOT / name
        if not path.is_file():
            continue
        blob = path.read_text(encoding="utf-8", errors="replace")
        found = HAN_RUN.findall(blob)
        assert not found, f"{name} carries written text: {found[:3]}"


def _messages() -> list[str] | None:
    """Every commit message in the repository, subject and body."""
    # A textual separator rather than a null byte, which this platform refuses
    # to carry through an argument list.
    mark = "-----end-of-commit-message-----"
    try:
        out = subprocess.run(["git", "log", f"--format=%s%n%b%n{mark}"], cwd=ROOT,
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return [m.strip() for m in out.stdout.split(mark) if m.strip()]


def test_no_commit_credits_a_tool_as_an_author():
    """A commit trailer is the one place an automated editor signs its work.

    Version control writes the trailer without being asked, it renders on the
    forge as co-authorship, and it survives every later check that looks only at
    the working tree. Tooling does not become a contributor by having been
    present, so the history is inspected rather than trusted.
    """
    messages = _messages()
    if messages is None:
        pytest.skip("not a git working tree")
    offenders = [m.splitlines()[0] for m in messages
                 if re.search(r"^\s*co-authored-by\s*:", m, re.I | re.M)]
    assert not offenders, f"commits credit a co-author: {offenders}"


def test_every_commit_carries_the_account_of_record():
    """One identity across the history, and it discloses nobody.

    A single commit made under a personal name or an institutional address
    undoes anonymised review for the whole archive, and it is the sort of thing
    that happens when a machine's global configuration is picked up silently.
    """
    try:
        out = subprocess.run(["git", "log", "--format=%an <%ae>|%cn <%ce>"],
                             cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("not a git working tree")
    identities = {part for line in out.stdout.splitlines() if line.strip()
                  for part in line.split("|")}
    assert len(identities) == 1, f"the history carries several identities: {identities}"
    only = identities.pop()
    assert "@users.noreply." not in only, (
        "the forge substituted a private address for the account of record, so "
        f"the history does not match the declared one: {only}")


def test_no_commit_message_carries_a_measurement():
    """A message is repository content, and findings do not belong in it.

    Every other guard here reads the working tree, which is exactly what a
    commit message is not. A quantity from the restricted session recorded in a
    message is published the moment the branch is pushed and cannot be recalled,
    so messages are held to describing the change and not its result.
    """
    messages = _messages()
    if messages is None:
        pytest.skip("not a git working tree")
    # Three digits is past any version number or exit code a message needs, and
    # a decimal fraction in a message is almost always an estimate.
    measurement = re.compile(r"(?<![\w.])\d{3,}(?![\w.])|\d+\.\d+")
    offenders = []
    for message in messages:
        hit = measurement.search(message)
        if hit:
            offenders.append(f"{message.splitlines()[0]!r} -> {hit.group(0)!r}")
    assert not offenders, f"commit messages carry measurements: {offenders}"


def test_released_text_names_no_author_and_no_working_label():
    """R13, R20 and R26, checked rather than remembered.

    Attribution travels with the paper. An archive that names a contributor, or
    that carries a manuscript's working label, defeats anonymised review for
    everyone on it.
    """
    banned = re.compile(
        r"Paper ?[123][ab]\b|orcid|@[\w.-]+\.(?:com|edu|ac|org)|"
        r"\bHuang ?Hai\b|Construct Validity and Provenance",
        re.IGNORECASE)
    checked = 0
    for path in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.md")) + \
            list(ROOT.rglob("*.toml")) + list(PROVENANCE.rglob("*.json")):
        if "__pycache__" in path.parts or ".git" in path.parts:
            continue
        if OUTPUTS in path.parents:
            continue
        # This file has to spell the banned strings out in order to look for
        # them, so scanning it would report itself and nothing else.
        if path.resolve() == Path(__file__).resolve():
            continue
        checked += 1
        hit = banned.search(path.read_text(encoding="utf-8", errors="replace"))
        assert not hit, f"{path.relative_to(ROOT)} carries {hit.group(0)!r}"
    assert checked > 10, "the scan found almost nothing and is probably misdirected"
