# -*- coding: utf-8 -*-
"""Tests of the arithmetic, against cases whose answers are known independently.

An audit of someone else's arithmetic has no standing unless its own arithmetic
is checked. These use constructed inputs rather than the session, so each one
has an answer that can be worked out by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metricaudit.models import (bca_interval, clopper_pearson,  # noqa: E402
                                exact_upper_bound, kendall_tau,
                                limits_of_agreement, spearman)
from metricaudit.parity import jaccard  # noqa: E402
from metricaudit.rules import BY_COLUMN, RULES, payload_value, round_mode  # noqa: E402


class TestRounding:
    """The convention has to be the stated one, not whatever Python does."""

    def test_half_up_rounds_a_half_upward(self):
        assert round_mode(84.5, "half_up") == 85
        assert round_mode(83.5, "half_up") == 84

    def test_half_even_differs_from_half_up_where_it_matters(self):
        # The case the audit exists to catch: the two conventions disagree, and
        # a package that inherited Python's default would report the second
        # while describing the first.
        assert round_mode(84.5, "half_even") == 84
        assert round_mode(85.5, "half_even") == 86
        assert round(84.5) == 84

    def test_floor_and_ceiling_bracket_the_others(self):
        for value in (80.1, 84.5, 90.9):
            lo = round_mode(value, "floor")
            hi = round_mode(value, "ceil")
            assert lo <= round_mode(value, "half_up") <= hi
            assert lo <= round_mode(value, "half_even") <= hi

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="unknown rounding mode"):
            round_mode(1.5, "nearest_tuesday")


class TestPayloadExtraction:
    """The extractor is the only route to a payload, so its width is the policy."""

    def test_it_returns_a_declared_numeric_key(self):
        assert payload_value('{"help_count": 7}', "help_count") == 7.0

    def test_it_returns_nothing_for_an_undeclared_key(self):
        assert payload_value('{"student_answer": "a sentence"}', "help_count") is None

    def test_it_refuses_to_return_text_even_when_asked_by_name(self):
        # The guarantee is not that text is dropped later. It is that text never
        # comes back from here at all, whatever the caller asks for.
        assert payload_value('{"note": "a sentence a child wrote"}', "note") is None

    def test_a_boolean_is_not_a_number(self):
        assert payload_value('{"is_correct": true}', "is_correct") is None

    def test_malformed_input_yields_nothing_rather_than_raising(self):
        for blob in ("", "not json", "[1,2,3]", None, 17):
            assert payload_value(blob, "help_count") is None


class TestRuleRegister:
    """G1 depends on the register being complete and internally consistent."""

    def test_every_rule_declares_its_assumptions(self):
        for rule in RULES:
            assert rule.statement.strip(), f"{rule.column} has no statement"
            assert rule.dedup.strip(), f"{rule.column} has no deduplication rule"
            assert rule.window.strip(), f"{rule.column} has no window"
            assert rule.missing.strip(), f"{rule.column} has no missing-id rule"

    def test_rule_columns_are_unique(self):
        columns = [rule.column for rule in RULES]
        assert len(columns) == len(set(columns))

    def test_an_unverifiable_rule_carries_no_recomputation(self):
        for rule in RULES:
            assert rule.verifiable == (rule.recompute is not None)

    def test_the_unverifiable_declaration_is_made_in_advance(self):
        """Declared before execution, not after a recomputation failed."""
        status = BY_COLUMN["group_status"]
        assert not status.verifiable
        assert status.source == "none"
        assert "no source path exists" in status.statement.lower()


class TestExactIntervals:
    def test_clopper_pearson_brackets_the_point_estimate(self):
        lower, upper = clopper_pearson(3, 12)
        assert lower < 3 / 12 < upper

    def test_a_zero_count_has_a_lower_bound_of_zero(self):
        lower, upper = clopper_pearson(0, 12)
        assert lower == 0.0
        assert 0 < upper < 1

    def test_a_full_count_has_an_upper_bound_of_one(self):
        lower, upper = clopper_pearson(12, 12)
        assert upper == 1.0

    def test_the_zero_bound_is_exact_not_the_rule_of_three(self):
        # The familiar 3 / n is the large-sample approximation and it is
        # anti-conservative at this size. Reporting it as though it were the
        # exact bound would overstate the room left by an observed zero.
        exact = exact_upper_bound(12)
        assert exact == pytest.approx(1 - 0.05 ** (1 / 12))
        assert exact < 3 / 12

    def test_no_trials_gives_no_bound(self):
        assert np.isnan(exact_upper_bound(0))
        assert all(np.isnan(v) for v in clopper_pearson(0, 0))


class TestBootstrap:
    def test_it_recovers_a_known_median(self):
        sample = np.arange(1.0, 13.0)
        out = bca_interval(sample, np.median, 2000, seed=1)
        assert out["estimate"] == pytest.approx(6.5)
        assert out["lower"] <= 6.5 <= out["upper"]

    def test_it_is_reproducible_under_a_fixed_seed(self):
        sample = np.array([1.0, 2.0, 2.5, 3.0, 9.0, 21.0, 2.2, 1.7])
        first = bca_interval(sample, np.mean, 1000, seed=7)
        second = bca_interval(sample, np.mean, 1000, seed=7)
        assert first == second

    def test_a_different_seed_moves_the_interval_but_not_the_estimate(self):
        sample = np.array([1.0, 2.0, 2.5, 3.0, 9.0, 21.0, 2.2, 1.7])
        first = bca_interval(sample, np.mean, 1000, seed=7)
        second = bca_interval(sample, np.mean, 1000, seed=8)
        assert first["estimate"] == second["estimate"]

    def test_too_few_units_is_declared_rather_than_estimated(self):
        out = bca_interval(np.array([1.0, 2.0]), np.mean, 1000, seed=1)
        assert np.isnan(out["lower"])
        assert "not estimable" in out["method"]

    def test_a_degenerate_sample_falls_back_without_pretending(self):
        # Every replicate equals the estimate, so the bias correction is
        # undefined. The interval returned has to say so rather than report a
        # correction it could not compute.
        out = bca_interval(np.ones(10), np.mean, 500, seed=3)
        assert not out["accelerated"]
        assert "undefined" in out["method"] or out["lower"] == out["upper"]


class TestAgreementAndAssociation:
    def test_limits_of_agreement_bracket_the_bias(self):
        diff = np.array([0.1, 0.2, 0.15, 0.3, 0.05, 0.25])
        out = limits_of_agreement(diff)
        assert out["lower"] < out["bias"] < out["upper"]

    def test_a_perfect_association_is_one(self):
        x = np.arange(10.0)
        assert spearman(x, 2 * x + 1)["rho"] == pytest.approx(1.0)

    def test_a_constant_variable_supports_no_estimate(self):
        out = spearman(np.ones(10), np.arange(10.0))
        assert not out["estimable"]
        assert np.isnan(out["rho"])

    def test_rank_agreement_detects_a_reordering(self):
        a = np.array([1.0, 2, 3, 4, 5])
        assert kendall_tau(a, a)["tau"] == pytest.approx(1.0)
        assert kendall_tau(a, a[::-1])["tau"] == pytest.approx(-1.0)


class TestParityArithmetic:
    def test_disjoint_sets_have_similarity_zero(self):
        assert jaccard(frozenset("abc"), frozenset("xyz")) == 0.0

    def test_identical_sets_have_similarity_one(self):
        assert jaccard(frozenset("abc"), frozenset("abc")) == 1.0

    def test_two_empty_sets_are_not_treated_as_agreeing(self):
        # Nothing compared to nothing is no evidence of agreement, and
        # returning one here would let an empty category look faithful.
        assert jaccard(frozenset(), frozenset()) == 0.0


class TestReconciliationBehaviour:
    """The classification itself, on frames small enough to verify by eye."""

    def _corpus(self, reported_help):
        from metricaudit.ingest import Corpus
        groups = pd.DataFrame({"id": ["g1", "g2"], "session_id": ["s", "s"],
                               "group_number": [1, 2], "name": ["A", "B"],
                               "current_stage": [1, 1], "status": ["active"] * 2,
                               "last_active_at": ["2024-01-01T00:00:00+00:00"] * 2,
                               "created_at": ["2024-01-01T00:00:00+00:00"] * 2})
        ops = pd.DataFrame({
            "id": ["e1", "e2", "e3"], "session_id": ["s"] * 3,
            "group_id": ["g1", "g1", "g2"], "stage": [0, 0, 0],
            "event_type": ["agent.help_click"] * 3, "target": ["b"] * 3,
            "payload_json": ['{"help_count": 1}'] * 3,
            "client_ts": ["2024-01-01T00:00:01+00:00"] * 3,
            "created_at": ["2024-01-01T00:00:01+00:00"] * 3})
        view = pd.DataFrame({"group_id": ["g1", "g2"], "session_id": ["s", "s"],
                             "group_number": [1, 2], "group_name": ["A", "B"],
                             "help_click_count": reported_help})
        return Corpus(frames={"groups": groups, "operational": ops, "view": view},
                      source="synthetic")

    def test_a_matching_column_is_classified_reconciled(self):
        from metricaudit.reconcile import apply_rules
        _, by_column = apply_rules(self._corpus([2, 1]))
        row = by_column.set_index("indicator").loc["help_click_count"]
        assert row["classification"] == "reconciled"
        assert row["n_diverging"] == 0

    def test_one_disagreeing_group_makes_the_column_divergent(self):
        from metricaudit.reconcile import apply_rules
        by_group, by_column = apply_rules(self._corpus([5, 1]))
        row = by_column.set_index("indicator").loc["help_click_count"]
        assert row["classification"] == "divergent"
        assert row["n_diverging"] == 1

        bad = by_group[(by_group["indicator"] == "help_click_count")
                       & (~by_group["matches"])].iloc[0]
        assert bad["reported"] == 5
        assert bad["recomputed"] == 2
        assert bad["ratio"] == pytest.approx(2.5)
        assert bad["direction"] == "reported higher"

    def test_a_column_absent_from_the_view_is_unverifiable_not_missing(self):
        from metricaudit.reconcile import apply_rules
        _, by_column = apply_rules(self._corpus([2, 1]))
        row = by_column.set_index("indicator").loc["quiz_wrong_count"]
        assert row["classification"] == "unverifiable"
        assert "absent" in row["reason"]
