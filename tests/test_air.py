"""Tests for hmda.fairness.air: the four-fifths adverse impact ratio (M5).

Runs against the committed 50k-row fixture only. Every number here is a
fixture-level number, not a national one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hmda.fairness.air import FOUR_FIFTHS_THRESHOLD, adverse_impact_ratio, flagged_lenders
from hmda.fairness.floors import DEFAULT_MIN_COUNT, below_floor


def _make_frame():
    # Group A: 8 applications, 2 denials -> approval rate 6/8 = 0.75
    # Group B: 10 applications, 1 denial -> approval rate 9/10 = 0.90 (reference: highest)
    # Group C: 5 applications, 4 denials -> approval rate 1/5 = 0.20
    # Group D: 1 application, 0 denials -> below the min_count=2 floor -> SUPPRESSED
    rows = []
    rows += [{"action_taken": "3", "derived_sex": "A"}] * 2
    rows += [{"action_taken": "1", "derived_sex": "A"}] * 6
    rows += [{"action_taken": "3", "derived_sex": "B"}] * 1
    rows += [{"action_taken": "1", "derived_sex": "B"}] * 9
    rows += [{"action_taken": "3", "derived_sex": "C"}] * 4
    rows += [{"action_taken": "1", "derived_sex": "C"}] * 1
    rows += [{"action_taken": "1", "derived_sex": "D"}] * 1
    frame = pd.DataFrame(rows)
    frame["lei"] = "L1"
    return frame


def test_air_hand_computed_ratio_against_the_reference_group():
    """HAND-COMPUTED expected ratio, written into this test by hand:
    reference group B has approval rate 9/10 = 0.90,
    group C has approval rate 1/5 = 0.20, so
    ratio(C) = 0.20 / 0.90 = 0.2222... (2/9 exactly)."""
    frame = _make_frame()
    rows = {r.group_value: r for r in adverse_impact_ratio(frame, "derived_sex", min_count=2)}

    assert rows["B"].ratio == 1.0
    assert rows["B"].reference_group_value == "B"

    assert rows["C"].ratio == pytest.approx(2 / 9)
    assert rows["C"].reference_group_value == "B"
    assert rows["C"].flagged is True  # 0.2222 < 0.80

    # HAND-COMPUTED: 0.75 / 0.90 = 0.8333..., which clears the four-fifths
    # line (>= 0.80), so A is not flagged.
    assert rows["A"].ratio == pytest.approx(0.75 / 0.90)
    assert rows["A"].flagged is False


def test_air_group_below_floor_is_suppressed_not_dropped_not_a_number():
    frame = _make_frame()
    rows = {r.group_value: r for r in adverse_impact_ratio(frame, "derived_sex", min_count=2)}
    assert "D" in rows  # never silently dropped
    assert rows["D"].ratio == "SUPPRESSED"
    assert rows["D"].flagged is False
    assert rows["D"].denominator == 1


def test_air_denominator_is_never_absent():
    frame = _make_frame()
    rows = adverse_impact_ratio(frame, "derived_sex", min_count=2)
    for r in rows:
        assert r.denominator is not None
        assert r.denominator >= 0


def test_air_excludes_purchased_loans():
    frame = pd.DataFrame(
        {
            "action_taken": ["3", "6", "1", "1"],  # denial, purchased (excluded), 2 approvals
            "lei": ["L1"] * 4,
            "derived_sex": ["Male"] * 4,
        }
    )
    rows = adverse_impact_ratio(frame, "derived_sex", min_count=1)
    assert len(rows) == 1
    # 1 denial of 3 applications (purchased loan excluded) -> approval rate 2/3
    assert rows[0].denominator == 3


def test_four_fifths_threshold_is_a_named_constant():
    assert FOUR_FIFTHS_THRESHOLD == 0.80


def test_below_floor_uses_the_documented_default_when_min_count_omitted():
    assert DEFAULT_MIN_COUNT == 100
    assert below_floor(99) is True
    assert below_floor(100) is False


def test_flagged_lenders_on_the_fixture_returns_a_list_of_lei_strings():
    """Integration check against the real 50k-row fixture. This is a
    fixture-level count, not a national figure."""
    from hmda.clean import load_fixture

    frame = load_fixture()
    flagged = flagged_lenders(frame, min_count=100)
    assert isinstance(flagged, list)
    assert len(set(flagged)) == len(flagged)  # no duplicates
    all_leis = set(frame["lei"].dropna())
    for lei in flagged:
        assert lei in all_leis


def test_zero_reference_approval_rate_is_undefined_not_flagged():
    """The smallest counterexample the property tests found: one group, one
    application, one denial, min_count=1. The reference approval rate is 0,
    so the ratio is 0/0. It must be "UNDEFINED" and not flagged, not 0.0 and
    flagged against itself."""
    frame = pd.DataFrame([{"action_taken": "3", "derived_sex": "A", "lei": "L1"}])
    (row,) = adverse_impact_ratio(frame, "derived_sex", min_count=1)
    assert row.reference_group_value == "A"
    assert row.ratio == "UNDEFINED"
    assert row.flagged is False


def test_lender_that_denied_every_application_is_not_flagged():
    """The national case: one lender, every application denied, and every
    protected-class column "not available", so its only group is the
    reference group. A second lender with a real gap is still flagged."""
    rows = [{"lei": "ALLDENIED", "action_taken": "3", "derived_race": "Race Not Available",
             "derived_ethnicity": "Ethnicity Not Available", "derived_sex": "Sex Not Available"}] * 150
    rows += [{"lei": "GAP", "action_taken": "1", "derived_race": "White",
              "derived_ethnicity": "Joint", "derived_sex": "Joint"}] * 100
    rows += [{"lei": "GAP", "action_taken": "3", "derived_race": "Asian",
              "derived_ethnicity": "Joint", "derived_sex": "Joint"}] * 100
    frame = pd.DataFrame(rows)
    for column in ("derived_race", "derived_ethnicity", "derived_sex"):
        (row,) = adverse_impact_ratio(frame, column, min_count=100, lei="ALLDENIED")
        assert row.ratio == "UNDEFINED" and row.flagged is False
        assert row.lei == "ALLDENIED" and row.group_column == column
    assert flagged_lenders(frame, min_count=100) == ["GAP"]
