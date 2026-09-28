"""Tests for hmda.fairness.rates (M5).

Runs against the committed 50k-row fixture (tests/fixtures/hmda_50k.parquet)
only. Every number here is a fixture-level number, not a national one.
"""

from __future__ import annotations

import pandas as pd

from hmda.fairness.rates import ALLOWED_GROUP_COLUMNS, denial_rate_by_group


def test_denial_rate_by_group_excludes_purchased_loans_action_taken_6():
    """action_taken == 6 is a purchased loan, not an
    application, and must never enter a denial rate's numerator or
    denominator."""
    frame = pd.DataFrame(
        {
            "action_taken": ["3", "6", "1"],  # denied, purchased (excluded), approved
            "lei": ["L1", "L1", "L1"],
            "derived_sex": ["Male", "Male", "Male"],
        }
    )
    rows = denial_rate_by_group(frame, "derived_sex")
    assert len(rows) == 1
    row = rows[0]
    # Only the denied row and the approved row count; the purchased loan
    # (action_taken == 6) is excluded from both denials and applications.
    assert row.applications == 2
    assert row.denials == 1
    assert row.denial_rate == 0.5


def test_denial_rate_by_group_hand_computed_two_groups():
    """A hand-computed case: Group A has 1 denial of 4 applications (0.25);
    Group B has 3 denials of 5 applications (0.60)."""
    frame = pd.DataFrame(
        {
            "action_taken": ["3", "1", "1", "1", "3", "3", "3", "1", "1"],
            "lei": ["L1"] * 9,
            "derived_race": ["A"] * 4 + ["B"] * 5,
        }
    )
    rows = {r.group_value: r for r in denial_rate_by_group(frame, "derived_race")}
    assert rows["A"].applications == 4
    assert rows["A"].denials == 1
    assert rows["A"].denial_rate == 0.25
    assert rows["B"].applications == 5
    assert rows["B"].denials == 3
    assert rows["B"].denial_rate == 0.6


def test_denial_rate_by_group_lei_scope_restricts_to_that_lender():
    frame = pd.DataFrame(
        {
            "action_taken": ["3", "1", "3", "1"],
            "lei": ["L1", "L1", "L2", "L2"],
            "derived_sex": ["Male", "Male", "Female", "Female"],
        }
    )
    rows = denial_rate_by_group(frame, "derived_sex", lei="L1")
    assert len(rows) == 1
    assert rows[0].applications == 2
    assert rows[0].denials == 1
    assert rows[0].lei == "L1"


def test_group_column_not_in_allowlist_raises():
    frame = pd.DataFrame({"action_taken": ["1"], "lei": ["L1"], "not_a_real_column": ["x"]})
    try:
        denial_rate_by_group(frame, "not_a_real_column")
        assert False, "expected ValueError for a group_column outside ALLOWED_GROUP_COLUMNS"
    except ValueError:
        pass


def test_allowed_group_columns_matches_protected_class_fields():
    assert ALLOWED_GROUP_COLUMNS == {"derived_race", "derived_ethnicity", "derived_sex"}


def test_denial_rate_by_group_on_the_fixture_national_scope():
    """Sanity check against the real 50k-row fixture: every row's
    denial_rate is a legitimate ratio and applications sums to the fixture's
    row count minus purchased loans (action_taken == 6)."""
    from hmda.clean import load_fixture

    frame = load_fixture()
    rows = denial_rate_by_group(frame, "derived_race")
    total_applications = sum(r.applications for r in rows)
    non_purchased = (frame["action_taken"] != "6").sum()
    assert total_applications == non_purchased
    for r in rows:
        assert 0.0 <= r.denial_rate <= 1.0
        assert r.denials <= r.applications
