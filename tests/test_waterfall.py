"""Tests for hmda.clean.waterfall."""

from __future__ import annotations

import pandas as pd
import pytest

from hmda.clean import load_fixture
from hmda.clean.filters import FILTER_REGISTRY
from hmda.clean.waterfall import build_waterfall, race_not_available_share


@pytest.fixture(scope="module")
def df():
    return load_fixture()


def test_waterfall_first_row_equals_raw_fixture_row_count(df):
    rows = build_waterfall(df, ["exclude_purchased_loans"])
    assert rows[0].step == "raw"
    assert rows[0].rows_before == rows[0].rows_after == 50000


def test_waterfall_last_row_equals_analysis_set_count(df):
    """Measured 2026-09-11: action_taken == 6 count is 5,700 of 50,000, so
    the analysis-set count after excluding purchased loans is 44,300."""
    rows = build_waterfall(df, ["exclude_purchased_loans"])
    assert rows[-1].rows_after == 44300


def test_waterfall_row_count_equals_filters_plus_one(df):
    filter_names = list(FILTER_REGISTRY.keys())
    rows = build_waterfall(df, filter_names)
    # one "raw" row, plus one row per registered filter
    assert len(rows) == len(filter_names) + 1


def test_every_filter_row_names_its_column_and_rule(df):
    filter_names = list(FILTER_REGISTRY.keys())
    rows = build_waterfall(df, filter_names)
    for row in rows[1:]:
        spec = FILTER_REGISTRY[row.step]
        assert spec.column in row.note
        assert spec.rule in row.note


def test_unknown_filter_name_raises_keyerror(df):
    with pytest.raises(KeyError):
        build_waterfall(df, ["this_filter_does_not_exist"])


def test_rows_dropped_is_never_negative(df):
    rows = build_waterfall(df, list(FILTER_REGISTRY.keys()))
    for row in rows:
        assert row.rows_dropped >= 0
        assert row.rows_dropped == row.rows_before - row.rows_after


def test_race_not_available_share_matches_measured_fixture_value(df):
    """Measured 2026-09-11: 'Race Not Available' is 14,284 of 50,000 rows
    (28.568%) in the fixture, the largest single derived_race category."""
    share = race_not_available_share(df)
    assert share == pytest.approx(14284 / 50000, abs=1e-9)


def test_race_not_available_share_on_empty_frame_is_zero():
    empty = pd.DataFrame({"derived_race": pd.Series([], dtype="string")})
    assert race_not_available_share(empty) == 0.0
