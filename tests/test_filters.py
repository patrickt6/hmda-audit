"""Tests for hmda.clean.filters."""

from __future__ import annotations

import pandas as pd
import pytest

from hmda.clean import load_fixture
from hmda.clean.filters import (
    FILTER_REGISTRY,
    FilterSpec,
    exclude_purchased_loans,
    register_filter,
)


@pytest.fixture(scope="module")
def df():
    return load_fixture()


def test_exclude_purchased_loans_drops_action_taken_6(df):
    """Measured 2026-09-11: action_taken == 6 (purchased loan) count is
    5,700 of 50,000 rows in the fixture."""
    before = len(df)
    out = exclude_purchased_loans(df)
    dropped = before - len(out)
    assert dropped == 5700
    assert (out["action_taken"].astype("string") != "6").all()


def test_exclude_purchased_loans_is_registered_with_column_and_rule():
    assert "exclude_purchased_loans" in FILTER_REGISTRY
    spec = FILTER_REGISTRY["exclude_purchased_loans"]
    assert spec.column == "action_taken"
    assert spec.rule  # a non-empty human description


def test_register_filter_rejects_duplicate_name():
    dummy = FilterSpec(
        name="exclude_purchased_loans",
        column="action_taken",
        rule="duplicate, must be rejected",
        apply=lambda frame: frame,
    )
    with pytest.raises(ValueError):
        register_filter(dummy)


def test_register_filter_adds_a_new_named_filter():
    spec = FilterSpec(
        name="__test_only_filter__",
        column="income",
        rule="test-only: keep everything",
        apply=lambda frame: frame,
    )
    try:
        register_filter(spec)
        assert FILTER_REGISTRY["__test_only_filter__"] is spec
    finally:
        FILTER_REGISTRY.pop("__test_only_filter__", None)
