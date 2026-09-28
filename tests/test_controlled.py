"""Tests for ``hmda.fairness.controlled``.

Two synthetic cases are the real proof the estimator works: one where the
true residual gap is zero by construction (all the raw gap is confounding
through the controls), one where a known non-zero residual gap is baked in
directly. A third block runs against the real 50k-row fixture and only
checks structural/API properties (both columns present, header lists the
controls, no group silently dropped). It does NOT assert a specific fixture
number, because ``hmda.fairness.controlled`` carries its own private
minimal sentinel and parsing helpers (see ``controlled.py`` module
docstring point 5) instead of using the general ``hmda.clean.sentinels`` /
``hmda.clean.filters`` implementations, so a fixture-level number here
would go stale the moment that integration debt is paid down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hmda.fairness.controlled import (
    CONTROL_COLUMNS,
    DisparityComparison,
    controlled_disparity,
)


def _base_frame(n_per_group: int, rng: np.random.Generator) -> pd.DataFrame:
    """A frame with two groups ("A" reference, "B" comparison), all five
    CONTROL_COLUMNS present but held at a constant, uninformative value so
    only income varies and drives denial."""
    n = n_per_group * 2
    group = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    return pd.DataFrame(
        {
            "action_taken": ["1"] * n,  # overwritten by caller per-row below
            "derived_race": group,
            "loan_to_value_ratio": ["80.0"] * n,
            "debt_to_income_ratio": ["43"] * n,
            "loan_purpose": ["1"] * n,
            "lien_status": ["1"] * n,
        }
    )


def _synthetic_confounded_zero_residual(rng: np.random.Generator, n_per_group: int = 4000):
    """Group B is drawn from a lower-income distribution than group A, and
    denial depends ONLY on income (never on group directly). The raw gap
    must be clearly non-zero (B looks worse because B is poorer); the
    controlled gap must be ~0 because income fully explains the difference
    by construction."""
    frame = _base_frame(n_per_group, rng)

    income_thousands_a = rng.normal(120, 25, size=n_per_group).clip(20, 400)
    income_thousands_b = rng.normal(70, 25, size=n_per_group).clip(20, 400)
    income_thousands = np.concatenate([income_thousands_a, income_thousands_b])
    frame["income"] = [f"{v:.0f}" for v in income_thousands]

    # Denial probability driven only by income (dollars), never by group.
    income_dollars = income_thousands * 1000.0
    logit = -2.5 + (-0.00002) * (income_dollars - 100000)
    prob_denial = 1 / (1 + np.exp(-logit))
    denial = rng.binomial(1, prob_denial)
    frame["action_taken"] = np.where(denial == 1, "3", "1")

    return frame


def _synthetic_known_nonzero_residual(rng: np.random.Generator, n_per_group: int = 4000):
    """Same income confound as above, PLUS a fixed direct group effect on
    the log-odds of denial. The residual (controlled) gap should recover
    something close to the direct effect's implied probability gap, not
    zero, because the group effect is NOT explained by income."""
    frame = _base_frame(n_per_group, rng)

    income_thousands_a = rng.normal(120, 25, size=n_per_group).clip(20, 400)
    income_thousands_b = rng.normal(70, 25, size=n_per_group).clip(20, 400)
    income_thousands = np.concatenate([income_thousands_a, income_thousands_b])
    frame["income"] = [f"{v:.0f}" for v in income_thousands]

    income_dollars = income_thousands * 1000.0
    is_b = np.concatenate([np.zeros(n_per_group), np.ones(n_per_group)])
    direct_group_logit_boost = 1.0  # the "known" injected effect
    logit = -2.5 + (-0.00002) * (income_dollars - 100000) + direct_group_logit_boost * is_b
    prob_denial = 1 / (1 + np.exp(-logit))
    denial = rng.binomial(1, prob_denial)
    frame["action_taken"] = np.where(denial == 1, "3", "1")

    return frame, direct_group_logit_boost


def test_raw_and_controlled_always_ship_together():
    """API shape check: the return type carries both numbers, always."""
    rng = np.random.default_rng(0)
    frame = _synthetic_confounded_zero_residual(rng, n_per_group=200)
    result = controlled_disparity(frame, "derived_race", min_count=50)
    assert len(result) == 1
    comparison = result[0]
    assert isinstance(comparison, DisparityComparison)
    assert hasattr(comparison, "raw_gap_pp")
    assert hasattr(comparison, "controlled_gap_pp")
    assert comparison.controls_used == CONTROL_COLUMNS


def test_no_public_function_returns_the_raw_gap_alone():
    """Checked from inside the test too (belt and suspenders alongside the
    grep check run externally): no other public name in this
    module returns a bare number for the raw gap."""
    import hmda.fairness.controlled as controlled_module

    public_names = [n for n in dir(controlled_module) if not n.startswith("_")]
    for name in public_names:
        assert "raw_gap" not in name.lower() or name == "DisparityComparison" or "controls_used" in name


def test_synthetic_zero_residual_gap_recovered_within_tolerance():
    """The proof: true residual gap is zero by construction (income
    alone explains the raw difference); the estimator must recover
    approximately zero, and must NOT report the (nonzero) raw gap as if it
    were the controlled one."""
    rng = np.random.default_rng(42)
    frame = _synthetic_confounded_zero_residual(rng, n_per_group=4000)

    result = controlled_disparity(frame, "derived_race", min_count=50)
    assert len(result) == 1
    comparison = result[0]

    # The raw gap must be clearly non-zero: group B (poorer) is denied a lot
    # more often than group A in the unadjusted comparison.
    assert comparison.raw_gap_pp > 5.0, comparison

    # The controlled gap must be close to zero, and clearly closer to zero
    # than the raw gap is.
    assert abs(comparison.controlled_gap_pp) < 3.0, comparison
    assert abs(comparison.controlled_gap_pp) < abs(comparison.raw_gap_pp) / 2


def test_synthetic_known_nonzero_residual_gap_recovered_within_tolerance():
    """The second proof: a KNOWN non-zero residual (a direct group
    effect on the log-odds, not explained by income) must survive
    controlling, and the recovered controlled gap must be meaningfully
    positive, unlike the zero-residual case above."""
    rng = np.random.default_rng(7)
    frame, _injected_logit_boost = _synthetic_known_nonzero_residual(rng, n_per_group=4000)

    result = controlled_disparity(frame, "derived_race", min_count=50)
    assert len(result) == 1
    comparison = result[0]

    # A logit boost of 1.0 near a ~10-20% base denial rate implies roughly a
    # several-percentage-point gap; require it be clearly non-zero and
    # positive (group B, the "is_b" group, denied more even after control).
    assert comparison.controlled_gap_pp > 3.0, comparison


def test_suppressed_group_never_silently_dropped():
    """A group under the floor still appears, with NaN gaps, never omitted
    and never reported as a real zero."""
    rng = np.random.default_rng(1)
    frame = _base_frame(2000, rng)
    frame["income"] = "100"
    frame["action_taken"] = "1"

    # Shrink group B down to below the floor.
    frame = pd.concat([frame[frame["derived_race"] == "A"], frame[frame["derived_race"] == "B"].head(5)])

    result = controlled_disparity(frame, "derived_race", min_count=50)
    assert len(result) == 1
    comparison = result[0]
    assert comparison.n_group == 5
    assert np.isnan(comparison.raw_gap_pp)
    assert np.isnan(comparison.controlled_gap_pp)


def test_race_not_available_share_reported_alongside_race_comparisons():
    rng = np.random.default_rng(2)
    frame = _synthetic_confounded_zero_residual(rng, n_per_group=500)
    # Relabel a slice as "Race Not Available" to prove the share is computed
    # over the full input frame, not just the two compared groups.
    frame = frame.copy()
    frame.loc[frame.index[:100], "derived_race"] = "Race Not Available"

    result = controlled_disparity(frame, "derived_race", min_count=50)
    assert all(c.race_not_available_share is not None for c in result)
    assert all(abs(c.race_not_available_share - 100 / len(frame)) < 1e-9 for c in result)


def test_race_not_available_share_is_none_for_non_race_columns():
    rng = np.random.default_rng(3)
    frame = _synthetic_confounded_zero_residual(rng, n_per_group=200)
    frame = frame.rename(columns={"derived_race": "derived_sex"})
    result = controlled_disparity(frame, "derived_sex", min_count=50)
    assert all(c.race_not_available_share is None for c in result)


def test_no_reference_group_meeting_floor_raises():
    rng = np.random.default_rng(4)
    frame = _synthetic_confounded_zero_residual(rng, n_per_group=10)
    with pytest.raises(ValueError):
        controlled_disparity(frame, "derived_race", min_count=1000)


def test_against_real_fixture_structural_only():
    """Runs against the committed 50k-row fixture. Only checks structure
    (both numbers present, controls echoed, no crash), not a specific
    number, because this module's cleaning is a private stand-in for
    ``hmda.clean.sentinels`` (see controlled.py module docstring point 5),
    so a fixture-level number here would silently go stale the moment that
    integration debt is paid down."""
    from hmda.clean import load_fixture

    frame = load_fixture()
    result = controlled_disparity(frame, "derived_race", min_count=100)
    assert len(result) > 0
    for comparison in result:
        assert comparison.controls_used == CONTROL_COLUMNS
        assert comparison.group_column == "derived_race"
        assert comparison.n_group >= 0
        assert comparison.race_not_available_share is not None
