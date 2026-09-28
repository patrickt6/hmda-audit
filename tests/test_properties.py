"""Property-based tests (Hypothesis) for the four-fifths screen, the denial
rates and the empirical-Bayes watch list.

Each property below is checked against the contract the code documents:

- hmda.fairness.air: reference group = highest approval rate among the
  groups that clear the floor; a group below the floor is kept with ratio
  "SUPPRESSED" and never dropped; a reference approval rate of 0 gives ratio
  "UNDEFINED" (0/0) and no flag; flagged means a numeric ratio below 0.8.
- hmda.fairness.rates: purchased loans (action_taken 6) never count;
  denial_rate = denials / applications.
- hmda.fairness.shrink: normal-normal empirical Bayes; posterior mean
  w*g + (1-w)*mu with w in [0, 1]; lenders under min_each are not scored;
  level must be strictly between 0 and 1.

Every test runs with derandomize=True and no example database, so CI sees
the same examples on every run. Example counts are sized to keep this file
well under a minute.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from hmda.fairness.air import FOUR_FIFTHS_THRESHOLD, _air_rows_from_raw, adverse_impact_ratio
from hmda.fairness.floors import below_floor
from hmda.fairness.rates import denial_rate_by_group
from hmda.fairness.shrink import shrink_lender_gaps

FAST = settings(max_examples=300, derandomize=True, database=None, deadline=None)
# Anything that opens a DuckDB connection per example runs fewer examples.
DUCK = settings(max_examples=40, derandomize=True, database=None, deadline=None,
                suppress_health_check=[HealthCheck.too_slow])

GROUPS = ["A", "B", "C", "D", "E", "F"]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def raw_group_rows(draw, min_groups=1):
    """Rows shaped like sql/air_by_group.sql output: one per distinct group."""
    names = draw(st.lists(st.sampled_from(GROUPS), min_size=min_groups, max_size=len(GROUPS), unique=True))
    rows = []
    for g in names:
        n = draw(st.integers(min_value=1, max_value=400))
        d = draw(st.integers(min_value=0, max_value=n))
        rows.append({"group_value": g, "denominator": n, "denials": d, "approval_rate": 1.0 - d / n})
    return rows


@st.composite
def application_frames(draw):
    """A small raw LAR-like frame: action_taken (text, as in the parquet), a
    protected-class column, and one lender."""
    n = draw(st.integers(min_value=1, max_value=60))
    actions = draw(st.lists(st.sampled_from(["1", "2", "3", "4", "5", "6", "7", "8"]), min_size=n, max_size=n))
    groups = draw(st.lists(st.sampled_from(GROUPS[:4]), min_size=n, max_size=n))
    return pd.DataFrame({"action_taken": actions, "derived_sex": groups, "lei": ["L1"] * n})


@st.composite
def lender_rows(draw, min_lenders=3):
    """Rows shaped like sql/air_by_lei_group.sql output for two groups B and W,
    unique per (lei, group). The case where BOTH pooled rates are 0 or 1 is
    excluded here and tested on its own below (it must raise ValueError)."""
    k = draw(st.integers(min_value=min_lenders, max_value=12))
    rows = []
    for i in range(k):
        for g in ("B", "W"):
            n = draw(st.integers(min_value=5, max_value=500))
            d = draw(st.integers(min_value=0, max_value=n))
            rows.append({"lei": f"L{i}", "group_value": g, "denominator": n, "denials": d})
    assume(_pooled_variance(rows) > 0)
    return rows


def _pooled_variance(rows):
    total = 0.0
    for g in ("B", "W"):
        p = sum(r["denials"] for r in rows if r["group_value"] == g) / \
            sum(r["denominator"] for r in rows if r["group_value"] == g)
        total += p * (1 - p)
    return total


def _by_group(air_rows):
    return {r.group_value: (r.denominator, r.ratio, r.flagged) for r in air_rows}


# ---------------------------------------------------------------------------
# air.py: the four-fifths screen
# ---------------------------------------------------------------------------

@FAST
@given(raw_group_rows(), st.integers(min_value=1, max_value=300), st.randoms(use_true_random=False))
def test_air_row_order_does_not_change_ratios_or_flags(rows, min_count, rnd):
    shuffled = list(rows)
    rnd.shuffle(shuffled)
    a = _by_group(_air_rows_from_raw(rows, "derived_sex", "L1", min_count))
    b = _by_group(_air_rows_from_raw(shuffled, "derived_sex", "L1", min_count))
    # reference_group_value is left out on purpose: max() breaks ties by
    # position, so with two equal top rates the label can differ. The ratio
    # does not, because both candidates have the same rate.
    assert a == b


@FAST
@given(raw_group_rows(), st.integers(min_value=1, max_value=300))
def test_air_groups_below_the_floor_are_suppressed_not_dropped(rows, min_count):
    out = _air_rows_from_raw(rows, "derived_sex", "L1", min_count)
    assert sorted(r.group_value for r in out) == sorted(r["group_value"] for r in rows)
    for r in out:
        if below_floor(r.denominator, min_count):
            assert r.ratio == "SUPPRESSED" and r.flagged is False


@FAST
@given(raw_group_rows(), st.integers(min_value=1, max_value=300))
def test_air_ratios_lie_in_zero_one_and_flag_means_below_threshold(rows, min_count):
    for r in _air_rows_from_raw(rows, "derived_sex", "L1", min_count):
        if r.ratio in ("SUPPRESSED", "UNDEFINED"):
            assert r.flagged is False
        else:
            # The reference is the maximum rate among eligible groups, and
            # only eligible groups get a number, so no ratio exceeds 1.
            assert 0.0 <= r.ratio <= 1.0 + 1e-12
            assert r.flagged == (r.ratio < FOUR_FIFTHS_THRESHOLD)


@FAST
@given(raw_group_rows(), st.integers(min_value=1, max_value=300))
def test_air_reference_group_ratio_is_one_when_its_approval_rate_is_positive(rows, min_count):
    out = _air_rows_from_raw(rows, "derived_sex", "L1", min_count)
    eligible = [r for r in rows if not below_floor(r["denominator"], min_count)]
    assume(eligible and max(r["approval_rate"] for r in eligible) > 0)
    ref = [r for r in out if r.group_value == r.reference_group_value]
    assert len(ref) == 1 and ref[0].ratio == 1.0 and not ref[0].flagged


@FAST
@given(raw_group_rows(), st.integers(min_value=1, max_value=300))
def test_air_reference_group_is_never_flagged_against_itself(rows, min_count):
    # This was a strict xfail until 2026-09-28. The property and
    # scripts/air_independent.py both found that when every floor-clearing
    # group had approval rate 0, air.py returned ratio 0.0 and flagged the
    # reference group against itself. A 0/0 ratio is now "UNDEFINED".
    out = _air_rows_from_raw(rows, "derived_sex", "L1", min_count)
    eligible = [r for r in rows if not below_floor(r["denominator"], min_count)]
    top = max((r["approval_rate"] for r in eligible), default=None)
    for r in out:
        if r.group_value == r.reference_group_value:
            assert not r.flagged
            assert r.ratio == ("UNDEFINED" if top == 0 else 1.0)


@FAST
@given(raw_group_rows(), st.integers(min_value=1, max_value=300))
def test_air_zero_reference_rate_makes_every_eligible_ratio_undefined(rows, min_count):
    # Deny every application in every floor-clearing group, so the
    # reference approval rate is 0. Groups under the floor keep their rates.
    rows = [dict(r, denials=r["denominator"], approval_rate=0.0)
            if not below_floor(r["denominator"], min_count) else r for r in rows]
    assume(any(not below_floor(r["denominator"], min_count) for r in rows))
    out = _air_rows_from_raw(rows, "derived_sex", "L1", min_count)
    for r in out:
        expected = "SUPPRESSED" if below_floor(r.denominator, min_count) else "UNDEFINED"
        assert r.ratio == expected and r.flagged is False


@FAST
@given(st.integers(min_value=1, max_value=400), st.data(), st.integers(min_value=1, max_value=300))
def test_air_two_identical_groups_give_ratio_one(n, data, min_count):
    d = data.draw(st.integers(min_value=0, max_value=n - 1))  # approval rate > 0
    row = {"denominator": n, "denials": d, "approval_rate": 1.0 - d / n}
    rows = [{"group_value": "A", **row}, {"group_value": "B", **row}]
    out = _air_rows_from_raw(rows, "derived_sex", "L1", min_count)
    if below_floor(n, min_count):
        assert all(r.ratio == "SUPPRESSED" for r in out)
    else:
        assert [r.ratio for r in out] == [1.0, 1.0]


@DUCK
@given(application_frames())
def test_air_through_duckdb_matches_a_hand_count_and_ignores_row_order(frame):
    kept = frame[frame["action_taken"] != "6"]
    assume(len(kept) > 0)
    min_count = 3
    out = adverse_impact_ratio(frame, "derived_sex", min_count, lei="L1")
    shuffled = adverse_impact_ratio(frame.sample(frac=1, random_state=0), "derived_sex", min_count, lei="L1")
    assert _by_group(out) == _by_group(shuffled)

    counts = kept.groupby("derived_sex")["action_taken"].agg(n="size", d=lambda s: int((s == "3").sum()))
    assert {r.group_value: r.denominator for r in out} == counts["n"].to_dict()
    rates = {g: 1 - c.d / c.n for g, c in counts.iterrows() if c.n >= min_count}
    for r in out:
        if r.ratio != "SUPPRESSED" and max(rates.values()) > 0:
            assert r.ratio == pytest.approx(rates[r.group_value] / max(rates.values()), abs=1e-12)


# ---------------------------------------------------------------------------
# rates.py: denial rates
# ---------------------------------------------------------------------------

@DUCK
@given(application_frames(), st.integers(min_value=0, max_value=10))
def test_rates_purchased_loans_never_change_the_result(frame, extra):
    base = denial_rate_by_group(frame, "derived_sex")
    padded = pd.concat([frame, pd.DataFrame({"action_taken": ["6"] * extra,
                                             "derived_sex": ["A"] * extra, "lei": ["L1"] * extra})])
    with_purchased = denial_rate_by_group(padded, "derived_sex")
    # A group made only of purchased loans must not appear at all.
    assert [(r.group_value, r.applications, r.denials) for r in base] == \
        [(r.group_value, r.applications, r.denials) for r in with_purchased]


@DUCK
@given(application_frames())
def test_rates_counts_add_up_and_rates_match_counts(frame):
    rows = denial_rate_by_group(frame, "derived_sex")
    kept = frame[frame["action_taken"] != "6"]
    assert sum(r.applications for r in rows) == len(kept)
    assert sum(r.denials for r in rows) == int((kept["action_taken"] == "3").sum())
    for r in rows:
        assert 0 <= r.denials <= r.applications
        assert r.denial_rate == r.denials / r.applications


# ---------------------------------------------------------------------------
# shrink.py: the empirical-Bayes watch list
# ---------------------------------------------------------------------------

@FAST
@given(lender_rows(), st.floats(min_value=0.01, max_value=0.99))
def test_shrink_weights_posteriors_and_probabilities_are_in_range(rows, level):
    res = shrink_lender_gaps(rows, "B", "W", min_each=5, level=level)
    assert res.tau_pp >= 0
    for l in res.lenders:
        assert 0.0 <= l.weight <= 1.0
        assert 0.0 <= l.p_above_typical <= 1.0
        lo, hi = sorted((l.raw_gap_pp, res.mu_pp))
        assert lo - 1e-9 <= l.shrunk_gap_pp <= hi + 1e-9
        assert l.on_watch_list == (l.p_above_typical >= level)
        assert math.isfinite(l.se_pp) and l.se_pp > 0


@FAST
@given(lender_rows(), st.integers(min_value=1, max_value=300))
def test_shrink_lenders_below_min_each_are_never_scored(rows, min_each):
    cells = {(r["lei"], r["group_value"]): r["denominator"] for r in rows}
    eligible = {lei for lei, _ in cells if cells[(lei, "B")] >= min_each and cells[(lei, "W")] >= min_each}
    assume(len(eligible) >= 3)
    assume(_pooled_variance([r for r in rows if r["lei"] in eligible]) > 0)
    res = shrink_lender_gaps(rows, "B", "W", min_each=min_each)
    assert {l.lei for l in res.lenders} == eligible
    assert res.lenders_scored == len(eligible)


@FAST
@given(lender_rows(), st.randoms(use_true_random=False))
def test_shrink_row_order_does_not_change_scores(rows, rnd):
    shuffled = list(rows)
    rnd.shuffle(shuffled)
    a = {l.lei: l for l in shrink_lender_gaps(rows, "B", "W").lenders}
    b = {l.lei: l for l in shrink_lender_gaps(shuffled, "B", "W").lenders}
    assert a.keys() == b.keys()
    for lei in a:
        # Float sums in a different order can differ in the last bits.
        assert a[lei].shrunk_gap_pp == pytest.approx(b[lei].shrunk_gap_pp, rel=1e-9, abs=1e-9)
        assert a[lei].p_above_typical == pytest.approx(b[lei].p_above_typical, rel=1e-9, abs=1e-9)


@FAST
@given(lender_rows(), st.one_of(st.floats(max_value=0.0), st.floats(min_value=1.0)))
def test_shrink_invalid_level_raises(rows, level):
    with pytest.raises(ValueError):
        shrink_lender_gaps(rows, "B", "W", level=level)


@FAST
@given(st.integers(min_value=3, max_value=12), st.data())
def test_shrink_degenerate_pooled_rates_raise_value_error(k, data):
    # Found by Hypothesis: when every B and every W application was denied
    # (or none was), each lender's variance is 0 and shrink_lender_gaps used
    # to crash with ZeroDivisionError. It now raises a ValueError that says why.
    all_b = data.draw(st.booleans())
    all_w = data.draw(st.booleans())
    rows = []
    for i in range(k):
        for g, denied_all in (("B", all_b), ("W", all_w)):
            n = data.draw(st.integers(min_value=5, max_value=500))
            rows.append({"lei": f"L{i}", "group_value": g, "denominator": n, "denials": n if denied_all else 0})
    with pytest.raises(ValueError, match="no sampling variance"):
        shrink_lender_gaps(rows, "B", "W")
