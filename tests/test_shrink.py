"""Empirical-Bayes lender watch list (hmda.fairness.shrink)."""

import pytest

from hmda.fairness.shrink import shrink_lender_gaps


def _rows(spec):
    """spec: {lei: (n_group, denials_group, n_ref, denials_ref)}"""
    out = []
    for lei, (ng, dg, nr, dr) in spec.items():
        out.append({"lei": lei, "group_value": "B", "denominator": ng, "denials": dg})
        out.append({"lei": lei, "group_value": "W", "denominator": nr, "denials": dr})
    return out


BASE = {f"L{i}": (1000, 150 + (i % 5) * 10, 1000, 100) for i in range(20)}


def test_small_lender_is_shrunk_harder_than_large_lender_with_same_raw_gap():
    spec = dict(BASE)
    spec["SMALL"] = (10, 6, 10, 1)        # raw gap 50 pp on 10 + 10 applications
    spec["LARGE"] = (5000, 3000, 5000, 500)  # raw gap 50 pp on 5000 + 5000
    res = shrink_lender_gaps(_rows(spec), "B", "W")
    by = {l.lei: l for l in res.lenders}
    assert by["SMALL"].raw_gap_pp == pytest.approx(by["LARGE"].raw_gap_pp)
    assert by["SMALL"].weight < by["LARGE"].weight
    assert by["SMALL"].shrunk_gap_pp < by["LARGE"].shrunk_gap_pp


def test_same_raw_gap_flags_the_large_lender_but_not_the_small_one():
    # Each outlier is scored against the ordinary population on its own, so it
    # cannot inflate the between-lender spread that the other is judged by.
    small = shrink_lender_gaps(_rows({**BASE, "SMALL": (10, 6, 10, 1)}), "B", "W")
    large = shrink_lender_gaps(_rows({**BASE, "LARGE": (5000, 3000, 5000, 500)}), "B", "W")
    assert not next(l for l in small.lenders if l.lei == "SMALL").on_watch_list
    assert next(l for l in large.lenders if l.lei == "LARGE").on_watch_list


def test_zero_denial_lender_does_not_get_zero_variance():
    spec = dict(BASE)
    spec["ZERO"] = (8, 0, 8, 0)
    res = shrink_lender_gaps(_rows(spec), "B", "W")
    zero = next(l for l in res.lenders if l.lei == "ZERO")
    assert zero.se_pp > 0


def test_lenders_missing_a_group_or_below_floor_are_not_scored():
    spec = dict(BASE)
    rows = _rows(spec) + [{"lei": "ONLY_W", "group_value": "W", "denominator": 500, "denials": 50}]
    rows += _rows({"TINY": (2, 1, 400, 40)})
    res = shrink_lender_gaps(rows, "B", "W", min_each=5)
    scored = {l.lei for l in res.lenders}
    assert "ONLY_W" not in scored and "TINY" not in scored
    assert res.lenders_scored == 20


def test_identical_lenders_give_zero_spread_and_empty_watch_list():
    spec = {f"L{i}": (1000, 150, 1000, 100) for i in range(10)}
    res = shrink_lender_gaps(_rows(spec), "B", "W")
    assert res.tau_pp == 0.0
    assert res.mu_pp == pytest.approx(5.0)
    assert res.watch_list == ()


def test_too_few_lenders_is_an_error():
    with pytest.raises(ValueError):
        shrink_lender_gaps(_rows({"A": (100, 10, 100, 5), "B2": (100, 10, 100, 5)}), "B", "W")
