"""Tests that pin details of the fairness functions' contracts.

Each test here was written because a mutation-testing run (mutmut, see
docs/VERIFICATION.md) changed that detail and the rest of the suite still
passed. They check the returned fields, the error messages, the boundary of
the four-fifths threshold, and the empirical-Bayes arithmetic against a
second, hand-written DerSimonian-Laird calculation.
"""

from __future__ import annotations

from math import sqrt
from statistics import NormalDist

import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from hmda.fairness.air import _air_rows_from_raw, adverse_impact_ratio, flagged_lenders
from hmda.fairness.rates import _validate_group_column, denial_rate_by_group
from hmda.fairness.shrink import shrink_lender_gaps

SETTINGS = settings(max_examples=200, derandomize=True, database=None, deadline=None)


def _raw(group, n, d):
    return {"group_value": group, "denominator": n, "denials": d, "approval_rate": 1.0 - d / n}


# ---------------------------------------------------------------------------
# air.py
# ---------------------------------------------------------------------------

def test_air_ratio_exactly_at_the_threshold_is_not_flagged():
    # 4/5 = 0.8 exactly: 80 approvals of 100 against 100 of 100.
    rows = [_raw("A", 100, 0), _raw("B", 100, 20)]
    out = {r.group_value: r for r in _air_rows_from_raw(rows, "derived_sex", "L1", 1)}
    assert out["B"].ratio == 0.8
    assert out["B"].flagged is False


def test_air_rows_carry_their_scope_fields():
    rows = [_raw("A", 100, 0), _raw("B", 100, 50), _raw("C", 3, 0)]
    out = _air_rows_from_raw(rows, "derived_sex", "L9", 10)
    for r in out:
        assert r.lei == "L9"
        assert r.group_column == "derived_sex"
        assert r.reference_group_value == "A"


def test_air_reference_label_when_no_group_clears_the_floor():
    out = _air_rows_from_raw([_raw("A", 3, 0), _raw("B", 4, 1)], "derived_sex", "L1", 10)
    assert {r.reference_group_value for r in out} == {"NONE_ELIGIBLE"}
    assert all(r.ratio == "SUPPRESSED" for r in out)


def test_adverse_impact_ratio_restricts_to_the_requested_lender():
    # L1: A 10/10 approved, B 5/10 approved -> ratio 0.5. L2 swamps B with
    # approvals, so a national (unfiltered) answer would be different.
    rows = [("L1", "A", "1")] * 10 + [("L1", "B", "1")] * 5 + [("L1", "B", "3")] * 5
    rows += [("L2", "B", "1")] * 200
    frame = pd.DataFrame(rows, columns=["lei", "derived_sex", "action_taken"])
    out = {r.group_value: r for r in adverse_impact_ratio(frame, "derived_sex", 1, lei="L1")}
    assert out["B"].denominator == 10
    assert out["B"].ratio == pytest.approx(0.5)
    assert out["B"].lei == "L1" and out["B"].group_column == "derived_sex"


def test_flagged_lenders_keeps_scanning_after_an_already_flagged_lender():
    # "A" is flagged in every column and sorts first. B, C and D are each
    # flagged in exactly one column. Whatever order the columns are visited
    # in, a scan that stops at an already-flagged lender would lose two of them.
    def block(lei, race, eth, sex, action, n):
        return [{"lei": lei, "derived_race": race, "derived_ethnicity": eth,
                 "derived_sex": sex, "action_taken": action}] * n

    rows = []
    # A: in every column, group x approves 100%, group y approves 0%.
    rows += block("A", "x", "x", "x", "1", 10) + block("A", "y", "y", "y", "3", 10)
    # B, C, D: only one column splits approvals; the other two are one group.
    rows += block("B", "x", "x", "x", "1", 10) + block("B", "x", "y", "x", "3", 10)
    rows += block("C", "x", "x", "x", "1", 10) + block("C", "x", "x", "y", "3", 10)
    rows += block("D", "x", "x", "x", "1", 10) + block("D", "y", "x", "x", "3", 10)
    assert flagged_lenders(pd.DataFrame(rows), 5) == ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# rates.py
# ---------------------------------------------------------------------------

def test_denial_rate_rows_carry_their_scope_fields():
    frame = pd.DataFrame({"lei": ["L1", "L1"], "derived_sex": ["A", "A"], "action_taken": ["3", "1"]})
    (row,) = denial_rate_by_group(frame, "derived_sex", lei="L1")
    assert (row.lei, row.group_column, row.group_value) == ("L1", "derived_sex", "A")


def test_unknown_group_column_error_names_the_column_and_the_allowlist():
    with pytest.raises(ValueError, match=r"'lei' is not in ALLOWED_GROUP_COLUMNS \['derived_ethnicity'"):
        _validate_group_column("lei")


# ---------------------------------------------------------------------------
# shrink.py: error messages
# ---------------------------------------------------------------------------

def _pairs(spec):
    out = []
    for lei, (ng, dg, nr, dr) in spec.items():
        out.append({"lei": lei, "group_value": "B", "denominator": ng, "denials": dg})
        out.append({"lei": lei, "group_value": "W", "denominator": nr, "denials": dr})
    return out


BASE = {f"L{i}": (1000, 150 + (i % 5) * 10, 1000, 100) for i in range(10)}


@pytest.mark.parametrize("level", [0.0, 1.0])
def test_shrink_invalid_level_message(level):
    with pytest.raises(ValueError, match=r"^level must be strictly between 0 and 1$"):
        shrink_lender_gaps(_pairs(BASE), "B", "W", level=level)


def test_shrink_too_few_lenders_message():
    with pytest.raises(ValueError, match=r"^only 2 lender\(s\) have both groups; too few to estimate the spread$"):
        shrink_lender_gaps(_pairs({"L0": BASE["L0"], "L1": BASE["L1"]}), "B", "W")


def test_shrink_degenerate_rates_message():
    spec = {f"L{i}": (10, 0, 10, 0) for i in range(4)}
    with pytest.raises(ValueError, match=r"^pooled denial rates are 0 or 1 in both groups; "
                                         r"the gaps have no sampling variance$"):
        shrink_lender_gaps(_pairs(spec), "B", "W")


def test_shrink_equal_pooled_variances_do_not_raise():
    # Both groups have pooled rate 0.2, so their variance terms are equal
    # and nonzero. That is an ordinary input, not the degenerate case.
    spec = {"L0": (100, 10, 100, 30), "L1": (100, 30, 100, 10), "L2": (100, 20, 100, 20)}
    res = shrink_lender_gaps(_pairs(spec), "B", "W")
    assert res.pooled_rate_group == res.pooled_rate_reference == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# shrink.py: arithmetic against a second, hand-written calculation
# ---------------------------------------------------------------------------

@st.composite
def lender_specs(draw):
    k = draw(st.integers(min_value=3, max_value=10))
    spec = {}
    for i in range(k):
        ng = draw(st.integers(min_value=5, max_value=2000))
        nr = draw(st.integers(min_value=5, max_value=2000))
        spec[f"L{i}"] = (ng, draw(st.integers(0, ng)), nr, draw(st.integers(0, nr)))
    return spec


def _reference_dl(spec):
    """DerSimonian-Laird written out again, straight from the formulas."""
    ng_t = sum(v[0] for v in spec.values()); dg_t = sum(v[1] for v in spec.values())
    nr_t = sum(v[2] for v in spec.values()); dr_t = sum(v[3] for v in spec.values())
    pg, pr = dg_t / ng_t, dr_t / nr_t
    g = {k: 100 * (dg / ng - dr / nr) for k, (ng, dg, nr, dr) in spec.items()}
    v = {k: 10_000 * (pg * (1 - pg) / ng + pr * (1 - pr) / nr) for k, (ng, _, nr, _) in spec.items()}
    w = {k: 1 / v[k] for k in spec}
    sw = sum(w.values())
    mu_fe = sum(w[k] * g[k] for k in spec) / sw
    q = sum(w[k] * (g[k] - mu_fe) ** 2 for k in spec)
    c = sw - sum(x * x for x in w.values()) / sw
    tau2 = max(0.0, (q - (len(spec) - 1)) / c)
    mu = sum(g[k] / (v[k] + tau2) for k in spec) / sum(1 / (v[k] + tau2) for k in spec)
    return pg, pr, g, v, tau2, mu


@SETTINGS
@given(lender_specs())
def test_shrink_matches_a_hand_written_dersimonian_laird(spec):
    pg = sum(x[1] for x in spec.values()) / sum(x[0] for x in spec.values())
    pr = sum(x[3] for x in spec.values()) / sum(x[2] for x in spec.values())
    assume(pg * (1 - pg) + pr * (1 - pr) > 0)  # the degenerate case raises; tested above
    pg, pr, g, v, tau2, mu = _reference_dl(spec)
    res = shrink_lender_gaps(_pairs(spec), "B", "W", min_each=5, level=0.9)

    assert (res.group_value, res.reference_group_value, res.level) == ("B", "W", 0.9)
    assert res.pooled_rate_group == pytest.approx(pg, rel=1e-12)
    assert res.pooled_rate_reference == pytest.approx(pr, rel=1e-12)
    assert res.tau_pp == pytest.approx(sqrt(tau2), rel=1e-7, abs=1e-7)
    assert res.mu_pp == pytest.approx(mu, rel=1e-9, abs=1e-9)

    ps = [l.p_above_typical for l in res.lenders]
    assert ps == sorted(ps, reverse=True)
    for l in res.lenders:
        ng, _, nr, _ = spec[l.lei]
        assert (l.n_group, l.n_reference) == (ng, nr)
        assert l.raw_gap_pp == pytest.approx(g[l.lei], rel=1e-12, abs=1e-12)
        assert l.se_pp == pytest.approx(sqrt(v[l.lei]), rel=1e-12)
        w = tau2 / (tau2 + v[l.lei]) if tau2 > 0 else 0.0
        assert l.weight == pytest.approx(w, rel=1e-6, abs=1e-9)
        post = w * g[l.lei] + (1 - w) * mu
        assert l.shrunk_gap_pp == pytest.approx(post, rel=1e-6, abs=1e-6)
        sd = sqrt(w * v[l.lei])
        assert l.posterior_sd_pp == pytest.approx(sd, rel=1e-6, abs=1e-9)
        if sd > 1e-6:
            expected_p = 1 - NormalDist(post, sd).cdf(mu)
            assert l.p_above_typical == pytest.approx(expected_p, abs=1e-6)
        elif sd == 0.0:
            assert l.p_above_typical == 0.0


def test_shrink_p_above_typical_hand_value():
    # One number checked by hand-written formula, not by the code under test:
    # the top lender's probability equals 1 - Phi((mu - post_mean) / post_sd).
    res = shrink_lender_gaps(_pairs({**BASE, "BIG": (2000, 500, 2000, 200)}), "B", "W")
    top = res.lenders[0]
    z = (res.mu_pp - top.shrunk_gap_pp) / top.posterior_sd_pp
    assert top.p_above_typical == pytest.approx(1 - NormalDist().cdf(z), rel=1e-12)
    assert 0.0 < top.weight < 1.0


def test_shrink_watch_list_includes_a_lender_exactly_at_the_level():
    rows = _pairs({**BASE, "BIG": (2000, 250, 2000, 200)})
    first = shrink_lender_gaps(rows, "B", "W")
    inside = [l for l in first.lenders if 0.0 < l.p_above_typical < 1.0]
    assert inside
    level = inside[0].p_above_typical
    again = shrink_lender_gaps(rows, "B", "W", level=level)
    same = next(l for l in again.lenders if l.lei == inside[0].lei)
    assert same.p_above_typical == level and same.on_watch_list is True


def test_shrink_zero_spread_gives_zero_weight_even_when_gaps_differ():
    # Noise-sized differences: tau^2 comes out 0, so every lender is pulled
    # all the way to mu (weight 0) and no one is on the watch list.
    spec = {"L0": (50, 10, 50, 10), "L1": (50, 11, 50, 10), "L2": (50, 9, 50, 10), "L3": (50, 10, 50, 11)}
    res = shrink_lender_gaps(_pairs(spec), "B", "W")
    assert res.tau_pp == 0.0
    for l in res.lenders:
        assert l.weight == 0.0
        assert l.shrunk_gap_pp == pytest.approx(res.mu_pp)
        assert l.posterior_sd_pp == 0.0 and l.p_above_typical == 0.0
    assert res.watch_list == ()


def test_shrink_small_positive_spread_still_shrinks():
    # tau^2 strictly between 0 and 1 (pp^2): weights must be positive.
    spec = {f"L{i}": (20000, 4000 + d, 20000, 4000) for i, d in enumerate([0, 60, 120, 180, 240, 300])}
    res = shrink_lender_gaps(_pairs(spec), "B", "W")
    assert 0.0 < res.tau_pp ** 2 < 1.0
    assert all(l.weight > 0.0 for l in res.lenders)


def test_load_sql_rejects_a_file_that_is_not_a_query(tmp_path, monkeypatch):
    import hmda.fairness.rates as rates

    (tmp_path / "notes.sql").write_text("-- just a comment\n")
    monkeypatch.setattr(rates, "_SQL_DIR", tmp_path)
    with pytest.raises(ValueError, match=r"notes\.sql does not look like a SQL query$"):
        rates._load_sql("notes.sql")


def test_controlled_raw_gap_has_the_right_sign_and_size():
    # A: 10 of 100 denied. B: 40 of 100 denied, plus 50 purchased loans that
    # must not count. Raw gap = 40% - 10% = +30 percentage points.
    from hmda.fairness.controlled import controlled_disparity

    def rows(group, action, n):
        return [{"derived_race": group, "action_taken": action, "income": "100",
                 "loan_to_value_ratio": "80", "debt_to_income_ratio": "36",
                 "loan_purpose": "1", "lien_status": "1"}] * n

    frame = pd.DataFrame(rows("A", "3", 10) + rows("A", "1", 90)
                         + rows("B", "3", 40) + rows("B", "1", 60) + rows("B", "6", 50))
    comps = controlled_disparity(frame, "derived_race", min_count=10, reference_group_value="A")
    b = next(c for c in comps if c.group_value == "B")
    assert b.raw_gap_pp == pytest.approx(30.0)
    assert (b.n_group, b.n_reference) == (100, 100)
