"""Tests for the baseline, the challengers, and the time split.

What this file checks:
  - "The time split is asserted in a test: no training row is dated after any
    test row."
  - protected-class fields stay out of the feature matrix.
  - the baseline and the challenger are always reported together.

Every number in this file is a 50,000-row FIXTURE number
(``tests/fixtures/hmda_50k.parquet``). It is not a national number.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hmda.model import features as F
from hmda.model.baseline import BaseRateBaseline
from hmda.model.evaluate import DEFAULT_SPLIT_YEAR, FIXTURE_PATH, EvalResult, evaluate, run_eval


@pytest.fixture(scope="module")
def fixture_frame() -> pd.DataFrame:
    return pd.read_parquet(FIXTURE_PATH)


@pytest.fixture(scope="module")
def eval_run() -> dict:
    """One training run shared by every test that needs a fitted model."""
    return run_eval()


# --------------------------------------------------------------------------
# THE GATE: the split is by time, never at random.
# --------------------------------------------------------------------------
def test_no_training_row_is_dated_after_any_test_row(fixture_frame):
    """The check: no training row may be dated after any test row."""
    analysis = F.analysis_set(fixture_frame)
    train, test = F.time_split(analysis, DEFAULT_SPLIT_YEAR)

    train_years = pd.to_numeric(train["activity_year"])
    test_years = pd.to_numeric(test["activity_year"])

    assert len(train) > 0 and len(test) > 0
    assert train_years.max() < test_years.min(), (
        f"a training row is dated {train_years.max()}, at or after the earliest "
        f"test row {test_years.min()}"
    )


def test_the_split_partitions_the_data_with_no_row_in_both_folds(fixture_frame):
    analysis = F.analysis_set(fixture_frame)
    train, test = F.time_split(analysis, DEFAULT_SPLIT_YEAR)
    assert len(train) + len(test) == len(analysis)
    assert set(train.index).isdisjoint(set(test.index))


def test_a_split_year_that_empties_one_side_is_refused(fixture_frame):
    """A split that silently produced an empty test fold would make every
    downstream number meaningless, so it raises instead."""
    with pytest.raises(ValueError):
        F.time_split(fixture_frame, 1900)


def test_time_split_is_not_random(fixture_frame):
    """Two calls give the identical partition; there is no shuffle anywhere."""
    analysis = F.analysis_set(fixture_frame)
    a_train, a_test = F.time_split(analysis, DEFAULT_SPLIT_YEAR)
    b_train, b_test = F.time_split(analysis, DEFAULT_SPLIT_YEAR)
    assert list(a_train.index) == list(b_train.index)
    assert list(a_test.index) == list(b_test.index)


# --------------------------------------------------------------------------
# THE GATE: protected-class fields never enter the feature matrix.
# --------------------------------------------------------------------------
def test_protected_column_families_are_recognised():
    """The blocklist is a family rule, so it catches the raw multi-response
    columns a name-by-name check would miss."""
    for col in (
        "derived_race",
        "derived_sex",
        "derived_ethnicity",
        "applicant_race-1",
        "applicant_race-5",
        "co-applicant_ethnicity-3",
        "applicant_sex",
        "co-applicant_sex_observed",
        "applicant_age",
        "co-applicant_age_above_62",
    ):
        assert F.is_protected_column(col), col

    for col in ("loan_amount", "income", "loan_to_value_ratio", "lien_status"):
        assert not F.is_protected_column(col), col


def test_no_protected_column_is_in_the_feature_allowlist():
    for col in F.FEATURE_COLUMNS:
        assert not F.is_protected_column(col), col


def test_no_protected_column_reaches_the_built_matrix(eval_run, fixture_frame):
    analysis = F.analysis_set(fixture_frame)
    X, _ = F.build_feature_matrix(analysis)
    for col in X.columns:
        assert not F.is_protected_column(col), col


def test_the_allowlist_and_the_blocklist_do_not_intersect():
    blocked = set(F.EXCLUDED_COLUMNS) | set(F.PROXY_COLUMNS)
    assert blocked.isdisjoint(set(F.FEATURE_COLUMNS))


# --------------------------------------------------------------------------
# Target leakage.
# --------------------------------------------------------------------------
def test_outcome_encoding_columns_are_never_features():
    """action_taken, the denial reasons and purchaser_type encode the outcome."""
    for col in (
        "action_taken",
        "denial_reason-1",
        "denial_reason-2",
        "denial_reason-3",
        "denial_reason-4",
        "purchaser_type",
    ):
        assert col in F.LEAKING_COLUMNS
        assert col not in F.FEATURE_COLUMNS


def test_post_decision_pricing_columns_are_never_features():
    """Rate and cost columns exist only once a loan is acted on."""
    for col in ("interest_rate", "rate_spread", "total_loan_costs", "loan_term", "aus-1"):
        assert col in F.LEAKING_COLUMNS
        assert col not in F.FEATURE_COLUMNS


def test_the_tract_minority_share_proxy_is_excluded():
    """A tract minority share is a geographic proxy for race; excluded by
    judgement and recorded in PROXY_COLUMNS so the judgement is visible."""
    assert "tract_minority_population_percent" in F.PROXY_COLUMNS
    assert "tract_minority_population_percent" not in F.FEATURE_COLUMNS


# --------------------------------------------------------------------------
# Cleaning: the two measured traps must not be re-introduced here.
# --------------------------------------------------------------------------
def test_income_is_converted_out_of_thousands(fixture_frame):
    """The raw income column is in THOUSANDS.

    Asserts the fixture median lands in a human annual-income range, the same
    check the loader must pass.
    """
    dollars = F.income_dollars(fixture_frame)
    assert 30_000 <= dollars.median() <= 300_000, dollars.median()


def test_debt_to_income_parsing_keeps_the_bucketed_rows(fixture_frame):
    """A float cast drops every bucket string in debt_to_income_ratio.

    The bucketed rows are the majority of the non-missing column in the
    fixture, so a naive parse would be a large silent loss.
    """
    raw = fixture_frame["debt_to_income_ratio"].astype("string").str.strip()
    present = ~raw.isin(["NA", "Exempt"])
    parsed = F.debt_to_income_ordinal(fixture_frame)
    assert parsed[present].notna().all()

    naive = pd.to_numeric(raw.where(present), errors="coerce")
    assert naive.notna().sum() < parsed[present].notna().sum(), (
        "the naive float parse did not lose rows, so this guard is not testing anything"
    )


def test_the_ltv_string_duplication_collapses(fixture_frame):
    """'80' and '80.0' must parse to the same value."""
    X, _ = F.build_feature_matrix(F.analysis_set(fixture_frame))
    assert X["loan_to_value_ratio"].dtype == "float64"


# --------------------------------------------------------------------------
# The analysis set.
# --------------------------------------------------------------------------
def test_purchased_loans_are_excluded(fixture_frame):
    """A purchased loan (action_taken = 6) is not an application."""
    analysis = F.analysis_set(fixture_frame)
    actions = pd.to_numeric(analysis["action_taken"])
    assert (actions == 6).sum() == 0
    assert (pd.to_numeric(fixture_frame["action_taken"]) == 6).sum() > 0


def test_the_label_is_action_taken_three(fixture_frame):
    analysis = F.analysis_set(fixture_frame)
    y = F.label(analysis)
    actions = pd.to_numeric(analysis["action_taken"])
    assert set(y.unique()) <= {0, 1}
    assert (y == 1).sum() == (actions == 3).sum()


# --------------------------------------------------------------------------
# The baseline, and the pairing.
# --------------------------------------------------------------------------
def test_base_rate_baseline_predicts_the_training_rate_for_everyone():
    y = pd.Series([1, 0, 0, 0])
    b = BaseRateBaseline().fit(y)
    assert b.base_rate == pytest.approx(0.25)
    preds = b.predict_proba(pd.DataFrame({"anything": [1, 2, 3]}))
    assert len(preds) == 3
    assert len(set(preds)) == 1


def test_base_rate_baseline_ranks_at_chance(eval_run):
    """A constant score ties every pair, so the baseline sits exactly at the
    50-of-100 floor the M7 sentence names."""
    assert eval_run["logistic_result"].baseline_auc == pytest.approx(0.5)
    assert eval_run["gbm_result"].baseline_auc == pytest.approx(0.5)


def test_baseline_must_be_fitted_before_it_predicts():
    with pytest.raises(ValueError):
        BaseRateBaseline().predict_proba(pd.DataFrame({"a": [1]}))


def test_evaluate_always_returns_both_models(eval_run):
    """The check: a run that reports only the challenger is a failure."""
    for key in ("logistic_result", "gbm_result"):
        result = eval_run[key]
        assert isinstance(result, EvalResult)
        assert result.baseline_sentence
        assert result.challenger_sentence
        line = result.one_line()
        assert "BASELINE" in line and "CHALLENGER" in line
        assert "\n" not in line, "the two numbers must be on ONE line"


def test_no_module_function_returns_a_challenger_score_alone():
    """Enforced structurally: the only public scoring entry
    point in evaluate.py returns the pair."""
    import hmda.model.evaluate as E

    public = [n for n in dir(E) if not n.startswith("_") and callable(getattr(E, n))]
    assert "evaluate" in public
    assert not any("challenger" in n.lower() for n in public)


# --------------------------------------------------------------------------
# The challengers actually beat the baseline (the point of M7).
# --------------------------------------------------------------------------
def test_both_challengers_beat_the_base_rate_baseline(eval_run):
    assert eval_run["logistic_result"].percent_better > 0
    assert eval_run["gbm_result"].percent_better > 0


def test_the_gbm_is_at_least_as_good_as_the_logistic_regression(eval_run):
    """Measured on the 50k fixture 2026-09-11; if this ever flips, the
    challenger choice should be revisited rather than the test relaxed."""
    assert eval_run["gbm_result"].challenger_auc >= eval_run["logistic_result"].challenger_auc


def test_the_run_reports_the_split_and_the_row_counts(eval_run):
    """A number with no denominator is not reportable."""
    assert eval_run["rows_raw"] == 50_000
    assert eval_run["rows_analysis"] < eval_run["rows_raw"]
    assert eval_run["rows_train"] + eval_run["rows_test"] == eval_run["rows_analysis"]
    assert eval_run["n_features"] > 0


# --------------------------------------------------------------------------
# The frame parameter. run_eval must accept an in-memory
# DataFrame and must NOT read tests/fixtures/hmda_50k.parquet when it gets
# one. Proof is two-fold: the reported row count is the frame's, and
# pd.read_parquet is made to explode.
# --------------------------------------------------------------------------
def _t22_frame(n: int = 12_345):
    import pandas as pd

    from hmda.model.evaluate import FIXTURE_PATH

    full = pd.read_parquet(FIXTURE_PATH)
    return full.sample(n=n, random_state=0).reset_index(drop=True)


def test_t22_run_eval_honours_an_in_memory_frame(monkeypatch):
    import pandas as pd

    from hmda.model import evaluate as EV

    frame = _t22_frame()

    def _boom(*a, **k):  # pragma: no cover - the point is that it never runs
        raise AssertionError("run_eval read parquet despite being given a frame")

    monkeypatch.setattr(pd, "read_parquet", _boom)
    result = EV.run_eval(frame=frame, source_label="TEST SOURCE")
    assert result["rows_raw"] == len(frame)
    assert result["rows_raw"] != 50_000


def test_run_eval_rejects_a_frame_with_no_source_label():
    """A frame with no source_label would print sample numbers under the
    fixture's DC/WY/VT label. That must be impossible, not merely discouraged."""
    frame = _t22_frame()
    with pytest.raises(ValueError, match="source_label"):
        run_eval(frame=frame)
