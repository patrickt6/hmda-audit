"""Tests for the M13 recourse study.

The load-bearing test is :func:`test_synthetic_known_increase_is_recovered`:
a model whose decision line is known BY CONSTRUCTION, so the correct answer
is arithmetic and not an opinion. Every other test in this file only checks
that the machinery around that search does not lie about counts, units or
scope.

Scope note, on every number: the fixture is ``tests/fixtures/hmda_50k.parquet``,
DC/WY/VT only, NOT a national sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hmda.fairness import recourse as R


class IncomeStepModel:
    """A model that denies exactly when income is below ``cutoff_usd``.

    ``predict_proba`` returns 1.0 for denial below the cutoff and 0.0 at or
    above it, so with any threshold in (0, 1) the required increase for a row
    is exactly ``max(0, cutoff_usd - income)``. Nothing about the answer
    depends on a fit, a seed or a tolerance of the search.
    """

    def __init__(self, cutoff_usd: float) -> None:
        self.cutoff_usd = cutoff_usd

    def predict_proba(self, X):
        income = np.asarray(X[R.INCOME_COLUMN], dtype="float64")
        denial = (income < self.cutoff_usd).astype("float64")
        return np.column_stack([1.0 - denial, denial])


def _frame(incomes, loan_amounts=None) -> pd.DataFrame:
    """A minimal feature matrix with the two income-dependent columns."""
    incomes = np.asarray(incomes, dtype="float64")
    if loan_amounts is None:
        loan_amounts = np.full(len(incomes), 300_000.0)
    loan_amounts = np.asarray(loan_amounts, dtype="float64")
    return pd.DataFrame(
        {
            R.INCOME_COLUMN: incomes,
            "loan_amount": loan_amounts,
            R.LOAN_TO_INCOME_COLUMN: loan_amounts / incomes,
        }
    )


# ---------------------------------------------------------------------------
# The synthetic proof: the required change is known by construction.
# ---------------------------------------------------------------------------


def test_synthetic_known_increase_is_recovered():
    """Recover a required income increase that is known exactly by construction.

    Cutoff $120,000. An applicant at $100,000 needs exactly $20,000; one at
    $73,500 needs exactly $46,500; one already at $150,000 needs nothing.
    The search grid has a stated resolution of ``GRID_STEP_USD``, so the
    tolerance is that one step and no more: it is the resolution of the
    instrument, not slack granted to make the test pass.
    """
    model = IncomeStepModel(cutoff_usd=120_000.0)
    X = _frame([100_000.0, 73_500.0, 150_000.0])

    got = R.required_income_increase(model, X, threshold=0.5)

    expected = np.array([20_000.0, 46_500.0, 0.0])
    assert np.all(np.abs(got - expected) <= R.GRID_STEP_USD), (got, expected)
    # The two that needed a change must not be understated: the smallest
    # grid point that truly crosses is at or above the exact answer.
    assert got[0] >= expected[0]
    assert got[1] >= expected[1]
    assert got[2] == 0.0


def test_synthetic_unreachable_is_nan_not_a_number():
    """An applicant no achievable increase can help gets NaN, never a filled-in figure."""
    model = IncomeStepModel(cutoff_usd=50_000_000.0)
    X = _frame([100_000.0])

    got = R.required_income_increase(model, X, threshold=0.5)

    assert np.isnan(got[0]), got


def test_loan_to_income_is_recomputed_at_every_candidate():
    """The counterfactual row stays internally consistent when income moves.

    A model that reads only ``loan_to_income_ratio`` must still be crossable
    by an income change. If the engineered ratio were held at its original
    value, this search would report every row unreachable.
    """

    class RatioModel:
        def predict_proba(self, X):
            ratio = np.asarray(X[R.LOAN_TO_INCOME_COLUMN], dtype="float64")
            denial = (ratio > 3.0).astype("float64")
            return np.column_stack([1.0 - denial, denial])

    # $300,000 loan on $60,000 income is a ratio of 5.0. Crossing 3.0 needs
    # income of $100,000, so exactly $40,000 more.
    X = _frame([60_000.0], [300_000.0])
    got = R.required_income_increase(RatioModel(), X, threshold=0.5)

    assert np.abs(got[0] - 40_000.0) <= R.GRID_STEP_USD, got


def test_grid_is_ordered_and_bounded():
    grid = R.income_grid()
    assert grid[0] == R.GRID_STEP_USD
    assert np.all(np.diff(grid) > 0)
    assert grid[-1] <= R.MAX_INCREASE_USD


# ---------------------------------------------------------------------------
# Units. Income is in THOUSANDS in the raw column.
# ---------------------------------------------------------------------------


def test_median_income_on_the_fixture_is_in_a_human_range():
    """The median computed income must land between $30,000 and $300,000.

    This is the 1000x guard. If ``income`` were read raw (thousands) the
    median would be about $114, and M13's money figure would be wrong by
    three orders of magnitude.
    """
    from hmda.model import features as F

    raw = pd.read_parquet(R.FIXTURE_PATH)
    X, _ = F.build_feature_matrix(F.analysis_set(raw))
    median = float(np.nanmedian(X[R.INCOME_COLUMN].to_numpy(dtype="float64")))

    assert 30_000.0 <= median <= 300_000.0, median


def test_income_column_comes_from_the_sentinel_aware_converter():
    """``income_dollars`` is exactly 1000x the raw column on non-sentinel rows."""
    from hmda.clean.sentinels import income_dollars

    raw = pd.Series(["100", "150", "NA", "73"], dtype="string")
    got = income_dollars(raw)

    assert got.iloc[0] == 100_000
    assert got.iloc[1] == 150_000
    assert pd.isna(got.iloc[2])
    assert got.iloc[3] == 73_000


# ---------------------------------------------------------------------------
# Actionability and scope.
# ---------------------------------------------------------------------------


def test_actionable_features_hold_no_protected_attribute():
    """No actionable input may name a protected attribute. An applicant cannot change race."""
    from hmda.model.features import is_protected_column

    for name in R.ACTIONABLE_FEATURES:
        assert not is_protected_column(name), name
    assert R.INCOME_COLUMN.startswith(R.ACTIONABLE_FEATURES[0])


def test_no_protected_field_name_appears_in_the_module_source():
    """The same check a grep enforces, asserted in code so it cannot silently regress."""
    from pathlib import Path

    import hmda.fairness.recourse as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    for stem in ("race", "sex", "ethnicity"):
        assert "derived_" + stem not in source, stem


def test_grouping_columns_come_from_the_shared_allowlist():
    """Grouping is by a protected column, which is a reporting split, not an input."""
    from hmda.fairness.rates import ALLOWED_GROUP_COLUMNS

    assert len(ALLOWED_GROUP_COLUMNS) == 3
    for column in ALLOWED_GROUP_COLUMNS:
        assert column not in R.ACTIONABLE_FEATURES


# ---------------------------------------------------------------------------
# Reporting contract: money, count, unreachable share. Never a probability.
# ---------------------------------------------------------------------------


def test_result_sentence_is_money_with_a_count_and_an_unreachable_share():
    result = R.RecourseResult(
        group_column="group",
        group_value="Some Group",
        applicants_count=200,
        reachable_count=40,
        unreachable_share=0.8,
        median_required_income_increase_usd=25_000.0,
        below_floor=False,
    )
    sentence = result.sentence()

    assert "$25,000" in sentence
    assert "40 of 200" in sentence
    assert "80.0% unreachable" in sentence
    assert "probability" not in sentence.lower()
    assert "distance" not in sentence.lower()


def test_small_group_is_flagged_not_dropped_and_gets_no_number():
    result = R.RecourseResult(
        group_column="group",
        group_value="Tiny Group",
        applicants_count=4,
        reachable_count=1,
        unreachable_share=0.75,
        median_required_income_increase_usd=None,
        below_floor=True,
    )
    sentence = result.sentence()

    assert "NOT REPORTED" in sentence
    assert str(R.MIN_GROUP_COUNT) in sentence
    assert "$" not in sentence


def test_recourse_by_group_counts_are_positional_and_survive_a_repeated_index():
    """A bootstrap resample repeats index labels; counts must not double.

    This guards a real defect found on 2026-09-11: label-based alignment of
    the grouping column against the selected rows raised an IndexError on a
    resampled frame and would otherwise have inflated every group count.
    """
    from hmda.model import features as F

    raw = pd.read_parquet(R.FIXTURE_PATH)
    analysis = F.analysis_set(raw).head(2_000)
    resampled = analysis.sample(n=500, replace=True, random_state=0)
    assert resampled.index.duplicated().any()

    X, y = F.build_feature_matrix(analysis)
    model = IncomeStepModel(cutoff_usd=200_000.0)

    results = R.recourse_by_group(
        resampled,
        model,
        sorted(__import__("hmda.fairness.rates", fromlist=["x"]).ALLOWED_GROUP_COLUMNS)[0],
        threshold=0.5,
        feature_columns=X.columns,
        min_group_count=1,
    )
    assert sum(r.applicants_count for r in results) <= len(resampled)
    for r in results:
        assert r.reachable_count <= r.applicants_count
        assert 0.0 <= r.unreachable_share <= 1.0


def test_main_runs_and_labels_the_fixture_scope(capsys):
    """The entry point exits 0 and never presents a fixture number as national."""
    assert R.main() == 0
    out = capsys.readouterr().out

    assert "FIXTURE ONLY" in out
    assert "NOT a national sample" in out
    assert "UNMEASURABLE" in out  # the M13 verdict is printed, not hidden
    for column in ("ethnicity", "sex"):
        assert column in out


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_already_approved_rows_need_nothing(threshold):
    """A row the model already puts on the approval side gets 0, not a positive figure."""
    model = IncomeStepModel(cutoff_usd=50_000.0)
    X = _frame([200_000.0, 120_000.0])

    got = R.required_income_increase(model, X, threshold=threshold)

    assert list(got) == [0.0, 0.0]


# --------------------------------------------------------------------------
# run_study() must accept an in-memory DataFrame and must NOT
# read tests/fixtures/hmda_50k.parquet when it gets one.
# --------------------------------------------------------------------------
def _t22_frame(n: int = 12_345):
    import pandas as pd

    full = pd.read_parquet(R.FIXTURE_PATH)
    return full.sample(n=n, random_state=0).reset_index(drop=True)


def test_t22_run_study_honours_an_in_memory_frame(monkeypatch):
    import pandas as pd

    frame = _t22_frame()

    def _boom(*a, **k):  # pragma: no cover - the point is that it never runs
        raise AssertionError("run_study read parquet despite being given a frame")

    monkeypatch.setattr(pd, "read_parquet", _boom)
    study = R.run_study(frame=frame, source_label="TEST SOURCE", sample_seed=7)
    assert study["rows_train"] + study["rows_test"] <= len(frame)
    assert study["rows_train"] + study["rows_test"] < 50_000
    assert study["source_label"] == "TEST SOURCE"
    assert study["sample_seed"] == 7


def test_run_study_rejects_a_frame_with_no_source_label():
    """A frame with no source_label would print sample numbers under the
    fixture's DC/WY/VT label. That must be impossible, not merely discouraged."""
    frame = _t22_frame()
    with pytest.raises(ValueError, match="source_label"):
        R.run_study(frame=frame)


# --------------------------------------------------------------------------
# Review finding: stability_check(frame=...) had the parameter
# but no test exercising it.
# --------------------------------------------------------------------------
def test_stability_check_honours_an_in_memory_frame(monkeypatch):
    """stability_check must use the given frame, not silently fall back to the fixture."""
    import pandas as pd

    frame = _t22_frame(n=2_000)

    def _boom(*a, **k):  # pragma: no cover - the point is that it never runs
        raise AssertionError("stability_check read parquet despite being given a frame")

    monkeypatch.setattr(pd, "read_parquet", _boom)
    out = R.stability_check("derived_ethnicity", n_bootstrap=2, frame=frame)
    # Proof it used the small frame, not the 50k fixture: with only 2,000 rows
    # sampled with replacement, no group can retain enough denied-and-modeled
    # applicants across every bootstrap draw to clear MIN_GROUP_COUNT in every
    # round for every group of the full fixture; a non-empty, well-formed
    # result confirms it ran end-to-end on the small frame without reading disk.
    assert isinstance(out, dict)
    for entry in out.values():
        assert len(entry["medians"]) == len(entry["unreachable"])
        assert len(entry["medians"]) <= 2
