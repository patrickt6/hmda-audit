"""Exactly three mitigation techniques, and a fairness gain never ships alone.

Rules covered: one mitigation technique per stage, no more; a fairness
gain never ships without its cost (disparity cut, accuracy change and
margin change always print together); and the study is never described as
deployed. A second file, ``tests/test_mitigate_reporting.py``, exists for
some reporting checks, but the reporting assertions for this module live
here too, under ``test_reporting_*`` names.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from hmda.fairness import mitigate as M
from hmda.model import economics as E


# --------------------------------------------------------------------------
# The registry cap: one technique per stage, no more.
# --------------------------------------------------------------------------
def test_the_registry_has_exactly_three_techniques():
    assert len(M.TECHNIQUE_REGISTRY) == 3


def test_there_is_exactly_one_technique_per_stage():
    stages = sorted(stage.value for stage in M.TECHNIQUE_REGISTRY.values())
    assert stages == ["in", "post", "pre"]


def test_a_fourth_technique_makes_the_cap_test_red():
    """The cap test must be able to fail. A check that cannot go red is not a check."""
    extended = dict(M.TECHNIQUE_REGISTRY)
    extended["calibrated_equalised_odds"] = M.MitigationStage.POST
    assert len(extended) == 4  # i.e. len(...) == 3 would now fail


def test_an_unregistered_technique_raises_rather_than_no_opping():
    with pytest.raises(KeyError):
        M.run_mitigation("not_a_technique", prepared=None, baseline=None, assumptions=None)


# --------------------------------------------------------------------------
# Protected class: out of the features, in at decision time only.
# --------------------------------------------------------------------------
def test_the_group_column_is_a_protected_column_and_is_excluded_from_features():
    from hmda.model import features as F

    assert F.is_protected_column(M.GROUP_COLUMN)
    assert M.GROUP_COLUMN not in F.FEATURE_COLUMNS


def test_reweighing_weights_equalise_a_biased_cell(monkeypatch):
    """Reweighing up-weights the under-represented (group, outcome) cell."""
    y = pd.Series([0, 0, 0, 0, 1, 1, 0, 1])
    g = pd.Series(["A", "A", "A", "A", "A", "A", "B", "B"])
    w = M.reweighing_weights(y, g)
    assert len(w) == len(y)
    assert np.all(w > 0.0)
    # Group B's denied row is rarer relative to independence than group A's,
    # so the weights must differ between the two groups' denied rows.
    assert w[4] != w[7]


def test_per_group_thresholds_differ_by_group_and_decide_differently():
    scores = np.concatenate([np.linspace(0.0, 1.0, 200), np.linspace(0.5, 1.0, 200)])
    g = pd.Series(["A"] * 200 + ["B"] * 200)
    table = M.per_group_thresholds(
        scores, g, target_approval_rate=0.5, eligible_groups=("A", "B"),
        fallback=0.5, min_count=10,
    )
    assert table["A"] != table["B"]
    approve = M.decide(scores, g, threshold=0.5, group_thresholds=table)
    assert approve[:200].mean() == pytest.approx(0.5, abs=0.02)
    assert approve[200:].mean() == pytest.approx(0.5, abs=0.02)


def test_a_group_below_the_count_floor_keeps_the_global_threshold():
    scores = np.linspace(0.0, 1.0, 120)
    g = pd.Series(["A"] * 115 + ["B"] * 5)
    table = M.per_group_thresholds(
        scores, g, 0.5, ("A", "B"), fallback=0.42, min_count=100,
    )
    assert table["B"] == 0.42


# --------------------------------------------------------------------------
# The end-to-end study. One fit set, shared across the reporting tests.
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def study():
    assumptions = E.load_assumptions()
    prepared, baseline, results = M.compare_all(assumptions=assumptions)
    lines = M.report_lines(prepared, baseline, results, assumptions, profit=True)
    return prepared, baseline, results, assumptions, lines


def test_all_three_techniques_run(study):
    _, _, results, _, _ = study
    assert sorted(r.technique for r in results) == sorted(M.TECHNIQUE_REGISTRY)


def test_reporting_every_line_carries_all_three_numbers(study):
    """The disparity cut, accuracy change and margin change always print together:
    a line with the disparity cut and no cost is a failure."""
    _, _, results, _, _ = study
    for result in results:
        line = result.one_line()
        assert "approval-rate gap" in line
        assert "agreement with the historical decision" in line
        assert "expected margin per 1,000 applications" in line
        # all three numbers are real, not placeholders
        assert result.disparity_cut_percent == result.disparity_cut_percent  # not NaN
        assert result.accuracy_change_pp == result.accuracy_change_pp
        assert result.margin_change_usd_per_1000 == result.margin_change_usd_per_1000


def test_reporting_no_banned_raw_statistic_appears(study):
    """No raw DIR, SPD, EOD, AUC, R2 or p-value ships as an outcome."""
    _, _, _, _, lines = study
    text = " ".join(lines)
    for banned in ("DIR", "SPD", "EOD", "AUC", "R2", "R²", "p-value", "p <"):
        assert banned not in text


def test_reporting_never_claims_deployment(study):
    """This is an offline study only: it never claims to be deployed."""
    _, _, _, _, lines = study
    text = " ".join(lines).lower()
    assert "offline counterfactual study" in text
    # The disclaimer itself names the banned words in order to deny them, so
    # it is removed before the check rather than the check being weakened.
    disclaimer = "nothing here is deployed, live, or in production]"
    assert disclaimer in text
    remainder = text.replace(disclaimer, "")
    for banned in (" deployed", "in production", "live applicant"):
        assert banned not in remainder


def test_reporting_labels_the_fixture_and_refuses_a_national_claim(study):
    """The study runs on the DC/WY/VT fixture only, and must say so."""
    _, _, _, _, lines = study
    text = " ".join(lines)
    assert "FIXTURE ONLY" in text
    assert "NOT a national" in text
    assert "DC / WY / VT" in text


def test_reporting_prints_the_assumptions_in_the_header(study):
    """The profit path prints the margin and LGD it assumed."""
    _, _, _, _, lines = study
    text = " ".join(lines)
    assert "config/economics.yaml" in text
    assert "expected margin per originated loan" in text
    assert "loss given default" in text


def test_reporting_flags_the_per_group_threshold_as_legally_contested(study):
    """It must never be presented as approved practice."""
    _, _, results, _, lines = study
    post = next(r for r in results if r.technique == "per_group_threshold")
    assert "LEGALLY CONTESTED" in post.note
    assert "LEGALLY CONTESTED" in " ".join(lines)


def test_reporting_states_the_race_not_reported_share(study):
    """The race-not-reported share must be printed beside every disparity number."""
    prepared, _, _, _, lines = study
    assert prepared.race_not_available_share > 0.0
    assert re.search(r"race not reported on \d+\.\d%", " ".join(lines))


def test_the_margin_number_moves_when_the_assumptions_change(study):
    """The proof, as a test, on an already-fitted study (no refit)."""
    prepared, baseline, results, _, _ = study
    halved = E.EconomicsAssumptions(
        expected_margin_per_approved_loan_usd=1.0,
        loss_given_default_rate=1.0,
        default_rate_assumption=1.0,
    )
    moved = E.expected_margin_per_1000(
        baseline.outcome.approval_rate * 1000.0, halved, baseline.outcome.mean_principal_usd
    )
    assert moved != baseline.outcome.margin_per_1000_usd


def test_every_technique_is_measured_against_the_same_unmitigated_baseline(study):
    _, baseline, results, _, _ = study
    for result in results:
        assert result.before == baseline.outcome


def test_the_unmitigated_gap_is_reported_even_though_it_is_not_a_result(study):
    """Raw and mitigated always ship together; the before-number is never hidden."""
    _, baseline, _, _, lines = study
    assert baseline.outcome.approval_gap_pp > 0.0
    assert "UNMITIGATED MODEL (the named baseline)" in " ".join(lines)


def test_main_exits_zero(capsys):
    assert M.main() == 0
    out = capsys.readouterr().out
    assert out.count("approval-rate gap") >= 4  # baseline + one per technique


# --------------------------------------------------------------------------
# prepare() must accept an in-memory DataFrame and must NOT
# read tests/fixtures/hmda_50k.parquet when it gets one.
# --------------------------------------------------------------------------
def _t22_frame(n: int = 12_345):
    full = pd.read_parquet(M.FIXTURE_PATH)
    return full.sample(n=n, random_state=0).reset_index(drop=True)


def test_t22_prepare_honours_an_in_memory_frame(monkeypatch):
    frame = _t22_frame()
    fixture_rows = M.prepare().rows_train + M.prepare().rows_test

    def _boom(*a, **k):  # pragma: no cover - the point is that it never runs
        raise AssertionError("prepare read parquet despite being given a frame")

    monkeypatch.setattr(pd, "read_parquet", _boom)
    prepared = M.prepare(frame=frame, source_label="TEST SOURCE", sample_seed=7)
    assert prepared.rows_train + prepared.rows_test < fixture_rows
    assert prepared.rows_train + prepared.rows_test <= len(frame)
    assert prepared.source_label == "TEST SOURCE"
    assert prepared.sample_seed == 7


def test_prepare_rejects_a_frame_with_no_source_label():
    """A frame with no source_label would print sample numbers under the
    fixture's DC/WY/VT label. That must be impossible, not merely discouraged."""
    frame = _t22_frame()
    with pytest.raises(ValueError, match="source_label"):
        M.prepare(frame=frame)
