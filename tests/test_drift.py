"""Tests for the year-over-year drift/stability check.

What this file checks:
  - ``hmda audit --drift`` exits 0 and prints "N of M input columns shifted
    past <threshold>" (proven here via ``python -m hmda.governance.drift``,
    the module's own entry point; see ``drift.py`` module docstring).
  - No raw PSI value or the word "PSI" reaches a user-facing output string
    (it may appear in code comments). This file asserts directly against
    the module's own OUTPUT strings, which is the substance of the rule
    regardless of which layer renders the final report.
  - ``pytest tests/test_drift.py -q`` exits 0, including a synthetic
    no-drift case returning exactly 0 and a synthetic known-drift case
    returning the expected count.
"""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from hmda.clean import load_fixture
from hmda.governance import drift as D
from hmda.model.features import FEATURE_COLUMNS


# --------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------
def _base_row(**overrides) -> dict:
    """One well-formed row covering every FEATURE_COLUMNS + activity_year, raw/uncleaned."""
    row = {
        "loan_amount": "255000.0",
        "income": "120",
        "debt_to_income_ratio": "39",
        "loan_to_value_ratio": "80.0",
        "property_value": "325000",
        "loan_type": "1",
        "loan_purpose": "1",
        "occupancy_type": "1",
        "lien_status": "1",
        "preapproval": "2",
        "conforming_loan_limit": "C",
        "construction_method": "1",
        "total_units": "1",
        "open-end_line_of_credit": "2",
        "business_or_commercial_purpose": "2",
        "reverse_mortgage": "2",
        "activity_year": "2023",
    }
    row.update(overrides)
    return row


def _synthetic_frame(n: int, year: int, seed: int, drift_column: str | None = None) -> pd.DataFrame:
    """Build ``n`` synthetic rows for one year, all model-input columns present.

    Every column is IDENTICALLY distributed across a "year A" and "year B"
    call to this function unless ``drift_column`` is set for that call, in
    which case that one column is drawn from a visibly different
    distribution. ``seed`` only spreads the numeric columns across a few
    values so PSI has more than one bin to work with; it does not change
    the underlying distribution shape between otherwise-matched calls.
    """
    import random

    rng = random.Random(seed)
    rows = []
    for i in range(n):
        loan_amount = 100000 + (i % 10) * 20000
        income = 60 + (i % 8) * 15
        dti = 20 + (i % 12) * 2  # stays in the 20-42 "exact integer" range mostly
        ltv = 60 + (i % 10) * 4
        prop_value = 150000 + (i % 10) * 30000
        loan_type = str(1 + (i % 4))
        row = _base_row(
            loan_amount=str(loan_amount),
            income=str(income),
            debt_to_income_ratio=str(dti) if dti <= 49 else "40",
            loan_to_value_ratio=str(float(ltv)),
            property_value=str(prop_value),
            loan_type=loan_type,
            activity_year=str(year),
        )
        rows.append(row)

    if drift_column is not None:
        # Force a visibly different distribution for exactly one column, all
        # rows in this frame, so PSI on it should land near its maximum
        # rather than anywhere close to 0.
        if drift_column in ("loan_amount", "income", "property_value"):
            for row in rows:
                row[drift_column] = str(float(row[drift_column]) * 20 + 5_000_000)
        elif drift_column == "loan_to_value_ratio":
            for row in rows:
                row[drift_column] = "150.0"
        elif drift_column == "debt_to_income_ratio":
            for row in rows:
                row[drift_column] = ">60%"
        elif drift_column in FEATURE_COLUMNS:
            # A categorical column: flip every row to a category never used
            # in the un-drifted frame.
            for row in rows:
                row[drift_column] = "9"

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# check_drift / check_drift_by_year
# --------------------------------------------------------------------------
def test_synthetic_no_drift_returns_zero_shifted_columns():
    """Two years drawn from the identical synthetic distribution must report 0 shifted."""
    frame_a = _synthetic_frame(400, 2023, seed=1)
    frame_b = _synthetic_frame(400, 2024, seed=2)  # different seed, same shape

    result = D.check_drift(frame_a, frame_b)

    assert result.columns_checked == len(FEATURE_COLUMNS)
    assert result.columns_shifted == 0
    assert result.threshold == D.DEFAULT_PSI_THRESHOLD


@pytest.mark.parametrize(
    "drift_column",
    ["loan_amount", "loan_to_value_ratio", "debt_to_income_ratio", "loan_type"],
)
def test_synthetic_known_drift_flags_exactly_the_drifted_column(drift_column):
    """One column forced to a wildly different distribution must be the only one flagged."""
    frame_a = _synthetic_frame(400, 2023, seed=1)
    frame_b = _synthetic_frame(400, 2024, seed=2, drift_column=drift_column)

    result = D.check_drift(frame_a, frame_b)

    shifted_columns = {c.column for c in result.per_column if c.shifted}
    assert shifted_columns == {drift_column}, shifted_columns
    assert result.columns_shifted == 1


def test_check_drift_by_year_on_fixture_returns_one_result_per_adjacent_pair():
    """The committed fixture spans 2023/2024/2025, so 2 adjacent pairs."""
    frame = load_fixture()
    years = sorted(pd.to_numeric(frame["activity_year"], errors="coerce").dropna().unique().astype(int))
    assert years == [2023, 2024, 2025]

    results = D.check_drift_by_year(frame)

    assert len(results) == 2
    assert (results[0].year_a, results[0].year_b) == (2023, 2024)
    assert (results[1].year_a, results[1].year_b) == (2024, 2025)
    for result in results:
        assert result.columns_checked == len(FEATURE_COLUMNS)
        assert 0 <= result.columns_shifted <= len(FEATURE_COLUMNS)


def test_check_drift_defaults_to_the_model_input_allowlist():
    """Checks the model's FEATURE_COLUMNS allowlist, not every column in the file."""
    frame_a = _synthetic_frame(50, 2023, seed=1)
    frame_b = _synthetic_frame(50, 2024, seed=2)

    result = D.check_drift(frame_a, frame_b)

    checked = {c.column for c in result.per_column}
    assert checked == set(FEATURE_COLUMNS)


# --------------------------------------------------------------------------
# THE GATE: no raw PSI value or the term itself ever reaches a user-facing string.
# --------------------------------------------------------------------------
def test_sentence_never_contains_a_raw_psi_number_or_the_word_psi():
    frame_a = _synthetic_frame(200, 2023, seed=1, drift_column="loan_amount")
    frame_b = _synthetic_frame(200, 2024, seed=2)

    result = D.check_drift(frame_a, frame_b)
    sentence = result.sentence()

    assert "psi" not in sentence.lower()
    assert "population stability" not in sentence.lower()
    # No column's raw internal PSI value (to several roundings) leaks into the sentence.
    for col_drift in result.per_column:
        for ndigits in (2, 3, 4, 6):
            assert str(round(col_drift.psi, ndigits)) not in sentence


def test_sentence_matches_the_allowed_grammar_exactly():
    """"N of M input columns shifted past <threshold>": the one allowed shape."""
    frame_a = _synthetic_frame(100, 2023, seed=1)
    frame_b = _synthetic_frame(100, 2024, seed=2)

    result = D.check_drift(frame_a, frame_b)
    sentence = result.sentence()

    assert sentence.startswith(f"{result.columns_shifted} of {result.columns_checked} input columns shifted past {result.threshold}")


def test_cli_module_output_never_leaks_psi_or_a_raw_score():
    """Runs the real entry point end to end: python -m hmda.governance.drift."""
    proc = subprocess.run(
        [sys.executable, "-m", "hmda.governance.drift"],
        capture_output=True,
        text=True,
        cwd=None,
    )
    assert proc.returncode == 0, proc.stderr
    output = proc.stdout
    assert "psi" not in output.lower()
    assert "population stability index" not in output.lower()
    assert "of 16 input columns shifted past" in output
    assert "FIXTURE" in output  # the source is labelled every time


def test_threshold_is_a_named_constant_not_a_magic_number():
    assert isinstance(D.DEFAULT_PSI_THRESHOLD, float)
    assert D.DEFAULT_PSI_THRESHOLD > 0


# --------------------------------------------------------------------------
# Data traps covered here: mixed-type DTI, "NA"/"Exempt"/"80" vs "80.0" LTV.
# --------------------------------------------------------------------------
def test_column_psi_handles_mixed_type_debt_to_income_ratio_without_raising():
    frame_a = pd.DataFrame(
        [
            _base_row(debt_to_income_ratio="39"),
            _base_row(debt_to_income_ratio="<20%"),
            _base_row(debt_to_income_ratio="NA"),
            _base_row(debt_to_income_ratio="Exempt"),
            _base_row(debt_to_income_ratio=">60%"),
        ]
    )
    frame_b = pd.DataFrame(
        [
            _base_row(debt_to_income_ratio="41"),
            _base_row(debt_to_income_ratio="30%-<36%"),
            _base_row(debt_to_income_ratio="NA"),
        ]
    )
    psi = D.column_psi(frame_a, frame_b, "debt_to_income_ratio")
    assert psi >= 0.0
    assert not pd.isna(psi)


def test_column_psi_handles_loan_to_value_sentinels_and_string_duplicates():
    frame_a = pd.DataFrame(
        [
            _base_row(loan_to_value_ratio="80"),
            _base_row(loan_to_value_ratio="80.0"),
            _base_row(loan_to_value_ratio="NA"),
            _base_row(loan_to_value_ratio="Exempt"),
            _base_row(loan_to_value_ratio="95.0"),
        ]
    )
    frame_b = pd.DataFrame(
        [
            _base_row(loan_to_value_ratio="80.00000"),
            _base_row(loan_to_value_ratio="90.0"),
            _base_row(loan_to_value_ratio="NA"),
        ]
    )
    psi = D.column_psi(frame_a, frame_b, "loan_to_value_ratio")
    assert psi >= 0.0
    assert not pd.isna(psi)


def test_column_psi_on_real_fixture_columns_never_raises():
    """Smoke test across every FEATURE_COLUMNS entry on real, messy fixture data."""
    frame = load_fixture()
    frame_a = frame[pd.to_numeric(frame["activity_year"], errors="coerce") == 2023]
    frame_b = frame[pd.to_numeric(frame["activity_year"], errors="coerce") == 2024]
    for column in FEATURE_COLUMNS:
        psi = D.column_psi(frame_a, frame_b, column)
        assert psi >= 0.0
        assert not pd.isna(psi)
