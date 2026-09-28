"""Tests for hmda.clean.sentinels.

Fixture facts used below are measured directly from
tests/fixtures/hmda_50k.parquet on 2026-09-11 with:
`.venv/bin/python -c "import pandas as pd; df=pd.read_parquet('tests/fixtures/hmda_50k.parquet'); print(df['<col>'].value_counts(dropna=False))"`
"""

from __future__ import annotations

import pandas as pd
import pytest

from hmda.clean import load_fixture
from hmda.clean.sentinels import (
    KNOWN_SENTINELS,
    UNCONFIRMED_SENTINELS,
    count_sentinels,
    income_dollars,
    normalize_loan_to_value,
    parse_debt_to_income,
    replace_sentinels,
    sentinels_for,
)


@pytest.fixture(scope="module")
def df():
    return load_fixture()


def test_sentinels_for_known_column():
    assert sentinels_for("applicant_age") == frozenset({"NA", "8888"})


def test_sentinels_for_unknown_column_falls_back_to_default():
    assert sentinels_for("some_column_never_mapped") == frozenset({"NA"})


def test_the_three_measured_sentinels_are_tracked():
    all_known = set().union(*KNOWN_SENTINELS.values())
    assert "NA" in all_known
    assert "Exempt" in all_known
    assert "8888" in all_known
    assert "1111" not in all_known


def test_1111_is_ordinary_income_data_not_a_sentinel():
    """CORRECTED 2026-09-13. National data (36,734,685-row `lar` table in
    data/hmda.duckdb) shows income=1111 (236 rows) is indistinguishable from
    its immediate neighbours (1108->274, 1109->248, 1110->334, 1112->262,
    1113->257, 1114->240), and shows none of the clustering that genuinely
    round values show (1100->1209, 1200->3537). income is denominated in
    thousands of dollars, so 1111 is an ordinary $1,111,000 income. Presence
    alone is not sentinel-hood; '1111' is NOT a sentinel for income."""
    assert "1111" not in sentinels_for("income")
    assert UNCONFIRMED_SENTINELS == frozenset()


def test_count_sentinels_matches_measured_applicant_age(df):
    # Measured 2026-09-11: applicant_age "8888" count is 6,644 of 50,000 rows.
    result = count_sentinels(df["applicant_age"], "applicant_age")
    assert result.total_rows == 50000
    assert result.counts["8888"] == 6644


def test_count_sentinels_matches_measured_ltv_exempt(df):
    # Measured 2026-09-11: loan_to_value_ratio "Exempt" count is 1,191.
    result = count_sentinels(df["loan_to_value_ratio"], "loan_to_value_ratio")
    assert result.counts["Exempt"] == 1191
    assert result.counts["NA"] == 16694


def test_replace_sentinels_nulls_only_sentinel_values():
    s = pd.Series(["NA", "Exempt", "80", "80.0"], dtype="string")
    out = replace_sentinels(s, "loan_to_value_ratio")
    assert out.isna().tolist() == [True, True, False, False]
    assert out.iloc[2] == "80"
    assert out.iloc[3] == "80.0"


def test_income_dollars_multiplies_raw_thousands_by_1000():
    raw = pd.Series(["100", "150", "NA"], dtype="string")
    dollars = income_dollars(raw)
    assert dollars.iloc[0] == 100_000
    assert dollars.iloc[1] == 150_000
    assert pd.isna(dollars.iloc[2])


def test_income_dollars_is_never_a_silent_float_cast_of_the_sentinel():
    """A naive astype(float) on the raw string 'NA' raises; a silent cast that
    instead produces a *number* for a sentinel row would hide the sentinel:
    'a naive astype(float) turns these into real numbers'. This asserts the
    sentinel survives as null, never as a numeric value, through this
    function specifically."""
    raw = pd.Series(["100", "NA"], dtype="string")
    dollars = income_dollars(raw)
    assert pd.isna(dollars.iloc[1])  # not silently coerced to 0 or any number


def test_income_dollars_median_lands_in_human_range(df):
    """Additional check: median loaded income must land between $30,000 and
    $300,000. Measured 2026-09-11: raw median non-sentinel income on the
    fixture is 120 (thousand) -> $120,000 after this function."""
    dollars = income_dollars(df["income"])
    median = dollars.dropna().median()
    assert 30_000 <= median <= 300_000


def test_a_sentinel_cast_directly_to_float_would_fail_this_module_never_does_it():
    """Proves the trap is real: astype(float) on the raw column raises on
    'NA', so income_dollars (which does not do this) is the only safe path."""
    raw = pd.Series(["100", "NA"], dtype="string")
    with pytest.raises((ValueError, TypeError)):
        raw.astype(float)


def test_parse_debt_to_income_keeps_every_non_sentinel_row(df):
    """Parsing debt_to_income_ratio keeps every non-NA row, both
    bucketed and bare-int. A naive float parse must fail this; prove it,
    then show the real parser passes."""
    raw = df["debt_to_income_ratio"]
    is_sentinel = raw.astype("string").isin({"NA", "Exempt"})
    expected_non_sentinel_count = int((~is_sentinel).sum())

    parsed = parse_debt_to_income(raw)
    non_null_count = int(pd.notna(parsed).sum())
    assert non_null_count == expected_non_sentinel_count


def test_naive_float_parse_of_debt_to_income_fails_or_drops_rows(df):
    """Proves the trap can fail: astype(float) either raises on bucket
    strings, or (if forced with errors='coerce') silently drops them. Either
    way it must NOT equal the real non-sentinel row count."""
    raw = df["debt_to_income_ratio"]
    is_sentinel = raw.astype("string").isin({"NA", "Exempt"})
    expected_non_sentinel_count = int((~is_sentinel).sum())

    coerced = pd.to_numeric(raw, errors="coerce")
    naive_non_null_count = int(coerced.notna().sum())
    assert naive_non_null_count != expected_non_sentinel_count
    assert naive_non_null_count < expected_non_sentinel_count


def test_parse_debt_to_income_returns_ordered_categorical():
    s = pd.Series(["<20%", "36", "49", ">60%", "NA"], dtype="string")
    parsed = parse_debt_to_income(s)
    assert isinstance(parsed, pd.Categorical)
    assert parsed.ordered
    # <20% sorts below the bare-int band, which sorts below >60%
    codes = dict(zip(parsed.categories, range(len(parsed.categories))))
    assert codes["<20%"] < codes["36"] < codes["49"] < codes[">60%"]


def test_parse_debt_to_income_treats_exempt_as_sentinel_not_a_category():
    """Observed 2026-09-11: 'Exempt' appears in debt_to_income_ratio in the
    fixture (1,191 rows), not just in loan_to_value_ratio/property_value.
    It must become null, never a fabricated category."""
    s = pd.Series(["Exempt", "36"], dtype="string")
    parsed = parse_debt_to_income(s)
    assert pd.isna(parsed[0])
    assert parsed[1] == "36"


def test_normalize_loan_to_value_collapses_duplicate_spellings():
    s = pd.Series(["80", "80.0", "80.00000", "NA", "Exempt"], dtype="string")
    out = normalize_loan_to_value(s)
    assert out.iloc[0] == out.iloc[1] == out.iloc[2] == 80.0
    assert pd.isna(out.iloc[3])
    assert pd.isna(out.iloc[4])
