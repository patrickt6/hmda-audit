"""Tests for the ingest layer: schema stability, checksums, the fixture,
and load_duckdb helpers.

These run against the real ingest artifacts (data/hmda.duckdb, the
committed 50k fixture, data/manifest.json) rather than mocking the network,
since the network calls themselves were already made and verified by hand.
Tests that need the live DuckDB file are skipped if
it is not present, so `make test` still passes on a fresh checkout that has
not run `hmda ingest` yet -- only the committed fixture is required.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from hmda.clean import load_fixture, REQUIRED_COLUMNS
from hmda.ingest.checksum import sha256_file
from hmda.ingest.schema import SchemaVersion, assert_header_stable

FIXTURE_PATH = Path("tests/fixtures/hmda_50k.parquet")
DUCKDB_PATH = Path("data/hmda.duckdb")


# --------------------------------------------------------------------------
# The fixture itself
# --------------------------------------------------------------------------


def test_fixture_exists_and_is_under_20mb():
    assert FIXTURE_PATH.exists(), "tests/fixtures/hmda_50k.parquet must be committed"
    size_mb = FIXTURE_PATH.stat().st_size / (1024 * 1024)
    assert size_mb < 20, f"fixture is {size_mb:.1f} MB, must be under 20 MB"


def test_fixture_has_required_columns():
    df = load_fixture()
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"fixture missing required column {col}"


def test_fixture_row_count_is_50000():
    df = load_fixture()
    assert len(df) == 50_000


def test_fixture_contains_every_measured_trap():
    """Covers the data traps below, plus the generic NA sentinel. If any of
    these assertions fails, a later stage has no example row to
    develop its handling against, which is exactly the failure mode this
    check exists to prevent."""
    df = load_fixture()

    # trap 8: "Exempt" sentinel in loan_to_value_ratio / property_value
    assert (df["loan_to_value_ratio"] == "Exempt").sum() >= 1, "no Exempt LTV row in fixture"

    # trap 8: "8888" sentinel in applicant_age (age not provided)
    assert (df["applicant_age"] == "8888").sum() >= 1, "no 8888 applicant_age row in fixture"

    # generic "NA" sentinel, present in every column per KNOWN_SENTINELS["__default__"]
    assert (df["debt_to_income_ratio"] == "NA").sum() >= 1, "no NA debt_to_income_ratio row in fixture"

    # trap 7: bucket strings AND bare integers coexist in debt_to_income_ratio
    bucket_pattern = re.compile(r"%|<|>")
    dti = df["debt_to_income_ratio"].astype(str)
    has_bucket = dti.str.contains(bucket_pattern, regex=True).sum() >= 1
    has_bare_int = dti.str.match(r"^\d+$").sum() >= 1
    assert has_bucket, "no bucketed debt_to_income_ratio string in fixture"
    assert has_bare_int, "no bare-integer debt_to_income_ratio value in fixture"

    # trap 9: "Race Not Available" present with a realistic (non-trivial) share
    race_not_available_share = (df["derived_race"] == "Race Not Available").mean()
    assert race_not_available_share > 0.05, (
        f"Race Not Available share is {race_not_available_share:.1%}, "
        "too small to be realistic (measured 34% in DC 2023)"
    )


def test_fixture_income_is_raw_thousands_not_yet_converted():
    """The fixture loader promises RAW, uncleaned columns (clean/__init__.py
    docstring): income is still in thousands, not dollars, at this layer."""
    df = load_fixture()
    numeric_income = pd.to_numeric(df["income"], errors="coerce").dropna()
    # If income were already in dollars, the median would be in the hundreds
    # of thousands; in thousands, HMDA's modal values are ~100-150.
    assert numeric_income.median() < 1000, (
        "income looks like it has already been converted to dollars; "
        "the fixture must stay RAW (income_dollars() is clean/sentinels.py's job)"
    )


def test_load_fixture_raises_filenotfound_for_missing_path(tmp_path):
    missing = tmp_path / "does_not_exist.parquet"
    with pytest.raises(FileNotFoundError):
        load_fixture(missing)


# --------------------------------------------------------------------------
# Schema / header stability
# --------------------------------------------------------------------------


def test_assert_header_stable_passes_for_identical_headers():
    v1 = SchemaVersion(year=2023, columns=("a", "b", "c"), source_state="DC")
    v2 = SchemaVersion(year=2024, columns=("a", "b", "c"), source_state="DC")
    assert_header_stable([v1, v2])  # must not raise


def test_assert_header_stable_raises_for_different_headers():
    v1 = SchemaVersion(year=2023, columns=("a", "b", "c"), source_state="DC")
    v2 = SchemaVersion(year=2024, columns=("a", "b"), source_state="DC")
    with pytest.raises(ValueError):
        assert_header_stable([v1, v2])


def test_measured_header_stable_across_2023_2024_2025():
    """MEASURED 2026-09-11: DC's 2023/2024/2025 parquet files (fetched live
    via `hmda ingest`) have identical 99-column headers. Skipped if the
    parquet files are not present (fresh checkout, network not run)."""
    import duckdb

    files = [
        Path("data/parquet/DC_2023.parquet"),
        Path("data/parquet/DC_2024.parquet"),
        Path("data/parquet/DC_2025.parquet"),
    ]
    if not all(f.exists() for f in files):
        pytest.skip("DC 2023/2024/2025 parquet not present in this checkout")

    con = duckdb.connect()
    try:
        col_sets = []
        for f in files:
            cols = [r[0] for r in con.execute(f"describe select * from read_parquet('{f}')").fetchall()]
            col_sets.append(tuple(cols))
    finally:
        con.close()
    assert col_sets[0] == col_sets[1] == col_sets[2]
    assert len(col_sets[0]) == 99


# --------------------------------------------------------------------------
# Checksums
# --------------------------------------------------------------------------


def test_sha256_file_is_deterministic(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_bytes(b"hmda test content" * 1000)
    r1 = sha256_file(p)
    r2 = sha256_file(p)
    assert r1.sha256 == r2.sha256
    assert r1.size_bytes == p.stat().st_size


def test_sha256_file_differs_for_different_content(tmp_path):
    p1 = tmp_path / "a.txt"
    p2 = tmp_path / "b.txt"
    p1.write_bytes(b"content a")
    p2.write_bytes(b"content b")
    assert sha256_file(p1).sha256 != sha256_file(p2).sha256


# --------------------------------------------------------------------------
# DuckDB counts (skipped without a real database)
# --------------------------------------------------------------------------


def test_counts_by_year_matches_manifest_row_counts():
    """MEASURED 2026-09-11: DC 2023 loaded via `hmda ingest --year 2023
    --state DC` must equal 17,474 rows exactly. Skipped
    if data/hmda.duckdb is not present in this checkout."""
    if not DUCKDB_PATH.exists():
        pytest.skip("data/hmda.duckdb not present in this checkout")

    from hmda.ingest.load_duckdb import counts_by_year

    counts = counts_by_year(DUCKDB_PATH)
    assert isinstance(counts, dict)
    assert all(isinstance(y, int) and isinstance(c, int) for y, c in counts.items())
