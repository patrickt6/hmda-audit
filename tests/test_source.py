"""The source abstraction must describe itself honestly and stay lazy."""

import pytest

from hmda.clean.source import open_source, FIXTURE, NATIONAL
from hmda.clean.source import _PARQUET_GLOB

# The national parquet files are gitignored (data/), so CI has only the fixture.
needs_national = pytest.mark.skipif(
    not any(_PARQUET_GLOB.parent.glob(_PARQUET_GLOB.name)),
    reason="national parquet files not present (data/ is gitignored)",
)


def test_fixture_source_reports_its_row_count_and_label():
    src = open_source(FIXTURE)
    assert src.rows == 50_000
    assert "50,000" in src.label
    assert "DC/WY/VT" in src.label


@needs_national
def test_national_source_reports_the_measured_national_count():
    src = open_source(NATIONAL)
    assert src.rows == 36_734_685
    assert "36,734,685" in src.label


@needs_national
def test_national_source_does_not_materialize_the_whole_table():
    """The national frame must be a lazy relation, never a DataFrame.

    A DataFrame here means 36.7M rows in RAM, which is the failure this
    module exists to prevent.
    """
    import pandas as pd

    src = open_source(NATIONAL)
    assert not isinstance(src.frame(), pd.DataFrame)


@needs_national
def test_sample_is_reproducible():
    a = open_source(NATIONAL).sample(1_000, seed=0)
    b = open_source(NATIONAL).sample(1_000, seed=0)
    assert a.equals(b)


@needs_national
def test_sample_returns_the_requested_row_count():
    """sample(n) must deliver n rows when the population has >= n rows.

    Finding 1: USING SAMPLE applied before the per-stratum WHERE filter
    made each stratum receive quota * (that stratum's share of the WHOLE
    table) rows instead of quota rows, so a requested 1,000 delivered only
    407 (measured against the unfixed source.py, before the stratified
    sampling fix).
    """
    src = open_source(NATIONAL)
    sample = src.sample(1_000, seed=0)
    assert len(sample) == 1_000


@needs_national
def test_sample_preserves_every_protected_group_present_in_the_population():
    """Every derived_race value in the population must appear in the sample.

    Finding 1: with the pre-fix bug, four protected groups -- American
    Indian or Alaska Native, Native Hawaiian or Other Pacific Islander, 2 or
    more minority races, and Free Form Text Only -- vanished entirely from
    a 1,000-row sample. Stratification exists precisely so this cannot
    happen; a test that only checks reproducibility cannot catch it.
    """
    import duckdb

    from hmda.clean.source import _STRATUM_COLUMN, register_frame

    src = open_source(NATIONAL)
    con = duckdb.connect()
    try:
        register_frame(con, src.frame())
        population_groups = {
            row[0]
            for row in con.execute(
                f"SELECT DISTINCT {_STRATUM_COLUMN} FROM frame"
            ).fetchall()
        }
    finally:
        con.close()

    sample = src.sample(5_000, seed=0)
    sample_groups = set(sample[_STRATUM_COLUMN].unique())

    missing = population_groups - sample_groups
    assert not missing, f"protected groups missing from sample: {missing}"


def test_unknown_source_name_raises():
    with pytest.raises(ValueError, match="unknown source"):
        open_source("nationwide")
