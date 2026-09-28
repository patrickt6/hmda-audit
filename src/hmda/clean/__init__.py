"""Cleaning layer: sentinels, row filters, the waterfall, and the fixture loader.

``load_fixture`` is the one function every downstream module's tests are
written against: a caller reading ``fairness/air.py`` already knows
exactly what ``clean.load_fixture()`` returns. Its contract is fixed
here and must not change without updating every caller that depends on it.
"""

from __future__ import annotations

from pathlib import Path

FIXTURE_PATH = Path("tests/fixtures/hmda_50k.parquet")

# Columns every loader (fixture or full) guarantees are present, named or
# derived. This is the minimum schema every downstream module (fairness,
# model, governance) may assume without re-checking. Derived/computed
# columns are named with a leading "derived_" comment where they are HMDA's
# own derived fields vs. this repo's own additions.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "action_taken",  # int, 1-8; denial is 3; purchased loans (6) excluded upstream
    "derived_race",  # str category, includes "Race Not Available" as largest bucket
    "derived_ethnicity",
    "derived_sex",
    "applicant_age",  # str bucket or "8888" sentinel (not provided)
    "loan_amount",
    "income",  # RAW, in THOUSANDS of dollars, pass through sentinels.income_dollars()
    "debt_to_income_ratio",  # RAW mixed-type, pass through sentinels.parse_debt_to_income()
    "loan_to_value_ratio",  # RAW, "NA"/"Exempt" sentinels, "80" vs "80.0" string dup
    "loan_type",
    "loan_purpose",
    "occupancy_type",
    "property_value",  # RAW, "NA"/"Exempt" sentinels
    "lien_status",
    "state_code",
    "county_code",
    "census_tract",
    "tract_minority_population_percent",
    "tract_to_msa_income_percentage",
    "lei",
    "denial_reason-1",
    "denial_reason-2",
    "denial_reason-3",
    "denial_reason-4",
    "activity_year",  # needed for the model's time-based train/test split
)


def load_fixture(path: Path | str = FIXTURE_PATH):
    """Load the committed 50k-row HMDA fixture as a ``pandas.DataFrame``.

    Every column in :data:`REQUIRED_COLUMNS` is present in the returned
    frame, RAW and UNCLEANED: sentinel strings ("NA", "Exempt", "8888") are
    still present as literal strings, ``income`` is still in THOUSANDS of
    dollars, and ``debt_to_income_ratio`` is still the mixed bucket/integer
    string column. Callers must run the relevant ``hmda.clean.sentinels``
    functions before treating any of those columns as numeric.

    This is the ONE loading entry point every fixture-only module in
    this repo is written against, from the fairness functions through
    the model figures. Its
    return type and column set are load-bearing: a caller that needs a
    different column added reports the need rather than changing this
    function directly.

    Raises FileNotFoundError if the fixture has not been committed yet:
    ``tests/fixtures/hmda_50k.parquet`` must exist, be under 20 MB,
    and be committed to the repository.
    """
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"fixture not found: {path}")
    return pd.read_parquet(path)
