"""Sentinel handling for HMDA LAR columns.

MEASURED (DC 2023, 17,474 rows, 2026-09-11): HMDA encodes missing or
not-applicable values as sentinel strings that differ by column. A naive
``astype(float)`` on a raw column silently turns these into real numbers or
NaN with no record of what happened. Every sentinel must be mapped explicitly
here and counted, never inferred at call sites.

Confirmed sentinels, by column:
    - ``"NA"``      : every column, generic not-available.
    - ``"Exempt"``  : ``loan_to_value_ratio``, ``property_value``.
    - ``"8888"``    : ``applicant_age`` (age not provided).

``"1111"`` is NOT a sentinel for ``income``. CORRECTED 2026-09-13: an earlier
version of this module read a bare non-zero count (236 rows for
``income = 1111`` in the 36,734,685-row national ``lar`` table in
``data/hmda.duckdb``) as proof of sentinel-hood. That rule is wrong. The
decisive test is the neighbour profile, measured with:
``.venv/bin/python -c "import duckdb; con = duckdb.connect('data/hmda.duckdb',
read_only=True); [print(f'income={v} rows=' +
str(con.execute(f\"select count(*) from lar where cast(income as
varchar)='{v}'\").fetchone()[0])) for v in (1108,1109,1110,1111,1112,1113,
1114,1100,1200)]"``, output: income=1108 rows=274, income=1109 rows=248,
income=1110 rows=334, income=1111 rows=236, income=1112 rows=262,
income=1113 rows=257, income=1114 rows=240, income=1100 rows=1209 (genuine
round-number clustering), income=1200 rows=3537 (genuine round-number
clustering). ``income`` is denominated in THOUSANDS of dollars, so
``income = 1111`` is an ordinary income of $1,111,000; its count is
indistinguishable from its immediate neighbours and shows none of the
clustering genuinely round values show. Verdict: ordinary data, not a
sentinel. ``"1111"`` was removed from ``KNOWN_SENTINELS["income"]``; it is
not carried in ``UNCONFIRMED_SENTINELS`` either, since the national data now
answers the question directly rather than leaving it open.

``loan_to_value_ratio`` also carries both the strings ``"80"`` and ``"80.0"``
for the same value: normalize numeric-looking strings
before comparing them, this module does not do that normalization itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Column name -> set of raw string values that mean "no value", not a number.
# This is the map every loader must consult before any numeric cast.
#
# "debt_to_income_ratio": frozenset({"NA", "Exempt"}): "Exempt" is NOT one of
# the sentinels HMDA's own coding names for this column (that coding only lists
# it for loan_to_value_ratio/property_value). It was found directly in
# tests/fixtures/hmda_50k.parquet on 2026-09-11 (observation):
# `df["debt_to_income_ratio"].value_counts()` shows "Exempt": 1191 rows,
# the identical count found for "Exempt" in loan_to_value_ratio and
# property_value in the same fixture (measured via
# `.venv/bin/python -c "import pandas as pd; df=pd.read_parquet('tests/fixtures/hmda_50k.parquet'); print(df['debt_to_income_ratio'].value_counts(dropna=False))"`).
# Interpretation: these are very likely the same exempt-transaction rows
# across all three columns, but that join was not verified row-by-row,
# so treat it as an observed sentinel value for this column,
# not a confirmed cross-column identity.
KNOWN_SENTINELS: dict[str, frozenset[str]] = {
    "loan_to_value_ratio": frozenset({"NA", "Exempt"}),
    "property_value": frozenset({"NA", "Exempt"}),
    "applicant_age": frozenset({"NA", "8888"}),
    "debt_to_income_ratio": frozenset({"NA", "Exempt"}),
    "income": frozenset({"NA"}),
    # Every other column at minimum carries the generic "NA" sentinel.
    "__default__": frozenset({"NA"}),
}

# Sentinels named in HMDA's published coding but not yet confirmed present in
# any file this repo has actually opened. Tracked separately so a future
# confirmation (or non-confirmation) against the national file is a one-line
# change here, not a re-derivation.
UNCONFIRMED_SENTINELS: frozenset[str] = frozenset()


@dataclass(frozen=True)
class SentinelCount:
    """One column's sentinel tally, produced by :func:`count_sentinels`.

    column: the HMDA column name counted.
    counts: mapping of the raw sentinel string to how many rows held it.
    total_rows: the row count the column was counted over, so a share can be
        computed by the caller without re-reading the data.
    """

    column: str
    counts: dict[str, int]
    total_rows: int


def sentinels_for(column: str) -> frozenset[str]:
    """Return the set of raw string values that are sentinels for ``column``.

    Falls back to ``KNOWN_SENTINELS["__default__"]`` (currently just
    ``{"NA"}``) for any column not given a more specific entry. Never raises
    for an unknown column name; an unmapped column still gets the generic
    sentinel set rather than silently treating every string as data.
    """
    return KNOWN_SENTINELS.get(column, KNOWN_SENTINELS["__default__"])


def count_sentinels(series, column: str) -> SentinelCount:
    """Count occurrences of each sentinel value for ``column`` in ``series``.

    ``series`` is expected to be the raw (pre-cast) string column, e.g. a
    ``pandas.Series`` of dtype object/string, as loaded straight from the
    HMDA CSV/parquet before any numeric conversion. Must be called, and its
    result surfaced in the data-quality report, before that column is ever
    cast to a numeric or categorical dtype: every sentinel must be mapped
    explicitly and counted in the data-quality report.

    Returns a :class:`SentinelCount` naming exactly which raw strings were
    treated as missing and how many rows each accounted for.
    """
    sentinels = sentinels_for(column)
    str_series = series.astype("string")
    counts = {value: int((str_series == value).sum()) for value in sorted(sentinels)}
    return SentinelCount(column=column, counts=counts, total_rows=int(len(series)))


def replace_sentinels(series, column: str):
    """Return a copy of ``series`` with sentinel strings replaced by null.

    Non-sentinel values are returned unchanged (still raw strings/ints at
    this stage: this function does not parse or cast, it only nulls out the
    values in :func:`sentinels_for`). Casting to a numeric or categorical
    dtype is the caller's job (see ``income_thousands_to_dollars`` and
    ``parse_debt_to_income`` for the two columns with non-trivial casts).
    """
    sentinels = sentinels_for(column)
    str_series = series.astype("string")
    return str_series.mask(str_series.isin(sentinels), other=pd.NA)


def income_dollars(income_thousands_series):
    """Convert the raw ``income`` column (THOUSANDS of dollars) to dollars.

    MEASURED (DC 2023, 2026-09-11): HMDA's ``income`` column
    is denominated in **thousands** of dollars: modal raw values are ``100``,
    ``150``, ``120``, meaning $100,000, $150,000, $120,000. This function
    must multiply by 1,000. Every downstream consumer of income (in
    particular the recourse study, a money claim) must call this
    function rather than read ``income`` raw, or the headline number is
    wrong by 1000x.

    A caller-side test must assert the resulting median lands in a human
    range (between $30,000 and $300,000); that assertion
    lives in ``tests/test_sentinels.py``, not in this function.
    """
    cleaned = replace_sentinels(income_thousands_series, "income")
    thousands = pd.to_numeric(cleaned, errors="raise")
    return thousands * 1000


def parse_debt_to_income(dti_series):
    """Parse the mixed-type ``debt_to_income_ratio`` column into an ordered category.

    MEASURED (DC 2023, 2026-09-11): this HMDA column holds
    BOTH bucket strings (``"<20%"``, ``"20%-<30%"``, ``"30%-<36%"``,
    ``">60%"``) AND bare integer strings (``"36"`` ... ``"49"``, the exact
    values HMDA reports only within the 36-50% band) in the same column: 21
    distinct values were seen in the DC sample alone. It must be parsed into
    a single ordered categorical (bucket boundary order), never cast
    directly to float: a naive ``astype(float)`` raises or silently drops
    every bucketed row, which in the DC sample was roughly two thirds of the
    non-null column (6,303 "NA" plus ~5,800 bucketed rows out of 17,474).

    Returns an ordered categorical series where every non-"NA" input row is
    represented: parsing it keeps every non-NA row.

    Observation, 2026-09-11: the fixture also carries the value
    ``"Exempt"`` in this column (1,191 of 50,000 rows, measured with
    ``df["debt_to_income_ratio"].value_counts()`` on
    ``tests/fixtures/hmda_50k.parquet``), the same count as ``"Exempt"`` in
    ``loan_to_value_ratio``/``property_value``. That is not one of the four
    sentinels HMDA's own coding names for this specific column, but it is
    plainly not a DTI value, so it is treated as a sentinel here too (see
    ``KNOWN_SENTINELS["debt_to_income_ratio"]`` above) and becomes null, not
    a synthetic bucket. "Every non-NA row" is read as "every
    row that is not a sentinel" for this reason: an "Exempt" row has no DTI
    to report, so counting it as a kept category would fabricate one.
    """
    cleaned = replace_sentinels(dti_series, "debt_to_income_ratio")

    bucket_order = ["<20%", "20%-<30%", "30%-<36%"]
    bare_int_order = [str(i) for i in range(36, 50)]
    tail_order = ["50%-60%", ">60%"]
    categories = bucket_order + bare_int_order + tail_order

    observed = set(cleaned.dropna().unique())
    unexpected = observed - set(categories)
    if unexpected:
        raise ValueError(
            "parse_debt_to_income: unmapped debt_to_income_ratio value(s) "
            f"found, not in the known bucket/bare-int/sentinel set: {sorted(unexpected)}"
        )

    return pd.Categorical(cleaned, categories=categories, ordered=True)


def normalize_loan_to_value(ltv_series):
    """Return ``loan_to_value_ratio`` as a nullable float, sentinels replaced first.

    MEASURED, 2026-09-11: the fixture carries duplicate string
    spellings of the same numeric LTV: e.g. ``"80"``, ``"80.0"`` and
    ``"80.00000"`` all present as distinct strings in
    ``tests/fixtures/hmda_50k.parquet`` (measured with
    ``df["loan_to_value_ratio"].astype(str).value_counts()``). The sentinels
    module docstring for this column explicitly defers this normalization
    ("normalize numeric-looking strings before comparing them, this module
    does not do that normalization itself"); this function is that
    normalization, added because no other module owns
    ``loan_to_value_ratio`` casting yet. It replaces sentinels first
    (:func:`replace_sentinels`), then casts the remaining numeric-looking
    strings to float, which collapses "80"/"80.0"/"80.00000" to the single
    float value 80.0.
    """
    cleaned = replace_sentinels(ltv_series, "loan_to_value_ratio")
    return pd.to_numeric(cleaned, errors="raise")
