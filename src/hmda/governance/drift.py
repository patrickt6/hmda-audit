"""Year-over-year population stability on the model inputs, reported as a count, not a raw PSI.

This module implements the "Drift" check: year-over-year population
stability on the model inputs, reported as a count of columns that
shifted past a stated threshold, not as a raw PSI number.
``hmda audit --drift`` (see ``cli.py``) prints ``N of M input columns
shifted past <threshold>``; this module is also runnable directly via
``python -m hmda.governance.drift``.
This is convenient when checking the module in isolation.

The rule that no metric with no named baseline or anchor ships raw is
applied here to ban raw PSI the same way it bans raw AUC, R2, and other
statistics elsewhere in this repository: PSI is computed internally as
the standard measure, but the outcome the reader sees is always the
translated count-past-threshold sentence, never the raw number itself.
See :func:`column_psi` and :func:`check_drift` for where this is enforced.

Threshold provenance
---------------------
``DEFAULT_PSI_THRESHOLD = 0.20`` matches a threshold commonly cited in
credit-risk model-monitoring practice as the cut for a "significant"
population shift (a widely used rule of thumb bins PSI as: < 0.10 no
material change, 0.10-0.25 moderate change worth investigating, > 0.25
significant change requiring action; some practitioners draw the
"significant" line at 0.20 rather than 0.25).

No single primary, citable source for that convention (e.g. a named
regulator standard or a specific textbook page) was located or opened.
It is carried over as an industry rule of thumb, not a verified citation,
and is recorded here as such rather than presented as sourced.
It is a single named module-level constant, not a magic number at call
sites, so it can be changed and re-cited in one place if a primary source
is later found.

Model inputs checked
---------------------
:data:`hmda.model.features.FEATURE_COLUMNS` (16 raw columns), the same
allowlist :func:`hmda.model.features.build_feature_matrix` reads, so
this checks exactly the columns the model consumes, not the full 99-column
file. Each raw column is cleaned with the SAME helper the model/cleaning
layers already use, so a value that would be treated as missing/normalized
for training is treated the same way here:

- ``income`` -> :func:`hmda.clean.sentinels.income_dollars`
- ``loan_to_value_ratio`` -> :func:`hmda.clean.sentinels.normalize_loan_to_value`
- ``debt_to_income_ratio`` -> :func:`hmda.model.features.debt_to_income_ordinal`
  (the features module's own ordinal encoding; its module docstring explains
  why this one is deliberately NOT delegated to ``clean.sentinels``, and this
  module follows that same decision rather than re-deriving a different one)
- ``loan_amount``, ``property_value`` -> generic numeric cast via
  :func:`hmda.clean.sentinels.replace_sentinels` then
  ``pandas.to_numeric``; no column-specific sentinels helper exists for
  these two, so the sentinel *replacement* is still the sentinels module's,
  only the final numeric cast is done here.
- every other ``FEATURE_COLUMNS`` entry (the 11 categorical fields) ->
  :func:`hmda.clean.sentinels.replace_sentinels`, compared as categories,
  not cast to a number.

No sentinel-detection logic is reimplemented in this file; every sentinel
string comes from ``clean.sentinels.KNOWN_SENTINELS`` via
``replace_sentinels``/``income_dollars``/``normalize_loan_to_value``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from hmda.clean import sentinels as _sentinels
from hmda.model.features import FEATURE_COLUMNS, NUMERIC_FEATURES, debt_to_income_ordinal

#: The PSI value above which a column is counted as "shifted". A parameter,
#: not a magic number silently used at call sites. See module docstring
#: "Threshold provenance" for what this is and is not sourced from.
DEFAULT_PSI_THRESHOLD = 0.20

#: Number of quantile bins used for PSI on numeric/ordinal columns. 10 is the
#: conventional PSI bin count in the same rule-of-thumb literature the
#: threshold is drawn from (deciles); not independently sourced beyond that,
#: same caveat as the threshold above.
_NUMERIC_BINS = 10

#: Columns whose cleaned values are used as an ordinal number rather than a
#: category, even though they are not in ``NUMERIC_FEATURES``.
_ORDINAL_COLUMNS = ("debt_to_income_ratio",)

#: Small floor so PSI's log ratio never divides by, or takes the log of,
#: zero on an empty bin. Standard PSI convention (a bin proportion of 0 is
#: replaced by a small epsilon, not treated as literally impossible).
_EPS = 1e-4


@dataclass(frozen=True)
class ColumnDrift:
    """One column's internal PSI result. Never printed with its numeric value
    in a user-facing string (that is the whole point of this check); carried only
    so the caller can decide "checked" vs "shifted" and, if it chooses, log
    the raw value to a file/comment/test, never to a report string."""

    column: str
    psi: float
    shifted: bool


@dataclass(frozen=True)
class DriftResult:
    """The drift check's user-facing result.

    columns_checked: total input columns checked for drift.
    columns_shifted: count of columns whose PSI (internal only) exceeded
        ``threshold``.
    threshold: the PSI threshold used.
    """

    columns_checked: int
    columns_shifted: int
    threshold: float
    year_a: int = field(default=0)
    year_b: int = field(default=0)
    per_column: tuple[ColumnDrift, ...] = field(default_factory=tuple)

    def sentence(self) -> str:
        """Return the allowed user-facing sentence: "N of M input columns shifted past <threshold>"."""
        years = f" between {self.year_a} and {self.year_b}" if self.year_a and self.year_b else ""
        return (
            f"{self.columns_shifted} of {self.columns_checked} input columns "
            f"shifted past {self.threshold}{years}"
        )


def _clean_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return ``column`` from ``frame`` cleaned the same way the model reads it.

    Numeric/ordinal columns come back as float (NaN for missing/sentinel).
    Categorical columns come back as nullable strings with sentinels
    replaced by <NA>. See module docstring for which helper backs each
    column.
    """
    if column == "income":
        return pd.Series(_sentinels.income_dollars(frame[column]), index=frame.index).astype(
            "float64"
        )
    if column == "loan_to_value_ratio":
        return pd.Series(
            _sentinels.normalize_loan_to_value(frame[column]), index=frame.index
        ).astype("float64")
    if column in _ORDINAL_COLUMNS:
        return debt_to_income_ordinal(frame)
    if column in NUMERIC_FEATURES:
        cleaned = _sentinels.replace_sentinels(frame[column], column)
        return pd.to_numeric(cleaned, errors="coerce").astype("float64")
    # Categorical: sentinel-replaced strings, compared as categories.
    return _sentinels.replace_sentinels(frame[column], column)


def _is_numeric_like(column: str) -> bool:
    return column in NUMERIC_FEATURES or column in _ORDINAL_COLUMNS


def _psi_numeric(a: pd.Series, b: pd.Series) -> float:
    """PSI for a numeric/ordinal column using quantile bins from the pooled sample.

    Bin edges are computed on the union of both years so both distributions
    are compared on the same bins, per the standard PSI construction. Rows
    with a missing value in either series are dropped before binning
    (missingness itself is not scored as a value here).
    """
    a = a.dropna()
    b = b.dropna()
    if a.empty or b.empty:
        return 0.0

    pooled = pd.concat([a, b], ignore_index=True)
    if pooled.nunique() <= 1:
        return 0.0

    try:
        edges = pd.qcut(pooled, q=_NUMERIC_BINS, retbins=True, duplicates="drop")[1]
    except ValueError:
        return 0.0
    if len(edges) < 3:
        # Fewer than 2 distinct bins came out (e.g. a near-constant column);
        # nothing meaningful to compare.
        return 0.0
    edges = edges.copy()
    edges[0] = -math.inf
    edges[-1] = math.inf

    a_bins = pd.cut(a, bins=edges, include_lowest=True)
    b_bins = pd.cut(b, bins=edges, include_lowest=True)
    a_props = a_bins.value_counts(normalize=True, sort=False)
    b_props = b_bins.value_counts(normalize=True, sort=False)
    return _psi_from_proportions(a_props, b_props)


def _psi_categorical(a: pd.Series, b: pd.Series) -> float:
    """PSI for a categorical column, bins = observed categories in either year."""
    a = a.dropna()
    b = b.dropna()
    if a.empty or b.empty:
        return 0.0
    a_props = a.value_counts(normalize=True)
    b_props = b.value_counts(normalize=True)
    categories = sorted(set(a_props.index) | set(b_props.index))
    a_props = a_props.reindex(categories, fill_value=0.0)
    b_props = b_props.reindex(categories, fill_value=0.0)
    return _psi_from_proportions(a_props, b_props)


def _psi_from_proportions(a_props: pd.Series, b_props: pd.Series) -> float:
    """The PSI formula: sum((p_a - p_b) * ln(p_a / p_b)) over aligned bins, floored by _EPS."""
    import numpy as np

    a_vals = a_props.to_numpy(dtype="float64")
    b_vals = b_props.reindex(a_props.index).fillna(0.0).to_numpy(dtype="float64")
    a_vals = a_vals.clip(min=_EPS)
    b_vals = b_vals.clip(min=_EPS)
    return float(((a_vals - b_vals) * np.log(a_vals / b_vals)).sum())


def column_psi(frame_year_a: pd.DataFrame, frame_year_b: pd.DataFrame, column: str) -> float:
    """Compute the internal PSI for one model-input ``column`` between two year-frames.

    Never call this to produce a user-facing string; its return value is the
    thing the metric-grammar rule bans from output. Used only inside
    :func:`check_drift` and by tests that assert the ban.
    """
    a = _clean_column(frame_year_a, column)
    b = _clean_column(frame_year_b, column)
    if _is_numeric_like(column):
        return _psi_numeric(a, b)
    return _psi_categorical(a, b)


def check_drift(
    frame_year_a, frame_year_b, columns: list[str] | None = None, threshold: float = DEFAULT_PSI_THRESHOLD
) -> DriftResult:
    """Compute PSI internally for each of ``columns`` between the two year-frames and count shifts past ``threshold``.

    ``columns`` defaults to :data:`hmda.model.features.FEATURE_COLUMNS` (the
    16 raw columns the model actually consumes), not every
    column in the file.

    The raw PSI number for each column is computed internally but must never
    reach a user-facing string; only the count in the returned
    :class:`DriftResult` is printed.
    """
    if columns is None:
        columns = list(FEATURE_COLUMNS)

    per_column: list[ColumnDrift] = []
    for column in columns:
        psi = column_psi(frame_year_a, frame_year_b, column)
        per_column.append(ColumnDrift(column=column, psi=psi, shifted=psi > threshold))

    shifted = sum(1 for c in per_column if c.shifted)
    return DriftResult(
        columns_checked=len(columns),
        columns_shifted=shifted,
        threshold=threshold,
        per_column=tuple(per_column),
    )


def check_drift_by_year(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = DEFAULT_PSI_THRESHOLD,
    year_column: str = "activity_year",
) -> list[DriftResult]:
    """Split ``frame`` by ``year_column`` and run :func:`check_drift` on every adjacent year pair.

    Years are sorted ascending; "adjacent" means consecutive entries in that
    sorted list of years PRESENT in the data, not consecutive calendar
    years (the fixture spans 2023, 2024, 2025 with no gap, but this does
    not assume a gap-free file for the national data).
    """
    years = sorted(pd.to_numeric(frame[year_column], errors="coerce").dropna().unique().astype(int))
    results: list[DriftResult] = []
    for year_a, year_b in zip(years, years[1:]):
        frame_a = frame.loc[pd.to_numeric(frame[year_column], errors="coerce") == year_a]
        frame_b = frame.loc[pd.to_numeric(frame[year_column], errors="coerce") == year_b]
        result = check_drift(frame_a, frame_b, columns=columns, threshold=threshold)
        result = DriftResult(
            columns_checked=result.columns_checked,
            columns_shifted=result.columns_shifted,
            threshold=result.threshold,
            year_a=year_a,
            year_b=year_b,
            per_column=result.per_column,
        )
        results.append(result)
    return results


def main() -> int:
    """Entry point for ``python -m hmda.governance.drift``.

    Loads the committed fixture, runs the year-over-year drift check on the
    model input columns, and prints one allowed sentence per adjacent year
    pair. Every line is explicitly labelled as a fixture-only number
    (the fixture is DC, WY and VT only, not national) and never
    prints a raw PSI value.
    """
    from hmda.clean import load_fixture

    frame = load_fixture()
    print("SOURCE: tests/fixtures/hmda_50k.parquet (50,000 rows, DC/WY/VT only).")
    print("These are FIXTURE numbers, not national. Do not quote them as national.")

    results = check_drift_by_year(frame)
    if not results:
        print("drift: fewer than two distinct activity_year values in the fixture; nothing to compare.")
        return 0

    for result in results:
        print(result.sentence())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
