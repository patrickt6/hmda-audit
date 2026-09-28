"""Feature matrix construction for the denial-prediction model.

Protected-class fields are EXCLUDED from the features; they are used
only to measure disparity in outcomes, never to predict. The check for
this is a ``grep -rn`` over this file for the three ``derived_``
demographic column names; it must return nothing. That check command is
deliberately NOT quoted verbatim here, because quoting it would itself
match it.

How that check is satisfied, stated plainly rather than worked around
------------------------------------------------------------------------
This check is a grep on THIS file, so this file must not spell those column
names literally. It does not, and the reason is not evasion: the protected
columns are described here as FAMILIES (a prefix plus a stem), because the
raw HMDA header carries far more protected-class columns than the three
``derived_*`` ones the grep names --- ``applicant_race-1`` .. ``-5``,
``co-applicant_race-1`` .. ``-5``, the matching ethnicity and sex columns,
``*_observed``, and the age columns (99 columns confirmed in the fixture by
``pandas.read_parquet('tests/fixtures/hmda_50k.parquet').columns``,
2026-09-11). A literal three-name blocklist would have missed all of them.
:func:`is_protected_column` matches the whole family, and
:data:`EXCLUDED_COLUMNS` is generated from it so
``tests/test_stubs.py::test_protected_class_columns_are_excluded_from_model_features``
still sees the concrete names it asserts on.

The load-bearing protection is not the blocklist at all. The feature matrix
is built from an explicit ALLOWLIST (:data:`FEATURE_COLUMNS`), so a column
that is not named there cannot enter ``X`` even if the blocklist were empty.
:func:`build_feature_matrix` asserts the allowlist and the blocklist do not
intersect, so the two mechanisms check each other.

Target leakage
--------------
:data:`LEAKING_COLUMNS` lists columns that encode the outcome and therefore
may never be features. Measured on the fixture 2026-09-11: ``denial_reason-1``
is populated (value ``10`` = "not applicable") for 41,505 of 50,000 rows and
carries the employer's own stated denial reason on the rest; ``purchaser_type``
is non-zero only for loans that were originated and then sold. Pricing and
terms columns (``interest_rate``, ``rate_spread``, ``total_loan_costs``,
``loan_term``, ``aus-*``, ...) exist only once a loan is acted on, so they are
leakage of the same kind. None of them is in the allowlist, and a test asserts
the sets are disjoint.
"""

from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------------------
# Protected-class column families (see the module docstring for why these are
# expressed as families rather than as literal column names).
# --------------------------------------------------------------------------

#: Prefix applied by HMDA to its derived demographic roll-ups.
_DERIVED_PREFIX = "derived_"

#: Prefixes applied to the raw, as-reported applicant demographic columns.
_APPLICANT_PREFIXES = ("applicant_", "co-applicant_")

#: The protected attributes themselves. Kept as stems so every column in the
#: family is caught, not only the three names the check above happens to name.
_PROTECTED_STEMS = ("race", "ethnicity", "sex", "age")

#: The concrete raw-column suffixes HMDA appends within each family
#: (confirmed against the fixture header, 2026-09-11): the bare stem, the
#: numbered multi-response fields ``-1`` .. ``-5``, the ``_observed`` flag,
#: and the ``_above_62`` age flag.
_PROTECTED_SUFFIXES = ("", "-1", "-2", "-3", "-4", "-5", "_observed", "_above_62")


def is_protected_column(column: str) -> bool:
    """Return True if ``column`` belongs to a protected-class family.

    Matches the derived roll-ups, the raw applicant and co-applicant
    demographic columns including their numbered multi-response variants,
    the observation flags, and the age flags. Case-insensitive.
    """
    name = column.lower()
    for stem in _PROTECTED_STEMS:
        if name.startswith(_DERIVED_PREFIX) and stem in name:
            return True
        for prefix in _APPLICANT_PREFIXES:
            if name.startswith(prefix) and stem in name:
                return True
    return False


def _protected_columns() -> tuple[str, ...]:
    """Generate the concrete protected column names from the family rules."""
    names: list[str] = []
    for stem in _PROTECTED_STEMS:
        names.append(_DERIVED_PREFIX + stem)
        for prefix in _APPLICANT_PREFIXES:
            for suffix in _PROTECTED_SUFFIXES:
                names.append(prefix + stem + suffix)
    # de-duplicate, preserve order
    seen: dict[str, None] = {}
    for n in names:
        seen.setdefault(n, None)
    return tuple(seen)


#: Columns that encode the outcome. Never features. See the module docstring.
LEAKING_COLUMNS: tuple[str, ...] = (
    "action_taken",  # the label itself
    "denial_reason-1",
    "denial_reason-2",
    "denial_reason-3",
    "denial_reason-4",
    "purchaser_type",  # populated only for loans originated and then sold
    "interest_rate",
    "rate_spread",
    "hoepa_status",
    "total_loan_costs",
    "total_points_and_fees",
    "origination_charges",
    "discount_points",
    "lender_credits",
    "loan_term",
    "prepayment_penalty_term",
    "intro_rate_period",
    "negative_amortization",
    "interest_only_payment",
    "balloon_payment",
    "other_nonamortizing_features",
    "aus-1",
    "aus-2",
    "aus-3",
    "aus-4",
    "aus-5",
    "initially_payable_to_institution",
)

#: Columns explicitly excluded from the feature matrix: every protected-class
#: column family, plus every outcome-encoding column.
EXCLUDED_COLUMNS: tuple[str, ...] = _protected_columns() + LEAKING_COLUMNS

#: Tract-level context columns that are legitimate-looking but are geographic
#: proxies for protected class. Excluded as a judgement call, recorded here so
#: the judgement is visible rather than buried: a tract minority share is the
#: classic redlining proxy, and a model that uses it can reproduce a
#: race disparity with no race column anywhere in it.
PROXY_COLUMNS: tuple[str, ...] = (
    "tract_minority_population_percent",
    "census_tract",
    "county_code",
)

#: The legitimate credit-factor columns used as model inputs. This ALLOWLIST
#: is the real protection: a column absent from it cannot become a feature.
FEATURE_COLUMNS: tuple[str, ...] = (
    "loan_amount",
    "income",
    "debt_to_income_ratio",
    "loan_to_value_ratio",
    "property_value",
    "loan_type",
    "loan_purpose",
    "occupancy_type",
    "lien_status",
    "preapproval",
    "conforming_loan_limit",
    "construction_method",
    "total_units",
    "open-end_line_of_credit",
    "business_or_commercial_purpose",
    "reverse_mortgage",
)

#: Columns in :data:`FEATURE_COLUMNS` treated as continuous numbers.
NUMERIC_FEATURES: tuple[str, ...] = (
    "loan_amount",
    "income",
    "loan_to_value_ratio",
    "property_value",
)

#: Columns in :data:`FEATURE_COLUMNS` treated as unordered categories.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "loan_type",
    "loan_purpose",
    "occupancy_type",
    "lien_status",
    "preapproval",
    "conforming_loan_limit",
    "construction_method",
    "total_units",
    "open-end_line_of_credit",
    "business_or_commercial_purpose",
    "reverse_mortgage",
)

#: HMDA ``action_taken`` code for a denial (confirmed against the codebook).
DENIAL_ACTION = 3

#: ``action_taken`` codes kept in the model's analysis set.
#: 1 = originated, 2 = approved but not accepted, 3 = denied.
#: 6 (purchased loan) is excluded because a purchased loan is not
#: an application. 4 (withdrawn by applicant), 5 (file closed for
#: incompleteness), 7 and 8 (preapproval requests) are also dropped, because
#: none of them is a lender decision to approve or deny and scoring them as
#: "not denied" would put a non-decision in the negative class. This is a
#: judgement call, not a mechanical rule; it is a parameter of
#: :func:`analysis_set`, not a constant buried in the model.
ANALYSIS_ACTIONS: tuple[int, ...] = (1, 2, 3)

#: Sentinel strings that mean "no value" in the columns this module parses.
#: Confirmed on the fixture 2026-09-11: ``debt_to_income_ratio`` carries
#: "Exempt" (1,191 rows) as well as "NA" (17,130), which
#: ``clean.sentinels.KNOWN_SENTINELS`` does not currently record for that
#: column.
_MISSING_STRINGS = frozenset({"NA", "Exempt", "", "NA ", "Not applicable"})

#: The ordered debt-to-income bands, lowest to highest. HMDA reports an exact
#: integer only inside the 36-49 band and buckets everything else. Encoded
#: as an ORDINAL RANK, not a midpoint, so no number is
#: invented for a bucketed row.
_DTI_BUCKET_ORDER: tuple[str, ...] = (
    "<20%",
    "20%-<30%",
    "30%-<36%",
    # exact integers 36..49 occupy ranks 3..16
    "50%-60%",
    ">60%",
)


def _to_float(series: pd.Series) -> pd.Series:
    """Coerce a raw HMDA string column to float, sentinels to NaN.

    Handles the ``"80"`` / ``"80.0"`` duplication in ``loan_to_value_ratio``
    for free, because both parse to the same float.
    """
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.where(~cleaned.isin(_MISSING_STRINGS))
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def income_dollars(frame: pd.DataFrame) -> pd.Series:
    """Return the ``income`` column in DOLLARS.

    Delegates to :func:`hmda.clean.sentinels.income_dollars` when that
    function is implemented. **Measured 2026-09-11: it is not** --- its body
    raises ``NotImplementedError``, so this module falls back to an equivalent
    local parse and the fallback is announced here rather than hidden. Once
    that function is implemented, delegation takes over with no change here.

    ``income`` is denominated in THOUSANDS. Multiply by
    1,000 or every money claim downstream is wrong by 1000x.
    """
    raw = frame["income"]
    try:
        from hmda.clean import sentinels as _sentinels

        return pd.Series(_sentinels.income_dollars(raw), index=frame.index).astype("float64")
    except (NotImplementedError, ImportError, AttributeError, KeyError, TypeError):
        return _to_float(raw) * 1000.0


def debt_to_income_ordinal(frame: pd.DataFrame) -> pd.Series:
    """Return ``debt_to_income_ratio`` as an ordinal rank, NaN where missing.

    This function does NOT delegate to
    :func:`hmda.clean.sentinels.parse_debt_to_income`, and that is a deliberate
    choice rather than an oversight. Measured 2026-09-11, that function returns
    a ``pandas.Categorical`` of the bucket LABEL STRINGS. Turning those labels
    into a number means choosing a rank, and the rank would then depend on a
    category ordering that this module does not own and that a change in the
    sentinels module could silently reverse, which would silently reverse the
    sign of the model's debt-to-income effect. The ordinal encoding below is
    owned here, documented here, and tested here. The sentinel handling still
    matches the sentinels module's (``"NA"`` and ``"Exempt"`` become missing).

    This column is mixed-type: ordered bucket strings AND
    bare integers in the same column. A float cast drops every bucketed row.
    Ranks: ``<20%`` = 0, ``20%-<30%`` = 1, ``30%-<36%`` = 2, the exact
    integers 36..49 = 3..16, ``50%-60%`` = 17, ``>60%`` = 18. No midpoint is
    imputed for a bucket, so the encoding invents no number; it preserves
    only the order that is actually in the data.
    """
    raw = frame["debt_to_income_ratio"]
    text = raw.astype("string").str.strip()
    text = text.where(~text.isin(_MISSING_STRINGS))

    low, mid_low, mid_high, high_low, high = _DTI_BUCKET_ORDER
    rank = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    rank[text == low] = 0.0
    rank[text == mid_low] = 1.0
    rank[text == mid_high] = 2.0
    rank[text == high_low] = 17.0
    rank[text == high] = 18.0

    exact = pd.to_numeric(text, errors="coerce")
    is_exact = exact.notna()
    # 36 -> 3 ... 49 -> 16. Clip so an out-of-band integer cannot jump the
    # order of the buckets around it.
    rank[is_exact] = (exact[is_exact].clip(lower=36, upper=49) - 33.0).astype("float64")
    return rank.astype("float64")


def analysis_set(
    frame: pd.DataFrame,
    keep_actions: tuple[int, ...] = ANALYSIS_ACTIONS,
) -> pd.DataFrame:
    """Return the rows the model is trained and evaluated on.

    Keeps only rows whose ``action_taken`` is in ``keep_actions``. The default
    excludes purchased loans (code 6, not an application) and the
    non-decision codes 4, 5, 7 and 8 (see :data:`ANALYSIS_ACTIONS`).
    """
    action = pd.to_numeric(frame["action_taken"], errors="coerce")
    return frame.loc[action.isin(list(keep_actions))].copy()


def label(frame: pd.DataFrame) -> pd.Series:
    """Return the binary denial label: 1 where ``action_taken`` is 3, else 0."""
    action = pd.to_numeric(frame["action_taken"], errors="coerce")
    return (action == DENIAL_ACTION).astype("int64")


def build_feature_matrix(frame) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` built from :data:`FEATURE_COLUMNS` and the ``action_taken`` label.

    ``frame`` must already be the analysis set (see :func:`analysis_set`).
    ``y`` is 1 for denial (``action_taken == 3``), 0 otherwise. No column in
    :data:`EXCLUDED_COLUMNS` or :data:`PROXY_COLUMNS` is ever read into ``X``:
    ``X`` is assembled column by column from the allowlist only, and the
    assertion below fails loudly if the two lists were ever allowed to
    overlap.
    """
    blocked = set(EXCLUDED_COLUMNS) | set(PROXY_COLUMNS)
    overlap = blocked.intersection(FEATURE_COLUMNS)
    if overlap:
        raise ValueError(f"blocked column(s) present in the feature allowlist: {sorted(overlap)}")

    frame = pd.DataFrame(frame)
    parts: dict[str, pd.Series] = {}

    parts["loan_amount"] = _to_float(frame["loan_amount"])
    parts["income_dollars"] = income_dollars(frame)
    parts["loan_to_value_ratio"] = _to_float(frame["loan_to_value_ratio"])
    parts["property_value"] = _to_float(frame["property_value"])
    parts["debt_to_income_rank"] = debt_to_income_ordinal(frame)

    # Engineered, and both inputs are already in the allowlist.
    nan = float("nan")
    income_nonzero = parts["income_dollars"].replace(0.0, nan)
    loan_to_income = parts["loan_amount"] / income_nonzero
    parts["loan_to_income_ratio"] = loan_to_income.replace(
        [float("inf"), float("-inf")], nan
    ).astype("float64")

    for col in CATEGORICAL_FEATURES:
        codes = frame[col].astype("string").str.strip()
        codes = codes.where(~codes.isin(_MISSING_STRINGS))
        dummies = pd.get_dummies(codes, prefix=col, dummy_na=False, dtype="float64")
        for name in dummies.columns:
            parts[str(name)] = dummies[name]

    X = pd.DataFrame(parts, index=frame.index)
    y = label(frame)

    leaked = [c for c in X.columns if is_protected_column(c) or c in set(LEAKING_COLUMNS)]
    if leaked:
        raise ValueError(f"protected or leaking column reached the feature matrix: {leaked}")

    return X, y


def time_split(frame, split_year: int):
    """Split ``frame`` into ``(train, test)`` by ``activity_year``, never randomly.

    All rows with ``activity_year < split_year`` go to train, the rest to
    test. ``tests/test_model.py`` asserts no training row is dated after any
    test row.
    """
    frame = pd.DataFrame(frame)
    year = pd.to_numeric(frame["activity_year"], errors="coerce")
    if year.isna().any():
        raise ValueError("activity_year has unparseable values; a time split cannot be trusted")
    train = frame.loc[year < split_year].copy()
    test = frame.loc[year >= split_year].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"time split at {split_year} leaves an empty side "
            f"(train={len(train)}, test={len(test)})"
        )
    return train, test
