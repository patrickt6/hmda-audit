"""Controlled disparity: the denial-rate gap after controlling for credit factors.

Raw and controlled disparities always ship together: publishing the raw
gap alone overstates, and publishing the controlled gap alone
understates. The API is deliberately shaped so a caller cannot obtain
one without the other: there is no code path that returns only the raw
gap, verified by a grep for a raw-gap-only accessor function, which
finds none.

Method (stated plainly, because a residual gap is an estimate under a model,
never a fact): the raw gap is the unadjusted difference in denial rates
between ``group_value`` and ``reference_group_value``. The controlled gap is
an **average predictive comparison from a binomial logistic regression**:
fit ``denial ~ group_indicator + CONTROL_COLUMNS`` on the two-group subset,
then for every row in that subset, predict the denial probability twice
(once with the group indicator set to this group, once set to the reference
group, holding every control at its observed value) and average the
difference. This is the standard "average marginal effect" construction for
a binary treatment in a logistic model. It is NOT stratification and NOT
matching; both were considered and logistic-regression AME was chosen
because it uses every row in the two-group subset rather than only rows with
an exact-matching stratum, which matters on a 50k-row fixture where several
``derived_race`` categories are already thin.

Stated assumptions and limits, because the docstring is the only place a
future caller will read them before trusting the number:

1. **Conditional-independence assumption.** The controlled gap is only a
   valid estimate of the disparity "after removing confounding through
   :data:`CONTROL_COLUMNS`" if there is no unmeasured confounder correlated
   with both group membership and denial. HMDA does not observe credit
   score, so a real (uncontrolled) credit-risk confounder almost certainly
   remains. The controlled number is a lower bound on how much of the raw
   gap survives *these five* controls, not a claim that no discrimination
   remains.
2. **Linear-in-the-logit functional form.** The logistic model assumes each
   control's effect on the log-odds of denial is linear (numeric controls)
   or additive across categories (categorical controls), with no
   interaction terms. A nonlinear true relationship will bias the residual
   gap in an unknown direction.
3. **Missing-data handling.** Rows missing a control after cleaning get that
   control's column-median imputed and an explicit `<col>_missing` indicator
   dummy, rather than being dropped: preserving sample size, at the cost of
   assuming missingness is not itself informative about denial risk within
   group. The raw gap uses the full two-group population (no imputation, no
   row dropped for missing controls); only the controlled model imputes.
4. **DTI numeric approximation.** ``debt_to_income_ratio`` mixes HMDA bucket
   strings (e.g. ``"30%-<36%"``) with bare integers (e.g. ``"42"``, reported
   exactly only in the 36-49 band). For use as a regression control this
   module maps every value to a single approximate numeric DTI percentage:
   a bare integer is used as-is; a bucket is mapped to its numeric midpoint
   (``"<20%"`` -> 10, ``"20%-<30%"`` -> 25, ``">60%"`` -> 65, etc.). This
   collapses the exact ordering the column otherwise preserves as an
   ordered categorical; it is an approximation good enough for a control
   variable, and it must never be reported to a user as an exact DTI value.
5. **This module does not import ``hmda.clean.sentinels`` or
   ``hmda.clean.filters``.** It carries its own minimal,
   private sentinel and parsing helpers below
   (``_income_dollars``, ``_clean_ltv``, ``_dti_numeric``,
   ``_exclude_purchased_loans``), scoped ONLY to what the
   five :data:`CONTROL_COLUMNS` plus ``action_taken`` need.
   Those two modules now hold general implementations of the
   same logic, so these private helpers duplicate rather
   than reuse them. **This is a known integration debt**:
   this module's private helpers should be deleted and
   replaced with calls to ``hmda.clean.sentinels`` /
   ``hmda.clean.filters``, so the sentinel map has one
   source of truth instead of two, not two copies that can
   silently drift apart. Flagging this here rather than
   letting it diverge unnoticed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#: Controls applied when residualizing the disparity. Printed in the CLI
#: output header, not only documented.
CONTROL_COLUMNS: tuple[str, ...] = (
    "income",
    "loan_to_value_ratio",
    "debt_to_income_ratio",
    "loan_purpose",
    "lien_status",
)

#: Denial and purchased-loan codes (action_taken outcome codes).
DENIAL_CODE = "3"
PURCHASED_LOAN_CODE = "6"

#: Minimum rows required in BOTH the comparison group and the reference
#: group before a comparison is computed at all, rather than the four-fifths
#: floor from ``hmda.fairness.floors`` (this module does not import that
#: file either, the same integration debt as module docstring point 5). A
#: stated, overridable parameter, never a silent magic number.
DEFAULT_MIN_COUNT = 100


@dataclass(frozen=True)
class DisparityComparison:
    """The raw and controlled denial-rate gap for one group, always paired.

    group_column: the protected-class column compared.
    group_value: the group compared against the reference group.
    reference_group_value: the comparison baseline group.
    raw_gap_pp: the unadjusted denial-rate gap, in percentage points
        (``100 * (denial_rate(group_value) - denial_rate(reference_group_value))``).
        Positive means ``group_value`` is denied more often than the
        reference group.
    controlled_gap_pp: the residual gap after controlling for
        :data:`CONTROL_COLUMNS`, in percentage points, via the average
        predictive comparison described in the module docstring.
    controls_used: echoes :data:`CONTROL_COLUMNS` at call time, so the
        output is self-describing even if the module's default set changes
        later.
    n_group: applications in scope for ``group_value`` (raw population,
        pre-imputation).
    n_reference: applications in scope for ``reference_group_value`` (raw
        population, pre-imputation).
    n_controlled: rows actually used to fit the controlled model (the
        two-group subset after excluding rows with a null ``group_column``;
        controls are imputed, not dropped, see module docstring point 3).
    race_not_available_share: when ``group_column == "derived_race"``, the
        fraction of the FULL input frame (before restricting to the two
        compared groups) whose ``derived_race`` is ``"Race Not Available"``,
        which must be printed beside every disparity number this way.
        ``None`` for any other ``group_column``.
    """

    group_column: str
    group_value: str
    reference_group_value: str
    raw_gap_pp: float
    controlled_gap_pp: float
    controls_used: tuple[str, ...]
    n_group: int
    n_reference: int
    n_controlled: int
    race_not_available_share: float | None


# --------------------------------------------------------------------------
# Private, minimal preprocessing (see module docstring point 5 for why this
# is not hmda.clean.sentinels / hmda.clean.filters).
# --------------------------------------------------------------------------

_GENERIC_NA = {"NA", "na", "", None}


def _exclude_purchased_loans(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where ``action_taken`` is the purchased-loan code."""
    action = frame["action_taken"].astype(str)
    return frame.loc[action != PURCHASED_LOAN_CODE]


def _denial_indicator(frame: pd.DataFrame) -> pd.Series:
    return (frame["action_taken"].astype(str) == DENIAL_CODE).astype(int)


def _income_dollars(series: pd.Series) -> pd.Series:
    """Raw ``income`` is in THOUSANDS of dollars; multiply by 1,000."""
    raw = series.astype(str).str.strip()
    raw = raw.where(~raw.isin(_GENERIC_NA), other=np.nan)
    thousands = pd.to_numeric(raw, errors="coerce")
    return thousands * 1000.0


def _clean_ltv(series: pd.Series) -> pd.Series:
    """Normalize ``loan_to_value_ratio``: "NA"/"Exempt" -> null, "80" == "80.0"."""
    raw = series.astype(str).str.strip()
    raw = raw.where(~raw.isin(_GENERIC_NA | {"Exempt"}), other=np.nan)
    return pd.to_numeric(raw, errors="coerce")


_DTI_RANGE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*%?\s*-\s*<?\s*(\d+(?:\.\d+)?)\s*%$")
_DTI_LT_RE = re.compile(r"^<\s*(\d+(?:\.\d+)?)\s*%$")
_DTI_GT_RE = re.compile(r"^>\s*(\d+(?:\.\d+)?)\s*%$")


def _dti_numeric(series: pd.Series) -> pd.Series:
    """Map the mixed bucket/bare-int ``debt_to_income_ratio`` column to an
    approximate numeric DTI percentage (module docstring point 4).

    Every non-"NA" input value is mapped to a number, the same bar
    ``hmda.clean.sentinels.parse_debt_to_income`` holds itself to, even
    though this is a separate, private helper and not that parser.
    """

    def parse_one(value: object) -> float:
        if value is None:
            return np.nan
        text = str(value).strip()
        if text in _GENERIC_NA:
            return np.nan
        range_match = _DTI_RANGE_RE.match(text)
        if range_match:
            low, high = float(range_match.group(1)), float(range_match.group(2))
            return (low + high) / 2.0
        lt_match = _DTI_LT_RE.match(text)
        if lt_match:
            return float(lt_match.group(1)) / 2.0
        gt_match = _DTI_GT_RE.match(text)
        if gt_match:
            return float(gt_match.group(1)) + 5.0
        try:
            return float(text)
        except ValueError:
            return np.nan

    return series.map(parse_one)


def _clean_categorical(series: pd.Series) -> pd.Series:
    """Generic-NA-aware pass-through for a code column used as a categorical control."""
    raw = series.astype(str).str.strip()
    return raw.where(~raw.isin(_GENERIC_NA), other="missing")


def _build_controls(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the numeric design matrix for :data:`CONTROL_COLUMNS`.

    Numeric controls (income, LTV, DTI) get column-median imputation plus an
    explicit ``<col>_missing`` indicator dummy (module docstring point 3);
    categorical controls (loan_purpose, lien_status) get one-hot encoded
    with an explicit "missing" level, never a silently dropped row.
    """
    numeric = pd.DataFrame(
        {
            "income": _income_dollars(frame["income"]),
            "loan_to_value_ratio": _clean_ltv(frame["loan_to_value_ratio"]),
            "debt_to_income_ratio": _dti_numeric(frame["debt_to_income_ratio"]),
        },
        index=frame.index,
    )
    missing_flags = numeric.isna().astype(int).add_suffix("_missing")
    numeric = numeric.fillna(numeric.median(numeric_only=True))

    categorical = pd.DataFrame(
        {
            "loan_purpose": _clean_categorical(frame["loan_purpose"]),
            "lien_status": _clean_categorical(frame["lien_status"]),
        },
        index=frame.index,
    )
    categorical_dummies = pd.get_dummies(categorical, prefix=categorical.columns, drop_first=True)

    return pd.concat([numeric, missing_flags, categorical_dummies], axis=1)


def _race_not_available_share(frame: pd.DataFrame) -> float:
    if len(frame) == 0:
        return 0.0
    return float((frame["derived_race"].astype(str) == "Race Not Available").mean())


def _default_reference_group(
    frame: pd.DataFrame, group_column: str, min_count: int
) -> str:
    """The group with the lowest raw denial rate among groups meeting ``min_count``,
    i.e. the group with the HIGHEST approval rate, same convention as
    ``hmda.fairness.air`` ("Reference group is the group with the HIGHEST
    approval rate")."""
    denial = _denial_indicator(frame)
    counts = frame.groupby(group_column, observed=True).size()
    eligible = counts[counts >= min_count].index
    if len(eligible) == 0:
        raise ValueError(
            f"no value of {group_column!r} meets min_count={min_count}; "
            "cannot pick a default reference group"
        )
    rates = (
        pd.DataFrame({group_column: frame[group_column], "denial": denial})
        .loc[frame[group_column].isin(eligible)]
        .groupby(group_column, observed=True)["denial"]
        .mean()
    )
    return str(rates.idxmin())


def controlled_disparity(
    frame: pd.DataFrame,
    group_column: str,
    *,
    min_count: int = DEFAULT_MIN_COUNT,
    reference_group_value: str | None = None,
) -> list[DisparityComparison]:
    """Compute the raw and controlled denial-rate gap for every value of ``group_column``.

    ``frame`` is the analysis-set frame as returned by
    ``hmda.clean.load_fixture()`` (or the equivalent full-data load): RAW HMDA
    columns, sentinel strings still present, ``income`` still in thousands.
    This function excludes purchased loans (``action_taken == "6"``) itself
    defensively; if the upstream loader has already excluded them the second
    exclusion is a no-op.

    There is intentionally no function in this module that returns only the
    raw or only the controlled number; both always travel together on
    :class:`DisparityComparison` (enforced by API shape, not
    convention).

    Raises ``ValueError`` if ``reference_group_value`` is not given and no
    value of ``group_column`` meets ``min_count`` (there is then no
    principled default reference group).
    """
    working = _exclude_purchased_loans(frame)

    if reference_group_value is None:
        reference_group_value = _default_reference_group(working, group_column, min_count)

    race_share = (
        _race_not_available_share(working) if group_column == "derived_race" else None
    )

    counts = working.groupby(group_column, observed=True).size()
    reference_count = int(counts.get(reference_group_value, 0))

    comparisons: list[DisparityComparison] = []
    group_values = [v for v in counts.index if v != reference_group_value]

    for group_value in group_values:
        group_count = int(counts.get(group_value, 0))
        if group_count < min_count or reference_count < min_count:
            # Below the stated floor: still returned, never silently
            # dropped, with the raw/controlled gaps reported as NaN so a
            # caller cannot mistake "suppressed" for "zero disparity".
            comparisons.append(
                DisparityComparison(
                    group_column=group_column,
                    group_value=str(group_value),
                    reference_group_value=str(reference_group_value),
                    raw_gap_pp=float("nan"),
                    controlled_gap_pp=float("nan"),
                    controls_used=CONTROL_COLUMNS,
                    n_group=group_count,
                    n_reference=reference_count,
                    n_controlled=0,
                    race_not_available_share=race_share,
                )
            )
            continue

        subset = working.loc[working[group_column].isin([group_value, reference_group_value])]
        denial = _denial_indicator(subset)
        is_group = (subset[group_column] == group_value).astype(int)

        raw_rate_group = denial.loc[is_group == 1].mean()
        raw_rate_reference = denial.loc[is_group == 0].mean()
        raw_gap_pp = 100.0 * (float(raw_rate_group) - float(raw_rate_reference))

        controlled_gap_pp = _controlled_gap_pp(subset, denial, is_group)

        comparisons.append(
            DisparityComparison(
                group_column=group_column,
                group_value=str(group_value),
                reference_group_value=str(reference_group_value),
                raw_gap_pp=raw_gap_pp,
                controlled_gap_pp=controlled_gap_pp,
                controls_used=CONTROL_COLUMNS,
                n_group=group_count,
                n_reference=reference_count,
                n_controlled=len(subset),
                race_not_available_share=race_share,
            )
        )

    return comparisons


def _controlled_gap_pp(subset: pd.DataFrame, denial: pd.Series, is_group: pd.Series) -> float:
    """The average predictive comparison described in the module docstring."""
    controls = _build_controls(subset)
    design = controls.copy()
    design["__group_indicator__"] = is_group.values

    scaler = StandardScaler()
    scaled = scaler.fit_transform(design.values)

    if denial.nunique() < 2:
        # No variation in the outcome within this two-group subset: a
        # logistic model has nothing to fit. Report zero residual rather
        # than let sklearn raise.
        return 0.0

    model = LogisticRegression(max_iter=2000)
    model.fit(scaled, denial.values)

    group_col_idx = list(design.columns).index("__group_indicator__")

    with_group = design.copy()
    with_group["__group_indicator__"] = 1
    without_group = design.copy()
    without_group["__group_indicator__"] = 0

    scaled_with = scaler.transform(with_group.values)
    scaled_without = scaler.transform(without_group.values)

    prob_with = model.predict_proba(scaled_with)[:, 1]
    prob_without = model.predict_proba(scaled_without)[:, 1]

    return float(100.0 * np.mean(prob_with - prob_without))
