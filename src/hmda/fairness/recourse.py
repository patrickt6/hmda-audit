"""Recourse / effort study (M13): the minimum actionable change to cross the approval line.

The recourse study is the repo's one original
research claim. **May legitimately fail**: if the method produces no stable
number, this module reports it as UNMEASURABLE with a written reason in
``docs/RECOURSE.md`` and never substitutes a proxy.

What this module computes
-------------------------
For each applicant the data records as DENIED, and whom the fitted model also
scores on the denial side of its decision line, the module searches for the
smallest INCOME INCREASE, in dollars, that moves the model's predicted denial
probability below that line, holding every other input fixed. The per-group
headline is the MEDIAN of those dollar increases, always printed with the
count of applicants it covers and the share for whom no increase inside the
search cap crosses the line.

**This is a property of a MODEL fitted to historical application records. It
is not a statement about what any real lender requires of any real applicant,
and the model makes and influences no credit decision**.

Method, and why this simple one
-------------------------------
A one-dimensional GRID search over income, not a bisection and not a
gradient/optimiser counterfactual search. A gradient-boosted model is a step
function and is not guaranteed monotone in income, so bisection can report a
crossing that does not exist or miss one that does. A grid evaluates the real
model at every candidate and takes the smallest candidate that actually
crosses; it cannot converge to a false root. The cost is resolution, which is
an explicit, reported parameter (:data:`GRID_STEP_USD`), not a hidden one.

Actionable inputs only
----------------------
:data:`ACTIONABLE_FEATURES` is the explicit list, in the order this module
declares them. Income is the one this module searches over. Loan amount and
loan-to-value are declared here and NOT searched, see
:data:`UNSEARCHED_ACTIONABLE` for why that is a stated limit rather than an
omission. A protected-class field is never an actionable input: an applicant
cannot change race, so a recourse number computed over such a feature would
be meaningless. The feature matrix this module perturbs comes from
``model.features.build_feature_matrix``, which is built from an allowlist
that contains no protected column at all, so no protected field can enter the
search even by accident.

Grouping is a separate question from actionability. The comparison is BY
protected group, so a protected column name is needed to split the rows; this
module never spells one, it iterates
``hmda.fairness.rates.ALLOWED_GROUP_COLUMNS``, and those columns are used only
to partition rows for reporting, never as an input the search may move.

Fixture scope
-------------
Every number this module prints is measured on
``tests/fixtures/hmda_50k.parquet``, which is DC / WY / VT only. It is NOT a
national sample, and no group may be characterised nationally from it. The
printed output carries that label on every run.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: The only inputs recourse may treat as changeable, in the order this
#: module declares them. Deliberately excludes every protected-class column.
ACTIONABLE_FEATURES: tuple[str, ...] = (
    "income",
    "loan_amount",
    "loan_to_value_ratio",
)

#: Of :data:`ACTIONABLE_FEATURES`, the ones this implementation declares but
#: does NOT search, recorded so the limit is visible rather than implied.
#: Income is searched because M13's required output shape is a MONEY figure
#: ("need [$X] more income"); a loan-amount or LTV change is a different
#: claim in different units and mixing the three into one "minimum cost"
#: would require a cost model trading dollars of income against dollars of
#: loan, which this repo has no evidence for and will not invent.
UNSEARCHED_ACTIONABLE: tuple[str, ...] = ("loan_amount", "loan_to_value_ratio")

#: The feature-matrix column holding income in DOLLARS. Produced by
#: ``model.features.build_feature_matrix``, which routes ``income`` through
#: ``clean.sentinels.income_dollars`` (x1000). Getting this
#: unit wrong makes the headline money figure wrong by 1000x.
INCOME_COLUMN = "income_dollars"

#: The engineered feature that depends on income and must be recomputed at
#: every grid point, or the counterfactual would be internally inconsistent.
LOAN_TO_INCOME_COLUMN = "loan_to_income_ratio"

#: Grid resolution, in dollars of additional annual income. The reported
#: median is therefore accurate to this step, and that is stated in the
#: output.
GRID_STEP_USD = 1_000.0

#: Largest income increase the search will consider. An applicant needing
#: more than this is counted as UNREACHABLE rather than assigned a number.
#: $1,000,000 of additional ANNUAL income is far outside any plausible
#: recourse, so treating it as "no achievable change" is conservative.
MAX_INCREASE_USD = 1_000_000.0

#: Minimum denied-applicant count before a group's median is reported at all
#: (small denominators give unstable numbers, and the floor
#: must be a stated parameter, not a magic number).
MIN_GROUP_COUNT = 30

#: The committed fixture. DC/WY/VT only, never national. Never reads ``data/``.
FIXTURE_PATH = Path("tests/fixtures/hmda_50k.parquet")

#: Same time split as ``model/evaluate.py`` uses, so the study runs on the
#: held-out fold of the same model.
DEFAULT_SPLIT_YEAR = 2025


@dataclass(frozen=True)
class RecourseResult:
    """The recourse figure for one protected group, over denied applicants.

    group_column: the protected-class column the rows were partitioned by.
    group_value: the group this row covers.
    applicants_count: denied applicants the figure was computed over, the
        ones the model also scores on the denial side of its line, so there
        is a line for them to cross. Always printed with the figure.
    reachable_count: of those, how many had SOME increase within
        :data:`MAX_INCREASE_USD` that crossed the line.
    unreachable_share: ``1 - reachable_count / applicants_count``; the share
        for whom no achievable income increase crosses the line.
    median_required_income_increase_usd: median additional ANNUAL income in
        real dollars over the reachable applicants, or ``None`` when the
        group is below :data:`MIN_GROUP_COUNT` or nobody was reachable.
    below_floor: True when ``applicants_count < MIN_GROUP_COUNT``.
    """

    group_column: str
    group_value: str
    applicants_count: int
    reachable_count: int
    unreachable_share: float
    median_required_income_increase_usd: float | None
    below_floor: bool

    def sentence(self) -> str:
        """One line: a MONEY figure, its count, and its unreachable share.

        Never a raw distance, a raw probability, or a cost in model
        units.
        """
        if self.below_floor:
            return (
                f"{self.group_value}: NOT REPORTED, only {self.applicants_count} denied "
                f"applicants, below the stated floor of {MIN_GROUP_COUNT}"
            )
        if self.median_required_income_increase_usd is None:
            return (
                f"{self.group_value}: UNMEASURABLE, no achievable income increase "
                f"crossed the line for any of {self.applicants_count} denied applicants "
                f"(unreachable share 100.0%)"
            )
        return (
            f"{self.group_value}: denied applicants need ${self.median_required_income_increase_usd:,.0f} "
            f"more annual income to reach the same approval line "
            f"(median over {self.reachable_count} of {self.applicants_count} denied applicants; "
            f"{self.unreachable_share * 100:.1f}% unreachable)"
        )


def income_grid(
    step_usd: float = GRID_STEP_USD, max_usd: float = MAX_INCREASE_USD
) -> np.ndarray:
    """Return the candidate income increases, in dollars, smallest first.

    A uniform grid at ``step_usd`` up to $100,000, then a coarser geometric
    tail to ``max_usd``. The fine head is where almost every real answer
    lands; the tail exists only to separate "needs an implausible amount"
    from "unreachable at any amount", so its resolution does not need to be
    fine and a fine one would cost a thousand model calls for no information.
    """
    head = np.arange(step_usd, min(100_000.0, max_usd) + step_usd, step_usd)
    if max_usd <= 100_000.0:
        return head
    tail = np.geomspace(110_000.0, max_usd, 40)
    return np.unique(np.concatenate([head, tail]))


def _denial_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Return one predicted denial probability per row."""
    return np.asarray(model.predict_proba(X))[:, 1]


def decision_threshold(model, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """Return the score above which the model is treated as saying 'deny'.

    Chosen as the base-rate cut: the quantile of the TRAINING scores that
    puts the same share of rows on the denial side as were actually denied
    in training. This is a stated modelling choice, not a lender's rule:
    no lender's threshold is known to this repo, and none is claimed. It is
    fitted on the training fold only, so the held-out study fold does not
    inform the line it is measured against.
    """
    scores = _denial_scores(model, X_train)
    base_rate = float(np.mean(np.asarray(y_train)))
    if not 0.0 < base_rate < 1.0:
        raise ValueError(f"training denial base rate {base_rate} leaves no decision line to draw")
    return float(np.quantile(scores, 1.0 - base_rate))


def required_income_increase(
    model,
    X: pd.DataFrame,
    threshold: float,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """For each row of ``X``, the smallest income increase that crosses ``threshold``.

    Returns a float array, one entry per row, in DOLLARS. ``NaN`` means no
    candidate on the grid brought the predicted denial probability below
    ``threshold`` (unreachable). ``0.0`` means the row is already below the
    line with no change at all.

    ``X`` must be a feature matrix from
    ``model.features.build_feature_matrix``: this function perturbs
    :data:`INCOME_COLUMN` and recomputes :data:`LOAN_TO_INCOME_COLUMN` so the
    counterfactual row stays internally consistent. Every other column is
    held fixed. The search is a grid, evaluated against the real model at
    every candidate, so it never assumes the model is monotone in income.
    """
    if INCOME_COLUMN not in X.columns:
        raise KeyError(f"feature matrix has no {INCOME_COLUMN!r} column; cannot search over income")
    if grid is None:
        grid = income_grid()

    n = len(X)
    answer = np.full(n, np.nan, dtype="float64")
    if n == 0:
        return answer

    base_income = X[INCOME_COLUMN].to_numpy(dtype="float64")
    has_loan_to_income = LOAN_TO_INCOME_COLUMN in X.columns
    loan_amount = (
        X[LOAN_TO_INCOME_COLUMN].to_numpy(dtype="float64") * base_income
        if has_loan_to_income
        else None
    )

    pending = _denial_scores(model, X) >= threshold
    answer[~pending] = 0.0

    work = X.copy()
    for increase in grid:
        if not pending.any():
            break
        idx = np.flatnonzero(pending)
        trial = work.iloc[idx].copy()
        new_income = base_income[idx] + float(increase)
        trial[INCOME_COLUMN] = new_income
        if has_loan_to_income:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(new_income != 0.0, loan_amount[idx] / new_income, np.nan)
            trial[LOAN_TO_INCOME_COLUMN] = ratio
        crossed = _denial_scores(model, trial) < threshold
        if crossed.any():
            answer[idx[crossed]] = float(increase)
            pending[idx[crossed]] = False
    return answer


def recourse_by_group(
    frame: pd.DataFrame,
    model,
    group_column: str,
    threshold: float,
    grid: np.ndarray | None = None,
    min_group_count: int = MIN_GROUP_COUNT,
    feature_columns: "pd.Index | None" = None,
) -> list[RecourseResult]:
    """Median required income increase, in dollars, per group, over denied applicants.

    ``frame`` is a RAW HMDA frame (it still carries ``action_taken`` and the
    grouping column). Rows are restricted to those the data records as denied
    AND that the model scores on the denial side of ``threshold``: a row the
    model already puts on the approval side has no line to cross and would
    contribute a meaningless 0. The feature matrix is rebuilt from the
    allowlist, so no protected column reaches the model or the search.

    Returns one :class:`RecourseResult` per observed group value, sorted by
    group value. Groups below ``min_group_count`` are returned flagged, never
    dropped silently.
    """
    from hmda.model import features as F

    frame = pd.DataFrame(frame)
    if group_column not in frame.columns:
        raise KeyError(f"frame has no grouping column {group_column!r}")

    X, y = F.build_feature_matrix(frame)
    if feature_columns is not None:
        # One-hot columns differ between folds; the model can only be scored
        # on the schema it was fitted on.
        X = X.reindex(columns=feature_columns, fill_value=0.0)
    groups = frame[group_column].astype("string").fillna("NA")

    scores = _denial_scores(model, X)
    denied_by_model = scores >= threshold
    selected = (y.to_numpy() == 1) & denied_by_model

    # POSITIONAL selection throughout. A label-based ``groups.loc[X_sel.index]``
    # silently duplicates rows when the frame carries a repeated index, which a
    # bootstrap resample always does; that raised an IndexError in a stability
    # check on 2026-09-11 and would otherwise have inflated a group's count.
    X_sel = X.loc[selected]
    groups_sel = pd.Series(groups.to_numpy()[selected], dtype="string")

    increases = required_income_increase(model, X_sel, threshold, grid)

    results: list[RecourseResult] = []
    for value in sorted(groups_sel.dropna().unique().tolist()):
        mask = (groups_sel == value).to_numpy()
        count = int(mask.sum())
        vals = increases[mask]
        reachable = vals[~np.isnan(vals)]
        reachable_count = int(len(reachable))
        unreachable_share = 1.0 - (reachable_count / count) if count else 1.0
        below_floor = count < min_group_count
        median = (
            float(np.median(reachable))
            if reachable_count > 0 and not below_floor
            else None
        )
        results.append(
            RecourseResult(
                group_column=group_column,
                group_value=str(value),
                applicants_count=count,
                reachable_count=reachable_count,
                unreachable_share=float(unreachable_share),
                median_required_income_increase_usd=median,
                below_floor=below_floor,
            )
        )
    return results


def run_study(
    fixture_path: Path = FIXTURE_PATH,
    split_year: int = DEFAULT_SPLIT_YEAR,
    frame=None,
    source_label: str | None = None,
    sample_seed: int | None = None,
) -> dict:
    """Fit the challenger on the training fold and run the study on the held-out fold.

    Returns the fitted model, the threshold, the median income sanity check,
    and one list of :class:`RecourseResult` per allowed grouping column.

    ``frame`` is an already-materialized ``pandas.DataFrame`` of RAW
    rows, normally ``hmda.clean.source.Source.sample(n, seed)``. When it is
    given, ``fixture_path`` is NOT read and every figure is a SAMPLE figure;
    ``source_label`` and ``sample_seed`` come back in the returned dict so the
    caller states them before any number. With ``frame=None`` the behaviour is
    unchanged. Per-group denied-applicant counts are already printed beside
    every figure by :meth:`RecourseResult.sentence`.
    """
    from hmda.fairness.rates import ALLOWED_GROUP_COLUMNS
    from hmda.model import features as F
    from hmda.model.gbm import fit_gbm

    if frame is not None and source_label is None:
        raise ValueError(
            "run_study() got a frame with no source_label: a sample number would be "
            "printed under the fixture's DC/WY/VT label. Pass source_label describing "
            "what frame is (e.g. from Source.sample()), or pass frame=None to use the "
            "committed fixture."
        )
    raw = frame if frame is not None else pd.read_parquet(fixture_path)
    analysis = F.analysis_set(raw)
    train_rows, test_rows = F.time_split(analysis, split_year)

    X_train, y_train = F.build_feature_matrix(train_rows)
    model = fit_gbm(X_train, y_train)

    threshold = decision_threshold(model, X_train, y_train)
    median_income = float(np.nanmedian(X_train[INCOME_COLUMN].to_numpy(dtype="float64")))

    by_column = {
        column: recourse_by_group(
            test_rows, model, column, threshold, feature_columns=X_train.columns
        )
        for column in sorted(ALLOWED_GROUP_COLUMNS)
    }
    return {
        "source_label": source_label,
        "sample_seed": sample_seed,
        "fixture_path": str(fixture_path),
        "rows_raw": len(raw),
        "rows_train": len(train_rows),
        "rows_test": len(test_rows),
        "split_year": split_year,
        "threshold": threshold,
        "median_income_dollars": median_income,
        "results": by_column,
    }


def stability_check(
    group_column: str,
    n_bootstrap: int = 10,
    fixture_path: Path = FIXTURE_PATH,
    split_year: int = DEFAULT_SPLIT_YEAR,
    frame=None,
) -> dict[str, dict]:
    """Resample the study fold and report how far each group's median moves.

    This is the function that decided M13's verdict, so it is shipped rather
    than run once in a scratch file and quoted. It refits nothing: the model
    and the decision line are held fixed and only the study fold is
    resampled, so what it measures is sampling variability alone, the
    smaller of the two instabilities found (the other is the model seed, see
    ``docs/RECOURSE.md``).

    Returns ``{group_value: {"medians": [...], "unreachable": [...]}}``.
    """
    from hmda.model import features as F
    from hmda.model.gbm import fit_gbm

    raw = frame if frame is not None else pd.read_parquet(fixture_path)
    train_rows, test_rows = F.time_split(F.analysis_set(raw), split_year)
    X_train, y_train = F.build_feature_matrix(train_rows)
    model = fit_gbm(X_train, y_train)
    threshold = decision_threshold(model, X_train, y_train)

    out: dict[str, dict] = {}
    for seed in range(n_bootstrap):
        sample = test_rows.sample(frac=1.0, replace=True, random_state=seed)
        for result in recourse_by_group(
            sample, model, group_column, threshold, feature_columns=X_train.columns
        ):
            if result.below_floor:
                continue
            entry = out.setdefault(result.group_value, {"medians": [], "unreachable": []})
            entry["medians"].append(result.median_required_income_increase_usd)
            entry["unreachable"].append(result.unreachable_share)
    return out


#: The M13 verdict, measured not asserted. See ``docs/RECOURSE.md`` for the
#: commands and the full tables behind every number quoted here.
M13_VERDICT = (
    "M13 VERDICT: UNMEASURABLE.\n"
    "  The per-group median above is NOT stable and must not be quoted as a result.\n"
    "  Measured 2026-09-11: across 3 model seeds x 3 decision lines the median for\n"
    "  one group moved from $38,000 to $507,041, and the ordering between groups\n"
    "  reversed. Across 10 bootstrap resamples of the held-out fold at a fixed\n"
    "  model and line, the same median moved from $20,500 to $79,000.\n"
    "  Root cause, and it is stable: for roughly 93% of denied applicants NO income\n"
    "  increase up to $1,000,000 crosses the line at all, so each median is taken\n"
    "  over a small, self-selected remainder (4 to 29 people per group).\n"
    "  The unreachable share IS stable. It is a diagnostic, not a money figure,\n"
    "  and it is NOT a substitute for M13. See docs/RECOURSE.md."
)


def main(
    argv: list[str] | None = None,
    frame=None,
    source_label: str | None = None,
    sample_seed: int | None = None,
) -> int:
    """Print the M13 recourse figures. A money figure, a count, an unreachable share.

    Pass ``frame`` (a ``Source.sample`` DataFrame) plus ``source_label``
    and ``sample_seed`` to run against the national data.

    This is the function ``hmda recourse --by-group`` calls
    (``src/hmda/cli.py``). It is also its own entry point, so the same
    output is measurable directly, without going through the CLI,
    with::

        .venv/bin/python -m hmda.fairness.recourse
    """
    from hmda.model.evaluate import provenance_lines

    study = run_study(frame=frame, source_label=source_label, sample_seed=sample_seed)
    print("hmda recourse --by-group  (M13)")
    # What the numbers were computed on, before any number.
    for line in provenance_lines(
        study["rows_raw"],
        study["source_label"],
        study["sample_seed"],
        study["fixture_path"],
    ):
        print(line)
    print(
        "  No group may be characterised nationally from these numbers.\n"
        "  These are properties of a model fitted to historical records. The model\n"
        "  makes and influences no credit decision, and this is not what any real\n"
        "  lender requires of any real applicant.\n"
        f"  model: gradient-boosted trees, trained on {study['rows_train']} rows before "
        f"activity_year {study['split_year']}; study run on the {study['rows_test']} "
        "held-out rows\n"
        f"  decision line: base-rate cut on the training fold\n"
        f"  search: income only, ${GRID_STEP_USD:,.0f} resolution, capped at "
        f"${MAX_INCREASE_USD:,.0f} of extra annual income\n"
        f"  median training income: ${study['median_income_dollars']:,.0f}"
    )
    for column, results in study["results"].items():
        print(f"\n  by {column}:")
        for result in results:
            print(f"    {result.sentence()}")
    print()
    print(M13_VERDICT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
