"""SHAP explainability pack (M14): top drivers overall and per protected group.

This module always prints the sample size it used; a run that hides
the sample size is a failure. No SHAP value ships as a bullet number:
SHAP is a governance artifact and a count (M14), not an outcome
metric.

Fixture only. ``tests/fixtures/hmda_50k.parquet`` is DC, WY and VT ONLY,
not a national sample. Every number this module prints or writes is
labelled as a fixture number, never a national one.

Design choices, recorded here rather than left implicit
---------------------------------------------------------
- The challenger explained is the LightGBM model from :mod:`hmda.model.gbm`,
  trained the same way ``evaluate.run_eval`` trains it (same feature
  matrix, same time split), because :func:`hmda.model.evaluate.run_eval`
  does not return the fitted model objects, only their scored results.
- SHAP is computed with ``shap.TreeExplainer`` on a deterministic sample of
  the held-out (test-fold) rows: ``X_test.sample(n=N, random_state=SEED)``
  where ``SEED`` is the same fixed seed :mod:`hmda.model.gbm` uses (``hmda.model.gbm.SEED``).
  Sampling the held-out fold, not the training fold, means the explanation
  is of predictions the model was not fitted on.
- One-hot dummy columns (e.g. ``loan_type_1``, ``loan_type_2``, ...) are
  reported as their PARENT feature ("loan type"), never as the dummy column
  name, because "loan_type_1" is not a plain-English driver and the dummy
  split has no meaning to a governance reader. Per-row importance for a
  parent feature is the sum of |SHAP| across its dummy columns; the reported
  driver score is the mean of that sum over the sampled rows.
- The protected-class dimension used for the per-group breakdown is
  ``derived_race`` (the column with the documented overstatement risk, so
  it is also the column this pack is most careful about). It is read
  directly off the held-out rows, never off the feature matrix:
  ``hmda.model.features`` excludes every protected-class column from
  ``X`` by allowlist, so the group labels are joined back in from the
  raw frame purely for reporting, and never re-enter the model.
- A minimum-count floor (:data:`MIN_GROUP_COUNT`) excludes any group whose
  sampled row count falls below it: small-denominator groups give
  unstable ratios. Excluded groups are named in the doc, not
  silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Fixed sample size for the SHAP pass. 2,000 held-out rows is large enough
#: to rank features stably and small enough that shap.TreeExplainer finishes
#: in seconds on a laptop. Fixed (not "as many as available") so the pack is
#: reproducible: a second run must produce a byte-identical result.
SAMPLE_SIZE = 2000

#: How many top drivers to list, overall and per group.
TOP_K = 5

#: Below this many sampled rows, a group's ranking is not reported:
#: small-denominator groups give unstable ratios. A parameter, not
#: a magic number buried at the call site.
MIN_GROUP_COUNT = 30

#: The protected-class column used for the per-group breakdown. Read from the
#: raw held-out frame for reporting only; it never enters the feature matrix
#: (see module docstring).
GROUP_COLUMN = "derived_race"

#: The fixture every number in this module is measured on. Not national
#: (DC + WY + VT only).
FIXTURE_PATH = Path("tests/fixtures/hmda_50k.parquet")

#: Where the plain-English label for each raw HMDA column, or each group of
#: raw columns behind an engineered feature, comes from. Every value here is
#: hand-written prose describing a column that has its own row in
#: ``docs/DATA-DICTIONARY.md`` -- see :func:`plain_english_label` and its
#: test, which asserts the raw column name(s) resolve to real dictionary
#: entries rather than trusting this table blindly.
RAW_COLUMN_LABELS: dict[str, str] = {
    "loan_amount": "loan amount",
    "income": "applicant income",
    "loan_to_value_ratio": "loan-to-value ratio",
    "property_value": "property value",
    "debt_to_income_ratio": "debt-to-income ratio",
    "loan_type": "loan type",
    "loan_purpose": "loan purpose",
    "occupancy_type": "occupancy type",
    "lien_status": "lien status",
    "preapproval": "preapproval request status",
    "conforming_loan_limit": "conforming loan limit flag",
    "construction_method": "construction method",
    "total_units": "number of dwelling units",
    "open-end_line_of_credit": "open-end line of credit flag",
    "business_or_commercial_purpose": "business or commercial purpose flag",
    "reverse_mortgage": "reverse mortgage flag",
}

#: The one engineered feature (see ``hmda.model.features.build_feature_matrix``),
#: mapped to the raw columns it is built from. Reported under its own plain
#: label but resolved, for the dictionary test, through both source columns.
ENGINEERED_FEATURE_SOURCES: dict[str, tuple[str, ...]] = {
    "loan_to_income_ratio": ("loan_amount", "income"),
}

ENGINEERED_FEATURE_LABELS: dict[str, str] = {
    "loan_to_income_ratio": "loan amount relative to applicant income (loan-to-income ratio)",
}


@dataclass(frozen=True)
class DriverScore:
    """One ranked driver: its plain-English label and the raw columns it resolves to."""

    label: str
    raw_columns: tuple[str, ...]
    mean_abs_shap: float  # internal magnitude, never printed to a user-facing surface


@dataclass(frozen=True)
class ExplainabilityResult:
    """The explainability pack's user-facing result.

    sample_size: how many rows SHAP was run over. Always printed; a run
        that hides it is a failure.
    n_features: how many plain-English (parent) features SHAP considered:
        the M14 count ("explained by SHAP over N features").
    top_drivers_overall: plain-English feature names (resolved through
        ``docs/DATA-DICTIONARY.md``, never raw column names), ranked by mean
        |SHAP value|.
    top_drivers_by_group: same, keyed by protected-class group value.
    group_counts: sampled row count per group value actually reported.
    excluded_groups: group values present in the sample but dropped for
        falling under :data:`MIN_GROUP_COUNT`, with their counts.
    missingness_note: a plain-English sentence recording whether a
        column this pack flagged as a top driver is also one with a large
        missingness gap between denied and non-denied rows, MEASURED on this
        same sample -- not asserted from an outside claim.
    """

    sample_size: int
    n_features: int
    top_drivers_overall: list[str]
    top_drivers_by_group: dict[str, list[str]]
    group_counts: dict[str, int] = field(default_factory=dict)
    excluded_groups: dict[str, int] = field(default_factory=dict)
    missingness_note: str = ""
    #: None means the rows came from the committed fixture. A string
    #: means they came from an in-memory sample of another source and this is
    #: the sentence that says which one and how many rows.
    source_label: str | None = None
    #: The seed of that sample, or None when there was no sample.
    sample_seed: int | None = None
    #: Rows the held-out fold was drawn from, before SHAP sampling.
    source_rows: int = 0


def _raw_group_for_column(column: str, categorical_features: tuple[str, ...]) -> tuple[str, str, tuple[str, ...]]:
    """Return ``(group_key, plain_label, raw_columns)`` for one feature-matrix column.

    ``group_key`` is the parent feature a one-hot dummy belongs to, or the
    column itself for a continuous feature. Unknown columns (should not
    occur given :mod:`hmda.model.features`' fixed schema) fall back to the
    raw column name as its own label, which will then correctly fail the
    dictionary-resolution test rather than being silently swallowed.
    """
    if column == "income_dollars":
        return "income", RAW_COLUMN_LABELS["income"], ("income",)
    if column == "debt_to_income_rank":
        return (
            "debt_to_income_ratio",
            RAW_COLUMN_LABELS["debt_to_income_ratio"],
            ("debt_to_income_ratio",),
        )
    if column in ENGINEERED_FEATURE_SOURCES:
        return column, ENGINEERED_FEATURE_LABELS[column], ENGINEERED_FEATURE_SOURCES[column]
    if column in RAW_COLUMN_LABELS:
        return column, RAW_COLUMN_LABELS[column], (column,)
    for cat in categorical_features:
        if column == cat or column.startswith(cat + "_"):
            return cat, RAW_COLUMN_LABELS.get(cat, cat), (cat,)
    return column, column, (column,)


def _grouped_importance(shap_values, columns, categorical_features) -> dict[str, "DriverScore"]:
    """Aggregate per-dummy-column |SHAP| into per-parent-feature :class:`DriverScore`."""
    import numpy as np

    abs_vals = np.abs(shap_values)  # (rows, columns)
    groups: dict[str, list[int]] = {}
    labels: dict[str, str] = {}
    raw_cols: dict[str, tuple[str, ...]] = {}
    for idx, col in enumerate(columns):
        key, label, raws = _raw_group_for_column(col, categorical_features)
        groups.setdefault(key, []).append(idx)
        labels[key] = label
        raw_cols[key] = raws

    scores: dict[str, DriverScore] = {}
    for key, idxs in groups.items():
        per_row_sum = abs_vals[:, idxs].sum(axis=1)
        scores[key] = DriverScore(
            label=labels[key],
            raw_columns=raw_cols[key],
            mean_abs_shap=float(per_row_sum.mean()),
        )
    return scores


def _top_k_labels(scores: dict[str, DriverScore], k: int) -> list[str]:
    ranked = sorted(scores.values(), key=lambda s: s.mean_abs_shap, reverse=True)
    return [s.label for s in ranked[:k]]


def _measure_missingness_note(test_rows, top_overall_raw_columns: set[str]) -> str:
    """Measure, on the SAME held-out rows this pack explains, whether a top
    driver also has a denial-vs-non-denial missingness gap.

    Only checks ``loan_to_value_ratio`` and ``property_value``, the two
    columns that carry the ``"Exempt"``/``"NA"`` sentinels
    (``docs/DATA-DICTIONARY.md``). Returns a sentence stating the measured
    gap plainly, or stating that no gap was found, never asserting a number
    it did not compute on this sample.
    """
    import pandas as pd

    from hmda.model import features as F

    candidates = [c for c in ("loan_to_value_ratio", "property_value") if c in top_overall_raw_columns]
    if not candidates:
        return (
            "Neither loan-to-value ratio nor property value ranked among the top overall "
            "drivers in this run, so this pack did not check their missingness rate."
        )

    denial = F.label(test_rows)
    lines = []
    for col in candidates:
        raw = test_rows[col].astype("string").str.strip()
        missing = raw.isin(F._MISSING_STRINGS) | raw.isna()
        denied_missing = float(missing[denial == 1].mean()) if (denial == 1).any() else float("nan")
        not_denied_missing = float(missing[denial == 0].mean()) if (denial == 0).any() else float("nan")
        lines.append(
            f"{RAW_COLUMN_LABELS[col]} is missing for {denied_missing:.1%} of denied applications "
            f"vs {not_denied_missing:.1%} of non-denied applications in this sample "
            # No source word here. The document header states the
            # source once; hardcoding "fixture" made a national run describe
            # its own numbers as fixture-derived.
            f"(n={len(test_rows)} held-out rows)."
        )
    return (
        "This pack ranked a column with a measured missingness gap among the top overall "
        "drivers. Measured on this pack's own held-out sample: " + " ".join(lines) + " "
        "Missingness itself appears to carry predictive signal here, which is a governance "
        "caveat, not a causal claim: the model was not told which values were missing on "
        "purpose, but LightGBM can still split on NaN, so an applicant's data being "
        "incomplete can influence the model's score."
    )


def _train_and_prepare(fixture_path=FIXTURE_PATH, frame=None):
    """Train the GBM the same way ``evaluate.run_eval`` does, and return
    everything :func:`explain` needs: the model, the training-fold columns,
    the held-out feature matrix, and the held-out RAW rows (for group labels
    and missingness), all index-aligned.

    This function used to read the module constant ``FIXTURE_PATH``
    directly with no parameter at all, which is why ``explain`` could not be
    pointed at the national data. ``frame`` is an already-materialized
    ``pandas.DataFrame`` of RAW rows (normally
    ``hmda.clean.source.Source.sample(n, seed)``); when it is given, nothing
    is read from disk. With ``frame=None`` the behaviour is unchanged.
    """
    import pandas as pd

    from hmda.model import features as F
    from hmda.model.evaluate import DEFAULT_SPLIT_YEAR
    from hmda.model.gbm import fit_gbm

    raw = frame if frame is not None else pd.read_parquet(fixture_path)
    analysis = F.analysis_set(raw)
    train_rows, test_rows = F.time_split(analysis, DEFAULT_SPLIT_YEAR)

    X_train, y_train = F.build_feature_matrix(train_rows)
    X_test, _y_test = F.build_feature_matrix(test_rows)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    model = fit_gbm(X_train, y_train)
    return model, X_test, test_rows


def explain(model, X_sample, group_labels, top_k: int = TOP_K) -> ExplainabilityResult:
    """Run SHAP over ``model`` on ``X_sample`` and rank the top ``top_k`` drivers overall and per group in ``group_labels``.

    Every reported driver name must resolve to a real entry in
    ``docs/DATA-DICTIONARY.md``: raw column names are
    never surfaced directly.

    ``group_labels`` must be a ``pandas.Series`` aligned to ``X_sample``'s
    index (e.g. ``derived_race`` pulled from the held-out raw rows). Groups
    with fewer than :data:`MIN_GROUP_COUNT` sampled rows are excluded from
    ``top_drivers_by_group`` and recorded in ``excluded_groups`` instead.
    """
    import shap

    from hmda.model import features as F

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_sample)
    # LightGBM binary classifier via shap.TreeExplainer: some shap/lightgbm
    # combinations return a list of two (rows, cols) arrays (class 0, class
    # 1); others return one (rows, cols) array already for the positive
    # class. Always resolve to the positive-class (denial) array.
    if isinstance(raw_shap, list):
        shap_values = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
    else:
        shap_values = raw_shap

    overall_scores = _grouped_importance(shap_values, list(X_sample.columns), F.CATEGORICAL_FEATURES)
    top_overall = _top_k_labels(overall_scores, top_k)
    top_overall_raw = {c for s in overall_scores.values() for c in s.raw_columns}

    by_group: dict[str, list[str]] = {}
    group_counts: dict[str, int] = {}
    excluded_groups: dict[str, int] = {}
    if group_labels is not None:
        aligned = group_labels.reindex(X_sample.index)
        for value, count in aligned.value_counts(dropna=True).items():
            count = int(count)
            if count < MIN_GROUP_COUNT:
                excluded_groups[str(value)] = count
                continue
            mask = (aligned == value).to_numpy()
            group_scores = _grouped_importance(shap_values[mask], list(X_sample.columns), F.CATEGORICAL_FEATURES)
            by_group[str(value)] = _top_k_labels(group_scores, top_k)
            group_counts[str(value)] = count

    missingness_note = ""
    if hasattr(X_sample, "attrs") and "test_rows" in X_sample.attrs:
        missingness_note = _measure_missingness_note(X_sample.attrs["test_rows"], top_overall_raw)

    return ExplainabilityResult(
        sample_size=int(len(X_sample)),
        n_features=len(overall_scores),
        top_drivers_overall=top_overall,
        top_drivers_by_group=by_group,
        group_counts=group_counts,
        excluded_groups=excluded_groups,
        missingness_note=missingness_note,
    )


def run_explain(
    top_k: int = TOP_K,
    sample_size: int = SAMPLE_SIZE,
    fixture_path=FIXTURE_PATH,
    frame=None,
    source_label: str | None = None,
    sample_seed: int | None = None,
) -> ExplainabilityResult:
    """Train the challenger, sample the held-out fold deterministically, and run :func:`explain`.

    ``frame`` is an already-materialized ``pandas.DataFrame`` of RAW
    rows, normally ``hmda.clean.source.Source.sample(n, seed)``. When it is
    given, ``fixture_path`` is NOT read. ``source_label`` and ``sample_seed``
    are recorded on the result so every surface that prints a number can print
    what it was computed on first. With ``frame=None`` the behaviour is
    unchanged.

    Note there are TWO sample sizes once ``frame`` is used: the rows drawn
    from the source (``source_rows``), and the SHAP sample drawn from the
    held-out fold of those (``sample_size``). Both are reported; conflating
    them would misstate the basis of every driver ranking.
    """
    import dataclasses

    from hmda.model.gbm import SEED

    if frame is not None and source_label is None:
        raise ValueError(
            "run_explain() got a frame with no source_label: a sample number would be "
            "printed under the fixture's DC/WY/VT label. Pass source_label describing "
            "what frame is (e.g. from Source.sample()), or pass frame=None to use the "
            "committed fixture."
        )
    model, X_test, test_rows = _train_and_prepare(fixture_path, frame)
    n = min(sample_size, len(X_test))
    sample_idx = X_test.sample(n=n, random_state=SEED).sort_index().index
    X_sample = X_test.loc[sample_idx]
    rows_sample = test_rows.loc[sample_idx]
    X_sample = X_sample.copy()
    X_sample.attrs["test_rows"] = rows_sample

    group_labels = rows_sample[GROUP_COLUMN] if GROUP_COLUMN in rows_sample.columns else None
    result = explain(model, X_sample, group_labels, top_k=top_k)
    return dataclasses.replace(
        result,
        source_label=source_label,
        sample_seed=sample_seed,
        source_rows=int(len(frame)) if frame is not None else int(len(X_test)),
    )


def render_explainability_doc(result: ExplainabilityResult, out_path) -> None:
    """Render ``result`` into ``docs/EXPLAINABILITY.md``, stably (re-render must be a no-op diff)."""
    # The provenance NOUN follows the run, it is never hardcoded.
    # ``source_label is None`` is the only case in which these numbers really
    # are the committed DC/WY/VT fixture's.
    basis = "fixture" if result.source_label is None else "sample"
    lines: list[str] = []
    lines.append("# EXPLAINABILITY")
    lines.append("")
    lines.append(
        "GENERATED by `hmda explain --report` "
        "(`.venv/bin/python -m hmda.governance.explain`). Never hand-edited."
    )
    lines.append("")
    lines.append(
        (
            "**SAMPLE-ONLY.** Every number below is measured on a sample, not on "
            f"the full file: {result.source_label}, {result.source_rows:,} rows "
            f"drawn with seed {result.sample_seed}. Sampling is proportional on "
            "`derived_race`, so a small group gets few rows; the per-group counts "
            "below are the denominators and a thin group must not be quoted."
        )
        if result.source_label is not None
        else (
            "**FIXTURE-ONLY.** Every number below is measured on "
            f"`{FIXTURE_PATH}`: **DC, WY and VT only**, not "
            "a national sample. No number in this file may be quoted as national "
            "or used to characterise any group nationally."
        )
    )
    lines.append("")
    lines.append("## M14")
    lines.append("")
    lines.append(
        f"Explained by SHAP over **{result.n_features} features**, with the "
        f"top **{TOP_K}** drivers listed per group. Sample size: "
        f"**{result.sample_size} held-out {basis} rows** "
        f"(`SAMPLE_SIZE = {SAMPLE_SIZE}` in `src/hmda/governance/explain.py`, "
        f"capped at the held-out fold's actual row count). This is a COUNT, "
        "not an outcome metric: no SHAP magnitude is reported in this "
        "document."
    )
    lines.append("")
    lines.append("## What was explained")
    lines.append("")
    lines.append(
        "The challenger is the LightGBM gradient-boosted model from "
        "`hmda.model.gbm.fit_gbm`, trained on the same allowlisted feature "
        "columns and the same time split as `hmda model --eval` "
        "(`src/hmda/model/evaluate.py`). SHAP values come from "
        "`shap.TreeExplainer` over a fixed, seeded sample of the HELD-OUT "
        "(test-fold) rows, so this pack explains predictions the model was "
        "not fitted on."
    )
    lines.append("")
    lines.append(
        "The protected-class fields are never inputs to this model: "
        "`hmda.model.features` excludes every `derived_*` / `applicant_*` / "
        "`co-applicant_*` demographic column family by allowlist (see that "
        "module's docstring). The `derived_race` values used below to split "
        "the sample into groups are read only from the raw held-out rows, "
        "for reporting, and never re-enter the feature matrix."
    )
    lines.append("")
    lines.append(f"## Top drivers overall ({basis})")
    lines.append("")
    for i, label in enumerate(result.top_drivers_overall, start=1):
        lines.append(f"{i}. {label}")
    lines.append("")
    lines.append(f"## Top drivers per protected group ({basis}, `derived_race`)")
    lines.append("")
    lines.append(
        "**Descriptive, not causal.** A feature ranking high for one group "
        "does not mean the model treats that group differently on purpose: "
        "the protected fields are not model inputs at all (see above). It "
        "means that, among the sampled rows for that group, that feature "
        "moved the model's score the most on average. Differences between "
        "groups' driver lists can also reflect differences in each group's "
        "distribution of the (non-protected) input columns."
    )
    lines.append("")
    for group in sorted(result.top_drivers_by_group):
        count = result.group_counts.get(group, 0)
        drivers = result.top_drivers_by_group[group]
        lines.append(f"### {group} (n={count} sampled rows)")
        lines.append("")
        for i, label in enumerate(drivers, start=1):
            lines.append(f"{i}. {label}")
        lines.append("")
    if result.excluded_groups:
        lines.append("### Excluded groups (below the minimum-count floor)")
        lines.append("")
        lines.append(
            f"`derived_race` values present in the sample with fewer than "
            f"{MIN_GROUP_COUNT} rows are not ranked (small-denominator "
            "groups give unstable ratios). This floor is a "
            "stated parameter (`MIN_GROUP_COUNT` in "
            "`src/hmda/governance/explain.py`), not a magic number."
        )
        lines.append("")
        for group in sorted(result.excluded_groups):
            lines.append(f"- {group} (n={result.excluded_groups[group]})")
        lines.append("")
    lines.append("## Missingness")
    lines.append("")
    lines.append(
        result.missingness_note
        or "This pack did not measure a missingness gap for its top drivers."
    )
    lines.append("")
    lines.append("## Limits")
    lines.append("")
    lines.append(
        (
            f"- Sample-only: every number above is measured on "
            f"{result.source_label}, {result.source_rows:,} rows drawn with "
            f"seed {result.sample_seed}, and on the "
            f"{result.sample_size}-row held-out SHAP sample of those. It is a "
            "sample of that source, never the full file."
        )
        if result.source_label is not None
        else (
            "- Fixture-only: DC, WY and VT. Re-measure "
            "once the national database lands."
        )
    )
    lines.append(
        "- SHAP attributes a prediction to inputs of a model that is not "
        "used to make credit decisions (see the model card). It explains "
        "the model, not the underlying lending process."
    )
    lines.append(
        "- Per-group driver lists are descriptive of this sample's SHAP "
        "distribution, not a test of disparate treatment."
    )
    lines.append(
        "- `derived_race` is a derived, coarse field, not self-identification; "
        "the per-group breakdown inherits that limitation."
    )
    lines.append("")

    text = "\n".join(lines) + "\n"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)


#: Where the rendered doc goes, relative to the repo root.
DOC_PATH = Path("docs/EXPLAINABILITY.md")


def main(
    argv: list[str] | None = None,
    frame=None,
    source_label: str | None = None,
    sample_seed: int | None = None,
) -> int:
    """Run the SHAP explainability pack and render `docs/EXPLAINABILITY.md`. Always prints the sample size.

    Pass ``frame`` (a ``Source.sample`` DataFrame) plus ``source_label``
    and ``sample_seed`` to run against the national data.
    """
    from hmda.model.evaluate import provenance_lines

    result = run_explain(
        frame=frame, source_label=source_label, sample_seed=sample_seed
    )
    print("hmda explain --report")
    # What the numbers were computed on, before any number.
    for line in provenance_lines(
        result.source_rows, result.source_label, result.sample_seed, FIXTURE_PATH
    ):
        print(line)
    print(
        f"  sample size: {result.sample_size} held-out rows SHAP was run over\n"
        f"  features considered: {result.n_features}\n"
        f"  top {TOP_K} drivers overall: {', '.join(result.top_drivers_overall)}"
    )
    for group in sorted(result.top_drivers_by_group):
        print(f"  [{group}] (n={result.group_counts.get(group, 0)}): "
              f"{', '.join(result.top_drivers_by_group[group])}")
    if result.excluded_groups:
        print(f"  excluded (below MIN_GROUP_COUNT={MIN_GROUP_COUNT}): {result.excluded_groups}")
    render_explainability_doc(result, DOC_PATH)
    print(f"  wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
