"""Evaluate baseline and challenger models on the same held-out time split.

``hmda model --eval`` exits 0 and prints the baseline number and the
challenger number on the same line. A run that prints only the
challenger is a failure.

The pairing is enforced by the type, not by discipline: :func:`evaluate`
returns an :class:`EvalResult` carrying both models, and there is no public
function in this module that returns a challenger score alone.

CLI wiring
----------
``hmda model --eval`` (see ``src/hmda/cli.py``) calls this module's
:func:`main` directly. The bare ``hmda model`` command, with no ``--eval``
flag, still falls through to ``_not_implemented("model")``: training-only
mode is not wired up yet. :func:`main` is also runnable on its own:

    .venv/bin/python -m hmda.model.evaluate

which is useful for checking the number without going through the
CLI.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from hmda.model import translate as _translate

#: The fixture every number in this module is measured on. 50,000 rows, committed.
FIXTURE_PATH = Path("tests/fixtures/hmda_50k.parquet")

#: Rows with ``activity_year`` at or after this go to the test fold. Chosen
#: because the fixture holds 2023 (20,867 rows), 2024 (21,399) and 2025
#: (7,734), measured 2026-09-11 with ``value_counts()`` on the fixture.
DEFAULT_SPLIT_YEAR = 2025


def provenance_lines(
    rows: int,
    source_label: str | None,
    sample_seed: int | None,
    fixture_path: "Path | str" = FIXTURE_PATH,
) -> list[str]:
    """The lines that say what the numbers below were computed on.

    Defined once here and imported by ``fairness/mitigate.py``,
    ``fairness/recourse.py`` and ``governance/explain.py`` so the four
    model-training commands cannot drift into four different disclosures.

    Callers MUST print these BEFORE any number. When ``source_label`` is set
    the rows came from an in-memory sample, and the sample size and seed are
    stated explicitly: a model number computed on a sample is not a
    full-file number, and printing it without its n is the precise
    overstatement this repository exists to prevent.
    """
    if source_label is None:
        return [
            f"  SOURCE: fixture -- {rows:,} rows from {fixture_path}",
            "  FIXTURE ONLY: DC / WY / VT rows only. NOT a national sample, and no",
            "  number below may be used to characterise any group nationally.",
        ]
    seed_text = "unstated" if sample_seed is None else str(sample_seed)
    return [
        f"  {source_label}",
        f"  SAMPLE: every number below is computed on {rows:,} rows sampled from that",
        f"  source with seed {seed_text}, NOT on the full file. Sampling is proportional",
        "  on derived_race, so a small group gets few rows; per-group row counts are",
        "  printed beside every per-group number and a thin group must not be quoted.",
    ]


@dataclass(frozen=True)
class EvalResult:
    """Paired baseline/challenger evaluation, always reported together.

    baseline_auc: raw AUC of the base-rate baseline (internal only: never
        printed raw to a user-facing surface; see ``translate.py``).
    challenger_auc: raw AUC of the challenger model (same caveat).
    baseline_sentence: the translated, allowed sentence for the baseline,
        per ``translate.auc_to_sentence``.
    challenger_sentence: the translated, allowed sentence for the
        challenger.
    """

    baseline_auc: float
    challenger_auc: float
    baseline_sentence: str
    challenger_sentence: str

    @property
    def percent_better(self) -> float:
        """The M7 percent: how much better the challenger ranks than the baseline."""
        return _translate.percent_better_than_baseline(self.challenger_auc, self.baseline_auc)

    def one_line(self, challenger_name: str = "challenger") -> str:
        """Both numbers, on ONE line, baseline first. This is the M7 output."""
        return (
            f"BASELINE (always guess the base rate): {self.baseline_sentence} | "
            f"CHALLENGER ({challenger_name}): {self.challenger_sentence}"
        )


def _score(model, X) -> "list[float]":
    """Return one denial score per row, for either kind of model."""
    from hmda.model.baseline import BaseRateBaseline

    if isinstance(model, BaseRateBaseline):
        return model.predict_proba(X)
    proba = model.predict_proba(X)
    return proba[:, 1]


def evaluate(baseline, challenger, X_test, y_test) -> EvalResult:
    """Score both ``baseline`` and ``challenger`` on the same ``X_test``/``y_test`` and pair the results.

    Always returns both models' numbers in one :class:`EvalResult`; there is
    no function in this module that returns the challenger's score alone.
    """
    from sklearn.metrics import roc_auc_score

    baseline_auc = float(roc_auc_score(y_test, _score(baseline, X_test)))
    challenger_auc = float(roc_auc_score(y_test, _score(challenger, X_test)))
    return EvalResult(
        baseline_auc=baseline_auc,
        challenger_auc=challenger_auc,
        baseline_sentence=_translate.baseline_sentence(baseline_auc),
        challenger_sentence=_translate.auc_to_sentence(challenger_auc, baseline_auc),
    )


def run_eval(
    fixture_path: Path = FIXTURE_PATH,
    split_year: int = DEFAULT_SPLIT_YEAR,
    frame=None,
    source_label: str | None = None,
    sample_seed: int | None = None,
) -> dict:
    """Train the baseline and both challengers and evaluate them.

    Returns a dict with the fitted objects, the split sizes, and one
    :class:`EvalResult` per challenger.

    ``frame`` is an already-materialized ``pandas.DataFrame`` of RAW
    rows, normally ``hmda.clean.source.Source.sample(n, seed)``. When it is
    given, ``fixture_path`` is NOT read at all and the numbers are SAMPLE
    numbers. ``source_label`` and ``sample_seed`` are carried through into the
    returned dict so the caller prints what the numbers were computed on
    before it prints any number. When ``frame`` is None the behaviour is
    unchanged: it reads the committed 50k fixture only and never touches
    ``data/``.
    """
    import pandas as pd

    from hmda.model import features as F
    from hmda.model.baseline import BaseRateBaseline
    from hmda.model.gbm import fit_gbm, fit_logistic_regression

    if frame is not None and source_label is None:
        raise ValueError(
            "run_eval() got a frame with no source_label: a sample number would be "
            "printed under the fixture's DC/WY/VT label. Pass source_label describing "
            "what frame is (e.g. from Source.sample()), or pass frame=None to use the "
            "committed fixture."
        )
    raw = frame if frame is not None else pd.read_parquet(fixture_path)
    analysis = F.analysis_set(raw)
    train_rows, test_rows = F.time_split(analysis, split_year)

    X_train, y_train = F.build_feature_matrix(train_rows)
    X_test, y_test = F.build_feature_matrix(test_rows)
    # One-hot columns can differ between folds; align on the training schema
    # so the test fold never introduces a column the model never saw.
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    baseline = BaseRateBaseline().fit(y_train)
    logistic = fit_logistic_regression(X_train, y_train)
    gbm = fit_gbm(X_train, y_train)

    return {
        "source_label": source_label,
        "sample_seed": sample_seed,
        "rows_raw": len(raw),
        "rows_analysis": len(analysis),
        "rows_train": len(train_rows),
        "rows_test": len(test_rows),
        "n_features": X_train.shape[1],
        "split_year": split_year,
        "train_denial_rate": float(y_train.mean()),
        "test_denial_rate": float(y_test.mean()),
        "baseline": baseline,
        "logistic_result": evaluate(baseline, logistic, X_test, y_test),
        "gbm_result": evaluate(baseline, gbm, X_test, y_test),
    }


def main(
    argv: list[str] | None = None,
    frame=None,
    source_label: str | None = None,
    sample_seed: int | None = None,
) -> int:
    """Print the M7 evaluation. Baseline and challenger always on the same line.

    Pass ``frame`` (a ``Source.sample`` DataFrame) plus ``source_label``
    and ``sample_seed`` to run against the national data.
    """
    result = run_eval(frame=frame, source_label=source_label, sample_seed=sample_seed)
    print("hmda model --eval")
    for line in provenance_lines(
        result["rows_raw"], result["source_label"], result["sample_seed"]
    ):
        print(line)
    print(
        f"  rows: {result['rows_raw']} raw -> {result['rows_analysis']} analysis "
        f"(purchased and non-decision actions excluded)\n"
        f"  time split at activity_year {result['split_year']}: "
        f"{result['rows_train']} train rows (earlier years) / "
        f"{result['rows_test']} test rows (later years), no random shuffle\n"
        f"  features: {result['n_features']} columns, none of them protected-class\n"
        f"  denial rate: {result['train_denial_rate'] * 100:.1f}% train, "
        f"{result['test_denial_rate'] * 100:.1f}% test"
    )
    print(result["logistic_result"].one_line("logistic regression"))
    print(result["gbm_result"].one_line("gradient-boosted trees"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
