"""Exactly three mitigation techniques: reweighing (pre), fairness-constrained GBM (in), per-group thresholds (post).

One mitigation technique per stage, no more: the technique
registry below has exactly 3 entries and ``tests/test_mitigate.py`` asserts
it. This is an OFFLINE COUNTERFACTUAL STUDY over historical applications:
nothing here scores a live applicant, nothing here is
deployed, and no output may be described as live or in production.

Protected-class fields stay out of the feature matrix, as in the unmitigated
model's pipeline (``model/features.py`` builds ``X`` from an allowlist). Two of the three
techniques read the protected class outside the feature matrix, and each use
is different in kind, so each is named here rather than lumped together:

* **reweighing** uses it at TRAINING time, to weight rows. The fitted model
  still has no protected column in it.
* **fair_constrained_gbm** uses it at TRAINING time, in the same way, once
  per round.
* **per_group_threshold** uses it at DECISION time: the model is the
  unmitigated one, and the applicant's group selects which cut-off applies.

**Per-group thresholds are legally contested in the United States.** Setting
a different approval cut-off by race is disparate treatment on its face, and
the Supreme Court's *Ricci v. DeStefano* (2009) line of reasoning constrains
adjusting outcomes by protected class to fix a disparity. This module prints
that warning beside the technique's own result line and ``docs/LIMITS.md``
repeats it. It is measured here because a trade-off study that omits the
post-processing corner is incomplete, NOT because it is approved practice.

What is measured, and against what
----------------------------------
Every technique is compared to the UNMITIGATED baseline gradient-boosted model,
on the SAME time split that baseline uses (``model.evaluate.DEFAULT_SPLIT_YEAR``),
through that baseline's own feature pipeline. No second split and no second feature
set is invented here.

Three numbers ship together on one line, always:
the disparity cut, the accuracy change, and the margin change.
:class:`MitigationResult` carries all three, and there is no function in
this module that returns the disparity cut alone.

FIXTURE ONLY. Every number this module prints is measured on
``tests/fixtures/hmda_50k.parquet``, which is
**DC (20,592), WY (15,085) and VT (14,323) only**, not a national sample,
and two of the three are among the least racially diverse states. No number
here characterises any group nationally. The output repeats this warning.

Provenance of the technique definitions
---------------------------------------
Each technique's provenance is stated plainly below rather than left to
cite a library's documentation. Measured 2026-09-11 in the project venv::

    .venv/bin/python -c "import fairlearn"  -> ModuleNotFoundError
    .venv/bin/python -c "import aif360"     -> ModuleNotFoundError

Neither library is installed, and none is installed here, so all three
techniques are implemented directly on ``lightgbm`` 4.7.0 and ``numpy``.
They follow the standard published definitions (Kamiran and Calders'
reweighing; a reduction-style reweighted booster in the spirit of Agarwal et
al.'s exponentiated gradient; Hardt et al.'s post-hoc thresholding, applied
to demographic parity rather than equalised odds). **The implementations
were not checked against those papers**, so they are stated below in full
rather than cited as equivalent to a library's. What is in the code is the
specification; the names are given for orientation only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from hmda.fairness import floors as _floors
from hmda.model import economics as _economics
from hmda.model import evaluate as _evaluate
from hmda.model import features as _features
from hmda.model import gbm as _gbm

#: The committed fixture, reused from model evaluation so both measure the same rows.
FIXTURE_PATH = _evaluate.FIXTURE_PATH

#: The time split, reused from model evaluation. Not re-derived here.
SPLIT_YEAR = _evaluate.DEFAULT_SPLIT_YEAR

#: The protected-class column the disparity is measured over. Used for
#: measurement, for training weights, and (for the post technique) at
#: decision time. Never a model feature.
GROUP_COLUMN = "derived_race"

#: Values of :data:`GROUP_COLUMN` that are not a racial group and therefore
#: cannot carry a disparity claim. "Race Not Available" is 28.6% of the
#: fixture and its share is printed beside every
#: result rather than being quietly dropped.
NON_GROUP_VALUES: frozenset[str] = frozenset(
    {"Race Not Available", "Free Form Text Only", "Joint", "NA", ""}
)

#: Rounds and step size for the in-processing technique. Fixed constants,
#: declared here, NOT tuned against the test fold: the reported result is
#: whatever the final round produces. Tuning these until the number looked
#: better would make the result a selection artifact rather than a
#: measurement.
FAIR_GBM_ROUNDS = 8
FAIR_GBM_STEP = 2.0


class MitigationStage(str, Enum):
    PRE = "pre"
    IN = "in"
    POST = "post"


@dataclass(frozen=True)
class Outcome:
    """What one decision rule did on the held-out fold. Not a technique's result.

    approval_gap_pp: the widest approval-rate gap between two eligible
        racial groups, in percentage points.
    accuracy_pct: share of held-out applications where the rule's decision
        matched the lender's historical decision, in percent.
    margin_per_1000_usd: expected margin per 1,000 applications, computed
        from ``config/economics.yaml`` plus the measured mean principal.
    approval_rate: overall share approved.
    group_rates: approval rate per eligible group.
    """

    approval_gap_pp: float
    accuracy_pct: float
    margin_per_1000_usd: float
    approval_rate: float
    group_rates: dict[str, float]
    high_group: str
    low_group: str
    mean_principal_usd: float = 0.0
    median_principal_usd: float = 0.0


@dataclass(frozen=True)
class MitigationResult:
    """One technique's measured effect against the unmitigated baseline model.

    Carries the disparity cut, the accuracy change AND the margin change
    together: a run that prints the
    disparity cut alone is a failure. The three cannot be separated: they
    are fields of one frozen record and :meth:`one_line` prints all three.
    """

    technique: str
    stage: MitigationStage
    before: Outcome
    after: Outcome
    note: str = ""

    @property
    def disparity_cut_pp(self) -> float:
        """Reduction in the approval-rate gap, in percentage points. Positive = narrower."""
        return self.before.approval_gap_pp - self.after.approval_gap_pp

    @property
    def disparity_cut_percent(self) -> float:
        """The M11 percent: how much of the unmitigated gap was removed."""
        if self.before.approval_gap_pp <= 0.0:
            raise ValueError(
                "the unmitigated approval-rate gap is not positive, so a percent cut "
                "against it is undefined; report the percentage-point gaps instead"
            )
        return self.disparity_cut_pp / self.before.approval_gap_pp * 100.0

    @property
    def accuracy_change_pp(self) -> float:
        """Change in agreement with the historical decision, in percentage points."""
        return self.after.accuracy_pct - self.before.accuracy_pct

    @property
    def margin_change_usd_per_1000(self) -> float:
        """Change in expected margin per 1,000 applications, in dollars."""
        return self.after.margin_per_1000_usd - self.before.margin_per_1000_usd

    def one_line(self) -> str:
        """The M11 + M12 line. All three numbers, or nothing.

        Required shape: "cut the approval-rate gap [P]% while
        holding accuracy within [Q] points of the unmitigated model", "at a
        cost of [$X] in expected margin per 1,000 applications". No raw DIR,
        SPD, EOD, AUC or p-value appears.
        """
        cut = self.disparity_cut_percent
        verb = "cut" if cut >= 0.0 else "WIDENED"
        acc = self.accuracy_change_pp
        acc_word = "within" if abs(acc) < 1.0 else "at a cost of"
        margin = self.margin_change_usd_per_1000
        money = (
            f"${margin:,.0f} MORE expected margin per 1,000 applications"
            if margin >= 0.0
            else f"${abs(margin):,.0f} of expected margin per 1,000 applications"
        )
        return (
            f"{self.technique:<22} [{self.stage.value:<4}] "
            f"{verb} the approval-rate gap {abs(cut):.1f}% "
            f"({self.before.approval_gap_pp:.1f} -> {self.after.approval_gap_pp:.1f} "
            f"percentage points), {acc_word} {abs(acc):.2f} points of the unmitigated "
            f"model's agreement with the historical decision, at {money}"
        )


#: Exactly three entries, one per stage. A fourth technique must not be added;
#: ``tests/test_mitigate.py`` asserts ``len(TECHNIQUE_REGISTRY) == 3``.
TECHNIQUE_REGISTRY: dict[str, MitigationStage] = {
    "reweighing": MitigationStage.PRE,
    "fair_constrained_gbm": MitigationStage.IN,
    "per_group_threshold": MitigationStage.POST,
}


# --------------------------------------------------------------------------
# Data preparation. Reuses the unmitigated model's pipeline; invents no second split.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Prepared:
    X_train: pd.DataFrame
    y_train: pd.Series
    g_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    g_test: pd.Series
    principal_test: pd.Series
    eligible_groups: tuple[str, ...]
    target_approval_rate: float
    race_not_available_share: float
    min_count: int
    fixture_path: str
    rows_train: int
    rows_test: int
    #: None means the rows came from ``fixture_path``. A string means
    #: they came from an in-memory sample and this is the sentence that says
    #: which source and how many rows.
    source_label: str | None = None
    #: The seed of that sample, or None when there was no sample.
    sample_seed: int | None = None
    #: Held-out row count per protected group, so no per-group number is
    #: ever printed without the denominator it rests on. Empty dict only when
    #: a caller constructs Prepared directly.
    test_group_counts: dict = field(default_factory=dict)

    #: RAW rows handed in, BEFORE the analysis-set exclusions. This, not
    #: ``rows_train + rows_test``, is the sample size the provenance line must
    #: quote: those two are what survived the exclusions and quoting them as
    #: "rows sampled" understates the draw.
    rows_raw: int = 0


def prepare(
    fixture_path: Path = FIXTURE_PATH,
    split_year: int = SPLIT_YEAR,
    min_count: int = _floors.DEFAULT_MIN_COUNT,
    frame=None,
    source_label: str | None = None,
    sample_seed: int | None = None,
) -> Prepared:
    """Build exactly the same analysis set, split and features as the unmitigated model.

    The protected-class series is carried ALONGSIDE the feature matrix, never
    inside it: ``build_feature_matrix`` raises if a protected column reaches
    ``X`` (``model/features.py``), and this function never adds one.

    ``frame`` is an already-materialized ``pandas.DataFrame`` of RAW
    rows, normally ``hmda.clean.source.Source.sample(n, seed)``. When it is
    given, ``fixture_path`` is NOT read and every number downstream is a
    SAMPLE number; ``source_label`` and ``sample_seed`` are carried onto
    :class:`Prepared` so the report says so before it says anything else.
    With ``frame=None`` the behaviour is unchanged.
    """
    if frame is not None and source_label is None:
        raise ValueError(
            "prepare() got a frame with no source_label: a sample number would be "
            "printed under the fixture's DC/WY/VT label. Pass source_label describing "
            "what frame is (e.g. from Source.sample()), or pass frame=None to use the "
            "committed fixture."
        )
    raw = frame if frame is not None else pd.read_parquet(fixture_path)
    analysis = _features.analysis_set(raw)
    train_rows, test_rows = _features.time_split(analysis, split_year)

    X_train, y_train = _features.build_feature_matrix(train_rows)
    X_test, y_test = _features.build_feature_matrix(test_rows)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    g_train = train_rows[GROUP_COLUMN].astype("string").fillna("NA")
    g_test = test_rows[GROUP_COLUMN].astype("string").fillna("NA")

    principal_test = pd.to_numeric(test_rows["loan_amount"], errors="coerce").astype("float64")

    counts = g_test.value_counts()
    eligible = tuple(
        sorted(
            str(value)
            for value, n in counts.items()
            if str(value) not in NON_GROUP_VALUES
            and not _floors.below_floor(int(n), min_count)
        )
    )
    if len(eligible) < 2:
        raise ValueError(
            f"fewer than two racial groups clear the {min_count}-application floor in the "
            "held-out fold; no approval-rate gap can be reported"
        )

    not_available = int((g_test == "Race Not Available").sum())

    return Prepared(
        X_train=X_train,
        y_train=y_train,
        g_train=g_train,
        X_test=X_test,
        y_test=y_test,
        g_test=g_test,
        principal_test=principal_test,
        eligible_groups=eligible,
        target_approval_rate=float((y_train == 0).mean()),
        race_not_available_share=not_available / len(g_test) if len(g_test) else 0.0,
        min_count=min_count,
        fixture_path=str(fixture_path),
        rows_train=len(train_rows),
        rows_test=len(test_rows),
        source_label=source_label,
        sample_seed=sample_seed,
        test_group_counts={str(v): int(n) for v, n in counts.items()},
        rows_raw=len(raw),
    )


# --------------------------------------------------------------------------
# The decision rule. Scores are denial probabilities; APPROVE when the score
# is below the cut-off. Every technique uses this same rule, so the
# comparison is of models and cut-offs, never of two different rules.
# --------------------------------------------------------------------------
def denial_scores(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X))[:, 1]


def global_threshold(train_scores: np.ndarray, target_approval_rate: float) -> float:
    """The single cut-off that approves ``target_approval_rate`` of the TRAINING fold.

    Chosen on training scores only. Calibrating it on the held-out fold would
    leak the answer into the decision rule.
    """
    return float(np.quantile(train_scores, target_approval_rate))


def per_group_thresholds(
    train_scores: np.ndarray,
    g_train: pd.Series,
    target_approval_rate: float,
    eligible_groups: tuple[str, ...],
    fallback: float,
    min_count: int,
) -> dict[str, float]:
    """One cut-off per eligible group, each approving ``target_approval_rate`` of that group's TRAINING rows.

    This equalises the approval rate by construction on the training fold; how
    much of that survives on the held-out fold is the measurement. Groups
    below the count floor keep the global cut-off, so a tiny group never gets
    a cut-off fitted on a handful of rows.
    """
    thresholds: dict[str, float] = {}
    groups = np.asarray(g_train.astype(str))
    for value in eligible_groups:
        mask = groups == value
        n = int(mask.sum())
        if _floors.below_floor(n, min_count):
            thresholds[value] = fallback
            continue
        thresholds[value] = float(np.quantile(train_scores[mask], target_approval_rate))
    return thresholds


def decide(
    scores: np.ndarray,
    g: pd.Series,
    threshold: float,
    group_thresholds: dict[str, float] | None = None,
) -> np.ndarray:
    """Return 1 where the rule APPROVES, 0 where it denies.

    When ``group_thresholds`` is given, the applicant's protected class
    selects the cut-off. That is the only place in this repo where a
    protected class touches a decision, it happens at decision time only, and
    it is the legally contested technique flagged in the module docstring.
    """
    cutoff = np.full(len(scores), float(threshold))
    if group_thresholds:
        groups = np.asarray(g.astype(str))
        for value, t in group_thresholds.items():
            cutoff[groups == value] = float(t)
    return (scores < cutoff).astype(int)


def measure(
    approve: np.ndarray,
    prepared: Prepared,
    assumptions: _economics.EconomicsAssumptions,
) -> Outcome:
    """Turn a set of held-out decisions into the three reportable numbers."""
    historical_approve = (np.asarray(prepared.y_test) == 0).astype(int)
    accuracy_pct = float((approve == historical_approve).mean() * 100.0)

    groups = np.asarray(prepared.g_test.astype(str))
    group_rates: dict[str, float] = {}
    for value in prepared.eligible_groups:
        mask = groups == value
        group_rates[value] = float(approve[mask].mean())

    high = max(group_rates, key=lambda k: group_rates[k])
    low = min(group_rates, key=lambda k: group_rates[k])
    gap_pp = (group_rates[high] - group_rates[low]) * 100.0

    approved_mask = approve == 1
    principal = np.asarray(prepared.principal_test, dtype="float64")[approved_mask]
    principal = principal[~np.isnan(principal)]
    mean_principal = float(principal.mean()) if principal.size else 0.0
    median_principal = float(np.median(principal)) if principal.size else 0.0
    approval_rate = float(approve.mean())
    margin = _economics.expected_margin_per_1000(
        approvals_per_1000=approval_rate * 1000.0,
        assumptions=assumptions,
        mean_principal_usd=mean_principal,
    )

    return Outcome(
        approval_gap_pp=gap_pp,
        accuracy_pct=accuracy_pct,
        margin_per_1000_usd=margin,
        approval_rate=approval_rate,
        group_rates=group_rates,
        high_group=high,
        low_group=low,
        mean_principal_usd=mean_principal,
        median_principal_usd=median_principal,
    )


# --------------------------------------------------------------------------
# The three techniques. One per stage. No more.
# --------------------------------------------------------------------------
def reweighing_weights(y: pd.Series, g: pd.Series) -> np.ndarray:
    """Kamiran-and-Calders reweighing weights for the TRAINING fold.

    ``w(group, label) = P(group) * P(label) / P(group, label)``, so a
    (group, outcome) cell that is under-represented relative to independence
    is up-weighted. The protected class never enters the feature matrix; it
    only sets these weights. Cells with no rows are skipped rather than
    given an infinite weight.
    """
    labels = np.asarray(y)
    groups = np.asarray(g.astype(str))
    n = len(labels)
    weights = np.ones(n, dtype="float64")
    for value in np.unique(groups):
        g_mask = groups == value
        p_group = g_mask.mean()
        for label_value in np.unique(labels):
            y_mask = labels == label_value
            p_label = y_mask.mean()
            cell = g_mask & y_mask
            p_cell = cell.mean()
            if p_cell <= 0.0:
                continue
            weights[cell] = p_group * p_label / p_cell
    return weights


def fit_reweighed(prepared: Prepared):
    """PRE-processing: fit the same baseline booster on reweighed training rows."""
    weights = reweighing_weights(prepared.y_train, prepared.g_train)
    return _fit_weighted(prepared, weights)


def _fit_weighted(prepared: Prepared, weights: np.ndarray):
    """Fit the baseline's booster with ``sample_weight``. Same hyperparameters, same seed."""
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=_gbm.SEED,
        n_jobs=1,
        verbose=-1,
        deterministic=True,
        force_row_wise=True,
    )
    model.fit(prepared.X_train, prepared.y_train, sample_weight=weights)
    return model


def fit_fair_constrained(prepared: Prepared):
    """IN-processing: a demographic-parity-constrained booster, by reweighted rounds.

    The constraint is enforced by a reduction, which is the standard shape of
    a fairness-constrained booster: repeatedly fit the ordinary booster on
    reweighted rows, where the weights move against whichever group the
    current model is approving least. Concretely, at each round the
    training-fold approval rate is measured per eligible group at the global
    cut-off, and for a group approved at rate ``r`` against the target
    ``t``, its approved (non-denied) rows are multiplied by
    ``exp(step * (t - r))`` and its denied rows by the reciprocal. A group
    approved too rarely therefore gains weight on the evidence that it should
    be approved.

    :data:`FAIR_GBM_ROUNDS` and :data:`FAIR_GBM_STEP` are fixed constants and
    the FINAL round's model is returned unconditionally. No round is selected
    on its result, on either fold. Selecting the best-looking round would be
    tuning the reported number into existence.
    """
    weights = np.ones(len(prepared.y_train), dtype="float64")
    groups = np.asarray(prepared.g_train.astype(str))
    labels = np.asarray(prepared.y_train)
    target = prepared.target_approval_rate
    model = _fit_weighted(prepared, weights)

    for _ in range(FAIR_GBM_ROUNDS):
        scores = denial_scores(model, prepared.X_train)
        cutoff = global_threshold(scores, target)
        approve = (scores < cutoff).astype(int)
        for value in prepared.eligible_groups:
            mask = groups == value
            if not mask.any():
                continue
            rate = float(approve[mask].mean())
            adjust = float(np.exp(FAIR_GBM_STEP * (target - rate)))
            weights[mask & (labels == 0)] *= adjust
            weights[mask & (labels == 1)] /= adjust
        weights = weights / weights.mean()
        model = _fit_weighted(prepared, weights)
    return model


# --------------------------------------------------------------------------
# Running one technique, and all three.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Baseline:
    """The unmitigated baseline model and what it did. Every technique is measured against this."""

    model: object
    threshold: float
    outcome: Outcome


def fit_unmitigated(prepared: Prepared, assumptions) -> Baseline:
    """Fit the baseline gradient-boosted model with no mitigation, and measure what it did.

    This is the named baseline every disparity-cut and margin-change number
    is stated against: a metric with no named baseline is not
    a metric.
    """
    model = _gbm.fit_gbm(prepared.X_train, prepared.y_train)
    threshold = global_threshold(
        denial_scores(model, prepared.X_train), prepared.target_approval_rate
    )
    approve = decide(denial_scores(model, prepared.X_test), prepared.g_test, threshold)
    return Baseline(model=model, threshold=threshold, outcome=measure(approve, prepared, assumptions))


def run_mitigation(
    technique: str,
    prepared: Prepared,
    baseline: Baseline,
    assumptions: _economics.EconomicsAssumptions,
) -> MitigationResult:
    """Run one registered mitigation technique and compare it to the unmitigated model.

    ``technique`` must be a key of :data:`TECHNIQUE_REGISTRY`; any other
    value raises KeyError rather than silently no-op'ing. Always returns all
    three of disparity cut, accuracy change and margin change together:
    there is no code path in this module that returns the
    disparity cut alone.
    """
    stage = TECHNIQUE_REGISTRY[technique]
    note = ""

    if technique == "reweighing":
        model = fit_reweighed(prepared)
        threshold = global_threshold(
            denial_scores(model, prepared.X_train), prepared.target_approval_rate
        )
        approve = decide(denial_scores(model, prepared.X_test), prepared.g_test, threshold)
    elif technique == "fair_constrained_gbm":
        model = fit_fair_constrained(prepared)
        threshold = global_threshold(
            denial_scores(model, prepared.X_train), prepared.target_approval_rate
        )
        approve = decide(denial_scores(model, prepared.X_test), prepared.g_test, threshold)
    elif technique == "per_group_threshold":
        model = baseline.model
        train_scores = denial_scores(model, prepared.X_train)
        table = per_group_thresholds(
            train_scores,
            prepared.g_train,
            prepared.target_approval_rate,
            prepared.eligible_groups,
            baseline.threshold,
            prepared.min_count,
        )
        approve = decide(
            denial_scores(model, prepared.X_test),
            prepared.g_test,
            baseline.threshold,
            group_thresholds=table,
        )
        note = (
            "LEGALLY CONTESTED: a different approval cut-off by race is disparate "
            "treatment on its face in the US. Measured for completeness of the "
            "trade-off study, NOT presented as approved practice. See docs/LIMITS.md."
        )
    else:  # pragma: no cover - unreachable: the registry lookup above raises first
        raise KeyError(technique)

    return MitigationResult(
        technique=technique,
        stage=stage,
        before=baseline.outcome,
        after=measure(approve, prepared, assumptions),
        note=note,
    )


def compare_all(
    prepared: Prepared | None = None,
    assumptions: _economics.EconomicsAssumptions | None = None,
    frame=None,
    source_label: str | None = None,
    sample_seed: int | None = None,
) -> tuple[Prepared, Baseline, list[MitigationResult]]:
    """Run every technique in :data:`TECHNIQUE_REGISTRY` and return all results.

    This is what ``hmda mitigate --compare`` prints, one line per
    technique.
    """
    if assumptions is None:
        assumptions = _economics.load_assumptions()
    if prepared is None:
        prepared = prepare(
            frame=frame, source_label=source_label, sample_seed=sample_seed
        )
    baseline = fit_unmitigated(prepared, assumptions)
    results = [
        run_mitigation(name, prepared, baseline, assumptions) for name in TECHNIQUE_REGISTRY
    ]
    return prepared, baseline, results


# --------------------------------------------------------------------------
# Output. This module carries its own entry
# point, and the CLI wiring in cli.py is a one-line call to main().
# --------------------------------------------------------------------------
def _fixture_warning(prepared: Prepared) -> list[str]:
    # The provenance block comes FIRST and states the sample size and
    # seed, because nothing below it is meaningful without them.
    groups = ", ".join(
        f"{g} (n={prepared.test_group_counts.get(g, 0)})"
        for g in prepared.eligible_groups
    )
    return [
        f"hmda mitigate  [OFFLINE COUNTERFACTUAL STUDY over historical applications;",
        "                nothing here is deployed, live, or in production]",
        *_evaluate.provenance_lines(
            prepared.rows_raw,
            prepared.source_label,
            prepared.sample_seed,
            prepared.fixture_path,
        ),
        f"  time split at activity_year {SPLIT_YEAR}: {prepared.rows_train} train rows / "
        f"{prepared.rows_test} held-out rows (the baseline model's split, not a new one)",
        f"  groups compared, with held-out row count: {groups} "
        f"(minimum {prepared.min_count} held-out applications per group)",
        f"  race not reported on {prepared.race_not_available_share * 100:.1f}% of held-out "
        "applications; those rows carry no group and are excluded from the gap",
    ]


def report_lines(
    prepared: Prepared,
    baseline: Baseline,
    results: list[MitigationResult],
    assumptions: _economics.EconomicsAssumptions,
    profit: bool = False,
) -> list[str]:
    """Every line both --compare and --profit print. One line per technique, three numbers each."""
    lines = _fixture_warning(prepared)
    lines.append("")
    lines.extend(assumptions.header_lines())
    lines.append(
        f"  principal of approved applications, measured from the data, not assumed: "
        f"mean ${baseline.outcome.mean_principal_usd:,.0f}, median "
        f"${baseline.outcome.median_principal_usd:,.0f}. The mean is the right aggregate "
        "for an expected loss but it is heavily right-skewed by a few very large "
        "multifamily loans, so both are printed."
    )
    lines.append(
        "  SIGN WARNING: the direction of every margin change below depends on whether the "
        "assumed per-loan margin exceeds the assumed expected loss "
        "(default rate x loss given default x principal). Under this file's assumptions it "
        f"does: ${assumptions.expected_value_per_approved_loan(baseline.outcome.mean_principal_usd):,.0f} "
        "expected value per approved application, so approving MORE people earns more. Set a "
        "margin below the expected loss and every sign flips. That is a property of the "
        "assumptions, not a finding about lending."
    )
    lines.append("")
    lines.append(
        f"UNMITIGATED MODEL (the named baseline): approval-rate gap "
        f"{baseline.outcome.approval_gap_pp:.1f} percentage points between "
        f"{baseline.outcome.high_group} ({baseline.outcome.group_rates[baseline.outcome.high_group] * 100:.1f}% approved) "
        f"and {baseline.outcome.low_group} ({baseline.outcome.group_rates[baseline.outcome.low_group] * 100:.1f}% approved); "
        f"agrees with the historical decision {baseline.outcome.accuracy_pct:.2f}% of the time; "
        f"${baseline.outcome.margin_per_1000_usd:,.0f} expected margin per 1,000 applications"
    )
    lines.append("")
    for result in results:
        lines.append(result.one_line())
        if result.note:
            lines.append(f"    ! {result.note}")
    if profit:
        lines.append("")
        lines.append("EXPECTED MARGIN PER 1,000 APPLICATIONS (M12):")
        lines.append(
            f"  unmitigated ............ ${baseline.outcome.margin_per_1000_usd:,.0f}"
        )
        for result in results:
            lines.append(
                f"  {result.technique:<22} ${result.after.margin_per_1000_usd:,.0f} "
                f"({result.margin_change_usd_per_1000:+,.0f} vs unmitigated)"
            )
        lines.append(
            "  LIMITATION: one assumed default rate is applied to every approval, "
            "including applications the historical lender denied. Those are plausibly "
            "riskier, so this UNDERSTATES the cost of approving more people."
        )
    return lines


def main(
    argv: list[str] | None = None,
    frame=None,
    source_label: str | None = None,
    sample_seed: int | None = None,
) -> int:
    """Print the M11 comparison and the M12 profit table. Both always print all three numbers.

    Pass ``frame`` (a ``Source.sample`` DataFrame) plus ``source_label``
    and ``sample_seed`` to run against the national data.
    """
    assumptions = _economics.load_assumptions()
    prepared, baseline, results = compare_all(
        assumptions=assumptions,
        frame=frame,
        source_label=source_label,
        sample_seed=sample_seed,
    )
    for line in report_lines(prepared, baseline, results, assumptions, profit=True):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
