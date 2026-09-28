"""Translate a raw AUC into an allowed, plain-English sentence.

Raw AUC never ships as an outcome: it is always translated into a
plain-English sentence before it reaches a reader. A test asserts a raw
AUC string never appears in this module's output.

What an honest translation of AUC actually is
=============================================
AUC has one plain-English reading that is exactly true rather than
approximately true. For a scoring model, AUC is the probability that a
randomly chosen DENIED application is scored above a randomly chosen
APPROVED one, counting a tie as half. It is a **pair-ordering rate**, and
nothing else. This is the precise translation used for AUC:
``AUC`` becomes "ranks defaulting loans above healthy ones X%
of the time".

The base-rate baseline gives every applicant the same score, so every pair is
a tie, so it orders exactly 50 of every 100 pairs correctly. That 50 is not
an estimate; it is what a constant score means. It is also the anchor
that the rule "a metric with no named baseline is not a metric"
demands.

So the percent in the M7 sentence is:

    P = (model_pairs_correct - baseline_pairs_correct) / baseline_pairs_correct

which with the base-rate baseline is ``(auc - 0.5) / 0.5``.

**Why this is honest, and what it is deliberately NOT.**

- It is a ratio of two quantities in the SAME unit --- share of
  denied/approved pairs ordered correctly --- so "P% better than" means the
  ordinary thing a reader assumes it means: P% more pairs put in the right
  order than the baseline manages.
- It is bounded and it cannot be inflated. The best possible model orders
  100 of 100 pairs, so P can never exceed 100%. A translation that produced
  "300% better" would be a sign the formula was wrong.
- It is **not** an accuracy claim. Ranking and deciding are different: a
  model can order pairs well and still be badly calibrated, and it names no
  threshold, so it never implies anyone was approved or denied: this
  module forbids implying deployment, and the wording here keeps to ordering.
- It is **not** a lift, a capture rate, or a Gini "score". Those need their
  own definitions and a reader cannot price them either.
- The percent is stated against a named baseline that the same command
  prints on the same line, so a reader can check the arithmetic.

A rejected alternative, recorded so nobody re-proposes it: reporting
``(auc - 0.5) / (1.0 - 0.5)`` as "closes P% of the gap to a perfect model".
It is the same number when the baseline is 0.5, but its wording implies the
model is P% of the way to perfection, which is a much stronger claim than
the arithmetic supports, and it silently breaks if the baseline is ever not
0.5. The pair-ordering wording is kept because it survives both.

**What is not translatable, and is therefore refused.** Some banned
statistics have no faithful one-sentence plain-English form at all. A
p-value and a confidence interval are statements about a procedure's
long-run behaviour, not about the size of an effect, and every popular
plain-English rendering of them ("95% sure", "significant") is wrong.
This repository allows them only as a rigor clause beside a
result. :func:`translate_or_abstain` therefore refuses them by name rather
than inventing a sentence. Refusing is the correct output, not a failure.
"""

from __future__ import annotations

#: The base-rate baseline's pair-ordering rate. A constant score makes every
#: denied/approved pair a tie, and a tie counts half.
BASE_RATE_BASELINE_AUC = 0.5

#: Statistics with no faithful plain-English outcome sentence. See the module
#: docstring. These are refused, never rendered.
UNTRANSLATABLE: dict[str, str] = {
    "p_value": (
        "a p-value states how surprising the data would be under a null "
        "hypothesis, not how large or how certain an effect is; it has no "
        "faithful one-sentence plain-English form and ships only as a rigor "
        "clause beside a result"
    ),
    "confidence_interval": (
        "a confidence interval describes the long-run behaviour of the "
        "interval-making procedure, not the probability that this interval "
        "holds the true value; it ships only as a rigor clause"
    ),
    "brier_score": (
        "a Brier score is a squared error on probabilities and has no "
        "meaning without a stated comparison; translate it as a percent "
        "against the base-rate baseline instead of rendering it"
    ),
}


def _pairs_per_hundred(auc: float) -> int:
    """Return how many of every 100 denied/approved pairs are ordered correctly."""
    return int(round(auc * 100))


def percent_better_than_baseline(auc: float, baseline_auc: float = BASE_RATE_BASELINE_AUC) -> float:
    """Return P: the percent more denied/approved pairs ordered correctly than the baseline.

    ``P = (auc - baseline_auc) / baseline_auc * 100``. Raises ``ValueError``
    on a non-positive baseline, because the ratio is undefined there and a
    silently substituted default would be an invented number.
    """
    if baseline_auc <= 0.0:
        raise ValueError("baseline ranking quality must be positive to state a percent against it")
    if auc != auc:  # NaN
        raise ValueError("ranking quality is not a number; nothing can be stated about it")
    return (auc - baseline_auc) / baseline_auc * 100.0


def auc_to_sentence(
    auc: float,
    baseline_auc: float = BASE_RATE_BASELINE_AUC,
    include_pair_clause: bool = True,
) -> str:
    """Convert ``auc`` into the allowed sentence shape.

    The M7 shape: "ranks denials ``[P]``% better than always guessing the
    base rate". ``[P]`` is the percent MORE denied/approved pairs the model
    orders correctly than the baseline does, per
    :func:`percent_better_than_baseline` --- never the raw number itself.

    With ``include_pair_clause`` (the default), a parenthetical states both
    models' pair-ordering rates out of 100, so a reader can check the
    percent rather than take it on trust. That parenthetical is the exact
    translation this module always uses for AUC.

    The returned string never contains the substring "AUC" and never
    contains the raw decimal, in any rounding.

    When the model does not beat the baseline, that is stated plainly. An
    honest weak result is the required output; there is no wording here that
    turns a non-result into one.
    """
    percent = percent_better_than_baseline(auc, baseline_auc)
    model_pairs = _pairs_per_hundred(auc)
    baseline_pairs = _pairs_per_hundred(baseline_auc)

    pair_clause = (
        f" (it puts {model_pairs} of every 100 denied/approved pairs in the right order, "
        f"against {baseline_pairs} for the baseline)"
    )

    if percent <= 0.0:
        body = "does not rank denials any better than always guessing the base rate"
        if percent < 0.0:
            body = (
                "ranks denials WORSE than always guessing the base rate, by "
                f"{abs(percent):.0f}%"
            )
        return body + (pair_clause if include_pair_clause else "")

    body = f"ranks denials {percent:.0f}% better than always guessing the base rate"
    return body + (pair_clause if include_pair_clause else "")


def baseline_sentence(baseline_auc: float = BASE_RATE_BASELINE_AUC) -> str:
    """State what the baseline itself does, in the same units as the challenger.

    The baseline must be printed beside the challenger: a run that prints
    only the challenger is a failure, and it needs a
    sentence of its own so the comparison is legible without arithmetic.
    """
    return (
        "always guessing the base rate puts "
        f"{_pairs_per_hundred(baseline_auc)} of every 100 denied/approved pairs in the "
        "right order, which is what an unordered guess achieves"
    )


def translate_or_abstain(statistic: str, value: float | None = None) -> str:
    """Translate ``statistic`` if a faithful translation exists, otherwise refuse.

    ``statistic`` is a key such as ``"auc"``, ``"p_value"``,
    ``"confidence_interval"`` or ``"brier_score"``. For anything in
    :data:`UNTRANSLATABLE`, this returns a refusal that names the reason. It
    does not invent a sentence, and it does not fall through to a generic
    rendering: it either translates or it refuses.
    """
    key = statistic.strip().lower()
    if key in UNTRANSLATABLE:
        return f"NOT TRANSLATABLE: {UNTRANSLATABLE[key]}"
    if key in {"auc", "roc_auc", "ranking_quality"}:
        if value is None:
            raise ValueError("a ranking-quality value is required to translate it")
        return auc_to_sentence(float(value))
    return (
        f"NOT TRANSLATABLE: no translation rule is defined for {statistic!r}; "
        "add a translation rule before it ships"
    )
