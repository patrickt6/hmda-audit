"""Tests for the metric-translation layer.

What this file checks: a raw AUC string never appears in translate.py's
output.

The binding rule: a banned statistic must be translated before it ships.
``AUC`` must become "ranks defaulting loans above healthy ones X% of the
time", and a metric with no named baseline is not a metric.
"""

from __future__ import annotations

import pytest

from hmda.model import translate as T

#: A spread of plausible ranking qualities, including the exact values this
#: repo measured on the fixture on 2026-09-11 (logistic 0.8038, GBM 0.8750).
SAMPLE_AUCS = [0.5, 0.55, 0.62, 0.7123, 0.8038, 0.8750, 0.93, 0.999, 1.0]


def _all_outputs() -> list[str]:
    """Every user-facing string this module can produce, for the banned-string sweep."""
    out: list[str] = []
    for auc in SAMPLE_AUCS:
        out.append(T.auc_to_sentence(auc))
        out.append(T.auc_to_sentence(auc, include_pair_clause=False))
        out.append(T.translate_or_abstain("auc", auc))
    out.append(T.auc_to_sentence(0.43))  # a model worse than chance
    out.append(T.baseline_sentence())
    out.extend(T.translate_or_abstain(k) for k in T.UNTRANSLATABLE)
    out.append(T.translate_or_abstain("some_unknown_statistic"))
    return out


# --------------------------------------------------------------------------
# THE GATE: a raw AUC string never reaches user-facing output.
# --------------------------------------------------------------------------
def test_raw_auc_string_never_appears_in_output():
    """No output contains the token 'AUC', nor the raw number in any rounding.

    This is the check that matters here. 'Raw AUC string' is read as both the NAME of the
    statistic and its DECIMAL VALUE at any sensible precision, because either
    one reaching a reader defeats the translation.
    """
    for text in _all_outputs():
        assert "auc" not in text.lower(), text
        assert "roc_auc" not in text.lower(), text
        assert "roc curve" not in text.lower(), text
        for auc in SAMPLE_AUCS + [0.43]:
            for places in (2, 3, 4):
                raw = f"{auc:.{places}f}"
                assert raw not in text, f"raw value {raw!r} leaked into: {text!r}"


def test_no_other_banned_statistic_name_appears():
    """Banned outputs: R2, Brier, p-value and confidence interval never ship raw.

    The refusal strings are allowed to NAME the statistic they refuse --- that
    is the point of a refusal --- so only the sentences that actually render a
    result are swept here.
    """
    rendered = [T.auc_to_sentence(a) for a in SAMPLE_AUCS] + [T.baseline_sentence()]
    for text in rendered:
        lowered = text.lower()
        for banned in ("r2", "r-squared", "brier", "p-value", "p value", "confidence interval"):
            assert banned not in lowered, text


# --------------------------------------------------------------------------
# The formula is the one documented, and it does not overstate.
# --------------------------------------------------------------------------
def test_percent_is_the_documented_ratio_against_the_named_baseline():
    """P = (auc - baseline) / baseline * 100, computed by hand here."""
    assert T.percent_better_than_baseline(0.75, 0.5) == pytest.approx(50.0)
    assert T.percent_better_than_baseline(0.5, 0.5) == pytest.approx(0.0)
    assert T.percent_better_than_baseline(1.0, 0.5) == pytest.approx(100.0)
    # Hand-computed from the fixture measurement of 2026-09-11:
    # (0.8750 - 0.5) / 0.5 * 100 = 75.0
    assert T.percent_better_than_baseline(0.8750, 0.5) == pytest.approx(75.0)


def test_percent_can_never_exceed_one_hundred():
    """A perfect model orders every pair correctly, so 100% is the ceiling.

    A translation that could print "300% better" would be arithmetically
    wrong, and this test is what would catch it.
    """
    for auc in SAMPLE_AUCS:
        assert T.percent_better_than_baseline(auc) <= 100.0 + 1e-9


def test_sentence_names_its_baseline():
    """A metric with no named baseline is not a metric."""
    text = T.auc_to_sentence(0.8750)
    assert "always guessing the base rate" in text
    assert "%" in text


def test_sentence_states_the_pair_ordering_translation_the_grammar_prescribes():
    """AUC becomes a rate of correctly ordered pairs."""
    text = T.auc_to_sentence(0.8750)
    assert "88 of every 100 denied/approved pairs" in text
    assert "against 50 for the baseline" in text


def test_pair_clause_can_be_switched_off_and_the_percent_survives():
    text = T.auc_to_sentence(0.8750, include_pair_clause=False)
    assert text == "ranks denials 75% better than always guessing the base rate"


# --------------------------------------------------------------------------
# A weak or failed result is stated plainly, never dressed up.
# --------------------------------------------------------------------------
def test_a_model_that_does_not_beat_the_baseline_is_said_so():
    text = T.auc_to_sentence(0.5)
    assert "does not rank denials any better" in text
    assert "better than always guessing the base rate" not in text.replace(
        "does not rank denials any better than always guessing the base rate", ""
    )


def test_a_model_worse_than_chance_is_reported_as_worse():
    text = T.auc_to_sentence(0.43)
    assert "WORSE" in text


def test_a_non_number_is_refused_rather_than_rendered():
    with pytest.raises(ValueError):
        T.percent_better_than_baseline(float("nan"))


def test_a_non_positive_baseline_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError):
        T.percent_better_than_baseline(0.75, 0.0)


# --------------------------------------------------------------------------
# Abstention: where no faithful translation exists, say so.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("statistic", sorted(T.UNTRANSLATABLE))
def test_untranslatable_statistics_are_refused_with_a_reason(statistic):
    """Translate or drop, never invent."""
    text = T.translate_or_abstain(statistic)
    assert text.startswith("NOT TRANSLATABLE:")
    assert len(text) > len("NOT TRANSLATABLE: ") + 20  # it carries a reason


def test_an_unknown_statistic_is_refused_not_guessed():
    text = T.translate_or_abstain("mean_absolute_percentage_error")
    assert text.startswith("NOT TRANSLATABLE:")


def test_translating_auc_without_a_value_raises():
    with pytest.raises(ValueError):
        T.translate_or_abstain("auc")


def test_baseline_ranking_quality_constant_is_one_half():
    """A constant score makes every pair a tie, and a tie counts half."""
    assert T.BASE_RATE_BASELINE_AUC == 0.5
