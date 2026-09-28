"""C-37: the model card states verbatim that this model makes no credit decisions.

docs/CONTROLS.md C-37: "The model card states verbatim that this model is not
used to make credit decisions." Implemented at
src/hmda/governance/model_card.py:18, which is the constant
``NO_CREDIT_DECISIONS_SENTENCE``.

WHAT THESE TESTS DO AND DO NOT COVER (read before trusting them).

These tests check the two things the control needs directly:

  * the constant the renderer is contractually required to emit, and
  * the shipped artifact docs/MODEL-CARD.md, which contains the sentence
    verbatim (a generated file; see its header).

That is a real control: deleting the sentence from either place turns these
red, which is exactly what C-37 asks for. It does not exercise
``render_model_card`` itself (tests/test_model_card.py covers rendering
against the live database); it is backed against the shipped artifact and
the constant.

Proven red on 2026-09-13 by deleting the sentence from docs/MODEL-CARD.md and
by mutating the constant.
"""

from __future__ import annotations

from hmda.governance import model_card
from hmda.governance.controls import REPO_ROOT

MODEL_CARD_DOC = REPO_ROOT / "docs" / "MODEL-CARD.md"


def test_the_no_credit_decisions_sentence_says_what_it_claims_to_say():
    """The constant is the control's wording; a reworded constant is a changed control."""
    sentence = model_card.NO_CREDIT_DECISIONS_SENTENCE
    lowered = sentence.lower()
    assert "not used to make credit decisions" in lowered, (
        f"NO_CREDIT_DECISIONS_SENTENCE no longer states the exclusion: {sentence!r}"
    )
    assert lowered.startswith("this model is not"), (
        f"NO_CREDIT_DECISIONS_SENTENCE is no longer an unqualified negation: {sentence!r}"
    )


def test_the_shipped_model_card_contains_the_sentence_verbatim():
    """C-37: the artifact a reader opens carries the out-of-scope-use sentence."""
    assert MODEL_CARD_DOC.is_file(), f"{MODEL_CARD_DOC} does not exist"
    text = MODEL_CARD_DOC.read_text(encoding="utf-8")
    assert model_card.NO_CREDIT_DECISIONS_SENTENCE in text, (
        f"{MODEL_CARD_DOC} does not contain "
        f"{model_card.NO_CREDIT_DECISIONS_SENTENCE!r} verbatim. C-37 requires the "
        "model record to state its out-of-scope use."
    )


def test_the_model_card_never_claims_the_model_makes_credit_decisions():
    """The inverse guard: an affirming sentence beside the disclaimer would void it."""
    text = MODEL_CARD_DOC.read_text(encoding="utf-8").lower()
    for banned in (
        "is used to make credit decisions",
        "makes credit decisions",
        "used for credit decisions",
    ):
        assert banned not in text, (
            f"{MODEL_CARD_DOC} contains {banned!r}, which contradicts C-37."
        )
