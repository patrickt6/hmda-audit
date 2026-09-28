"""The model card is generated, and its numbers cannot be hand-edited.

What these tests cover, and what they do not.

COVERED WITHOUT THE DATABASE: the token grammar the verifier relies on, the
fact-shape rules (a sample-based fact must carry its sample size), and the
two failure modes ``verify_card`` exists to catch -- a hand-edited number and
a deleted C-37 sentence -- both exercised against a synthetic card built from
synthetic facts, so they run anywhere.

COVERED ONLY WITH ``data/hmda.duckdb`` PRESENT: that ``render_model_card``
really reads the national file, that a re-render is byte-stable, and that the
shipped ``docs/MODEL-CARD.md`` verifies against today's sources. Those tests
skip when the 3 GB database is absent (it is git-ignored), because a
missing large artifact is not a code defect.

NOT COVERED: whether the sample numbers recorded in ``docs/VERIFICATION.md``
section 7 ("Recorded model runs") are themselves correct. This module checks
that the card cannot drift from that document; it cannot check the document
against the runs that produced it.
"""

from __future__ import annotations

import pytest

from hmda.governance import model_card
from hmda.governance.controls import REPO_ROOT

DB_PATH = REPO_ROOT / "data" / "hmda.duckdb"
CARD_PATH = REPO_ROOT / "docs" / "MODEL-CARD.md"
NUMBERS_PATH = REPO_ROOT / "docs" / "VERIFICATION.md"
CONTROLS_PATH = REPO_ROOT / "docs" / "CONTROLS.md"

needs_db = pytest.mark.skipif(
    not DB_PATH.is_file(), reason=f"{DB_PATH} not present (git-ignored national database)"
)


# --------------------------------------------------------------------------
# The token grammar the verifier depends on
# --------------------------------------------------------------------------
def test_metric_and_control_identifiers_are_not_read_as_numbers():
    """M13 and C-37 are labels. Reading them as numbers would make every card fail."""
    tokens = model_card.number_tokens("M13 is unmeasurable and C-37 is backed by 238 tests.")
    assert tokens == ["238"], tokens


def test_a_thousands_separated_number_is_one_token():
    assert model_card.number_tokens("36,734,685 applications") == ["36,734,685"]


def test_percentages_and_decimals_survive_tokenizing():
    assert model_card.number_tokens("26.73% and 0.59 points") == ["26.73%", "0.59"]


def test_banned_abbreviations_are_detected_case_insensitively():
    assert model_card.banned_terms_in("the auc was high") == ["auc"]
    assert model_card.banned_terms_in("ranks denials better than chance") == []


# --------------------------------------------------------------------------
# verify_card's two failure modes, on a synthetic card
# --------------------------------------------------------------------------
def _synthetic_facts():
    return {
        "applications": model_card.Fact("applications", "1,234", "full-file, measured", "SELECT 1"),
    }


def _write_card(tmp_path, body: str):
    path = tmp_path / "MODEL-CARD.md"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture()
def stub_facts(monkeypatch):
    """Replace fact collection so these tests need no database and no docs."""
    monkeypatch.setattr(model_card, "collect_facts", lambda **kwargs: _synthetic_facts())


def test_a_card_whose_numbers_are_all_facts_verifies(tmp_path, stub_facts):
    card = _write_card(
        tmp_path, f"{model_card.NO_CREDIT_DECISIONS_SENTENCE}\n\n1,234 applications.\n"
    )
    assert model_card.verify_card(card_path=card) == []


def test_a_hand_edited_number_fails_verification(tmp_path, stub_facts):
    """The check this module exists for: one changed digit must go red."""
    card = _write_card(
        tmp_path, f"{model_card.NO_CREDIT_DECISIONS_SENTENCE}\n\n1,235 applications.\n"
    )
    problems = model_card.verify_card(card_path=card)
    assert problems, "a hand-edited number verified clean; the check cannot fail"
    assert "1,235" in problems[0]


def test_a_card_missing_the_c37_sentence_fails_verification(tmp_path, stub_facts):
    card = _write_card(tmp_path, "1,234 applications.\n")
    problems = model_card.verify_card(card_path=card)
    assert any("C-37" in p for p in problems), problems


def test_a_banned_abbreviation_fails_verification(tmp_path, stub_facts):
    card = _write_card(
        tmp_path,
        f"{model_card.NO_CREDIT_DECISIONS_SENTENCE}\n\n1,234 applications, AUC reported.\n",
    )
    problems = model_card.verify_card(card_path=card)
    assert any("banned abbreviation" in p for p in problems), problems


def test_a_missing_card_is_a_problem_not_a_pass(tmp_path, stub_facts):
    problems = model_card.verify_card(card_path=tmp_path / "absent.md")
    assert problems and "does not exist" in problems[0]


# --------------------------------------------------------------------------
# Fact shape
# --------------------------------------------------------------------------
def test_recorded_sample_facts_carry_their_sample_size():
    """A sample number is never presented as a full-file number."""
    facts = model_card._recorded_facts(NUMBERS_PATH)
    sampled = [f for f in facts if f.key.startswith(("ranking_", "gap_cut_", "accuracy_cost_"))]
    assert sampled, "no sample-based facts were parsed"
    for fact in sampled:
        assert "sample-based" in fact.basis and "n=" in fact.basis, fact


def test_control_counts_are_read_from_the_registry_not_written_down():
    """Both control numbers come from docs/CONTROLS.md, so they cannot overstate coverage."""
    facts = {f.key: f for f in model_card._control_facts(CONTROLS_PATH)}
    documented = int(facts["controls_documented"].value.replace(",", ""))
    backed = int(facts["controls_backed"].value.replace(",", ""))
    assert documented >= backed
    assert documented > 0


def test_a_reworded_measurement_doc_raises_rather_than_guessing(tmp_path):
    """If an anchor stops matching, the render must stop, not fall back on a constant."""
    broken = tmp_path / "VERIFICATION.md"
    broken.write_text("nothing measured here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        model_card._recorded_facts(broken)


# --------------------------------------------------------------------------
# The real render (needs the national database)
# --------------------------------------------------------------------------
@needs_db
def test_rendering_produces_the_c37_sentence_verbatim(tmp_path):
    """C-37 against the GENERATOR, not only against the shipped artifact."""
    text = model_card.render_model_card(db_path=DB_PATH, out_path=tmp_path / "card.md")
    assert model_card.NO_CREDIT_DECISIONS_SENTENCE in text


@needs_db
def test_the_rendered_card_carries_no_banned_abbreviation(tmp_path):
    text = model_card.render_model_card(db_path=DB_PATH, out_path=tmp_path / "card.md")
    assert model_card.banned_terms_in(text) == []


@needs_db
def test_rendering_twice_is_byte_stable(tmp_path):
    """The check that matters: git diff --exit-code after a re-run must be clean."""
    first = model_card.render_model_card(db_path=DB_PATH, out_path=tmp_path / "a.md")
    second = model_card.render_model_card(db_path=DB_PATH, out_path=tmp_path / "b.md")
    assert first == second


@needs_db
def test_the_rendered_card_states_the_model_was_never_deployed(tmp_path):
    """This is offline study only: nothing here ever scored a real applicant."""
    text = model_card.render_model_card(db_path=DB_PATH, out_path=tmp_path / "card.md").lower()
    assert "never deployed" in text
    assert "never scored a real applicant" in text


@needs_db
def test_the_shipped_card_verifies_against_todays_sources():
    assert model_card.verify_card(card_path=CARD_PATH, db_path=DB_PATH) == []
