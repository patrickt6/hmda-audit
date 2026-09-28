"""Tests for the SR 11-7 / OSFI E-23 control mapping (metric M10).

The check this file protects: ``hmda verify --governance`` must go RED when a
control names a test that does not exist. A check that cannot fail is not a
check, so that case is tested directly against a synthetic table rather than
only against the real one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hmda.governance import controls as ctl

REPO_ROOT = ctl.REPO_ROOT
CONTROLS_DOC = REPO_ROOT / "docs" / "CONTROLS.md"


@pytest.fixture(scope="module")
def real_controls():
    return ctl.load_controls(CONTROLS_DOC)


# --------------------------------------------------------------------------
# the real table
# --------------------------------------------------------------------------
def test_the_controls_doc_parses_into_at_least_one_control(real_controls):
    assert len(real_controls) > 0


def test_every_parsed_row_has_all_six_fields(real_controls):
    for control in real_controls:
        assert control.control_id
        assert control.assertion
        assert control.implementing_file
        assert control.test
        assert control.sr_11_7_clause
        assert control.osfi_e23_clause


def test_control_ids_are_unique(real_controls):
    ids = [c.control_id for c in real_controls]
    assert len(ids) == len(set(ids))


def test_parsed_count_equals_the_row_count_in_the_markdown(real_controls):
    """The count M10 reports must equal the rows a human can count in the file."""
    text = CONTROLS_DOC.read_text(encoding="utf-8")
    hand_counted = sum(1 for line in text.splitlines() if line.strip().startswith("| C-"))
    assert len(real_controls) == hand_counted


def test_every_named_test_exists(real_controls):
    missing = ctl.missing_tests(real_controls)
    assert missing == [], [c.control_id for c in missing]


def test_every_implementing_file_line_exists(real_controls):
    missing = ctl.missing_implementations(real_controls)
    assert missing == [], [c.control_id for c in missing]


def test_verify_controls_passes_on_the_real_table(real_controls):
    assert ctl.verify_controls(real_controls) is True


def test_main_exits_zero_on_the_real_table(capsys):
    assert ctl.main(CONTROLS_DOC) == 0
    assert "governance controls documented:" in capsys.readouterr().out


def test_main_prints_the_control_count_that_is_metric_m10(real_controls, capsys):
    ctl.main(CONTROLS_DOC)
    out = capsys.readouterr().out
    assert f"governance controls documented: {len(real_controls)}" in out


def test_no_framework_column_cites_a_numbered_clause(real_controls):
    """No citation may carry a section/clause number: none was opened.

    A fabricated "SR 11-7 §3.2" is worse than a named principle, because a
    bank reviewer will check it.
    """
    import re

    numbered = re.compile(r"(§|\bsection\s+\d|\bclause\s+\d|\b\d+\.\d+\b)", re.IGNORECASE)
    offenders = [
        c.control_id
        for c in real_controls
        if numbered.search(c.sr_11_7_clause) or numbered.search(c.osfi_e23_clause)
    ]
    assert offenders == [], offenders


def test_the_doc_carries_its_sourcing_caveat():
    text = CONTROLS_DOC.read_text(encoding="utf-8")
    assert "did **not**" in text or "did not" in text
    assert "approximate" in text


# --------------------------------------------------------------------------
# the check can go red
# --------------------------------------------------------------------------
_HEADER = (
    "| control_id | assertion | implementing_file | test | SR 11-7 clause | OSFI E-23 clause |\n"
    "|---|---|---|---|---|---|\n"
)


def _write_table(tmp_path: Path, rows: str) -> Path:
    path = tmp_path / "CONTROLS.md"
    path.write_text("# CONTROLS\n\n" + _HEADER + rows, encoding="utf-8")
    return path


def test_a_control_naming_a_nonexistent_test_fails_verification(tmp_path):
    doc = _write_table(
        tmp_path,
        "| C-99 | a | src/hmda/governance/controls.py:1 | "
        "tests/test_controls.py::test_this_name_does_not_exist | x | y |\n",
    )
    parsed = ctl.load_controls(doc)
    assert ctl.verify_controls(parsed) is False
    assert [c.control_id for c in ctl.missing_tests(parsed)] == ["C-99"]


def test_a_control_naming_a_nonexistent_test_file_fails_verification(tmp_path):
    doc = _write_table(
        tmp_path,
        "| C-98 | a | src/hmda/governance/controls.py:1 | "
        "tests/test_no_such_file.py::test_x | x | y |\n",
    )
    assert ctl.verify_controls(ctl.load_controls(doc)) is False


def test_main_exits_nonzero_when_a_named_test_is_missing(tmp_path, capsys):
    doc = _write_table(
        tmp_path,
        "| C-97 | a | src/hmda/governance/controls.py:1 | "
        "tests/test_controls.py::test_absent | x | y |\n",
    )
    assert ctl.main(doc) == 1
    assert "FAIL C-97" in capsys.readouterr().out


def test_a_not_implemented_control_is_counted_but_does_not_fail_the_gate(tmp_path, capsys):
    doc = _write_table(
        tmp_path,
        "| C-96 | a | src/hmda/governance/controls.py:1 | NOT IMPLEMENTED | x | y |\n",
    )
    parsed = ctl.load_controls(doc)
    assert len(parsed) == 1
    assert parsed[0].implemented is False
    assert ctl.verify_controls(parsed) is True
    assert ctl.main(doc) == 0
    assert "NOT IMPLEMENTED: 1" in capsys.readouterr().out


def test_a_bad_implementing_file_line_fails_the_gate(tmp_path, capsys):
    doc = _write_table(
        tmp_path,
        "| C-95 | a | src/hmda/governance/controls.py:999999 | NOT IMPLEMENTED | x | y |\n",
    )
    assert ctl.main(doc) == 1
    assert "C-95" in capsys.readouterr().out


def test_a_nonexistent_implementing_file_fails_the_gate(tmp_path):
    doc = _write_table(
        tmp_path,
        "| C-94 | a | src/hmda/no_such_module.py:1 | NOT IMPLEMENTED | x | y |\n",
    )
    assert ctl.main(doc) == 1


def test_a_row_with_the_wrong_cell_count_raises_rather_than_being_skipped(tmp_path):
    """A silently dropped control is a silently lowered M10."""
    doc = _write_table(tmp_path, "| C-93 | a | b |\n")
    with pytest.raises(ValueError, match="C-93"):
        ctl.load_controls(doc)


def test_a_duplicate_control_id_raises(tmp_path):
    row = "| C-92 | a | src/hmda/governance/controls.py:1 | NOT IMPLEMENTED | x | y |\n"
    doc = _write_table(tmp_path, row + row)
    with pytest.raises(ValueError, match="duplicate"):
        ctl.load_controls(doc)


def test_non_control_markdown_tables_are_ignored(tmp_path):
    doc = _write_table(
        tmp_path,
        "| C-91 | a | src/hmda/governance/controls.py:1 | NOT IMPLEMENTED | x | y |\n"
        "\n| other | table |\n|---|---|\n| some | row |\n",
    )
    assert [c.control_id for c in ctl.load_controls(doc)] == ["C-91"]


# --------------------------------------------------------------------------
# resolver unit tests
# --------------------------------------------------------------------------
def test_test_exists_finds_a_test_in_this_very_file():
    assert ctl.test_exists("tests/test_controls.py::test_control_ids_are_unique") is True


def test_test_exists_rejects_a_node_id_with_no_double_colon():
    assert ctl.test_exists("tests/test_controls.py") is False


def test_test_exists_tolerates_a_parametrisation_suffix():
    assert ctl.test_exists("tests/test_controls.py::test_control_ids_are_unique[0]") is True


def test_implementation_exists_accepts_a_real_line():
    assert ctl.implementation_exists("src/hmda/governance/controls.py:1") is True


def test_implementation_exists_rejects_a_locator_with_no_line_number():
    assert ctl.implementation_exists("src/hmda/governance/controls.py") is False


# --------------------------------------------------------------------------
# the validation document
# --------------------------------------------------------------------------
def test_validation_doc_names_the_sr_11_7_validation_elements():
    text = (REPO_ROOT / "docs" / "VALIDATION.md").read_text(encoding="utf-8").lower()
    for heading in ("conceptual soundness", "outcomes analysis", "benchmarking", "ongoing monitoring"):
        assert heading in text, heading


def test_validation_doc_records_the_unstable_mitigation_finding():
    """A validation report that records only successes is worthless."""
    text = (REPO_ROOT / "docs" / "VALIDATION.md").read_text(encoding="utf-8")
    assert "UNSTABLE" in text
    assert "500,000" in text
    assert "1,500,000" in text


def test_validation_doc_has_a_not_validated_section():
    text = (REPO_ROOT / "docs" / "VALIDATION.md").read_text(encoding="utf-8").lower()
    assert "not validated" in text
