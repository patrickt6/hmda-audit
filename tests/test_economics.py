"""The M12 assumptions are a config file, not a constant in the code.

The tests here exist to make one property mechanical: the
printed profit number is a function of ``config/economics.yaml``, so editing
the yaml moves the number and a reader can falsify it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hmda.model import economics as E

CONFIG = Path("config/economics.yaml")


def test_the_shipped_config_loads_and_states_all_three_assumptions():
    a = E.load_assumptions(CONFIG)
    assert a.expected_margin_per_approved_loan_usd > 0.0
    assert 0.0 <= a.loss_given_default_rate <= 1.0
    assert 0.0 <= a.default_rate_assumption <= 1.0


def test_a_null_assumption_is_refused_rather_than_defaulted(tmp_path):
    """A missing assumption must raise. A silent default is the unfalsifiable case."""
    path = tmp_path / "economics.yaml"
    path.write_text(
        "expected_margin_per_approved_loan_usd: null\n"
        "loss_given_default_rate: 0.25\n"
        "default_rate_assumption: 0.02\n"
    )
    with pytest.raises(ValueError, match="unset"):
        E.load_assumptions(path)


def test_a_fraction_outside_the_unit_interval_is_refused(tmp_path):
    path = tmp_path / "economics.yaml"
    path.write_text(
        "expected_margin_per_approved_loan_usd: 2000.0\n"
        "loss_given_default_rate: 1.5\n"
        "default_rate_assumption: 0.02\n"
    )
    with pytest.raises(ValueError, match="fraction"):
        E.load_assumptions(path)


def test_a_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        E.load_assumptions(tmp_path / "nope.yaml")


def _write(path: Path, margin: str, lgd: str, default_rate: str) -> None:
    path.write_text(
        f"expected_margin_per_approved_loan_usd: {margin}\n"
        f"loss_given_default_rate: {lgd}\n"
        f"default_rate_assumption: {default_rate}\n"
    )


def test_changing_the_yaml_changes_the_number(tmp_path):
    """The check that matters here: the margin is a function of the file."""
    low = tmp_path / "low.yaml"
    high = tmp_path / "high.yaml"
    _write(low, "2000.0", "0.25", "0.02")
    _write(high, "4000.0", "0.25", "0.02")

    kwargs = dict(approvals_per_1000=800.0, mean_principal_usd=300000.0)
    a = E.expected_margin_per_1000(assumptions=E.load_assumptions(low), **kwargs)
    b = E.expected_margin_per_1000(assumptions=E.load_assumptions(high), **kwargs)
    assert b > a
    assert b - a == pytest.approx(800.0 * 2000.0)


def test_raising_the_loss_given_default_lowers_the_margin(tmp_path):
    low = tmp_path / "low.yaml"
    high = tmp_path / "high.yaml"
    _write(low, "2000.0", "0.25", "0.02")
    _write(high, "2000.0", "0.75", "0.02")
    kwargs = dict(approvals_per_1000=800.0, mean_principal_usd=300000.0)
    a = E.expected_margin_per_1000(assumptions=E.load_assumptions(low), **kwargs)
    b = E.expected_margin_per_1000(assumptions=E.load_assumptions(high), **kwargs)
    assert b < a


def test_no_rate_literal_is_hardcoded_in_the_module():
    """The same rule a grep check enforces, kept here as a test so it cannot rot."""
    import re

    text = Path(E.__file__).read_text()
    assert re.search(r"0\.0[0-9]", text) is None


def test_the_header_prints_both_assumptions_the_gate_names():
    a = E.load_assumptions(CONFIG)
    header = "\n".join(a.header_lines())
    assert "margin" in header.lower()
    assert "loss given default" in header.lower()
    assert str(CONFIG) in header
    assert "ILLUSTRATIVE" in header
