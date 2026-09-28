"""M2 regression: `hmda audit --all-years` must actually run the four audit
branches (waterfall, air, controlled, drift), not just print a SOURCE line.

Bug history: `--all-years` satisfied
the CLI's `if not any([...])` usage guard, so the command exited 0, but no
`if waterfall:` / `if air:` / `if controlled:` / `if drift:` block was ever
gated on `all_years`, so the flag ran nothing -- confirmed by a 0.09s wall
time against a 36.7M-row national file (`/usr/bin/time -p .venv/bin/hmda
audit --all-years --source national`, real=0.09, EXIT=0). This test drives
the real CLI through click's CliRunner (not a reimplementation of the
branch logic) on the small fixture source, and asserts output specific to
each of the four branches is present. Before the fix in src/hmda/cli.py
(forcing waterfall=air=controlled=drift=True when all_years is set), this
test fails because none of those four blocks' output strings appear.
"""

from __future__ import annotations

from click.testing import CliRunner

from hmda.cli import main


def _run(*args) -> str:
    result = CliRunner().invoke(main, list(args))
    assert result.exit_code == 0, (
        f"hmda {' '.join(args)} exited {result.exit_code}\n"
        f"{result.output}\n{result.exception!r}"
    )
    return result.output


def test_all_years_runs_the_waterfall_branch():
    """--all-years must print the waterfall table (step/rows_before/rows_after/dropped)."""
    output = _run("audit", "--all-years", "--source", "fixture")
    assert "rows_before" in output and "rows_after" in output and "dropped" in output, (
        "hmda audit --all-years printed no waterfall table.\n" + output
    )


def test_all_years_runs_the_air_branch():
    """--all-years must print the four-fifths screen table and its flag count."""
    output = _run("audit", "--all-years", "--source", "fixture")
    assert "Four-fifths screen" in output, (
        "hmda audit --all-years printed no four-fifths (AIR) screen.\n" + output
    )
    assert "Lenders flagged for review" in output, (
        "hmda audit --all-years printed no flagged-lender count.\n" + output
    )


def test_all_years_runs_the_controlled_branch():
    """--all-years must print the raw/controlled gap comparison."""
    output = _run("audit", "--all-years", "--source", "fixture")
    assert "raw gap" in output and "controlled gap" in output, (
        "hmda audit --all-years printed no controlled-disparity comparison.\n" + output
    )


def test_all_years_runs_the_drift_branch():
    """--all-years must print the drift module's own output shape: 'N of M ... shifted'."""
    output = _run("audit", "--all-years", "--source", "fixture")
    assert "shifted" in output.lower(), (
        "hmda audit --all-years printed no drift/stability output.\n" + output
    )


def test_all_years_alone_is_not_a_no_op():
    """Regression for the exact defect: at minimum, more than just the SOURCE line.

    The pre-fix behavior printed only the source label (and, in an earlier
    revision, nothing at all) and exited 0 in well under a second regardless
    of dataset size, which is what let 0.09s real time on a 36.7M-row file be
    mistaken for a completed full pass. A single-line output is the fingerprint
    of that no-op; this asserts the output is not that.
    """
    output = _run("audit", "--all-years", "--source", "fixture")
    non_blank_lines = [line for line in output.splitlines() if line.strip()]
    assert len(non_blank_lines) > 5, (
        "hmda audit --all-years produced suspiciously little output "
        f"({len(non_blank_lines)} non-blank lines) -- looks like the M2 no-op again.\n"
        + output
    )
