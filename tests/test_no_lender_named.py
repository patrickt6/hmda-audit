"""C-36: no individual lender is ever named as discriminating.

docs/CONTROLS.md C-36: "No individual lender is ever named as discriminating;
flagged lender identity is an internal artifact only."
Implemented at src/hmda/fairness/air.py:116 (``flagged_lenders`` returns the
identities, and the CLI reduces them to a count).

The control is about what a USER SEES, so these tests drive the real command
path through click's CliRunner and inspect the rendered output. Checking the
return value of ``flagged_lenders`` would prove the opposite of the control:
that function is SUPPOSED to hold identities. The control is that they stop
there.

Two things make these tests non-vacuous:

  * they assert first that the fixture actually HAS flagged lenders
    (8, measured 2026-09-13 with ``flagged_lenders(open_source("fixture")
    .frame(), min_count=100)``). An output with nothing to leak cannot
    demonstrate that nothing leaked.
  * they check the specific identities that the audit computed on this very
    run, not a hard-coded list that could drift out of date and silently
    stop matching.

Proven red on 2026-09-13 by making the CLI print the flagged LEI list instead
of its length.
"""

from __future__ import annotations

import re

import pytest
from click.testing import CliRunner

from hmda.cli import main
from hmda.clean.source import open_source
from hmda.fairness.air import flagged_lenders

#: The min-count the CLI defaults to; the tests invoke the default path.
DEFAULT_MIN_COUNT = 100

#: An LEI is 20 uppercase alphanumerics. Matching the SHAPE, not just the known
#: values, catches a leak of a lender this fixture does not happen to flag.
_LEI_SHAPE_RE = re.compile(r"\b[A-Z0-9]{20}\b")


@pytest.fixture(scope="module")
def flagged():
    """The LEIs the audit flags on the fixture. Must be non-empty or the tests prove nothing."""
    leis = flagged_lenders(open_source("fixture").frame(), min_count=DEFAULT_MIN_COUNT)
    assert leis, (
        "the fixture flagged no lender, so a 'no lender is named' assertion would "
        "pass vacuously. Fix the fixture or the min-count before trusting C-36."
    )
    return leis


def _run(*args) -> str:
    result = CliRunner().invoke(main, list(args))
    assert result.exit_code == 0, (
        f"hmda {' '.join(args)} exited {result.exit_code}\n"
        f"{result.output}\n{result.exception!r}"
    )
    return result.output


def test_air_output_names_no_flagged_lender(flagged):
    """C-36: not one flagged LEI appears in the user-facing --air output."""
    output = _run("audit", "--air", "--source", "fixture")
    leaked = [lei for lei in flagged if lei in output]
    assert not leaked, (
        f"hmda audit --air disclosed {len(leaked)} flagged lender identifier(s): "
        f"{leaked}. C-36 permits aggregate counts only.\n" + output
    )


def test_air_output_contains_no_lei_shaped_token_at_all(flagged):
    """The stronger form: no 20-character LEI-shaped token reaches the user.

    The per-identity check above only catches the lenders this fixture flags.
    This catches a leak of any lender, flagged or not.
    """
    output = _run("audit", "--air", "--source", "fixture")
    found = _LEI_SHAPE_RE.findall(output)
    assert not found, (
        f"hmda audit --air printed LEI-shaped token(s) {found}.\n" + output
    )


def test_air_output_reports_flagged_lenders_only_as_a_count(flagged):
    """The control is disclosure-limiting, not information-destroying.

    C-36 withholds identity; it does not withhold the finding. The aggregate
    count must still be reported, and must agree with the computation, or a
    future "fix" could satisfy the two tests above by deleting the line.
    """
    output = _run("audit", "--air", "--source", "fixture")
    match = re.search(r"Lenders flagged for review:\s*(\d+)", output)
    assert match is not None, (
        "hmda audit --air no longer reports an aggregate flagged-lender count. "
        "C-36 withholds identity, not the finding.\n" + output
    )
    assert int(match.group(1)) == len(flagged), (
        f"the CLI reported {match.group(1)} flagged lenders; the audit computed "
        f"{len(flagged)}."
    )


def test_controlled_output_names_no_lender_either(flagged):
    """The controlled-disparity branch is user-facing output too."""
    output = _run("audit", "--controlled", "--source", "fixture")
    leaked = [lei for lei in flagged if lei in output]
    assert not leaked, (
        f"hmda audit --controlled disclosed lender identifier(s): {leaked}\n" + output
    )
    assert not _LEI_SHAPE_RE.findall(output), output
