"""C-35: every disparity output the CLI prints carries the interpretation limit.

docs/CONTROLS.md C-35: "The CLI states on every disparity output that a flag
is a statistical disparity warranting review and NEVER a finding of
discrimination." Implemented at src/hmda/cli.py:266.

These tests drive the REAL command path through click's CliRunner, not a
reimplementation of it: a test that builds its own string and inspects that
string proves nothing about what a user sees.

The assertion is SEMANTIC, not a literal sentence match. An exact-string test
would go red on a harmless rewording and would tempt a future editor to delete
the control rather than fix the test. What is asserted is the pair of ideas the
control actually requires, in the same output:

  1. the result is named as a STATISTICAL disparity (or equivalent), and
  2. it is explicitly disclaimed as NOT a finding of discrimination.

Deleting the disclaimer line, or softening it to only half of the pair, turns
these red. Proven red on 2026-09-13 by commenting out the click.echo at
src/hmda/cli.py:266.
"""

from __future__ import annotations

import re

from click.testing import CliRunner

from hmda.cli import main

#: "statistical disparity", "statistical disparities", "disparity ... statistical".
_STATISTICAL_DISPARITY_RE = re.compile(r"statistical\s+disparit", re.IGNORECASE)

#: "NEVER a finding of discrimination", "not a finding of discrimination",
#: "is not evidence of discrimination".
_NOT_A_FINDING_RE = re.compile(
    r"(never|not)\s+(a\s+finding|evidence|proof|a\s+determination)\s+of\s+discrimination",
    re.IGNORECASE,
)


def _run(*args) -> str:
    """Invoke the real CLI and return its combined output, failing loudly on a crash."""
    result = CliRunner().invoke(main, list(args))
    assert result.exit_code == 0, (
        f"hmda {' '.join(args)} exited {result.exit_code}\n"
        f"{result.output}\n{result.exception!r}"
    )
    return result.output


def test_air_output_names_the_result_a_statistical_disparity():
    """C-35, half one: the --air output calls a flag a statistical disparity."""
    output = _run("audit", "--air", "--source", "fixture")
    assert _STATISTICAL_DISPARITY_RE.search(output), (
        "hmda audit --air printed no 'statistical disparity' framing. C-35 requires "
        "the permitted interpretation to be stated WITH the output.\n" + output
    )


def test_air_output_disclaims_a_finding_of_discrimination():
    """C-35, half two: the --air output says a flag is not a finding of discrimination."""
    output = _run("audit", "--air", "--source", "fixture")
    assert _NOT_A_FINDING_RE.search(output), (
        "hmda audit --air printed no 'not a finding of discrimination' disclaimer. "
        "This is the sentence the project's credibility rests on.\n" + output
    )


def test_the_disclaimer_is_printed_above_the_disparity_numbers():
    """The limit has to reach a reader who stops reading at the table.

    A disclaimer printed after 40 rows of ratios is not "stated with the
    output" in any useful sense. This asserts ordering against the real
    rendered text: the disclaimer precedes the first ratio row.
    """
    output = _run("audit", "--air", "--source", "fixture")
    disclaimer = _NOT_A_FINDING_RE.search(output)
    assert disclaimer is not None, output
    # The table header line the ratio rows follow, printed by src/hmda/cli.py.
    header = output.index("ratio")
    assert disclaimer.start() < header, (
        "the C-35 disclaimer appears AFTER the disparity table began.\n" + output
    )


def test_controlled_output_also_disclaims_discrimination():
    """--controlled is a disparity output too, and carries its own limit line.

    C-35 says "every disparity output", not "the --air output". The controlled
    branch prints a different sentence (src/hmda/cli.py, controlled branch);
    both must survive.
    """
    output = _run("audit", "--controlled", "--source", "fixture")
    assert _NOT_A_FINDING_RE.search(output), (
        "hmda audit --controlled printed no discrimination disclaimer.\n" + output
    )


def test_no_disparity_output_asserts_discrimination_outright():
    """The inverse guard: the CLI must not state discrimination as a finding.

    A disclaimer plus a contradicting sentence is worse than no disclaimer, so
    the negative case is checked as well as the positive one.
    """
    for branch in ("--air", "--controlled"):
        output = _run("audit", branch, "--source", "fixture")
        for banned in (
            "finding of discrimination",
            "evidence of discrimination",
            "proof of discrimination",
        ):
            for match in re.finditer(re.escape(banned), output, re.IGNORECASE):
                window = output[max(0, match.start() - 40) : match.start()]
                assert _NOT_A_FINDING_RE.search(
                    output[max(0, match.start() - 40) : match.end()]
                ), (
                    f"hmda audit {branch} says {banned!r} without a negation in front "
                    f"of it: ...{window}{banned}...\n" + output
                )
