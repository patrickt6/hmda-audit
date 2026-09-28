"""Control-to-standard mapping: control id, assertion, implementing file, test, SR 11-7/OSFI E-23 clause.

``hmda verify --governance`` exits non-zero if any control names a test
that does not exist. A control with no test is listed as NOT IMPLEMENTED,
never omitted.

``docs/CONTROLS.md`` is the source of truth. This module only parses it and
checks it against the repository, so the table a reviewer reads and the
table this check runs against can never drift apart.

Sourcing note (binding, see the repo provenance contract): neither
SR 11-7 nor OSFI E-23 was opened while writing this mapping. Every framework
citation in ``docs/CONTROLS.md`` therefore names the framework and the
principle IN WORDS and carries no clause or section number. See the
"Sources and their limits" section of ``docs/CONTROLS.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: The literal value a control uses when no test proves it yet. Such a
#: control is listed, never omitted, and never counted as implemented.
NOT_IMPLEMENTED = "NOT IMPLEMENTED"

#: Default location of the control table.
CONTROLS_DOC_PATH = Path("docs/CONTROLS.md")

#: Repository root: this file is ``<root>/src/hmda/governance/controls.py``.
REPO_ROOT = Path(__file__).resolve().parents[3]

_COLUMNS = 6
_SEPARATOR_RE = re.compile(r"^\|[\s:|-]+\|$")
_CONTROL_ID_RE = re.compile(r"^C-\d{2}$")
_IMPL_RE = re.compile(r"^(?P<path>[^:`]+):(?P<line>\d+)$")


@dataclass(frozen=True)
class Control:
    """One governance control row.

    control_id: short stable identifier (e.g. "C-01").
    assertion: what the control asserts, in one sentence.
    implementing_file: the file path that implements it, as ``path:line``.
    test: the pytest node id that proves it, or the literal string
        "NOT IMPLEMENTED" if none exists yet.
    sr_11_7_clause: the SR 11-7 clause this maps to, with the source
        document/URL this was actually read from. The rule: read the two
        standards before writing the mapping, and never cite a clause that
        was not opened.
    osfi_e23_clause: the OSFI E-23 clause this maps to, same sourcing rule.
    """

    control_id: str
    assertion: str
    implementing_file: str
    test: str
    sr_11_7_clause: str
    osfi_e23_clause: str

    @property
    def implemented(self) -> bool:
        """True iff this control names a test rather than ``NOT IMPLEMENTED``."""
        return self.test != NOT_IMPLEMENTED


def _strip_markdown(cell: str) -> str:
    """Remove backticks and surrounding whitespace from one table cell."""
    return cell.strip().strip("`").strip()


def load_controls(controls_doc_path=CONTROLS_DOC_PATH) -> list[Control]:
    """Parse the control table out of ``docs/CONTROLS.md`` into a list of :class:`Control`.

    The count this function returns must equal the row count in
    ``docs/CONTROLS.md``.

    Only rows whose first cell matches ``C-\\d\\d`` are controls; every other
    markdown table in the document is ignored, so the file can carry prose
    and other tables without corrupting the count. A row with the right id
    shape but the wrong number of cells raises, rather than being skipped:
    a silently dropped control is a silently lowered M10.
    """
    path = Path(controls_doc_path)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    text = path.read_text(encoding="utf-8")

    controls: list[Control] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line.startswith("|") or _SEPARATOR_RE.match(line):
            continue
        cells = [_strip_markdown(c) for c in line.strip("|").split("|")]
        if not cells or not _CONTROL_ID_RE.match(cells[0]):
            continue
        if len(cells) != _COLUMNS:
            raise ValueError(
                f"{path}:{lineno}: control row {cells[0]} has {len(cells)} cells, "
                f"expected {_COLUMNS}"
            )
        if cells[0] in seen:
            raise ValueError(f"{path}:{lineno}: duplicate control id {cells[0]}")
        seen.add(cells[0])
        controls.append(Control(*cells))
    return controls


def test_exists(node_id: str, root: Path = REPO_ROOT) -> bool:
    """True iff ``node_id`` ("tests/test_x.py::test_name") names a real test.

    Checks two things, both of which have to hold: the test FILE exists, and
    a ``def <test_name>(`` appears in it. Checking only the file would let a
    renamed test pass, which is the exact failure this check exists to catch.
    """
    if "::" not in node_id:
        return False
    rel, _, name = node_id.partition("::")
    name = name.split("[", 1)[0]  # drop any parametrisation suffix
    test_file = root / rel
    if not test_file.is_file():
        return False
    source = test_file.read_text(encoding="utf-8")
    return re.search(rf"^\s*def {re.escape(name)}\s*\(", source, re.MULTILINE) is not None


def implementation_exists(locator: str, root: Path = REPO_ROOT) -> bool:
    """True iff ``locator`` ("path/to/file.py:123") points at a real file and a real line."""
    match = _IMPL_RE.match(locator.strip())
    if match is None:
        return False
    target = root / match.group("path")
    if not target.is_file():
        return False
    wanted = int(match.group("line"))
    if wanted < 1:
        return False
    with target.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle) >= wanted


def missing_tests(controls: list[Control]) -> list[Control]:
    """Controls that name a test which does not exist. ``NOT IMPLEMENTED`` is not missing."""
    return [c for c in controls if c.implemented and not test_exists(c.test)]


def missing_implementations(controls: list[Control]) -> list[Control]:
    """Controls whose ``implementing_file`` locator does not resolve to a real file:line."""
    return [c for c in controls if not implementation_exists(c.implementing_file)]


def verify_controls(controls: list[Control]) -> bool:
    """Return True iff every control whose ``test`` is not "NOT IMPLEMENTED" names a test that actually exists.

    This is what ``hmda verify --governance`` calls; it must exit non-zero
    when this returns False.
    """
    return not missing_tests(controls)


def main(controls_doc_path=CONTROLS_DOC_PATH) -> int:
    """Body of ``hmda verify --governance``. Prints M10 and returns a process exit code.

    Exit 0 only when every control row resolves: its test exists (or is
    explicitly ``NOT IMPLEMENTED``) and its ``implementing_file`` locator
    points at a real line of a real file. Both failures print the offending
    control id, so a red result names what to fix.
    """
    controls = load_controls(controls_doc_path)
    bad_tests = missing_tests(controls)
    bad_impls = missing_implementations(controls)
    implemented = [c for c in controls if c.implemented]
    unbacked = [c for c in controls if not c.implemented]

    for control in bad_tests:
        print(
            f"FAIL {control.control_id}: names test {control.test!r}, which does not exist"
        )
    for control in bad_impls:
        print(
            f"FAIL {control.control_id}: implementing_file {control.implementing_file!r} "
            f"does not resolve to a real file:line"
        )

    print(f"governance controls documented: {len(controls)}")
    print(f"  backed by a test: {len(implemented)}")
    print(f"  {NOT_IMPLEMENTED}: {len(unbacked)}")
    if unbacked:
        print("  unbacked: " + ", ".join(c.control_id for c in unbacked))

    if bad_tests or bad_impls:
        print("hmda verify --governance: FAIL")
        return 1
    print("hmda verify --governance: PASS")
    return 0
