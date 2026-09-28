"""Tests for bench/compare_pandas.py.

The benchmark's only real claim is "these two paths computed the same table".
Everything here exists to keep that claim honest: that the comparison is
exact and order-sensitive, that a scale point is a well-defined set of rows,
and that the two implementations agree end to end on the committed fixture.

These tests never touch ``data/parquet/``. That directory is gitignored and
36.7M rows; the fixture is 50,000 rows and committed, so the end-to-end
identity check runs in CI as well as here.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCH = _REPO_ROOT / "bench"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import compare_pandas as cp  # noqa: E402

FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "hmda_50k.parquet"


# --------------------------------------------------------------------------
# The comparison itself
# --------------------------------------------------------------------------


def test_tables_match_is_true_for_equal_tables():
    a = [("White", 10, 2, 0.2), ("Asian", 5, 1, 0.2)]
    assert cp.tables_match(a, list(a)) is True


def test_tables_match_is_false_on_a_single_differing_count():
    a = [("White", 10, 2, 0.2)]
    b = [("White", 10, 3, 0.3)]
    assert cp.tables_match(a, b) is False


def test_tables_match_is_order_sensitive():
    """Right rows, wrong order, is not a reproduction of the query.

    sql/rates_by_group.sql ends `order by group_value`, so order is part of
    the result table.
    """
    a = [("Asian", 5, 1, 0.2), ("White", 10, 2, 0.2)]
    b = [("White", 10, 2, 0.2), ("Asian", 5, 1, 0.2)]
    assert cp.tables_match(a, b) is False


def test_tables_match_is_false_when_one_table_is_short():
    a = [("Asian", 5, 1, 0.2), ("White", 10, 2, 0.2)]
    assert cp.tables_match(a, a[:1]) is False


def test_tables_match_rejects_a_float_that_is_merely_close():
    """No tolerance. Both sides compute denials/applications on the same ints."""
    a = [("White", 3, 1, 1 / 3)]
    b = [("White", 3, 1, 0.3333333333333333 + 1e-17)]
    assert cp.tables_match(a, b) == (a[0][3] == b[0][3])


def test_describe_mismatch_names_the_differing_row():
    a = [("White", 10, 2, 0.2)]
    b = [("White", 10, 3, 0.3)]
    lines = cp.describe_mismatch(a, b)
    assert any("row 0" in line for line in lines)


def test_describe_mismatch_reports_a_length_difference():
    a = [("White", 10, 2, 0.2), ("Asian", 5, 1, 0.2)]
    lines = cp.describe_mismatch(a, a[:1])
    assert any("row count differs" in line for line in lines)


def test_describe_mismatch_is_empty_for_equal_tables():
    a = [("White", 10, 2, 0.2)]
    assert cp.describe_mismatch(a, list(a)) == []


# --------------------------------------------------------------------------
# Scale points are whole files, so both paths see exactly the same rows
# --------------------------------------------------------------------------


def test_select_files_takes_the_shortest_prefix_reaching_the_target():
    paths = ["a", "b", "c"]
    counts = [10, 10, 10]
    scope, rows = cp._select_files(paths, counts, 15)
    assert scope == ["a", "b"]
    assert rows == 20


def test_select_files_reports_the_realised_row_count_not_the_target():
    scope, rows = cp._select_files(["a", "b"], [7, 7], 8)
    assert rows == 14 and rows != 8
    assert scope == ["a", "b"]


def test_select_files_none_target_means_every_file():
    paths = ["a", "b", "c"]
    scope, rows = cp._select_files(paths, [1, 2, 3], None)
    assert scope == paths
    assert rows == 6


def test_select_files_returns_everything_when_the_target_is_unreachable():
    paths = ["a", "b"]
    scope, rows = cp._select_files(paths, [1, 1], 10_000)
    assert scope == paths
    assert rows == 2


def test_select_files_exact_boundary_stops_at_that_file():
    scope, rows = cp._select_files(["a", "b", "c"], [10, 10, 10], 20)
    assert scope == ["a", "b"]
    assert rows == 20


# --------------------------------------------------------------------------
# Measurement plumbing
# --------------------------------------------------------------------------


def test_ru_maxrss_unit_matches_the_platform():
    """macOS reports ru_maxrss in bytes; Linux reports kilobytes.

    Getting this wrong misstates M4 by a factor of 1024 in either direction.
    """
    assert cp.RU_MAXRSS_UNIT == (1 if sys.platform == "darwin" else 1024)


def test_memory_cap_is_below_physical_memory():
    """A cap above RAM is not a cap; the machine would swap first."""
    import subprocess

    if sys.platform != "darwin":
        pytest.skip("hw.memsize is a darwin sysctl")
    out = subprocess.run(
        ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False
    ).stdout.strip()
    if not out.isdigit():
        pytest.skip("hw.memsize unavailable")
    assert cp.MEM_CAP_BYTES < int(out)


def test_rss_bytes_of_a_dead_pid_is_zero():
    assert cp._rss_bytes(999_999_999) == 0


def test_rss_bytes_of_this_process_is_positive():
    import os

    assert cp._rss_bytes(os.getpid()) > 0


def test_bench_view_quotes_are_escaped():
    """The file list is spliced into CREATE VIEW; quotes must not break out."""
    captured: list[str] = []

    class FakeCon:
        def execute(self, sql):
            captured.append(sql)

    cp.BenchView(["/tmp/o'brien.parquet"]).register_into(FakeCon())
    assert "'/tmp/o''brien.parquet'" in captured[0]


def test_bench_view_registers_under_the_name_frame():
    """Every sql/*.sql file queries `from frame`."""
    captured: list[str] = []

    class FakeCon:
        def execute(self, sql):
            captured.append(sql)

    cp.BenchView(["a.parquet", "b.parquet"]).register_into(FakeCon())
    assert "CREATE OR REPLACE VIEW frame AS" in captured[0]
    assert "'a.parquet', 'b.parquet'" in captured[0]


_BENCH_PATHS = [
    _REPO_ROOT / "bench" / "compare_pandas.py",
    _REPO_ROOT / "src" / "hmda" / "fairness" / "rates.py",
    _REPO_ROOT / "src" / "hmda" / "clean" / "source.py",
]


def _connects_with_an_argument(source: str) -> list[int]:
    """Line numbers of real ``duckdb.connect(<something>)`` CALLS in ``source``.

    Parsed, not grepped. Two text-matching versions of this check were written
    first and both were defeated by the same thing: the file's own prose. A
    ``grep -c "hmda.duckdb"`` matched the sentences saying the file is never
    opened, and a regex for ``duckdb.connect([^)]`` matched the worked example
    in the docstring that demonstrates what a violation looks like. The AST
    sees only executable code, so documentation about the rule can never trip
    the rule.
    """
    tree = ast.parse(source)
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "connect"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "duckdb"
        and (node.args or node.keywords)
    ]


def test_no_duckdb_connect_takes_a_database_path():
    """Neither benchmark path may open data/hmda.duckdb.

    Both paths must read the same parquet bytes, or "TABLES IDENTICAL" is
    comparing two different datasets. Opening the 3 GB database would also
    take a file lock that other processes on this machine contend on -- it
    lives in a cloud-synced folder that fileproviderd holds open.

    ``duckdb.connect()`` with no argument is in-memory; any argument is a
    database path.
    """
    offenders = [
        f"{p}:{line}"
        for p in _BENCH_PATHS
        for line in _connects_with_an_argument(p.read_text())
    ]
    assert offenders == [], "\n".join(offenders)


def test_the_connect_check_flags_a_real_database_open():
    """The check must be able to go red, or it proves nothing."""
    assert _connects_with_an_argument('duckdb.connect("data/hmda.duckdb")') == [1]
    assert _connects_with_an_argument("duckdb.connect(path)") == [1]
    assert _connects_with_an_argument("duckdb.connect(database=p)") == [1]


def test_the_connect_check_passes_an_in_memory_connection():
    assert _connects_with_an_argument("con = duckdb.connect()") == []


def test_the_connect_check_is_not_fooled_by_prose():
    """The trap that broke both earlier text-matching versions of this check."""
    source = '"""Never opens data/hmda.duckdb, unlike duckdb.connect(\'x.duckdb\')."""'
    assert _connects_with_an_argument(source) == []


def test_perturb_pandas_is_off_in_committed_code():
    """The deliberate-failure switch must never be left on."""
    assert cp.PERTURB_PANDAS is False


# --------------------------------------------------------------------------
# End to end, on the committed fixture
# --------------------------------------------------------------------------


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture parquet not present")
def test_duckdb_and_pandas_paths_agree_on_the_fixture():
    """The core claim: the two implementations compute the same table."""
    paths = [str(FIXTURE)]
    duck = cp.duckdb_table(paths)
    pand = cp.pandas_table(paths)
    assert duck, "the fixture produced no rows; the comparison would be vacuous"
    assert cp.tables_match(duck, pand), "\n".join(cp.describe_mismatch(duck, pand))


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture parquet not present")
def test_the_fixture_agreement_test_can_fail():
    """Prove the end-to-end check is not vacuous by perturbing pandas.

    A comparison that cannot come out False proves nothing about the one that
    comes out True.
    """
    paths = [str(FIXTURE)]
    duck = cp.duckdb_table(paths)
    cp.PERTURB_PANDAS = True
    try:
        perturbed = cp.pandas_table(paths)
    finally:
        cp.PERTURB_PANDAS = False
    assert not cp.tables_match(duck, perturbed)
    assert cp.PERTURB_PANDAS is False


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture parquet not present")
def test_child_actually_receives_the_perturbation():
    """The perturbation must survive the hop into the worker.

    Regression test. ``_child`` sets its own copy of PERTURB_PANDAS from the
    argument it is passed, and ``main`` used to compute that argument from the
    command-line flag alone. Editing PERTURB_PANDAS to True in the source was
    therefore overwritten with False inside the child, and the deliberately
    broken run reported TABLES IDENTICAL: True -- a check that was supposed
    to catch the break but could not fail. Measured 2026-09-13.
    """

    class Q:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    duck = cp.duckdb_table([str(FIXTURE)])

    plain, broken = Q(), Q()
    try:
        cp._child("pandas", [str(FIXTURE)], False, plain)
        cp._child("pandas", [str(FIXTURE)], True, broken)
    finally:
        cp.PERTURB_PANDAS = False

    assert plain.items[0]["kind"] == "ok", plain.items[0]["detail"]
    assert broken.items[0]["kind"] == "ok", broken.items[0]["detail"]
    assert cp.tables_match(duck, [tuple(r) for r in plain.items[0]["table"]])
    assert not cp.tables_match(duck, [tuple(r) for r in broken.items[0]["table"]])


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture parquet not present")
def test_pandas_path_excludes_purchased_loans():
    """action_taken = 6 is a purchased loan, not an application, so it is excluded and no group counts them."""
    import pandas as pd

    df = pd.read_parquet(FIXTURE)
    purchased = (pd.to_numeric(df["action_taken"], errors="coerce") == 6).sum()
    total = len(df)
    counted = sum(row[1] for row in cp.pandas_table([str(FIXTURE)]))
    assert purchased > 0, "fixture has no purchased loans; the test is vacuous"
    assert counted == total - purchased


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture parquet not present")
def test_pandas_path_rows_are_sorted_by_group_value():
    rows = cp.pandas_table([str(FIXTURE)])
    values = [r[0] for r in rows]
    assert values == sorted(values)


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture parquet not present")
def test_denial_rate_equals_denials_over_applications():
    for group_value, applications, denials, rate in cp.pandas_table([str(FIXTURE)]):
        assert rate == (denials / applications if applications else 0.0), group_value
