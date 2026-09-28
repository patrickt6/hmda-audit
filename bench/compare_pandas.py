"""M3/M4 baseline: run the same audit through DuckDB and through pandas, compare.

Prints both wall times, both peak-memory
figures (via ``resource.getrusage``), the machine spec, and
``TABLES IDENTICAL: True``.

WHAT IS BEING COMPARED
----------------------
One audit computation: the national denial rate by ``derived_race``, over the
parquet files under ``data/parquet/``.

* The **DuckDB path** is the production path. It calls
  :func:`hmda.fairness.rates.denial_rate_by_group`, which loads
  ``sql/rates_by_group.sql`` and runs it against a registered ``frame``. This
  benchmark does not reimplement it and does not copy the SQL.
* The **pandas path** is a straightforward conventional implementation:
  ``pd.read_parquet`` every file, ``pd.concat`` them into one DataFrame,
  filter, ``groupby``, aggregate. That is what the engineering claim is
  measured against, so it is deliberately not optimised: no column
  projection, no chunked per-file aggregation. Both of those would be
  reasonable engineering and both would be a different claim.

**Read the ratio with this caveat.** The audit reads 2 of the 99 columns.
DuckDB's parquet scan projects to those two; the pandas baseline materializes
all 99, because that is what ``pd.read_parquet(path)`` does. A large part of
the gap is therefore column projection rather than aggregation speed. That is
a genuine property of the engine and not an artefact of the harness: not
having to think about projection is the point, but a reader who assumes the
gap is "DuckDB groups by faster" has misread it. The gap on a query that
needed all 99 columns would be much smaller, and this benchmark does not
measure that case.

WHAT THIS BENCHMARK OPENS, AND WHAT IT DOES NOT
-----------------------------------------------
Both paths read ``data/parquet/*.parquet`` and nothing else. **Neither path
opens ``data/hmda.duckdb``.** The DuckDB side uses an in-memory connection
(``duckdb.connect()`` with no path, at ``hmda/fairness/rates.py:73`` and
``:97``) and attaches the parquet files as a view; the pandas side calls
``pd.read_parquet`` on the same files. So the two engines read identical
bytes from an identical source, which is what "TABLES IDENTICAL" has to mean
to be worth anything, and no file lock is taken that another process could
contend on.

The check for that is
``tests/test_bench.py::test_no_duckdb_connect_takes_a_database_path``. It
parses this file and the two ``hmda`` modules and looks for a
``duckdb.connect`` CALL that is given any argument, which is the only way
this code could open a database file. It is an AST check rather than a grep
because two text-matching versions were written first and both were defeated
by this very docstring: ``grep -c "hmda.duckdb"`` matched the sentences above
promising the file is never opened, and a regex for ``duckdb.connect([^)]``
matched the line that demonstrated what a violation looks like. Prose about a
rule must not be able to trip the rule. Three sibling tests prove the check
still goes red on a genuine database open.

For a quick manual look, every connection on this path::

    $ grep -rn "duckdb\\.connect(" src/hmda/fairness/rates.py \\
          src/hmda/clean/source.py
    src/hmda/clean/source.py:158:        con = duckdb.connect()
    src/hmda/clean/source.py:236:    con = duckdb.connect()
    src/hmda/fairness/rates.py:73:    con = duckdb.connect()
    src/hmda/fairness/rates.py:97:    con = duckdb.connect()

This matters because ``data/hmda.duckdb`` is a contended resource on this
machine: the repository lives in a cloud-synced folder and ``fileproviderd``
holds a write descriptor on that 3 GB file. A benchmark that took a lock on
it would be measuring sync scheduling.

The pandas path reproduces SQL's three-valued logic on purpose (a NULL
``action_taken`` is dropped by ``where action_taken != 6``, whereas pandas'
``!=`` would keep it). That is fidelity to the thing being compared, not a
thumb on the scale. Measured 2026-09-13, ``action_taken`` has no NULLs in this
dataset, so the guard changes nothing here; it is there so the two paths do
not diverge silently if the data changes.

MEASURING PEAK MEMORY
---------------------
``resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`` is a per-process
high-water mark that never goes down, so running both paths in one process
would report the larger of the two twice. Each path therefore runs in its own
spawned child process and reports its own ``ru_maxrss``.

**On macOS ``ru_maxrss`` is BYTES, not kilobytes as on Linux.** Verified on
this machine rather than assumed: allocating and touching a 629,145,600-byte
bytearray moved ``ru_maxrss`` by 629,719,040.
:data:`RU_MAXRSS_UNIT` encodes that,
and is selected from ``sys.platform``.

THE MEMORY CAP, AND WHY IT IS A WATCHDOG
----------------------------------------
A runaway pandas allocation on a 36 GiB machine swaps for a long time instead
of failing. The obvious guard, ``resource.setrlimit(resource.RLIMIT_AS, ...)``,
**is not usable here**: on this machine (macOS 26.5.2, arm64) both
``RLIMIT_AS`` and ``RLIMIT_DATA`` are reported as ``RLIM_INFINITY`` and any
``setrlimit`` on them raises ``ValueError: current limit exceeds maximum
limit``, even when the hard limit is left at infinity. Measured 2026-09-13.

So the cap is enforced from the parent instead: a watchdog polls the child's
resident size with ``ps -o rss=`` and ``SIGKILL``s it above
:data:`MEM_CAP_BYTES`, or above :data:`WALL_CAP_S` seconds. The cap is part of
the measurement's meaning: "died at a 16 GiB cap" is a statement about the cap
as much as about pandas, and it is printed with every result.
"""

from __future__ import annotations

import glob
import multiprocessing as mp
import os
import platform
import queue as _queue
import resource
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BENCH_DIR.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

PARQUET_DIR = _REPO_ROOT / "data" / "parquet"

# The one audit computation both paths must produce identically.
GROUP_COLUMN = "derived_race"

# ru_maxrss units. Linux reports kilobytes; macOS reports bytes. Verified on
# this machine for darwin (see module docstring).
RU_MAXRSS_UNIT = 1 if sys.platform == "darwin" else 1024

# Parent-enforced caps on any single child run. See module docstring for why
# this is a watchdog and not setrlimit.
MEM_CAP_BYTES = 16 * 1024**3
WALL_CAP_S = 1200.0
# Polled often, because a pandas.concat can add several GiB between two polls
# and the watchdog is the only thing standing between this benchmark and a
# machine that swaps for an hour.
POLL_INTERVAL_S = 0.2

# How often the parent prints that it is alive. See run_path.
HEARTBEAT_S = 15.0

# Row counts to scale up through. The last entry, None, means every file.
# Scale points land on whole-file boundaries (see _select_files), so the
# realised row count is at or just above the target and is always printed.
SCALE_TARGETS: list[int | None] = [1_000_000, 5_000_000, 10_000_000, 20_000_000, None]

# Set by the deliberate-failure self-check only. Never true in committed code.
PERTURB_PANDAS = False


# --------------------------------------------------------------------------
# Scope selection: which parquet files make up a scale point
# --------------------------------------------------------------------------


def _file_row_counts(paths: list[str]) -> list[int]:
    """Row count per parquet file, read from footer metadata, not from data.

    Progress is reported because this step can stall. It normally takes about
    a second for all 168 files; on 2026-09-13, with ``fileproviderd`` at 120%
    CPU over this cloud-synced folder, the same loop did not finish in 120
    seconds. Without a progress line that stall is indistinguishable from a
    hang -- it is a blocking filesystem read, so CPU is 0% and memory is flat
    -- and a run was killed on exactly that appearance.
    """
    import pyarrow.parquet as pq

    counts: list[int] = []
    started = time.perf_counter()
    next_beat = started + HEARTBEAT_S
    for i, p in enumerate(paths, 1):
        counts.append(pq.ParquetFile(p).metadata.num_rows)
        now = time.perf_counter()
        if now >= next_beat:
            print(
                f"    [alive] reading parquet footers: {i}/{len(paths)} "
                f"after {now - started:.0f}s (blocking filesystem reads; "
                f"0% CPU here is normal)",
                flush=True,
            )
            next_beat = now + HEARTBEAT_S
    return counts


def _select_files(
    paths: list[str], counts: list[int], target: int | None
) -> tuple[list[str], int]:
    """The shortest prefix of ``paths`` holding at least ``target`` rows.

    Whole files only. Truncating the last file would need a row order that
    parquet does not promise and that ``LIMIT`` without ``ORDER BY`` does not
    give, so the two paths could silently disagree about which rows are in
    scope. A scale point is therefore "the nearest whole-file row count at or
    above the target", and the realised count is returned so it can be printed
    instead of the target.

    ``target=None`` means every file.
    """
    if target is None:
        return list(paths), sum(counts)
    running = 0
    for i, c in enumerate(counts):
        running += c
        if running >= target:
            return list(paths[: i + 1]), running
    return list(paths), running


# --------------------------------------------------------------------------
# The two implementations
# --------------------------------------------------------------------------


class BenchView:
    """A DuckDB view over an explicit list of parquet files.

    Duck-types :class:`hmda.clean.source.ParquetView`: it exposes
    ``register_into`` so the production :func:`hmda.clean.source.register_frame`
    attaches it under the name ``frame`` without materializing anything. A list
    is needed rather than the production glob because a scale point is a subset
    of the files.
    """

    def __init__(self, paths: list[str]) -> None:
        self.paths = list(paths)

    def register_into(self, con) -> None:
        # DuckDB cannot prepare a parameter inside CREATE VIEW (documented in
        # hmda.clean.source.ParquetView.register_into), so the list is spliced.
        # These paths come from glob() over a module-level directory, never
        # from caller input, and single quotes are still escaped.
        literals = ", ".join("'" + p.replace("'", "''") + "'" for p in self.paths)
        con.execute(
            "CREATE OR REPLACE VIEW frame AS "
            f"SELECT * FROM read_parquet([{literals}])"
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"BenchView({len(self.paths)} files)"


def duckdb_table(paths: list[str]) -> list[tuple]:
    """Run the audit through the production DuckDB path."""
    from hmda.fairness.rates import denial_rate_by_group

    rows = denial_rate_by_group(BenchView(paths), GROUP_COLUMN)
    return [
        (r.group_value, int(r.applications), int(r.denials), float(r.denial_rate))
        for r in rows
    ]


def pandas_table(paths: list[str]) -> list[tuple]:
    """Run the same audit through a straightforward pandas implementation.

    Deliberately conventional: read every file whole, concatenate, then
    aggregate in memory. See the module docstring for why it is not optimised.
    """
    import pandas as pd

    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    del frames

    # action_taken is stored as VARCHAR in these parquet files; the SQL
    # compares it against the integers 6 and 3, so DuckDB casts. Cast here too.
    action = pd.to_numeric(df["action_taken"], errors="coerce")
    group = df[GROUP_COLUMN]

    # SQL's `where action_taken != 6` drops NULLs, because NULL != 6 is NULL,
    # not TRUE. pandas' `!=` keeps them. Reproduce SQL.
    in_scope = action.notna() & (action != 6)

    # --- PERTURBATION POINT (deliberate-failure self-check) ----------------
    # Flipping the denial code here makes the pandas table disagree with the
    # DuckDB table, which must make this script exit non-zero.
    denial_code = 4 if PERTURB_PANDAS else 3
    # -----------------------------------------------------------------------

    scoped = pd.DataFrame(
        {
            "group_value": group[in_scope],
            "is_denial": (action[in_scope] == denial_code).astype("int64"),
        }
    )
    del df, action, group, in_scope

    agg = scoped.groupby("group_value", dropna=False)["is_denial"].agg(
        ["size", "sum"]
    )

    out: list[tuple] = []
    # sql/rates_by_group.sql ends `order by group_value`; DuckDB sorts ASC with
    # NULLS LAST by default, so sort the same way.
    index = list(agg.index)
    non_null = sorted(v for v in index if v is not None and v == v)
    nulls = [v for v in index if v is None or v != v]
    for value in non_null + nulls:
        applications = int(agg.loc[value, "size"])
        denials = int(agg.loc[value, "sum"])
        # hmda.fairness.rates.denial_rate_by_group computes the rate in Python
        # as denials / applications, guarding a zero denominator. Same here.
        rate = denials / applications if applications else 0.0
        out.append((value, applications, denials, float(rate)))
    return out


# --------------------------------------------------------------------------
# Running one path in its own process, under the watchdog
# --------------------------------------------------------------------------


@dataclass
class Outcome:
    """What happened to one (path, scale) run."""

    kind: str  # "ok" | "mem_killed" | "timeout" | "error" | "signal"
    wall_s: float
    peak_bytes: int  # child ru_maxrss on success; watchdog high-water on a kill
    detail: str = ""
    table: list[tuple] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.kind == "ok"


def _child(kind: str, paths: list[str], perturb: bool, out_q) -> None:
    """Child-process entry point. Runs one path and reports its own rusage."""
    global PERTURB_PANDAS
    PERTURB_PANDAS = perturb
    started = time.perf_counter()
    try:
        table = duckdb_table(paths) if kind == "duckdb" else pandas_table(paths)
    except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
        out_q.put(
            {
                "kind": "error",
                "wall_s": time.perf_counter() - started,
                "peak_bytes": _self_peak_bytes(),
                "detail": f"{type(exc).__name__}: {exc}",
                "table": [],
            }
        )
        return
    out_q.put(
        {
            "kind": "ok",
            "wall_s": time.perf_counter() - started,
            "peak_bytes": _self_peak_bytes(),
            "detail": "",
            "table": table,
        }
    )


def _self_peak_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * RU_MAXRSS_UNIT


def _rss_bytes(pid: int) -> int:
    """Resident size of ``pid`` in bytes, via ``ps``. 0 if the pid is gone.

    macOS ``ps -o rss=`` reports kibibytes.
    """
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        return 0
    return int(out) * 1024 if out.isdigit() else 0


def run_path(
    kind: str, paths: list[str], perturb: bool = False, rows: int = 0
) -> Outcome:
    """Run one path in a spawned child, under the memory and wall watchdog.

    A heartbeat is printed while the child works. The parent does nothing but
    poll, so its own CPU use is ~0% and its resident size sits flat near
    100 MB for the whole run. On 2026-09-13 that was mistaken for a hung
    process and the benchmark was killed nine minutes in. The parent being
    idle IS the design -- all the work is in the child -- so it now says so
    out loud, and prints the child's resident size, which is the number that
    actually moves.
    """
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    proc = ctx.Process(target=_child, args=(kind, paths, perturb, out_q))
    started = time.perf_counter()
    proc.start()

    watched_peak = 0
    verdict = ""
    next_beat = started + HEARTBEAT_S
    while proc.is_alive():
        rss = _rss_bytes(proc.pid)
        watched_peak = max(watched_peak, rss)
        elapsed = time.perf_counter() - started
        now = time.perf_counter()
        if now >= next_beat:
            print(
                f"    [alive] {kind} at {rows:,} rows: {elapsed:.0f}s, "
                f"child pid {proc.pid} rss {_gib(rss)} "
                f"(cap {_gib(MEM_CAP_BYTES)}). The parent polls and is idle "
                f"by design.",
                flush=True,
            )
            next_beat = now + HEARTBEAT_S
        if rss > MEM_CAP_BYTES:
            verdict = "mem_killed"
            os.kill(proc.pid, signal.SIGKILL)
            break
        if elapsed > WALL_CAP_S:
            verdict = "timeout"
            os.kill(proc.pid, signal.SIGKILL)
            break
        time.sleep(POLL_INTERVAL_S)

    try:
        payload = out_q.get(timeout=10)
    except _queue.Empty:
        payload = None
    proc.join(timeout=30)
    wall = time.perf_counter() - started

    if verdict == "mem_killed":
        return Outcome(
            kind="mem_killed",
            wall_s=wall,
            peak_bytes=watched_peak,
            detail=(
                f"SIGKILLed by the benchmark watchdog: resident size "
                f"{watched_peak / 1024**3:.2f} GiB exceeded the "
                f"{MEM_CAP_BYTES / 1024**3:.0f} GiB cap"
            ),
        )
    if verdict == "timeout":
        return Outcome(
            kind="timeout",
            wall_s=wall,
            peak_bytes=watched_peak,
            detail=f"SIGKILLed by the watchdog after {WALL_CAP_S:.0f}s",
        )
    if payload is None:
        # No result and no watchdog verdict: the process died on its own.
        # Negative exitcode is a signal number -- on macOS an out-of-memory
        # process is killed with SIGKILL by the kernel, which looks like -9.
        code = proc.exitcode
        return Outcome(
            kind="signal",
            wall_s=wall,
            peak_bytes=watched_peak,
            detail=(
                f"child exited with code {code}"
                + (f" (signal {-code})" if code is not None and code < 0 else "")
                + ", no result returned"
            ),
        )
    return Outcome(
        kind=payload["kind"],
        wall_s=payload["wall_s"],
        peak_bytes=max(int(payload["peak_bytes"]), watched_peak),
        detail=payload["detail"],
        table=[tuple(r) for r in payload["table"]],
    )


# --------------------------------------------------------------------------
# Comparison and reporting
# --------------------------------------------------------------------------


def tables_match(a: list[tuple], b: list[tuple]) -> bool:
    """Exact, order-sensitive equality of two result tables.

    Counts are integers and the rate is the same Python ``int / int`` on both
    sides, so there is nothing to round and no tolerance is allowed. Order is
    part of the table: ``sql/rates_by_group.sql`` ends ``order by
    group_value``, and a path that returned the right rows in the wrong order
    has not reproduced the query.
    """
    return a == b


def describe_mismatch(a: list[tuple], b: list[tuple]) -> list[str]:
    """Human-readable first differences between two tables."""
    lines: list[str] = []
    if len(a) != len(b):
        lines.append(f"  row count differs: duckdb={len(a)} pandas={len(b)}")
    for i, (ra, rb) in enumerate(zip(a, b)):
        if ra != rb:
            lines.append(f"  row {i}: duckdb={ra!r}  pandas={rb!r}")
    return lines


def machine_spec() -> list[str]:
    """The machine, read from the machine, never hardcoded."""

    def sysctl(key: str) -> str:
        try:
            r = subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, check=False
            )
            return r.stdout.strip() or "unavailable"
        except OSError:
            return "unavailable"

    lines = [
        f"  platform            : {platform.platform()}",
        f"  machine             : {platform.machine()}",
    ]
    if sys.platform == "darwin":
        memsize = sysctl("hw.memsize")
        lines += [
            f"  model               : {sysctl('hw.model')}",
            f"  cpu                 : {sysctl('machdep.cpu.brand_string')}",
            f"  logical cpus        : {sysctl('hw.ncpu')}",
            f"  physical memory     : {memsize} bytes"
            + (
                f" ({int(memsize) / 1024**3:.1f} GiB)"
                if memsize.isdigit()
                else ""
            ),
        ]
    else:  # pragma: no cover - this benchmark's numbers were taken on darwin
        lines.append(f"  logical cpus        : {os.cpu_count()}")
    return lines


def version_lines() -> list[str]:
    """Interpreter and library versions actually imported by this run."""
    import duckdb
    import numpy
    import pandas
    import pyarrow

    return [
        f"  executable          : {sys.executable}",
        f"  python              : {sys.version.splitlines()[0]}",
        f"  pandas              : {pandas.__version__}",
        f"  numpy               : {numpy.__version__}",
        f"  duckdb              : {duckdb.__version__}",
        f"  pyarrow             : {pyarrow.__version__}",
    ]


def _gib(n: int) -> str:
    return f"{n / 1024**3:.2f} GiB"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Either route into the perturbation may be used: the --perturb-pandas
    # flag, or editing PERTURB_PANDAS to True in the source. The module
    # constant is OR-ed in rather than ignored because the child process sets
    # its own copy from this value; reading only the flag here silently
    # overwrote a source-level True with False in the child, and the
    # deliberately-broken run came back "TABLES IDENTICAL: True". Found
    # 2026-09-13 by running this deliberate-failure check, which is what it
    # is for.
    perturb = PERTURB_PANDAS or ("--perturb-pandas" in argv)

    # Line-buffer stdout. Redirected to a file, Python block-buffers it, so a
    # run that is working normally shows an EMPTY output file for minutes and
    # looks hung. Every progress line below is useless without this.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):  # pragma: no cover - non-standard stdout
        pass

    paths = sorted(glob.glob(str(PARQUET_DIR / "*.parquet")))
    if not paths:
        print(f"no parquet files under {PARQUET_DIR}", file=sys.stderr)
        return 2
    counts = _file_row_counts(paths)

    print("=" * 74)
    print("M3/M4 -- DuckDB vs pandas, same audit, same rows")
    print("=" * 74)
    print(f"computation         : denial rate by {GROUP_COLUMN}, "
          f"via sql/rates_by_group.sql")
    print(f"data                : {PARQUET_DIR} "
          f"({len(paths)} files, {sum(counts):,} rows)")
    print()
    print("MACHINE")
    for line in machine_spec():
        print(line)
    print()
    print("INTERPRETER AND LIBRARIES")
    for line in version_lines():
        print(line)
    print()
    print("MEASUREMENT")
    print(f"  peak memory         : resource.getrusage(RUSAGE_SELF).ru_maxrss "
          f"in a spawned child")
    print(f"  ru_maxrss unit      : "
          f"{'bytes (darwin)' if RU_MAXRSS_UNIT == 1 else 'kilobytes'}")
    print(f"  memory cap          : {_gib(MEM_CAP_BYTES)} resident, enforced by a "
          f"parent watchdog (ps -o rss=)")
    print(f"  wall cap            : {WALL_CAP_S:.0f}s per run")
    # Wall time is only meaningful next to what else the machine was doing.
    # This repository's working convention is that a number names its
    # conditions; a 1-minute load average above the CPU count means the
    # wall times below are contended and should be read as an upper bound.
    print(f"  load avg at start   : "
          f"{', '.join(f'{x:.2f}' for x in os.getloadavg())} "
          f"(1/5/15 min, {os.cpu_count()} logical cpus)")
    print(f"  why not setrlimit   : setrlimit(RLIMIT_AS) raises ValueError on "
          f"this macOS; see module docstring")
    if perturb:
        print("  PERTURBED           : pandas denial code forced to 4 "
              "(--perturb-pandas)")
    print()

    header = (
        f"{'rows':>12}  {'files':>5}  {'duckdb s':>9}  {'duckdb peak':>12}  "
        f"{'pandas s':>9}  {'pandas peak':>12}  {'identical':>9}"
    )
    print("SCALING")
    print(header)
    print("-" * len(header))

    all_identical = True
    any_compared = False
    notes: list[str] = []
    pandas_failed_at: tuple[int, str] | None = None
    full_scale: tuple[int, Outcome, Outcome] | None = None

    for target in SCALE_TARGETS:
        scope, rows = _select_files(paths, counts, target)
        d = run_path("duckdb", scope, rows=rows)
        if pandas_failed_at is None:
            p = run_path("pandas", scope, perturb=perturb, rows=rows)
        else:
            p = Outcome(
                kind="skipped",
                wall_s=0.0,
                peak_bytes=0,
                detail=(
                    "not attempted: pandas already failed at "
                    f"{pandas_failed_at[0]:,} rows"
                ),
            )

        if target is None:
            full_scale = (rows, d, p)

        if d.ok and p.ok:
            any_compared = True
            identical = tables_match(d.table, p.table)
            all_identical = all_identical and identical
            verdict = str(identical)
            if not identical:
                notes.append(f"MISMATCH at {rows:,} rows:")
                notes.extend(describe_mismatch(d.table, p.table))
        else:
            verdict = "n/a"

        d_s = f"{d.wall_s:9.2f}" if d.ok else f"{d.kind:>9}"
        p_s = f"{p.wall_s:9.2f}" if p.ok else f"{p.kind:>9}"
        print(
            f"{rows:>12,}  {len(scope):>5}  {d_s}  {_gib(d.peak_bytes):>12}  "
            f"{p_s}  {_gib(p.peak_bytes):>12}  {verdict:>9}"
        )

        if not d.ok:
            notes.append(f"duckdb at {rows:,} rows: {d.kind} -- {d.detail}")
        if not p.ok and p.kind != "skipped":
            notes.append(f"pandas at {rows:,} rows: {p.kind} -- {p.detail}")
            if pandas_failed_at is None:
                pandas_failed_at = (rows, f"{p.kind}: {p.detail}")
        elif p.kind == "skipped":
            notes.append(f"pandas at {rows:,} rows: {p.detail}")

    print()
    print(f"load avg at end       : "
          f"{', '.join(f'{x:.2f}' for x in os.getloadavg())}")
    print()
    if notes:
        print("NOTES")
        for line in notes:
            print(line)
        print()

    if pandas_failed_at is not None:
        n, why = pandas_failed_at
        print("M3/M4 OUTCOME")
        print(f"  pandas did NOT complete the national audit. It failed at "
              f"{n:,} rows.")
        print(f"  failure mode        : {why}")
        print(f"  M3 must be stated as: 'completes where the pandas baseline "
              f"exhausts memory at {n / 1_000_000:.1f}M rows'.")
        print("  No speed ratio against the full national data exists, and none "
              "is extrapolated.")
        print()
    elif full_scale is not None and full_scale[1].ok and full_scale[2].ok:
        rows, d, p = full_scale
        print("M3/M4 OUTCOME")
        print(f"  Both paths completed the full {rows:,}-row audit and produced "
              f"the same table.")
        speed = f"{p.wall_s / d.wall_s:.0f}x" if d.wall_s else "not divisible"
        mem = f"{p.peak_bytes / d.peak_bytes:.0f}x" if d.peak_bytes else "n/a"
        print(f"  M3 wall time        : DuckDB {d.wall_s:.2f}s vs pandas "
              f"{p.wall_s:.2f}s ({speed})")
        print(f"  M4 peak memory      : DuckDB {_gib(d.peak_bytes)} vs pandas "
              f"{_gib(p.peak_bytes)} ({mem})")
        print("  Both ratios are measured at full scale, not extrapolated from a "
              "smaller one.")
        print("  Read them with the caveat in this file's docstring: the audit "
              "needs 2 of 99")
        print("  columns, and column projection is a large part of why the "
              "DuckDB side is cheap.")
        print()

    if not any_compared:
        print("TABLES IDENTICAL: n/a (no scale point completed on both paths)")
        return 1
    print(f"TABLES IDENTICAL: {all_identical}")
    return 0 if all_identical else 1


if __name__ == "__main__":
    sys.exit(main())
