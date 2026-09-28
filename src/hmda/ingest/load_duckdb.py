"""Load per-state-year parquet files into the single DuckDB file, unioned.

DuckDB, not Spark, not pandas: the "national scale on a
laptop" claim is the interesting one. This module owns ``data/hmda.duckdb``,
a single on-disk, no-server, no-cloud-cost columnar file.

Also owns the streaming CSV -> parquet conversion. Ingest must
stream: do not download all three years as CSV first and convert later.
DuckDB reads the CSV directly and writes parquet without pandas ever holding
the whole file in memory, which keeps peak RSS low for the eventual
national run.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

DUCKDB_PATH = Path("data/hmda.duckdb")


def csv_to_parquet(csv_path: Path, parquet_path: Path) -> int:
    """Convert one state-year CSV to parquet via DuckDB's own CSV reader.

    Reads every column as VARCHAR (``all_varchar=True``): the known data
    traps (sentinel strings, mixed-type debt_to_income_ratio, the
    "80"/"80.0" duplicate) are all string-level problems, and casting
    anything at ingest time would silently pre-decide the fix before
    ``hmda.clean`` gets to make it. Returns the row count written.
    """
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        # DuckDB's COPY ... TO clause does not accept a bound parameter for
        # the destination path, so the path is escaped and interpolated
        # directly; both csv_path and parquet_path are produced by this
        # repo's own download/naming code, never external input.
        csv_literal = str(csv_path).replace("'", "''")
        parquet_literal = str(parquet_path).replace("'", "''")
        con.execute(
            f"COPY (SELECT * FROM read_csv('{csv_literal}', all_varchar=True, header=True)) "
            f"TO '{parquet_literal}' (FORMAT PARQUET)"
        )
        row_count = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [str(parquet_path)]
        ).fetchone()[0]
    finally:
        con.close()
    return row_count


def load_parquet_dir(parquet_dir: Path, db_path: Path = DUCKDB_PATH) -> int:
    """Union every ``*.parquet`` file in ``parquet_dir`` into ``db_path`` as one ``lar`` table.

    Each input parquet is one state-year (see ``download.py``). Returns the
    total row count loaded, which callers use as M1's raw count for the
    row-count waterfall (``hmda.clean.waterfall``). Safe to re-run: the
    whole ``lar`` table is rebuilt from the current contents of
    ``parquet_dir`` each call (a DROP + CREATE), so re-loading the same
    directory never duplicates rows, and adding a newly-completed
    state-year's parquet before the next call picks it up automatically.
    """
    parquet_files = sorted(Path(parquet_dir).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files found in {parquet_dir}")

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        glob_pattern = str(Path(parquet_dir) / "*.parquet")
        con.execute("DROP TABLE IF EXISTS lar")
        con.execute(
            "CREATE TABLE lar AS SELECT * FROM read_parquet(?, union_by_name=True)",
            [glob_pattern],
        )
        row_count = con.execute("SELECT COUNT(*) FROM lar").fetchone()[0]
    finally:
        con.close()
    return row_count


def counts_by_year(db_path: Path = DUCKDB_PATH) -> dict[int, int]:
    """Return ``{activity_year: row_count}`` read directly from ``db_path``.

    This is M1's row-count-per-year figure, and it must be read from
    DuckDB, never from a doc: ``hmda verify --counts`` prints row count
    per year and distinct lei count, read straight from DuckDB.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT CAST(activity_year AS INTEGER) AS yr, COUNT(*) "
            "FROM lar GROUP BY yr ORDER BY yr"
        ).fetchall()
    finally:
        con.close()
    return {int(yr): int(cnt) for yr, cnt in rows}


def distinct_lei_count(db_path: Path = DUCKDB_PATH) -> int:
    """Return the count of distinct ``lei`` (lender) values across the whole table.

    This is M1's lender-count figure ("audited N million mortgage
    applications from K lenders across Y years").
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        result = con.execute("SELECT COUNT(DISTINCT lei) FROM lar").fetchone()[0]
    finally:
        con.close()
    return int(result)
