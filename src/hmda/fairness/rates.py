"""Denial rate by protected class, per lender and nationally.

SQL lives in ``sql/`` as reviewable files,
never as an f-string: no .py file contains a
SELECT in an f-string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# register_frame supplies the table name ``frame`` that every sql/*.sql file
# queries: con.register() for an in-memory pandas frame (the fixture), a
# streamed CREATE VIEW over parquet for the national source. No .sql
# file is aware of the difference.
from hmda.clean.source import register_frame

# Protected-class columns this module (and hmda.fairness.air) is permitted
# to group by. group_column is spliced into the .sql file text as an
# IDENTIFIER (SQL has no safe parameter binding for identifiers), so it is
# validated against this allowlist AND against _IDENTIFIER_RE before it
# ever reaches a query string -- this is the "no f-string SELECT" rule
# satisfied by never letting the *aggregation logic*
# live in Python: only a pre-validated column name is substituted into SQL
# that already lives in sql/*.sql.
ALLOWED_GROUP_COLUMNS: frozenset[str] = frozenset(
    {"derived_race", "derived_ethnicity", "derived_sex"}
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_SQL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "sql"


def _validate_group_column(group_column: str) -> None:
    if group_column not in ALLOWED_GROUP_COLUMNS:
        raise ValueError(
            f"group_column {group_column!r} is not in ALLOWED_GROUP_COLUMNS "
            f"{sorted(ALLOWED_GROUP_COLUMNS)}"
        )
    if not _IDENTIFIER_RE.match(group_column):
        # Unreachable given the allowlist above, but kept as a second,
        # independent guard against ever splicing an unsafe identifier.
        raise ValueError(f"group_column {group_column!r} is not a safe SQL identifier")


def _load_sql(name: str) -> str:
    path = _SQL_DIR / name
    text = path.read_text()
    if "SELECT" not in text.upper():
        raise ValueError(f"{path} does not look like a SQL query")
    return text


def run_group_query(sql_filename: str, frame, group_column: str, lei: str | None):
    """Render ``sql_filename`` with a validated ``group_column`` and run it.

    ``frame`` is registered with an in-process DuckDB connection as the
    table ``frame``; ``lei`` is bound as a query VALUE parameter (never
    spliced into the SQL text). Returns a list of DuckDB result rows as
    dicts. This is the one place group_column identifier-substitution
    happens, so every .sql file under sql/ can assume a fixed, reviewable
    aggregation shape.
    """
    import duckdb

    _validate_group_column(group_column)
    sql_template = _load_sql(sql_filename)
    sql = sql_template.replace("{group_column}", group_column)

    con = duckdb.connect()
    try:
        register_frame(con, frame)
        result = con.execute(sql, [lei, lei]).fetchall()
        columns = [d[0] for d in con.description]
    finally:
        con.close()

    return [dict(zip(columns, row)) for row in result]


def run_group_query_no_params(sql_filename: str, frame, group_column: str):
    """Like :func:`run_group_query`, for .sql files with no ``?`` placeholders.

    Used by ``hmda.fairness.air.flagged_lenders``, whose query groups by
    ``lei`` itself rather than filtering to one, so there is no lei value to
    bind.
    """
    import duckdb

    _validate_group_column(group_column)
    sql_template = _load_sql(sql_filename)
    sql = sql_template.replace("{group_column}", group_column)

    con = duckdb.connect()
    try:
        register_frame(con, frame)
        result = con.execute(sql).fetchall()
        columns = [d[0] for d in con.description]
    finally:
        con.close()

    return [dict(zip(columns, row)) for row in result]


@dataclass(frozen=True)
class DenialRateRow:
    """One group's denial rate, for one lender or nationally.

    lei: the lender identifier, or ``None`` for the national aggregate.
    group_column: which protected-class column this row groups by
        (e.g. "derived_race").
    group_value: the value within that column (e.g. "Black or African
        American").
    denials: count of denied applications (``action_taken == 3``) in scope.
    applications: count of applications in scope (purchased loans already
        excluded upstream by ``hmda.clean.filters.exclude_purchased_loans``).
    denial_rate: ``denials / applications``.
    """

    lei: str | None
    group_column: str
    group_value: str
    denials: int
    applications: int
    denial_rate: float


def denial_rate_by_group(
    frame, group_column: str, lei: str | None = None
) -> list[DenialRateRow]:
    """Compute denial rate per distinct value of ``group_column``.

    ``frame`` must already be the cleaned analysis set (post-waterfall, see
    ``hmda.clean.waterfall.build_waterfall``): purchased loans excluded,
    sentinels resolved. If ``lei`` is given, restricts to that lender;
    otherwise computes the national aggregate. The underlying aggregation
    must be executed from a ``.sql`` file under ``sql/``, not an inline
    string.
    """
    rows = run_group_query("rates_by_group.sql", frame, group_column, lei)
    out: list[DenialRateRow] = []
    for row in rows:
        applications = int(row["applications"])
        denials = int(row["denials"])
        denial_rate = denials / applications if applications else 0.0
        out.append(
            DenialRateRow(
                lei=lei,
                group_column=group_column,
                group_value=row["group_value"],
                denials=denials,
                applications=applications,
                denial_rate=denial_rate,
            )
        )
    return out
