"""Which dataset the audit runs against, and the sentence that says so.

Two sources exist:

* ``FIXTURE`` -- the committed 50,000-row parquet fixture
  (``tests/fixtures/hmda_50k.parquet``, DC/WY/VT only), loaded through
  :func:`hmda.clean.load_fixture` as a ``pandas.DataFrame``.
* ``NATIONAL`` -- the 168 per-state/year parquet files under
  ``data/parquet/``, exposed as a DuckDB VIEW.

The design that makes one body of SQL serve both: every file under ``sql/``
queries ``from frame``. For the fixture that name is supplied by
``duckdb.Connection.register("frame", dataframe)``, which is fine at 50k
rows and fatal at 36.7M. For the national source the same name is supplied
by::

    CREATE OR REPLACE VIEW frame AS
    SELECT * FROM read_parquet('data/parquet/*.parquet')

Same SQL text, same table name, but DuckDB streams the scan and never
materializes the table. **No ``.sql`` file changes.** The registration
difference is hidden behind :func:`register_frame`, which the two
``con.register("frame", frame)`` sites in ``hmda.fairness.rates`` call
instead of registering directly.

The split that matters:

* **Aggregation paths take** :meth:`Source.frame` and stay out-of-core.
* **Model-training paths (mitigate, recourse, SHAP) take**
  :meth:`Source.sample`, because lightgbm and shap need materialized numpy
  arrays and cannot stream. Those paths must print the sample size, since a
  number computed on a sample is not a national number.

What this module deliberately does NOT do: it does not clean. Both sources
return RAW UNCLEANED rows, exactly as :func:`hmda.clean.load_fixture`
documents. ``hmda.clean.sentinels`` is written against ``pandas.Series``
(``astype``, ``pd.NA``, ``pd.Categorical``, ``pd.to_numeric``) and cannot
run on a DuckDB relation. The
``sql/`` aggregations do not depend on cleaning having run -- e.g.
``sql/rates_by_group.sql`` repeats the ``action_taken != 6`` exclusion
defensively -- so the aggregation path is correct on raw input. Any
national path that needs sentinel-resolved columns must go through
:meth:`Source.sample`, which returns pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FIXTURE = "fixture"
NATIONAL = "national"

# Repository root: src/hmda/clean/source.py -> clean -> hmda -> src -> root.
# Same derivation as hmda.fairness.rates._SQL_DIR.
_REPO_ROOT = Path(__file__).resolve().parents[3]

_PARQUET_GLOB = _REPO_ROOT / "data" / "parquet" / "*.parquet"

# The column whose proportions Source.sample() preserves. derived_race is the
# protected class the audit is about, and the small groups in it are exactly
# the ones a uniform sample can thin out.
_STRATUM_COLUMN = "derived_race"


class ParquetView:
    """A lazy handle on the national parquet files.

    It is NOT a ``pandas.DataFrame`` and holds no rows. It exists so
    :func:`register_frame` can attach the national data to any DuckDB
    connection under the name ``frame`` without materializing it, and so
    ``isinstance(src.frame(), pandas.DataFrame)`` is false for the national
    source.
    """

    def __init__(self, glob: str) -> None:
        self.glob = glob

    def register_into(self, con) -> None:
        """Create (or replace) the view ``frame`` on ``con``.

        DuckDB cannot prepare a parameter inside ``CREATE VIEW``
        ("Binder Error: Unexpected prepared parameter. This type of
        statement can't be prepared!", measured 2026-09-13), so the path is
        spliced. It is a module-level constant derived from the repository
        root, never caller input, and single quotes are still escaped.
        """
        literal = self.glob.replace("'", "''")
        con.execute(
            "CREATE OR REPLACE VIEW frame AS "
            f"SELECT * FROM read_parquet('{literal}')"
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ParquetView({self.glob!r})"


def register_frame(con, frame) -> None:
    """Expose ``frame`` to ``con`` under the table name ``frame``.

    A :class:`ParquetView` becomes a streamed view over parquet; anything
    else (a ``pandas.DataFrame``, an arrow table) is registered as before.
    This is the ONE behavioural difference between the fixture and national
    paths, and it is why no ``.sql`` file needs to change.
    """
    register_into = getattr(frame, "register_into", None)
    if callable(register_into):
        register_into(con)
    else:
        con.register("frame", frame)


@dataclass(frozen=True)
class Source:
    """A dataset the audit can run against, and the sentence describing it."""

    name: str  # "fixture" or "national"
    rows: int  # exact row count, measured at open time
    label: str  # the SOURCE: line printed by every command
    _relation: object  # ParquetView or pandas DataFrame; not public

    def frame(self):
        """The tabular object to hand to fairness/model functions.

        For the national source this is a :class:`ParquetView`, never a
        DataFrame. Aggregation paths use this.
        """
        return self._relation

    def sample(self, n: int, seed: int = 0):
        """A reproducible stratified ``pandas.DataFrame`` of at most n rows.

        Model training paths use this. Aggregation paths use :meth:`frame`.

        Stratified means proportional-to-size on ``derived_race``: each
        group keeps its national share of the sample, so a small group is
        not thinned out of the training set. Within a group the draw is
        DuckDB reservoir sampling with an explicit seed, on a single-threaded
        connection so the draw is reproducible run to run.

        Callers must print the returned length. A number computed here is a
        sample number, not a national number.

        If a stratum has FEWER rows than its quota (a genuinely small
        protected group may), that stratum contributes all of its rows and
        no more -- the total returned length is then less than ``n``. This
        is documented behaviour, not an accident: it is impossible to
        deliver a quota a stratum does not have without duplicating rows,
        and duplication would corrupt reproducibility and downstream model
        training. Callers that need to detect this compare the returned
        length against ``n`` themselves.
        """
        if n <= 0:
            raise ValueError(f"sample size must be positive, got {n!r}")

        import duckdb

        con = duckdb.connect()
        try:
            con.execute("SET threads TO 1")
            register_frame(con, self._relation)
            if n >= self.rows:
                return con.execute("SELECT * FROM frame").df()

            strata = con.execute(
                f"SELECT {_STRATUM_COLUMN} AS g, count(*) AS c "
                "FROM frame GROUP BY 1 ORDER BY 1"
            ).fetchall()

            quotas = _proportional_quotas([c for _, c in strata], n)

            parts = []
            for (group_value, _count), quota in zip(strata, quotas):
                if quota <= 0:
                    continue
                if group_value is None:
                    where = f"{_STRATUM_COLUMN} IS NULL"
                    params: list = []
                else:
                    where = f"{_STRATUM_COLUMN} = ?"
                    params = [group_value]
                part = con.execute(
                    f"SELECT * FROM (SELECT * FROM frame WHERE {where}) "
                    f"USING SAMPLE {int(quota)} ROWS (reservoir, {int(seed)})",
                    params,
                ).df()
                parts.append(part)

            import pandas as pd

            if not parts:
                return con.execute("SELECT * FROM frame LIMIT 0").df()
            combined = pd.concat(parts, ignore_index=True)
            # Rows come back grouped by stratum in block order,
            # which would badly skew any downstream train/test split that
            # slices rather than shuffles. Shuffle deterministically with
            # the same seed so reproducibility is preserved.
            return combined.sample(frac=1, random_state=seed).reset_index(
                drop=True
            )
        finally:
            con.close()


def _proportional_quotas(counts: list[int], n: int) -> list[int]:
    """Largest-remainder apportionment of ``n`` across strata of ``counts``.

    Deterministic, sums to exactly ``min(n, sum(counts))``, and never asks a
    stratum for more rows than it has.
    """
    total = sum(counts)
    if total == 0:
        return [0 for _ in counts]
    n = min(n, total)
    exact = [c * n / total for c in counts]
    quotas = [int(x) for x in exact]
    remainder = n - sum(quotas)
    # Largest fractional part first; index breaks ties, so the result does
    # not depend on dict or set ordering.
    order = sorted(
        range(len(counts)), key=lambda i: (-(exact[i] - quotas[i]), i)
    )
    for i in order:
        if remainder <= 0:
            break
        if quotas[i] < counts[i]:
            quotas[i] += 1
            remainder -= 1
    return quotas


def _count_rows(relation) -> int:
    """Measure the row count by querying. Never a hardcoded literal."""
    import duckdb

    con = duckdb.connect()
    try:
        register_frame(con, relation)
        return int(con.execute("SELECT count(*) FROM frame").fetchone()[0])
    finally:
        con.close()


def open_source(name: str) -> Source:
    """Open the named dataset, measuring its row count at open time.

    Raises ``ValueError`` on any unrecognised name. A typo must never
    silently fall back to the fixture: that is how a 50,000-row number ends
    up in a report wearing a national label.
    """
    if name == FIXTURE:
        from hmda.clean import FIXTURE_PATH, load_fixture

        relation = load_fixture()
        rows = _count_rows(relation)
        label = (
            f"SOURCE: fixture -- {rows:,} rows, DC/WY/VT only, "
            f"{FIXTURE_PATH} (raw, uncleaned)"
        )
        return Source(name=FIXTURE, rows=rows, label=label, _relation=relation)

    if name == NATIONAL:
        glob = str(_PARQUET_GLOB)
        relation = ParquetView(glob)
        rows = _count_rows(relation)
        label = (
            f"SOURCE: national -- {rows:,} rows, all states, "
            f"{glob} (raw, uncleaned)"
        )
        return Source(name=NATIONAL, rows=rows, label=label, _relation=relation)

    raise ValueError(
        f"unknown source: {name!r} (expected {FIXTURE!r} or {NATIONAL!r})"
    )
