"""Per-figure/table lineage: source file, checksum, SQL file, and commit.

Supports the report render's "Lineage" section. Mirrors this
repo's own provenance contract.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LineageRecord:
    """Provenance for one figure or table in the rendered report.

    artifact_name: the figure/table identifier as it appears in the report.
    source_file: the data file the number traces to (e.g. a parquet or the
        DuckDB file).
    source_checksum: SHA-256 of ``source_file`` at render time.
    sql_file: the ``sql/*.sql`` file that produced the aggregation, if any.
    commit: the git commit hash the render was produced at.
    """

    artifact_name: str
    source_file: str
    source_checksum: str
    sql_file: str | None
    commit: str


def record_lineage(artifact_name: str, source_file, sql_file=None) -> LineageRecord:
    """Build a :class:`LineageRecord` for ``artifact_name``, reading the current commit hash and file checksum."""
    raise NotImplementedError


def verify_lineage(records: list[LineageRecord]) -> bool:
    """Return True iff every record has a non-empty ``source_file``, ``source_checksum`` and ``commit``.

    This is what ``hmda verify --lineage`` calls; must exit non-zero when
    any figure lacks a source.
    """
    raise NotImplementedError
