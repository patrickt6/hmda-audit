"""The real HMDA LAR schema, read from the header, never hand-typed.

MEASURED (DC 2023, 2026-09-11): 99 columns, confirmed against the
real file header (``head -1 dc2023.csv | tr ',' '\\n' | wc -l`` -> 99).
There is NO ``combined_loan_to_value_ratio`` column in the real file.
``docs/DATA-DICTIONARY.md`` must be generated from this
module's output, never hand-typed, so it cannot drift from the
real header.

Header equality across 2023/2024/2025 is UNCONFIRMED: whether the
99-column header is identical across 2023, 2024 and 2025 was still
unknown as of 2026-09-11. :func:`assert_header_stable` exists to check
this, not assume it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SchemaVersion:
    """The column set observed for one HMDA activity year.

    year: the HMDA activity year the header was read from.
    columns: the ordered column names, exactly as they appear in the header
        row of the source CSV.
    source_state: the state code whose file was actually opened to read
        this header (e.g. "DC"), so the claim is traceable to one file, not
        assumed national.
    """

    year: int
    columns: tuple[str, ...]
    source_state: str


def read_header(csv_path) -> SchemaVersion:
    """Read the literal header row of ``csv_path`` and return it as a :class:`SchemaVersion`.

    Never hand-types or guesses a column list; this is the only function
    that may produce a :class:`SchemaVersion`. ``year`` and ``source_state``
    are parsed from the filename convention ``{STATE}_{YEAR}.csv`` used by
    ``download.py``; if that convention is not followed, both fall back to
    unknown/0 and the caller must set them.
    """
    csv_path = Path(csv_path)
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

    stem = csv_path.stem  # "DC_2023"
    year = 0
    source_state = "UNKNOWN"
    if "_" in stem:
        state_part, _, year_part = stem.rpartition("_")
        if year_part.isdigit():
            year = int(year_part)
            source_state = state_part

    return SchemaVersion(year=year, columns=tuple(header), source_state=source_state)


def assert_header_stable(versions: list[SchemaVersion]) -> None:
    """Raise ValueError if any two :class:`SchemaVersion` in ``versions`` have different columns.

    Header equality across 2023/2024/2025 is a measured fact this module
    must establish, not assume. If headers differ, the caller must add a
    reconciliation step and record the difference in
    ``docs/DATA-DICTIONARY.md`` rather than silently unioning mismatched
    columns.
    """
    if not versions:
        return
    reference = versions[0]
    for v in versions[1:]:
        if v.columns != reference.columns:
            ref_set = set(reference.columns)
            other_set = set(v.columns)
            only_in_ref = ref_set - other_set
            only_in_other = other_set - ref_set
            raise ValueError(
                f"header mismatch: year {reference.year} ({reference.source_state}) "
                f"vs year {v.year} ({v.source_state}). "
                f"Only in {reference.year}: {sorted(only_in_ref)}. "
                f"Only in {v.year}: {sorted(only_in_other)}."
            )
