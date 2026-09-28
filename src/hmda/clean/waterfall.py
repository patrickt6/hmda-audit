"""The row-count waterfall: raw rows -> after each named filter -> analysis rows.

Every filter's row drop must be visible:
``hmda audit --waterfall`` prints a table whose first row equals the raw
count and whose last row equals the analysis-set count.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WaterfallRow:
    """One step of the waterfall.

    step: the filter name (``"raw"`` for the first row, a
        :class:`hmda.clean.filters.FilterSpec` name for every row after),
        or ``"race_not_available"`` for the dedicated trap-9 row below.
    rows_before: row count entering this step.
    rows_after: row count leaving this step.
    rows_dropped: ``rows_before - rows_after``. Always non-negative.
    note: free-text context, e.g. the column and rule for a filter step.
    """

    step: str
    rows_before: int
    rows_after: int
    rows_dropped: int
    note: str


def build_waterfall(frame, filter_names: list[str]) -> list[WaterfallRow]:
    """Apply each named filter in ``hmda.clean.filters.FILTER_REGISTRY`` in order.

    ``filter_names`` must all be keys of
    ``hmda.clean.filters.FILTER_REGISTRY``; an unknown name raises KeyError
    rather than being silently skipped. Returns one :class:`WaterfallRow` per
    step, first row ``step="raw"`` with ``rows_before == rows_after`` equal to
    the input row count, last row's ``rows_after`` equal to the analysis-set
    row count fed to every downstream fairness/model computation.
    """
    from hmda.clean.filters import FILTER_REGISTRY

    rows: list[WaterfallRow] = []
    raw_count = int(len(frame))
    rows.append(
        WaterfallRow(
            step="raw",
            rows_before=raw_count,
            rows_after=raw_count,
            rows_dropped=0,
            note="raw fixture/loaded rows, before any exclusion filter",
        )
    )

    current = frame
    for name in filter_names:
        spec = FILTER_REGISTRY[name]  # KeyError on unknown name, by contract
        before = int(len(current))
        current = spec.apply(current)
        after = int(len(current))
        rows.append(
            WaterfallRow(
                step=spec.name,
                rows_before=before,
                rows_after=after,
                rows_dropped=before - after,
                note=f"column={spec.column}; rule={spec.rule}",
            )
        )

    return rows


def race_not_available_share(frame) -> float:
    """Return the fraction of rows where ``derived_race == "Race Not Available"``.

    MEASURED (DC 2023, 2026-09-11): "Race Not Available" was
    34% of DC rows (5,948 of 17,474) and is the single largest
    ``derived_race`` category, larger than "White". This function must be
    called and its result printed beside every disparity number: a
    fairness number computed with those rows dropped, shown without this
    share, is exactly the kind of overstatement this rule exists to
    stop.
    """
    total = len(frame)
    if total == 0:
        return 0.0
    not_available = int(
        (frame["derived_race"].astype("string") == "Race Not Available").sum()
    )
    return not_available / total
