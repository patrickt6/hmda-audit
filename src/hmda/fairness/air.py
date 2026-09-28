"""Adverse impact ratio (the four-fifths rule), per lender and nationally.

Screens K lenders against the four-fifths rule and flags F, the count that
fail it. Reference group is the group with the highest approval rate (i.e.
lowest denial rate) within each lender or scope.
"""

from __future__ import annotations

from dataclasses import dataclass

from hmda.fairness.floors import below_floor
from hmda.fairness.rates import run_group_query, run_group_query_no_params

FOUR_FIFTHS_THRESHOLD = 0.80

# The two non-numeric ``AIRRow.ratio`` values. A row carrying either one is
# never flagged.
SUPPRESSED = "SUPPRESSED"
UNDEFINED = "UNDEFINED"


@dataclass(frozen=True)
class AIRRow:
    """One group's adverse impact ratio against the reference group.

    lei: the lender identifier, or ``None`` for the national aggregate.
    group_column: the protected-class column (e.g. "derived_sex").
    group_value: the value within that column.
    reference_group_value: the group this row's approval rate is compared
        against (the group with the highest approval rate in scope).
    denominator: the applications count this group's ratio rests on. Never
        absent: every group gets a denominator, even a row that is
        suppressed.
    ratio: one of three things.

        - A float: approval_rate(group) / approval_rate(reference_group).
        - The literal string ``"SUPPRESSED"`` when ``denominator`` is below
          the floor, or when no group in scope clears it (see
          ``floors.py``). A suppressed row is never a bare number and never
          silently dropped.
        - The literal string ``"UNDEFINED"`` when the reference group's
          approval rate is 0. Then every eligible group was denied every
          time, the ratio is 0/0, and there is no comparison to make. The
          row is kept, like a suppressed row, and is not flagged. (Before
          2026-09-28 this case returned 0.0 and flagged the reference group
          against itself; see docs/VERIFICATION.md section 3.)
    flagged: True if ``ratio`` is a number and below
        :data:`FOUR_FIFTHS_THRESHOLD`. Always False for ``"SUPPRESSED"``
        and ``"UNDEFINED"``.
    """

    lei: str | None
    group_column: str
    group_value: str
    reference_group_value: str
    denominator: int
    ratio: float | str
    flagged: bool


def adverse_impact_ratio(frame, group_column: str, min_count: int, lei: str | None = None) -> list[AIRRow]:
    """Compute the four-fifths adverse impact ratio per group.

    ``min_count`` is the explicit, required floor (see ``floors.py``); there
    is no hidden default applied silently. Groups below the floor appear as
    an :class:`AIRRow` with ``ratio="SUPPRESSED"``, never omitted from the
    return list. If the reference group's approval rate is 0, every
    eligible group gets ``ratio="UNDEFINED"`` and is not flagged.
    """
    raw_rows = run_group_query("air_by_group.sql", frame, group_column, lei)
    return _air_rows_from_raw(raw_rows, group_column, lei, min_count)


def _air_rows_from_raw(raw_rows, group_column: str, lei: str | None, min_count: int) -> list[AIRRow]:
    """Turn a list of ``{group_value, denominator, denials, approval_rate}``
    dicts, all sharing one ``lei``/``group_column`` scope, into
    :class:`AIRRow` objects.

    Shared by :func:`adverse_impact_ratio` (one lei/group_column at a time,
    via ``air_by_group.sql``) and :func:`flagged_lenders` (every lei at
    once, via ``air_by_lei_group.sql``, split back into per-lei scopes
    here) so the reference-group and suppression rule lives in exactly one
    place.
    """
    if not raw_rows:
        return []

    # The reference group is the group with the HIGHEST approval rate
    # (= lowest denial rate) among groups that themselves clear the
    # min-count floor. A reference group whose own denominator is too thin
    # to trust would make every ratio in this scope untrustworthy too.
    eligible = [r for r in raw_rows if not below_floor(int(r["denominator"]), min_count)]
    reference_row = max(eligible, key=lambda r: r["approval_rate"]) if eligible else None

    out: list[AIRRow] = []
    for row in raw_rows:
        denominator = int(row["denominator"])
        group_value = row["group_value"]
        reference_value = reference_row["group_value"] if reference_row else "NONE_ELIGIBLE"

        if below_floor(denominator, min_count) or reference_row is None:
            out.append(
                AIRRow(
                    lei=lei,
                    group_column=group_column,
                    group_value=group_value,
                    reference_group_value=reference_value,
                    denominator=denominator,
                    ratio=SUPPRESSED,
                    flagged=False,
                )
            )
            continue

        ref_approval = reference_row["approval_rate"]
        if not ref_approval:
            # Every eligible group was denied every time, so the ratio is
            # 0/0. That is undefined, not 0.0, and it is not a disparity:
            # the reference group would be flagged against itself.
            out.append(
                AIRRow(
                    lei=lei,
                    group_column=group_column,
                    group_value=group_value,
                    reference_group_value=reference_value,
                    denominator=denominator,
                    ratio=UNDEFINED,
                    flagged=False,
                )
            )
            continue

        ratio = row["approval_rate"] / ref_approval
        out.append(
            AIRRow(
                lei=lei,
                group_column=group_column,
                group_value=group_value,
                reference_group_value=reference_value,
                denominator=denominator,
                ratio=ratio,
                flagged=ratio < FOUR_FIFTHS_THRESHOLD,
            )
        )
    return out


def flagged_lenders(frame, min_count: int) -> list[str]:
    """Return the list of ``lei`` values with at least one flagged group.

    This is the count of flagged lenders. Per the rule that no lender is
    ever named, this list is used only in aggregate counts in any public
    artifact: never printed with lender identity in the site or README.
    """
    from hmda.fairness.rates import ALLOWED_GROUP_COLUMNS

    flagged: set[str] = set()
    for group_column in ALLOWED_GROUP_COLUMNS:
        # One aggregation per group_column, grouped by lei AND group_value
        # together (air_by_lei_group.sql), instead of one query per lender
        # per column -- the fixture alone has 800+ distinct leis, so that
        # would be thousands of DuckDB round trips over the same frame.
        raw_rows = run_group_query_no_params("air_by_lei_group.sql", frame, group_column)

        by_lei: dict[str, list[dict]] = {}
        for row in raw_rows:
            by_lei.setdefault(row["lei"], []).append(row)

        for lei, rows_for_lei in by_lei.items():
            if lei in flagged:
                continue
            air_rows = _air_rows_from_raw(rows_for_lei, group_column, lei, min_count)
            if any(r.flagged for r in air_rows):
                flagged.add(lei)

    return sorted(flagged)
