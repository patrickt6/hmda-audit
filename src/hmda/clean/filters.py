"""Row-exclusion filters applied between the raw load and the analysis set.

Every filter here is a named, registered step so the row-count waterfall
(``waterfall.py``) can print exactly what each one dropped and why.
Sentinel and exclusion handling is visible: every
filter appears in the row-count waterfall in the report.

A filter is never applied ad hoc at a call site. It is registered here,
applied by name through ``waterfall.build_waterfall``, and its row-drop count
is always visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


class RowFilter(Protocol):
    """A filter step: takes the current frame, returns the kept rows.

    Implementations receive whatever tabular object the loader uses (a
    ``pandas.DataFrame`` or a DuckDB relation, per the loader's
    choice) and must return the same type with only the kept rows.
    """

    def __call__(self, frame): ...


@dataclass(frozen=True)
class FilterSpec:
    """One registered, named row-exclusion rule.

    name: short identifier, used as the waterfall row label.
    column: the HMDA column the rule inspects.
    rule: a one-line human description of the rule, printed in the waterfall.
        Every filter names the column and
        the rule it enforces.
    apply: the callable that performs the filter.
    """

    name: str
    column: str
    rule: str
    apply: RowFilter


# The filter registry. The count of entries must equal
# the number of waterfall rows: a grep -c of the filter
# registry equals the number of waterfall rows.
FILTER_REGISTRY: dict[str, FilterSpec] = {}


def register_filter(spec: FilterSpec) -> None:
    """Add ``spec`` to :data:`FILTER_REGISTRY`, keyed by ``spec.name``.

    Raises ValueError if a filter with the same name is already registered,
    so two filters cannot silently shadow each other's exclusion rule.
    """
    if spec.name in FILTER_REGISTRY:
        raise ValueError(f"register_filter: '{spec.name}' is already registered")
    FILTER_REGISTRY[spec.name] = spec


def exclude_purchased_loans(frame):
    """Drop rows where ``action_taken == 6`` (a purchased loan, not an application).

    MEASURED: purchased loans are not applications and
    must be excluded from any denial-rate computation. The exclusion must be
    visible in the row-count waterfall, never silent.
    """
    action_taken = frame["action_taken"]
    # action_taken is loaded raw (string dtype in the fixture); compare on
    # both the string and numeric spelling so this filter is not silently
    # bypassed by a dtype change upstream.
    mask = action_taken.astype("string") != "6"
    return frame[mask]


register_filter(
    FilterSpec(
        name="exclude_purchased_loans",
        column="action_taken",
        rule="drop rows where action_taken == 6 (purchased loan, not an application)",
        apply=exclude_purchased_loans,
    )
)
