"""The minimum-count floor for reporting any disparity ratio.

Small-denominator groups give unstable ratios. A minimum-count floor is
required before any disparity is reported, and the floor must be a
stated parameter, not a magic number.
"""

from __future__ import annotations

DEFAULT_MIN_COUNT = 100  # documented default; always overridable via --min-count


def below_floor(denominator: int, min_count: int = DEFAULT_MIN_COUNT) -> bool:
    """Return True if ``denominator`` is below ``min_count`` and must be suppressed.

    ``min_count`` is never hardcoded at a call site; every caller either
    passes the CLI's ``--min-count`` value through explicitly or this
    documented default.
    """
    return denominator < min_count
