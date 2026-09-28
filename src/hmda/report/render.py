"""Render the HTML (and optional PDF) report from the database.

The report renders from the database,
never from a markdown file a human edited. ``make
report`` exits 0 and writes ``out/report.html``; lender-level figure
captions carry the disparity-is-not-proof sentence.
"""

from __future__ import annotations

from pathlib import Path

DISPARITY_NOT_PROOF_SENTENCE = (
    "This shows a disparity that warrants review, not proof of discrimination "
    "by any named lender."
)

OUT_REPORT_PATH = Path("out/report.html")


def render_report(db_path: Path, out_path: Path = OUT_REPORT_PATH) -> Path:
    """Render the full HTML report from ``db_path`` using the Jinja2 templates in this package.

    Every lender-level figure's caption must include
    :data:`DISPARITY_NOT_PROOF_SENTENCE` verbatim: a count of
    "warrants review" in the output must be at least equal to the number
    of lender-level figures. Returns the path written.
    """
    raise NotImplementedError
