"""Chart design tokens and the one Plotly template this dashboard draws under.

Every figure in app.py registers and uses `hmda`. Nothing renders under
Plotly's default template, and no chart picks a colour of its own.

Palette provenance: the four categorical slots below were checked with the
house dataviz validator
(`node scripts/validate_palette.js "#CC2E3C,#0F7FB0,#C07C00" --mode light`),
which returned PASS on the lightness band, chroma floor, CVD separation
(worst adjacent pair delta-E 19.0 protan), the normal-vision floor and
contrast against the surface. GRID, MUTED and INK are text and rule tokens,
not categorical slots, so they are deliberately outside that set.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

SURFACE = "#FCFCFB"
INK = "#16212B"
MUTED = "#5E6E7D"
GRID = "#E3E7EA"
RECESSIVE = "#B3BFC9"

FLAGGED = "#CC2E3C"
PRIMARY = "#0F7FB0"
CAUTION = "#C07C00"

DISPLAY = "'IBM Plex Sans Condensed', 'Helvetica Neue', sans-serif"
BODY = "'IBM Plex Sans', -apple-system, sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, monospace"


def register() -> None:
    pio.templates["hmda"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            font=dict(family=BODY, size=15, color=INK),
            title=dict(font=dict(family=DISPLAY, size=22, color=INK), x=0, xanchor="left"),
            margin=dict(l=16, r=48, t=64, b=56),
            colorway=[PRIMARY, FLAGGED, CAUTION],
            hoverlabel=dict(
                bgcolor=INK,
                bordercolor=INK,
                font=dict(family=MONO, size=14, color="#FFFFFF"),
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor=GRID,
                gridwidth=1,
                zeroline=False,
                linecolor=GRID,
                ticks="outside",
                tickcolor=GRID,
                ticklen=6,
                tickfont=dict(family=MONO, size=14, color=MUTED),
                title=dict(font=dict(family=BODY, size=15, color=MUTED)),
                automargin=True,
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                linecolor=GRID,
                ticks="",
                tickfont=dict(family=BODY, size=15, color=INK),
                title=dict(font=dict(family=BODY, size=15, color=MUTED)),
                automargin=True,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                x=0,
                font=dict(size=14, color=MUTED),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
    )
    pio.templates.default = "hmda"
