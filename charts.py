"""Every chart on the dashboard, written as an explanatory graphic.

Each chart carries its own headline sentence, its own plain-English method
line, and its explanation on the graphic rather than in a caption beneath it.
A reader who sees only the headline still learns the finding.

Each function takes one artifact dict loaded from results/ and returns a
Plotly figure. No function computes a statistic: the only arithmetic here is
denials / applications, which reproduces the denial_rate column of the
full-file query recorded in results/denial_rates.json and was checked against it.

Every headline states what the data shows. None states or implies that any
lender discriminated: the measurement is statistical disparity in public
filings, which is a screening signal and not a finding of discrimination.
"""

from __future__ import annotations

import plotly.graph_objects as go

from theme import (
    BODY,
    CAUTION,
    DISPLAY,
    FLAGGED,
    GRID,
    INK,
    MONO,
    MUTED,
    PRIMARY,
    RECESSIVE,
    SURFACE,
)

PARTITION_ORDER = ["Sex", "Ethnicity", "Race"]


def _headline(fig: go.Figure, title: str, subtitle: str, size: int = 26) -> None:
    """One sentence a reader can take away, then the method in plain words.

    Called last, after the figure's height is set, so the title can be placed a
    fixed number of pixels below the top edge rather than a fraction of a
    height that differs from chart to chart.
    """
    height = fig.layout.height or 450
    fig.update_layout(
        title=dict(
            text=(
                f"<span style='font-size:{size}px'>{title}</span><br>"
                f"<span style='font-size:9px'>&nbsp;</span><br>"
                f"<span style='font-family:{BODY};font-size:15px;"
                f"font-weight:400;color:{MUTED}'>{subtitle}</span>"
            ),
            x=0,
            xanchor="left",
            xref="paper",
            y=1 - 20 / height,
            yanchor="top",
        )
    )


def four_fifths(data: dict) -> go.Figure:
    """The hero. All 18 comparisons as bars, one hard line at 80 percent.

    Bars run from zero, so the lengths are honest. Colour carries one thing:
    whether the bar stops short of the line. The three rows that are the
    baseline compared with itself are drawn hollow, because a group compared
    with itself is 100 percent by construction and is not a result.
    """
    threshold = data["threshold"]
    rows = sorted(data["rows"], key=lambda r: r["ratio"], reverse=True)

    labels = []
    for r in rows:
        name = r["group"]
        if sum(1 for x in rows if x["group"] == name) > 1:
            name = f"{name} ({r['partition'].lower()})"
        if r.get("is_reference"):
            name = f"{name} \u2014 baseline"
        labels.append(name)

    values = [r["ratio"] * 100 for r in rows]
    colors, texts = [], []
    for r, v in zip(rows, values):
        if r["flag"]:
            colors.append(FLAGGED)
            texts.append(f"<b>{v:.0f}%</b>")
        elif r.get("is_reference"):
            colors.append("rgba(0,0,0,0)")
            texts.append(f"{v:.0f}%")
        else:
            colors.append(RECESSIVE)
            texts.append(f"{v:.0f}%")

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(
                    color=[MUTED if r.get("is_reference") else "rgba(0,0,0,0)" for r in rows],
                    width=[1.5 if r.get("is_reference") else 0 for r in rows],
                ),
            ),
            width=0.66,
            text=texts,
            textposition="outside",
            textfont=dict(family=MONO, size=15, color=INK),
            cliponaxis=False,
            customdata=[[r["denom"], r["partition"]] for r in rows],
            hovertemplate="<b>%{y}</b><br>%{x:.2f}% of the baseline group's approval rate<br>"
            "%{customdata[0]:,} applications (%{customdata[1]})<extra></extra>",
        )
    )

    fig.add_shape(
        type="line",
        x0=threshold * 100,
        x1=threshold * 100,
        y0=-0.55,
        y1=len(rows) - 0.4,
        line=dict(color=FLAGGED, width=3, dash="dash"),
    )
    fig.add_annotation(
        x=threshold * 100,
        y=-0.72,
        text="<b>80%</b>: below this line, a gap gets flagged",
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        font=dict(family=BODY, size=16, color=FLAGGED),
    )
    for r, label in zip(rows, labels):
        if not r["flag"]:
            continue
        fig.add_annotation(
            x=r["ratio"] * 100,
            y=label,
            text=f"stops short \u00b7 {r['denom']:,} applications",
            showarrow=False,
            xanchor="left",
            xshift=48,
            font=dict(family=BODY, size=15, color=FLAGGED),
        )

    fig.update_layout(
        height=920,
        showlegend=False,
        margin=dict(l=16, r=64, t=178, b=76),
        bargap=0.34,
        xaxis=dict(
            range=[0, 118],
            tickvals=[0, 20, 40, 60, 80, 100],
            ticktext=["0", "20%", "40%", "60%", "80%", "100%"],
            title="Approval rate as a share of joint applicants' approval rate",
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(family=BODY, size=15, color=INK),
        ),
    )
    _headline(
        fig,
        "Only 2 of 18 groups fall below the 80 percent line",
        "Regulators flag a gap for review when one group is approved at less than 80 percent "
        "of a baseline<br>group's rate. Every group below is measured against joint applicants, "
        "across all 36.7 million US<br>mortgage applications filed from 2023 to 2025. Counted "
        "from the full file, not a sample. Three of the eighteen rows are the<br>baseline compared with itself, drawn hollow. A flag is a signal to look closer, not a verdict.",
    )
    return fig


def denial_rates(data: dict) -> go.Figure:
    """Denial rate by group, full file, with the headline group picked out."""
    rows = sorted(data["rows"], key=lambda r: r["denials"] / r["applications"])
    rates = [r["denials"] / r["applications"] * 100 for r in rows]
    labels = [r["group"] for r in rows]
    ref = data["reference_group"]
    highlight = "Black or African American"

    colors = []
    for lab in labels:
        if lab == highlight:
            colors.append(FLAGGED)
        elif lab == ref:
            colors.append(MUTED)
        else:
            colors.append(RECESSIVE)

    fig = go.Figure(
        go.Bar(
            x=rates,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            width=0.62,
            text=[f"<b>{v:.1f}%</b>" for v in rates],
            textposition="outside",
            textfont=dict(family=MONO, size=16, color=INK),
            cliponaxis=False,
            customdata=[[r["applications"], r["denials"]] for r in rows],
            hovertemplate="<b>%{y}</b><br>%{x:.2f}% turned down<br>"
            "%{customdata[1]:,} denied of %{customdata[0]:,} applications<extra></extra>",
        )
    )
    fig.add_annotation(
        x=rates[0],
        y=labels[0],
        text="the comparison group above",
        showarrow=False,
        xanchor="left",
        xshift=64,
        font=dict(family=BODY, size=14, color=MUTED),
    )
    fig.add_annotation(
        x=rates[labels.index(highlight)],
        y=highlight,
        text="about 1 in 4 turned down",
        showarrow=False,
        xanchor="left",
        xshift=70,
        font=dict(family=BODY, size=15, color=FLAGGED),
    )
    fig.add_annotation(
        x=rates[-1],
        y=labels[-1],
        text="only 9,032 applications",
        showarrow=False,
        xanchor="left",
        xshift=70,
        font=dict(family=BODY, size=14, color=MUTED),
    )
    fig.update_layout(
        height=620,
        showlegend=False,
        margin=dict(l=16, r=72, t=190, b=56),
        xaxis=dict(range=[0, 53], ticksuffix="%", title="Share of applications turned down"),
        bargap=0.38,
    )
    _headline(
        fig,
        "About 1 in 4 Black applicants were turned down.<br>For joint applicants it was about 1 in 6",
        "Share of applications denied, 2023 to 2025. Counted from every one of the "
        "32.6 million<br>applications in the file, not a sample. These are raw rates: they do "
        "not yet account<br>for income, loan size or debt.",
    )
    return fig


def controlled_gap(data: dict) -> go.Figure:
    """Before and after the statistical adjustment. One pair, one slope."""
    raw = data["raw_gap_pp"]
    ctrl = data["controlled_gap_pp"]
    share_left = ctrl / raw * 100

    fig = go.Figure()
    fig.add_shape(
        type="line",
        x0=ctrl,
        x1=raw,
        y0=0,
        y1=0,
        line=dict(color=RECESSIVE, width=10),
        layer="below",
    )
    for value, color, label, pos in (
        (raw, FLAGGED, "Before the adjustment", "top center"),
        (ctrl, CAUTION, "After the adjustment", "bottom center"),
    ):
        fig.add_trace(
            go.Scatter(
                x=[value],
                y=[0],
                mode="markers+text",
                marker=dict(size=28, color=color, line=dict(color=SURFACE, width=2)),
                text=[
                    f"<b>{value:.1f} points higher</b><br>"
                    f"<span style='font-size:14px;color:{MUTED}'>{label}</span>"
                ],
                textposition=pos,
                textfont=dict(family=BODY, size=17, color=INK),
                name=label,
                hovertemplate=f"{label}: {value:.2f} percentage points<extra></extra>",
            )
        )
    fig.add_annotation(
        x=(raw + ctrl) / 2,
        y=0.8,
        text=f"<b>{share_left:.0f}% of the gap is still there</b>",
        showarrow=False,
        font=dict(family=BODY, size=16, color=MUTED),
    )
    fig.update_layout(
        height=460,
        showlegend=False,
        margin=dict(l=16, r=48, t=214, b=76),
        xaxis=dict(
            range=[6.2, 14.6],
            ticksuffix=" pts",
            title="How much more often this group was turned down than joint applicants",
        ),
        yaxis=dict(visible=False, range=[-2.1, 2.1]),
    )
    _headline(
        fig,
        "Comparing like with like explains a third of the gap.<br>Two thirds of it is left over",
        "Black or African American applicants were turned down 12.4 percentage points more "
        "often than joint applicants.<br>Comparing only applications that look alike on income, "
        "how much of the home's value is being borrowed,<br>debt measured against income, what "
        "the loan is for, and whether it is a first or a second mortgage,<br>narrows that to 8.2 "
        "points. Estimated from a 200,000-application sample, not the full file. What is left<br>"
        "over is unexplained, which is not the same thing as discrimination.",
    )
    return fig


def engineering_pair(data: dict, which: str) -> go.Figure:
    """One measure, two tools. Never two scales on one axis."""
    if which == "speed":
        duck, pand = data["speed"]["duckdb_s"], data["speed"]["pandas_s"]
        fmt = "{:.2f} seconds"
        factor = data["speed"]["factor"].rstrip("x")
        title = f"{factor} times faster"
        subtitle = (
            "The same query over all 36,734,685 rows, on one laptop.<br>"
            "Both tools produced identical tables."
        )
        sliver = "DuckDB's bar is drawn to scale. It is just very short."
    else:
        duck, pand = data["memory"]["duckdb_gib"], data["memory"]["pandas_gib"]
        fmt = "{:.2f} GiB"
        factor = data["memory"]["factor"].rstrip("x")
        title = f"{factor} times less memory"
        subtitle = (
            "Peak memory for the same job. 10.5 GiB is more<br>"
            "than many laptops can spare. 0.09 GiB is not."
        )
        sliver = "Same scale, same story."

    fig = go.Figure(
        go.Bar(
            x=[duck, pand],
            y=["DuckDB", "pandas"],
            orientation="h",
            marker=dict(color=[PRIMARY, RECESSIVE], line=dict(width=0)),
            width=0.5,
            text=[f"<b>{fmt.format(duck)}</b>", f"<b>{fmt.format(pand)}</b>"],
            textposition="outside",
            textfont=dict(family=MONO, size=17, color=INK),
            cliponaxis=False,
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    fig.add_annotation(
        x=0,
        y=-0.85,
        text=sliver,
        showarrow=False,
        xanchor="left",
        font=dict(family=BODY, size=14, color=MUTED),
    )
    fig.update_layout(
        height=296,
        showlegend=False,
        margin=dict(l=16, r=120, t=134, b=24),
        xaxis=dict(range=[0, pand * 1.32], showticklabels=False, ticks="", title=""),
        yaxis=dict(
            tickfont=dict(family=MONO, size=16, color=INK),
            range=[-1.1, 1.6],
        ),
        bargap=0.42,
    )
    _headline(fig, title, subtitle, size=24)
    return fig
