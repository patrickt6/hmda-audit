"""Static SVG figure generation, matplotlib to SVG, no interactivity.

Four figures: the disparity bars, the four-fifths screen, the
trade-off frontier, and the recourse gap, written to
``out/figures/``. Axis labels are plain English; no lender name ever
appears anywhere on any figure.

Implemented 2026-09-14 to replace the ``NotImplementedError`` stubs that
an earlier standalone script (never wired to the CLI, since removed)
was duplicating around. ``hmda report --figures`` / ``--frontier``
(``cli.py``) are the only callers that should reach these functions in
production; call them directly if you need a figure outside the CLI
(e.g. a notebook).
"""

from __future__ import annotations

from pathlib import Path

FIGURES_DIR = Path("out/figures")

#: Fixed random seed so re-runs are byte-identical (the reproducibility
#: check depends on this). Matplotlib's own rendering is deterministic given
#: the same input data, so this seed is not consumed directly by this
#: module; it documents the contract for any caller that jitters points.
FIGURE_RANDOM_SEED = 20260911

#: Plain-English label formatting rules shared across figures:
#: no raw DIR, SPD, EOD, AUC or R^2 value in any axis label, tick label,
#: legend entry or title.
_FORBIDDEN_LABEL_TOKENS = ("DIR", "SPD", "EOD", "AUC", "R2", "R²")


def _check_no_forbidden_tokens(*texts: str) -> None:
    for text in texts:
        upper = text.upper()
        for token in _FORBIDDEN_LABEL_TOKENS:
            if token in upper:
                raise ValueError(f"figure label {text!r} contains forbidden statistical shorthand {token!r}")


def render_disparity_figure(group_rates, race_not_available_share: float, out_dir: Path = FIGURES_DIR) -> Path:
    """Render the hero disparity-bar figure (denial rate by group, sample size on every bar).

    Supports M6. Must NOT contain a raw DIR, SPD, EOD, AUC or R² value in
    any label. The "Race Not Available" share must appear
    somewhere in the figure or its caption text.

    ``group_rates``: iterable of ``(group_value, applications, denial_rate_pct)``.
    """
    import matplotlib

    matplotlib.use("svg")
    import matplotlib.pyplot as plt

    rows = list(group_rates)
    if not rows:
        raise ValueError("render_disparity_figure: no group rows supplied")
    rows = sorted(rows, key=lambda r: r[2], reverse=True)
    labels = [str(r[0]) for r in rows]
    rates = [float(r[2]) for r in rows]
    counts = [int(r[1]) for r in rows]

    _check_no_forbidden_tokens(*labels)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(labels, rates, color="#2b6cb0")
    ax.invert_yaxis()
    ax.set_xlabel("Denial rate (%)")
    ax.set_title("Denial rate by race/ethnicity group")
    for bar, n in zip(bars, counts):
        ax.text(
            bar.get_width() + max(rates) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"n={n:,}",
            va="center",
            fontsize=8,
        )
    caption = (
        f"'Race Not Available' share of all applications: {race_not_available_share:.1%}. "
        "This is a disparity that warrants review, not proof of discrimination by any named lender."
    )
    fig.text(0.01, 0.01, caption, fontsize=8, wrap=True)
    fig.subplots_adjust(left=0.28, bottom=0.16, right=0.95)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "disparity.svg"
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path


def render_four_fifths_figure(air_rows, threshold: float, out_dir: Path = FIGURES_DIR) -> Path:
    """Render lenders on the adverse-impact ratio, the 0.80 line drawn and labelled, flagged count called out.

    Supports M5. No lender is named on the figure.

    ``air_rows``: iterable of ``(label, ratio, flagged)``, label already
    scrubbed of any lender identifier (e.g. "Black or African American vs
    White"), ratio a float (SUPPRESSED rows must be filtered by the caller
    before this function is reached).
    """
    import matplotlib

    matplotlib.use("svg")
    import matplotlib.pyplot as plt

    rows = list(air_rows)
    if not rows:
        raise ValueError("render_four_fifths_figure: no rows supplied")

    labels = [str(r[0]) for r in rows]
    ratios = [float(r[1]) for r in rows]
    flagged = [bool(r[2]) for r in rows]
    _check_no_forbidden_tokens(*labels)

    flagged_count = sum(flagged)
    colors = ["#c53030" if f else "#2b6cb0" for f in flagged]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(rows))))
    y_pos = range(len(rows))
    ax.barh(list(y_pos), ratios, color=colors)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1)
    ax.text(threshold, len(rows), f" four-fifths line ({threshold:.2f})", fontsize=8, va="bottom")
    ax.set_xlabel("Approval-rate ratio vs. reference group")
    ax.set_title(f"Four-fifths screen — {flagged_count} of {len(rows)} comparisons flagged for review")
    fig.subplots_adjust(left=0.35, bottom=0.12, right=0.95)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "four_fifths.svg"
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path


def render_frontier_figure(mitigation_results, out_dir: Path = FIGURES_DIR) -> Path:
    """Render each mitigation as a point: disparity cut vs. accuracy cost, margin cost as size/second panel.

    Supports M11, M12. The unmitigated model is labelled.
    Axis labels in plain English, e.g. "Approval-rate gap (percentage
    points)", never "SPD".

    ``mitigation_results``: iterable of dicts with keys ``label`` and an
    optional ``unmitigated`` bool, plus EITHER a single-run shape
    (``disparity_cut_pct``, ``accuracy_change_pp``,
    ``margin_change_usd_per_1000``, each float or None) OR a multi-seed range
    shape (``disparity_cut_pct_min/max``, ``accuracy_change_pp_min/max``,
    ``margin_change_usd_per_1000_min/max``). The range shape is what
    ``report.frontier.build_frontier_range`` produces from three seeds'
    ``MitigationResult`` rows: no single seed's figure may be presented
    as the headline, and it is drawn as a line
    between the min and max draw, not a single dot, for each technique. The
    single-run shape (``report.frontier.build_frontier``) is still accepted
    and draws a single point, for callers that only have one seed.
    """
    import matplotlib

    matplotlib.use("svg")
    import matplotlib.pyplot as plt

    rows = list(mitigation_results)
    if not rows:
        raise ValueError("render_frontier_figure: no points supplied")

    labels = [str(r["label"]) for r in rows]
    _check_no_forbidden_tokens(*labels)

    is_range = any("disparity_cut_pct_min" in r for r in rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Panel 1: disparity cut vs accuracy cost, one point (or one min-max line,
    # for the range shape) per technique.
    for row in rows:
        if is_range:
            cut_lo, cut_hi = row.get("disparity_cut_pct_min"), row.get("disparity_cut_pct_max")
            acc_lo, acc_hi = row.get("accuracy_change_pp_min"), row.get("accuracy_change_pp_max")
            if cut_lo is None or cut_hi is None or acc_lo is None or acc_hi is None:
                continue
            marker = "*" if row.get("unmitigated") else "o"
            size = 220 if row.get("unmitigated") else 90
            if cut_lo == cut_hi and acc_lo == acc_hi:
                ax1.scatter(acc_lo, cut_lo, s=size, marker=marker, label=row["label"])
                ax1.annotate(row["label"], (acc_lo, cut_lo), fontsize=7, xytext=(4, 4), textcoords="offset points")
            else:
                (line,) = ax1.plot([acc_lo, acc_hi], [cut_lo, cut_hi], marker="o", markersize=6, label=row["label"])
                ax1.annotate(
                    row["label"],
                    (acc_hi, cut_hi),
                    fontsize=7,
                    xytext=(4, 4),
                    textcoords="offset points",
                    color=line.get_color(),
                )
        else:
            cut = row.get("disparity_cut_pct")
            acc = row.get("accuracy_change_pp")
            if cut is None or acc is None:
                continue
            marker = "*" if row.get("unmitigated") else "o"
            size = 220 if row.get("unmitigated") else 140
            ax1.scatter(acc, cut, s=size, marker=marker, label=row["label"])
            ax1.annotate(row["label"], (acc, cut), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax1.set_xlabel("Change in agreement with historical decision (percentage points)")
    ax1.set_ylabel("Approval-rate gap cut (%)")
    title = "Fairness vs. accuracy trade-off"
    if is_range:
        title += " (range across seeds)"
    ax1.set_title(title)
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.axvline(0, color="gray", linewidth=0.5)

    # Panel 2: margin cost per technique as a bar (or a min-max range bar, for
    # the range shape), so it reads as its own channel rather than a marker
    # size a reader has to decode.
    if is_range:
        margin_rows = [
            r for r in rows
            if r.get("margin_change_usd_per_1000_min") is not None and r.get("margin_change_usd_per_1000_max") is not None
        ]
        if margin_rows:
            margin_labels = [r["label"] for r in margin_rows]
            los = [r["margin_change_usd_per_1000_min"] for r in margin_rows]
            his = [r["margin_change_usd_per_1000_max"] for r in margin_rows]
            colors = ["#2b6cb0" if lo >= 0 else "#c53030" for lo in los]
            ax2.barh(margin_labels, [hi - lo for lo, hi in zip(los, his)], left=los, color=colors)
            ax2.set_xlabel("Change in expected margin ($ per 1,000 applications)")
            ax2.set_title("Margin cost by technique (range across seeds)")
            ax2.axvline(0, color="black", linewidth=0.5)
        else:
            ax2.axis("off")
            ax2.text(0.5, 0.5, "No margin figures supplied for these techniques", ha="center", va="center", wrap=True)
    else:
        margin_labels = [r["label"] for r in rows if r.get("margin_change_usd_per_1000") is not None]
        margins = [r["margin_change_usd_per_1000"] for r in rows if r.get("margin_change_usd_per_1000") is not None]
        if margins:
            colors = ["#2b6cb0" if m >= 0 else "#c53030" for m in margins]
            ax2.barh(margin_labels, margins, color=colors)
            ax2.set_xlabel("Change in expected margin ($ per 1,000 applications)")
            ax2.set_title("Margin cost by technique")
            ax2.axvline(0, color="black", linewidth=0.5)
        else:
            ax2.axis("off")
            ax2.text(0.5, 0.5, "No margin figures supplied for these techniques", ha="center", va="center", wrap=True)

    fig.legend(*ax1.get_legend_handles_labels(), loc="lower center", ncol=min(3, len(rows)), fontsize=8)
    fig.subplots_adjust(bottom=0.22, wspace=0.4)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "frontier.svg"
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path


def render_recourse_figure(recourse_results, out_dir: Path = FIGURES_DIR) -> Path:
    """Render extra income needed to cross the approval line, by group, in dollars.

    Supports M13. M13 was reported as UNMEASURABLE, so this figure is
    skipped entirely and the page ships with three figures instead of
    four.

    Left as a stub: M13 is UNMEASURABLE on the national data (see
    docs/RECOURSE.md), so there is no measured input this
    function could plot without inventing one. Wiring this up is only in
    scope again once a stable national recourse number exists.
    """
    raise NotImplementedError
