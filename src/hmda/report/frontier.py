"""The trade-off frontier figure: fairness against accuracy, margin cost as a channel.

The "Trade-off" figure. Every point on
the figure traces to a row produced by ``hmda mitigate --compare``, verified
through ``hmda verify --lineage``.

Relationship to ``hmda.report.figures``: this module does not draw anything
itself. ``build_frontier`` turns real ``MitigationResult`` rows (the only
lineage-traceable source) into plot-ready points, and ``render_frontier_svg``
hands those points to ``hmda.report.figures.render_frontier_figure``, which
owns the actual matplotlib call. So ``hmda report --frontier`` and the
frontier slot inside ``hmda report --figures`` write the SAME file
(``out/figures/frontier.svg``) through this same two-function path; they are
not two different artifacts under one name, see ``cli.py``'s ``report``
command docstring for the resolution.
"""

from __future__ import annotations

from pathlib import Path

from hmda.report.figures import render_frontier_figure

#: Plain-English labels for the TECHNIQUE_REGISTRY keys: plain
#: English on the figure, not an internal identifier.
_TECHNIQUE_LABELS = {
    "reweighing": "Reweighing",
    "fair_constrained_gbm": "Fairness-constrained GBM",
    "per_group_threshold": "Per-group threshold",
}


def build_frontier(mitigation_results) -> list[dict]:
    """Convert a list of ``hmda.fairness.mitigate.MitigationResult`` into plot-ready points.

    Each returned dict carries at minimum ``label``, ``disparity_cut_pct``,
    ``accuracy_change_pp`` and ``margin_change_usd_per_1000``, i.e. every
    field needed to draw the frontier and to trace each point back to its
    source ``MitigationResult`` for the lineage check (each dict also keeps
    the source ``technique`` name and ``stage`` for that trace). An
    unmitigated-model anchor point (0, 0, 0) is prepended so the figure can
    label it.
    """
    results = list(mitigation_results)
    if not results:
        raise ValueError("build_frontier: no MitigationResult rows supplied")

    points: list[dict] = [
        {
            "label": "Unmitigated model",
            "technique": None,
            "stage": None,
            "disparity_cut_pct": 0.0,
            "accuracy_change_pp": 0.0,
            "margin_change_usd_per_1000": 0.0,
            "unmitigated": True,
        }
    ]
    for result in results:
        try:
            cut_pct = result.disparity_cut_percent
        except ValueError:
            # before.approval_gap_pp <= 0.0: undefined percent cut. Report the
            # point with cut=None rather than fabricate a number or drop it
            # (the lineage check must still find one point per technique).
            cut_pct = None
        points.append(
            {
                "label": _TECHNIQUE_LABELS.get(result.technique, result.technique),
                "technique": result.technique,
                "stage": result.stage.value,
                "disparity_cut_pct": cut_pct,
                "accuracy_change_pp": result.accuracy_change_pp,
                "margin_change_usd_per_1000": result.margin_change_usd_per_1000,
                "unmitigated": False,
            }
        )
    return points


def build_frontier_range(results_by_seed: list[list]) -> list[dict]:
    """Convert several seeds' worth of ``MitigationResult`` lists into range points.

    ``results_by_seed`` is one list of ``MitigationResult`` per seed (the
    same technique order each time, e.g. three calls to
    ``hmda.fairness.mitigate.compare_all`` at ``sample_seed`` 0, 1, 2). No
    single seed's figure may be presented as the headline: the quotable
    form of M11 is a RANGE, not a
    point, so each returned dict spans every seed via ``*_min``/``*_max``
    pairs instead of a single run's value: ``disparity_cut_pct_min/max``,
    ``accuracy_change_pp_min/max``, ``margin_change_usd_per_1000_min/max``.
    An unmitigated-model anchor (min=max=0 on all three) is prepended, same
    as :func:`build_frontier`.
    """
    if not results_by_seed:
        raise ValueError("build_frontier_range: no seed runs supplied")
    n_techniques = len(results_by_seed[0])
    for seed_results in results_by_seed:
        if len(seed_results) != n_techniques:
            raise ValueError(
                "build_frontier_range: seeds returned different technique counts"
            )

    points: list[dict] = [
        {
            "label": "Unmitigated model",
            "technique": None,
            "stage": None,
            "disparity_cut_pct_min": 0.0,
            "disparity_cut_pct_max": 0.0,
            "accuracy_change_pp_min": 0.0,
            "accuracy_change_pp_max": 0.0,
            "margin_change_usd_per_1000_min": 0.0,
            "margin_change_usd_per_1000_max": 0.0,
            "unmitigated": True,
        }
    ]
    for i in range(n_techniques):
        per_seed = [seed_results[i] for seed_results in results_by_seed]
        technique = per_seed[0].technique
        stage = per_seed[0].stage

        cuts = []
        for result in per_seed:
            try:
                cuts.append(result.disparity_cut_percent)
            except ValueError:
                # before.approval_gap_pp <= 0.0 on this seed: undefined percent
                # cut. Excluded from the range rather than fabricated or
                # allowed to collapse the whole point.
                pass
        accs = [r.accuracy_change_pp for r in per_seed]
        margins = [r.margin_change_usd_per_1000 for r in per_seed]

        points.append(
            {
                "label": _TECHNIQUE_LABELS.get(technique, technique),
                "technique": technique,
                "stage": stage.value,
                "disparity_cut_pct_min": min(cuts) if cuts else None,
                "disparity_cut_pct_max": max(cuts) if cuts else None,
                "accuracy_change_pp_min": min(accs),
                "accuracy_change_pp_max": max(accs),
                "margin_change_usd_per_1000_min": min(margins),
                "margin_change_usd_per_1000_max": max(margins),
                "unmitigated": False,
            }
        )
    return points


def render_frontier_svg(points: list[dict], out_path: Path) -> Path:
    """Render ``points`` (from :func:`build_frontier`) to an SVG at ``out_path``.

    Delegates the actual plotting to
    ``hmda.report.figures.render_frontier_figure`` conventions (plain-English
    axis labels, unmitigated model labelled, caption states the margin
    assumptions in plain English).
    """
    out_path = Path(out_path)
    written = render_frontier_figure(points, out_dir=out_path.parent)
    if written != out_path:
        written.rename(out_path)
    return out_path
