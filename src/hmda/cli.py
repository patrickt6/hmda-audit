"""The `hmda` CLI.

Every subcommand and flag is registered here, so the CLI surface stays
fixed even where a command body still prints "not implemented" and exits
1. This lets other modules be written and tested against a stable CLI
without waiting on every command to be finished.

The M1-M14 metric table lines up with the subcommands below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click


def _not_implemented(command: str) -> None:
    click.echo(f"hmda {command}: not implemented", err=True)
    sys.exit(1)


@click.group()
@click.version_option(package_name="hmda-audit")
def main() -> None:
    """hmda: national-scale mortgage lending-fairness and model-governance audit."""


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------
@main.command()
@click.option("--year", type=int, help="HMDA activity year to ingest, e.g. 2023.")
@click.option("--state", type=str, default=None, help="Two-letter state code; omit for all states.")
def ingest(year: int | None, state: str | None) -> None:
    """Download, checksum, and load one year (optionally one state) of HMDA LAR data."""
    from hmda.ingest.download import all_states, fetch_state_year
    from hmda.ingest.checksum import (
        is_state_year_complete,
        load_state_manifest,
        record_completed_state_year,
    )
    from hmda.ingest.load_duckdb import csv_to_parquet, load_parquet_dir

    if year is None:
        click.echo("hmda ingest: --year is required", err=True)
        sys.exit(1)

    states = [state] if state else all_states()
    raw_dir = Path("data/raw")
    parquet_dir = Path("data/parquet")
    parquet_dir.mkdir(parents=True, exist_ok=True)

    for st in states:
        parquet_path = parquet_dir / f"{st}_{year}.parquet"
        if is_state_year_complete(year, st):
            manifest = load_state_manifest()
            rec = manifest[f"{year}:{st}"]
            click.echo(
                f"hmda ingest: {st} {year} already complete (skipped download). "
                f"url={rec['url']} sha256={rec['sha256']}"
            )
            continue

        result = fetch_state_year(year, st, raw_dir)
        row_count = csv_to_parquet(result.csv_path, parquet_path)
        result.csv_path.unlink()  # streaming requirement: delete CSV immediately

        record_completed_state_year(
            year=year,
            state=st,
            url=result.url,
            sha256=result.sha256,
            size_bytes=result.size_bytes,
            elapsed_seconds=result.elapsed_seconds,
            row_count=row_count,
            parquet_path=parquet_path,
        )
        click.echo(
            f"hmda ingest: {st} {year} url={result.url} sha256={result.sha256} "
            f"rows={row_count} size_bytes={result.size_bytes} "
            f"elapsed_s={result.elapsed_seconds:.2f}"
        )

    load_parquet_dir(parquet_dir)
    click.echo("hmda ingest: loaded into data/hmda.duckdb")


# --------------------------------------------------------------------------
# shared: source selection for the model-based commands
# --------------------------------------------------------------------------
_SOURCE_OPTIONS = [
    click.option(
        "--source",
        type=click.Choice(["fixture", "national"]),
        default="fixture",
        show_default=True,
        help="Dataset to run on. 'fixture' is 50k rows, DC/WY/VT only.",
    ),
    click.option("--sample-n", type=int, default=500_000, show_default=True,
                 help="Rows to draw when --source national. These paths train models and cannot stream."),
    click.option("--sample-seed", type=int, default=0, show_default=True,
                 help="Seed for the stratified draw, so a run is reproducible."),
]


def source_options(fn):
    """Attach --source/--sample-n/--sample-seed to a model-based command."""
    for opt in reversed(_SOURCE_OPTIONS):
        fn = opt(fn)
    return fn


def resolve_model_source(source: str, sample_n: int, sample_seed: int):
    """Return (frame, source_label, seed) for the model-path main() functions.

    Returns (None, None, None) for the fixture, which is those functions'
    unchanged default. For national it draws a stratified sample, because
    lightgbm and shap need materialized arrays and cannot read a DuckDB view.

    The label is never omitted: supplying a frame without one raises in all
    four modules, so a sample number can never print under a fixture label.
    """
    if source == "fixture":
        return None, None, None
    from hmda.clean.source import open_source

    src = open_source(source)
    frame = src.sample(sample_n, seed=sample_seed)
    label = f"{src.label} | drew {len(frame):,} of {src.rows:,} rows, seed {sample_seed}"
    return frame, label, sample_seed


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------
@main.command()
@click.option("--all-years", is_flag=True, help="Run the full audit across every ingested year (M2).")
@click.option("--waterfall", is_flag=True, help="Print the row-count waterfall.")
@click.option("--air", is_flag=True, help="Screen lenders against the four-fifths adverse-impact rule (M5).")
@click.option("--min-count", type=int, default=100, show_default=True, help="Minimum group denominator before a ratio is reported (M5).")
@click.option("--controlled", is_flag=True, help="Print the raw and controlled denial-rate gap side by side (M6).")
@click.option("--drift", is_flag=True, help="Print year-over-year input drift as a column count, not a raw PSI.")
@click.option(
    "--source",
    type=click.Choice(["fixture", "national"]),
    default="fixture",
    show_default=True,
    help="Which dataset to audit. 'fixture' is 50k rows, DC/WY/VT only; 'national' is the full file.",
)
@click.option(
    "--sample-rows",
    type=int,
    default=200_000,
    show_default=True,
    help="Row budget for the pandas-bound branches (--waterfall, --controlled) when --source national.",
)
def audit(
    all_years: bool,
    waterfall: bool,
    air: bool,
    min_count: int,
    controlled: bool,
    drift: bool,
    source: str,
    sample_rows: int,
) -> None:
    """Run fairness audit computations: waterfall, four-fifths screen, controlled disparity, drift.

    Every disparity branch prints the "Race Not Available" share beside its
    numbers. That category is a large share of applications, so a
    disparity computed over the remainder and presented as if it covered the
    file is an overstatement.
    """
    from hmda.clean.source import open_source

    if not any([waterfall, air, controlled, drift, all_years]):
        raise click.UsageError("Pick one of --waterfall, --air, --controlled, --drift, --all-years.")

    if all_years:
        # M2: a full pass over every ingested year, timed. `--all-years` is a
        # shorthand that turns on all four real audit branches below
        # (waterfall, air, controlled, drift) rather than running a separate,
        # narrower code path of its own.
        #
        # Bug history, recorded so it is not silently reintroduced: this flag
        # used to run nothing at all while still printing a SOURCE line (a
        # pure no-op, 0.09s wall time), then
        # was patched to run a hand-rolled subset (rows-per-year + AIR only,
        # skipping waterfall/controlled/drift entirely) that still did not
        # match its own docstring ("waterfall, four-fifths screen, controlled
        # disparity, drift"). Forcing the four flags true and falling through
        # to the real per-branch code below is what actually runs all four.
        waterfall = air = controlled = drift = True

    if drift:
        from hmda.governance.drift import main as drift_main

        drift_main()
        if not any([waterfall, air, controlled, all_years]):
            return

    src = open_source(source)
    click.echo(src.label)
    if source == "fixture":
        click.echo("These are FIXTURE numbers, not national. Do not quote them as national.")

    # The aggregation branches run as SQL and stream; the pandas-bound branches
    # cannot, so on national they take a sample and say so.
    #
    streaming_frame = src.frame()
    pandas_frame = None

    def _pandas_frame():
        """Materialize the pandas-bound branches' input, once, and announce it."""
        nonlocal pandas_frame
        if pandas_frame is None:
            if source == "national":
                pandas_frame = src.sample(sample_rows, seed=0)
                click.echo(
                    f"\nPANDAS-BOUND BRANCH: computed on a stratified sample of "
                    f"{len(pandas_frame):,} rows (seed 0), not all {src.rows:,}. "
                    "Do not quote these as full-file counts."
                )
            else:
                pandas_frame = src.frame()
        return pandas_frame

    def _rna_share() -> float:
        """Race Not Available share, computed against the FULL source either way.

        This must appear beside every disparity number. Taking it
        from a sample would understate or overstate the very caveat it exists
        to carry, so national computes it in SQL over every row.
        """
        if source == "national":
            import duckdb

            from hmda.clean.source import register_frame

            con = duckdb.connect()
            try:
                register_frame(con, src.frame())
                total, rna = con.execute(
                    "SELECT count(*), "
                    "count(*) FILTER (WHERE derived_race = 'Race Not Available') "
                    "FROM frame"
                ).fetchone()
            finally:
                con.close()
            return 0.0 if total == 0 else rna / total
        from hmda.clean.waterfall import race_not_available_share

        return race_not_available_share(src.frame())

    if waterfall:
        from hmda.clean.filters import FILTER_REGISTRY
        from hmda.clean.waterfall import build_waterfall

        wf = _pandas_frame()
        click.echo(f"\nRace Not Available share: {_rna_share():.2%}")
        click.echo(f"{'step':<28}{'rows_before':>13}{'rows_after':>12}{'dropped':>10}")
        for row in build_waterfall(wf, list(FILTER_REGISTRY)):
            click.echo(f"{row.step:<28}{row.rows_before:>13,}{row.rows_after:>12,}{row.rows_dropped:>10,}")

    if air:
        from hmda.fairness.air import adverse_impact_ratio, flagged_lenders

        click.echo(f"\nRace Not Available share: {_rna_share():.2%}")
        click.echo(f"Four-fifths screen, min-count {min_count}.")
        click.echo("A flag is a statistical disparity that warrants review, NEVER a finding of discrimination.")
        click.echo(f"\n{'group':<42}{'reference':<20}{'ratio':>11}{'denom':>14}{'flag':>6}")
        for column in ("derived_race", "derived_ethnicity", "derived_sex"):
            for row in adverse_impact_ratio(streaming_frame, column, min_count=min_count):
                ratio = row.ratio if isinstance(row.ratio, str) else f"{row.ratio:.4f}"
                flag = "YES" if row.flagged else ""
                click.echo(
                    f"{str(row.group_value)[:41]:<42}{str(row.reference_group_value)[:19]:<20}"
                    f"{ratio:>11}{row.denominator:>14,}{flag:>6}"
                )
        flagged = flagged_lenders(streaming_frame, min_count=min_count)
        click.echo(f"\nLenders flagged for review: {len(flagged)} (identifiers withheld by policy: no lender is named).")

    if controlled:
        from hmda.fairness.controlled import controlled_disparity

        full_rna = _rna_share()
        for cmp in controlled_disparity(_pandas_frame(), "derived_race", min_count=min_count):
            click.echo(f"\ncontrols used: {', '.join(cmp.controls_used)}")
            click.echo(f"Race Not Available share (full file): {full_rna:.2%}")
            if source == "national":
                # controlled_disparity computes its own share from the rows it was
                # handed, which is the SAMPLE. Printing that under the same label as
                # the full-file share gave two different numbers one name.
                click.echo(
                    f"Race Not Available share (this sample): {cmp.race_not_available_share:.2%}"
                )
            click.echo(f"{cmp.group_value} vs {cmp.reference_group_value}:")
            click.echo(f"  raw gap        {cmp.raw_gap_pp:+.2f} pp")
            click.echo(f"  controlled gap {cmp.controlled_gap_pp:+.2f} pp")
        click.echo("\nThe controlled gap is unexplained variation under a stated model. It is not evidence of discrimination.")


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
@main.command()
@click.option("--eval", "eval_", is_flag=True, help="Print baseline and challenger ranking quality on the same line (M7).")
@source_options
def model(eval_: bool, source: str, sample_n: int, sample_seed: int) -> None:
    """Train and evaluate the baseline and challenger denial-prediction models."""
    if eval_:
        from hmda.model import evaluate as _evaluate

        frame, label, seed = resolve_model_source(source, sample_n, sample_seed)
        sys.exit(_evaluate.main(frame=frame, source_label=label, sample_seed=seed))
    _not_implemented("model")


# --------------------------------------------------------------------------
# mitigate
# --------------------------------------------------------------------------
@main.command()
@click.option("--compare", is_flag=True, help="Print disparity cut, accuracy change and margin change per technique, one line each (M11).")
@click.option("--profit", is_flag=True, help="Print expected margin per 1,000 applications, with assumptions in the header (M12).")
@source_options
def mitigate(compare: bool, profit: bool, source: str, sample_n: int, sample_seed: int) -> None:
    """Run the three offline mitigation techniques (reweighing, fairness-constrained GBM, per-group thresholds).

    OFFLINE COUNTERFACTUAL STUDY ONLY. Nothing here scores a live applicant,
    ever.
    """
    from hmda.fairness import mitigate as _mitigate

    frame, label, seed = resolve_model_source(source, sample_n, sample_seed)
    sys.exit(_mitigate.main(frame=frame, source_label=label, sample_seed=seed))


# --------------------------------------------------------------------------
# recourse
# --------------------------------------------------------------------------
@main.command()
@click.option("--by-group", is_flag=True, help="Print the recourse (minimum actionable change) figure per protected group (M13).")
@source_options
def recourse(by_group: bool, source: str, sample_n: int, sample_seed: int) -> None:
    """Compute the recourse/effort study: minimum actionable-feature change to cross the approval line.

    M13 was measured and found UNMEASURABLE: the per-group medians are not
    stable under resampling (the ordering between groups reverses with the
    seed). The command prints the finding AND the verdict. Do not quote the
    medians. See docs/RECOURSE.md.
    """
    from hmda.fairness import recourse as _recourse

    frame, label, seed = resolve_model_source(source, sample_n, sample_seed)
    sys.exit(_recourse.main(frame=frame, source_label=label, sample_seed=seed))


# --------------------------------------------------------------------------
# explain
# --------------------------------------------------------------------------
@main.command()
@click.option("--report", "report_", is_flag=True, help="Run SHAP and print the sample size and top drivers (M14).")
@source_options
def explain(report_: bool, source: str, sample_n: int, sample_seed: int) -> None:
    """Generate the SHAP explainability pack."""
    from hmda.governance import explain as _explain

    frame, label, seed = resolve_model_source(source, sample_n, sample_seed)
    sys.exit(_explain.main(frame=frame, source_label=label, sample_seed=seed))


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
@main.command()
@click.option("--model-card", "model_card", is_flag=True, help="Regenerate docs/MODEL-CARD.md from the database.")
@click.option("--figures", is_flag=True, help="Write the three static SVG figures (disparity, four-fifths, frontier) to out/figures/.")
@click.option("--frontier", is_flag=True, help="Write just the trade-off frontier SVG to out/figures/frontier.svg.")
@click.option("--min-count", type=int, default=100, show_default=True, help="Minimum group denominator before a four-fifths ratio is plotted (M5, same floor as `audit --air`).")
@source_options
def report(model_card: bool, figures: bool, frontier: bool, min_count: int, source: str, sample_n: int, sample_seed: int) -> None:
    """Render the HTML report (and, with flags, individual artifacts) from the database.

    Fix, 2026-09-14: a bare `hmda report` used to call `_not_implemented`
    unconditionally, so `make reproduce`'s report step failed deterministically
    no matter what the first two steps produced. Following the same
    convention `audit --all-years` uses (the covering flag turns on the
    specific ones and falls through to real code): a bare `report` now
    defaults to running every artifact that is actually implemented
    today.

    Fix, 2026-09-14: `--figures` and `--frontier` are wired to the real
    figure-generation code (`hmda.report.figures`,
    `hmda.report.frontier`) instead of `_not_implemented`. This replaces
    an earlier standalone script that duplicated this data
    gathering outside the CLI (some of it via a `subprocess` call parsing
    `hmda audit --air`'s stdout, some of it hard-coded from a markdown doc)
    so a figure's source could silently drift from the number it plots.
    `--figures` writes all three figures the current data supports
    (disparity, four-fifths, frontier); the fourth, recourse, stays a
    stub because M13 is UNMEASURABLE nationally, see docs/RECOURSE.md,
    and shipping without it is accepted.

    `--frontier` writes the SAME file as the frontier slot inside
    `--figures` (`out/figures/frontier.svg`), through the same
    `hmda.report.frontier.build_frontier` -> `render_frontier_svg` ->
    `hmda.report.figures.render_frontier_figure` path, see
    `report/frontier.py`'s module docstring. It exists as its own flag so
    `hmda verify --lineage` can regenerate just that one artifact
    without re-running the other two.
    """
    if not (model_card or figures or frontier):
        model_card = True

    ran_something = False
    if model_card:
        from hmda.governance.model_card import render_main

        code = render_main()
        if code != 0:
            sys.exit(code)
        ran_something = True

    if figures or frontier:
        from hmda.clean.source import open_source
        from hmda.fairness import mitigate as _mitigate
        from hmda.fairness.air import adverse_impact_ratio
        from hmda.fairness.rates import denial_rate_by_group
        from hmda.report import figures as _figures
        from hmda.report.frontier import build_frontier_range, render_frontier_svg

        out_dir = _figures.FIGURES_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        src = open_source(source)
        click.echo(src.label)
        streaming_frame = src.frame()

        if figures:
            group_rows = denial_rate_by_group(streaming_frame, "derived_race")
            group_rates = [(r.group_value, r.applications, r.denial_rate * 100.0) for r in group_rows]
            import duckdb

            from hmda.clean.source import register_frame

            con = duckdb.connect()
            try:
                register_frame(con, streaming_frame)
                total, rna = con.execute(
                    "SELECT count(*), count(*) FILTER (WHERE derived_race = 'Race Not Available') FROM frame"
                ).fetchone()
            finally:
                con.close()
            rna_share = 0.0 if total == 0 else rna / total

            p1 = _figures.render_disparity_figure(group_rates, rna_share, out_dir=out_dir)
            click.echo(f"wrote {p1}")

            air_rows = []
            for column in ("derived_race", "derived_ethnicity", "derived_sex"):
                for row in adverse_impact_ratio(streaming_frame, column, min_count=min_count):
                    if isinstance(row.ratio, str):
                        continue  # SUPPRESSED (below the floor) or UNDEFINED (0/0): never plotted
                    air_rows.append((f"{row.group_value} vs {row.reference_group_value}", row.ratio, row.flagged))
            p2 = _figures.render_four_fifths_figure(air_rows, threshold=0.80, out_dir=out_dir)
            click.echo(f"wrote {p2}")

        if figures or frontier:
            # M11 is quoted as a range across three independent
            # 1.5-million-row national samples, not a single run, and
            # not a single point taken from one `--sample-seed` run.
            #
            # So this always runs it at all three seeds, 0, 1, 2 (the
            # runs recorded in results/metrics_ledger.json's M11
            # entry), at n=1,500,000, and plots each technique's point
            # as a min-max range across the three runs, not one seed's
            # single point. `--sample-n`/`--sample-seed` are ignored
            # here (frontier always uses the full 3-seed/1.5M protocol)
            # so the figure cannot silently drift to the unstable
            # 500k default or to a single seed.
            frontier_seeds = (0, 1, 2)
            frontier_sample_n = 1_500_000
            results_by_seed = []
            for frontier_seed in frontier_seeds:
                model_frame, label, seed = resolve_model_source(
                    source, frontier_sample_n, frontier_seed
                )
                click.echo(f"frontier: running mitigate --compare at seed {seed} ({label})")
                _, _, mitigation_results = _mitigate.compare_all(
                    frame=model_frame, source_label=label, sample_seed=seed
                )
                results_by_seed.append(mitigation_results)
            points = build_frontier_range(results_by_seed)
            p3 = render_frontier_svg(points, out_dir / "frontier.svg")
            click.echo(f"wrote {p3}")

        ran_something = True

    if not ran_something:
        _not_implemented("report")

    sys.exit(0)


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------
@main.command()
@click.option("--counts", is_flag=True, help="Print row count per year and distinct lei count, read from DuckDB (M1).")
@click.option("--governance", is_flag=True, help="Verify every governance control names a test that exists (M10).")
@click.option("--card", is_flag=True, help="Verify every number in docs/MODEL-CARD.md exists in the database.")
@click.option("--lineage", is_flag=True, help="Verify every report figure/table has a recorded source.")
def verify(counts: bool, governance: bool, card: bool, lineage: bool) -> None:
    """Verification checks: counts, governance controls, model-card numbers, report lineage."""
    if counts:
        from hmda.ingest.load_duckdb import counts_by_year, distinct_lei_count

        db_path = Path("data/hmda.duckdb")
        if not db_path.exists():
            click.echo("hmda verify --counts: data/hmda.duckdb does not exist yet", err=True)
            sys.exit(1)

        by_year = counts_by_year(db_path)
        for yr in sorted(by_year):
            click.echo(f"year {yr}: {by_year[yr]} rows")
        click.echo(f"total rows: {sum(by_year.values())}")
        click.echo(f"distinct lei: {distinct_lei_count(db_path)}")
        return

    if governance:
        from hmda.governance.controls import main as controls_main

        sys.exit(controls_main())

    if card:
        from hmda.governance.model_card import verify_main

        sys.exit(verify_main())

    _not_implemented("verify")


if __name__ == "__main__":
    main()
