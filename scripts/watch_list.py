"""Build the empirical-Bayes lender watch list on the national file.

    .venv/bin/python scripts/watch_list.py [--source national|fixture] [--top 25]

Writes two files:

- ``results/watch_list.json``: aggregate counts only, no lender identity in
  any public artifact. Safe to commit.
- ``out/watch_list_local.csv``: the named list, for local review. ``out/`` is
  gitignored and this file must stay out of the repo.

For the top ``--top`` watch-list lenders it also computes the controlled gap
(``controlled.controlled_disparity``) on that lender's own applications, so the
raw and controlled numbers ship together.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import duckdb

from hmda.clean.source import FIXTURE, NATIONAL, _PARQUET_GLOB, open_source
from hmda.fairness.air import FOUR_FIFTHS_THRESHOLD, _air_rows_from_raw
from hmda.fairness.controlled import DEFAULT_MIN_COUNT, controlled_disparity
from hmda.fairness.rates import run_group_query_no_params
from hmda.fairness.shrink import DEFAULT_LEVEL, DEFAULT_MIN_EACH, shrink_lender_gaps

ROOT = Path(__file__).resolve().parents[1]
GROUP, REFERENCE = "Black or African American", "White"


def _four_fifths_flagged(rows, min_count: int) -> set[str]:
    """Leis whose GROUP approval-rate ratio vs REFERENCE is under 0.8, at min_count."""
    by_lei: dict[str, list[dict]] = {}
    for r in rows:
        by_lei.setdefault(r["lei"], []).append(r)
    out = set()
    for lei, rs in by_lei.items():
        for a in _air_rows_from_raw(rs, "derived_race", lei, min_count):
            if a.group_value == GROUP and a.flagged:
                out.add(lei)
    return out


def _lender_frame(source_name: str, lei: str, fixture_frame):
    if source_name == FIXTURE:
        return fixture_frame[fixture_frame["lei"] == lei]
    con = duckdb.connect()
    try:
        return con.execute(
            "select * from read_parquet(?) where lei = ?", [str(_PARQUET_GLOB), lei]
        ).df()
    finally:
        con.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=NATIONAL, choices=[NATIONAL, FIXTURE])
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--level", type=float, default=DEFAULT_LEVEL)
    ap.add_argument("--min-each", type=int, default=DEFAULT_MIN_EACH)
    args = ap.parse_args(argv)

    src = open_source(args.source)
    print(src.label)
    rows = run_group_query_no_params("air_by_lei_group.sql", src.frame(), "derived_race")

    res = shrink_lender_gaps(rows, GROUP, REFERENCE, min_each=args.min_each, level=args.level)
    scored = {l.lei for l in res.lenders}
    ff_any = _four_fifths_flagged(rows, 1) & scored
    ff_floor = _four_fifths_flagged(rows, DEFAULT_MIN_COUNT) & scored
    watch = {l.lei for l in res.watch_list}

    fixture_frame = src.frame() if args.source == FIXTURE else None
    top = sorted(res.watch_list, key=lambda l: l.shrunk_gap_pp, reverse=True)[: args.top]
    controlled = {}
    not_estimable: list[str] = []
    for l in top:
        lf = _lender_frame(args.source, l.lei, fixture_frame)
        try:
            comps = controlled_disparity(lf, "derived_race", min_count=args.min_each,
                                         reference_group_value=REFERENCE)
        except ValueError:
            # A control with no usable value for this whole lender (for example every
            # LTV reported "Exempt") leaves the median fill empty; no controlled gap.
            not_estimable.append(l.lei)
            continue
        c = next((c for c in comps if c.group_value == GROUP), None)
        controlled[l.lei] = None if c is None else c.controlled_gap_pp

    top_controlled = [v for v in controlled.values() if v is not None]
    summary = {
        # The label holds an absolute local path; keep only the repo-relative part.
        "source": src.label.replace(str(ROOT) + "/", ""),
        "comparison": f"{GROUP} vs {REFERENCE}, denial-rate gap in percentage points, all years pooled",
        "method": "normal-normal empirical Bayes, DerSimonian-Laird moments, pooled-rate variances",
        "min_each": args.min_each,
        "level": res.level,
        "lenders_scored": res.lenders_scored,
        "typical_gap_pp": round(res.mu_pp, 2),
        "between_lender_sd_pp": round(res.tau_pp, 2),
        "four_fifths_flagged_no_floor": len(ff_any),
        "four_fifths_flagged_min_count_100": len(ff_floor),
        "watch_list": len(watch),
        "watch_list_and_four_fifths_no_floor": len(watch & ff_any),
        "four_fifths_no_floor_not_on_watch_list": len(ff_any - watch),
        "watch_list_not_four_fifths_min_count_100": len(watch - ff_floor),
        "top_n_with_controlled_gap": len(top_controlled),
        "top_n_controlled_gap_still_positive": sum(1 for v in top_controlled if v > 0),
        "top_n_controlled_not_estimable": len(not_estimable),
    }

    results = ROOT / "results" / ("watch_list.json" if args.source == NATIONAL else "watch_list_fixture.json")
    results.write_text(json.dumps(summary, indent=2) + "\n")

    local = ROOT / "out" / f"watch_list_local_{args.source}.csv"
    local.parent.mkdir(exist_ok=True)
    with local.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lei", "n_black", "n_white", "raw_gap_pp", "shrunk_gap_pp", "posterior_sd_pp",
                    "p_above_typical", "on_watch_list", "four_fifths_flag_floor100", "controlled_gap_pp"])
        for l in res.lenders:
            w.writerow([l.lei, l.n_group, l.n_reference, round(l.raw_gap_pp, 2), round(l.shrunk_gap_pp, 2),
                        round(l.posterior_sd_pp, 2), round(l.p_above_typical, 4), l.on_watch_list,
                        l.lei in ff_floor, "" if controlled.get(l.lei) is None else round(controlled[l.lei], 2)])

    print(json.dumps(summary, indent=2))
    print(f"aggregate -> {results.relative_to(ROOT)}; named list (local only) -> {local.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
