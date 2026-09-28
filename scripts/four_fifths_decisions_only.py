"""National four-fifths ratios under the decisions-only definition of approval.

    .venv/bin/python scripts/four_fifths_decisions_only.py [--source national|fixture]

The four-fifths screen (``hmda audit --air``) computes approval rate over
every record except purchased loans (``sql/air_by_group.sql``), so withdrawn,
incomplete and preapproval records count as not denied. The models keep
only lender decisions, action_taken 1, 2 and 3. This script runs the
national ratios under the second definition, with the same reference rule
and floor as the screen (``hmda.fairness.air._air_rows_from_raw``), so both
sets of national ratios come from committed files. It does not change the
screen or its lender count.

Writes ``results/four_fifths_decisions_only.json`` (national source) or
``results/four_fifths_decisions_only_fixture.json`` (fixture).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hmda.clean.source import FIXTURE, NATIONAL, open_source
from hmda.fairness.air import FOUR_FIFTHS_THRESHOLD, _air_rows_from_raw
from hmda.fairness.rates import run_group_query

ROOT = Path(__file__).resolve().parents[1]
SQL = "air_by_group_decisions_only.sql"
PARTITIONS = {"Race": "derived_race", "Ethnicity": "derived_ethnicity", "Sex": "derived_sex"}
MIN_COUNT = 100


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=NATIONAL, choices=[NATIONAL, FIXTURE])
    args = ap.parse_args(argv)

    src = open_source(args.source)
    print(src.label)
    frame = src.frame()

    rows, rows_counted = [], None
    for partition, column in PARTITIONS.items():
        raw = run_group_query(SQL, frame, column, None)
        total = sum(int(r["denominator"]) for r in raw)
        rows_counted = total if rows_counted is None else rows_counted
        if total != rows_counted:
            raise RuntimeError(f"{column}: {total} rows, expected {rows_counted}")
        by_value = {r["group_value"]: r for r in raw}
        for a in _air_rows_from_raw(raw, column, None, MIN_COUNT):
            r = by_value[a.group_value]
            rows.append({
                "partition": partition,
                "group": a.group_value,
                "decisions": int(r["denominator"]),
                "denials": int(r["denials"]),
                "approval_rate": round(float(r["approval_rate"]), 4),
                "reference_group": a.reference_group_value,
                "ratio": a.ratio if isinstance(a.ratio, str) else round(a.ratio, 4),
                "flag": a.flagged,
                **({"is_reference": True} if a.group_value == a.reference_group_value else {}),
            })

    race = [r for r in rows if r["partition"] == "Race"]
    below = [r["group"] for r in race if r["flag"]]
    black = next(r for r in race if r["group"] == "Black or African American")
    out = {
        "command": ".venv/bin/python scripts/four_fifths_decisions_only.py",
        "sql": f"sql/{SQL}",
        # The label holds an absolute local path; keep only the repo-relative part.
        "source": src.label.replace(str(ROOT) + "/", ""),
        "status": "MEASURED" if args.source == NATIONAL else "FIXTURE",
        "definition": "approval rate = 1 - denials / decisions, over action_taken 1, 2 and 3 only",
        "screen_definition": "the four-fifths screen (results/four_fifths.json) uses every record except action_taken 6",
        "threshold": FOUR_FIFTHS_THRESHOLD,
        "min_count": MIN_COUNT,
        "reference_rule": "highest approval rate among groups with at least min_count decisions, as in hmda.fairness.air",
        "action_taken_kept": [1, 2, 3],
        "rows_counted": rows_counted,
        "black_vs_reference_ratio": black["ratio"],
        "black_reference_group": black["reference_group"],
        "race_groups_below_threshold": below,
        "race_groups_below_threshold_count": len(below),
        "race_groups_below_threshold_excluding_free_form": len([g for g in below if g != "Free Form Text Only"]),
        "rows": rows,
    }
    name = "four_fifths_decisions_only.json" if args.source == NATIONAL else "four_fifths_decisions_only_fixture.json"
    path = ROOT / "results" / name
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"-> {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
