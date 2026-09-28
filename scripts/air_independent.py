"""Recompute the per-lender four-fifths screen without the repo's SQL, then diff.

    .venv/bin/python scripts/air_independent.py [--min-count 100]

Why this exists: the Spark/DuckDB parity check (databricks/compare_national.py)
runs the SAME SQL text on two engines. It catches engine bugs, not logic bugs:
if sql/air_by_lei_group.sql had the wrong filter, both engines would agree on
the wrong answer. This script computes the same numbers a second way, from
the written definition, with pyarrow instead of SQL, and compares every row.

Honesty about independence. The author of this script read sql/*.sql,
hmda/fairness/air.py and hmda/fairness/rates.py before writing it (they were
part of the review). So the CODE is independent (no file in sql/ is read, no
function from rates.py or air.py is called in part 1), but the author is not.
A misunderstanding shared by both authors would not be caught.

The definition used, written out:

- Purchased loans (action_taken 6) are not applications and are dropped.
  Nothing else is dropped. (There are no null action_taken values in the
  national file, so whether nulls count does not change any number here.)
- Denial means action_taken 3. Approval rate = 1 - denials / applications.
- Within one lender and one protected-class column, the reference group is
  the group with the highest approval rate among the groups that clear the
  floor. This rule is the one stated in the hmda/fairness/air.py module
  docstring and in its comment above the reference choice.
- Floor: a group with fewer than MIN_COUNT applications is suppressed (kept
  in the output, marked, never given a ratio). MIN_COUNT = 100 and the rule
  "denominator < min_count means suppress" are copied by hand from
  hmda/fairness/floors.py (DEFAULT_MIN_COUNT and below_floor). floors.py was
  read only for that constant and that rule; it is not imported.
- Ratio = group approval rate / reference approval rate. Flagged means the
  ratio is below 0.8.
- If the reference approval rate is 0 (every eligible group was denied
  every time), the ratio is 0/0. This script calls it undefined and does
  not flag it. Until 2026-09-28 air.py set the ratio to 0.0 and flagged it,
  and that one rule was the whole difference between the two flagged sets
  (615 against 614). air.py now returns "UNDEFINED" and does not flag it,
  so the two sides must agree exactly. The script still counts the 0/0
  cases so they can be seen.

Part 2 runs the repo's own SQL path and compares. Part 3 checks that the
per-state and per-year totals add up to the national totals.

Output: results/independent_check.json, aggregate counts only. No lender
identifiers and no absolute paths.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PARQUET_DIR = ROOT / "data" / "parquet"
OUT = ROOT / "results" / "independent_check.json"

MIN_COUNT = 100  # copied from hmda/fairness/floors.py DEFAULT_MIN_COUNT, not imported
THRESHOLD = 0.8
COLUMNS = ("derived_race", "derived_ethnicity", "derived_sex")
TOL = 1e-12


# ---------------------------------------------------------------------------
# Part 1: the independent computation (pyarrow only, no sql/, no hmda import)
# ---------------------------------------------------------------------------

def _count_true(mask) -> int:
    """Number of True values in a boolean array (nulls count as False; empty is 0)."""
    return pc.sum(pc.cast(pc.fill_null(mask, False), "int64")).as_py() or 0


def independent_counts():
    """One pass over every parquet file, reading only the needed columns.

    Returns (per_column, totals) where per_column[col][(lei, group)] is
    [applications, denials] summed over all files, and totals holds the
    per-file, per-state-column and per-year-column application and denial
    counts for part 3.
    """
    per_column = {c: defaultdict(lambda: [0, 0]) for c in COLUMNS}
    by_file_state = defaultdict(lambda: [0, 0])
    by_file_year = defaultdict(lambda: [0, 0])
    by_state_code = defaultdict(lambda: [0, 0])
    by_activity_year = defaultdict(lambda: [0, 0])
    state_mismatch_rows = 0
    year_mismatch_rows = 0
    rows_read = 0
    files = sorted(PARQUET_DIR.glob("*.parquet"))
    need = ["lei", "action_taken", *COLUMNS, "state_code", "activity_year"]

    for path in files:
        file_state, file_year = path.stem.split("_")
        t = pq.read_table(path, columns=need)
        rows_read += t.num_rows
        not_purchased = pc.fill_null(pc.not_equal(t["action_taken"], "6"), True)
        t = t.filter(not_purchased)
        denied = pc.cast(pc.fill_null(pc.equal(t["action_taken"], "3"), False), "int64")
        t = t.append_column("denied", denied)

        n, d = t.num_rows, pc.sum(denied).as_py() or 0
        by_file_state[file_state][0] += n
        by_file_state[file_state][1] += d
        by_file_year[file_year][0] += n
        by_file_year[file_year][1] += d
        for key_col, acc in (("state_code", by_state_code), ("activity_year", by_activity_year)):
            g = t.group_by(key_col).aggregate([("denied", "count"), ("denied", "sum")])
            for k, cnt, s in zip(g[key_col].to_pylist(), g["denied_count"].to_pylist(), g["denied_sum"].to_pylist()):
                acc[k][0] += cnt
                acc[k][1] += s
        state_mismatch_rows += t.num_rows - _count_true(pc.equal(t["state_code"], file_state))
        year_mismatch_rows += t.num_rows - _count_true(pc.equal(t["activity_year"], file_year))

        with_lei = t.filter(pc.is_valid(t["lei"]))
        for col in COLUMNS:
            g = with_lei.group_by(["lei", col]).aggregate([("denied", "count"), ("denied", "sum")])
            acc = per_column[col]
            for lei, grp, cnt, s in zip(g["lei"].to_pylist(), g[col].to_pylist(),
                                        g["denied_count"].to_pylist(), g["denied_sum"].to_pylist()):
                acc[(lei, grp)][0] += cnt
                acc[(lei, grp)][1] += s

    totals = {
        "files": len(files), "rows_read": rows_read,
        "by_file_state": by_file_state, "by_file_year": by_file_year,
        "by_state_code": by_state_code, "by_activity_year": by_activity_year,
        "state_mismatch_rows": state_mismatch_rows, "year_mismatch_rows": year_mismatch_rows,
    }
    return per_column, totals


def independent_screen(counts, min_count: int):
    """Apply the four-fifths rule per lender to {(lei, group): [n, d]}.

    Returns {(lei, group): (status, ratio)} with status one of "ratio",
    "suppressed", "undefined", plus the set of flagged leis and the set of
    leis whose reference approval rate is 0 (the undefined 0/0 case).
    """
    by_lei = defaultdict(dict)
    for (lei, grp), (n, d) in counts.items():
        by_lei[lei][grp] = (n, d)

    out, flagged, undefined = {}, set(), set()
    for lei, groups in by_lei.items():
        rate = {g: 1.0 - d / n for g, (n, d) in groups.items()}
        eligible = [g for g, (n, _) in groups.items() if not n < min_count]
        ref = max((rate[g] for g in eligible), default=None)
        for g, (n, _) in groups.items():
            if n < min_count or ref is None:
                out[(lei, g)] = ("suppressed", None)
            elif ref == 0.0:
                out[(lei, g)] = ("undefined", None)
                undefined.add(lei)
            else:
                r = rate[g] / ref
                out[(lei, g)] = ("ratio", r)
                if r < THRESHOLD:
                    flagged.add(lei)
    return out, flagged, undefined


# ---------------------------------------------------------------------------
# Part 2: the repo's SQL path, and the row-by-row diff
# ---------------------------------------------------------------------------

def sql_path(min_count: int):
    sys.path.insert(0, str(ROOT / "src"))
    from hmda.clean.source import NATIONAL, open_source
    from hmda.fairness.air import _air_rows_from_raw, flagged_lenders
    from hmda.fairness.rates import run_group_query_no_params

    frame = open_source(NATIONAL).frame()
    rows = run_group_query_no_params("air_by_lei_group.sql", frame, "derived_race")
    by_lei = defaultdict(list)
    for r in rows:
        by_lei[r["lei"]].append(r)
    screen, race_flagged = {}, set()
    for lei, rs in by_lei.items():
        for a in _air_rows_from_raw(rs, "derived_race", lei, min_count):
            screen[(lei, a.group_value)] = a
            if a.flagged:
                race_flagged.add(lei)
    t0 = time.time()
    all_flagged = set(flagged_lenders(frame, min_count))
    return rows, screen, race_flagged, all_flagged, time.time() - t0


def diff_rows(sql_rows, ind_counts):
    sql = {(r["lei"], r["group_value"]): r for r in sql_rows}
    keys = sql.keys() & ind_counts.keys()
    mism = {"applications": 0, "denials": 0, "approval_rate": 0}
    max_rate_diff = 0.0
    for k in keys:
        n, d = ind_counts[k]
        s = sql[k]
        mism["applications"] += int(s["denominator"]) != n
        mism["denials"] += int(s["denials"]) != d
        diff = abs(float(s["approval_rate"]) - (1.0 - d / n))
        max_rate_diff = max(max_rate_diff, diff)
        mism["approval_rate"] += diff > TOL
    return {
        "rows_sql": len(sql), "rows_independent": len(ind_counts), "rows_compared": len(keys),
        "only_in_sql": len(sql.keys() - ind_counts.keys()),
        "only_in_independent": len(ind_counts.keys() - sql.keys()),
        "mismatches": mism, "max_abs_approval_rate_diff": max_rate_diff,
    }


def diff_screen(sql_screen, ind_screen):
    keys = sql_screen.keys() & ind_screen.keys()
    status, ratio, undefined_rows, max_ratio_diff = 0, 0, 0, 0.0
    for k in keys:
        a = sql_screen[k]
        st, r = ind_screen[k]
        sql_status = {"SUPPRESSED": "suppressed", "UNDEFINED": "undefined"}.get(a.ratio, "ratio") \
            if isinstance(a.ratio, str) else "ratio"
        if sql_status != st:
            status += 1
        elif st == "undefined":
            undefined_rows += 1
        elif st == "ratio":
            dd = abs(float(a.ratio) - r)
            max_ratio_diff = max(max_ratio_diff, dd)
            ratio += dd > TOL
    return {"rows_compared": len(keys), "status_mismatches": status,
            "ratio_mismatches": ratio, "max_abs_ratio_diff": max_ratio_diff,
            "rows_undefined_on_both_sides": undefined_rows}


def set_cmp(a: set, b: set, undefined_leis: set) -> dict:
    """Compare two flagged sets. ``explained_by_undefined_ratio`` counts the
    lenders in the symmetric difference that are exactly the 0/0 lenders."""
    sym = a ^ b
    return {"sql": len(a), "independent": len(b), "both": len(a & b),
            "symmetric_difference": len(sym),
            "explained_by_undefined_ratio": len(sym & undefined_leis),
            "unexplained": len(sym - undefined_leis)}


# ---------------------------------------------------------------------------
# Part 3: totals add up
# ---------------------------------------------------------------------------

def totals_checks(totals, sql_rows):
    nat_n = sum(int(r["denominator"]) for r in sql_rows)
    nat_d = sum(int(r["denials"]) for r in sql_rows)

    def sums(acc):
        return sum(v[0] for v in acc.values()), sum(v[1] for v in acc.values())

    checks = {}
    for name in ("by_file_state", "by_file_year", "by_state_code", "by_activity_year"):
        n, d = sums(totals[name])
        checks[f"{name}_sums_to_national"] = {
            "groups": len(totals[name]), "applications": n, "denials": d,
            "pass": n == nat_n and d == nat_d}
    checks["national_from_sql"] = {"applications": nat_n, "denials": nat_d}
    checks["file_state_equals_state_code_column"] = {
        "rows_disagreeing": totals["state_mismatch_rows"], "pass": totals["state_mismatch_rows"] == 0}
    checks["file_year_equals_activity_year_column"] = {
        "rows_disagreeing": totals["year_mismatch_rows"], "pass": totals["year_mismatch_rows"] == 0}
    return checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=MIN_COUNT)
    args = ap.parse_args(argv)
    t_start = time.time()

    t0 = time.time()
    per_column, totals = independent_counts()
    screens = {c: independent_screen(per_column[c], args.min_count) for c in COLUMNS}
    t_ind = time.time() - t0

    t0 = time.time()
    sql_rows, sql_screen, sql_race_flagged, sql_all_flagged, t_flagged = sql_path(args.min_count)
    t_sql = time.time() - t0

    ind_race_screen, ind_race_flagged, race_undefined = screens["derived_race"]
    ind_all_flagged = set().union(*(screens[c][1] for c in COLUMNS))
    all_undefined = set().union(*(screens[c][2] for c in COLUMNS))
    checks = totals_checks(totals, sql_rows)
    counts_diff = diff_rows(sql_rows, per_column["derived_race"])
    screen_diff = diff_screen(sql_screen, ind_race_screen)

    summary = {
        "command": ".venv/bin/python scripts/air_independent.py",
        "source": "data/parquet/*.parquet (national, raw), read column-projected with pyarrow",
        "status": "MEASURED",
        "min_count": args.min_count,
        "threshold": THRESHOLD,
        "files_read": totals["files"],
        "rows_in_files": totals["rows_read"],
        "group_column_compared": "derived_race",
        "counts": counts_diff,
        "screen": screen_diff,
        "flagged_race_only": set_cmp(sql_race_flagged, ind_race_flagged, race_undefined),
        "flagged_all_three_columns": set_cmp(sql_all_flagged, ind_all_flagged, all_undefined),
        "undefined_ratio_lenders_by_column": {c: len(screens[c][2]) for c in COLUMNS},
        "undefined_ratio_note": ("A lender whose floor-clearing groups were all denied every time has a "
                                 "reference approval rate of 0, so every ratio is 0/0. Both air.py "
                                 "(ratio 'UNDEFINED', since 2026-09-28) and this script treat it as "
                                 "undefined and do not flag it. Before that fix air.py returned 0.0 and "
                                 "flagged the reference group against itself, which made SQL 615 against "
                                 "independent 614."),
        "files_with_no_applications": sorted(k for k, v in totals["by_file_state"].items() if v[0] == 0),
        "totals": checks,
        "all_totals_checks_pass": all(v.get("pass", True) for v in checks.values()),
        "seconds_independent": round(t_ind, 1),
        "seconds_sql_path": round(t_sql, 1),
        "seconds_sql_flagged_lenders": round(t_flagged, 1),
        "seconds_total": round(time.time() - t_start, 1),
        "independence_note": ("pyarrow code written from the definition; reads no file in sql/ and calls "
                              "nothing in rates.py or air.py; floor constant copied from floors.py. "
                              "The author had read the SQL and air.py first, so shared misreadings "
                              "would not be caught."),
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    ok = (counts_diff["only_in_sql"] == 0 and counts_diff["only_in_independent"] == 0
          and not any(counts_diff["mismatches"].values())
          and screen_diff["status_mismatches"] == 0 and screen_diff["ratio_mismatches"] == 0
          and summary["flagged_race_only"]["symmetric_difference"] == 0
          and summary["flagged_all_three_columns"]["symmetric_difference"] == 0
          and summary["all_totals_checks_pass"])
    print("AGREE" if ok else "DISAGREE (see the counts above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
