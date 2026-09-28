"""Generate the small precomputed JSON artifacts the Streamlit dashboard reads.

This script is the only thing that turns the recorded measurements into the
dashboard's data. app.py never parses markdown and never opens
data/hmda.duckdb; it only reads the JSON and SVG files this script writes
under results/.

Run it from the repo root: `.venv/bin/python scripts/gen_dashboard_artifacts.py`

It is safe to re-run. It does not invent numbers: the governance table is
parsed out of docs/CONTROLS.md's own markdown table (so it cannot drift from
the source of truth `hmda verify --governance` checks against), and the
metric, mitigation and engineering figures below are copied from the console
output of the national runs, each with the command that produced it. The
sample-based model and mitigation runs are also written up in
docs/VERIFICATION.md, section 7.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def parse_controls_table() -> list[dict]:
    """Parse the 38-row control table out of docs/CONTROLS.md verbatim."""
    text = (ROOT / "docs" / "CONTROLS.md").read_text()
    rows = []
    for line in text.splitlines():
        if not line.startswith("| C-"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 6:
            continue
        control_id, assertion, implementing_file, test, sr11_7, osfi_e23 = cells
        backed = test != "NOT IMPLEMENTED"
        rows.append(
            {
                "control_id": control_id,
                "assertion": assertion,
                "implementing_file": implementing_file,
                "test": test,
                "backed": backed,
                "sr_11_7": sr11_7,
                "osfi_e_23": osfi_e23,
            }
        )
    return rows


def governance_summary(rows: list[dict]) -> dict:
    backed = sum(1 for r in rows if r["backed"])
    unbacked = [r["control_id"] for r in rows if not r["backed"]]
    return {
        "source": "docs/CONTROLS.md (parsed table), cross-checked against "
        "`hmda verify --governance` output",
        "total_controls": len(rows),
        "backed_by_test": backed,
        "not_implemented": len(unbacked),
        "not_implemented_ids": unbacked,
        "verify_command": "hmda verify --governance",
        "controls": rows,
    }


# The fourteen headline metrics, M1 to M14, each with the command that
# measures it and a one-line summary of the last national run. Status is
# MEASURED (full national file), SAMPLE_BASED (a stratified sample of it),
# UNMEASURABLE, or PARTIALLY_FIXED_UNVERIFIED.
METRICS_LEDGER = [
    {
        "id": "M1", "status": "MEASURED",
        "command": "hmda verify --counts",
        "summary": "36,734,685 rows, 5,329 LEIs, full DuckDB SQL.",
    },
    {
        "id": "M2", "status": "SAMPLE_BASED",
        "command": "time hmda audit --all-years --source national",
        "summary": "--all-years runs all four audit branches. "
        "Re-measured 64.17s real, exit 0. Waterfall/AIR/controlled branches "
        "carry their own sampling caveats; drift stays fixture-scoped.",
    },
    {
        "id": "M3", "status": "MEASURED",
        "command": ".venv/bin/python bench/compare_pandas.py",
        "summary": "DuckDB 0.13s vs pandas 24.36s at the full 36,734,685 rows "
        "= 185x. TABLES IDENTICAL: True. An earlier same-day run measured "
        "208x; machine load varies between runs, and 185x is the figure "
        "kept as canonical (results/engineering.json).",
    },
    {
        "id": "M4", "status": "MEASURED",
        "command": ".venv/bin/python bench/compare_pandas.py --mem",
        "summary": "DuckDB 0.09 GiB vs pandas 10.49 GiB peak = 119x. Same run "
        "as M3; an earlier same-day run measured 93x for the same reason.",
    },
    {
        "id": "M5", "status": "MEASURED",
        "command": "hmda audit --air --min-count 100 --source national",
        "summary": "Full-file DuckDB SQL, 614 lenders flagged at --min-count 100, "
        "re-run 2026-09-28 after the 0/0 fix in air.py.",
        "correction": "Was 615 (2026-09-13). air.py gave a ratio of 0.0 and a "
        "flag when the reference approval rate was 0, so one lender whose only "
        "group was the reference group, with every application denied, was "
        "flagged against itself. The ratio is now UNDEFINED and not flagged "
        "(docs/VERIFICATION.md, section 3).",
    },
    {
        "id": "M6", "status": "SAMPLE_BASED",
        "command": "hmda audit --controlled --source national",
        "summary": "National, 200,000-row stratified sample (seed 0); the "
        "--controlled branch is pandas-bound.",
    },
    {
        "id": "M7", "status": "SAMPLE_BASED",
        "command": "hmda model --eval --source national --sample-n 1500000",
        "summary": "National, 1,500,000 rows, seeds 0/1/2. GBM ranks 72% "
        "better than base rate, logistic 57%, stable across seeds, below the "
        "fixture's inflated 75%/61%.",
    },
    {
        "id": "M8", "status": "PARTIALLY_FIXED_UNVERIFIED",
        "command": "hmda report",
        "summary": "A bare `hmda report` builds the model card. --figures "
        "and --frontier write the three SVG figures to out/figures/. The "
        "full HTML report (src/hmda/report/render.py) is not implemented.",
    },
    {
        "id": "M9", "status": "MEASURED",
        "command": "pytest -q",
        "summary": "Most recent full-suite run: 326 passed, 6 skipped "
        "(2026-09-28, after the 0/0 fix in air.py).",
        "superseded": "290 passed (2026-09-14), and 322 passed, 6 skipped, "
        "1 xfailed (2026-09-28, before the 0/0 fix). Both kept for the record.",
    },
    {
        "id": "M10", "status": "MEASURED",
        "command": "hmda verify --governance",
        "summary": "38 governance controls, 37 backed by a test, 1 (C-38) "
        "NOT IMPLEMENTED. hmda verify --governance exits 0.",
    },
    {
        "id": "M11", "status": "SAMPLE_BASED",
        "command": "hmda mitigate --compare --source national",
        "summary": "National, 1,500,000 rows, seeds 0/1/2. Per-group "
        "thresholding cuts the gap 55-61%, fairness-constrained GBM "
        "10.7-11.6%, reweighing 0.2-3.9%, reported as ranges, no "
        "single-seed headline.",
    },
    {
        "id": "M12", "status": "SAMPLE_BASED",
        "command": "hmda mitigate --profit --source national",
        "summary": "National, 500,000 rows, seed 0 only. Sign reversed from "
        "the fixture: per-group thresholding costs $26,882 per 1,000 "
        "applications nationally, vs. the fixture's claimed +$296,556 gain.",
    },
    {
        "id": "M13", "status": "UNMEASURABLE",
        "command": "hmda recourse --by-group --source national",
        "summary": "Recourse number is unstable across seeds/bootstraps on "
        "the national data, so the verdict is UNMEASURABLE: the negative "
        "result is reported instead of a number. No dollar figure is quotable; "
        "see docs/RECOURSE.md.",
    },
    {
        "id": "M14", "status": "SAMPLE_BASED",
        "command": "hmda explain --report --source national",
        "summary": "National, 2,000 held-out rows, seed 0 only, no "
        "multi-seed stability claim. Counts only: no SHAP magnitude ships "
        "(docs/EXPLAINABILITY.md).",
    },
]

MITIGATION_FINDING = {
    "source": "hmda mitigate --compare --profit --source national, run by hand; "
    "see docs/VERIFICATION.md, section 7",
    "fixture_claim_per_1000": 296556,
    "fixture_claim_direction": "gain",
    "national_result_per_1000": -26882,
    "national_result_direction": "cost",
    "national_sample_n": 500000,
    "national_sample_seed": 0,
    "provenance_note": "The margin run was done by hand (seed 0, 500,000 "
    "rows) and only its result was written down, not the console output. "
    "It has not been checked a second time, so it is shown with that caveat "
    "attached, not as an independently re-measured number.",
    "uncertainty_note": "Single seed at n=500,000, the sample size M11 "
    "shows is unreliable for this same technique (a 40.4-point spread in gap "
    "reduction across seeds 0/1/2 at n=500,000). No three-seed spread "
    "exists for the dollar figure: uncertainty on $26,882 is unquantified "
    "and probably large.",
    "gap_reduction_range_pct": {
        "per_group_threshold": [55, 61],
        "fair_constrained_gbm": [10.7, 11.6],
        "reweighing": [0.2, 3.9],
        "fixture_per_group_threshold_pct": 76.8,
        "note": "The fixture's 76.8% for per-group thresholding sits above "
        "the entire national band.",
    },
    "legal_flag": "per_group_threshold carries a LEGALLY CONTESTED warning "
    "(C-25): a different approval cut-off by race is disparate treatment "
    "on its face in the US. Measured for completeness, not presented as "
    "approved practice.",
}

ENGINEERING = {
    "source": "bench/compare_pandas.py on the full national file, 2026-09-14",
    "command": ".venv/bin/python bench/compare_pandas.py --mem",
    "rows": 36734685,
    "speed": {"duckdb_s": 0.13, "pandas_s": 24.36, "factor": "185x"},
    "memory": {"duckdb_gib": 0.09, "pandas_gib": 10.49, "factor": "119x"},
    "tables_identical": True,
    "footnote": "An earlier same-day run on the same machine measured "
    "208x / 93x. That run was superseded, not deleted, because machine "
    "load varies between runs; 185x/119x is the later run and is treated "
    "as canonical. Publishing the lower, superseding number and saying why "
    "is deliberate.",
}


# ---------------------------------------------------------------------------
# Row-level chart data, copied from the national runs' console output so
# app.py never parses markdown and never hardcodes a number of its own.
# ---------------------------------------------------------------------------

# `hmda audit --air --min-count 100 --source national`, re-run 2026-09-28
# after the 0/0 fix, EXIT=0. Eighteen rows in three protected-class
# partitions; each partition's denominators sum to 32,620,789, which is the
# 36,734,685-row file after the `action_taken != 6` exclusion (purchased loans
# are not applications).
FOUR_FIFTHS = {
    "source": "console output of the command below, 2026-09-28, after the 0/0 fix in air.py",
    "command": "hmda audit --air --min-count 100 --source national",
    "status": "MEASURED",
    "threshold": 0.80,
    "reference_group": "Joint",
    "lenders_screened": 5329,
    "lenders_flagged": 614,
    "superseded": "615 lenders flagged (2026-09-13), before the 0/0 fix in "
    "air.py. The difference is one lender whose reference approval rate was 0.",
    "min_count": 100,
    "rows_screened": 32620789,
    "rows_in_file": 36734685,
    "exclusion_note": "Rows screened is the file after `action_taken != 6`; "
    "purchased loans are not applications.",
    "rows": [
        {"partition": "Race", "group": "2 or more minority races", "ratio": 0.8326, "denom": 84000, "flag": False},
        {"partition": "Race", "group": "American Indian or Alaska Native", "ratio": 0.8496, "denom": 243804, "flag": False},
        {"partition": "Race", "group": "Asian", "ratio": 0.9875, "denom": 1976817, "flag": False},
        {"partition": "Race", "group": "Black or African American", "ratio": 0.8497, "denom": 2893734, "flag": False},
        {"partition": "Race", "group": "Free Form Text Only", "ratio": 0.6876, "denom": 9032, "flag": True},
        {"partition": "Race", "group": "Joint", "ratio": 1.0000, "denom": 702245, "flag": False, "is_reference": True},
        {"partition": "Race", "group": "Native Hawaiian or Other Pacific Islander", "ratio": 0.8317, "denom": 79000, "flag": False},
        {"partition": "Race", "group": "Race Not Available", "ratio": 0.9441, "denom": 6038818, "flag": False},
        {"partition": "Race", "group": "White", "ratio": 0.9774, "denom": 20593339, "flag": False},
        {"partition": "Ethnicity", "group": "Ethnicity Not Available", "ratio": 0.9646, "denom": 5607045, "flag": False},
        {"partition": "Ethnicity", "group": "Free Form Text Only", "ratio": 0.7701, "denom": 15582, "flag": True},
        {"partition": "Ethnicity", "group": "Hispanic or Latino", "ratio": 0.9147, "denom": 3878586, "flag": False},
        {"partition": "Ethnicity", "group": "Joint", "ratio": 1.0000, "denom": 823547, "flag": False, "is_reference": True},
        {"partition": "Ethnicity", "group": "Not Hispanic or Latino", "ratio": 0.9853, "denom": 22296029, "flag": False},
        {"partition": "Sex", "group": "Female", "ratio": 0.9040, "denom": 7312149, "flag": False},
        {"partition": "Sex", "group": "Joint", "ratio": 1.0000, "denom": 11015122, "flag": False, "is_reference": True},
        {"partition": "Sex", "group": "Male", "ratio": 0.9309, "denom": 11156760, "flag": False},
        {"partition": "Sex", "group": "Sex Not Available", "ratio": 0.9487, "denom": 3136758, "flag": False},
    ],
}

# Measured 2026-09-14. Full file, no sampling, queried from data/hmda.duckdb's
# `lar` table with the same `action_taken != 6` exclusion that
# sql/rates_by_group.sql applies. Denial rate is denials / applications.
DENIAL_RATES = {
    "source": "direct DuckDB query on the full file, 2026-09-14",
    "command": "duckdb query on data/hmda.duckdb `lar`, action_taken != 6",
    "status": "MEASURED",
    "rows_screened": 32620789,
    "reference_group": "Joint",
    "race_not_available_share_pct": 26.73,
    "rows": [
        {"group": "White", "applications": 20593339, "denials": 3576956},
        {"group": "Race Not Available", "applications": 6038818, "denials": 1218839},
        {"group": "Black or African American", "applications": 2893734, "denials": 815053},
        {"group": "Asian", "applications": 1976817, "denials": 326450},
        {"group": "Joint", "applications": 702245, "denials": 108563},
        {"group": "American Indian or Alaska Native", "applications": 243804, "denials": 68681},
        {"group": "2 or more minority races", "applications": 84000, "denials": 24875},
        {"group": "Native Hawaiian or Other Pacific Islander", "applications": 79000, "denials": 23455},
        {"group": "Free Form Text Only", "applications": 9032, "denials": 3782},
    ],
}

# `hmda audit --controlled --source national`, 200,000-row sample, seed 0.
CONTROLLED = {
    "source": "console output of the command below, 200,000-row stratified sample, seed 0",
    "command": "hmda audit --controlled --source national",
    "status": "SAMPLE_BASED",
    "sample_n": 200000,
    "sample_seed": 0,
    "group": "Black or African American",
    "reference_group": "Joint",
    "raw_gap_pp": 12.44,
    "controlled_gap_pp": 8.23,
    "covariates_source": "src/hmda/fairness/controlled.py::_build_controls",
    "covariates_plain": "income, how much of the home's value is being "
    "borrowed, debt measured against income, what the loan is for, and "
    "whether it is a first or a second mortgage",
    "note": "The --controlled branch is pandas-bound and ran on a 200,000-row "
    "stratified sample, not the full file. No comparable fixture number "
    "exists, so this is a new national baseline, not a delta.",
    "caveat": "The controlled gap is unexplained variation under a stated "
    "model. It is not evidence of discrimination.",
}

# The mitigation provenance gap, recorded rather than resolved.
MITIGATION_CONFLICT = {
    "source": "docs/VERIFICATION.md, section 7; src/hmda/cli.py (the mitigate command)",
    "headline": "The national mitigation figures were recorded by hand, without the console output.",
    "claims": [
        {
            "locator": "docs/VERIFICATION.md, section 7 (Mitigation)",
            "quote": "Only the result figures below were written down from these runs, not the "
            "full console output, so they have not been checked a second time.",
        },
        {
            "locator": "src/hmda/cli.py, the mitigate command",
            "quote": "`mitigate` takes `--source`, `--sample-n` and `--sample-seed`, so the "
            "national run can be repeated with `hmda mitigate --compare --profit --source "
            "national --sample-n 1500000 --sample-seed S`.",
        },
    ],
    "resolution": "Unresolved, and presented as unresolved. Every other "
    "SAMPLE_BASED result in this repository was copied from the console "
    "output of its run; these figures were written down by hand without "
    "that output, so no national mitigation number is charted "
    "on this page and none is quoted in the text above. The measurement "
    "ledger lower down still lists the M11 and M12 figures as they were "
    "recorded, dollar figures included, so the claim and the caveat sit in "
    "the same place.",
}


# Scale headline. `hmda verify --counts` on the full file (M1, MEASURED).
SCALE = {
    "source": "full DuckDB SQL over data/hmda.duckdb",
    "command": "hmda verify --counts",
    "status": "MEASURED",
    "applications": 36734685,
    "lenders": 5329,
    "years": 3,
    "years_label": "2023, 2024, 2025",
}

# Measurement-class badge for each committed figure, so a chart never
# implies full-national measurement when the underlying number is
# SAMPLE_BASED. disparity.svg and four_fifths.svg are full-file (M5 and the
# denial-rate query above). frontier.svg draws on two different
# measurements: the left (fairness/accuracy) panel is M11 (SAMPLE_BASED,
# national, 1,500,000 rows, seeds 0/1/2); the right (margin cost) panel is
# M12 (SAMPLE_BASED, national, 500,000 rows, single seed 0, recorded by hand).
FIGURE_BADGES = {
    "disparity.svg": {
        "status": "MEASURED",
        "caption": "Full national file, no sampling (36,734,685 rows).",
        "source": "results/denial_rates.json",
    },
    "four_fifths.svg": {
        "status": "MEASURED",
        "caption": "Full national file, DuckDB SQL, no sampling.",
        "source": "results/four_fifths.json (M5)",
    },
    "frontier.svg": {
        "status": "SAMPLE_BASED",
        "caption": "Left panel (fairness/accuracy): national, 1,500,000 "
        "rows, seeds 0/1/2 (M11). Right panel (margin cost): national, "
        "500,000 rows, single seed 0, not independently verified (M12).",
        "source": "results/metrics_ledger.json (M11, M12); docs/VERIFICATION.md, section 7",
    },
}


# `hmda model --eval --source national --sample-n 1500000 --sample-seed S`
# at seeds 0, 1 and 2 (M7). Also written up in docs/VERIFICATION.md,
# section 7, which the model card parses.
MODEL_EVAL = {
    "source": "console output of the command below at seeds 0, 1 and 2",
    "command": "hmda model --eval --source national --sample-n 1500000 --sample-seed S",
    "status": "SAMPLE_BASED",
    "sample_n": 1500000,
    "seeds": [0, 1, 2],
    "test_year": 2025,
    "metric": "how much better than a base-rate guess the model orders a random "
    "denied/approved pair (a base-rate guess orders 50 of 100 pairs correctly)",
    "better_than_base_rate_pct": {"logistic_regression": 57, "gradient_boosted_trees": 72},
    "pairs_ordered_per_100": {"base_rate": 50, "logistic_regression": 78, "gradient_boosted_trees": 86},
    "by_seed": [
        {"seed": 0, "analysis_rows": 1058400, "logistic_regression_pct": 57, "gradient_boosted_trees_pct": 72},
        {"seed": 1, "analysis_rows": 1060642, "logistic_regression_pct": 57, "gradient_boosted_trees_pct": 72},
        {"seed": 2, "analysis_rows": 1060029, "logistic_regression_pct": 57, "gradient_boosted_trees_pct": 72},
    ],
    "fixture_pct": {"logistic_regression": 61, "gradient_boosted_trees": 75},
    "note": "Sample numbers, not full-file numbers. The CLI prints whole "
    "percentages, so the three seeds agree to 1 point.",
}


def copy_figures() -> dict:
    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_figs = ROOT / "out" / "figures"
    provenance = {}
    for name in ("disparity.svg", "four_fifths.svg", "frontier.svg"):
        src = out_figs / name
        if not src.exists():
            continue
        dst = fig_dir / name
        shutil.copy2(src, dst)
        m = re.search(r"<dc:date>([^<]+)</dc:date>", src.read_text(errors="ignore"))
        provenance[name] = {
            "source_path": f"out/figures/{name}",
            "mtime": src.stat().st_mtime,
            "embedded_dc_date": m.group(1) if m else None,
            **FIGURE_BADGES.get(name, {}),
        }
    return provenance


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    controls = parse_controls_table()
    assert len(controls) == 38, f"expected 38 controls, parsed {len(controls)}"
    (RESULTS / "governance.json").write_text(
        json.dumps(governance_summary(controls), indent=2)
    )

    (RESULTS / "metrics_ledger.json").write_text(
        json.dumps(
            {
                "source": "the command on each row, national data",
                "metrics": METRICS_LEDGER,
            },
            indent=2,
        )
    )

    (RESULTS / "mitigation.json").write_text(json.dumps(MITIGATION_FINDING, indent=2))
    (RESULTS / "engineering.json").write_text(json.dumps(ENGINEERING, indent=2))
    (RESULTS / "scale.json").write_text(json.dumps(SCALE, indent=2))
    (RESULTS / "four_fifths.json").write_text(json.dumps(FOUR_FIFTHS, indent=2))
    (RESULTS / "denial_rates.json").write_text(json.dumps(DENIAL_RATES, indent=2))
    (RESULTS / "controlled.json").write_text(json.dumps(CONTROLLED, indent=2))
    (RESULTS / "model_eval.json").write_text(json.dumps(MODEL_EVAL, indent=2))
    (RESULTS / "mitigation_conflict.json").write_text(
        json.dumps(MITIGATION_CONFLICT, indent=2)
    )

    fig_provenance = copy_figures()
    (RESULTS / "figures_provenance.json").write_text(
        json.dumps(fig_provenance, indent=2)
    )

    print(f"wrote {len(controls)} controls, {len(METRICS_LEDGER)} metrics, "
          f"{len(fig_provenance)} figures to {RESULTS}")


if __name__ == "__main__":
    main()
