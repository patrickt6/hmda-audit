"""Record the committed results in a claimtrail store and state the headline claims.

Every number in the README's results table must trace to a record in
`.claimtrail/`. This script builds those records from the small JSON files
under `results/`, which the national runs produced, and states each headline
claim with a structured assertion that claimtrail re-checks on every run.

It is safe to re-run. Unchanged results collapse onto the existing records.
If a results file changes after it was recorded, registration fails loudly
(a collision), because the claims that cite it must be reviewed before they
are re-linked. That is the point: a number cannot move under a sentence
without someone noticing.

A reviewed change (a code fix that moves a number) is recorded by giving the
results entry a ``code_sha`` field: the commit of the code that produced the
new number. claimtrail puts code_sha into the computation id, so the new
result gets a new record and the old record stays in the store. The claim's
``expect`` values are then updated by hand in this file, in the same commit.
The steps are in docs/CLAIMS.md.

    .venv/bin/python scripts/record_claims.py
    claimtrail check --paper hmda-audit
    claimtrail audit-report README.md

Basis tags carry each result's status: MEASURED (full national file),
SAMPLE_BASED (a stratified sample), or REPORTED (a recorded run that this
script does not re-measure).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import claimtrail

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = "hmda-audit"

claimtrail.set_store_root(ROOT / ".claimtrail")

# Claim ids whose stored row may be replaced on this run (see --reviewed).
REVIEWED: set[str] = set()


def load(stem: str) -> dict:
    return json.loads((RESULTS / f"{stem}.json").read_text(encoding="utf-8"))


def register_result(stem: str, basis: str | None = None) -> str:
    """Register one committed results file as an external computation."""
    data = load(stem)
    return claimtrail.register_external(
        function_name=f"hmda.results.{stem}",
        inputs={
            "command": data.get("command") or data.get("verify_command"),
            "source": data.get("source"),
        },
        outputs=data,
        code_sha=data.get("code_sha"),
        source_file=f"results/{stem}.json",
        tags={"report": REPORT, "basis": data.get("status") or basis or "UNSPECIFIED"},
    )


def register_reported_test_run() -> str:
    """The last full test run, as results/metrics_ledger.json records it (M9).

    This number is reported, not re-measured here. Re-running the suite is
    a separate verification step:

        .venv/bin/pytest -q            # note the "N passed" line
        echo '{"passed": N}' > /tmp/run.json
        claimtrail verify <id> --against /tmp/run.json
    """
    ledger = load("metrics_ledger")
    m9 = next(m for m in ledger["metrics"] if m["id"] == "M9")
    passed = int(re.search(r"(\d+) passed", m9["summary"]).group(1))
    skipped = int(re.search(r"(\d+) skipped", m9["summary"]).group(1))
    return claimtrail.register_external(
        function_name="hmda.reported_test_run",
        inputs={"metric": "M9", "command": m9["command"]},
        outputs={"passed": passed, "skipped": skipped},
        code_sha=m9.get("code_sha"),
        source_file="results/metrics_ledger.json",
        notes=f"Reported in results/metrics_ledger.json (M9): {m9['summary']}",
        tags={"report": REPORT, "basis": "REPORTED"},
    )


def state(claim_id: str, text: str, computation_id: str, basis: str, expect: list[str]) -> None:
    tags = {"paper": REPORT, "basis": basis}
    try:
        claimtrail.claim(text, claim_id=claim_id, computation_id=computation_id,
                         expect=expect, tags=tags)
    except claimtrail.ClaimtrailCollisionError:
        # claim() checks every assertion before it inserts, so reaching a
        # collision means the new record already satisfies `expect`. The
        # stored claim row is replaced only for ids named with --reviewed.
        # Without the flag (as in CI) any moved number fails here.
        if claim_id not in REVIEWED:
            raise
        _replace_claim(claim_id, text, computation_id, expect, tags)


def _replace_claim(claim_id, text, computation_id, expect, tags) -> None:
    """Overwrite a claim row in place after a reviewed change.

    claimtrail's INTEGRATION.md says a stable claim_id "overwrites in place",
    but claim() raises a collision instead and does not expose force. This
    builds the same row claim() would and inserts it with force=True. Old
    computation rows are kept: a reviewed result carries a new code_sha, so
    it gets a new computation id.
    """
    from claimtrail import assertions as _assertions
    from claimtrail.store import Claim, get_store, utc_now_iso

    expectations = [_assertions.parse_expectation(e) for e in expect]
    rec = Claim(
        id=claim_id, text=text, value_numeric=None, computation_id=computation_id,
        created_at=utc_now_iso(), notes=None, tags={k: str(v) for k, v in tags.items()},
        assertions=_assertions.dumps(expectations),
    )
    get_store().insert_claim(rec, force=True)
    print(f"replaced reviewed claim {claim_id} -> computation {computation_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviewed", default="",
                    help="comma-separated claim ids whose stored row may be replaced "
                         "after a reviewed change (see docs/CLAIMS.md)")
    REVIEWED.update(x for x in ap.parse_args().reviewed.split(",") if x)

    scale = register_result("scale")
    air = register_result("four_fifths")
    controlled = register_result("controlled")
    model = register_result("model_eval")
    watch = register_result("watch_list", basis="MEASURED")
    spark = register_result("spark_parity", basis="MEASURED")
    independent = register_result("independent_check")
    probe = register_result("nn_probe")
    mutation = register_result("mutation")
    engineering = register_result("engineering", basis="MEASURED")
    governance = register_result("governance", basis="MEASURED")
    tests = register_reported_test_run()

    state("hmda-scale", "Applications and lenders in the national file, 2023 to 2025",
          scale, "MEASURED",
          ["outputs.applications == 36734685", "outputs.lenders == 5329", "outputs.years == 3"])
    state("hmda-four-fifths", "Lenders flagged by the full-file four-fifths screen",
          air, "MEASURED",
          ["outputs.lenders_flagged == 614", "outputs.lenders_screened == 5329",
           "outputs.threshold == 0.8", "outputs.min_count == 100"])
    state("hmda-controlled", "Black and Joint applicants' denial-rate gap, raw and controlled",
          controlled, "SAMPLE_BASED",
          ["outputs.raw_gap_pp == 12.44", "outputs.controlled_gap_pp == 8.23",
           "outputs.sample_n == 200000", "outputs.sample_seed == 0"])
    state("hmda-model", "Denial ranking against a base-rate guess, national sample",
          model, "SAMPLE_BASED",
          ["outputs.better_than_base_rate_pct.gradient_boosted_trees == 72",
           "outputs.better_than_base_rate_pct.logistic_regression == 57",
           "outputs.sample_n == 1500000"])
    state("hmda-watch-list", "Empirical-Bayes lender watch list, Black against White denial gap",
          watch, "MEASURED",
          ["outputs.watch_list == 404", "outputs.lenders_scored == 3454",
           "outputs.watch_list_and_four_fifths_no_floor == 382"])
    state("hmda-spark-parity", "The four-fifths SQL on Spark and on DuckDB, full file",
          spark, "MEASURED",
          ["outputs.rows_spark == 31793", "outputs.rows_duckdb == 31793",
           "outputs.value_mismatches == 0", "outputs.only_in_spark == 0",
           "outputs.only_in_duckdb == 0"])
    state("hmda-independent", "Independent pyarrow reimplementation of the four-fifths screen",
          independent, "MEASURED",
          ["outputs.counts.rows_compared == 31793",
           "outputs.flagged_all_three_columns.both == 614",
           "outputs.flagged_all_three_columns.symmetric_difference == 0"])
    state("hmda-race-probe", "Linear probe for race on a denial network that never sees race",
          probe, "SAMPLE_BASED",
          ["outputs.aggregate.race_probe_auc_linear.input_features.mean == 0.6783",
           "outputs.aggregate.race_probe_auc_linear.hidden_layer_2.mean == 0.659",
           "outputs.aggregate.race_probe_auc_linear.control_shuffled_labels_layer_2.mean == 0.4994"])
    state("hmda-mutation", "Mutation score of the four-fifths and watch-list code",
          mutation, "MEASURED",
          ["outputs.rerun_2026_09_28.run.killed == 476",
           "outputs.rerun_2026_09_28.run.total == 487",
           "outputs.rerun_2026_09_28.run.score == 0.977"])
    state("hmda-engineering", "DuckDB against pandas at full national scale",
          engineering, "MEASURED",
          ["outputs.speed.duckdb_s == 0.13", "outputs.speed.pandas_s == 24.36",
           "outputs.memory.duckdb_gib == 0.09", "outputs.memory.pandas_gib == 10.49",
           "outputs.tables_identical == true"])
    state("hmda-governance", "Governance controls and their test backing",
          governance, "MEASURED",
          ["outputs.total_controls == 38", "outputs.backed_by_test == 37",
           "outputs.not_implemented == 1"])
    state("hmda-tests", "Most recent full test-suite run, as recorded in results/metrics_ledger.json",
          tests, "REPORTED", ["outputs.passed == 326", "outputs.skipped == 6"])
    print("recorded: hmda-scale hmda-four-fifths hmda-controlled hmda-model hmda-watch-list "
          "hmda-spark-parity hmda-independent hmda-race-probe hmda-mutation "
          "hmda-engineering hmda-governance hmda-tests")

if __name__ == "__main__":
    main()
