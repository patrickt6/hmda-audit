# hmda-audit

A fair-lending and model-governance audit of US mortgage lending, run on a
laptop over the full public HMDA Loan/Application Register: 36,734,685
records from 5,329 lenders, 2023 to 2025, which is 32,620,789 applications
once purchased loans are taken out. It screens every lender with
the four-fifths adverse-impact rule, measures how much of the denial gap
between Black applicants and the best-approved group survives basic loan
controls, trains denial models that never see race and documents them the
way a bank's model-risk team would, and then checks its own numbers several
independent ways. Every number in the results table below comes from a
committed file in `results/`, and CI fails if the table and those files
disagree.

## Headline results

| Result | Number | Basis | Source |
|---|---|---|---|
| National file | 36,734,685 HMDA records from 5,329 lenders; 32,620,789 applications after excluding purchased loans <!-- ct:hmda-scale --> <!-- ct:hmda-four-fifths --> | full file | `results/scale.json`, `results/four_fifths.json` |
| Four-fifths screen | 614 of 5,329 lenders have a group approved at below 0.8 times the best-approved group, counting only groups with at least 100 applications. The screen counts every record except purchased loans, so withdrawn, incomplete and preapproval records count as not denied <!-- ct:hmda-four-fifths --> | full file | `results/four_fifths.json` |
| Four-fifths, two definitions | nationally, Black applicants' approval rate is 0.8497 times the Joint group's under the screen's definition, and 0.7862 when only lender decisions count (action_taken 1, 2 and 3, the definition the models use). Under the second definition 4 named race groups fall below 0.8, so the national picture depends on the definition <!-- ct:hmda-four-fifths --> <!-- ct:hmda-four-fifths-decisions --> | full file | `results/four_fifths.json`, `results/four_fifths_decisions_only.json` |
| Black applicants vs the Joint group (mixed-race joint applications, the best-approved group) | denial gap of 12.44 percentage points raw, 8.23 after controls for income, loan-to-value, debt-to-income, loan purpose and lien <!-- ct:hmda-controlled --> | 200,000-row sample, seed 0 | `results/controlled.json` |
| Denial model ranking | gradient-boosted trees AUC 0.86 (Gini 0.72), logistic regression AUC 0.78 (Gini 0.57), trained on 2023 and 2024 and scored on the 2025 holdout. A constant score has AUC 0.5 <!-- ct:hmda-model --> <!-- ct:hmda-model-auc --> | 1,500,000-row sample, three seeds, 380,272 to 380,365 test rows per seed | `results/model_auc.json`, `results/model_eval.json` |
| Empirical-Bayes watch list | 404 of 3,454 scored lenders have a Black and White denial gap above the typical lender's, with at least 95% posterior probability; 382 of them also fail the four-fifths ratio when no size floor is applied <!-- ct:hmda-watch-list --> | full file | `results/watch_list.json` |
| Spark parity | the four-fifths SQL gives the same 31,793 lender-by-race rows (one row per lender and race group) and 32,620,789 applications on Databricks Spark and on DuckDB, 0 rows differ <!-- ct:hmda-spark-parity --> | full file | `results/spark_parity.json` |
| Independent reimplementation | separate pyarrow code, written from the definition by the same author, matches all 31,793 lender-by-race rows and flags the same 614 lenders, symmetric difference 0. The first comparison disagreed by one lender and exposed a 0/0 reference-group bug in `air.py`, which was then fixed (`docs/VERIFICATION.md`) <!-- ct:hmda-independent --> | full file | `results/independent_check.json` |
| Race probe | a linear probe reads Black vs White from the allowed credit fields at AUC 0.68, and from the network's last hidden layer at 0.66; shuffled labels give 0.50 <!-- ct:hmda-race-probe --> | 500,000-row sample, five seeds | `results/nn_probe.json` |
| Mutation testing | 476 of 487 mutants of `air.py`, `rates.py`, `shrink.py` and `floors.py` (the screen and watch-list code) are killed by the tests (97.7%). `controlled.py`, run as a separate pass, scored 72.6% (260 of 358), and its 98 survivors have not been reviewed <!-- ct:hmda-mutation --> | current code | `results/mutation.json` |
| DuckDB vs pandas | 0.13 s against 24.36 s, and 0.09 GiB against 10.49 GiB peak memory, same output table <!-- ct:hmda-engineering --> | full file | `results/engineering.json` |
| Governance controls | 38 controls documented, 37 backed by an automated test <!-- ct:hmda-governance --> | registry | `results/governance.json` |
| Test suite | 326 passed, 6 skipped <!-- ct:hmda-tests --> | last full run, with the national data | `results/metrics_ledger.json` |

Gini is 2 x AUC - 1. It is the same number that `hmda model --eval`
prints as "percent better than a base-rate guess" (72% and 57%).

The model, controlled-gap and probe numbers are sample numbers. The model
and mitigation stages train models and do not fit the whole file in memory,
so they run on stratified national samples; each table row says which.

## How it works

1. `hmda ingest` and `scripts/ingest_all.py` download each state and year
   from the FFIEC HMDA Data Browser API, convert it to parquet, and load a
   DuckDB database.
2. Every aggregation is a reviewable `.sql` file in `sql/`, run by DuckDB
   straight over the parquet files.
3. The four-fifths screen, the controlled gap and the watch list live in
   `src/hmda/fairness/`. Raw and controlled gaps are always printed together.
4. The models in `src/hmda/model/` use an allowlist of loan and property
   fields, a time split (train 2023 and 2024, test 2025), and a base-rate
   baseline beside every score. The baseline gives everyone the same score,
   so it cannot order any pair and its AUC is 0.5.
5. `src/hmda/governance/` holds the control registry (mapped to SR 11-7 and
   OSFI E-23 principles), the generated model card, SHAP explanations and
   drift checks.

The full walkthrough, from the HMDA fields to each SQL file and formula, is
[`docs/HOW-IT-WORKS.md`](docs/HOW-IT-WORKS.md). How each number is checked,
and what each check does not prove, is in
[`docs/VERIFICATION.md`](docs/VERIFICATION.md).

## Run it

```
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
make test
```

The tests run on a committed 50,000-row fixture (`tests/fixtures/hmda_50k.parquet`).
Tests that need the national file skip when it is absent. To get the
national data (168 state-year files), run:

```
.venv/bin/python scripts/ingest_all.py
```

It is resumable and skips state-years it already has. Then:

```
.venv/bin/hmda verify --counts                                # rows, lenders, years
.venv/bin/hmda audit --air --min-count 100 --source national  # four-fifths screen
.venv/bin/hmda audit --controlled --source national           # raw and controlled gaps
.venv/bin/hmda model --eval --source national --sample-n 1500000
.venv/bin/hmda verify --governance                            # every control names a real test
.venv/bin/python scripts/watch_list.py                        # empirical-Bayes watch list
.venv/bin/python scripts/air_independent.py                   # independent recount
```

`app.py` is a small Streamlit dashboard over the committed `results/` files
(`pip install -e ".[app]"`, then `streamlit run app.py`). The claims check
that CI runs is described in [`docs/CLAIMS.md`](docs/CLAIMS.md).

## What this is not

- Not a claim that any named lender discriminates. No lender is named
  anywhere in the repo or its output. A flag is a screening signal that
  warrants review, not a legal finding.
- Not proof of discrimination. A controlled gap is unexplained variation
  under a stated model.
- Not a live underwriting system. The models and the mitigation study run
  offline on historical records and never score a real applicant.

## Limits

- HMDA has no credit score, so every controlled gap leaves out the largest
  single factor in real underwriting. The gap probably overstates what would
  survive a full set of controls.
- The data records outcomes, not reasons, so it cannot show intent.
- Race is not reported on 26.73% of national applications
  (`results/denial_rates.json`), so every group comparison is conditional
  on who reported.
- The four-fifths ratio compares each group with the lender's own
  best-approved group, and counts withdrawn and incomplete files as not
  denied. Groups under 100 applications are suppressed, not scored. This
  choice matters: counting lender decisions only, Black applicants fall
  below 0.8 of the Joint group nationally (0.786 against 0.850), as do
  three other named race groups. Both sets of national ratios are in
  `results/`.
- The recourse study (how much an applicant would need to change to be
  approved) did not give stable answers across seeds, and it is reported as
  a negative result. See [`docs/RECOURSE.md`](docs/RECOURSE.md).
- The national mitigation figures were recorded by hand without their
  console output. The dashboard does not chart them.

More in [`docs/LIMITS.md`](docs/LIMITS.md) and the model card,
[`docs/MODEL-CARD.md`](docs/MODEL-CARD.md).

## About this repository

The development history was squashed into a single commit before
publishing. The work began on 2026-09-11: the first line of
`logs/ingest_all_20260911_145810.log` is the national ingest starting at
2026-09-11T18:58:10Z.

## License

MIT. See [`LICENSE`](LICENSE). The HMDA data is public and comes from the
FFIEC; it is not redistributed here.
