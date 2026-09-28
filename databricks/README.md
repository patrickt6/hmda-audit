# HMDA audit on Databricks

The DuckDB audit, ported to Databricks and checked against the original.
Built 2026-09-28 on the 50,000-row test sample (`tests/fixtures/hmda_50k.parquet`).

## What is where

| Thing | Where |
|---|---|
| Workspace | Databricks Free Edition (any workspace works) |
| Raw file (volume) | `/Volumes/workspace/hmda/raw/hmda_50k.parquet` |
| Delta table | `workspace.hmda.lar_50k` (catalog `workspace`, schema `hmda`) |
| Four-fifths in Spark SQL | `01_four_fifths.sql` (run on the "Serverless Starter Warehouse") |
| PySpark + MLflow notebook | `02_pyspark_mlflow.py`, imported to `/Users/<you>/hmda-audit/02_pyspark_mlflow` |
| MLflow experiment | `/Users/<you>/hmda-audit/hmda_models` (runs `logistic_regression`, `lightgbm_gbm`) |
| DuckDB reference numbers | `answer_key_50k.md` |

## What was checked

1. **Spark SQL = DuckDB.** `01_four_fifths.sql` gave the same n, denials, approval rate and ratio for all 9 race groups as `answer_key_50k.md` (output saved in `spark_sql_out.txt`).
2. **PySpark = DuckDB.** The notebook's `assert got == expected` passed (job run 701948730687956, TERMINATED SUCCESS, 2026-09-28).
3. **Models on Databricks = models on the laptop.** The notebook runs the repo's own `hmda.model.evaluate.run_eval` on the Delta table. AUC: logistic 0.8038, LightGBM 0.8750, the same as the local run to 1e-4 (60.8% and 75.0% better than the base-rate baseline). These are 50k-sample numbers. The headline national numbers (57% / 72%) come from the 1.5M-row sample, so they differ on purpose.

## National scale (2026-09-28)

All 168 state-year parquet files (36,734,685 rows, 5,329 lenders) were uploaded to
`/Volumes/workspace/hmda/raw/national/` and loaded into the Delta table
`workspace.hmda.lar_national`. `compare_national.py` runs `sql/air_by_lei_group.sql`
in Spark SQL on that table and in DuckDB on the local files, then compares every
(lender, race) row:

| | Spark SQL | DuckDB |
|---|---|---|
| (lender, race) rows | 31,793 | 31,793 |
| Applications (purchased loans excluded) | 32,620,789 | 32,620,789 |
| Denials | 6,166,654 | 6,166,654 |
| Rows that differ | 0 | |

The lender watch list (`scripts/watch_list.py`, `src/hmda/fairness/shrink.py`) is
computed from these per-lender counts, so it is the same on both engines.

## Walkthrough

1. **Terms.**
   - *Delta table*: parquet files plus a transaction log, so the table has versions, and updates are safe.
   - *Unity Catalog three-part name*: `catalog.schema.table`, for example `workspace.hmda.lar_50k`.
   - *SQL warehouse* runs SQL. *Serverless compute* runs notebooks. Both start when needed.
   - *Spark*: runs a query in parallel across machines. Here the data is small, but the code is the same at 36.7M rows.
   - *MLflow*: records each training run (settings, metrics, model), so a reviewer can see what was trained and how well it did.
2. **Open `01_four_fifths.sql` in the SQL editor and run it.** Read the comments line by line. Change `'White'` to `'Asian'` and see how the ratios move.
3. **Open the notebook and run it cell by cell.** Compare the PySpark cell with the SQL file. `filter` = WHERE, `groupBy` + `agg` = GROUP BY, `withColumn` = a new column in SELECT.
4. **Open the MLflow experiment** (left bar, Experiments). Click a run and find the AUC.

## FAQ
- Why exclude `action_taken = 6`? (A purchased loan is not an application; the lender made no decision.)
- Why is a ratio below 0.8 a flag, not a finding? (It does not control for income, DTI or loan type. The controlled model does that.)
- Two small groups (58 and 13 rows) are flagged. Is that meaningful? (Probably not; the repo sets a minimum group size for this reason.)
- Why use `toPandas()` before training? (scikit-learn and LightGBM run on one machine with pandas; 50k rows fits. At full scale, sample first or use Spark ML.)
- What does AUC 0.80 mean? (Pick one denied and one approved application at random. 80 times in 100, the model gives the denied one the higher risk score.)

## Serverless note
MLflow on serverless compute could not read `spark.mlflow.modelRegistryUri`, so the notebook sets the tracking and registry URIs explicitly and turns off autologging.
