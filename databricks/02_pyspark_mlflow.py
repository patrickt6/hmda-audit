# Databricks notebook source
# MAGIC %md
# MAGIC # HMDA audit on Databricks: PySpark four-fifths check + models tracked with MLflow
# MAGIC
# MAGIC What this notebook shows:
# MAGIC 1. The four-fifths screen written in **PySpark** (the DataFrame API), not SQL.
# MAGIC    It must give the same numbers as `01_four_fifths.sql` and the DuckDB answer key.
# MAGIC 2. The repo's own model code (`hmda.model.evaluate.run_eval`) trained on data read
# MAGIC    from the **Delta table**, with each model's result logged to **MLflow**.
# MAGIC
# MAGIC Data: `workspace.hmda.lar_50k` (the 50,000-row test sample).

# COMMAND ----------

# MAGIC %pip install scikit-learn==1.9.1 lightgbm==4.7.0 --quiet

# COMMAND ----------

# Restart Python so the versions installed above are the ones imported.
dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import functions as F

# spark.table reads a Delta table by its three-part name: catalog.schema.table.
lar = spark.table("workspace.hmda.lar_50k")

# Keep decided applications only. Code 6 = purchased loan (no decision), so drop it.
decided = lar.filter(F.col("action_taken") != "6")

# One row per race group: size, denials (code 3), approval rate.
by_group = (
    decided.groupBy(F.col("derived_race").alias("grp"))
    .agg(
        F.count("*").alias("n"),
        F.sum(F.when(F.col("action_taken") == "3", 1).otherwise(0)).alias("denials"),
    )
    .withColumn("approval_rate", 1.0 - F.col("denials") / F.col("n"))
)

# The reference group's approval rate, pulled out as a plain Python number.
white_rate = by_group.filter(F.col("grp") == "White").first()["approval_rate"]

# Four-fifths ratio. Below 0.8 means "flag for review" (a flag is not a finding).
air = (
    by_group.withColumn("air", F.col("approval_rate") / F.lit(white_rate))
    .withColumn("flag", F.col("air") < 0.8)
    .orderBy(F.col("n").desc())
)
display(air)

# COMMAND ----------

# Check against the DuckDB answer key (databricks/answer_key_50k.md). Values copied from it.
expected = {
    "White": (27426, 4152, 1.0),
    "Race Not Available": (9166, 1791, 0.9481),
    "Black or African American": (4997, 1390, 0.8506),
    "Asian": (1201, 176, 1.0057),
    "Joint": (1075, 133, 1.0326),
    "American Indian or Alaska Native": (263, 82, 0.811),
    "2 or more minority races": (101, 23, 0.91),
    "Native Hawaiian or Other Pacific Islander": (58, 20, 0.7721),
    "Free Form Text Only": (13, 6, 0.6345),
}
got = {r["grp"]: (r["n"], r["denials"], round(r["air"], 4)) for r in air.collect()}
assert got == expected, f"mismatch: {got}"
print("PySpark four-fifths matches the DuckDB answer key for all", len(got), "groups")

# COMMAND ----------

# Use the repo's real model code. Its source was uploaded to the volume as a zip.
import sys, zipfile, shutil, os

src_dir = "/tmp/hmda_src"
shutil.rmtree(src_dir, ignore_errors=True)
zipfile.ZipFile("/Volumes/workspace/hmda/raw/hmda_src.zip").extractall(src_dir)
sys.path.insert(0, src_dir)

import sklearn, lightgbm
from hmda.model.evaluate import run_eval
from hmda.model.translate import percent_better_than_baseline

print("scikit-learn", sklearn.__version__, "| lightgbm", lightgbm.__version__)

# COMMAND ----------

import mlflow

# Serverless compute blocks the Spark setting MLflow reads by default, so set the
# tracking server and model registry explicitly. Turn off autologging: we log on purpose below.
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.autolog(disable=True)

# Spark does the reading; the models are scikit-learn / LightGBM, which need pandas.
# 50k rows fits in memory easily. (At full scale you would sample first, as the repo does.)
raw = lar.toPandas()

result = run_eval(frame=raw, source_label="workspace.hmda.lar_50k (Delta, 50k test sample)")

# The experiment lives in your own workspace folder; current_user() gives your user name.
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/hmda-audit/hmda_models")

for name, key in [("logistic_regression", "logistic_result"), ("lightgbm_gbm", "gbm_result")]:
    r = result[key]
    better = percent_better_than_baseline(r.challenger_auc, r.baseline_auc)
    with mlflow.start_run(run_name=name):
        mlflow.log_params({
            "data": "workspace.hmda.lar_50k",
            "split_year": result["split_year"],
            "rows_train": result["rows_train"],
            "rows_test": result["rows_test"],
            "n_features": result["n_features"],
        })
        mlflow.log_metrics({
            "auc": r.challenger_auc,
            "baseline_auc": r.baseline_auc,
            "pct_better_than_baseline": better,
        })
    print(f"{name}: AUC {r.challenger_auc:.6f}, {better:.1f}% better than base-rate baseline")

# COMMAND ----------

# Check against the local run of the same code on the same 50k rows (laptop, 2026-09-28):
# logistic 0.803754, gbm 0.875008.
assert abs(result["logistic_result"].challenger_auc - 0.803754) < 1e-4
assert abs(result["gbm_result"].challenger_auc - 0.875008) < 1e-4
print("Databricks model results match the local run")
