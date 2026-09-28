"""Check that Spark SQL on Databricks and DuckDB on a laptop give the same
per-lender, per-race counts on the full national file.

    .venv/bin/python databricks/compare_national.py <warehouse_id>

Runs sql/air_by_lei_group.sql (group column derived_race) both ways and
compares every (lei, group) row: application count and denials. Uses the
Databricks CLI (`databricks api`) for the Spark side, so it needs a logged-in
profile. Prints aggregate counts only, no lender identities.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
SQL = (ROOT / "sql" / "air_by_lei_group.sql").read_text().replace("{group_column}", "derived_race")
SQL = "\n".join(line for line in SQL.splitlines() if not line.lstrip().startswith("--"))


def _api(method: str, path: str, body: dict | None = None) -> dict:
    cmd = ["databricks", "api", method, path]
    if body is not None:
        cmd += ["--json", json.dumps(body)]
    return json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)


def spark_rows(warehouse_id: str) -> dict[tuple[str, str], tuple[int, int]]:
    # Spark: the frame is the national Delta table; action_taken is text there,
    # so compare as text, exactly as DuckDB coerces it.
    stmt = SQL.replace("from frame", "from workspace.hmda.lar_national")
    stmt = stmt.replace("action_taken = 3", "action_taken = '3'").replace("action_taken != 6", "action_taken != '6'")
    r = _api("post", "/api/2.0/sql/statements", {
        "warehouse_id": warehouse_id, "statement": stmt, "wait_timeout": "50s",
        "disposition": "INLINE", "format": "JSON_ARRAY"})
    sid = r["statement_id"]
    while r["status"]["state"] in ("PENDING", "RUNNING"):
        time.sleep(3)
        r = _api("get", f"/api/2.0/sql/statements/{sid}")
    if r["status"]["state"] != "SUCCEEDED":
        raise SystemExit(f"Spark query failed: {r['status']}")
    data = list(r["result"].get("data_array", []))
    chunk = r["result"]
    while chunk.get("next_chunk_index") is not None:
        chunk = _api("get", f"/api/2.0/sql/statements/{sid}/result/chunks/{chunk['next_chunk_index']}")
        data += chunk.get("data_array", [])
    return {(row[0], row[1]): (int(row[2]), int(row[3])) for row in data}


def duckdb_rows() -> dict[tuple[str, str], tuple[int, int]]:
    con = duckdb.connect()
    con.execute(f"create view frame as select * from read_parquet('{ROOT}/data/parquet/*.parquet')")
    return {(r[0], r[1]): (int(r[2]), int(r[3])) for r in con.execute(SQL).fetchall()}


def main() -> int:
    wh = sys.argv[1]
    t0 = time.time(); s = spark_rows(wh); ts = time.time() - t0
    t0 = time.time(); d = duckdb_rows(); td = time.time() - t0
    only_s, only_d = s.keys() - d.keys(), d.keys() - s.keys()
    diff = [k for k in s.keys() & d.keys() if s[k] != d[k]]
    summary = {
        "rows_spark": len(s), "rows_duckdb": len(d),
        "only_in_spark": len(only_s), "only_in_duckdb": len(only_d), "value_mismatches": len(diff),
        "total_applications_spark": sum(v[0] for v in s.values()),
        "total_applications_duckdb": sum(v[0] for v in d.values()),
        "total_denials_spark": sum(v[1] for v in s.values()),
        "total_denials_duckdb": sum(v[1] for v in d.values()),
        "seconds_spark_incl_api": round(ts, 1), "seconds_duckdb": round(td, 1),
    }
    print(json.dumps(summary, indent=2))
    out = dict(summary)
    out["checked_on"] = date.today().isoformat()
    out["engine_a"] = "Spark SQL on Databricks (Delta table workspace.hmda.lar_national)"
    out["engine_b"] = "DuckDB on local parquet"
    (ROOT / "results" / "spark_parity.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0 if not (only_s or only_d or diff) else 1


if __name__ == "__main__":
    raise SystemExit(main())
