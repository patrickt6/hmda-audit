#!/usr/bin/env python
"""National ingest runner: loops every state x {2023, 2024, 2025}.

The national runner: loops all
states x {2023,2024,2025}, is safe to run detached with nohup, writes
progress and errors to a log file with timestamps, and is interruptible/
resumable.

Usage (foreground, small test):
    .venv/bin/python scripts/ingest_all.py --years 2023

Usage (the real national run, detached; run this separately from
day-to-day testing):
    nohup .venv/bin/python scripts/ingest_all.py \
        > logs/ingest_all_$(date +%Y%m%d_%H%M%S).log 2>&1 &

Resumability: each state-year is a separate call into
``hmda.ingest.checksum.is_state_year_complete`` (via the same manifest
``download.py``/``checksum.py`` use), so re-running this script after an
interruption or a crash skips every state-year already recorded in
``data/manifest.json`` and only retries the ones that never finished. A
single state's failure is caught, logged, and the loop continues; it does
not abort the whole run: a failed state is retried alone,
not from the start.

This script is not run to completion during normal development. The
real national pass is launched separately, once ingest is confirmed
working end to end.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hmda.ingest.checksum import is_state_year_complete, record_completed_state_year
from hmda.ingest.download import all_states, fetch_state_year
from hmda.ingest.load_duckdb import csv_to_parquet, load_parquet_dir

DEFAULT_YEARS = (2023, 2024, 2025)


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(DEFAULT_YEARS),
        help="Activity years to ingest (default: 2023 2024 2025).",
    )
    parser.add_argument(
        "--states",
        type=str,
        nargs="+",
        default=None,
        help="States to restrict to (default: every state from all_states()).",
    )
    args = parser.parse_args()

    states = args.states or all_states()
    raw_dir = Path("data/raw")
    parquet_dir = Path("data/parquet")
    parquet_dir.mkdir(parents=True, exist_ok=True)

    total = len(states) * len(args.years)
    done = 0
    failed: list[tuple[int, str, str]] = []

    log(f"ingest_all starting: {len(states)} states x {len(args.years)} years = {total} state-years")

    for year in args.years:
        for state in states:
            done += 1
            if is_state_year_complete(year, state):
                log(f"[{done}/{total}] {state} {year}: already complete, skipping")
                continue

            log(f"[{done}/{total}] {state} {year}: fetching")
            try:
                start = time.monotonic()
                result = fetch_state_year(year, state, raw_dir)
                parquet_path = parquet_dir / f"{state}_{year}.parquet"
                row_count = csv_to_parquet(result.csv_path, parquet_path)
                result.csv_path.unlink()  # stream: delete CSV immediately

                record_completed_state_year(
                    year=year,
                    state=state,
                    url=result.url,
                    sha256=result.sha256,
                    size_bytes=result.size_bytes,
                    elapsed_seconds=result.elapsed_seconds,
                    row_count=row_count,
                    parquet_path=parquet_path,
                )
                elapsed = time.monotonic() - start
                log(
                    f"[{done}/{total}] {state} {year}: OK rows={row_count} "
                    f"size_bytes={result.size_bytes} elapsed_s={elapsed:.2f}"
                )
            except Exception as exc:  # noqa: BLE001 - log and continue, never abort the run
                log(f"[{done}/{total}] {state} {year}: FAILED - {exc!r}")
                failed.append((year, state, repr(exc)))
                continue

    log(f"ingest_all: rebuilding data/hmda.duckdb from data/parquet/")
    row_total = load_parquet_dir(parquet_dir)
    log(f"ingest_all: done. {row_total} total rows loaded. {len(failed)} state-years failed.")
    for year, state, err in failed:
        log(f"  FAILED: {year} {state}: {err}")


if __name__ == "__main__":
    main()
