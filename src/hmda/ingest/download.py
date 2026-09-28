"""Download HMDA LAR CSVs from the FFIEC Data Browser API, per state-year.

MEASURED (2026-09-11): the working endpoint is
``https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?years=<YEAR>&states=<ST>``,
answers HTTP 301 and redirects to ``files.ffiec.cfpb.gov`` (``curl -L
--compressed`` required). Confirmed years: 2018, 2020, 2023, 2024, 2025; the
three most recent are 2023/2024/2025. The S3 snapshot path
(``s3.amazonaws.com/cfpb-hmda-public/...``) returned 403 for all four years
tried and must NOT be used.

Disk constraint: 36 GB free measured 2026-09-11, an estimated
13-19 GB for three CSV years. The ingest streams per state-year: CSV ->
parquet -> delete CSV. Peak disk use over a full run must stay under
5 GB.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import requests

FFIEC_CSV_URL = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"

# All 50 states + DC + the 5 populated territories HMDA covers, plus "PR".
# This is the loop set for scripts/ingest_all.py. Small states are fetched
# first in local testing (DC, VT, WY) but the national runner walks
# the whole list.
US_STATES: tuple[str, ...] = (
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
    "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
    "WV", "WI", "WY", "PR", "GU", "AS", "VI", "MP",
)

MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 2.0


@dataclass(frozen=True)
class DownloadResult:
    """One completed state-year download.

    url: the exact URL fetched (after following the redirect), so it can be
        printed verbatim, so the URL and SHA-256 always print together.
    year: the HMDA activity year requested.
    state: the two-letter state code requested.
    csv_path: where the CSV was written before conversion (deleted after
        the parquet conversion succeeds, per the streaming requirement).
    size_bytes: size of the downloaded CSV, in bytes.
    sha256: hex digest of the downloaded CSV.
    elapsed_seconds: wall-clock time the download took, for a download-rate
        estimate (bytes / elapsed_seconds).
    """

    url: str
    year: int
    state: str
    csv_path: Path
    size_bytes: int
    sha256: str
    elapsed_seconds: float


def fetch_state_year(year: int, state: str, dest_dir: Path) -> DownloadResult:
    """Download one state-year LAR CSV into ``dest_dir``, following redirects.

    Uses ``requests`` with ``stream=True`` (handles redirects and gzip
    negotiation, the ``requests`` equivalent of ``curl -L --compressed``),
    streams the response body directly to disk in chunks (never loads the
    whole file into memory), and computes the SHA-256 incrementally over
    the same chunks. Retries with exponential backoff
    (:data:`BACKOFF_BASE_SECONDS` * 2**attempt) up to :data:`MAX_RETRIES` on
    a network error or non-200 final status.

    Does not delete its own output; the caller converts to parquet and
    deletes the CSV (streaming requirement).
    """
    import hashlib

    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dest_dir / f"{state}_{year}.csv"
    params = {"years": year, "states": state}

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        start = time.monotonic()
        try:
            with requests.get(
                FFIEC_CSV_URL,
                params=params,
                stream=True,
                allow_redirects=True,
                headers={"Accept-Encoding": "gzip, deflate"},
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                resolved_url = resp.url
                hasher = hashlib.sha256()
                size_bytes = 0
                with open(csv_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        hasher.update(chunk)
                        size_bytes += len(chunk)
            elapsed = time.monotonic() - start
            return DownloadResult(
                url=resolved_url,
                year=year,
                state=state,
                csv_path=csv_path,
                size_bytes=size_bytes,
                sha256=hasher.hexdigest(),
                elapsed_seconds=elapsed,
            )
        except (requests.RequestException, OSError) as exc:
            last_exc = exc
            csv_path.unlink(missing_ok=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))
            continue

    raise RuntimeError(
        f"fetch_state_year({year}, {state}) failed after {MAX_RETRIES} attempts"
    ) from last_exc


def all_states() -> list[str]:
    """Return the list of two-letter state/territory codes to loop over.

    The ingest is per-state, not one national call: the tested endpoint
    takes a states parameter, so the loop writes one parquet per
    state-year, then unions in DuckDB. This is also the resumable
    design: a failed state is retried alone, not from the start, and
    does not restart the whole run.
    """
    return list(US_STATES)
