"""SHA-256 checksums for every downloaded file, and the manifest they land in.

Every number the ingest step reports must be traceable: the exact URL, the
file size, the SHA-256, the publication year, and the schema version. This
module owns computing and recording that checksum; it does not own the
download itself (see ``download.py``).

The manifest is also the resumability ledger: ``ingest`` re-runs consult it
to skip a state-year whose parquet + sha256 are already recorded, so a
re-run re-downloads nothing and exits 0.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_MANIFEST_PATH = Path("data/manifest.json")


@dataclass(frozen=True)
class ChecksumRecord:
    """One file's recorded checksum, for the ingest manifest.

    path: the file checksummed.
    sha256: hex digest.
    size_bytes: file size in bytes at the time of checksumming.
    """

    path: Path
    sha256: str
    size_bytes: int


def sha256_file(path: Path) -> ChecksumRecord:
    """Compute the SHA-256 of ``path`` by streaming it, without loading it whole into memory.

    Required for large per-state CSVs (some tens of MB) and must not read
    the whole file into RAM at once, to stay inside the streaming
    disk budget.
    """
    hasher = hashlib.sha256()
    size_bytes = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            size_bytes += len(chunk)
    return ChecksumRecord(path=path, sha256=hasher.hexdigest(), size_bytes=size_bytes)


def write_manifest(records: list[ChecksumRecord], manifest_path: Path) -> None:
    """Write ``records`` to ``manifest_path`` as a reviewable, diffable file (JSON lines).

    This manifest is the source the reproducibility check (M8) compares
    against on a second run: the same inputs must produce the same
    manifest hash.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        for rec in records:
            row = asdict(rec)
            row["path"] = str(row["path"])
            f.write(json.dumps(row, sort_keys=True) + "\n")


# --------------------------------------------------------------------------
# The state-year completion manifest (resumability + idempotency).
#
# This is a separate, richer manifest from write_manifest() above: it
# records one JSON object per completed state-year (year, state, url,
# sha256, size_bytes, elapsed_seconds, row_count, parquet_path), keyed so a
# re-run of `hmda ingest --year Y --state ST` can look itself up and skip
# the download entirely if the parquet already exists on disk with a
# recorded sha256. This makes the download step idempotent.
# --------------------------------------------------------------------------


def load_state_manifest(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> dict:
    """Load the state-year completion manifest as {"YEAR:STATE": {...}}.

    Returns an empty dict if the manifest does not exist yet (first run).
    """
    if not manifest_path.exists():
        return {}
    with open(manifest_path) as f:
        return json.load(f)


def record_completed_state_year(
    year: int,
    state: str,
    url: str,
    sha256: str,
    size_bytes: int,
    elapsed_seconds: float,
    row_count: int,
    parquet_path: Path,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> None:
    """Append/overwrite one state-year's completion record in the manifest.

    Called only after the parquet conversion has succeeded and the CSV has
    been deleted, so a manifest entry always means "safe to skip on re-run".
    """
    data = load_state_manifest(manifest_path)
    key = f"{year}:{state}"
    data[key] = {
        "year": year,
        "state": state,
        "url": url,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "elapsed_seconds": elapsed_seconds,
        "row_count": row_count,
        "parquet_path": str(parquet_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def is_state_year_complete(
    year: int, state: str, manifest_path: Path = DEFAULT_MANIFEST_PATH
) -> bool:
    """True if ``year``/``state`` has a manifest record AND its parquet file exists on disk."""
    data = load_state_manifest(manifest_path)
    key = f"{year}:{state}"
    if key not in data:
        return False
    parquet_path = Path(data[key]["parquet_path"])
    return parquet_path.exists()
