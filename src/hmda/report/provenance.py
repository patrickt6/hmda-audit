"""Report-level provenance manifest: every figure/table's lineage, and the reproducibility manifest.

Wraps ``hmda.governance.lineage`` for the report
build, and writes the output-hash manifest the reproducibility check (M8) compares on a second run.
"""

from __future__ import annotations

from pathlib import Path

MANIFEST_PATH = Path("out/manifest.json")


def write_manifest(lineage_records, out_path: Path = MANIFEST_PATH) -> Path:
    """Write a JSON manifest of output-file hashes plus their lineage records to ``out_path``.

    A second ``make reproduce`` run must produce a byte-identical manifest,
    so this function must not embed a wall-clock
    timestamp or any other non-deterministic value.
    """
    raise NotImplementedError


def manifest_hash(manifest_path: Path = MANIFEST_PATH) -> str:
    """Return the SHA-256 of the manifest file at ``manifest_path``."""
    raise NotImplementedError
