"""Content-addressed manifests for public protein CSV shards."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, List, Tuple


AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO-.*")
REQUIRED_COLUMNS = ("id", "name", "text")


class _HashingRawReader(io.RawIOBase):
    def __init__(self, handle: BinaryIO, hasher: "hashlib._Hash") -> None:
        self.handle = handle
        self.hasher = hasher

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        chunk = self.handle.read(len(buffer))
        size = len(chunk)
        if size:
            buffer[:size] = chunk
            self.hasher.update(chunk)
        return size


def _scan_csv(path: Path) -> Dict[str, Any]:
    hasher = hashlib.sha256()
    records = 0
    residues = 0
    min_length = None
    max_length = 0
    invalid_sequences = 0

    with path.open("rb") as binary:
        raw = _HashingRawReader(binary, hasher)
        with io.TextIOWrapper(io.BufferedReader(raw), encoding="utf-8", newline="") as text:
            reader = csv.DictReader(text)
            if reader.fieldnames is None:
                raise ValueError(f"{path} is missing a CSV header")
            missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
            if missing:
                raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
            for row in reader:
                sequence = row["text"].strip().upper()
                length = len(sequence)
                records += 1
                residues += length
                min_length = length if min_length is None else min(min_length, length)
                max_length = max(max_length, length)
                if not sequence or not set(sequence) <= AMINO_ACIDS:
                    invalid_sequences += 1

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hasher.hexdigest(),
        "records": records,
        "residues": residues,
        "min_length": 0 if min_length is None else min_length,
        "max_length": max_length,
        "invalid_sequences": invalid_sequences,
    }


def _iter_split_files(dataset_root: Path) -> Iterable[Tuple[str, Path]]:
    for split in ("train", "validation", "test"):
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.glob("*.csv")):
            yield split, path


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build_csv_manifest(
    dataset_root: str,
    output_path: str,
    source: Dict[str, Any],
) -> Dict[str, Any]:
    """Scan CSV files once, write an atomic manifest, and return it."""

    root = Path(dataset_root).resolve()
    files: List[Dict[str, Any]] = []
    split_totals: Dict[str, Dict[str, int]] = {}
    for split, path in _iter_split_files(root):
        receipt = _scan_csv(path)
        receipt["path"] = str(path.relative_to(root))
        receipt["split"] = split
        files.append(receipt)
        totals = split_totals.setdefault(
            split,
            {"files": 0, "bytes": 0, "records": 0, "residues": 0, "invalid_sequences": 0},
        )
        totals["files"] += 1
        for key in ("bytes", "records", "residues", "invalid_sequences"):
            totals[key] += int(receipt[key])

    if not files:
        raise ValueError(f"no CSV shards found under {root}")

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "required_columns": list(REQUIRED_COLUMNS),
        "source": source,
        "split_totals": split_totals,
        "files": files,
        "test_policy": "sealed: never used for optimization or checkpoint selection",
    }
    _atomic_json(Path(output_path), manifest)
    return manifest
