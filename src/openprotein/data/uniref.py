"""Deterministic, content-addressed UniRef FASTA to CSV conversion."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterator, List, Optional, TextIO, Tuple


ALLOWED_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")
SPLIT_BUCKETS = 10_000


class _DigestingRawReader(io.RawIOBase):
    def __init__(self, handle: BinaryIO) -> None:
        self.handle = handle
        self.sha256 = hashlib.sha256()
        self.md5 = hashlib.md5()  # noqa: S324 - required to verify the upstream receipt
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        chunk = self.handle.read(len(buffer))
        size = len(chunk)
        if size:
            buffer[:size] = chunk
            self.sha256.update(chunk)
            self.md5.update(chunk)
            self.bytes_read += size
        return size


class _HashingRawWriter(io.RawIOBase):
    def __init__(self, handle: BinaryIO) -> None:
        self.handle = handle
        self.sha256 = hashlib.sha256()
        self.bytes_written = 0

    def writable(self) -> bool:
        return True

    def write(self, data: bytes) -> int:
        written = self.handle.write(data)
        if written:
            self.sha256.update(data[:written])
            self.bytes_written += written
        return written

    def flush(self) -> None:
        self.handle.flush()

    def close(self) -> None:
        if not self.closed:
            try:
                super().close()
            finally:
                self.handle.close()


class _CSVShardWriter:
    def __init__(self, path: Path, relative_path: str, split: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.relative_path = relative_path
        self.split = split
        self.binary = path.open("wb")
        self.raw = _HashingRawWriter(self.binary)
        self.buffered = io.BufferedWriter(self.raw, buffer_size=1024 * 1024)
        self.text = io.TextIOWrapper(self.buffered, encoding="utf-8", newline="")
        self.writer = csv.DictWriter(
            self.text, fieldnames=("id", "name", "text"), lineterminator="\n"
        )
        self.writer.writeheader()
        self.records = 0
        self.residues = 0
        self.min_length: Optional[int] = None
        self.max_length = 0
        self.closed = False

    def write(self, record_id: str, name: str, sequence: str) -> None:
        self.writer.writerow({"id": record_id, "name": name, "text": sequence})
        length = len(sequence)
        self.records += 1
        self.residues += length
        self.min_length = length if self.min_length is None else min(self.min_length, length)
        self.max_length = max(self.max_length, length)

    def close(self) -> Dict[str, Any]:
        if not self.closed:
            self.text.close()
            self.closed = True
        return {
            "path": self.relative_path,
            "split": self.split,
            "bytes": self.raw.bytes_written,
            "sha256": self.raw.sha256.hexdigest(),
            "records": self.records,
            "residues": self.residues,
            "min_length": 0 if self.min_length is None else self.min_length,
            "max_length": self.max_length,
            "invalid_sequences": 0,
        }


def iter_fasta(handle: TextIO) -> Iterator[Tuple[str, str, str]]:
    """Yield ``(record_id, description, sequence)`` from a FASTA stream."""

    header: Optional[str] = None
    sequence_parts: List[str] = []
    for line_number, raw_line in enumerate(handle, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                record_id, _, description = header.partition(" ")
                yield record_id, description, "".join(sequence_parts).upper()
            header = line[1:].strip()
            if not header:
                raise ValueError(f"empty FASTA header at line {line_number}")
            sequence_parts = []
        else:
            if header is None:
                raise ValueError(f"sequence appears before the first FASTA header at line {line_number}")
            sequence_parts.append(line)
    if header is not None:
        record_id, _, description = header.partition(" ")
        yield record_id, description, "".join(sequence_parts).upper()


def _assignment(
    record_id: str,
    train_basis_points: int,
    validation_basis_points: int,
    shard_counts: Dict[str, int],
) -> Tuple[str, int]:
    digest = hashlib.sha256(record_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % SPLIT_BUCKETS
    if bucket < train_basis_points:
        split = "train"
    elif bucket < train_basis_points + validation_basis_points:
        split = "validation"
    else:
        split = "test"
    shard = int.from_bytes(digest[8:16], "big") % shard_counts[split]
    return split, shard


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build_uniref_csv(
    source_fasta: str,
    output_root: str,
    source_definition_path: str,
    shard_counts: Optional[Dict[str, int]] = None,
    train_basis_points: int = 9_800,
    validation_basis_points: int = 100,
    min_length: int = 1,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Build deterministic CSV shards and fail closed on source-receipt drift."""

    counts = shard_counts or {"train": 96, "validation": 1, "test": 1}
    if set(counts) != {"train", "validation", "test"} or any(
        int(value) <= 0 for value in counts.values()
    ):
        raise ValueError("shard_counts must contain positive train, validation, and test counts")
    if not 0 < train_basis_points < SPLIT_BUCKETS:
        raise ValueError("train_basis_points must be in (0, 10000)")
    if validation_basis_points <= 0 or train_basis_points + validation_basis_points >= SPLIT_BUCKETS:
        raise ValueError("the validation and test splits must both be non-empty")
    if min_length <= 0 or (max_length is not None and max_length < min_length):
        raise ValueError("invalid sequence-length filter")

    source_path = Path(source_fasta).resolve()
    final_root = Path(output_root).resolve()
    if final_root.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset root: {final_root}")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{final_root.name}.tmp-", dir=final_root.parent)
    )
    source_definition = json.loads(Path(source_definition_path).read_text(encoding="utf-8"))
    writers: Dict[Tuple[str, int], _CSVShardWriter] = {}
    files: List[Dict[str, Any]] = []
    records_seen = 0
    filter_totals = {"empty": 0, "invalid_residue": 0, "too_short": 0, "too_long": 0}
    input_reader: Optional[_DigestingRawReader] = None

    try:
        for split in ("train", "validation", "test"):
            for shard in range(int(counts[split])):
                relative = f"{split}/{split}-{shard:05d}-of-{int(counts[split]):05d}.csv"
                writers[(split, shard)] = _CSVShardWriter(
                    temporary_root / relative, relative, split
                )

        with source_path.open("rb") as binary:
            input_reader = _DigestingRawReader(binary)
            buffered = io.BufferedReader(input_reader, buffer_size=1024 * 1024)
            with gzip.GzipFile(fileobj=buffered, mode="rb") as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    for record_id, description, sequence in iter_fasta(text):
                        records_seen += 1
                        if not sequence:
                            filter_totals["empty"] += 1
                            continue
                        if not set(sequence) <= ALLOWED_RESIDUES:
                            filter_totals["invalid_residue"] += 1
                            continue
                        if len(sequence) < min_length:
                            filter_totals["too_short"] += 1
                            continue
                        if max_length is not None and len(sequence) > max_length:
                            filter_totals["too_long"] += 1
                            continue
                        split, shard = _assignment(
                            record_id, train_basis_points, validation_basis_points, counts
                        )
                        writers[(split, shard)].write(record_id, description, sequence)
            if input_reader.bytes_read < source_path.stat().st_size:
                while buffered.read(1024 * 1024):
                    pass

        for key in sorted(writers):
            files.append(writers[key].close())

        expected_bytes = int(source_definition["fasta_bytes"])
        expected_records = int(source_definition["records"])
        observed_sha256 = input_reader.sha256.hexdigest()
        observed_md5 = input_reader.md5.hexdigest()
        if input_reader.bytes_read != expected_bytes:
            raise RuntimeError(
                f"source byte count mismatch: {input_reader.bytes_read} != {expected_bytes}"
            )
        if observed_md5 != source_definition["fasta_md5"]:
            raise RuntimeError("source MD5 does not match the pinned release receipt")
        if observed_sha256 != source_definition["fasta_sha256"]:
            raise RuntimeError("source SHA-256 does not match the local retrieval receipt")
        if records_seen != expected_records:
            raise RuntimeError(f"FASTA record count mismatch: {records_seen} != {expected_records}")

        split_totals: Dict[str, Dict[str, int]] = {}
        for split in ("train", "validation", "test"):
            selected = [item for item in files if item["split"] == split]
            split_totals[split] = {
                "files": len(selected),
                "bytes": sum(int(item["bytes"]) for item in selected),
                "records": sum(int(item["records"]) for item in selected),
                "residues": sum(int(item["residues"]) for item in selected),
            }

        manifest: Dict[str, Any] = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_root": str(final_root),
            "required_columns": ["id", "name", "text"],
            "source": source_definition,
            "source_input": {
                "path": str(source_path),
                "bytes": input_reader.bytes_read,
                "md5": observed_md5,
                "sha256": observed_sha256,
                "records_seen": records_seen,
            },
            "split_policy": {
                "algorithm": "sha256(record_id), first uint64 modulo 10000",
                "train_basis_points": train_basis_points,
                "validation_basis_points": validation_basis_points,
                "test_basis_points": SPLIT_BUCKETS
                - train_basis_points
                - validation_basis_points,
                "shard_algorithm": "sha256(record_id), second uint64 modulo split shard count",
                "shard_counts": counts,
            },
            "filters": {
                "allowed_residues": "".join(sorted(ALLOWED_RESIDUES)),
                "min_length": min_length,
                "max_length": max_length,
                "dropped": filter_totals,
            },
            "split_totals": split_totals,
            "files": files,
            "test_policy": "sealed: never used for optimization or checkpoint selection",
        }
        _atomic_json(temporary_root / "manifest.json", manifest)
        os.replace(temporary_root, final_root)
        return manifest
    except Exception:
        for writer in writers.values():
            if not writer.closed:
                writer.close()
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
