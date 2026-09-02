"""Content-addressed uint8 protein shards for high-throughput pretraining."""

from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from openprotein.data.streaming import ProteinMLMCollator
from openprotein.data.tokenizer import EsmcTokenizer
from openprotein.data.uniref import ALLOWED_RESIDUES, iter_fasta
from openprotein.provenance import sha256_file


PACKED_SCHEMA_VERSION = 1
INDEX_MAGIC = b"OPSEQIDX"
INDEX_HEADER = struct.Struct("<8sIIQQ")
INDEX_RECORD = struct.Struct("<QII")
IDENTIFIER_RECORD = struct.Struct("<Q")


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _source_file_map(manifest: Dict[str, Any], manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError(f"source manifest has no file list: {manifest_path}")
    result: Dict[str, Dict[str, Any]] = {}
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("source manifest contains an invalid file entry")
        result[item["path"]] = item
    return result


def build_packed_csv_dataset(
    input_paths: Iterable[str],
    output_root: str,
    tokenizer_json: str,
    source_manifest_path: str,
    source_root: str,
    split: str = "train",
) -> Dict[str, Any]:
    """Convert hash-bound CSV shards to indexed uint8 sequence shards."""

    paths = [Path(path).resolve() for path in input_paths]
    if not paths:
        raise ValueError("packed builder requires at least one input shard")
    if split != "train":
        raise ValueError("the pretraining packed builder accepts train split only")
    final_root = Path(output_root).resolve()
    if final_root.exists():
        raise FileExistsError(f"refusing to overwrite packed dataset: {final_root}")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{final_root.name}.tmp-", dir=final_root.parent)
    )
    tokenizer_path = Path(tokenizer_json).resolve()
    tokenizer = EsmcTokenizer.from_tokenizer_json(str(tokenizer_path))
    if max(tokenizer.vocabulary.values()) >= 256:
        raise ValueError("packed uint8 format requires a vocabulary smaller than 256")
    source_manifest_file = Path(source_manifest_path).resolve()
    source_manifest = json.loads(source_manifest_file.read_text(encoding="utf-8"))
    source_files = _source_file_map(source_manifest, source_manifest_file)
    source_dataset_root = Path(source_root).resolve()

    output_files: List[Dict[str, Any]] = []
    total_records = 0
    total_residues = 0
    total_unknown = 0
    try:
        for shard_number, input_path in enumerate(paths):
            try:
                source_relative = input_path.relative_to(source_dataset_root).as_posix()
            except ValueError as exc:
                raise ValueError(f"input is outside source_root: {input_path}") from exc
            source_entry = source_files.get(source_relative)
            if source_entry is None:
                raise RuntimeError(f"input is absent from source manifest: {source_relative}")
            if source_entry.get("split") != split:
                raise RuntimeError(
                    f"input shard is not in the declared train split: {source_relative}"
                )
            if input_path.stat().st_size != int(source_entry["bytes"]):
                raise RuntimeError(f"source shard size changed: {source_relative}")
            if sha256_file(input_path) != source_entry["sha256"]:
                raise RuntimeError(f"source shard hash changed: {source_relative}")

            stem = f"{split}-{shard_number:05d}-of-{len(paths):05d}"
            binary_relative = f"{split}/{stem}.bin"
            index_relative = f"{split}/{stem}.idx"
            binary_path = temporary_root / binary_relative
            index_path = temporary_root / index_relative
            binary_path.parent.mkdir(parents=True, exist_ok=True)
            records = 0
            residues = 0
            unknown_residues = 0
            minimum_length: Optional[int] = None
            maximum_length = 0
            with input_path.open("r", encoding="utf-8", newline="") as source, binary_path.open(
                "wb"
            ) as binary, index_path.open("w+b") as index:
                reader = csv.DictReader(source)
                if reader.fieldnames is None or "text" not in reader.fieldnames:
                    raise ValueError(f"CSV shard has no text column: {input_path}")
                index.write(
                    INDEX_HEADER.pack(
                        INDEX_MAGIC, PACKED_SCHEMA_VERSION, INDEX_RECORD.size, 0, 0
                    )
                )
                offset = 0
                for row_number, row in enumerate(reader, start=2):
                    sequence = row["text"].strip().upper()
                    if not sequence:
                        raise ValueError(f"empty sequence at {input_path}:{row_number}")
                    if len(sequence) > 0xFFFFFFFF:
                        raise ValueError(
                            "sequence exceeds packed uint32 length at "
                            f"{input_path}:{row_number}"
                        )
                    encoded = bytes(
                        tokenizer.vocabulary.get(residue, tokenizer.unk_token_id)
                        for residue in sequence
                    )
                    unknown_residues += encoded.count(tokenizer.unk_token_id)
                    binary.write(encoded)
                    index.write(INDEX_RECORD.pack(offset, len(encoded), 0))
                    offset += len(encoded)
                    records += 1
                    residues += len(encoded)
                    minimum_length = (
                        len(encoded)
                        if minimum_length is None
                        else min(minimum_length, len(encoded))
                    )
                    maximum_length = max(maximum_length, len(encoded))
                index.seek(0)
                index.write(
                    INDEX_HEADER.pack(
                        INDEX_MAGIC,
                        PACKED_SCHEMA_VERSION,
                        INDEX_RECORD.size,
                        records,
                        residues,
                    )
                )

            binary_sha256 = sha256_file(binary_path)
            index_sha256 = sha256_file(index_path)
            output_files.extend(
                [
                    {
                        "path": binary_relative,
                        "kind": "tokens",
                        "bytes": binary_path.stat().st_size,
                        "sha256": binary_sha256,
                        "records": records,
                        "residues": residues,
                        "unknown_residues": unknown_residues,
                        "min_length": minimum_length or 0,
                        "max_length": maximum_length,
                        "source_path": source_relative,
                        "source_sha256": source_entry["sha256"],
                    },
                    {
                        "path": index_relative,
                        "kind": "index",
                        "bytes": index_path.stat().st_size,
                        "sha256": index_sha256,
                        "records": records,
                        "residues": residues,
                        "token_path": binary_relative,
                    },
                ]
            )
            total_records += records
            total_residues += residues
            total_unknown += unknown_residues

        manifest: Dict[str, Any] = {
            "schema_version": PACKED_SCHEMA_VERSION,
            "format": "openprotein_uint8_sequence_index_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_root": str(final_root),
            "split": split,
            "test_data_accessed": False,
            "tokenizer": {
                "path": str(tokenizer_path),
                "sha256": sha256_file(tokenizer_path),
                "vocab_size": max(tokenizer.vocabulary.values()) + 1,
                "cls_token_id": tokenizer.cls_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "unk_token_id": tokenizer.unk_token_id,
                "mask_token_id": tokenizer.mask_token_id,
            },
            "source_manifest": {
                "path": str(source_manifest_file),
                "sha256": sha256_file(source_manifest_file),
                "dataset_root": str(source_dataset_root),
            },
            "totals": {
                "shards": len(paths),
                "records": total_records,
                "residues": total_residues,
                "unknown_residues": total_unknown,
            },
            "files": output_files,
        }
        _atomic_json(temporary_root / "manifest.json", manifest)
        os.replace(temporary_root, final_root)
        return manifest
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def verify_packed_manifest(
    manifest_path: str,
    expected_index_paths: Optional[Iterable[str]] = None,
    expected_tokenizer_sha256: Optional[str] = None,
    content_shard_index: int = 0,
    content_shard_count: int = 1,
) -> Dict[str, Any]:
    if content_shard_count < 1:
        raise ValueError("content_shard_count must be positive")
    if not 0 <= content_shard_index < content_shard_count:
        raise ValueError("content_shard_index must be in [0, content_shard_count)")
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PACKED_SCHEMA_VERSION:
        raise RuntimeError("packed manifest schema is unsupported")
    if manifest.get("format") != "openprotein_uint8_sequence_index_v1":
        raise RuntimeError("packed manifest format is unsupported")
    if manifest.get("split") != "train" or manifest.get("test_data_accessed") is not False:
        raise RuntimeError("packed pretraining manifest is not train-only")
    if expected_tokenizer_sha256 is not None and manifest.get("tokenizer", {}).get(
        "sha256"
    ) != expected_tokenizer_sha256:
        raise RuntimeError("packed manifest tokenizer differs from the model tokenizer")
    root = Path(manifest["dataset_root"]).resolve()
    index_paths = set()
    token_entries: Dict[str, Dict[str, Any]] = {}
    files = manifest.get("files", [])
    if not isinstance(files, list):
        raise RuntimeError("packed manifest file list is invalid")
    if any(
        not isinstance(entry, dict)
        or not isinstance(entry.get("path"), str)
        or not isinstance(entry.get("bytes"), int)
        or int(entry["bytes"]) < 0
        for entry in files
    ):
        raise RuntimeError("packed manifest contains an invalid file entry")
    verification_loads = [0] * content_shard_count
    verification_assignments: List[List[int]] = [
        [] for _ in range(content_shard_count)
    ]
    for file_index in sorted(
        range(len(files)),
        key=lambda index: (-int(files[index]["bytes"]), files[index]["path"]),
    ):
        assigned_rank = min(
            range(content_shard_count),
            key=lambda rank: (verification_loads[rank], rank),
        )
        verification_assignments[assigned_rank].append(file_index)
        verification_loads[assigned_rank] += int(files[file_index]["bytes"])
    verification_indices = set(verification_assignments[content_shard_index])
    for file_index, entry in enumerate(files):
        relative = entry.get("path")
        if not isinstance(relative, str):
            raise RuntimeError("packed manifest contains an invalid file path")
        candidate = root / relative
        if not candidate.is_file():
            raise RuntimeError(f"packed data file is missing: {relative}")
        if candidate.stat().st_size != int(entry["bytes"]):
            raise RuntimeError(f"packed data file size changed: {relative}")
        if (
            file_index in verification_indices
            and sha256_file(candidate) != entry["sha256"]
        ):
            raise RuntimeError(f"packed data file hash changed: {relative}")
        if entry.get("kind") == "index":
            index_paths.add(str(candidate.resolve()))
        elif entry.get("kind") == "tokens":
            token_entries[relative] = entry
        elif entry.get("kind") == "identifiers_u64":
            if int(entry["bytes"]) != int(entry["records"]) * IDENTIFIER_RECORD.size:
                raise RuntimeError(f"packed identifier byte count is inconsistent: {relative}")
        else:
            raise RuntimeError(f"packed manifest has unknown file kind: {relative}")
    if expected_index_paths is not None:
        expected = {str(Path(item).resolve()) for item in expected_index_paths}
        if index_paths != expected:
            raise RuntimeError("train_glob does not exactly match the packed manifest indices")
    if len(index_paths) != int(manifest["totals"]["shards"]):
        raise RuntimeError("packed manifest shard total is inconsistent")
    if len(token_entries) != len(index_paths):
        raise RuntimeError("packed manifest does not pair every index with a token shard")
    return manifest


class PackedSequenceShard:
    def __init__(self, index_path: str) -> None:
        self.index_path = Path(index_path).resolve()
        if self.index_path.suffix != ".idx":
            raise ValueError("packed sequence index must end in .idx")
        self.binary_path = self.index_path.with_suffix(".bin")
        if not self.binary_path.is_file():
            raise FileNotFoundError(f"packed token file is missing: {self.binary_path}")
        self.index_fd = os.open(self.index_path, os.O_RDONLY)
        self.binary_fd = os.open(self.binary_path, os.O_RDONLY)
        header = os.pread(self.index_fd, INDEX_HEADER.size, 0)
        if len(header) != INDEX_HEADER.size:
            raise RuntimeError("packed index header is truncated")
        magic, version, record_size, records, residues = INDEX_HEADER.unpack(header)
        if magic != INDEX_MAGIC or version != PACKED_SCHEMA_VERSION:
            raise RuntimeError("packed index header is unsupported")
        if record_size != INDEX_RECORD.size:
            raise RuntimeError("packed index record size differs")
        expected_bytes = INDEX_HEADER.size + records * INDEX_RECORD.size
        if self.index_path.stat().st_size != expected_bytes:
            raise RuntimeError("packed index byte count is inconsistent")
        if self.binary_path.stat().st_size != residues:
            raise RuntimeError("packed token byte count is inconsistent")
        self.records = records
        self.residues = residues

    def close(self) -> None:
        if self.index_fd >= 0:
            os.close(self.index_fd)
            self.index_fd = -1
        if self.binary_fd >= 0:
            os.close(self.binary_fd)
            self.binary_fd = -1

    def __del__(self) -> None:
        self.close()

    def read(self, index: int) -> bytes:
        if not 0 <= index < self.records:
            raise IndexError("packed sequence index is outside the shard")
        position = INDEX_HEADER.size + index * INDEX_RECORD.size
        raw = os.pread(self.index_fd, INDEX_RECORD.size, position)
        if len(raw) != INDEX_RECORD.size:
            raise RuntimeError("packed index record is truncated")
        offset, length, _ = INDEX_RECORD.unpack(raw)
        sequence = os.pread(self.binary_fd, length, offset)
        if len(sequence) != length:
            raise RuntimeError("packed token record is truncated")
        return sequence


class _DigestingRawReader(io.RawIOBase):
    def __init__(self, handle: Any) -> None:
        self.handle = handle
        self.sha256 = hashlib.sha256()
        self.md5 = hashlib.md5()  # noqa: S324 - upstream MGnify publishes MD5
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


class _SelectedPackedWriter:
    def __init__(self, root: Path, split: str, shard: int, shard_count: int) -> None:
        stem = f"{split}-{shard:05d}-of-{shard_count:05d}"
        self.binary_relative = f"{split}/{stem}.bin"
        self.index_relative = f"{split}/{stem}.idx"
        self.identifiers_relative = f"{split}/{stem}.ids"
        self.binary_path = root / self.binary_relative
        self.index_path = root / self.index_relative
        self.identifiers_path = root / self.identifiers_relative
        self.binary_path.parent.mkdir(parents=True, exist_ok=True)
        self.binary = self.binary_path.open("wb", buffering=256 * 1024)
        self.index = self.index_path.open("w+b", buffering=256 * 1024)
        self.identifiers = self.identifiers_path.open("wb", buffering=256 * 1024)
        self.index.write(
            INDEX_HEADER.pack(INDEX_MAGIC, PACKED_SCHEMA_VERSION, INDEX_RECORD.size, 0, 0)
        )
        self.records = 0
        self.residues = 0
        self.unknown_residues = 0
        self.minimum_length: Optional[int] = None
        self.maximum_length = 0
        self.closed = False

    def write(self, accession: int, encoded: bytes, unknown_token_id: int) -> None:
        self.binary.write(encoded)
        self.index.write(INDEX_RECORD.pack(self.residues, len(encoded), 0))
        self.identifiers.write(IDENTIFIER_RECORD.pack(accession))
        self.records += 1
        self.residues += len(encoded)
        self.unknown_residues += encoded.count(unknown_token_id)
        self.minimum_length = (
            len(encoded) if self.minimum_length is None else min(self.minimum_length, len(encoded))
        )
        self.maximum_length = max(self.maximum_length, len(encoded))

    def close(self) -> List[Dict[str, Any]]:
        if not self.closed:
            self.index.seek(0)
            self.index.write(
                INDEX_HEADER.pack(
                    INDEX_MAGIC,
                    PACKED_SCHEMA_VERSION,
                    INDEX_RECORD.size,
                    self.records,
                    self.residues,
                )
            )
            self.binary.close()
            self.index.close()
            self.identifiers.close()
            self.closed = True
        common = {
            "records": self.records,
            "residues": self.residues,
            "unknown_residues": self.unknown_residues,
            "min_length": self.minimum_length or 0,
            "max_length": self.maximum_length,
        }
        return [
            {
                "path": self.binary_relative,
                "kind": "tokens",
                "bytes": self.binary_path.stat().st_size,
                "sha256": sha256_file(self.binary_path),
                **common,
            },
            {
                "path": self.index_relative,
                "kind": "index",
                "bytes": self.index_path.stat().st_size,
                "sha256": sha256_file(self.index_path),
                "token_path": self.binary_relative,
                **common,
            },
            {
                "path": self.identifiers_relative,
                "kind": "identifiers_u64",
                "bytes": self.identifiers_path.stat().st_size,
                "sha256": sha256_file(self.identifiers_path),
                "index_path": self.index_relative,
                **common,
            },
        ]

    def abort(self) -> None:
        if not self.closed:
            self.binary.close()
            self.index.close()
            self.identifiers.close()
            self.closed = True


def _mgnify_assignment(
    record_id: str,
    selection_basis_points: int,
    train_basis_points: int,
    validation_basis_points: int,
    shard_counts: Dict[str, int],
) -> Optional[Tuple[str, int]]:
    digest = hashlib.sha256(record_id.encode("utf-8")).digest()
    if int.from_bytes(digest[:8], "big") % 10_000 >= selection_basis_points:
        return None
    split_bucket = int.from_bytes(digest[8:16], "big") % 10_000
    if split_bucket < train_basis_points:
        split = "train"
    elif split_bucket < train_basis_points + validation_basis_points:
        split = "validation"
    else:
        split = "test"
    shard = int.from_bytes(digest[16:24], "big") % shard_counts[split]
    return split, shard


def build_selected_mgnify_packed_dataset(
    source_fasta: str,
    output_root: str,
    tokenizer_json: str,
    source_definition_path: str,
    retrieval_receipt_path: str,
    selection_basis_points: int = 2_500,
    train_basis_points: int = 9_800,
    validation_basis_points: int = 100,
    shard_counts: Optional[Dict[str, int]] = None,
    minimum_train_residues: int = 0,
) -> Dict[str, Any]:
    """Stream a verified MGnify90 source into a deterministic selected subset."""

    counts = shard_counts or {"train": 128, "validation": 4, "test": 4}
    if set(counts) != {"train", "validation", "test"} or any(
        int(value) <= 0 for value in counts.values()
    ):
        raise ValueError("shard_counts must contain positive train/validation/test counts")
    if not 0 < selection_basis_points <= 10_000:
        raise ValueError("selection_basis_points must be in (0, 10000]")
    if not 0 < train_basis_points < 10_000:
        raise ValueError("train_basis_points must be in (0, 10000)")
    if validation_basis_points <= 0 or train_basis_points + validation_basis_points >= 10_000:
        raise ValueError("validation and test splits must both be non-empty")
    if minimum_train_residues < 0:
        raise ValueError("minimum_train_residues must be non-negative")

    source_path = Path(source_fasta).resolve()
    definition_path = Path(source_definition_path).resolve()
    receipt_path = Path(retrieval_receipt_path).resolve()
    tokenizer_path = Path(tokenizer_json).resolve()
    final_root = Path(output_root).resolve()
    if final_root.exists():
        raise FileExistsError(f"refusing to overwrite packed dataset: {final_root}")
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_status = receipt.get("status")
    if receipt_status == "VERIFIED_RAW_INPUT":
        if Path(receipt["path"]).resolve() != source_path:
            raise RuntimeError("MGnify retrieval receipt points to another source")
        if int(receipt["bytes"]) != int(definition["fasta_bytes"]):
            raise RuntimeError("MGnify retrieval byte receipt differs from source definition")
        if receipt["md5"] != definition["fasta_md5"]:
            raise RuntimeError("MGnify retrieval MD5 differs from source definition")
    elif receipt_status == "VERIFIED_SHARDED_INPUT":
        transport = definition.get("mirror_transport", {})
        if transport.get("format") != "fasta_zstd_shards":
            raise RuntimeError("MGnify mirror definition has an unsupported format")
        if Path(receipt["dataset_root"]).resolve() != source_path:
            raise RuntimeError("MGnify shard receipt points to another source root")
        for field in ("repo_id", "revision"):
            if receipt.get(field) != transport.get(field):
                raise RuntimeError(f"MGnify shard receipt has a different {field}")
        totals = receipt.get("totals", {})
        if int(totals.get("shards", -1)) != int(transport["shards"]):
            raise RuntimeError("MGnify shard receipt has the wrong shard count")
        if int(totals.get("bytes", -1)) != int(transport["bytes"]):
            raise RuntimeError("MGnify shard receipt has the wrong byte count")
    else:
        raise RuntimeError("MGnify retrieval receipt is not verified")
    tokenizer = EsmcTokenizer.from_tokenizer_json(str(tokenizer_path))
    if max(tokenizer.vocabulary.values()) >= 256:
        raise ValueError("packed uint8 format requires a vocabulary smaller than 256")

    final_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{final_root.name}.tmp-", dir=final_root.parent)
    )
    writers = {
        (split, shard): _SelectedPackedWriter(temporary_root, split, shard, counts[split])
        for split in ("train", "validation", "test")
        for shard in range(counts[split])
    }
    records_seen = 0
    residues_seen = 0
    selected_records = 0
    selected_residues = 0
    invalid_selected = 0
    raw_reader: Optional[_DigestingRawReader] = None
    source_summary: Dict[str, Any] = {}

    def consume_records(text: Any) -> None:
        nonlocal records_seen, residues_seen, selected_records, selected_residues
        nonlocal invalid_selected
        for record_id, _, sequence in iter_fasta(text):
            records_seen += 1
            residues_seen += len(sequence)
            assignment = _mgnify_assignment(
                record_id,
                selection_basis_points,
                train_basis_points,
                validation_basis_points,
                counts,
            )
            if assignment is None:
                continue
            if not sequence or not set(sequence) <= ALLOWED_RESIDUES:
                invalid_selected += 1
                continue
            if not record_id.startswith("MGYP") or not record_id[4:].isdigit():
                raise RuntimeError(f"unexpected MGnify accession: {record_id}")
            encoded = bytes(
                tokenizer.vocabulary.get(residue, tokenizer.unk_token_id)
                for residue in sequence
            )
            writers[assignment].write(int(record_id[4:]), encoded, tokenizer.unk_token_id)
            selected_records += 1
            selected_residues += len(sequence)

    try:
        if receipt_status == "VERIFIED_RAW_INPUT":
            with source_path.open("rb") as binary:
                raw_reader = _DigestingRawReader(binary)
                buffered = io.BufferedReader(raw_reader, buffer_size=4 * 1024 * 1024)
                with gzip.GzipFile(fileobj=buffered, mode="rb") as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                        consume_records(text)
                if raw_reader.bytes_read < source_path.stat().st_size:
                    while buffered.read(4 * 1024 * 1024):
                        pass
            if raw_reader.bytes_read != int(definition["fasta_bytes"]):
                raise RuntimeError("MGnify source byte count differs from the definition")
            if raw_reader.md5.hexdigest() != definition["fasta_md5"]:
                raise RuntimeError("MGnify source MD5 differs from the official release")
            if raw_reader.sha256.hexdigest() != receipt["sha256"]:
                raise RuntimeError("MGnify source SHA-256 differs from the retrieval receipt")
            source_summary = {
                "transport": "official_gzip",
                "bytes": raw_reader.bytes_read,
                "md5": raw_reader.md5.hexdigest(),
                "sha256": raw_reader.sha256.hexdigest(),
            }
        else:
            try:
                import zstandard
            except ImportError as exc:
                raise RuntimeError("zstandard is required for sharded MGnify input") from exc
            receipt_files = receipt.get("files")
            if not isinstance(receipt_files, list):
                raise RuntimeError("MGnify shard receipt has no file list")
            compressed_bytes = 0
            verified_files = 0
            for item in sorted(receipt_files, key=lambda entry: int(entry["shard_index"])):
                shard_path = source_path / item["local_path"]
                if shard_path.stat().st_size != int(item["bytes"]):
                    raise RuntimeError(f"MGnify shard size changed: {item['local_path']}")
                with shard_path.open("rb") as binary:
                    shard_reader = _DigestingRawReader(binary)
                    buffered = io.BufferedReader(shard_reader, buffer_size=4 * 1024 * 1024)
                    decompressor = zstandard.ZstdDecompressor()
                    with decompressor.stream_reader(buffered, closefd=False) as compressed:
                        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                            consume_records(text)
                    if shard_reader.bytes_read < shard_path.stat().st_size:
                        while buffered.read(4 * 1024 * 1024):
                            pass
                if shard_reader.bytes_read != int(item["bytes"]):
                    raise RuntimeError(f"MGnify shard byte count changed: {item['local_path']}")
                if shard_reader.sha256.hexdigest() != item["sha256"]:
                    raise RuntimeError(f"MGnify shard SHA-256 changed: {item['local_path']}")
                compressed_bytes += shard_reader.bytes_read
                verified_files += 1
            source_summary = {
                "transport": "pinned_fasta_zstd_shards",
                "bytes": compressed_bytes,
                "shards": verified_files,
                "repo_id": receipt["repo_id"],
                "revision": receipt["revision"],
            }

        expected_records = int(definition["cluster_representatives"])
        expected_residues = int(definition["cluster_representative_residues"])
        if records_seen != expected_records or residues_seen != expected_residues:
            raise RuntimeError(
                "MGnify source totals differ from the release statistics: "
                f"records observed={records_seen} expected={expected_records}; "
                f"residues observed={residues_seen} expected={expected_residues}"
            )

        split_files: Dict[str, List[Dict[str, Any]]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        for (split, _), writer in sorted(writers.items()):
            split_files[split].extend(writer.close())
        tokenizer_receipt = {
            "path": str(tokenizer_path),
            "sha256": sha256_file(tokenizer_path),
            "vocab_size": max(tokenizer.vocabulary.values()) + 1,
            "cls_token_id": tokenizer.cls_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "unk_token_id": tokenizer.unk_token_id,
            "mask_token_id": tokenizer.mask_token_id,
        }
        split_totals: Dict[str, Dict[str, int]] = {}
        for split, files in split_files.items():
            token_files = [entry for entry in files if entry["kind"] == "tokens"]
            split_totals[split] = {
                "shards": len(token_files),
                "records": sum(int(entry["records"]) for entry in token_files),
                "residues": sum(int(entry["residues"]) for entry in token_files),
                "unknown_residues": sum(
                    int(entry["unknown_residues"]) for entry in token_files
                ),
            }
            if split == "train" and split_totals[split]["residues"] < minimum_train_residues:
                raise RuntimeError("selected MGnify train residues are below the required floor")
            split_manifest = {
                "schema_version": PACKED_SCHEMA_VERSION,
                "format": "openprotein_uint8_sequence_index_v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_root": str(final_root),
                "split": split,
                "test_data_accessed": False,
                "tokenizer": tokenizer_receipt,
                "source_manifest": {
                    "path": str(definition_path),
                    "sha256": sha256_file(definition_path),
                    "retrieval_receipt_path": str(receipt_path),
                    "retrieval_receipt_sha256": sha256_file(receipt_path),
                },
                "selection_policy": {
                    "algorithm": "sha256(MGYP accession), disjoint uint64 words",
                    "selection_basis_points": selection_basis_points,
                    "train_basis_points": train_basis_points,
                    "validation_basis_points": validation_basis_points,
                    "test_basis_points": 10_000
                    - train_basis_points
                    - validation_basis_points,
                    "shard_counts": counts,
                },
                "totals": split_totals[split],
                "files": files,
            }
            _atomic_json(temporary_root / f"{split}-manifest.json", split_manifest)

        manifest: Dict[str, Any] = {
            "schema_version": 1,
            "format": "openprotein_mgnify_selected_packed_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_root": str(final_root),
            "source": {
                "definition_path": str(definition_path),
                "definition_sha256": sha256_file(definition_path),
                "retrieval_receipt_path": str(receipt_path),
                "retrieval_receipt_sha256": sha256_file(receipt_path),
                "records": records_seen,
                "residues": residues_seen,
                **source_summary,
            },
            "selection_basis_points": selection_basis_points,
            "selected_records": selected_records,
            "selected_residues": selected_residues,
            "invalid_selected_records": invalid_selected,
            "split_totals": split_totals,
            "split_manifests": {
                split: {
                    "path": f"{split}-manifest.json",
                    "sha256": sha256_file(temporary_root / f"{split}-manifest.json"),
                }
                for split in ("train", "validation", "test")
            },
        }
        _atomic_json(temporary_root / "manifest.json", manifest)
        os.replace(temporary_root, final_root)
        return manifest
    except Exception:
        for writer in writers.values():
            writer.abort()
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


class PackedProteinBatcher:
    """Deterministic, directly restartable batching over packed sequence shards."""

    def __init__(
        self,
        index_paths: List[str],
        tokenizer: EsmcTokenizer,
        sequence_length: int,
        token_budget: int,
        max_batch_size: int,
        seed: int,
        mlm_probability: float,
        rank: int = 0,
        world_size: int = 1,
        batch_lookahead: int = 1,
    ) -> None:
        self.index_paths = [str(Path(path).resolve()) for path in index_paths]
        if not self.index_paths:
            raise ValueError("packed batcher requires at least one index shard")
        if len(self.index_paths) < world_size:
            raise ValueError("packed batcher requires at least one shard per rank")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if sequence_length < 3:
            raise ValueError("sequence_length must fit CLS, a residue, and EOS")
        if token_budget < sequence_length or max_batch_size <= 0:
            raise ValueError("invalid packed micro-batch limits")
        if batch_lookahead <= 0:
            raise ValueError("batch_lookahead must be positive")
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.token_budget = token_budget
        self.max_batch_size = max_batch_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.batch_lookahead = batch_lookahead
        self.epoch = 0
        self.file_cursor = 0
        self.record_cursor = 0
        self.pending: Optional[Dict[str, object]] = None
        self.lookahead_pool: List[Dict[str, object]] = []
        self.collator = ProteinMLMCollator(tokenizer, seed + 104_729 * rank, mlm_probability)
        self._open_path: Optional[str] = None
        self._open_shard: Optional[PackedSequenceShard] = None
        self._migration_paths: Dict[str, Dict[str, object]] = {}
        self._migration_current_path: Optional[str] = None
        self._migration_pending: List[Dict[str, object]] = []
        self._migration_completed = False

    def _owned_paths_for(self, epoch: int, rank: int, world_size: int) -> List[str]:
        keyed = []
        for path in self.index_paths:
            digest = hashlib.sha256(f"{self.seed}:{epoch}:{path}".encode()).digest()
            keyed.append((digest, path))
        ordered = [path for _, path in sorted(keyed)]
        return [path for index, path in enumerate(ordered) if index % world_size == rank]

    def _owned_paths(self) -> List[str]:
        return self._owned_paths_for(self.epoch, self.rank, self.world_size)

    @staticmethod
    def _record_path(example: Dict[str, object]) -> str:
        record_id = example.get("record_id")
        if not isinstance(record_id, str) or record_id.count(":") < 2:
            raise RuntimeError("topology migration found an invalid packed record id")
        return record_id.rsplit(":", 2)[0]

    def load_topology_migration_state(
        self,
        states: List[Dict[str, Any]],
        migration_seed: int,
    ) -> None:
        """Repartition exact unconsumed packed records across a new world size.

        Model and optimizer tensors are resharded by distributed checkpointing.
        This method separately converts the topology-bound packed-data cursors:
        completed paths stay completed, each partial path keeps its exact cursor,
        and records already read into a lookahead pool remain pending rather than
        being silently skipped or replayed.
        """

        if not states:
            raise RuntimeError("topology migration requires old rank states")
        paths_sha256 = hashlib.sha256("\n".join(self.index_paths).encode()).hexdigest()
        old_world_size = int(states[0].get("world_size", -1))
        epochs = {int(state.get("epoch", -1)) for state in states}
        ranks = {int(state.get("rank", -1)) for state in states}
        if old_world_size != len(states) or ranks != set(range(old_world_size)):
            raise RuntimeError("topology migration rank states are incomplete")
        if len(epochs) != 1:
            raise RuntimeError("topology migration requires one shared packed-data epoch")
        epoch = epochs.pop()
        migrated: Dict[str, Dict[str, object]] = {}
        for state in sorted(states, key=lambda item: int(item["rank"])):
            if state.get("format") != "openprotein_uint8_sequence_index_v1":
                raise RuntimeError("topology migration batcher format differs")
            if state.get("paths_sha256") != paths_sha256:
                raise RuntimeError("topology migration packed shard list differs")
            if int(state.get("world_size", -1)) != old_world_size:
                raise RuntimeError("topology migration old world sizes differ")
            rank = int(state["rank"])
            owned = self._owned_paths_for(epoch, rank, old_world_size)
            file_cursor = int(state.get("file_cursor", -1))
            if not 0 <= file_cursor <= len(owned):
                raise RuntimeError("topology migration file cursor is invalid")
            for path in owned[:file_cursor]:
                migrated[path] = {
                    "record_cursor": 0,
                    "completed": True,
                    "pending": [],
                }
            if file_cursor < len(owned):
                path = owned[file_cursor]
                record_cursor = int(state.get("record_cursor", -1))
                if record_cursor < 0:
                    raise RuntimeError("topology migration record cursor is invalid")
                migrated[path] = {
                    "record_cursor": record_cursor,
                    "completed": False,
                    "pending": [],
                }
            pools = []
            pending = state.get("pending")
            if pending is not None:
                pools.append(copy.deepcopy(pending))
            pools.extend(copy.deepcopy(list(state.get("lookahead_pool", []))))
            for example in pools:
                path = self._record_path(example)
                if path not in self.index_paths:
                    raise RuntimeError("topology migration pending record uses another shard")
                entry = migrated.setdefault(
                    path,
                    {"record_cursor": 0, "completed": False, "pending": []},
                )
                entry["pending"].append(example)

        owned_now = set(self._owned_paths_for(epoch, self.rank, self.world_size))
        self.epoch = epoch
        self.file_cursor = 0
        self.record_cursor = 0
        self.pending = None
        self.lookahead_pool = []
        self._migration_paths = {
            path: copy.deepcopy(entry)
            for path, entry in migrated.items()
            if path in owned_now
        }
        self._migration_current_path = None
        self._migration_pending = []
        self._migration_completed = False
        self.collator.generator.manual_seed(int(migration_seed) + 104_729 * self.rank)
        if self._open_shard is not None:
            self._open_shard.close()
        self._open_shard = None
        self._open_path = None

    def _activate_migration_path(self, path: str) -> None:
        if self._migration_current_path == path:
            return
        if self._migration_pending:
            raise RuntimeError("topology migration changed paths with pending records")
        entry = self._migration_paths.pop(path, None)
        self._migration_current_path = path
        self._migration_pending = []
        self._migration_completed = False
        if entry is not None:
            self.record_cursor = int(entry["record_cursor"])
            self._migration_pending = copy.deepcopy(list(entry.get("pending", [])))
            self._migration_completed = bool(entry.get("completed", False))

    def _shard(self, path: str) -> PackedSequenceShard:
        if self._open_path != path:
            if self._open_shard is not None:
                self._open_shard.close()
            self._open_path = path
            self._open_shard = PackedSequenceShard(path)
        return self._open_shard

    def _permutation(self, path: str, records: int) -> Tuple[int, int]:
        if records <= 1:
            return 0, 1
        digest = hashlib.sha256(f"perm:{self.seed}:{self.epoch}:{path}".encode()).digest()
        start = int.from_bytes(digest[:8], "big") % records
        stride = 1 + int.from_bytes(digest[8:16], "big") % (records - 1)
        while math.gcd(stride, records) != 1:
            stride = 1 if stride + 1 >= records else stride + 1
        return start, stride

    def _crop_start(self, path: str, record_index: int, maximum: int) -> int:
        if maximum <= 0:
            return 0
        digest = hashlib.sha256(
            f"crop:{self.seed}:{self.epoch}:{path}:{record_index}".encode()
        ).digest()
        return int.from_bytes(digest[:8], "big") % (maximum + 1)

    def _next_example(self) -> Dict[str, object]:
        while True:
            paths = self._owned_paths()
            if self.file_cursor >= len(paths):
                self.epoch += 1
                self.file_cursor = 0
                self.record_cursor = 0
                self._migration_current_path = None
                self._migration_pending = []
                self._migration_completed = False
                continue
            path = paths[self.file_cursor]
            self._activate_migration_path(path)
            if self._migration_pending:
                return self._migration_pending.pop(0)
            shard = self._shard(path)
            if (
                self._migration_completed
                or shard.records == 0
                or self.record_cursor >= shard.records
            ):
                self.file_cursor += 1
                self.record_cursor = 0
                self._migration_current_path = None
                self._migration_completed = False
                continue
            start, stride = self._permutation(path, shard.records)
            record_index = (start + self.record_cursor * stride) % shard.records
            self.record_cursor += 1
            sequence = shard.read(record_index)
            content_length = self.sequence_length - 2
            crop_start = self._crop_start(
                path, record_index, max(0, len(sequence) - content_length)
            )
            sequence = sequence[crop_start : crop_start + content_length]
            return {
                "input_ids": [self.tokenizer.cls_token_id, *sequence, self.tokenizer.eos_token_id],
                "record_id": f"{path}:{record_index}:{crop_start}",
            }

    def _read_lookahead_window(self) -> List[Dict[str, object]]:
        window = [self._next_example() for _ in range(self.batch_lookahead)]
        window.sort(key=lambda example: len(example["input_ids"]), reverse=True)
        return window

    def next(self) -> Dict[str, torch.Tensor]:
        if self.batch_lookahead > 1:
            if not self.lookahead_pool:
                self.lookahead_pool = self._read_lookahead_window()
            maximum_length = len(self.lookahead_pool[0]["input_ids"])
            target_count = min(
                self.max_batch_size,
                max(1, self.token_budget // maximum_length),
            )
            if len(self.lookahead_pool) < target_count:
                examples = self.lookahead_pool
                refill = self._read_lookahead_window()
                while refill and len(examples) < self.max_batch_size:
                    candidate = refill[-1]
                    next_maximum = max(maximum_length, len(candidate["input_ids"]))
                    if next_maximum * (len(examples) + 1) > self.token_budget:
                        break
                    examples.append(refill.pop())
                    maximum_length = next_maximum
                self.lookahead_pool = refill
            else:
                examples = self.lookahead_pool[:target_count]
                del self.lookahead_pool[:target_count]
            return self.collator(examples)

        examples: List[Dict[str, object]] = []
        maximum_length = 0
        while len(examples) < self.max_batch_size:
            if self.pending is not None:
                candidate = self.pending
                self.pending = None
            else:
                candidate = self._next_example()
            candidate_length = len(candidate["input_ids"])
            next_maximum = max(maximum_length, candidate_length)
            if examples and next_maximum * (len(examples) + 1) > self.token_budget:
                self.pending = candidate
                break
            examples.append(candidate)
            maximum_length = next_maximum
        return self.collator(examples)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "format": "openprotein_uint8_sequence_index_v1",
            "paths_sha256": hashlib.sha256("\n".join(self.index_paths).encode()).hexdigest(),
            "rank": self.rank,
            "world_size": self.world_size,
            "epoch": self.epoch,
            "file_cursor": self.file_cursor,
            "record_cursor": self.record_cursor,
            "pending": copy.deepcopy(self.pending),
            "batch_lookahead": self.batch_lookahead,
            "lookahead_pool": copy.deepcopy(self.lookahead_pool),
            "collator": self.collator.state_dict(),
            "topology_migration": {
                "paths": copy.deepcopy(self._migration_paths),
                "current_path": self._migration_current_path,
                "current_pending": copy.deepcopy(self._migration_pending),
                "current_completed": self._migration_completed,
            },
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        expected_paths = hashlib.sha256("\n".join(self.index_paths).encode()).hexdigest()
        if state.get("format") != "openprotein_uint8_sequence_index_v1":
            raise RuntimeError("checkpoint batcher format differs from packed data")
        if state.get("paths_sha256") != expected_paths:
            raise RuntimeError("checkpoint packed shard list differs")
        if int(state["rank"]) != self.rank or int(state["world_size"]) != self.world_size:
            raise RuntimeError("checkpoint packed batcher topology differs")
        if int(state.get("batch_lookahead", 1)) != self.batch_lookahead:
            raise RuntimeError("checkpoint packed batch lookahead differs")
        self.epoch = int(state["epoch"])
        self.file_cursor = int(state["file_cursor"])
        self.record_cursor = int(state["record_cursor"])
        self.pending = copy.deepcopy(state.get("pending"))
        self.lookahead_pool = copy.deepcopy(list(state.get("lookahead_pool", [])))
        self.collator.load_state_dict(state["collator"])
        migration = state.get("topology_migration") or {}
        self._migration_paths = copy.deepcopy(dict(migration.get("paths", {})))
        self._migration_current_path = migration.get("current_path")
        self._migration_pending = copy.deepcopy(
            list(migration.get("current_pending", []))
        )
        self._migration_completed = bool(migration.get("current_completed", False))
        if self._open_shard is not None:
            self._open_shard.close()
        self._open_shard = None
        self._open_path = None
