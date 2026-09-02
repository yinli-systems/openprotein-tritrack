"""Build a compact sequence-only view of the official TorchProtein EC benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, List, Tuple

from openprotein.provenance import sha256_file


TEST_CUTOFFS = (0.3, 0.4, 0.5, 0.7, 0.95)
_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SEC": "U",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "PYL": "O",
}


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - upstream publishes this exact digest
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_by_suffix(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one outer ZIP member ending in {suffix}")
    return matches[0]


def _copy_member(archive: zipfile.ZipFile, member: str, output: Path) -> None:
    with archive.open(member) as source, output.open("wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


def _parse_annotations(payload: str) -> Tuple[List[str], Dict[str, List[int]]]:
    rows = list(csv.reader(payload.splitlines(), delimiter="\t"))
    if len(rows) < 4 or rows[0] != ["### EC-numbers"]:
        raise ValueError("unexpected EC annotation header")
    tasks = rows[1]
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("EC task vocabulary is empty or duplicated")
    if rows[2] != ["### PDB-chain", "EC-numbers"]:
        raise ValueError("unexpected EC record header")
    task_to_index = {task: index for index, task in enumerate(tasks)}
    annotations: Dict[str, List[int]] = {}
    for row in rows[3:]:
        if len(row) != 2 or not row[0] or not row[1]:
            raise ValueError("malformed EC annotation row")
        if row[0] in annotations:
            raise ValueError(f"duplicate EC annotation ID: {row[0]}")
        labels = [task_to_index[label] for label in row[1].split(",")]
        annotations[row[0]] = labels
    return tasks, annotations


def _parse_test_cutoffs(payload: str) -> Dict[str, Tuple[int, ...]]:
    rows = csv.reader(payload.splitlines())
    header = next(rows)
    expected = ["PDB-chain", "<30%", "<40%", "<50%", "<70%", "<95%"]
    if header != expected:
        raise ValueError("unexpected EC identity-cutoff header")
    result: Dict[str, Tuple[int, ...]] = {}
    for row in rows:
        if len(row) != len(expected) or row[0] in result:
            raise ValueError("malformed or duplicate EC identity-cutoff row")
        flags = tuple(int(value) for value in row[1:])
        if any(value not in {0, 1} for value in flags):
            raise ValueError("EC identity-cutoff flags must be binary")
        result[row[0]] = flags
    return result


def _sequence_from_pdb(handle: BinaryIO) -> Tuple[str, int]:
    residues: Dict[Tuple[bytes, bytes, bytes], Tuple[Tuple[int, bytes], str]] = {}
    for line in handle:
        if not line.startswith(b"ATOM  ") or len(line) < 27:
            continue
        if line[12:16].strip() != b"CA":
            continue
        residue_key = (line[21:22], line[22:26], line[26:27])
        alternate_location = line[16:17]
        priority = (
            0 if alternate_location == b" " else 1 if alternate_location == b"A" else 2,
            alternate_location,
        )
        existing = residues.get(residue_key)
        if existing is None or priority < existing[0]:
            residues[residue_key] = (
                priority,
                _THREE_TO_ONE.get(line[17:20].decode("ascii"), "X"),
            )
    if not residues:
        raise ValueError("EC benchmark PDB member contains no CA residues")
    fallback_residues = sum(priority[0] == 2 for priority, _ in residues.values())
    return "".join(residue for _, residue in residues.values()), fallback_residues


def _iter_split(
    nested_zip: Path,
    split: str,
) -> Iterable[Tuple[str, str, int]]:
    with zipfile.ZipFile(nested_zip) as archive:
        members = sorted(
            name for name in archive.namelist() if name.startswith(f"{split}/") and name.endswith(".pdb")
        )
        if not members:
            raise RuntimeError(f"EC nested ZIP contains no {split} PDB members")
        for member in members:
            filename = Path(member).name
            record_id = filename.split("_", 1)[0]
            with archive.open(member) as handle:
                sequence, fallback_residues = _sequence_from_pdb(handle)
                yield record_id, sequence, fallback_residues


def build_ec_benchmark(
    source_zip_path: str,
    source_definition_path: str,
    output_directory: str,
) -> Dict[str, Any]:
    """Verify the official archive and publish a deterministic sequence/label table."""

    source = Path(source_zip_path).resolve()
    definition_path = Path(source_definition_path).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite EC benchmark: {output}")
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    if source.stat().st_size != int(definition["bytes"]):
        raise RuntimeError("EC source byte count differs from its definition")
    observed_md5 = _md5_file(source)
    if observed_md5 != definition["md5"]:
        raise RuntimeError("EC source MD5 differs from its definition")
    observed_sha256 = sha256_file(source)
    expected_sha256 = definition.get("sha256")
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise RuntimeError("EC source SHA-256 differs from its definition")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    split_counts: Dict[str, int] = {}
    cutoff_counts = {f"below_{int(cutoff * 100)}_percent": 0 for cutoff in TEST_CUTOFFS}
    alternate_location_fallbacks: List[Dict[str, Any]] = []
    split_record_ids = {split: set() for split in ("train", "valid", "test")}
    try:
        with zipfile.ZipFile(source) as outer:
            annotation_member = _member_by_suffix(outer, "EnzymeCommission/nrPDB-EC_annot.tsv")
            cutoff_member = _member_by_suffix(outer, "EnzymeCommission/nrPDB-EC_test.csv")
            tasks, annotations = _parse_annotations(
                outer.read(annotation_member).decode("utf-8")
            )
            test_cutoffs = _parse_test_cutoffs(outer.read(cutoff_member).decode("utf-8"))
            nested_paths: Dict[str, Path] = {}
            for split in ("train", "valid", "test"):
                member = _member_by_suffix(outer, f"EnzymeCommission/{split}.zip")
                nested = staging / f"{split}.zip"
                _copy_member(outer, member, nested)
                nested_paths[split] = nested

        labels_path = staging / "labels.json"
        labels_path.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
        table_path = staging / "benchmark.csv"
        fields = [
            "id",
            "split",
            "sequence",
            "label_indices_json",
            *cutoff_counts,
        ]
        seen_ids = set()
        with table_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for split in ("train", "valid", "test"):
                count = 0
                for record_id, sequence, fallback_residues in _iter_split(
                    nested_paths[split], split
                ):
                    if record_id in seen_ids:
                        raise RuntimeError(f"EC benchmark ID crosses splits: {record_id}")
                    seen_ids.add(record_id)
                    split_record_ids[split].add(record_id)
                    if record_id not in annotations:
                        raise RuntimeError(f"EC benchmark has no labels for {record_id}")
                    flags = test_cutoffs.get(record_id)
                    if fallback_residues:
                        alternate_location_fallbacks.append(
                            {
                                "id": record_id,
                                "split": split,
                                "residues": fallback_residues,
                            }
                        )
                    row: Dict[str, Any] = {
                        "id": record_id,
                        "split": split,
                        "sequence": sequence,
                        "label_indices_json": json.dumps(
                            annotations[record_id], separators=(",", ":")
                        ),
                    }
                    for index, field in enumerate(cutoff_counts):
                        value = "" if flags is None else flags[index]
                        row[field] = value
                        if split == "test" and value == 1:
                            cutoff_counts[field] += 1
                    writer.writerow(row)
                    count += 1
                split_counts[split] = count

        for nested in nested_paths.values():
            nested.unlink()
        receipt: Dict[str, Any] = {
            "schema_version": 1,
            "status": "passed",
            "evidence_level": "PUBLIC_BENCHMARK_DATASET_BUILD",
            "test_data_accessed": True,
            "proteingym_rows_opened": False,
            "benchmark": "TorchProtein EnzymeCommission",
            "protocol": (
                "official train/valid/test splits and published 30/40/50/70/95 percent "
                "test identity-cutoff masks; sequence-only CA-residue view"
            ),
            "source_definition": {
                "path": str(definition_path),
                "sha256": sha256_file(definition_path),
            },
            "source": {
                **definition,
                "path": str(source),
                "observed_md5": observed_md5,
                "observed_sha256": observed_sha256,
            },
            "records": split_counts,
            "labels": len(tasks),
            "test_identity_cutoff_counts": cutoff_counts,
            "identity_cutoff_alignment": {
                "test_records_without_published_cutoff": sorted(
                    split_record_ids["test"] - set(test_cutoffs)
                ),
                "published_cutoff_records_not_in_test_archive": sorted(
                    set(test_cutoffs) - split_record_ids["test"]
                ),
                "policy": (
                    "retain every official test-archive record; records without a published "
                    "cutoff remain outside all cutoff-specific metric subsets"
                ),
            },
            "sequence_parsing": {
                "residue_source": "unique ATOM CA records in supplied PDB chains",
                "alternate_location_policy": (
                    "prefer blank, then A, then lexicographically first available alternate "
                    "location independently per residue"
                ),
                "records_requiring_non_primary_alternate_location": alternate_location_fallbacks,
            },
            "table": {
                "path": "benchmark.csv",
                "bytes": table_path.stat().st_size,
                "sha256": sha256_file(table_path),
            },
            "label_vocabulary": {
                "path": "labels.json",
                "bytes": labels_path.stat().st_size,
                "sha256": sha256_file(labels_path),
            },
            "comparability_limitations": [
                "The sequence view uses unique CA residues present in each supplied PDB chain.",
                "Identity cutoffs constrain benchmark train-test homology, not overlap with either model's pretraining corpus.",
            ],
        }
        receipt_path = staging / "receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
        return receipt
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
