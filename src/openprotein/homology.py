"""Manifest-bound train-to-validation homology audit helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

from openprotein.provenance import sha256_file


MMSEQS_COLUMNS = (
    "query",
    "target",
    "pident",
    "alnlen",
    "qstart",
    "qend",
    "qlen",
    "tstart",
    "tend",
    "tlen",
    "evalue",
    "bits",
    "qcov",
    "tcov",
)

HOMOLOGY_THRESHOLDS: Mapping[str, Mapping[str, float]] = {
    "exact_duplicate": {"pident": 99.9, "qcov": 0.99, "tcov": 0.99},
    "near_duplicate": {"pident": 95.0, "qcov": 0.90},
    "identity_50_query_coverage_80": {"pident": 50.0, "qcov": 0.80},
    "identity_50_bidirectional_coverage_80": {
        "pident": 50.0,
        "qcov": 0.80,
        "tcov": 0.80,
    },
    "identity_30_query_coverage_80": {"pident": 30.0, "qcov": 0.80},
}


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _manifest_split_files(manifest_path: Path, split: str) -> Tuple[Dict[str, Any], List[Path]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_root = Path(manifest["dataset_root"]).resolve()
    selected = [item for item in manifest["files"] if item["split"] == split]
    if not selected:
        raise ValueError(f"manifest contains no {split} files: {manifest_path}")
    paths = []
    for item in selected:
        path = (dataset_root / item["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"manifest-bound shard is missing: {path}")
        if path.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"manifest-bound shard byte count changed: {path}")
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"manifest-bound shard SHA-256 changed: {path}")
        paths.append(path)
    return manifest, paths


def _fasta_bytes(record_id: str, sequence: str) -> bytes:
    if not record_id or any(character.isspace() for character in record_id):
        raise ValueError(f"FASTA identifier is empty or contains whitespace: {record_id!r}")
    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError(f"empty sequence for {record_id}")
    return f">{record_id}\n{sequence}\n".encode("ascii")


def _export_split_fasta(
    paths: Iterable[Path],
    output: Path,
    sequence_column: str,
    id_column: str = "id",
    ids_output: Path | None = None,
) -> Dict[str, Any]:
    digest = hashlib.sha256()
    ids_digest = hashlib.sha256()
    records = 0
    residues = 0
    seen_ids = set() if ids_output is not None else None
    id_handle = ids_output.open("wb") if ids_output is not None else None
    try:
        with output.open("wb") as output_handle:
            for path in paths:
                with path.open("r", encoding="utf-8", newline="") as source_handle:
                    reader = csv.DictReader(source_handle)
                    required = {id_column, sequence_column}
                    if reader.fieldnames is None or not required <= set(reader.fieldnames):
                        raise ValueError(f"{path} lacks required columns {sorted(required)}")
                    for row_index, row in enumerate(reader, start=2):
                        record_id = row[id_column].strip()
                        if seen_ids is not None:
                            if record_id in seen_ids:
                                raise ValueError(
                                    f"duplicate query ID {record_id} in {path}:{row_index}"
                                )
                            seen_ids.add(record_id)
                        payload = _fasta_bytes(record_id, row[sequence_column])
                        output_handle.write(payload)
                        digest.update(payload)
                        sequence = row[sequence_column].strip()
                        records += 1
                        residues += len(sequence)
                        if id_handle is not None:
                            id_payload = f"{record_id}\n".encode("utf-8")
                            id_handle.write(id_payload)
                            ids_digest.update(id_payload)
    finally:
        if id_handle is not None:
            id_handle.close()
    receipt: Dict[str, Any] = {
        "path": output.name,
        "bytes": output.stat().st_size,
        "sha256": digest.hexdigest(),
        "records": records,
        "residues": residues,
    }
    if ids_output is not None:
        receipt["ids"] = {
            "path": ids_output.name,
            "bytes": ids_output.stat().st_size,
            "sha256": ids_digest.hexdigest(),
        }
    return receipt


def prepare_homology_inputs(
    train_manifest_path: str,
    validation_manifest_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Export only manifest train rows and manifest validation rows as FASTA."""

    train_manifest_file = Path(train_manifest_path).resolve()
    validation_manifest_file = Path(validation_manifest_path).resolve()
    work = Path(output_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    prepare_path = work / "prepare.json"
    train_fasta = work / "train.fasta"
    query_fasta = work / "queries.fasta"
    query_ids = work / "queries.ids.txt"
    for path in (prepare_path, train_fasta, query_fasta, query_ids):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite homology input: {path}")

    train_manifest, train_paths = _manifest_split_files(train_manifest_file, "train")
    validation_manifest, validation_paths = _manifest_split_files(
        validation_manifest_file, "validation"
    )
    train = _export_split_fasta(train_paths, train_fasta, sequence_column="text")
    queries = _export_split_fasta(
        validation_paths,
        query_fasta,
        sequence_column="sequence",
        ids_output=query_ids,
    )
    train_expected = train_manifest["split_totals"]["train"]
    validation_expected = validation_manifest["split_totals"]["validation"]
    if train["records"] != int(train_expected["records"]):
        raise RuntimeError("exported train FASTA record count differs from manifest")
    if train["residues"] != int(train_expected["residues"]):
        raise RuntimeError("exported train FASTA residue count differs from manifest")
    if queries["records"] != int(validation_expected["records"]):
        raise RuntimeError("exported validation FASTA record count differs from manifest")
    if queries["residues"] != int(validation_expected["residues"]):
        raise RuntimeError("exported validation FASTA residue count differs from manifest")

    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": "DECONTAMINATION_INPUT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "split_contract": {
            "target_database": "train only",
            "queries": "validation only",
        },
        "train_manifest": {
            "path": str(train_manifest_file),
            "sha256": sha256_file(train_manifest_file),
            "files": len(train_paths),
        },
        "validation_manifest": {
            "path": str(validation_manifest_file),
            "sha256": sha256_file(validation_manifest_file),
            "files": len(validation_paths),
        },
        "train": train,
        "queries": queries,
    }
    _atomic_json(prepare_path, receipt)
    return receipt


def _read_query_ids(path: Path) -> List[str]:
    query_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not query_ids or any(not query_id for query_id in query_ids):
        raise ValueError("query ID file is empty or malformed")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query ID file contains duplicates")
    return query_ids


def _parse_mmseqs(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, fieldnames=MMSEQS_COLUMNS, delimiter="\t")
        for row_index, row in enumerate(reader, start=1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"malformed MMseqs row {row_index}")
            yield {
                "query": row["query"],
                "target": row["target"],
                "pident": float(row["pident"]),
                "alnlen": int(row["alnlen"]),
                "qlen": int(row["qlen"]),
                "tlen": int(row["tlen"]),
                "evalue": float(row["evalue"]),
                "bits": float(row["bits"]),
                "qcov": float(row["qcov"]),
                "tcov": float(row["tcov"]),
            }


def _passes(hit: Mapping[str, Any], threshold: Mapping[str, float]) -> bool:
    return all(float(hit[key]) >= minimum for key, minimum in threshold.items())


def _write_ids(path: Path, values: Sequence[str]) -> Dict[str, Any]:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    path.write_bytes(payload)
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def summarize_homology_hits(
    prepare_path: str,
    hits_path: str,
    output_path: str,
    clean_ids_path: str,
    excluded_ids_path: str,
    top_hits_path: str,
    mmseqs_version: str,
    mmseqs_command: Sequence[str],
    search_slurm_job_id: str | None = None,
) -> Dict[str, Any]:
    """Summarize train hits and materialize a <50%-identity validation subset."""

    prepare_file = Path(prepare_path).resolve()
    hits_file = Path(hits_path).resolve()
    output_file = Path(output_path).resolve()
    clean_file = Path(clean_ids_path).resolve()
    excluded_file = Path(excluded_ids_path).resolve()
    top_file = Path(top_hits_path).resolve()
    for path in (output_file, clean_file, excluded_file, top_file):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite homology result: {path}")
    prepare = json.loads(prepare_file.read_text(encoding="utf-8"))
    if prepare.get("test_data_accessed") is not False:
        raise RuntimeError("homology preparation receipt does not prove test isolation")
    query_ids_file = Path(prepare["queries"]["ids"]["path"])
    if not query_ids_file.is_absolute():
        query_ids_file = prepare_file.parent / query_ids_file
    if sha256_file(query_ids_file) != prepare["queries"]["ids"]["sha256"]:
        raise RuntimeError("query ID list changed after preparation")
    query_ids = _read_query_ids(query_ids_file)
    query_set = set(query_ids)
    threshold_queries = {name: set() for name in HOMOLOGY_THRESHOLDS}
    best_hits: Dict[str, Dict[str, Any]] = {}
    rows = 0
    for hit in _parse_mmseqs(hits_file):
        rows += 1
        query_id = hit["query"]
        if query_id not in query_set:
            raise RuntimeError(f"MMseqs emitted an unknown query ID: {query_id}")
        for name, threshold in HOMOLOGY_THRESHOLDS.items():
            if _passes(hit, threshold):
                threshold_queries[name].add(query_id)
        previous = best_hits.get(query_id)
        key = (hit["pident"], hit["qcov"], hit["tcov"], hit["bits"])
        if previous is None or key > (
            previous["pident"],
            previous["qcov"],
            previous["tcov"],
            previous["bits"],
        ):
            best_hits[query_id] = hit

    excluded = sorted(threshold_queries["identity_50_query_coverage_80"])
    excluded_set = set(excluded)
    clean = [query_id for query_id in query_ids if query_id not in excluded_set]
    clean_receipt = _write_ids(clean_file, clean)
    excluded_receipt = _write_ids(excluded_file, excluded)
    top_fields = ("query", "target", "pident", "qcov", "tcov", "evalue", "bits")
    with top_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=top_fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for query_id in query_ids:
            if query_id in best_hits:
                writer.writerow({field: best_hits[query_id][field] for field in top_fields})

    exact = len(threshold_queries["exact_duplicate"])
    identity_50 = len(threshold_queries["identity_50_query_coverage_80"])
    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "status": "passed" if exact == 0 and identity_50 == 0 else "decontamination_required",
        "evidence_level": "TRAIN_VALIDATION_HOMOLOGY_AUDIT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "prepare_receipt": {"path": prepare_file.name, "sha256": sha256_file(prepare_file)},
        "hits": {
            "path": hits_file.name,
            "bytes": hits_file.stat().st_size,
            "sha256": sha256_file(hits_file),
            "rows": rows,
        },
        "queries": len(query_ids),
        "queries_with_reported_hits": len(best_hits),
        "queries_without_reported_hits": len(query_ids) - len(best_hits),
        "thresholds": {
            name: {
                **threshold,
                "queries": len(threshold_queries[name]),
                "fraction": len(threshold_queries[name]) / len(query_ids),
            }
            for name, threshold in HOMOLOGY_THRESHOLDS.items()
        },
        "decontamination_policy": (
            "exclude a validation query when any train hit has >=50% identity and >=80% "
            "query coverage"
        ),
        "audit_scope": (
            "This audit covers the manifest-bound OpenProtein UniRef50 train split only. "
            "The record-level pretraining corpora of the warm-start ESM-2 backbone and the "
            "public ProTrek baseline are not fully exposed and cannot be cleared by this receipt."
        ),
        "clean_queries": {**clean_receipt, "records": len(clean)},
        "excluded_queries": {**excluded_receipt, "records": len(excluded)},
        "top_hits": {
            "path": top_file.name,
            "bytes": top_file.stat().st_size,
            "sha256": sha256_file(top_file),
        },
        "mmseqs": {"version": mmseqs_version, "command": list(mmseqs_command)},
        "mmseqs_search_slurm_job_id": search_slurm_job_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(output_file, receipt)
    return receipt
