"""Filter a verified EC sequence view by a verified TorchDrug eligibility audit."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from openprotein.function_probe import EC_CUTOFF_FIELDS
from openprotein.provenance import sha256_file


def _verified_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise RuntimeError(f"receipt did not pass: {path}")
    if payload.get("test_data_accessed") is not True:
        raise RuntimeError(f"receipt does not declare public-test access: {path}")
    if payload.get("proteingym_rows_opened") is not False:
        raise RuntimeError(f"receipt opened protected ProteinGym rows: {path}")
    return payload


def _verify_artifact(root: Path, artifact: dict[str, Any]) -> Path:
    path = root / artifact["path"]
    if path.stat().st_size != int(artifact["bytes"]):
        raise RuntimeError(f"artifact byte count changed: {path}")
    if sha256_file(path) != artifact["sha256"]:
        raise RuntimeError(f"artifact SHA-256 changed: {path}")
    return path


def filter_ec_benchmark(
    benchmark_directory: str,
    rdkit_audit_directory: str,
    output_directory: str,
) -> dict[str, Any]:
    """Publish a sequence table restricted to TorchDrug-compatible record IDs."""

    benchmark_root = Path(benchmark_directory).resolve()
    audit_root = Path(rdkit_audit_directory).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite filtered EC benchmark: {output}")
    benchmark_receipt_path = benchmark_root / "receipt.json"
    audit_receipt_path = audit_root / "receipt.json"
    benchmark_receipt = _verified_json(benchmark_receipt_path)
    audit_receipt = _verified_json(audit_receipt_path)
    if benchmark_receipt["source"]["observed_sha256"] != audit_receipt["source"][
        "observed_sha256"
    ]:
        raise RuntimeError("EC benchmark and RDKit audit used different source archives")

    benchmark_table = _verify_artifact(benchmark_root, benchmark_receipt["table"])
    labels_path = _verify_artifact(
        benchmark_root, benchmark_receipt["label_vocabulary"]
    )
    valid_records_path = _verify_artifact(audit_root, audit_receipt["valid_records"])
    with valid_records_path.open("r", encoding="utf-8", newline="") as handle:
        valid_rows = list(csv.DictReader(handle))
    valid_splits = {row["id"]: row["split"] for row in valid_rows}
    if len(valid_splits) != len(valid_rows):
        raise RuntimeError("RDKit audit valid-record table contains duplicate IDs")
    expected_valid_counts = {
        split: int(audit_receipt["records"][split]["valid"])
        for split in ("train", "valid", "test")
    }
    observed_valid_counts = {
        split: sum(row["split"] == split for row in valid_rows)
        for split in ("train", "valid", "test")
    }
    if observed_valid_counts != expected_valid_counts:
        raise RuntimeError("RDKit audit valid-record counts changed")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        provenance_directory = staging / "provenance"
        provenance_directory.mkdir()
        portable_benchmark_receipt = provenance_directory / "raw-benchmark-receipt.json"
        portable_audit_receipt = provenance_directory / "rdkit-audit-receipt.json"
        portable_valid_records = provenance_directory / "rdkit-valid-records.csv"
        shutil.copyfile(benchmark_receipt_path, portable_benchmark_receipt)
        shutil.copyfile(audit_receipt_path, portable_audit_receipt)
        shutil.copyfile(valid_records_path, portable_valid_records)
        table_path = staging / "benchmark.csv"
        cutoff_counts = {field: 0 for field in EC_CUTOFF_FIELDS}
        records = {split: 0 for split in ("train", "valid", "test")}
        seen_valid_ids = set()
        with benchmark_table.open("r", encoding="utf-8", newline="") as source_handle:
            reader = csv.DictReader(source_handle)
            if reader.fieldnames is None:
                raise RuntimeError("raw EC benchmark has no header")
            with table_path.open("w", encoding="utf-8", newline="") as target_handle:
                writer = csv.DictWriter(
                    target_handle, fieldnames=reader.fieldnames, lineterminator="\n"
                )
                writer.writeheader()
                for row in reader:
                    expected_split = valid_splits.get(row["id"])
                    if expected_split is None:
                        continue
                    if row["split"] != expected_split:
                        raise RuntimeError(f"EC split changed for {row['id']}")
                    writer.writerow(row)
                    seen_valid_ids.add(row["id"])
                    records[row["split"]] += 1
                    if row["split"] == "test":
                        for field in EC_CUTOFF_FIELDS:
                            cutoff_counts[field] += row[field] == "1"
        if seen_valid_ids != set(valid_splits):
            missing = sorted(set(valid_splits) - seen_valid_ids)
            raise RuntimeError(f"RDKit-valid IDs absent from raw EC view: {missing[:10]}")
        if records != expected_valid_counts:
            raise RuntimeError("filtered EC split counts differ from the RDKit audit")

        filtered_labels_path = staging / "labels.json"
        shutil.copyfile(labels_path, filtered_labels_path)
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "status": "passed",
            "evidence_level": "PUBLIC_BENCHMARK_TORCHDRUG_COMPATIBLE_SEQUENCE_VIEW",
            "test_data_accessed": True,
            "proteingym_rows_opened": False,
            "benchmark": "TorchProtein EnzymeCommission",
            "protocol": (
                "verified CA-residue sequence view restricted to records accepted by the "
                "pinned RDKit and TorchDrug 0.2.1 dataset-loader policy"
            ),
            "source": benchmark_receipt["source"],
            "raw_benchmark_receipt": {
                "path": str(portable_benchmark_receipt.relative_to(staging)),
                "sha256": sha256_file(portable_benchmark_receipt),
            },
            "rdkit_audit_receipt": {
                "path": str(portable_audit_receipt.relative_to(staging)),
                "sha256": sha256_file(portable_audit_receipt),
                "parser": audit_receipt["parser"],
            },
            "rdkit_valid_records": {
                "path": str(portable_valid_records.relative_to(staging)),
                "bytes": portable_valid_records.stat().st_size,
                "sha256": sha256_file(portable_valid_records),
            },
            "records": records,
            "labels": int(benchmark_receipt["labels"]),
            "test_identity_cutoff_counts": cutoff_counts,
            "identity_cutoff_alignment": audit_receipt["identity_cutoff_alignment"],
            "table": {
                "path": table_path.name,
                "bytes": table_path.stat().st_size,
                "sha256": sha256_file(table_path),
            },
            "label_vocabulary": {
                "path": filtered_labels_path.name,
                "bytes": filtered_labels_path.stat().st_size,
                "sha256": sha256_file(filtered_labels_path),
            },
            "comparability_limitations": [
                "The model receives a sequence-only CA-residue view, not the GearNet graph.",
                "Identity cutoffs do not clear overlap with either pretraining corpus.",
            ],
        }
        (staging / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
        return receipt
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
