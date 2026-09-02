"""Leakage-safe joins from UniRef representatives to curated annotations."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from openprotein.provenance import sha256_file


_REPID = re.compile(r"(?:^|[;\s])RepID=([^;\s]+)")
_OUTPUT_FIELDS = (
    "id",
    "accession",
    "name",
    "sequence",
    "text",
    "sequence_sha256",
    "text_sha256",
    "evidence_codes_json",
)


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _source_shards(root: Path) -> List[Path]:
    return [
        path
        for split in ("train", "validation", "test")
        for path in sorted((root / split).glob("*.csv"))
    ]


def join_uniref_swissprot(
    uniref_root: str,
    swissprot_database: str,
    swissprot_manifest: str,
    output_root: str,
) -> Dict[str, Any]:
    """Join only exact Swiss-Prot representatives and preserve UniRef splits."""

    sequence_root = Path(uniref_root).resolve()
    sequence_manifest_path = sequence_root / "manifest.json"
    annotation_database = Path(swissprot_database).resolve()
    annotation_manifest_path = Path(swissprot_manifest).resolve()
    final_root = Path(output_root).resolve()
    for required in (sequence_manifest_path, annotation_database, annotation_manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"required join input is missing: {required}")
    if final_root.exists():
        raise FileExistsError(f"refusing to overwrite multimodal dataset: {final_root}")
    source_shards = _source_shards(sequence_root)
    if not source_shards:
        raise ValueError("UniRef dataset contains no CSV shards")

    final_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{final_root.name}.tmp-", dir=final_root.parent)
    )
    join_counts = {
        "uniref_records": 0,
        "headers_without_repid": 0,
        "representatives_without_swissprot": 0,
        "sequence_mismatches": 0,
        "exact_pairs": 0,
    }
    files: List[Dict[str, Any]] = []

    try:
        with sqlite3.connect(f"file:{annotation_database}?mode=ro", uri=True) as connection:
            entry_names = {row[0] for row in connection.execute("SELECT entry_name FROM entries")}
            for source_path in source_shards:
                relative = source_path.relative_to(sequence_root)
                split = relative.parts[0]
                output_path = temporary_root / relative
                output_path.parent.mkdir(parents=True, exist_ok=True)
                records = 0
                residues = 0
                annotation_bytes = 0
                with source_path.open("r", encoding="utf-8", newline="") as source_handle:
                    reader = csv.DictReader(source_handle)
                    if reader.fieldnames is None or not {"id", "name", "text"} <= set(
                        reader.fieldnames
                    ):
                        raise ValueError(f"unexpected UniRef schema: {source_path}")
                    with output_path.open("w", encoding="utf-8", newline="") as output_handle:
                        writer = csv.DictWriter(
                            output_handle, fieldnames=_OUTPUT_FIELDS, lineterminator="\n"
                        )
                        writer.writeheader()
                        for row in reader:
                            join_counts["uniref_records"] += 1
                            match = _REPID.search(row["name"])
                            if match is None:
                                join_counts["headers_without_repid"] += 1
                                continue
                            entry_name = match.group(1)
                            if entry_name not in entry_names:
                                join_counts["representatives_without_swissprot"] += 1
                                continue
                            annotation = connection.execute(
                                """
                                SELECT accession, sequence, sequence_sha256, annotation_text,
                                       annotation_sha256, evidence_codes_json
                                FROM entries WHERE entry_name = ?
                                """,
                                (entry_name,),
                            ).fetchone()
                            if annotation is None:
                                raise RuntimeError("Swiss-Prot key set and database diverged")
                            accession, expected_sequence, sequence_sha, text, text_sha, evidence = (
                                annotation
                            )
                            sequence = row["text"].strip().upper()
                            if sequence != expected_sequence:
                                join_counts["sequence_mismatches"] += 1
                                continue
                            writer.writerow(
                                {
                                    "id": row["id"],
                                    "accession": accession,
                                    "name": entry_name,
                                    "sequence": sequence,
                                    "text": text,
                                    "sequence_sha256": sequence_sha,
                                    "text_sha256": text_sha,
                                    "evidence_codes_json": evidence,
                                }
                            )
                            records += 1
                            residues += len(sequence)
                            annotation_bytes += len(text.encode("utf-8"))
                            join_counts["exact_pairs"] += 1
                files.append(
                    {
                        "path": relative.as_posix(),
                        "split": split,
                        "bytes": output_path.stat().st_size,
                        "sha256": sha256_file(output_path),
                        "records": records,
                        "residues": residues,
                        "annotation_utf8_bytes": annotation_bytes,
                    }
                )

        split_totals: Dict[str, Dict[str, int]] = {}
        for split in ("train", "validation", "test"):
            selected = [item for item in files if item["split"] == split]
            split_totals[split] = {
                "files": len(selected),
                "bytes": sum(item["bytes"] for item in selected),
                "records": sum(item["records"] for item in selected),
                "residues": sum(item["residues"] for item in selected),
                "annotation_utf8_bytes": sum(
                    item["annotation_utf8_bytes"] for item in selected
                ),
            }
        if join_counts["exact_pairs"] == 0:
            raise RuntimeError("Swiss-Prot join produced zero exact sequence pairs")
        manifest: Dict[str, Any] = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "builder_code_sha256": sha256_file(Path(__file__).resolve()),
            "dataset_root": str(final_root),
            "columns": list(_OUTPUT_FIELDS),
            "sources": {
                "uniref_manifest": {
                    "path": str(sequence_manifest_path),
                    "sha256": sha256_file(sequence_manifest_path),
                },
                "swissprot_manifest": {
                    "path": str(annotation_manifest_path),
                    "sha256": sha256_file(annotation_manifest_path),
                },
                "swissprot_database": {
                    "path": str(annotation_database),
                    "sha256": sha256_file(annotation_database),
                },
            },
            "join_policy": (
                "UniRef header RepID equals Swiss-Prot entry_name and canonical sequences are "
                "byte-identical; output inherits the UniRef split and shard"
            ),
            "join_counts": join_counts,
            "split_totals": split_totals,
            "files": files,
            "test_policy": "sealed: never used for optimization or checkpoint selection",
        }
        _atomic_json(temporary_root / "manifest.json", manifest)
        os.replace(temporary_root, final_root)
        return manifest
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
