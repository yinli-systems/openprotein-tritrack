"""Homology-clustered Swiss-Prot/AlphaFold multimodal dataset construction."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Iterable
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openprotein.data.structure import verify_structure_database
from openprotein.provenance import sha256_file

OUTPUT_FIELDS = (
    "id",
    "cluster_id",
    "accession",
    "name",
    "sequence",
    "text",
    "sequence_sha256",
    "text_sha256",
    "evidence_codes_json",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _verify_swissprot_database(database: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", 0)) != 1:
        raise RuntimeError("unsupported Swiss-Prot manifest schema")
    expected = manifest.get("database")
    if not isinstance(expected, dict):
        raise RuntimeError("Swiss-Prot manifest has no database receipt")
    if database != Path(expected["path"]).resolve():
        raise RuntimeError("Swiss-Prot database path differs from its manifest")
    if database.stat().st_size != int(expected["bytes"]):
        raise RuntimeError("Swiss-Prot database byte size differs from its manifest")
    digest = sha256_file(database)
    if digest != expected["sha256"]:
        raise RuntimeError("Swiss-Prot database SHA-256 differs from its manifest")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "database_path": str(database),
        "database_sha256": digest,
        "records": int(expected["records"]),
        "records_with_text": int(expected["records_with_text"]),
        "residues": int(expected["residues"]),
    }


def export_swissprot_fasta(
    database_path: str,
    manifest_path: str,
    output_fasta: str,
    output_receipt: str,
) -> dict[str, Any]:
    """Export every verified Swiss-Prot canonical sequence for external clustering."""

    database = Path(database_path).resolve()
    manifest = Path(manifest_path).resolve()
    output = Path(output_fasta).resolve()
    receipt_path = Path(output_receipt).resolve()
    if output.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite clustered-data preparation artifacts")
    source = _verify_swissprot_database(database, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    records = 0
    residues = 0
    try:
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                for accession, sequence in connection.execute(
                    "SELECT accession, sequence FROM entries ORDER BY accession"
                ):
                    if not accession or not sequence or any(c.isspace() for c in accession):
                        raise RuntimeError("Swiss-Prot database contains an invalid FASTA record")
                    handle.write(f">{accession}\n{sequence}\n")
                    records += 1
                    residues += len(sequence)
        if records != source["records"] or residues != source["residues"]:
            raise RuntimeError("FASTA export totals differ from the Swiss-Prot manifest")
        os.replace(temporary, output)
        receipt = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "purpose": "homology_clustering_input",
            "source": source,
            "fasta": {
                "path": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
                "records": records,
                "residues": residues,
            },
        }
        _atomic_json(receipt_path, receipt)
        return receipt
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_clusters(path: Path, expected_members: int) -> tuple[dict[str, str], int]:
    member_to_cluster: dict[str, str] = {}
    clusters = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if len(row) != 2 or not row[0] or not row[1]:
                raise ValueError(f"invalid MMseqs cluster row at line {line_number}")
            representative, member = row
            if member in member_to_cluster:
                raise RuntimeError(f"MMseqs member occurs more than once: {member}")
            member_to_cluster[member] = representative
            clusters.add(representative)
    if len(member_to_cluster) != expected_members:
        raise RuntimeError(
            "MMseqs member count differs from Swiss-Prot: "
            f"{len(member_to_cluster)} != {expected_members}"
        )
    if not clusters <= set(member_to_cluster):
        raise RuntimeError("MMseqs output contains a representative absent from its members")
    return member_to_cluster, len(clusters)


def _split(cluster_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{cluster_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    if bucket < 9_800:
        return "train"
    if bucket < 9_900:
        return "validation"
    return "test"


def _paths_for_split(root: Path, split: str, shards: int) -> Iterable[Path]:
    count = shards if split == "train" else 1
    for index in range(count):
        yield root / split / f"{split}-{index:05d}-of-{count:05d}.csv"


def build_clustered_multimodal_dataset(
    swissprot_database: str,
    swissprot_manifest: str,
    structure_database: str,
    structure_manifest: str,
    cluster_tsv: str,
    output_root: str,
    mmseqs_version: str,
    mmseqs_command: Iterable[str],
    seed: int = 260903,
    train_shards: int = 96,
) -> dict[str, Any]:
    """Create a cluster-disjoint curated sequence/structure/text corpus."""

    if train_shards <= 0:
        raise ValueError("train_shards must be positive")
    annotations = Path(swissprot_database).resolve()
    annotations_manifest = Path(swissprot_manifest).resolve()
    structures = Path(structure_database).resolve()
    structures_manifest = Path(structure_manifest).resolve()
    clusters_path = Path(cluster_tsv).resolve()
    final_root = Path(output_root).resolve()
    if final_root.exists():
        raise FileExistsError(f"refusing to overwrite multimodal dataset: {final_root}")
    swissprot_receipt = _verify_swissprot_database(annotations, annotations_manifest)
    structure_receipt = verify_structure_database(str(structures), str(structures_manifest))
    member_to_cluster, cluster_count = _read_clusters(
        clusters_path, swissprot_receipt["records"]
    )

    final_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{final_root.name}.tmp-", dir=final_root.parent)
    )
    totals = {
        split: {"records": 0, "residues": 0, "annotation_utf8_bytes": 0, "clusters": set()}
        for split in ("train", "validation", "test")
    }
    skips = {"missing_structure": 0, "sequence_mismatch": 0, "empty_text": 0}
    output_paths = {
        split: list(_paths_for_split(temporary_root, split, train_shards))
        for split in totals
    }

    try:
        with ExitStack() as stack:
            writers = {}
            for split, paths in output_paths.items():
                for path in paths:
                    path.parent.mkdir(parents=True, exist_ok=True)
                handles = [
                    stack.enter_context(path.open("w", encoding="utf-8", newline=""))
                    for path in paths
                ]
                writers[split] = [
                    csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
                    for handle in handles
                ]
                for writer in writers[split]:
                    writer.writeheader()

            with sqlite3.connect(f"file:{annotations}?mode=ro", uri=True) as annotation_db:
                with sqlite3.connect(f"file:{structures}?mode=ro", uri=True) as structure_db:
                    rows = annotation_db.execute(
                        """
                        SELECT entry_name, accession, sequence, sequence_sha256,
                               annotation_text, annotation_sha256, evidence_codes_json
                        FROM entries ORDER BY accession
                        """
                    )
                    for (
                        entry_name,
                        accession,
                        sequence,
                        sequence_sha256,
                        text,
                        text_sha256,
                        evidence_codes_json,
                    ) in rows:
                        if not text:
                            skips["empty_text"] += 1
                            continue
                        structure = structure_db.execute(
                            "SELECT sequence FROM structures WHERE accession = ? AND fragment = 1",
                            (accession,),
                        ).fetchone()
                        if structure is None:
                            skips["missing_structure"] += 1
                            continue
                        if structure[0] != sequence:
                            skips["sequence_mismatch"] += 1
                            continue
                        cluster_id = member_to_cluster.get(accession)
                        if cluster_id is None:
                            raise RuntimeError(
                                f"Swiss-Prot accession is absent from clusters: {accession}"
                            )
                        split = _split(cluster_id, seed)
                        shard = (
                            int.from_bytes(hashlib.sha256(accession.encode()).digest()[:8], "big")
                            % len(writers[split])
                        )
                        writers[split][shard].writerow(
                            {
                                "id": accession,
                                "cluster_id": cluster_id,
                                "accession": accession,
                                "name": entry_name,
                                "sequence": sequence,
                                "text": text,
                                "sequence_sha256": sequence_sha256,
                                "text_sha256": text_sha256,
                                "evidence_codes_json": evidence_codes_json,
                            }
                        )
                        totals[split]["records"] += 1
                        totals[split]["residues"] += len(sequence)
                        totals[split]["annotation_utf8_bytes"] += len(text.encode("utf-8"))
                        totals[split]["clusters"].add(cluster_id)

        cluster_sets = [totals[split]["clusters"] for split in totals]
        if any(cluster_sets[i] & cluster_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise RuntimeError("homology cluster leaked across dataset splits")
        if any(totals[split]["records"] == 0 for split in totals):
            raise RuntimeError("clustered multimodal dataset contains an empty split")
        files = []
        for split, paths in output_paths.items():
            for path in paths:
                files.append(
                    {
                        "path": path.relative_to(temporary_root).as_posix(),
                        "split": split,
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        serializable_totals = {
            split: {
                **{key: value for key, value in values.items() if key != "clusters"},
                "clusters": len(values["clusters"]),
            }
            for split, values in totals.items()
        }
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "dataset_root": str(final_root),
            "split_policy": {
                "unit": "MMseqs2 homology cluster",
                "assignment": "sha256(seed:representative) modulo 10000",
                "thresholds": {"train": 9800, "validation": 9900, "test": 10000},
                "seed": seed,
                "cross_split_cluster_overlap": 0,
                "test_policy": "sealed: never used for optimization or checkpoint selection",
            },
            "mmseqs": {
                "version": mmseqs_version,
                "command": list(mmseqs_command),
                "cluster_tsv_path": str(clusters_path),
                "cluster_tsv_bytes": clusters_path.stat().st_size,
                "cluster_tsv_sha256": sha256_file(clusters_path),
                "members": len(member_to_cluster),
                "clusters": cluster_count,
            },
            "swissprot": swissprot_receipt,
            "structure": structure_receipt,
            "join_skips": skips,
            "split_totals": serializable_totals,
            "files": files,
        }
        _atomic_json(temporary_root / "manifest.json", manifest)
        os.replace(temporary_root, final_root)
        return manifest
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
