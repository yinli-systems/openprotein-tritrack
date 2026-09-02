import csv
import hashlib
import json
import sqlite3

import pytest

from openprotein.data.clustered_multimodal import (
    _split,
    build_clustered_multimodal_dataset,
    export_swissprot_fasta,
)


def _representative(split: str, seed: int) -> str:
    for index in range(100_000):
        candidate = f"CLUSTER{index:05d}"
        if _split(candidate, seed) == split:
            return candidate
    raise AssertionError(f"unable to find a representative for {split}")


def _write_sources(tmp_path, seed: int):
    representatives = {
        split: _representative(split, seed)
        for split in ("train", "validation", "test")
    }
    train_member = "TRAINMEMBER"
    sequences = {
        representatives["train"]: "ACDEFG",
        train_member: "HIKLMN",
        representatives["validation"]: "PQRSTV",
        representatives["test"]: "WYACDE",
    }

    annotations = tmp_path / "swissprot.sqlite"
    with sqlite3.connect(annotations) as connection:
        connection.execute(
            """
            CREATE TABLE entries (
                entry_name TEXT, accession TEXT, sequence TEXT,
                sequence_sha256 TEXT, annotation_text TEXT,
                annotation_sha256 TEXT, evidence_codes_json TEXT
            )
            """
        )
        for accession, sequence in sequences.items():
            annotation = f"curated annotation for {accession}"
            connection.execute(
                "INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    f"{accession}_ENTRY",
                    accession,
                    sequence,
                    hashlib.sha256(sequence.encode()).hexdigest(),
                    annotation,
                    hashlib.sha256(annotation.encode()).hexdigest(),
                    '["ECO:0000269"]',
                ),
            )
    swissprot_manifest = tmp_path / "swissprot-manifest.json"
    swissprot_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "database": {
                    "path": str(annotations),
                    "bytes": annotations.stat().st_size,
                    "sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
                    "records": len(sequences),
                    "records_with_text": len(sequences),
                    "residues": sum(map(len, sequences.values())),
                },
            }
        )
    )

    structures = tmp_path / "structures.sqlite"
    with sqlite3.connect(structures) as connection:
        connection.execute(
            "CREATE TABLE structures (accession TEXT, fragment INTEGER, sequence TEXT)"
        )
        for accession, sequence in sequences.items():
            connection.execute(
                "INSERT INTO structures VALUES (?, ?, ?)", (accession, 1, sequence)
            )
    structure_manifest = tmp_path / "structure-manifest.json"
    structure_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "database": {
                    "path": str(structures),
                    "bytes": structures.stat().st_size,
                    "sha256": hashlib.sha256(structures.read_bytes()).hexdigest(),
                    "structures": len(sequences),
                    "residues": sum(map(len, sequences.values())),
                },
            }
        )
    )

    clusters = tmp_path / "clusters.tsv"
    clusters.write_text(
        "\n".join(
            [
                f"{representatives['train']}\t{representatives['train']}",
                f"{representatives['train']}\t{train_member}",
                f"{representatives['validation']}\t{representatives['validation']}",
                f"{representatives['test']}\t{representatives['test']}",
            ]
        )
        + "\n"
    )
    return annotations, swissprot_manifest, structures, structure_manifest, clusters


def test_clustered_multimodal_build_is_cluster_disjoint_and_content_addressed(tmp_path):
    seed = 260903
    annotations, annotation_manifest, structures, structure_manifest, clusters = (
        _write_sources(tmp_path, seed)
    )
    fasta = tmp_path / "swissprot.fasta"
    fasta_receipt = tmp_path / "fasta-receipt.json"
    export = export_swissprot_fasta(
        str(annotations), str(annotation_manifest), str(fasta), str(fasta_receipt)
    )
    assert export["fasta"]["records"] == 4
    assert fasta.read_text().count(">") == 4

    output = tmp_path / "clustered"
    manifest = build_clustered_multimodal_dataset(
        str(annotations),
        str(annotation_manifest),
        str(structures),
        str(structure_manifest),
        str(clusters),
        str(output),
        "15-6f452",
        ["mmseqs", "easy-cluster", "swissprot.fasta", "clusters", "tmp"],
        seed=seed,
        train_shards=2,
    )

    assert {key: value["records"] for key, value in manifest["split_totals"].items()} == {
        "train": 2,
        "validation": 1,
        "test": 1,
    }
    assert manifest["split_policy"]["cross_split_cluster_overlap"] == 0
    assert manifest["split_policy"]["test_policy"].startswith("sealed")
    assert (output / "manifest.json").is_file()
    assert len(manifest["files"]) == 4
    for receipt in manifest["files"]:
        artifact = output / receipt["path"]
        assert artifact.stat().st_size == receipt["bytes"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == receipt["sha256"]
    with (output / "test" / "test-00000-of-00001.csv").open(newline="") as handle:
        test_rows = list(csv.DictReader(handle))
    assert len(test_rows) == 1


def test_clustered_multimodal_build_fails_closed_on_manifest_path_drift(tmp_path):
    seed = 260903
    annotations, annotation_manifest, _, _, _ = _write_sources(tmp_path, seed)
    moved = tmp_path / "copied.sqlite"
    moved.write_bytes(annotations.read_bytes())
    with pytest.raises(RuntimeError, match="path differs"):
        export_swissprot_fasta(
            str(moved),
            str(annotation_manifest),
            str(tmp_path / "out.fasta"),
            str(tmp_path / "out.json"),
        )
