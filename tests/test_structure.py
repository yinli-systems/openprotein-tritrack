import csv
import hashlib
import io
import json
import sqlite3
import tarfile
from array import array
from pathlib import Path

import torch
import pytest

from openprotein.data.streaming import (
    JointMultimodalCSVStream,
    SequenceStructureCollator,
    SequenceStructureTextCollator,
    StructureCSVStream,
)
from openprotein.data.structure import (
    STRUCTURE_FEATURE_DIMENSION,
    build_alphafold_structure_index,
    ca_trace_features,
    parse_pdb_ca,
    sample_geometry_pairs,
    verify_structure_database,
)
from openprotein.data.tokenizer import EsmcTokenizer


ROOT = Path(__file__).resolve().parents[1]


def _atom(serial, residue, position, x, confidence):
    return (
        f"ATOM  {serial:5d}  CA  {residue:>3s} A{position:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{confidence:6.2f}           C  \n"
    )


def test_pdb_ca_parser_and_linear_space_pair_sampling():
    pdb = "".join(
        _atom(index, "ALA" if index % 2 else "GLY", index, float(index), 90.0)
        for index in range(1, 11)
    )
    parsed = parse_pdb_ca(io.BytesIO(pdb.encode()))
    assert parsed["sequence"] == "AGAGAGAGAG"
    coordinates = torch.tensor(parsed["coordinates"])
    confidence = torch.tensor(parsed["confidence"])
    pairs, labels = sample_geometry_pairs(
        coordinates,
        confidence,
        num_pairs=12,
        generator=torch.Generator().manual_seed(9),
        min_sequence_separation=3,
    )
    assert pairs.shape == (12, 2)
    assert labels.shape == (12,)
    assert torch.all((pairs[:, 1] - pairs[:, 0]).ge(3))
    assert len({tuple(pair) for pair in pairs.tolist()}) == 12


def test_ca_trace_features_are_rigid_motion_invariant():
    torch.manual_seed(2)
    coordinates = torch.randn(12, 3)
    confidence = torch.full((12,), 90.0)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    transformed = coordinates @ rotation + torch.tensor([11.0, -3.0, 7.0])
    original_features, original_mask = ca_trace_features(coordinates, confidence)
    moved_features, moved_mask = ca_trace_features(transformed, confidence)
    assert original_features.shape == (12, STRUCTURE_FEATURE_DIMENSION)
    assert torch.equal(original_mask, moved_mask)
    assert torch.allclose(original_features, moved_features, atol=2e-6)


def test_structure_index_records_builder_code_hash(tmp_path):
    pdb = "".join(_atom(index, "ALA", index, float(index), 90.0) for index in range(1, 5))
    source_tar = tmp_path / "structures.tar"
    with tarfile.open(source_tar, "w") as archive:
        payload = pdb.encode()
        member = tarfile.TarInfo("AF-P12345-F1-model_v6.pdb")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    source_definition = tmp_path / "source.json"
    source_definition.write_text(json.dumps({"pdb_tar_bytes": source_tar.stat().st_size}))

    manifest = build_alphafold_structure_index(
        str(source_tar),
        str(tmp_path / "structures.sqlite"),
        str(tmp_path / "manifest.json"),
        str(source_definition),
    )

    assert len(manifest["builder_code_sha256"]) == 64
    assert manifest["database"]["structures"] == 1
    assert manifest["database"]["residues"] == 4


def test_structure_database_verification_fails_closed_on_drift(tmp_path):
    database = tmp_path / "structures.sqlite"
    database.write_bytes(b"database")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "database": {
                    "path": str(database),
                    "bytes": database.stat().st_size,
                    "sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
                    "structures": 1,
                    "residues": 4,
                },
            }
        )
    )
    receipt = verify_structure_database(str(database), str(manifest_path))
    assert receipt["database_sha256"] == hashlib.sha256(b"database").hexdigest()
    database.write_bytes(b"databAse")
    with pytest.raises(RuntimeError, match="SHA-256"):
        verify_structure_database(str(database), str(manifest_path))


def test_structure_stream_exact_join_and_geometry_collation(tmp_path):
    sequence = "ACDEFGHIKL"
    database = tmp_path / "structures.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE structures (
                accession TEXT, fragment INTEGER, sequence TEXT,
                coordinates_f32le BLOB, confidence_f32le BLOB
            )
            """
        )
        coordinates = [value for index in range(len(sequence)) for value in (index, 0, 0)]
        confidence = [90.0] * len(sequence)
        connection.execute(
            "INSERT INTO structures VALUES (?, ?, ?, ?, ?)",
            (
                "P12345",
                1,
                sequence,
                array("f", coordinates).tobytes(),
                array("f", confidence).tobytes(),
            ),
        )
        connection.execute(
            "INSERT INTO structures VALUES (?, ?, ?, ?, ?)",
            (
                "P00000",
                1,
                sequence,
                array("f", coordinates).tobytes(),
                array("f", [10.0] * len(sequence)).tobytes(),
            ),
        )
    pairs = tmp_path / "pairs.csv"
    with pairs.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "accession", "sequence"])
        writer.writeheader()
        writer.writerow({"id": "cluster", "accession": "P12345", "sequence": sequence})
        writer.writerow({"id": "low", "accession": "P00000", "sequence": sequence})
    tokenizer = EsmcTokenizer.from_tokenizer_json(str(ROOT / "tests/fixtures/tiny_tokenizer.json"))
    examples = list(StructureCSVStream([str(pairs)], tokenizer, str(database), 12, seed=3))
    assert [example["record_id"] for example in examples] == ["cluster"]
    filtered = list(
        StructureCSVStream(
            [str(pairs)],
            tokenizer,
            str(database),
            12,
            seed=3,
            include_record_ids={"not-cluster"},
        )
    )
    assert filtered == []
    batch = SequenceStructureCollator(
        tokenizer, seed=4, geometry_pairs=8, distance_bins=64, mlm_probability=0.15
    )(examples)
    assert batch["pair_indices"].shape == (1, 8, 2)
    assert batch["distance_labels"].ge(0).all()
    assert batch["pair_indices"].min() >= 1


class _FakeTextTokenizer:
    def encode(self, text, max_length=None):
        ids = [2, *[4 + (ord(character) % 10) for character in text], 3]
        return ids if max_length is None else ids[: max_length - 1] + [3]

    def pad(self, examples):
        rows = list(examples)
        width = max(map(len, rows))
        ids = torch.zeros((len(rows), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = torch.tensor(row)
            mask[index, : len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def test_joint_stream_and_collator_emit_all_supervision_paths(tmp_path):
    sequence = "ACDEFGHIKLMN"
    database = tmp_path / "structures.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE structures (
                accession TEXT, fragment INTEGER, sequence TEXT,
                coordinates_f32le BLOB, confidence_f32le BLOB
            )
            """
        )
        coordinates = [
            value
            for index in range(len(sequence))
            for value in (float(index), float(index % 3), float(index % 2))
        ]
        connection.execute(
            "INSERT INTO structures VALUES (?, ?, ?, ?, ?)",
            (
                "P12345",
                1,
                sequence,
                array("f", coordinates).tobytes(),
                array("f", [90.0] * len(sequence)).tobytes(),
            ),
        )
    pairs = tmp_path / "pairs.csv"
    with pairs.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "accession", "sequence", "text"]
        )
        writer.writeheader()
        for index in range(2):
            writer.writerow(
                {
                    "id": f"cluster-{index}",
                    "accession": "P12345",
                    "sequence": sequence,
                    "text": "curated enzyme annotation",
                }
            )
    protein_tokenizer = EsmcTokenizer.from_tokenizer_json(
        str(ROOT / "tests/fixtures/tiny_tokenizer.json")
    )
    text_tokenizer = _FakeTextTokenizer()
    examples = list(
        JointMultimodalCSVStream(
            [str(pairs)],
            protein_tokenizer,
            text_tokenizer,
            str(database),
            protein_max_length=14,
            text_max_length=16,
            seed=3,
        )
    )
    batch = SequenceStructureTextCollator(
        protein_tokenizer,
        text_tokenizer,
        seed=4,
        geometry_pairs=8,
        distance_bins=64,
        structure_mask_probability=0.5,
    )(examples)
    assert set(batch) == {
        "protein_input_ids",
        "protein_attention_mask",
        "protein_labels",
        "text_input_ids",
        "text_attention_mask",
        "text_labels",
        "structure_input_features",
        "structure_attention_mask",
        "structure_labels",
        "structure_label_mask",
        "pair_indices",
        "distance_labels",
        "contrastive_group_ids",
    }
    assert batch["structure_input_features"].shape[-1] == STRUCTURE_FEATURE_DIMENSION
    assert batch["structure_label_mask"].any()
    assert batch["contrastive_group_ids"].shape == (2,)
    assert batch["contrastive_group_ids"].unique().numel() == 1
    assert torch.equal(
        batch["structure_input_features"][batch["structure_label_mask"]],
        torch.zeros_like(batch["structure_input_features"][batch["structure_label_mask"]]),
    )
