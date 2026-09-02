import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest
import torch

from openprotein.data.packed import (
    IDENTIFIER_RECORD,
    PackedProteinBatcher,
    PackedSequenceShard,
    build_packed_csv_dataset,
    build_selected_mgnify_packed_dataset,
    verify_packed_manifest,
)
from openprotein.data.tokenizer import EsmcTokenizer


ROOT = Path(__file__).resolve().parents[1]


def _build_fixture(tmp_path: Path):
    source_root = tmp_path / "csv"
    train = source_root / "train"
    train.mkdir(parents=True)
    paths = []
    entries = []
    sequences = [["ACDEFG", "MNPQRS", "WYAC"], ["VVVV", "BXZUO", "KLMNPQRST"]]
    for index, rows in enumerate(sequences):
        path = train / f"train-{index:05d}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("id", "name", "text"))
            writer.writeheader()
            for row_number, sequence in enumerate(rows):
                writer.writerow(
                    {
                        "id": f"{index}-{row_number}",
                        "name": "fixture",
                        "text": sequence,
                    }
                )
        paths.append(path)
        entries.append(
            {
                "path": path.relative_to(source_root).as_posix(),
                "split": "train",
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    source_manifest = source_root / "manifest.json"
    source_manifest.write_text(json.dumps({"schema_version": 1, "files": entries}))
    output = tmp_path / "packed"
    manifest = build_packed_csv_dataset(
        [str(path) for path in paths],
        str(output),
        str(ROOT / "configs/scratch/tokenizer/esm2-33.json"),
        str(source_manifest),
        str(source_root),
    )
    return output, manifest


def test_packed_builder_reader_and_manifest_are_exact(tmp_path):
    output, manifest = _build_fixture(tmp_path)
    assert manifest["totals"] == {
        "shards": 2,
        "records": 6,
        "residues": 34,
        "unknown_residues": 0,
    }
    indices = sorted(str(path) for path in output.glob("train/*.idx"))
    verified = verify_packed_manifest(
        str(output / "manifest.json"),
        expected_index_paths=indices,
        expected_tokenizer_sha256=manifest["tokenizer"]["sha256"],
    )
    assert verified["test_data_accessed"] is False
    shard = PackedSequenceShard(indices[0])
    assert shard.records == 3
    assert [shard.read(index) for index in range(3)] == [
        bytes([5, 23, 13, 9, 18, 6]),
        bytes([20, 17, 14, 16, 10, 8]),
        bytes([22, 19, 5, 23]),
    ]
    shard.close()


def test_packed_batcher_restart_replays_pending_record(tmp_path):
    output, _ = _build_fixture(tmp_path)
    indices = sorted(str(path) for path in output.glob("train/*.idx"))
    tokenizer = EsmcTokenizer.from_tokenizer_json(
        str(ROOT / "configs/scratch/tokenizer/esm2-33.json")
    )
    batcher = PackedProteinBatcher(indices, tokenizer, 8, 8, 4, 17, 0.15)
    batcher.next()
    state = batcher.state_dict()
    expected = batcher.next()
    restored = PackedProteinBatcher(indices, tokenizer, 8, 8, 4, 17, 0.15)
    restored.load_state_dict(state)
    actual = restored.next()
    for key in expected:
        assert torch.equal(expected[key], actual[key]), key


def test_packed_batcher_lookahead_is_restartable_and_length_bucketed(tmp_path):
    output, _ = _build_fixture(tmp_path)
    indices = sorted(str(path) for path in output.glob("train/*.idx"))
    tokenizer = EsmcTokenizer.from_tokenizer_json(
        str(ROOT / "configs/scratch/tokenizer/esm2-33.json")
    )
    batcher = PackedProteinBatcher(
        indices, tokenizer, 8, 16, 4, 17, 0.15, batch_lookahead=5
    )
    first = batcher.next()
    assert first["protein_input_ids"].shape[0] >= 2
    assert batcher.lookahead_pool
    cursor_after_window_fill = (batcher.epoch, batcher.file_cursor, batcher.record_cursor)
    state = batcher.state_dict()
    expected = batcher.next()
    assert (batcher.epoch, batcher.file_cursor, batcher.record_cursor) == cursor_after_window_fill

    restored = PackedProteinBatcher(
        indices, tokenizer, 8, 16, 4, 17, 0.15, batch_lookahead=5
    )
    restored.load_state_dict(state)
    actual = restored.next()
    for key in expected:
        assert torch.equal(expected[key], actual[key]), key

    boundary_state = batcher.state_dict()
    assert len(batcher.lookahead_pool) == 1
    expected = batcher.next()
    assert expected["protein_input_ids"].shape[0] >= 2
    assert (batcher.epoch, batcher.file_cursor, batcher.record_cursor) != (
        cursor_after_window_fill
    )
    restored = PackedProteinBatcher(
        indices, tokenizer, 8, 16, 4, 17, 0.15, batch_lookahead=5
    )
    restored.load_state_dict(boundary_state)
    actual = restored.next()
    for key in expected:
        assert torch.equal(expected[key], actual[key]), key

    mismatched = PackedProteinBatcher(
        indices, tokenizer, 8, 16, 4, 17, 0.15, batch_lookahead=4
    )
    with pytest.raises(RuntimeError, match="lookahead differs"):
        mismatched.load_state_dict(state)


def test_packed_batcher_lookahead_refill_consumes_tail_without_underfill(tmp_path):
    output, _ = _build_fixture(tmp_path)
    indices = sorted(str(path) for path in output.glob("train/*.idx"))
    tokenizer = EsmcTokenizer.from_tokenizer_json(
        str(ROOT / "configs/scratch/tokenizer/esm2-33.json")
    )
    batcher = PackedProteinBatcher(
        indices, tokenizer, 8, 16, 4, 17, 0.15, batch_lookahead=5
    )
    batcher.collator = lambda examples: examples
    batcher.next()
    batcher.next()
    assert len(batcher.lookahead_pool) == 1
    tail_record = batcher.lookahead_pool[0]["record_id"]
    cursor_before_refill = (batcher.epoch, batcher.file_cursor, batcher.record_cursor)

    boundary = batcher.next()

    assert len(boundary) >= 2
    assert tail_record in {example["record_id"] for example in boundary}
    assert (batcher.epoch, batcher.file_cursor, batcher.record_cursor) != cursor_before_refill


def test_packed_batcher_topology_migration_preserves_pending_and_unread_records(
    tmp_path,
):
    output, _ = _build_fixture(tmp_path)
    indices = sorted(str(path) for path in output.glob("train/*.idx"))
    tokenizer = EsmcTokenizer.from_tokenizer_json(
        str(ROOT / "configs/scratch/tokenizer/esm2-33.json")
    )
    old = [
        PackedProteinBatcher(
            indices,
            tokenizer,
            8,
            8,
            4,
            17,
            0.15,
            rank=rank,
            world_size=2,
            batch_lookahead=2,
        )
        for rank in range(2)
    ]
    pending = [batcher._next_example() for batcher in old]
    for batcher, example in zip(old, pending):
        batcher.lookahead_pool = [example]

    migrated = PackedProteinBatcher(
        indices,
        tokenizer,
        8,
        8,
        4,
        17,
        0.15,
        rank=0,
        world_size=1,
        batch_lookahead=2,
    )
    migrated.load_topology_migration_state(
        [batcher.state_dict() for batcher in old], migration_seed=1234
    )
    observed = [migrated._next_example()["record_id"] for _ in range(6)]

    reference = PackedProteinBatcher(
        indices, tokenizer, 8, 8, 4, 17, 0.15, batch_lookahead=2
    )
    expected = [reference._next_example()["record_id"] for _ in range(6)]
    assert set(observed) == set(expected)
    assert len(observed) == len(set(observed))
    assert {example["record_id"] for example in pending} <= set(observed)


def test_packed_batcher_topology_migration_state_is_restartable(tmp_path):
    output, _ = _build_fixture(tmp_path)
    indices = sorted(str(path) for path in output.glob("train/*.idx"))
    tokenizer = EsmcTokenizer.from_tokenizer_json(
        str(ROOT / "configs/scratch/tokenizer/esm2-33.json")
    )
    old = []
    for rank in range(2):
        batcher = PackedProteinBatcher(
            indices, tokenizer, 8, 8, 4, 17, 0.15, rank=rank, world_size=2
        )
        batcher.lookahead_pool = [batcher._next_example()]
        old.append(batcher.state_dict())
    migrated = PackedProteinBatcher(indices, tokenizer, 8, 8, 4, 17, 0.15)
    migrated.load_topology_migration_state(old, migration_seed=1234)
    migrated._next_example()
    state = migrated.state_dict()
    expected = migrated._next_example()

    restored = PackedProteinBatcher(indices, tokenizer, 8, 8, 4, 17, 0.15)
    restored.load_state_dict(state)
    assert restored._next_example() == expected


def test_packed_manifest_detects_corruption(tmp_path):
    output, _ = _build_fixture(tmp_path)
    token_path = next(output.glob("train/*.bin"))
    token_path.write_bytes(token_path.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="size changed"):
        verify_packed_manifest(str(output / "manifest.json"))


def test_packed_manifest_content_hashes_partition_exactly_across_ranks(
    tmp_path, monkeypatch
):
    output, manifest = _build_fixture(tmp_path)
    expected = {str((output / item["path"]).resolve()) for item in manifest["files"]}
    observed = []
    original = hashlib.sha256

    def recording_sha256_file(path):
        observed.append(str(Path(path).resolve()))
        digest = original()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    monkeypatch.setattr("openprotein.data.packed.sha256_file", recording_sha256_file)
    rank_files = []
    for rank in range(2):
        start = len(observed)
        verify_packed_manifest(
            str(output / "manifest.json"),
            content_shard_index=rank,
            content_shard_count=2,
        )
        rank_files.append(set(observed[start:]))

    assert rank_files[0].isdisjoint(rank_files[1])
    assert rank_files[0] | rank_files[1] == expected
    rank_bytes = [sum(Path(path).stat().st_size for path in paths) for paths in rank_files]
    assert max(rank_bytes) - min(rank_bytes) <= max(
        Path(path).stat().st_size for path in expected
    )
    with pytest.raises(ValueError, match="content_shard_count"):
        verify_packed_manifest(
            str(output / "manifest.json"), content_shard_index=0, content_shard_count=0
        )
    with pytest.raises(ValueError, match="content_shard_index"):
        verify_packed_manifest(
            str(output / "manifest.json"), content_shard_index=2, content_shard_count=2
        )


def test_mgnify_builder_verifies_full_source_and_emits_split_manifests(tmp_path):
    fasta = tmp_path / "mgy_clusters.fa.gz"
    records = {
        "MGYP000000000001": "ACDEFG",
        "MGYP000000000002": "MNPQRS",
        "MGYP000000000003": "VVVV",
        "MGYP000000000004": "BXZUO",
    }
    content = "".join(
        f">{accession} FL=1\n{sequence}\n" for accession, sequence in records.items()
    ).encode()
    with gzip.GzipFile(filename=fasta, mode="wb", mtime=0) as handle:
        handle.write(content)
    raw = fasta.read_bytes()
    definition = tmp_path / "definition.json"
    definition.write_text(
        json.dumps(
            {
                "release": "2026_07",
                "fasta_bytes": len(raw),
                "fasta_md5": hashlib.md5(raw).hexdigest(),
                "cluster_representatives": len(records),
                "cluster_representative_residues": sum(map(len, records.values())),
            }
        )
    )
    receipt = tmp_path / "retrieval.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "VERIFIED_RAW_INPUT",
                "path": str(fasta),
                "bytes": len(raw),
                "md5": hashlib.md5(raw).hexdigest(),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    )
    output = tmp_path / "mgnify-packed"
    manifest = build_selected_mgnify_packed_dataset(
        str(fasta),
        str(output),
        str(ROOT / "configs/scratch/tokenizer/esm2-33.json"),
        str(definition),
        str(receipt),
        selection_basis_points=10_000,
        train_basis_points=8_000,
        validation_basis_points=1_000,
        shard_counts={"train": 2, "validation": 1, "test": 1},
    )
    assert manifest["source"]["records"] == 4
    assert manifest["source"]["residues"] == 21
    assert manifest["selected_records"] == 4
    assert manifest["selected_residues"] == 21
    assert sum(item["records"] for item in manifest["split_totals"].values()) == 4
    identifiers = set()
    for path in output.glob("*/*.ids"):
        raw_ids = path.read_bytes()
        for offset in range(0, len(raw_ids), IDENTIFIER_RECORD.size):
            identifiers.add(IDENTIFIER_RECORD.unpack_from(raw_ids, offset)[0])
    assert identifiers == {1, 2, 3, 4}
    train_indices = sorted(str(path) for path in output.glob("train/*.idx"))
    verified = verify_packed_manifest(
        str(output / "train-manifest.json"), expected_index_paths=train_indices
    )
    assert verified["split"] == "train"


def test_mgnify_builder_accepts_hash_bound_zstd_shards(tmp_path):
    zstandard = pytest.importorskip("zstandard")
    source_root = tmp_path / "shards"
    source_root.mkdir()
    records = [
        ("MGYP000000000001", "ACDEFG"),
        ("MGYP000000000002", "MNPQRS"),
        ("MGYP000000000003", "VVVV"),
        ("MGYP000000000004", "BXZUO"),
    ]
    receipt_files = []
    for shard_index, shard_records in enumerate((records[:2], records[2:]), start=1):
        name = f"shard-{shard_index:06d}.fasta.zst"
        content = "".join(
            f">{accession} FL=1\n{sequence}\n" for accession, sequence in shard_records
        ).encode()
        path = source_root / name
        path.write_bytes(zstandard.ZstdCompressor().compress(content))
        receipt_files.append(
            {
                "path": f"upstream/{name}",
                "local_path": name,
                "shard_index": shard_index,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    revision = "a" * 40
    definition = tmp_path / "definition.json"
    definition.write_text(
        json.dumps(
            {
                "release": "2024_04",
                "cluster_representatives": len(records),
                "cluster_representative_residues": sum(len(item[1]) for item in records),
                "mirror_transport": {
                    "format": "fasta_zstd_shards",
                    "repo_id": "fixture/mgnify",
                    "revision": revision,
                    "shards": len(receipt_files),
                    "bytes": sum(item["bytes"] for item in receipt_files),
                },
            }
        )
    )
    receipt = tmp_path / "retrieval.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "VERIFIED_SHARDED_INPUT",
                "dataset_root": str(source_root),
                "repo_id": "fixture/mgnify",
                "revision": revision,
                "totals": {
                    "shards": len(receipt_files),
                    "bytes": sum(item["bytes"] for item in receipt_files),
                },
                "files": receipt_files,
            }
        )
    )
    output = tmp_path / "packed-zstd"
    manifest = build_selected_mgnify_packed_dataset(
        str(source_root),
        str(output),
        str(ROOT / "configs/scratch/tokenizer/esm2-33.json"),
        str(definition),
        str(receipt),
        selection_basis_points=10_000,
        train_basis_points=8_000,
        validation_basis_points=1_000,
        shard_counts={"train": 2, "validation": 1, "test": 1},
        minimum_train_residues=1,
    )
    assert manifest["source"]["transport"] == "pinned_fasta_zstd_shards"
    assert manifest["source"]["records"] == 4
    assert manifest["source"]["residues"] == 21
    assert manifest["source"]["shards"] == 2


def test_mgnify_builder_reports_observed_and_expected_source_totals(tmp_path):
    fasta = tmp_path / "mgy_clusters.fa.gz"
    content = b">MGYP000000000001 FL=1\nACDEFG\n"
    with gzip.GzipFile(filename=fasta, mode="wb", mtime=0) as handle:
        handle.write(content)
    raw = fasta.read_bytes()
    definition = tmp_path / "definition.json"
    definition.write_text(
        json.dumps(
            {
                "release": "fixture",
                "fasta_bytes": len(raw),
                "fasta_md5": hashlib.md5(raw).hexdigest(),
                "cluster_representatives": 2,
                "cluster_representative_residues": 11,
            }
        )
    )
    receipt = tmp_path / "retrieval.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "VERIFIED_RAW_INPUT",
                "path": str(fasta),
                "bytes": len(raw),
                "md5": hashlib.md5(raw).hexdigest(),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "records observed=1 expected=2; "
            "residues observed=6 expected=11"
        ),
    ):
        build_selected_mgnify_packed_dataset(
            str(fasta),
            str(tmp_path / "rejected"),
            str(ROOT / "configs/scratch/tokenizer/esm2-33.json"),
            str(definition),
            str(receipt),
            selection_basis_points=10_000,
            train_basis_points=8_000,
            validation_basis_points=1_000,
            shard_counts={"train": 1, "validation": 1, "test": 1},
        )
    assert not (tmp_path / "rejected").exists()
