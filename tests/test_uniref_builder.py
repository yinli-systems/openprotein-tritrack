import gzip
import hashlib
import json
from pathlib import Path

import pytest

from openprotein.data.uniref import build_uniref_csv


def _source_fixture(tmp_path: Path):
    fasta = tmp_path / "uniref50.fasta.gz"
    content = (
        ">UniRef50_A alpha protein\nACDEFG\n"
        ">UniRef50_B beta protein\nMNPQRS\n"
        ">UniRef50_C ambiguous protein\nBXZUO\n"
        ">UniRef50_D short protein\nA\n"
        ">UniRef50_E invalid protein\nACD?\n"
    ).encode()
    with gzip.GzipFile(filename=fasta, mode="wb", mtime=0) as handle:
        handle.write(content)
    raw = fasta.read_bytes()
    definition = tmp_path / "source.json"
    definition.write_text(
        json.dumps(
            {
                "name": "fixture",
                "records": 5,
                "fasta_bytes": len(raw),
                "fasta_md5": hashlib.md5(raw).hexdigest(),
                "fasta_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    )
    return fasta, definition


def test_uniref_builder_is_deterministic_and_content_addressed(tmp_path):
    fasta, definition = _source_fixture(tmp_path)
    first = build_uniref_csv(
        str(fasta),
        str(tmp_path / "first"),
        str(definition),
        shard_counts={"train": 2, "validation": 1, "test": 1},
        train_basis_points=8000,
        validation_basis_points=1000,
        min_length=2,
    )
    second = build_uniref_csv(
        str(fasta),
        str(tmp_path / "second"),
        str(definition),
        shard_counts={"train": 2, "validation": 1, "test": 1},
        train_basis_points=8000,
        validation_basis_points=1000,
        min_length=2,
    )
    assert first["source_input"]["records_seen"] == 5
    assert first["filters"]["dropped"] == {
        "empty": 0,
        "invalid_residue": 1,
        "too_long": 0,
        "too_short": 1,
    }
    assert sum(split["records"] for split in first["split_totals"].values()) == 3
    assert [item["sha256"] for item in first["files"]] == [
        item["sha256"] for item in second["files"]
    ]
    assert "ACDEFG" not in (tmp_path / "first/manifest.json").read_text()


def test_uniref_builder_fails_closed_on_source_digest_drift(tmp_path):
    fasta, definition = _source_fixture(tmp_path)
    metadata = json.loads(definition.read_text())
    metadata["fasta_sha256"] = "0" * 64
    definition.write_text(json.dumps(metadata))
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="SHA-256"):
        build_uniref_csv(
            str(fasta),
            str(output),
            str(definition),
            shard_counts={"train": 1, "validation": 1, "test": 1},
        )
    assert not output.exists()
