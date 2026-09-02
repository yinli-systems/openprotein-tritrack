import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

from openprotein.ec_benchmark import _sequence_from_pdb, build_ec_benchmark


def _pdb(sequence: str) -> bytes:
    residue_names = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU"}
    lines = []
    for index, residue in enumerate(sequence, start=1):
        lines.append(
            f"ATOM  {index:5d}  CA  {residue_names[residue]:>3s} A{index:4d}    "
            f"{float(index):8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{90.0:6.2f}           C  \n"
        )
    return "".join(lines).encode()


def _nested(split: str, record_id: str) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{split}/{record_id}_1.pdb", _pdb("ACDE"))
    return payload.getvalue()


def test_build_ec_benchmark_preserves_splits_and_cutoffs(tmp_path: Path) -> None:
    source = tmp_path / "ec.zip"
    annotation = (
        "### EC-numbers\n1.1.1.-\t2.2.2.-\n### PDB-chain\tEC-numbers\n"
        "TRAIN-A\t1.1.1.-\nVALID-A\t2.2.2.-\nTEST-A\t1.1.1.-,2.2.2.-\n"
    )
    cutoffs = (
        "PDB-chain,<30%,<40%,<50%,<70%,<95%\n"
        "TEST-A,1,1,1,1,1\nSTALE-A,1,1,1,1,1\n"
    )
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        root = "EnzymeCommission"
        archive.writestr(f"{root}/nrPDB-EC_annot.tsv", annotation)
        archive.writestr(f"{root}/nrPDB-EC_test.csv", cutoffs)
        archive.writestr(f"{root}/train.zip", _nested("train", "TRAIN-A"))
        archive.writestr(f"{root}/valid.zip", _nested("valid", "VALID-A"))
        archive.writestr(f"{root}/test.zip", _nested("test", "TEST-A"))
    definition = tmp_path / "source.json"
    definition.write_text(
        json.dumps(
            {
                "bytes": source.stat().st_size,
                "md5": hashlib.md5(source.read_bytes()).hexdigest(),  # noqa: S324
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    receipt = build_ec_benchmark(str(source), str(definition), str(output))
    assert receipt["records"] == {"train": 1, "valid": 1, "test": 1}
    assert receipt["labels"] == 2
    assert receipt["test_identity_cutoff_counts"]["below_30_percent"] == 1
    with (output / "benchmark.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["split"] for row in rows] == ["train", "valid", "test"]
    assert rows[-1]["label_indices_json"] == "[0,1]"
    assert receipt["test_data_accessed"] is True
    assert receipt["proteingym_rows_opened"] is False
    assert receipt["identity_cutoff_alignment"] == {
        "test_records_without_published_cutoff": [],
        "published_cutoff_records_not_in_test_archive": ["STALE-A"],
        "policy": (
            "retain every official test-archive record; records without a published "
            "cutoff remain outside all cutoff-specific metric subsets"
        ),
    }


def test_sequence_parser_uses_non_primary_altloc_only_when_required() -> None:
    payload = (
        "ATOM      1  CA BSER B  21     203.258  50.788 -28.599  0.50 41.29           C  \n"
        "ATOM      2  CA AALA B  22     201.832  50.468 -28.306  0.50 41.25           C  \n"
        "ATOM      3  CA BCYS B  22     201.832  50.468 -28.306  0.50 41.25           C  \n"
        "ATOM      4  CA  ASP B  23     201.676  50.004 -26.862  1.00 42.42           C  \n"
    ).encode()
    sequence, fallback_residues = _sequence_from_pdb(io.BytesIO(payload))
    assert sequence == "SAD"
    assert fallback_residues == 1
