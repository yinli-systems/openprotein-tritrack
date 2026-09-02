import csv
import hashlib
import json
from pathlib import Path

from openprotein.homology import prepare_homology_inputs, summarize_homology_hits


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path, split: str, path: Path, records: int, residues: int) -> Path:
    manifest = {
        "dataset_root": str(root),
        "files": [
            {
                "path": str(path.relative_to(root)),
                "split": split,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "records": records,
                "residues": residues,
            }
        ],
        "split_totals": {split: {"records": records, "residues": residues}},
    }
    output = root / "manifest.json"
    output.write_text(json.dumps(manifest))
    return output


def test_prepare_and_summarize_homology_audit(tmp_path):
    train_root = tmp_path / "train-data"
    validation_root = tmp_path / "validation-data"
    (train_root / "train").mkdir(parents=True)
    (validation_root / "validation").mkdir(parents=True)
    train_csv = train_root / "train/train.csv"
    with train_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "name", "text"))
        writer.writeheader()
        writer.writerow({"id": "train-a", "name": "A", "text": "ACDE"})
        writer.writerow({"id": "train-b", "name": "B", "text": "VVVV"})
    validation_csv = validation_root / "validation/validation.csv"
    with validation_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "sequence"))
        writer.writeheader()
        writer.writerow({"id": "query-a", "sequence": "ACDE"})
        writer.writerow({"id": "query-b", "sequence": "AAAA"})
        writer.writerow({"id": "query-c", "sequence": "CCCC"})
    train_manifest = _manifest(train_root, "train", train_csv, records=2, residues=8)
    validation_manifest = _manifest(
        validation_root, "validation", validation_csv, records=3, residues=12
    )
    output = tmp_path / "audit"
    receipt = prepare_homology_inputs(
        str(train_manifest), str(validation_manifest), str(output)
    )
    assert receipt["test_data_accessed"] is False
    assert receipt["train"]["records"] == 2
    assert receipt["queries"]["records"] == 3

    hits = output / "hits.tsv"
    hits.write_text(
        "query-a\ttrain-a\t100.0\t4\t1\t4\t4\t1\t4\t4\t1e-9\t50\t1.0\t1.0\n"
        "query-b\ttrain-b\t55.0\t4\t1\t4\t4\t1\t4\t4\t1e-4\t20\t1.0\t1.0\n"
        "query-c\ttrain-b\t35.0\t4\t1\t4\t4\t1\t4\t4\t1e-3\t10\t1.0\t1.0\n"
    )
    result = summarize_homology_hits(
        str(output / "prepare.json"),
        str(hits),
        str(output / "homology.json"),
        str(output / "clean.txt"),
        str(output / "excluded.txt"),
        str(output / "top.tsv"),
        "test-version",
        ["mmseqs", "easy-search"],
    )
    assert result["status"] == "decontamination_required"
    assert result["thresholds"]["exact_duplicate"]["queries"] == 1
    assert result["thresholds"]["identity_50_query_coverage_80"]["queries"] == 2
    assert (output / "clean.txt").read_text().splitlines() == ["query-c"]
    assert (output / "excluded.txt").read_text().splitlines() == ["query-a", "query-b"]
    assert result["clean_queries"]["path"] == "clean.txt"
    assert result["hits"]["path"] == "hits.tsv"
