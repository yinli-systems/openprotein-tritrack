import csv
import json
from pathlib import Path

from openprotein.ec_benchmark_filter import filter_ec_benchmark
from openprotein.provenance import sha256_file


def _artifact(path: Path) -> dict[str, object]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def test_filter_ec_benchmark_binds_both_receipts(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    labels = raw / "labels.json"
    labels.write_text('["1.1.1.-"]\n', encoding="utf-8")
    table = raw / "benchmark.csv"
    fields = [
        "id",
        "split",
        "sequence",
        "label_indices_json",
        "below_30_percent",
        "below_40_percent",
        "below_50_percent",
        "below_70_percent",
        "below_95_percent",
    ]
    rows = [
        {"id": "TRAIN-A", "split": "train", "sequence": "AC", "label_indices_json": "[0]"},
        {"id": "TRAIN-B", "split": "train", "sequence": "AD", "label_indices_json": "[0]"},
        {"id": "VALID-A", "split": "valid", "sequence": "AE", "label_indices_json": "[0]"},
        {
            "id": "TEST-A",
            "split": "test",
            "sequence": "AF",
            "label_indices_json": "[0]",
            **{field: "1" for field in fields[4:]},
        },
    ]
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    source_sha = "a" * 64
    (raw / "receipt.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "test_data_accessed": True,
                "proteingym_rows_opened": False,
                "source": {"observed_sha256": source_sha},
                "table": _artifact(table),
                "label_vocabulary": _artifact(labels),
                "labels": 1,
            }
        ),
        encoding="utf-8",
    )

    audit = tmp_path / "audit"
    audit.mkdir()
    valid = audit / "rdkit-valid-records.csv"
    valid.write_text("id,split\nTRAIN-A,train\nVALID-A,valid\nTEST-A,test\n", encoding="utf-8")
    (audit / "receipt.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "test_data_accessed": True,
                "proteingym_rows_opened": False,
                "source": {"observed_sha256": source_sha},
                "parser": {"version": "2022.09.5"},
                "records": {
                    "train": {"valid": 1},
                    "valid": {"valid": 1},
                    "test": {"valid": 1},
                },
                "valid_records": _artifact(valid),
                "identity_cutoff_alignment": {},
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "filtered"
    receipt = filter_ec_benchmark(str(raw), str(audit), str(output))
    assert receipt["records"] == {"train": 1, "valid": 1, "test": 1}
    assert all(value == 1 for value in receipt["test_identity_cutoff_counts"].values())
    assert receipt["proteingym_rows_opened"] is False
    assert (output / receipt["raw_benchmark_receipt"]["path"]).is_file()
    assert (output / receipt["rdkit_audit_receipt"]["path"]).is_file()
    assert (output / receipt["rdkit_valid_records"]["path"]).is_file()
    with (output / "benchmark.csv").open(newline="") as handle:
        filtered = list(csv.DictReader(handle))
    assert [row["id"] for row in filtered] == ["TRAIN-A", "VALID-A", "TEST-A"]
