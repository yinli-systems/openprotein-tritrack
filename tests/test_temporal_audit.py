import csv
import gzip
import json
from pathlib import Path

from openprotein.provenance import sha256_file
from openprotein.temporal_audit import audit_swissprot_temporal_split


def _write_split(root: Path, split: str, rows: list[tuple[str, str]]) -> dict[str, object]:
    path = root / split / "part.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "accession"])
        writer.writeheader()
        for record_id, accession in rows:
            writer.writerow({"id": record_id, "accession": accession})
    return {
        "path": str(path.relative_to(root)),
        "split": split,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def test_temporal_audit_detects_post_cutoff_training_rows(tmp_path: Path) -> None:
    data = tmp_path / "data"
    files = [
        _write_split(data, "train", [("train-a", "P00001")]),
        _write_split(data, "validation", [("val-a", "P00002")]),
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"dataset_root": str(data), "files": files}), encoding="utf-8"
    )
    raw = tmp_path / "sprot.dat.gz"
    with gzip.open(raw, "wt", encoding="utf-8") as handle:
        handle.write(
            "AC   P00001;\n"
            "DT   01-JAN-2020, integrated into UniProtKB/Swiss-Prot.\n"
            "DT   01-JAN-2020, sequence version 1.\n"
            "DT   02-FEB-2025, entry version 2.\n"
            "//\n"
            "AC   P00002;\n"
            "DT   01-JAN-2025, integrated into UniProtKB/Swiss-Prot.\n"
            "DT   01-JAN-2025, sequence version 1.\n"
            "DT   02-FEB-2025, entry version 1.\n"
            "//\n"
        )
    output = tmp_path / "temporal.json"
    receipt = audit_swissprot_temporal_split(
        str(manifest), str(raw), "2024-01-01", str(output)
    )
    assert receipt["status"] == "temporal_split_absent"
    assert receipt["test_data_accessed"] is False
    assert receipt["splits"]["train"]["annotation_update_date"][
        "on_or_after_cutoff"
    ] == 1
