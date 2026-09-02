import csv
import json
from pathlib import Path

import pytest
import torch

from openprotein.provenance import sha256_file
from openprotein.retrieval_stress import (
    build_retrieval_stress_benchmark,
    large_pool_retrieval_metrics,
    load_retrieval_stress_rows,
)


def _write_split(root: Path, split: str, rows: list[tuple[str, str]]) -> dict[str, object]:
    path = root / split / "part.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "accession",
                "name",
                "sequence",
                "text",
                "sequence_sha256",
            ],
        )
        writer.writeheader()
        for record_id, sequence in rows:
            writer.writerow(
                {
                    "id": record_id,
                    "accession": record_id.upper(),
                    "name": f"name-{record_id}",
                    "sequence": sequence,
                    "text": f"function {record_id}",
                    "sequence_sha256": f"sha-{record_id}",
                }
            )
    return {
        "path": str(path.relative_to(root)),
        "split": split,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def test_build_retrieval_stress_is_manifest_bound_and_ordered(tmp_path: Path) -> None:
    root = tmp_path / "data"
    files = [
        _write_split(root, "train", [(f"t{i}", "AAAA") for i in range(5)]),
        _write_split(root, "validation", [("q0", "CCCC"), ("q1", "DDDD")]),
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"dataset_root": str(root), "files": files}), encoding="utf-8"
    )
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("q1\nq0\n", encoding="utf-8")
    output = tmp_path / "artifact"
    receipt = build_retrieval_stress_benchmark(
        str(manifest),
        str(allowlist),
        str(output / "benchmark.csv"),
        str(output / "benchmark.json"),
        distractors=3,
        seed=7,
    )
    rows, loaded = load_retrieval_stress_rows(
        str(output / "benchmark.csv"), str(output / "benchmark.json")
    )
    assert [row["id"] for row in rows[:2]] == ["q1", "q0"]
    assert len(rows) == 5
    assert receipt["benchmark"]["distractors"] == 3
    assert loaded["test_data_accessed"] is False


def test_large_pool_metrics_report_exact_map_and_stable_ties() -> None:
    protein = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    text = protein.clone()
    metrics = large_pool_retrieval_metrics(
        protein, text, query_count=2, device_name="cpu", chunk_size=1
    )
    for direction in ("protein_to_text", "text_to_protein"):
        assert metrics[direction]["mean_average_precision"] == pytest.approx(1.0)
        assert metrics[direction]["recall_at_1"] == pytest.approx(1.0)
