import csv
import json
from pathlib import Path

import torch

from openprotein.data.manifest import build_csv_manifest
from openprotein.data.masking import mask_protein_tokens


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "name", "text"])
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_hashes_and_counts_without_exposing_rows(tmp_path):
    _write_csv(
        tmp_path / "train/train_00.csv",
        [
            {"id": "1", "name": "a", "text": "ACDE"},
            {"id": "2", "name": "b", "text": "GGX"},
        ],
    )
    _write_csv(
        tmp_path / "validation/validation_00.csv",
        [{"id": "3", "name": "c", "text": "MNPQ"}],
    )
    output = tmp_path / "manifest.json"
    manifest = build_csv_manifest(
        str(tmp_path),
        str(output),
        {"name": "fixture", "url": "https://example.invalid", "release": "1", "license": "CC0"},
    )
    assert manifest["split_totals"]["train"]["records"] == 2
    assert manifest["split_totals"]["train"]["residues"] == 7
    assert len(manifest["files"][0]["sha256"]) == 64
    serialized = output.read_text()
    assert "ACDE" not in serialized
    assert json.loads(serialized)["test_policy"].startswith("sealed")


def test_masking_is_deterministic_and_never_masks_special_tokens():
    ids = torch.tensor([[0, 5, 6, 7, 1]])
    attention = torch.ones_like(ids)
    first = mask_protein_tokens(
        ids,
        attention,
        mask_token_id=9,
        vocabulary_size=10,
        special_token_ids=[0, 1],
        generator=torch.Generator().manual_seed(123),
        mlm_probability=1.0,
    )
    second = mask_protein_tokens(
        ids,
        attention,
        mask_token_id=9,
        vocabulary_size=10,
        special_token_ids=[0, 1],
        generator=torch.Generator().manual_seed(123),
        mlm_probability=1.0,
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[1][0, 0].item() == -100
    assert first[1][0, 4].item() == -100
    assert first[1][0, 1:4].tolist() == [5, 6, 7]


def test_masking_selects_one_fallback_per_eligible_row():
    ids = torch.tensor([[0, 5, 6, 1], [0, 7, 8, 1]])
    attention = torch.ones_like(ids)
    _, labels = mask_protein_tokens(
        ids,
        attention,
        mask_token_id=9,
        vocabulary_size=10,
        special_token_ids=[0, 1],
        generator=torch.Generator().manual_seed(123),
        mlm_probability=1e-12,
    )
    assert labels.ne(-100).sum(dim=1).tolist() == [1, 1]
