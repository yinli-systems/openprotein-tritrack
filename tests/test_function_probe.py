import json

import torch

from openprotein.function_probe import (
    run_frozen_ec_probe,
    torchdrug_f1_max,
    torchdrug_micro_auprc,
)


def test_torchdrug_metrics_rank_perfect_predictions_first() -> None:
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    perfect = torch.tensor([[3.0, -3.0], [-3.0, 3.0]])
    reversed_logits = -perfect
    assert torchdrug_micro_auprc(perfect, targets) == 1.0
    assert torchdrug_f1_max(perfect, targets) == 1.0
    assert torchdrug_micro_auprc(reversed_logits, targets) < 1.0
    assert torchdrug_f1_max(reversed_logits, targets) < 1.0


def test_frozen_probe_uses_all_identity_cutoffs() -> None:
    rows = []
    for index in range(12):
        split = "train" if index < 6 else "valid" if index < 8 else "test"
        row = {
            "id": str(index),
            "split": split,
            "sequence": "ACDE",
            "label_indices_json": json.dumps([index % 2]),
        }
        for field in (
            "below_30_percent",
            "below_40_percent",
            "below_50_percent",
            "below_70_percent",
            "below_95_percent",
        ):
            row[field] = "1" if split == "test" else ""
        rows.append(row)
    embeddings = torch.tensor(
        [[1.0, 0.0] if index % 2 == 0 else [0.0, 1.0] for index in range(12)]
    )
    result = run_frozen_ec_probe(
        embeddings,
        rows,
        label_count=2,
        device_name="cpu",
        seeds=(1,),
        steps=20,
        validation_interval=5,
        learning_rate=0.1,
    )
    assert result["records"]["train"] == 6
    assert set(result["aggregate"]) == {
        "below_30_percent",
        "below_40_percent",
        "below_50_percent",
        "below_70_percent",
        "below_95_percent",
    }
    assert result["aggregate"]["below_30_percent"]["micro_auprc"]["mean"] == 1.0
    assert [item["step"] for item in result["runs"][0]["validation_history"]] == [
        5,
        10,
        15,
        20,
    ]
    assert result["runs"][0]["actual_steps"] == 20
    assert result["convergence"]["runs_early_stopped"] == 0
