"""Frozen-encoder multilabel probes with TorchDrug-compatible EC metrics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from openprotein.provenance import sha256_file


EC_CUTOFF_FIELDS = (
    "below_30_percent",
    "below_40_percent",
    "below_50_percent",
    "below_70_percent",
    "below_95_percent",
)


def load_ec_benchmark(
    table_path: str,
    receipt_path: str,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Verify a compact EC benchmark receipt and return its ordered rows."""

    table = Path(table_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    if receipt.get("status") != "passed":
        raise RuntimeError("EC benchmark receipt did not pass")
    if receipt.get("test_data_accessed") is not True:
        raise RuntimeError("EC benchmark receipt does not declare public-test access")
    if receipt.get("proteingym_rows_opened") is not False:
        raise RuntimeError("EC benchmark build opened ProteinGym rows")
    table_receipt = receipt["table"]
    if table.stat().st_size != int(table_receipt["bytes"]):
        raise RuntimeError("EC benchmark table byte count changed")
    if sha256_file(table) != table_receipt["sha256"]:
        raise RuntimeError("EC benchmark table SHA-256 changed")
    with table.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "split", "sequence", "label_indices_json", *EC_CUTOFF_FIELDS}
    if not rows or not required <= set(rows[0]):
        raise ValueError("EC benchmark table lacks required columns")
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "valid", "test")}
    if counts != {key: int(value) for key, value in receipt["records"].items()}:
        raise RuntimeError("EC benchmark split counts changed")
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("EC benchmark table contains duplicate IDs")
    return rows, receipt


def torchdrug_micro_auprc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Exact flattened AUPRC definition used by TorchDrug's EC task."""

    predictions = logits.flatten()
    labels = targets.flatten().to(torch.long)
    order = predictions.argsort(descending=True)
    labels = labels[order]
    precision = labels.cumsum(0) / torch.arange(
        1, len(labels) + 1, device=labels.device
    )
    score = precision[labels == 1].sum() / ((labels == 1).sum() + 1e-10)
    return float(score)


def torchdrug_f1_max(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Exact sample-averaged F1-max definition used by TorchDrug."""

    target = targets.to(logits)
    order = logits.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target, dtype=torch.bool)
    is_start[:, 0] = True
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = logits.flatten().argsort(descending=True)
    offsets = torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    flattened_order = (order + offsets).flatten()
    inverse_order = torch.zeros_like(flattened_order)
    inverse_order[flattened_order] = torch.arange(
        flattened_order.shape[0], device=order.device
    )
    is_start = is_start.flatten()[all_order]
    all_order = inverse_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    previous_precision = torch.where(
        is_start, torch.zeros_like(precision), precision[all_order - 1]
    )
    all_precision = (precision[all_order] - previous_precision).cumsum(0)
    all_precision = all_precision / is_start.cumsum(0)
    previous_recall = torch.where(
        is_start, torch.zeros_like(recall), recall[all_order - 1]
    )
    all_recall = (recall[all_order] - previous_recall).cumsum(0) / logits.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return float(all_f1.max())


def _dense_targets(rows: Sequence[Dict[str, str]], labels: int) -> torch.Tensor:
    targets = torch.zeros((len(rows), labels), dtype=torch.float32)
    for index, row in enumerate(rows):
        positive = json.loads(row["label_indices_json"])
        if not isinstance(positive, list) or not positive:
            raise ValueError("EC benchmark row has no positive labels")
        if any(not isinstance(value, int) or not 0 <= value < labels for value in positive):
            raise ValueError("EC benchmark row has an invalid label index")
        targets[index, positive] = 1.0
    return targets


def _summary(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": mean(values),
        "sample_standard_deviation": stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def run_frozen_ec_probe(
    embeddings: torch.Tensor,
    rows: Sequence[Dict[str, str]],
    label_count: int,
    device_name: str,
    seeds: Sequence[int] = (1701, 2701, 3701),
    steps: int = 300,
    validation_interval: int = 10,
    learning_rate: float = 1e-2,
    minimum_steps: Optional[int] = None,
    patience_validations: int = 30,
) -> Dict[str, Any]:
    """Fit deterministic full-batch linear probes and evaluate each public test cutoff."""

    if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
        raise ValueError("EC embeddings must align one-to-one with benchmark rows")
    minimum_steps = min(300, steps) if minimum_steps is None else minimum_steps
    if (
        min(label_count, steps, validation_interval, minimum_steps, patience_validations) <= 0
        or minimum_steps > steps
        or learning_rate <= 0
    ):
        raise ValueError("invalid EC probe settings")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("EC probe seeds must be nonempty and unique")
    torch.use_deterministic_algorithms(True)
    device = torch.device(device_name)
    features = embeddings.float().to(device)
    targets = _dense_targets(rows, label_count).to(device)
    train_indices = torch.tensor(
        [index for index, row in enumerate(rows) if row["split"] == "train"],
        device=device,
    )
    valid_indices = torch.tensor(
        [index for index, row in enumerate(rows) if row["split"] == "valid"],
        device=device,
    )
    test_indices = {
        field: torch.tensor(
            [
                index
                for index, row in enumerate(rows)
                if row["split"] == "test" and row[field] == "1"
            ],
            device=device,
        )
        for field in EC_CUTOFF_FIELDS
    }
    if not len(train_indices) or not len(valid_indices) or any(
        not len(indices) for indices in test_indices.values()
    ):
        raise RuntimeError("EC benchmark has an empty probe split or cutoff")

    runs: List[Dict[str, Any]] = []
    for seed in seeds:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        head = nn.Linear(features.shape[1], label_count).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
        best_valid_auprc = -math.inf
        best_step = 0
        best_state = None
        validation_history: List[Dict[str, float | int]] = []
        validations_without_improvement = 0
        actual_steps = 0
        for step in range(1, steps + 1):
            head.train()
            optimizer.zero_grad(set_to_none=True)
            logits = head(features.index_select(0, train_indices))
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits,
                targets.index_select(0, train_indices),
            )
            loss.backward()
            optimizer.step()
            actual_steps = step
            if step % validation_interval == 0 or step == steps:
                head.eval()
                with torch.no_grad():
                    valid_logits = head(features.index_select(0, valid_indices))
                    valid_targets = targets.index_select(0, valid_indices)
                    valid_auprc = torchdrug_micro_auprc(valid_logits, valid_targets)
                validation_history.append(
                    {"step": step, "validation_micro_auprc": valid_auprc}
                )
                if valid_auprc > best_valid_auprc:
                    best_valid_auprc = valid_auprc
                    best_step = step
                    best_state = {
                        key: value.detach().cpu().clone() for key, value in head.state_dict().items()
                    }
                    validations_without_improvement = 0
                else:
                    validations_without_improvement += 1
                if (
                    step >= minimum_steps
                    and validations_without_improvement >= patience_validations
                ):
                    break
        if best_state is None:
            raise RuntimeError("EC probe did not select a validation checkpoint")
        head.load_state_dict(best_state)
        head.eval()
        test_metrics: Dict[str, Dict[str, float | int]] = {}
        with torch.no_grad():
            for field, indices in test_indices.items():
                cutoff_logits = head(features.index_select(0, indices))
                cutoff_targets = targets.index_select(0, indices)
                test_metrics[field] = {
                    "records": len(indices),
                    "micro_auprc": torchdrug_micro_auprc(
                        cutoff_logits, cutoff_targets
                    ),
                    "f1_max": torchdrug_f1_max(cutoff_logits, cutoff_targets),
                }
        runs.append(
            {
                "seed": seed,
                "best_step": best_step,
                "actual_steps": actual_steps,
                "early_stopped": actual_steps < steps,
                "validation_checks_after_selected_checkpoint": (
                    len(validation_history)
                    - 1
                    - next(
                        index
                        for index, item in enumerate(validation_history)
                        if item["step"] == best_step
                    )
                ),
                "best_validation_micro_auprc": best_valid_auprc,
                "validation_history": validation_history,
                "test": test_metrics,
            }
        )

    aggregate: Dict[str, Dict[str, Dict[str, float] | int]] = {}
    for field, indices in test_indices.items():
        aggregate[field] = {
            "records": len(indices),
            "micro_auprc": _summary(
                [float(run["test"][field]["micro_auprc"]) for run in runs]
            ),
            "f1_max": _summary(
                [float(run["test"][field]["f1_max"]) for run in runs]
            ),
        }
    return {
        "probe": {
            "kind": "frozen_encoder_linear",
            "head": "single affine layer",
            "loss": "unweighted binary cross entropy with logits",
            "optimizer": "Adam",
            "training": "full batch",
            "learning_rate": learning_rate,
            "steps": steps,
            "validation_interval": validation_interval,
            "minimum_steps_before_early_stopping": minimum_steps,
            "validation_patience": patience_validations,
            "selection": "maximum validation micro-AUPR independently per seed",
            "seeds": list(seeds),
            "deterministic_algorithms": True,
        },
        "embedding_dimension": int(features.shape[1]),
        "records": {
            "train": len(train_indices),
            "valid": len(valid_indices),
            "test_cutoffs": {field: len(indices) for field, indices in test_indices.items()},
        },
        "runs": runs,
        "convergence": {
            "runs_early_stopped": sum(bool(run["early_stopped"]) for run in runs),
            "runs_selecting_last_evaluated_checkpoint": sum(
                int(run["best_step"]) == int(run["actual_steps"]) for run in runs
            ),
            "minimum_validation_checks_after_selected_checkpoint": min(
                int(run["validation_checks_after_selected_checkpoint"]) for run in runs
            ),
        },
        "aggregate": aggregate,
        "metric_compatibility": (
            "micro-AUPR and F1-max reproduce TorchDrug 0.2.1 definitions from commit "
            "6066fbd82360abb5f270cba1eca560af01b8cc90"
        ),
    }
