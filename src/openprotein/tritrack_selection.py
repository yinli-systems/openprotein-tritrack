"""Fail-closed throughput selection for TriTrack batch-cap probes."""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openprotein.provenance import sha256_file, source_tree_sha256
from openprotein.training import TrainConfig


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _read_status(path: Path) -> dict[str, int]:
    statuses: dict[str, int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) != 2 or not fields[0]:
            raise ValueError(f"invalid probe status at line {line_number}")
        if fields[0] in statuses:
            raise RuntimeError(f"duplicate probe status: {fields[0]}")
        statuses[fields[0]] = int(fields[1])
    return statuses


def _read_metrics(path: Path, required_steps: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if [int(row["step"]) for row in rows] != list(range(1, required_steps + 1)):
        raise RuntimeError("probe metrics do not contain the exact required step sequence")
    for row in rows:
        for field in ("loss", "grad_norm", "tokens_per_second", "peak_memory_mib"):
            value = float(row[field])
            if not math.isfinite(value):
                raise RuntimeError(f"probe metric {field} is non-finite")
        if float(row["tokens_per_second"]) <= 0.0 or int(row["tokens"]) <= 0:
            raise RuntimeError("probe throughput must be positive")
    return rows


def _validate_formal_pair(probe: TrainConfig, formal: TrainConfig, target_tokens: int) -> None:
    fields = (
        "task",
        "model_config",
        "model_artifact_manifest",
        "tokenizer_json",
        "text_tokenizer_model",
        "train_glob",
        "data_manifest",
        "structure_database",
        "structure_manifest",
        "replay_train_glob",
        "replay_data_manifest",
        "replay_every",
        "replay_loss_weight",
        "sequence_length",
        "text_length",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "mlm_probability",
        "structure_mask_probability",
        "structure_dropout_probability",
        "attention_policy",
        "geometry_pairs",
        "min_structure_confidence",
        "detach_geometry_backbone",
        "loss_weights",
        "learning_rate",
        "min_learning_rate",
        "weight_decay",
        "max_grad_norm",
        "seed",
        "precision",
    )
    for field in fields:
        if getattr(probe, field) != getattr(formal, field):
            raise RuntimeError(f"probe/formal config mismatch: {field}")
    if formal.max_tokens != target_tokens or formal.warmup_tokens is None:
        raise RuntimeError("formal config has the wrong token contract")


def select_tritrack_batchcap(
    policy_path: str,
    status_path: str,
    output_path: str,
) -> dict[str, Any]:
    """Select maximum robust throughput subject to a physical-memory headroom gate."""

    policy_file = Path(policy_path).resolve()
    status_file = Path(status_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite selection receipt: {output}")
    policy = json.loads(policy_file.read_text(encoding="utf-8"))
    if int(policy.get("schema_version", 0)) != 1:
        raise RuntimeError("unsupported TriTrack selection policy schema")
    project_root = policy_file.parents[2]
    physical_memory = float(policy["physical_gpu_memory_mib"])
    minimum_headroom = float(policy["minimum_headroom_fraction"])
    required_steps = int(policy["required_steps"])
    discard_steps = int(policy["discard_initial_steps"])
    target_tokens = int(policy["formal_target_tokens"])
    world_size = int(policy["world_size"])
    if not 0.0 < minimum_headroom < 1.0:
        raise ValueError("minimum headroom must be in (0, 1)")
    if not 0 <= discard_steps < required_steps:
        raise ValueError("discard step count must be smaller than required steps")
    if world_size <= 1:
        raise ValueError("TriTrack efficiency selection requires distributed probes")
    statuses = _read_status(status_file)
    expected = {str(candidate["probe_config"]) for candidate in policy["candidates"]}
    if set(statuses) != expected:
        raise RuntimeError("probe status set differs from the frozen policy")

    candidates = []
    for specification in policy["candidates"]:
        probe_relative = str(specification["probe_config"])
        formal_relative = str(specification["formal_config"])
        probe_path = project_root / probe_relative
        formal_path = project_root / formal_relative
        probe = TrainConfig.from_json(str(probe_path))
        formal = TrainConfig.from_json(str(formal_path))
        _validate_formal_pair(probe, formal, target_tokens)
        result: dict[str, Any] = {
            "probe_config": probe_relative,
            "probe_config_sha256": sha256_file(probe_path),
            "formal_config": formal_relative,
            "formal_config_sha256": sha256_file(formal_path),
            "micro_batch_size_per_rank": probe.micro_batch_size,
            "gradient_accumulation_steps": probe.gradient_accumulation_steps,
            "exit_code": statuses[probe_relative],
            "admissible": False,
        }
        if statuses[probe_relative] != 0:
            result["rejected_reason"] = "probe process exited non-zero"
            candidates.append(result)
            continue
        metrics_path = Path(probe.output_dir) / "metrics.jsonl"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"successful probe has no metrics: {metrics_path}")
        rows = _read_metrics(metrics_path, required_steps)
        measured = rows[discard_steps:]
        peak_memory = max(float(row["peak_memory_mib"]) for row in rows)
        headroom = (physical_memory - peak_memory) / physical_memory
        throughput = [float(row["tokens_per_second"]) for row in measured]
        result.update(
            {
                "metrics_path": str(metrics_path),
                "metrics_sha256": sha256_file(metrics_path),
                "measured_steps": len(measured),
                "peak_memory_mib": peak_memory,
                "headroom_fraction": headroom,
                "median_tokens_per_second": statistics.median(throughput),
                "mean_tokens_per_second": statistics.fmean(throughput),
                "minimum_tokens_per_second": min(throughput),
                "maximum_tokens_per_second": max(throughput),
            }
        )
        if headroom < minimum_headroom:
            result["rejected_reason"] = "physical-memory headroom below policy minimum"
        else:
            result["admissible"] = True
        candidates.append(result)

    admissible = [candidate for candidate in candidates if candidate["admissible"]]
    if not admissible:
        raise RuntimeError("no TriTrack batch-cap candidate passed the frozen gates")
    selected = max(admissible, key=lambda candidate: candidate["median_tokens_per_second"])
    selected["selected"] = True
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "maximize median aggregate tokens/s after discarded startup steps, subject to "
            "successful exact-step completion, finite metrics, and memory headroom"
        ),
        "policy_path": str(policy_file),
        "policy_sha256": sha256_file(policy_file),
        "status_path": str(status_file),
        "status_sha256": sha256_file(status_file),
        "source_tree_sha256": source_tree_sha256(project_root),
        "physical_gpu_memory_mib": physical_memory,
        "minimum_headroom_fraction": minimum_headroom,
        "required_steps": required_steps,
        "discard_initial_steps": discard_steps,
        "formal_target_tokens": target_tokens,
        "world_size": world_size,
        "candidates": candidates,
        "selected_probe_config": selected["probe_config"],
        "selected_formal_config": selected["formal_config"],
    }
    _atomic_json(output, receipt)
    return receipt
