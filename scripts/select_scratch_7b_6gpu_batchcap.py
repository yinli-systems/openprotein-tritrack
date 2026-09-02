#!/usr/bin/env python3
"""Fail-closed selector for the focused six-GPU batch-cap benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def require_equal(observed: Any, expected: Any, message: str) -> None:
    if observed != expected:
        raise RuntimeError(f"{message}: observed={observed!r} expected={expected!r}")


def evaluate_candidate(
    source_dir: Path,
    candidate: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    config_path = source_dir / candidate["config"]
    require_equal(sha256_file(config_path), candidate["sha256"], "candidate config hash")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require_equal(
        int(config["max_micro_batch_size"]),
        int(candidate["max_micro_batch_size"]),
        "candidate max micro-batch size",
    )
    require_equal(
        int(config["micro_batch_token_budget"])
        * int(config["gradient_accumulation_steps"])
        * int(gate["world_size"]),
        int(candidate["theoretical_global_token_capacity"]),
        "candidate global token capacity",
    )
    output_dir = Path(config["output_dir"])
    receipt_path = output_dir / f"receipt-step-{int(gate['step']):08d}.json"
    metrics_path = output_dir / "metrics.jsonl"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require_equal(receipt.get("status"), gate["receipt_status"], "receipt status")
    require_equal(receipt.get("evidence_level"), gate["evidence_level"], "evidence level")
    require_equal(int(receipt.get("step", -1)), int(gate["step"]), "receipt step")
    require_equal(int(receipt.get("world_size", -1)), int(gate["world_size"]), "world size")
    require_equal(
        int(receipt.get("parameter_count", -1)),
        int(gate["parameter_count"]),
        "parameter count",
    )
    require_equal(receipt.get("gpu_name"), gate["gpu_name"], "GPU model")
    require_equal(
        receipt.get("provenance", {}).get("train_config_sha256"),
        candidate["sha256"],
        "receipt config hash",
    )
    require_equal(
        receipt.get("provenance", {}).get("source_tree_sha256"),
        gate["source_snapshot_sha256"],
        "receipt source hash",
    )
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    require_equal(len(rows), int(gate["metrics_rows"]), "metrics row count")
    require_equal(
        [int(row.get("step", -1)) for row in rows],
        list(range(1, int(gate["step"]) + 1)),
        "continuous metric steps",
    )
    numeric_fields = (
        "loss",
        "grad_norm",
        "tokens",
        "padded_tokens",
        "packing_efficiency",
        "tokens_per_second",
        "step_seconds",
        "peak_memory_mib",
    )
    if any(
        not isinstance(row.get(field), (int, float)) or not math.isfinite(float(row[field]))
        for row in rows
        for field in numeric_fields
    ):
        raise RuntimeError("candidate contains non-finite or missing metrics")
    first, last = [int(value) for value in gate["selection_step_range"]]
    selected_rows = [row for row in rows if first <= int(row["step"]) <= last]
    require_equal(len(selected_rows), last - first + 1, "selection row count")
    throughput = [float(row["tokens_per_second"]) for row in selected_rows]
    median_tps = statistics.median(throughput)
    p10_tps = nearest_rank_percentile(throughput, 0.10)
    mean_packing = statistics.mean(
        float(row["packing_efficiency"]) for row in selected_rows
    )
    peak_memory = max(float(row["peak_memory_mib"]) for row in rows)
    stability_ratio = p10_tps / median_tps
    passed = (
        mean_packing >= float(gate["minimum_mean_packing_efficiency"])
        and peak_memory <= float(gate["maximum_peak_memory_mib"])
        and stability_ratio >= float(gate["minimum_p10_to_median_ratio"])
    )
    return {
        "config": candidate["config"],
        "config_sha256": candidate["sha256"],
        "max_micro_batch_size": int(candidate["max_micro_batch_size"]),
        "mean_packing_efficiency": mean_packing,
        "median_tokens_per_second": median_tps,
        "p10_tokens_per_second": p10_tps,
        "p10_to_median_ratio": stability_ratio,
        "peak_memory_mib": peak_memory,
        "status": "passed" if passed else "failed_gate",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    policy_path = Path(args.policy).resolve()
    output_path = Path(args.output).resolve()
    require_equal(sha256_file(policy_path), args.policy_sha256, "selection policy hash")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    require_equal(policy.get("status"), "predeclared", "policy status")
    require_equal(source_dir.name, policy["source_snapshot_sha256"], "source snapshot")
    if os.access(source_dir, os.W_OK):
        raise RuntimeError("source snapshot is writable")
    if os.access(policy_path, os.W_OK):
        raise RuntimeError("selection policy is writable")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite selection output: {output_path}")

    results = []
    for candidate in policy["candidates"]:
        try:
            results.append(evaluate_candidate(source_dir, candidate, policy["candidate_gate"]))
        except Exception as exc:
            results.append(
                {
                    "config": candidate["config"],
                    "config_sha256": candidate["sha256"],
                    "max_micro_batch_size": int(candidate["max_micro_batch_size"]),
                    "status": "failed_evidence",
                    "error": str(exc),
                }
            )
    passing = [result for result in results if result["status"] == "passed"]
    passing.sort(
        key=lambda result: (
            -float(result["median_tokens_per_second"]),
            -float(result["p10_tokens_per_second"]),
            float(result["peak_memory_mib"]),
        )
    )
    selected = passing[0] if passing else None
    receipt = {
        "schema_version": 1,
        "status": "selected" if selected is not None else "no_selection",
        "evidence_level": "SCRATCH_6GPU_BATCHCAP_SELECTION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "selection_policy_path": str(policy_path),
        "selection_policy_sha256": args.policy_sha256,
        "source_snapshot_sha256": source_dir.name,
        "candidate_results": results,
        "selected": selected,
        "claim_boundary": policy["claim_boundary"],
        "test_data_accessed": False,
        "proteingym_accessed": False,
    }
    atomic_json(output_path, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    if selected is None:
        raise SystemExit("no candidate passed the predeclared selection gates")


if __name__ == "__main__":
    main()
