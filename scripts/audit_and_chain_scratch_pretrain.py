#!/usr/bin/env python3
"""Fail-closed scratch segment audit and single-successor submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_exclusive_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise RuntimeError(f"{label} mismatch: {observed!r} != {expected!r}")


def require_completed_job(job_id: str) -> None:
    result = subprocess.run(
        [
            "sacct",
            "-j",
            job_id,
            "--format=JobIDRaw,State,ExitCode",
            "-n",
            "-P",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    matches = [line.split("|") for line in result.stdout.splitlines() if line]
    roots = [fields for fields in matches if fields[0] == job_id]
    if len(roots) != 1:
        raise RuntimeError(f"expected one sacct root row for {job_id}, found {len(roots)}")
    require_equal(roots[0][1], "COMPLETED", "completed job state")
    require_equal(roots[0][2], "0:0", "completed job exit code")


def learning_rate(config: Dict[str, Any], metric_step: int, horizon: int) -> float:
    optimizer_step = metric_step - 1
    warmup = int(config["warmup_steps"])
    maximum = float(config["learning_rate"])
    minimum = float(config["min_learning_rate"])
    if optimizer_step < warmup:
        return maximum * float(optimizer_step + 1) / max(1, warmup)
    progress = (optimizer_step - warmup) / max(1, horizon - warmup)
    return maximum + min(1.0, progress) * (minimum - maximum)


def finite_metric_rows(path: Path) -> list[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("metrics file is empty")
    numeric = (
        "step",
        "tokens_seen",
        "loss",
        "grad_norm",
        "learning_rate",
        "tokens_per_second",
        "step_seconds",
        "masked_tokens",
        "tokens",
        "packing_efficiency",
    )
    for row in rows:
        if not all(math.isfinite(float(row[name])) for name in numeric):
            raise RuntimeError("metrics contain a non-finite numeric field")
        if row.get("run_kind") != "pretrain":
            raise RuntimeError("metrics contain a non-pretrain row")
    steps = [int(row["step"]) for row in rows]
    tokens = [int(row["tokens_seen"]) for row in rows]
    if steps[0] != 10 or any(right - left != 10 for left, right in zip(steps, steps[1:])):
        raise RuntimeError("metrics are not exactly 10-step interval-continuous")
    if any(right <= left for left, right in zip(tokens, tokens[1:])):
        raise RuntimeError("metrics tokens are not strictly increasing")
    return rows


def find_completion_receipt(run_dir: Path, segment_start: int, segment_end: int) -> Path:
    candidates = []
    for path in sorted(run_dir.glob("receipt-step-*.json")):
        payload = json.loads(path.read_text())
        step = int(payload.get("step", -1))
        if segment_start < step <= segment_end and payload.get("status") == "passed":
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one passed completion receipt in ({segment_start}, {segment_end}], "
            f"found {len(candidates)}"
        )
    return candidates[0]


def verify_checkpoint(checkpoint: Path, expected: Dict[str, Any]) -> Dict[str, Any]:
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require_equal(manifest["schema_version"], 1, "checkpoint schema")
    require_equal(manifest["step"], expected["step"], "checkpoint step")
    require_equal(manifest["tokens_seen"], expected["tokens"], "checkpoint tokens")
    require_equal(
        manifest["world_size"], expected["world_size"], "checkpoint world size"
    )
    require_equal(manifest["parameter_count"], 7_065_227_297, "checkpoint parameters")
    require_equal(
        manifest["trainable_parameter_count"],
        7_065_227_297,
        "checkpoint trainable parameters",
    )
    require_equal(manifest["initialization"]["type"], "random", "initialization type")
    require_equal(
        manifest["initialization"]["external_checkpoint_loaded"],
        False,
        "external checkpoint",
    )
    for key, value in expected["provenance"].items():
        require_equal(manifest["provenance"][key], value, f"checkpoint {key}")
    declared = manifest["files"]
    actual = {path.name for path in checkpoint.iterdir() if path.is_file()}
    require_equal(actual, set(declared) | {"manifest.json"}, "checkpoint exact file set")
    total_bytes = 0
    for name, record in sorted(declared.items()):
        path = checkpoint / name
        require_equal(path.stat().st_size, record["bytes"], f"checkpoint size {name}")
        require_equal(sha256_file(path), record["sha256"], f"checkpoint hash {name}")
        total_bytes += int(record["bytes"])
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "declared_files_verified": len(declared),
        "declared_bytes_verified": total_bytes,
        "exact_file_set_verified": True,
        "all_declared_sha256_verified": True,
    }


def audit_segment(args: argparse.Namespace, verify_slurm: bool = True) -> Dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    source_dir = Path(args.source_dir).resolve()
    config_path = Path(args.config).resolve()
    audit_sbatch_path = Path(args.audit_sbatch).resolve()
    operations_dir = audit_sbatch_path.parent.parent
    if verify_slurm:
        require_completed_job(args.completed_job_id)
    require_equal(source_dir.name, args.source_tree_sha256, "source directory digest")
    if os.access(source_dir, os.W_OK):
        raise RuntimeError("source snapshot is writable")
    require_equal(sha256_file(config_path), args.train_config_sha256, "train config hash")
    require_equal(
        sha256_file(Path(args.trainer_sbatch).resolve()),
        args.trainer_sbatch_sha256,
        "trainer sbatch hash",
    )
    require_equal(
        sha256_file(audit_sbatch_path),
        args.audit_sbatch_sha256,
        "audit sbatch hash",
    )
    require_equal(
        Path(__file__).resolve().parent.parent,
        operations_dir,
        "audit operations snapshot",
    )
    require_equal(
        sha256_file(Path(__file__).resolve()),
        args.audit_script_sha256,
        "audit script hash",
    )
    config = json.loads(config_path.read_text())
    require_equal(config["max_steps"], args.lr_schedule_end_step, "global LR horizon")
    require_equal(config["target_tokens"], args.target_tokens, "target tokens")
    require_equal(Path(config["output_dir"]).resolve(), run_dir, "config output directory")

    receipt_path = find_completion_receipt(run_dir, args.segment_start, args.segment_end)
    receipt = json.loads(receipt_path.read_text())
    require_equal(receipt["segment_end_step"], args.segment_end, "receipt segment endpoint")
    require_equal(
        receipt["learning_rate_schedule_end_step"],
        args.lr_schedule_end_step,
        "receipt LR horizon",
    )
    require_equal(receipt["target_tokens"], args.target_tokens, "receipt target tokens")
    require_equal(receipt["world_size"], args.expected_world_size, "receipt world size")
    require_equal(receipt["parameter_count"], 7_065_227_297, "receipt parameters")
    for key, value in {
        "source_tree_sha256": args.source_tree_sha256,
        "training_code_sha256": args.training_code_sha256,
        "train_config_sha256": args.train_config_sha256,
        "data_manifest_sha256": args.data_manifest_sha256,
        "model_config_sha256": args.model_config_sha256,
        "tokenizer_sha256": args.tokenizer_sha256,
    }.items():
        require_equal(receipt["provenance"][key], value, f"receipt {key}")

    target_reached = bool(receipt["target_reached"])
    if target_reached:
        require_equal(receipt["evidence_level"], "SCRATCH_PRETRAIN", "completion evidence")
        require_equal(receipt["stopped_by"], "token_budget", "completion stop reason")
    else:
        require_equal(
            receipt["evidence_level"],
            "SCRATCH_PRETRAIN_SEGMENT",
            "segment evidence",
        )
        require_equal(receipt["step"], args.segment_end, "segment completion step")
        require_equal(receipt["stopped_by"], "segment_checkpoint", "segment stop reason")

    step = int(receipt["step"])
    tokens = int(receipt["tokens"])
    checkpoint = run_dir / "checkpoints" / f"step-{step:08d}"
    expected_provenance = {
        "source_tree_sha256": args.source_tree_sha256,
        "training_code_sha256": args.training_code_sha256,
        "train_config_sha256": args.train_config_sha256,
        "data_manifest_sha256": args.data_manifest_sha256,
        "model_config_sha256": args.model_config_sha256,
        "tokenizer_sha256": args.tokenizer_sha256,
    }
    checkpoint_audit = verify_checkpoint(
        checkpoint,
        {
            "step": step,
            "tokens": tokens,
            "world_size": args.expected_world_size,
            "provenance": expected_provenance,
        },
    )

    metrics_path = run_dir / "metrics.jsonl"
    rows = finite_metric_rows(metrics_path)
    expected_last_logged = step - (step % 10)
    require_equal(int(rows[-1]["step"]), expected_last_logged, "last logged metrics step")
    if step % 10 == 0:
        require_equal(int(rows[-1]["tokens_seen"]), tokens, "metrics final tokens")
    for row in rows:
        expected_lr = learning_rate(config, int(row["step"]), args.lr_schedule_end_step)
        if not math.isclose(
            float(row["learning_rate"]), expected_lr, rel_tol=0.0, abs_tol=1e-15
        ):
            raise RuntimeError(f"learning rate mismatch at step {row['step']}")

    return {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": "SCRATCH_SEGMENT_AUDIT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_job_id": args.completed_job_id,
        "segment_start_step": args.segment_start,
        "declared_segment_end_step": args.segment_end,
        "completed_step": step,
        "tokens": tokens,
        "target_tokens": args.target_tokens,
        "target_reached": target_reached,
        "learning_rate_schedule_end_step": args.lr_schedule_end_step,
        "world_size": args.expected_world_size,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_audit": checkpoint_audit,
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "metrics_rows": len(rows),
        "metrics_all_finite": True,
        "metrics_interval_continuous": True,
        "metrics_tokens_strictly_increasing": True,
        "learning_rates_exact": True,
        "audit_script_sha256": args.audit_script_sha256,
        "audit_sbatch_sha256": args.audit_sbatch_sha256,
        "chain_policy_sha256": args.chain_policy_sha256,
        "provenance": expected_provenance,
        "test_data_accessed": False,
        "proteingym_access_permitted": False,
    }


def run_sbatch(command: Iterable[str]) -> str:
    result = subprocess.run(list(command), check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"sbatch returned invalid job id: {result.stdout!r}")
    return job_id


def submit_successor(args: argparse.Namespace, audit: Dict[str, Any]) -> Dict[str, Any]:
    if audit["target_reached"]:
        return {"target_reached": True, "successor_submitted": False}
    completed_step = int(audit["completed_step"])
    next_end = min(completed_step + args.segment_size, args.lr_schedule_end_step)
    checkpoint = audit["checkpoint_path"]
    operations_dir = Path(args.audit_sbatch).resolve().parent.parent
    receipts_dir = Path(args.archive_receipts_dir).resolve()
    intent_path = receipts_dir / (
        f"{args.chain_id}-chain-after-step{completed_step:08d}-intent.json"
    )
    intent = {
        "schema_version": 1,
        "status": "intent_locked",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_receipt_sha256": args.current_audit_receipt_sha256,
        "chain_policy_sha256": args.chain_policy_sha256,
        "completed_step": completed_step,
        "next_segment_end_step": next_end,
        "resume_checkpoint": checkpoint,
    }
    atomic_exclusive_json(intent_path, intent)

    export = ",".join(
        [
            "ALL",
            f"SOURCE_DIR={args.source_dir}",
            f"CONFIG={args.config}",
            f"SEGMENT_END_STEP={next_end}",
            f"RESUME={checkpoint}",
        ]
    )
    training_job = run_sbatch(
        ["sbatch", "--parsable", f"--export={export}", args.trainer_sbatch]
    )
    training_submission = {
        "schema_version": 1,
        "status": "submitted",
        "evidence_level": "SCRATCH_CONTINUATION_SUBMISSION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": training_job,
        "segment_start_step": completed_step,
        "segment_end_step": next_end,
        "resume_checkpoint": checkpoint,
        "source_tree_sha256": args.source_tree_sha256,
        "training_code_sha256": args.training_code_sha256,
        "train_config_sha256": args.train_config_sha256,
        "learning_rate_schedule_end_step": args.lr_schedule_end_step,
        "audit_receipt_sha256": args.current_audit_receipt_sha256,
        "chain_policy_sha256": args.chain_policy_sha256,
    }
    training_submission_path = (
        receipts_dir / f"{args.chain_id}-step{next_end:08d}-submission.json"
    )
    atomic_exclusive_json(training_submission_path, training_submission)

    audit_export = ",".join(
        [
            "ALL",
            f"OPS_DIR={operations_dir}",
            f"CHAIN_ID={args.chain_id}",
            f"TRAINER_SOURCE={args.source_dir}",
            f"CONFIG={args.config}",
            f"RUN_DIR={args.run_dir}",
            f"TRAINER_SBATCH={args.trainer_sbatch}",
            f"EXPECTED_WORLD_SIZE={args.expected_world_size}",
            f"SOURCE_TREE_SHA256={args.source_tree_sha256}",
            f"TRAINING_CODE_SHA256={args.training_code_sha256}",
            f"TRAIN_CONFIG_SHA256={args.train_config_sha256}",
            f"DATA_MANIFEST_SHA256={args.data_manifest_sha256}",
            f"MODEL_CONFIG_SHA256={args.model_config_sha256}",
            f"TOKENIZER_SHA256={args.tokenizer_sha256}",
            f"COMPLETED_JOB_ID={training_job}",
            f"SEGMENT_START={completed_step}",
            f"SEGMENT_END={next_end}",
            f"TRAINER_SBATCH_SHA256={args.trainer_sbatch_sha256}",
            f"AUDIT_SBATCH_SHA256={args.audit_sbatch_sha256}",
            f"AUDIT_SCRIPT_SHA256={args.audit_script_sha256}",
            f"CHAIN_POLICY_SHA256={args.chain_policy_sha256}",
        ]
    )
    audit_job = run_sbatch(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterok:{training_job}",
            f"--export={audit_export}",
            args.audit_sbatch,
        ]
    )
    audit_submission = {
        "schema_version": 1,
        "status": "submitted",
        "evidence_level": "SCRATCH_SEGMENT_AUDIT_SUBMISSION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": audit_job,
        "dependency": f"afterok:{training_job}",
        "segment_start_step": completed_step,
        "segment_end_step": next_end,
        "audit_sbatch_sha256": args.audit_sbatch_sha256,
        "audit_script_sha256": args.audit_script_sha256,
        "chain_policy_sha256": args.chain_policy_sha256,
    }
    audit_submission_path = (
        receipts_dir / f"{args.chain_id}-step{next_end:08d}-audit-submission.json"
    )
    atomic_exclusive_json(audit_submission_path, audit_submission)
    return {
        "target_reached": False,
        "successor_submitted": True,
        "training_job_id": training_job,
        "audit_job_id": audit_job,
        "next_segment_end_step": next_end,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    for name in (
        "run-dir",
        "source-dir",
        "config",
        "completed-job-id",
        "source-tree-sha256",
        "training-code-sha256",
        "train-config-sha256",
        "data-manifest-sha256",
        "model-config-sha256",
        "tokenizer-sha256",
        "trainer-sbatch",
        "trainer-sbatch-sha256",
        "audit-sbatch",
        "audit-sbatch-sha256",
        "audit-script-sha256",
        "archive-receipts-dir",
        "current-audit-receipt-sha256",
        "chain-policy-sha256",
        "chain-id",
    ):
        result.add_argument(f"--{name}", required=True)
    for name in (
        "segment-start",
        "segment-end",
        "segment-size",
        "target-tokens",
        "lr-schedule-end-step",
        "expected-world-size",
    ):
        result.add_argument(f"--{name}", required=True, type=int)
    result.add_argument("--audit-only", action="store_true")
    result.add_argument("--skip-slurm-check", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    audit = audit_segment(args, verify_slurm=not args.skip_slurm_check)
    audit_path = Path(args.archive_receipts_dir) / (
        f"{args.chain_id}-step{audit['completed_step']:08d}-audit.json"
    )
    atomic_exclusive_json(audit_path, audit)
    args.current_audit_receipt_sha256 = sha256_file(audit_path)
    if not args.audit_only:
        chain = submit_successor(args, audit)
        print(json.dumps({"audit": audit, "chain": chain}, indent=2, sort_keys=True))
    else:
        print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
