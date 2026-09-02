import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "scratch_chain", ROOT / "scripts/audit_and_chain_scratch_pretrain.py"
)
assert SPEC and SPEC.loader
CHAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHAIN)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    source = tmp_path / "sourcehash"
    (source / "scripts").mkdir(parents=True)
    trainer = source / "scripts/run_scratch_pretrain.sbatch"
    trainer.write_text("#!/bin/bash\n")
    run = tmp_path / "run"
    checkpoint = run / "checkpoints/step-00000020"
    checkpoint.mkdir(parents=True)
    shard = checkpoint / "shard.bin"
    shard.write_bytes(b"checkpoint")
    config = source / "config.json"
    config.write_text(
        json.dumps(
            {
                "max_steps": 100,
                "target_tokens": 1000,
                "output_dir": str(run),
                "warmup_steps": 10,
                "learning_rate": 0.001,
                "min_learning_rate": 0.0001,
            }
        )
    )
    provenance = {
        "source_tree_sha256": source.name,
        "training_code_sha256": "train-code",
        "train_config_sha256": digest(config),
        "data_manifest_sha256": "data",
        "model_config_sha256": "model",
        "tokenizer_sha256": "tokenizer",
    }
    manifest = {
        "schema_version": 1,
        "step": 20,
        "tokens_seen": 200,
        "world_size": 8,
        "parameter_count": 7_065_227_297,
        "trainable_parameter_count": 7_065_227_297,
        "initialization": {"type": "random", "external_checkpoint_loaded": False},
        "provenance": provenance,
        "files": {"shard.bin": {"bytes": 10, "sha256": digest(shard)}},
    }
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    receipt = {
        "status": "passed",
        "step": 20,
        "tokens": 200,
        "target_tokens": 1000,
        "target_reached": False,
        "segment_end_step": 20,
        "learning_rate_schedule_end_step": 100,
        "world_size": 8,
        "parameter_count": 7_065_227_297,
        "evidence_level": "SCRATCH_PRETRAIN_SEGMENT",
        "stopped_by": "segment_checkpoint",
        "provenance": provenance,
    }
    (run / "receipt-step-00000020.json").write_text(json.dumps(receipt))
    rows = []
    for step, tokens in ((10, 100), (20, 200)):
        rows.append(
            {
                "step": step,
                "tokens_seen": tokens,
                "loss": 2.0,
                "grad_norm": 1.0,
                "learning_rate": CHAIN.learning_rate(json.loads(config.read_text()), step, 100),
                "tokens_per_second": 10.0,
                "step_seconds": 1.0,
                "masked_tokens": 1,
                "tokens": 100,
                "packing_efficiency": 1.0,
                "run_kind": "pretrain",
            }
        )
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    args = argparse.Namespace(
        run_dir=str(run),
        source_dir=str(source),
        config=str(config),
        completed_job_id="1",
        segment_start=0,
        segment_end=20,
        segment_size=20,
        target_tokens=1000,
        lr_schedule_end_step=100,
        expected_world_size=8,
        source_tree_sha256=source.name,
        training_code_sha256="train-code",
        train_config_sha256=digest(config),
        data_manifest_sha256="data",
        model_config_sha256="model",
        tokenizer_sha256="tokenizer",
        trainer_sbatch=str(trainer),
        trainer_sbatch_sha256=digest(trainer),
        audit_sbatch=str(ROOT / "scripts/audit_and_chain_scratch_pretrain.sbatch"),
        audit_sbatch_sha256=digest(
            ROOT / "scripts/audit_and_chain_scratch_pretrain.sbatch"
        ),
        audit_script_sha256=digest(
            ROOT / "scripts/audit_and_chain_scratch_pretrain.py"
        ),
        archive_receipts_dir=str(tmp_path / "receipts"),
        current_audit_receipt_sha256="audit-receipt",
        chain_policy_sha256="policy",
        chain_id="fixture-chain",
    )
    return args, run


def test_segment_audit_verifies_checkpoint_metrics_lr_and_provenance(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    monkeypatch.setattr(CHAIN.os, "access", lambda *_: False)
    audit = CHAIN.audit_segment(args, verify_slurm=False)
    assert audit["status"] == "passed"
    assert audit["learning_rates_exact"] is True
    assert audit["checkpoint_audit"]["all_declared_sha256_verified"] is True


def test_segment_audit_rejects_segment_dependent_learning_rate(tmp_path, monkeypatch):
    args, run = fixture(tmp_path)
    monkeypatch.setattr(CHAIN.os, "access", lambda *_: False)
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    rows[-1]["learning_rate"] = CHAIN.learning_rate(
        json.loads(Path(args.config).read_text()), 20, 20
    )
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(RuntimeError, match="learning rate mismatch"):
        CHAIN.audit_segment(args, verify_slurm=False)


def test_successor_propagates_immutable_operations_snapshot_and_hashes(
    tmp_path, monkeypatch
):
    args, _ = fixture(tmp_path)
    commands = []

    def fake_sbatch(command):
        commands.append(list(command))
        return str(100 + len(commands))

    monkeypatch.setattr(CHAIN, "run_sbatch", fake_sbatch)
    result = CHAIN.submit_successor(
        args,
        {
            "target_reached": False,
            "completed_step": 20,
            "checkpoint_path": str(tmp_path / "checkpoint"),
        },
    )

    assert result["training_job_id"] == "101"
    assert result["audit_job_id"] == "102"
    audit_export = next(part for part in commands[1] if part.startswith("--export="))
    expected_ops_dir = (ROOT / "scripts/audit_and_chain_scratch_pretrain.sbatch").parent.parent
    assert f"OPS_DIR={expected_ops_dir}" in audit_export
    assert f"TRAINER_SBATCH_SHA256={args.trainer_sbatch_sha256}" in audit_export
    assert f"AUDIT_SBATCH_SHA256={args.audit_sbatch_sha256}" in audit_export
    assert f"AUDIT_SCRIPT_SHA256={args.audit_script_sha256}" in audit_export
    assert f"CHAIN_POLICY_SHA256={args.chain_policy_sha256}" in audit_export


def test_audit_sbatch_requires_explicit_operations_snapshot():
    script = (ROOT / "scripts/audit_and_chain_scratch_pretrain.sbatch").read_text()
    assert 'ops_dir="${OPS_DIR:?' in script
    assert "BASH_SOURCE" not in script
