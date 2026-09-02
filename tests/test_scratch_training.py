import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from openprotein.scratch.training import (
    ScratchTrainConfig,
    _completion_fields,
    _learning_rate,
    _learning_rate_schedule_end_step,
    _mlm_loss_scale,
    _prune_checkpoints,
    _resolve_target_steps,
    _topology_migration_required,
    _validate_resume_metrics,
    run_scratch_training,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_config(tmp_path, output_name="run"):
    raw = json.loads((ROOT / "configs/scratch/train/tiny-recovery.json").read_text())
    raw.update(
        {
            "model_config": str(
                ROOT / "configs/scratch/model/openprotein-scratch-tiny.json"
            ),
            "tokenizer_json": str(ROOT / "configs/scratch/tokenizer/esm2-33.json"),
            "train_glob": str(ROOT / "tests/fixtures/train/*.csv"),
            "output_dir": str(tmp_path / output_name),
        }
    )
    path = tmp_path / f"{output_name}.json"
    path.write_text(json.dumps(raw))
    return path


def test_scratch_checkpoint_resume_matches_uninterrupted(tmp_path):
    config = _write_config(tmp_path)
    first = run_scratch_training(str(config), max_steps_override=2, device_name="cpu")
    checkpoint = tmp_path / "run/checkpoints/step-00000002"
    assert first["evidence_level"] == "SCALE_TEST"
    assert first["rank"] == 0
    assert first["initialization"]["external_checkpoint_loaded"] is False
    assert first["loss_normalization"] == "global_masked_tokens"
    assert checkpoint.is_dir()

    resumed = run_scratch_training(
        str(config), resume=str(checkpoint), max_steps_override=4, device_name="cpu"
    )
    assert resumed["step"] == 4
    assert resumed["resumed_from"]["step"] == 2

    uninterrupted_config = _write_config(tmp_path, "uninterrupted")
    run_scratch_training(str(uninterrupted_config), device_name="cpu")
    resumed_rank = torch.load(
        tmp_path / "run/checkpoints/step-00000004/rank-00000.pt",
        map_location="cpu",
        weights_only=False,
    )
    uninterrupted_rank = torch.load(
        tmp_path / "uninterrupted/checkpoints/step-00000004/rank-00000.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert resumed_rank["tokens_seen"] == uninterrupted_rank["tokens_seen"]
    assert resumed_rank["batcher"]["records_consumed"] == uninterrupted_rank["batcher"][
        "records_consumed"
    ]


def test_formal_pretrain_contract_rejects_small_or_unbound_run(tmp_path):
    raw = json.loads((ROOT / "configs/scratch/train/tiny-recovery.json").read_text())
    raw["run_kind"] = "pretrain"
    raw["target_tokens"] = 1_000_000_000
    raw["data_manifest"] = None
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="at least"):
        ScratchTrainConfig.from_json(str(path))


def test_topology_migration_config_is_hash_bound_and_probe_is_checkpoint_free():
    base = ScratchTrainConfig.from_json(
        str(ROOT / "configs/scratch/train/tiny-recovery.json")
    )
    migration = {
        "from_world_size": 1,
        "to_world_size": 2,
        "from_step": 2,
        "from_tokens": 100,
        "checkpoint_manifest_sha256": "a" * 64,
        "source_tree_sha256": "b" * 64,
        "training_code_sha256": "c" * 64,
        "train_config_sha256": "d" * 64,
    }
    probe = replace(
        base,
        fsdp_shard_size=2,
        checkpoint_at_start=False,
        checkpoint_at_end=False,
        topology_migration=migration,
    )
    probe.validate()

    with pytest.raises(ValueError, match="checkpoint-free"):
        replace(probe, checkpoint_at_end=True).validate()
    with pytest.raises(ValueError, match="must equal fsdp_shard_size"):
        replace(probe, fsdp_shard_size=1).validate()
    with pytest.raises(ValueError, match="must be SHA-256"):
        replace(
            probe,
            topology_migration={**migration, "checkpoint_manifest_sha256": "bad"},
        ).validate()

    assert _topology_migration_required({"world_size": 1}, probe, 2) is True
    assert _topology_migration_required({"world_size": 2}, probe, 2) is False
    with pytest.raises(RuntimeError, match="runtime world size"):
        _topology_migration_required({"world_size": 1}, probe, 3)
    with pytest.raises(RuntimeError, match="outside the frozen topology lineage"):
        _topology_migration_required({"world_size": 4}, probe, 2)


def test_formal_pretrain_segments_are_checkpoint_bound_and_not_completion_claims():
    base = ScratchTrainConfig.from_json(
        str(ROOT / "configs/scratch/train/tiny-recovery.json")
    )
    config = replace(
        base,
        run_kind="pretrain",
        target_tokens=50_000_000_000,
        data_format="packed_v1",
        data_manifest="manifest.json",
        max_steps=100,
        checkpoint_every=10,
        checkpoint_at_end=True,
    )
    config.validate()

    assert _resolve_target_steps(config, 0, None, 10) == 10
    with pytest.raises(ValueError, match="checkpoint boundary"):
        _resolve_target_steps(config, 0, None, 11)
    with pytest.raises(ValueError, match="forbids max_steps_override"):
        _resolve_target_steps(config, 0, 10, None)
    with pytest.raises(ValueError, match="follow the current checkpoint"):
        _resolve_target_steps(config, 10, None, 10)

    completed, evidence, stopped_by = _completion_fields(
        config, False, 10, 10, 10
    )
    assert completed is True
    assert evidence == "SCRATCH_PRETRAIN_SEGMENT"
    assert stopped_by == "segment_checkpoint"

    completed, evidence, stopped_by = _completion_fields(
        config, True, 9, 10, 10
    )
    assert completed is True
    assert evidence == "SCRATCH_PRETRAIN"
    assert stopped_by == "token_budget"


def test_formal_pretrain_segments_do_not_restart_learning_rate_schedule():
    base = ScratchTrainConfig.from_json(
        str(ROOT / "configs/scratch/train/tiny-recovery.json")
    )
    config = replace(
        base,
        run_kind="pretrain",
        target_tokens=50_000_000_000,
        data_format="packed_v1",
        data_manifest="manifest.json",
        max_steps=100,
        warmup_steps=10,
        checkpoint_every=10,
        checkpoint_at_end=True,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
    )
    config.validate()

    first_segment_target = _resolve_target_steps(config, 0, None, 20)
    second_segment_target = _resolve_target_steps(config, 20, None, 40)
    first_horizon = _learning_rate_schedule_end_step(config, first_segment_target)
    second_horizon = _learning_rate_schedule_end_step(config, second_segment_target)

    assert first_horizon == second_horizon == config.max_steps
    assert _learning_rate(config, 20, first_horizon) == _learning_rate(
        config, 20, second_horizon
    )
    assert _learning_rate(config, 20, first_horizon) == pytest.approx(9e-4)

    scale_config = replace(config, run_kind="scale_test", target_tokens=None)
    assert _learning_rate_schedule_end_step(scale_config, 20) == 20


def test_resume_metrics_must_not_extend_beyond_checkpoint(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            json.dumps({"step": step, "tokens_seen": step * 100})
            for step in (1, 2, 3)
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="extend beyond"):
        _validate_resume_metrics(metrics, resumed_step=2, resumed_tokens=200)

    metrics.write_text(json.dumps({"step": 2, "tokens_seen": 201}) + "\n")
    with pytest.raises(RuntimeError, match="disagree"):
        _validate_resume_metrics(metrics, resumed_step=2, resumed_tokens=200)

    metrics.write_text(json.dumps({"step": 2, "tokens_seen": 200}) + "\n")
    _validate_resume_metrics(metrics, resumed_step=2, resumed_tokens=200)


def test_checkpoint_retention_keeps_only_latest_restart(tmp_path):
    config = _write_config(tmp_path, "retained")
    raw = json.loads(config.read_text())
    raw["checkpoint_keep_last"] = 1
    config.write_text(json.dumps(raw))

    receipt = run_scratch_training(str(config), device_name="cpu")

    checkpoint_root = tmp_path / "retained/checkpoints"
    assert receipt["step"] == 4
    assert not (checkpoint_root / "step-00000000").exists()
    assert not (checkpoint_root / "step-00000002").exists()
    assert (checkpoint_root / "step-00000004").is_dir()


def test_checkpoint_retention_supports_symlinked_checkpoint_root(tmp_path):
    output_dir = tmp_path / "run"
    target = tmp_path / "hdd-checkpoints"
    target.mkdir()
    for step in (2500, 5000):
        checkpoint = target / f"step-{step:08d}"
        checkpoint.mkdir()
        (checkpoint / "manifest.json").write_text(json.dumps({"step": step}))

    output_dir.mkdir()
    (output_dir / "checkpoints").symlink_to(target, target_is_directory=True)

    pruned = _prune_checkpoints(output_dir, keep_last=1)

    assert [path.name for path in pruned] == ["step-00002500"]
    assert not (target / "step-00002500").exists()
    assert (target / "step-00005000").is_dir()


def test_scale_test_can_skip_large_checkpoints_but_pretrain_cannot(tmp_path):
    config = _write_config(tmp_path, "throughput-only")
    raw = json.loads(config.read_text())
    raw.update(
        {
            "checkpoint_at_start": False,
            "checkpoint_at_end": False,
            "checkpoint_every": 100,
        }
    )
    config.write_text(json.dumps(raw))

    receipt = run_scratch_training(str(config), device_name="cpu")
    assert receipt["status"] == "passed"
    assert receipt["checkpoint_at_end"] is False
    assert not (tmp_path / "throughput-only/checkpoints").exists()

    raw.update(
        {
            "run_kind": "pretrain",
            "target_tokens": 50_000_000_000,
            "data_format": "packed_v1",
            "data_manifest": "manifest.json",
            "checkpoint_at_start": True,
        }
    )
    config.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="final checkpoint"):
        ScratchTrainConfig.from_json(str(config))


def test_scratch_trainer_has_no_initialization_checkpoint_argument():
    with pytest.raises(TypeError, match="init_checkpoint"):
        run_scratch_training("unused.json", init_checkpoint="forbidden.pt")


def test_distributed_mlm_scaling_matches_global_masked_token_mean():
    torch.manual_seed(23)
    logits = torch.randn(10, 5, requires_grad=True)
    labels = torch.tensor([1, -100, 2, -100, -100, 3, 4, 0, -100, 2])
    local_counts = [int(labels[:5].ne(-100).sum()), int(labels[5:].ne(-100).sum())]
    global_count = sum(local_counts)
    local_losses = [
        F.cross_entropy(logits[:5], labels[:5], ignore_index=-100),
        F.cross_entropy(logits[5:], labels[5:], ignore_index=-100),
    ]
    emulated_fsdp_average = sum(
        loss * _mlm_loss_scale(count, global_count, data_parallel_size=2)
        for loss, count in zip(local_losses, local_counts)
    ) / 2
    emulated_fsdp_average.backward()
    distributed_gradient = logits.grad.detach().clone()

    reference_logits = logits.detach().clone().requires_grad_(True)
    global_loss = F.cross_entropy(reference_logits, labels, ignore_index=-100)
    global_loss.backward()

    assert torch.allclose(emulated_fsdp_average.detach(), global_loss.detach())
    assert torch.allclose(distributed_gradient, reference_logits.grad)
