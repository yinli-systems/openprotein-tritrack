from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from openprotein.data.tokenizer import EsmcTokenizer
from openprotein.training import RestartableBatcher, TrainConfig, _learning_rate, run_training


ROOT = Path(__file__).resolve().parents[1]


def test_token_cosine_schedule_reaches_floor_at_exact_budget():
    base = TrainConfig.from_json(str(ROOT / "configs/train/tiny-recovery.json"))
    config = replace(base, max_tokens=1_000, warmup_tokens=100)
    assert _learning_rate(config, 0, projected_tokens_seen=50) == pytest.approx(
        0.5 * config.learning_rate
    )
    assert _learning_rate(config, 10, projected_tokens_seen=100) == pytest.approx(
        config.learning_rate
    )
    assert _learning_rate(config, 20, projected_tokens_seen=1_000) == pytest.approx(
        config.min_learning_rate
    )


def test_tiny_checkpoint_resume(tmp_path, monkeypatch):
    source = ROOT / "configs/train/tiny-recovery.json"
    raw = source.read_text().replace("artifacts/tiny-recovery", str(tmp_path / "run"))
    config = tmp_path / "tiny-recovery.json"
    config.write_text(raw)
    # The temporary config is not under the repository, so make its relative
    # fixtures absolute while retaining the same content hash across both runs.
    raw = (
        config.read_text()
        .replace("configs/model/tiny.json", str(ROOT / "configs/model/tiny.json"))
        .replace("tests/fixtures/tiny_tokenizer.json", str(ROOT / "tests/fixtures/tiny_tokenizer.json"))
        .replace("tests/fixtures/train/*.csv", str(ROOT / "tests/fixtures/train/*.csv"))
    )
    config.write_text(raw)

    first = run_training(str(config), max_steps_override=2, device_name="cpu")
    checkpoint = tmp_path / "run/checkpoints/step-00000002"
    assert checkpoint.exists()
    second = run_training(
        str(config), resume=str(checkpoint), max_steps_override=4, device_name="cpu"
    )
    assert first["step"] == 2
    assert second["step"] == 4
    state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=False)
    assert state["step"] == 2
    assert state["rank_states"][0]["batcher"]["records_consumed"] == 4
    metrics = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert metrics[0]["losses"].keys() == {"protein_mlm"}

    uninterrupted_config = tmp_path / "tiny-uninterrupted.json"
    uninterrupted_config.write_text(
        config.read_text().replace(str(tmp_path / "run"), str(tmp_path / "uninterrupted"))
    )
    run_training(str(uninterrupted_config), max_steps_override=4, device_name="cpu")
    resumed_final = torch.load(
        tmp_path / "run/checkpoints/step-00000004/state.pt",
        map_location="cpu",
        weights_only=False,
    )
    uninterrupted_final = torch.load(
        tmp_path / "uninterrupted/checkpoints/step-00000004/state.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert resumed_final["rank_states"][0]["batcher"]["records_consumed"] == (
        uninterrupted_final["rank_states"][0]["batcher"]["records_consumed"]
    )
    for name, tensor in resumed_final["trainable_model"].items():
        assert torch.equal(tensor, uninterrupted_final["trainable_model"][name]), name


def test_cross_stage_initialization_loads_only_trainable_weights(tmp_path):
    source = ROOT / "configs/train/tiny-recovery.json"
    first_config = tmp_path / "stage-one.json"
    first_config.write_text(
        source.read_text()
        .replace("artifacts/tiny-recovery", str(tmp_path / "stage-one"))
        .replace("configs/model/tiny.json", str(ROOT / "configs/model/tiny.json"))
        .replace(
            "tests/fixtures/tiny_tokenizer.json",
            str(ROOT / "tests/fixtures/tiny_tokenizer.json"),
        )
        .replace("tests/fixtures/train/*.csv", str(ROOT / "tests/fixtures/train/*.csv"))
    )
    run_training(str(first_config), max_steps_override=1, device_name="cpu")
    initialization = tmp_path / "stage-one/checkpoints/step-00000001"

    second_config = tmp_path / "stage-two.json"
    second_config.write_text(
        first_config.read_text()
        .replace('"name": "tiny-recovery"', '"name": "tiny-stage-two"')
        .replace(str(tmp_path / "stage-one"), str(tmp_path / "stage-two"))
    )
    receipt = run_training(
        str(second_config),
        max_steps_override=1,
        device_name="cpu",
        init_checkpoint=str(initialization),
    )
    assert receipt["initialization_checkpoint"]["step"] == 1
    assert receipt["initialization_checkpoint"]["optimizer_loaded"] is False
    assert receipt["tokens"] > 0

    resumed = run_training(
        str(second_config),
        resume=str(tmp_path / "stage-two/checkpoints/step-00000001"),
        max_steps_override=2,
        device_name="cpu",
    )
    assert resumed["resumed_checkpoint_schema"] == 4
    assert resumed["initialization_checkpoint"]["step"] == 1
    assert resumed["initialization_checkpoint"]["state_sha256"] == receipt[
        "initialization_checkpoint"
    ]["state_sha256"]

    with pytest.raises(ValueError, match="mutually exclusive"):
        run_training(
            str(second_config),
            resume=str(initialization),
            init_checkpoint=str(initialization),
            device_name="cpu",
        )


def test_dynamic_batcher_replays_prefetched_example_after_restart():
    tokenizer = EsmcTokenizer.from_tokenizer_json(
        str(ROOT / "tests/fixtures/tiny_tokenizer.json")
    )
    paths = [str(ROOT / "tests/fixtures/train/train.csv")]

    batcher = RestartableBatcher(
        paths,
        tokenizer,
        sequence_length=16,
        batch_size=1,
        seed=31,
        mlm_probability=0.5,
        micro_batch_token_budget=16,
        max_micro_batch_size=8,
    )
    first = batcher.next()
    state = batcher.state_dict()
    second = batcher.next()

    restored = RestartableBatcher(
        paths,
        tokenizer,
        sequence_length=16,
        batch_size=1,
        seed=31,
        mlm_probability=0.5,
        micro_batch_token_budget=16,
        max_micro_batch_size=8,
    )
    restored.load_state_dict(state)
    replay = restored.next()

    assert first["protein_input_ids"].numel() <= 16
    for key, value in second.items():
        assert torch.equal(value, replay[key]), key


def test_exact_token_budget_emits_pretrain_receipt(tmp_path):
    source = json.loads((ROOT / "configs/train/tiny-recovery.json").read_text())
    source.update(
        {
            "name": "tiny-token-budget",
            "model_config": str(ROOT / "configs/model/tiny.json"),
            "tokenizer_json": str(ROOT / "tests/fixtures/tiny_tokenizer.json"),
            "train_glob": str(ROOT / "tests/fixtures/train/*.csv"),
            "output_dir": str(tmp_path / "token-budget"),
            "micro_batch_size": 1,
            "micro_batch_token_budget": 16,
            "max_micro_batch_size": 8,
            "max_tokens": 50,
            "max_steps": 3,
        }
    )
    config = tmp_path / "tiny-token-budget.json"
    config.write_text(json.dumps(source))

    receipt = run_training(str(config), device_name="cpu")

    assert receipt["status"] == "passed"
    assert receipt["evidence_level"] == "PRETRAIN"
    assert receipt["token_budget_reached"] is True
    assert receipt["tokens"] >= 50
    assert receipt["step"] == 2
    assert receipt["stopped_by"] == "token_budget"
