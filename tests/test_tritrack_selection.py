import json
from pathlib import Path

import pytest

from openprotein.training import TrainConfig
from openprotein.tritrack_selection import select_tritrack_batchcap

ROOT = Path(__file__).resolve().parents[1]


def _config(name: str, output: Path, micro_batch_size: int, formal: bool) -> dict:
    config = {
        "name": name,
        "task": "sequence_structure_text",
        "model_config": "configs/model/model.json",
        "model_artifact_manifest": "/artifacts/model.json",
        "tokenizer_json": "configs/tokenizer.json",
        "text_tokenizer_model": "/data/tokenizer.model",
        "train_glob": "/data/train/*.csv",
        "data_manifest": "/data/manifest.json",
        "structure_database": "/data/structures.sqlite",
        "structure_manifest": "/data/structures.json",
        "replay_train_glob": "/data/replay/*.csv",
        "replay_data_manifest": "/data/replay.json",
        "replay_every": 3,
        "replay_loss_weight": 1.0,
        "output_dir": str(output),
        "sequence_length": 512,
        "text_length": 128,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": 1,
        "max_steps": 100_000 if formal else 20,
        "learning_rate": 0.00008,
        "min_learning_rate": 0.000008,
        "warmup_steps": 1000 if formal else 5,
        "weight_decay": 0.05,
        "max_grad_norm": 1.0,
        "mlm_probability": 0.15,
        "structure_mask_probability": 0.15,
        "structure_dropout_probability": 0.1,
        "attention_policy": "prefix_lm",
        "geometry_pairs": 256,
        "min_structure_confidence": 70.0,
        "detach_geometry_backbone": False,
        "loss_weights": {
            "protein_mlm": 1.0,
            "text_lm": 0.5,
            "contrastive": 0.2,
            "geometry": 0.25,
            "structure_denoising": 0.25,
        },
        "seed": 260903,
        "log_every": 10 if formal else 1,
        "checkpoint_every": 2500 if formal else 20,
        "precision": "bf16",
    }
    if formal:
        config["max_tokens"] = 3_000_000_000
        config["warmup_tokens"] = 100_000_000
    return config


def _metrics(path: Path, throughput: float, peak_memory: float) -> None:
    path.mkdir(parents=True)
    rows = [
        {
            "step": step,
            "loss": 5.0 - step / 100,
            "grad_norm": 1.0,
            "tokens": 10_000,
            "tokens_per_second": throughput + step,
            "peak_memory_mib": peak_memory,
        }
        for step in range(1, 21)
    ]
    (path / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _policy_tree(tmp_path: Path):
    config_dir = tmp_path / "configs" / "train"
    config_dir.mkdir(parents=True)
    candidates = []
    outputs = {}
    for micro_batch_size in (12, 16, 20):
        probe_relative = f"configs/train/probe-{micro_batch_size}.json"
        formal_relative = f"configs/train/formal-{micro_batch_size}.json"
        output = tmp_path / "runs" / f"probe-{micro_batch_size}"
        outputs[micro_batch_size] = output
        (tmp_path / probe_relative).write_text(
            json.dumps(_config(f"probe-{micro_batch_size}", output, micro_batch_size, False))
        )
        (tmp_path / formal_relative).write_text(
            json.dumps(
                _config(
                    f"formal-{micro_batch_size}",
                    tmp_path / "runs" / f"formal-{micro_batch_size}",
                    micro_batch_size,
                    True,
                )
            )
        )
        candidates.append(
            {"probe_config": probe_relative, "formal_config": formal_relative}
        )
    policy = config_dir / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "world_size": 6,
                "physical_gpu_memory_mib": 24_564,
                "minimum_headroom_fraction": 0.15,
                "required_steps": 20,
                "discard_initial_steps": 5,
                "formal_target_tokens": 3_000_000_000,
                "candidates": candidates,
            }
        )
    )
    return policy, candidates, outputs


def test_selection_prefers_fastest_candidate_that_passes_memory_gate(tmp_path):
    policy, candidates, outputs = _policy_tree(tmp_path)
    _metrics(outputs[12], throughput=100.0, peak_memory=15_000.0)
    _metrics(outputs[16], throughput=140.0, peak_memory=21_500.0)
    status = tmp_path / "status.tsv"
    status.write_text(
        f"{candidates[0]['probe_config']}\t0\n"
        f"{candidates[1]['probe_config']}\t0\n"
        f"{candidates[2]['probe_config']}\t1\n"
    )
    output = tmp_path / "selection.json"

    receipt = select_tritrack_batchcap(str(policy), str(status), str(output))

    assert receipt["selected_formal_config"] == candidates[0]["formal_config"]
    assert receipt["candidates"][0]["selected"] is True
    assert receipt["candidates"][1]["admissible"] is False
    assert "headroom" in receipt["candidates"][1]["rejected_reason"]
    assert receipt["candidates"][2]["exit_code"] == 1
    assert json.loads(output.read_text())["status"] == "passed"


def test_selection_rejects_status_set_drift(tmp_path):
    policy, candidates, _ = _policy_tree(tmp_path)
    status = tmp_path / "status.tsv"
    status.write_text(f"{candidates[0]['probe_config']}\t0\n")
    with pytest.raises(RuntimeError, match="status set"):
        select_tritrack_batchcap(str(policy), str(status), str(tmp_path / "selection.json"))


def test_production_tritrack_policy_binds_valid_probe_and_formal_configs():
    policy = json.loads(
        (ROOT / "configs/train/tritrack-1.2b-hp6-batchcap-policy.json").read_text()
    )
    assert policy["world_size"] == 6
    assert policy["formal_target_tokens"] == 3_000_000_000
    for candidate in policy["candidates"]:
        probe = TrainConfig.from_json(str(ROOT / candidate["probe_config"]))
        formal = TrainConfig.from_json(str(ROOT / candidate["formal_config"]))
        assert probe.micro_batch_size == formal.micro_batch_size
        assert probe.max_steps == 20
        assert formal.max_tokens == policy["formal_target_tokens"]
        assert formal.warmup_tokens == 100_000_000
