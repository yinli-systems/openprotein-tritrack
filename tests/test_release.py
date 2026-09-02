import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from openprotein.artifacts import build_artifact_manifest
from openprotein.config import load_model_config
from openprotein.model import OpenProteinModel
from openprotein.provenance import sha256_file
from openprotein.release import export_trainable_adapter, load_released_adapter

ROOT = Path(__file__).resolve().parents[1]


def test_adapter_export_excludes_optimizer_and_binds_dependencies(tmp_path):
    config = tmp_path / "model.json"
    tokenizer = tmp_path / "tokenizer.json"
    config.write_bytes((ROOT / "configs/model/tiny.json").read_bytes())
    tokenizer.write_text("{}")
    artifact_root = tmp_path / "backbone"
    artifact_root.mkdir()
    (artifact_root / "weights.bin").write_bytes(b"backbone")
    artifact_manifest = tmp_path / "artifact.json"
    build_artifact_manifest(
        str(artifact_root), str(artifact_manifest), ["*.bin"], {"name": "fixture"}
    )
    provenance = {
        "model_config_sha256": sha256_file(config),
        "tokenizer_sha256": sha256_file(tokenizer),
        "model_artifact_manifest_sha256": sha256_file(artifact_manifest),
    }
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    state = checkpoint / "state.pt"
    torch.save(
        {
            "step": 11,
            "provenance": provenance,
            "trainable_model": {"bridge.weight": torch.arange(6, dtype=torch.float32)},
            "optimizer": {"must_not_ship": torch.ones(100)},
        },
        state,
    )
    (checkpoint / "manifest.json").write_text(
        json.dumps(
            {
                "state_bytes": state.stat().st_size,
                "state_sha256": sha256_file(state),
            }
        )
    )
    release = tmp_path / "release"
    manifest = export_trainable_adapter(
        str(checkpoint),
        str(config),
        str(tokenizer),
        str(artifact_manifest),
        str(release),
    )
    exported = load_file(str(release / "adapter.safetensors"))
    assert exported.keys() == {"bridge.weight"}
    assert manifest["adapter"]["parameters"] == 6
    assert (release / "adapter.safetensors").stat().st_size < state.stat().st_size
    assert (release / "model-config.json").read_bytes() == config.read_bytes()


def test_release_reconstruction_verifies_and_loads_every_trainable_tensor(tmp_path):
    config_path = tmp_path / "model.json"
    config_path.write_bytes((ROOT / "configs/model/tiny.json").read_bytes())
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    text_tokenizer = tmp_path / "text-tokenizer.model"
    text_tokenizer.write_bytes(b"trained-text-tokenizer")
    artifact_root = tmp_path / "backbone"
    artifact_root.mkdir()
    (artifact_root / "weights.bin").write_bytes(b"backbone")
    artifact_manifest = tmp_path / "artifact.json"
    build_artifact_manifest(
        str(artifact_root), str(artifact_manifest), ["*.bin"], {"name": "fixture"}
    )
    config = load_model_config(str(config_path))
    torch.manual_seed(29)
    source_model = OpenProteinModel.from_config(config, dtype=torch.float32)
    trainable = {
        name: parameter.detach().clone()
        for name, parameter in source_model.named_parameters()
        if parameter.requires_grad
    }
    provenance = {
        "model_config_sha256": sha256_file(config_path),
        "tokenizer_sha256": sha256_file(tokenizer),
        "model_artifact_manifest_sha256": sha256_file(artifact_manifest),
        "text_tokenizer_sha256": sha256_file(text_tokenizer),
    }
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    state = checkpoint / "state.pt"
    torch.save(
        {
            "step": 7,
            "provenance": provenance,
            "trainable_model": trainable,
            "optimizer": {"must_not_ship": torch.ones(100)},
        },
        state,
    )
    (checkpoint / "manifest.json").write_text(
        json.dumps({"state_bytes": state.stat().st_size, "state_sha256": sha256_file(state)})
    )
    pretrain_receipt = tmp_path / "pretrain-receipt.json"
    pretrain_receipt.write_text(
        json.dumps(
            {
                "status": "passed",
                "evidence_level": "PRETRAIN",
                "token_budget_reached": True,
                "tokens": 101,
                "target_tokens": 100,
                "step": 7,
                "provenance": provenance,
                "trainable_parameters": sum(tensor.numel() for tensor in trainable.values()),
            }
        )
    )
    release = tmp_path / "release"
    export_trainable_adapter(
        str(checkpoint),
        str(config_path),
        str(tokenizer),
        str(artifact_manifest),
        str(release),
        text_tokenizer_model=str(text_tokenizer),
        pretrain_receipt=str(pretrain_receipt),
    )
    assert (release / "text-tokenizer.model").read_bytes() == text_tokenizer.read_bytes()
    reconstructed, receipt = load_released_adapter(
        str(release), str(tokenizer), str(artifact_manifest), device="cpu"
    )
    assert receipt["status"] == "passed"
    assert receipt["pretrain_receipt_sha256"] == sha256_file(pretrain_receipt)
    assert reconstructed.training is False
    for name, parameter in reconstructed.named_parameters():
        if parameter.requires_grad:
            assert torch.equal(parameter, trainable[name])

    adapter = release / "adapter.safetensors"
    adapter.write_bytes(adapter.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="byte size differs"):
        load_released_adapter(
            str(release), str(tokenizer), str(artifact_manifest), device="cpu"
        )


def test_adapter_export_rejects_pretrain_receipt_for_another_checkpoint(tmp_path):
    config = tmp_path / "model.json"
    tokenizer = tmp_path / "tokenizer.json"
    config.write_bytes((ROOT / "configs/model/tiny.json").read_bytes())
    tokenizer.write_text("{}")
    artifact_root = tmp_path / "backbone"
    artifact_root.mkdir()
    (artifact_root / "weights.bin").write_bytes(b"backbone")
    artifact_manifest = tmp_path / "artifact.json"
    build_artifact_manifest(
        str(artifact_root), str(artifact_manifest), ["*.bin"], {"name": "fixture"}
    )
    provenance = {
        "model_config_sha256": sha256_file(config),
        "tokenizer_sha256": sha256_file(tokenizer),
        "model_artifact_manifest_sha256": sha256_file(artifact_manifest),
    }
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    state = checkpoint / "state.pt"
    torch.save(
        {
            "step": 11,
            "provenance": provenance,
            "trainable_model": {"bridge.weight": torch.arange(6, dtype=torch.float32)},
        },
        state,
    )
    (checkpoint / "manifest.json").write_text(
        json.dumps({"state_bytes": state.stat().st_size, "state_sha256": sha256_file(state)})
    )
    receipt = tmp_path / "pretrain.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "passed",
                "evidence_level": "PRETRAIN",
                "token_budget_reached": True,
                "tokens": 100,
                "target_tokens": 100,
                "step": 10,
                "provenance": provenance,
                "trainable_parameters": 6,
            }
        )
    )
    with pytest.raises(RuntimeError, match="step differs"):
        export_trainable_adapter(
            str(checkpoint),
            str(config),
            str(tokenizer),
            str(artifact_manifest),
            str(tmp_path / "release"),
            pretrain_receipt=str(receipt),
        )
