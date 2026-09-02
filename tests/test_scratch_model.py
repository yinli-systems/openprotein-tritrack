import json
from pathlib import Path

import pytest
import torch

from openprotein.scratch.config import ScratchModelConfig, load_scratch_model_config
from openprotein.scratch.model import ScratchProteinModel


ROOT = Path(__file__).resolve().parents[1]


def _tiny_config():
    return load_scratch_model_config(
        str(ROOT / "configs/scratch/model/openprotein-scratch-tiny.json")
    )


def test_scratch_7b_parameter_contract_is_exact():
    config = load_scratch_model_config(
        str(ROOT / "configs/scratch/model/openprotein-scratch-7b.json")
    )
    assert config.initialization == "random"
    assert config.parameter_count == 7_065_227_297
    assert config.declared_parameter_count == config.parameter_count
    assert config.is_7b_scale


def test_scale_ladder_parameter_contracts_are_exact():
    expected = {
        "openprotein-scratch-125m.json": 127_049_761,
        "openprotein-scratch-1b.json": 1_011_437_601,
        "openprotein-scratch-7b.json": 7_065_227_297,
    }
    for filename, parameter_count in expected.items():
        config = load_scratch_model_config(str(ROOT / "configs/scratch/model" / filename))
        assert config.parameter_count == parameter_count


def test_scratch_tiny_analytical_count_matches_instantiated_model():
    config = _tiny_config()
    model = ScratchProteinModel(config)
    instantiated = sum(parameter.numel() for parameter in model.parameters())
    assert instantiated == config.parameter_count == 106_529
    assert model.lm_head.embedding_weight.data_ptr() == model.token_embedding.weight.data_ptr()
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_scratch_forward_backward_and_padding_are_finite():
    torch.manual_seed(7)
    model = ScratchProteinModel(_tiny_config())
    input_ids = torch.tensor([[0, 5, 32, 7, 2, 1], [0, 8, 9, 2, 1, 1]])
    attention_mask = input_ids.ne(1)
    labels = torch.tensor([[-100, -100, 6, -100, -100, -100], [-100, 8, -100, -100, -100, -100]])
    output = model(input_ids, attention_mask, labels)
    assert output["logits"].shape == (2, 6, 33)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_padding_token_values_cannot_change_valid_logits():
    torch.manual_seed(11)
    model = ScratchProteinModel(_tiny_config()).eval()
    first = torch.tensor([[0, 5, 6, 2, 1, 1]])
    second = first.clone()
    second[:, 4:] = torch.tensor([17, 23])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        first_logits = model(first, mask)["logits"][:, :4]
        second_logits = model(second, mask)["logits"][:, :4]
    assert torch.equal(first_logits, second_logits)


def test_scratch_config_rejects_nonrandom_or_checkpoint_fields():
    path = ROOT / "configs/scratch/model/openprotein-scratch-tiny.json"
    raw = json.loads(path.read_text())
    raw["initialization"] = "pretrained"
    with pytest.raises(ValueError, match="initialization"):
        ScratchModelConfig.from_dict(raw)
    raw["initialization"] = "random"
    raw["model_path"] = "/forbidden/checkpoint"
    with pytest.raises(ValueError, match="unknown fields"):
        ScratchModelConfig.from_dict(raw)
