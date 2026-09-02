from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from openprotein.config import load_model_config
from openprotein.model import OpenProteinModel


ROOT = Path(__file__).resolve().parents[1]


def _tiny_model():
    config = load_model_config(str(ROOT / "configs/model/tiny.json"))
    return OpenProteinModel.from_config(config, dtype=torch.float32)


def test_tiny_multitask_forward_backward_is_finite():
    torch.manual_seed(7)
    model = _tiny_model()
    protein_ids = torch.randint(0, 40, (2, 12))
    protein_mask = torch.tensor([[1] * 12, [1] * 9 + [0] * 3])
    text_ids = torch.randint(0, 128, (2, 7))
    text_mask = torch.tensor([[1] * 7, [1] * 5 + [0] * 2])

    protein_labels = protein_ids.clone()
    protein_labels[:, 1::2] = -100
    text_labels = text_ids.clone()
    text_labels[text_mask == 0] = -100
    pair_indices = torch.tensor([[[0, 3], [2, 8]], [[0, 2], [3, 7]]])
    distance_labels = torch.tensor([[1, 4], [2, 6]])

    output = model(
        protein_ids,
        protein_mask,
        text_ids,
        text_mask,
        protein_labels=protein_labels,
        text_labels=text_labels,
        pair_indices=pair_indices,
        distance_labels=distance_labels,
        attention_policy="prefix_lm",
    )
    assert set(output["losses"]) == {"protein_mlm", "text_lm", "contrastive", "geometry"}
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.bridge[0].qkv.weight.grad is not None
    assert torch.isfinite(model.bridge[0].qkv.weight.grad).all()


def test_sequence_only_control_bypasses_bridge_exactly():
    torch.manual_seed(11)
    model = _tiny_model().eval()
    protein_ids = torch.randint(0, 40, (1, 9))
    protein_mask = torch.ones_like(protein_ids)
    output = model(protein_ids, protein_mask, use_bridge=False)
    assert torch.equal(output["base_protein_logits"], output["protein_logits"])


def test_prefix_lm_forward_has_expected_shapes():
    model = _tiny_model().eval()
    output = model(
        torch.randint(0, 40, (2, 8)),
        torch.ones((2, 8), dtype=torch.long),
        torch.randint(0, 128, (2, 5)),
        torch.ones((2, 5), dtype=torch.long),
        attention_policy="prefix_lm",
    )
    assert output["protein_logits"].shape == (2, 8, 40)
    assert output["text_logits"].shape == (2, 5, 128)
    assert output["protein_embedding"].shape == (2, 32)
    assert output["text_embedding"].shape == (2, 32)


def test_tied_text_parameterization_scales_inputs_and_logits():
    torch.manual_seed(13)
    model = _tiny_model().eval()
    text_ids = torch.tensor([[3, 7]])
    with torch.no_grad():
        model.text_embedding.weight.fill_(1.0)
        model.modality_embedding.weight[1].fill_(3.0)
        embedded = model._embed_text(text_ids)
        hidden = torch.randn(1, 2, model.config.hidden_size)
        logits = model._text_logits(hidden)
    scale = model.config.hidden_size**-0.5
    assert torch.equal(embedded, torch.full_like(embedded, 4.0 * scale))
    assert torch.allclose(
        logits,
        F.linear(hidden, model.text_embedding.weight) * scale,
    )


def test_text_lm_rejects_bidirectional_target_leakage():
    model = _tiny_model().eval()
    with torch.no_grad(), pytest.raises(ValueError, match="target leakage"):
        model(
            torch.randint(0, 40, (1, 8)),
            torch.ones((1, 8), dtype=torch.long),
            torch.randint(0, 128, (1, 5)),
            torch.ones((1, 5), dtype=torch.long),
            text_labels=torch.randint(0, 128, (1, 5)),
            attention_policy="bidirectional",
        )


def test_requested_contrastive_objective_cannot_disappear_at_batch_one():
    model = _tiny_model().eval()
    with torch.no_grad(), pytest.raises(ValueError, match="contrastive"):
        model(
            torch.randint(0, 40, (1, 8)),
            torch.ones((1, 8), dtype=torch.long),
            torch.randint(0, 128, (1, 5)),
            torch.ones((1, 5), dtype=torch.long),
            text_labels=torch.randint(0, 128, (1, 5)),
            attention_policy="prefix_lm",
            loss_weights={"text_lm": 0.5, "contrastive": 0.1},
        )


def test_detached_geometry_objective_does_not_update_shared_bridge():
    torch.manual_seed(17)
    model = _tiny_model()
    protein_ids = torch.randint(0, 40, (1, 10))
    protein_labels = protein_ids.clone()
    pair_indices = torch.tensor([[[1, 4], [2, 7]]])
    output = model(
        protein_ids,
        torch.ones_like(protein_ids),
        protein_labels=protein_labels,
        pair_indices=pair_indices,
        distance_labels=torch.tensor([[2, 5]]),
        detach_geometry_backbone=True,
        loss_weights={"protein_mlm": 0.0, "geometry": 1.0},
    )
    assert output["loss"] == output["losses"]["geometry"]
    output["loss"].backward()
    assert model.geometry_head.classifier.weight.grad is not None
    assert model.bridge[0].qkv.weight.grad is None
    assert model.text_embedding.weight.grad is None


def test_zero_initialized_structure_adapter_preserves_warm_start_then_learns():
    torch.manual_seed(23)
    base_config = load_model_config(str(ROOT / "configs/model/tiny.json"))
    config = replace(base_config, structure_feature_dimension=12)
    model = OpenProteinModel.from_config(config, dtype=torch.float32).eval()
    protein_ids = torch.randint(0, 40, (2, 10))
    protein_mask = torch.ones_like(protein_ids)
    text_ids = torch.randint(0, 128, (2, 5))
    text_mask = torch.ones_like(text_ids)
    structure = torch.randn(2, 10, 12)
    structure_mask = torch.ones((2, 10), dtype=torch.bool)
    structure_label_mask = torch.zeros((2, 10), dtype=torch.bool)
    structure_label_mask[:, 3] = True
    with torch.no_grad():
        warm = model(protein_ids, protein_mask, text_ids, text_mask)
        gated = model(
            protein_ids,
            protein_mask,
            text_ids,
            text_mask,
            structure_input_features=structure,
            structure_attention_mask=structure_mask,
        )
    assert torch.equal(warm["protein_logits"], gated["protein_logits"])
    assert torch.equal(warm["text_logits"], gated["text_logits"])

    output = model(
        protein_ids,
        protein_mask,
        text_ids,
        text_mask,
        structure_input_features=structure,
        structure_attention_mask=structure_mask,
        structure_labels=structure,
        structure_label_mask=structure_label_mask,
        loss_weights={"structure_denoising": 1.0},
    )
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.structure_gate.grad is not None
    assert torch.isfinite(model.structure_gate.grad).all()


def test_contextual_multi_positive_contrastive_and_residue_structure_gate():
    torch.manual_seed(29)
    base = load_model_config(str(ROOT / "configs/model/tiny.json"))
    config = replace(
        base,
        structure_feature_dimension=12,
        structure_fusion="residue_gate",
        structure_mask_token=True,
        contrastive_text_layers=1,
        multi_positive_contrastive=True,
    )
    model = OpenProteinModel.from_config(config, dtype=torch.float32)
    protein_ids = torch.randint(0, 40, (3, 10))
    protein_mask = torch.ones_like(protein_ids)
    text_ids = torch.randint(0, 128, (3, 6))
    text_mask = torch.ones_like(text_ids)
    structure = torch.randn(3, 10, 12)
    structure_mask = torch.ones((3, 10), dtype=torch.bool)
    label_mask = torch.zeros((3, 10), dtype=torch.bool)
    label_mask[:, 2] = True
    output = model(
        protein_ids,
        protein_mask,
        text_ids,
        text_mask,
        structure_input_features=structure,
        structure_attention_mask=structure_mask,
        structure_labels=structure,
        structure_label_mask=label_mask,
        contrastive_group_ids=torch.tensor([7, 7, 11]),
        attention_policy="prefix_lm",
        loss_weights={"contrastive": 1.0, "structure_denoising": 1.0},
    )
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.contrastive_logit_scale.grad is not None
    assert model.structure_gate_projection.weight.grad is not None
    assert model.structure_mask_embedding.grad is not None
