import json

import pytest
import torch

from openprotein.provenance import sha256_file
from openprotein.validation import (
    _geometry_prior_diagnostics,
    load_trainable_checkpoint,
    retrieval_metrics,
)


class TinyCheckpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable = torch.nn.Parameter(torch.zeros(2))
        self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)


def _checkpoint(tmp_path, provenance):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    state = checkpoint / "state.pt"
    torch.save(
        {
            "schema_version": 3,
            "step": 7,
            "provenance": provenance,
            "trainable_model": {"trainable": torch.tensor([2.0, 3.0])},
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
    return checkpoint


def test_validation_checkpoint_binds_selected_training_provenance(tmp_path):
    expected = {
        "model_config_sha256": "model",
        "tokenizer_sha256": "tokenizer",
        "data_manifest_sha256": "data",
    }
    checkpoint = _checkpoint(tmp_path, expected)
    model = TinyCheckpointModel()
    receipt = load_trainable_checkpoint(model, str(checkpoint), expected)
    assert receipt["step"] == 7
    assert torch.equal(model.trainable, torch.tensor([2.0, 3.0]))

    with pytest.raises(RuntimeError, match="data_manifest_sha256"):
        load_trainable_checkpoint(
            TinyCheckpointModel(),
            str(checkpoint),
            {**expected, "data_manifest_sha256": "other"},
        )


def test_retrieval_metrics_require_correct_diagonal_pairs():
    embeddings = torch.eye(4)
    perfect = retrieval_metrics(embeddings, embeddings)
    assert perfect["protein_to_text_recall_at_1"] == 1.0
    assert perfect["protein_to_text_recall_at_5"] == 1.0
    assert perfect["text_to_protein_recall_at_10"] == 1.0
    assert perfect["text_to_protein_mrr"] == 1.0

    shifted = retrieval_metrics(embeddings, embeddings.roll(1, dims=0))
    assert shifted["protein_to_text_recall_at_1"] == 0.0
    assert shifted["protein_to_text_recall_at_5"] == 1.0


def test_retrieval_metrics_allow_equivalent_duplicate_text_groups():
    protein = torch.eye(3)
    text = torch.stack((protein[1], protein[0], protein[2]))
    exact = retrieval_metrics(protein, text)
    grouped = retrieval_metrics(protein, text, match_group_ids=["same", "same", "other"])
    assert exact["protein_to_text_recall_at_1"] == pytest.approx(1 / 3)
    assert grouped["protein_to_text_recall_at_1"] == 1.0
    assert grouped["matching_policy"] == "normalized_text_group"
    assert grouped["random_reference_recall_at_1"] == pytest.approx(1 / 3)


def test_geometry_prior_diagnostics_use_calibration_counts_only():
    counts = torch.ones((6, 4), dtype=torch.long)
    counts[:, 2] = 20
    buckets = torch.tensor([0, 1, 5, 3])
    labels = torch.tensor([2, 2, 2, 1])
    receipt = _geometry_prior_diagnostics(
        counts,
        buckets,
        labels,
        model_evaluation_nll=2.0,
        model_evaluation_correct=3,
    )
    assert receipt["evaluation_targets"] == 4
    assert receipt["model_clean_sequence_accuracy_on_evaluation_half"] == 0.75
    assert receipt["sequence_separation_prior_accuracy"] == 0.75
    assert receipt["global_marginal_prior_accuracy"] == 0.75
