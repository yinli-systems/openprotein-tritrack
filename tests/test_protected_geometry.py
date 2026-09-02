import copy
import json
import os
from pathlib import Path

import pytest

from openprotein.protected_geometry import finalize_protected_geometry
from openprotein.provenance import sha256_file, source_tree_sha256


def _write(path, payload):
    path.write_text(json.dumps(payload))
    return path


def _validation(checkpoint, settings):
    return {
        "status": "passed",
        "evidence_level": "VALIDATION",
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "checkpoint": checkpoint,
        "evaluation_code_sha256": "filled-later",
        "evaluation_settings": settings,
    }


def _fixture(tmp_path):
    snapshot = tmp_path / "snapshot"
    package = snapshot / "src/openprotein"
    package.mkdir(parents=True)
    (package / "validation.py").write_text("# frozen\n")
    scripts = snapshot / "scripts"
    scripts.mkdir()
    for name in (
        "run_protected_geometry_gate.sbatch",
        "validate_sequence_gate.sbatch",
        "validate_text_gate.sbatch",
        "validate_structure_gate.sbatch",
        "audit_checkpoint_delta.sbatch",
    ):
        (scripts / name).write_text("#!/usr/bin/env bash\n")
    evaluator_hash = source_tree_sha256(package)
    source_hash = source_tree_sha256(snapshot)

    initializer_checkpoint = {
        "step": 10000,
        "manifest_sha256": "initializer-manifest",
        "state_sha256": "initializer-state",
        "training_provenance": {"train_config_sha256": "sequence-text"},
    }
    target_provenance = {"train_config_sha256": "geometry"}
    target_checkpoint = {
        "step": 1000,
        "manifest_sha256": "target-manifest",
        "state_sha256": "target-state",
        "training_provenance": target_provenance,
    }
    upstream = {
        "status": "passed",
        "evidence_level": "VALIDATION_SELECTION",
        "candidate_set_complete": True,
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "selected": {"checkpoint": initializer_checkpoint},
    }
    upstream_path = _write(tmp_path / "upstream.json", upstream)
    train = {
        "status": "passed",
        "evidence_level": "PILOT",
        "task": "sequence_structure",
        "step": 1000,
        "source_tree_sha256": source_hash,
        "provenance": target_provenance,
        "initialization_checkpoint": initializer_checkpoint,
    }
    train_path = _write(tmp_path / "train.json", train)

    sequence_settings = {
        "max_batches": 1000,
        "sequence_length": 256,
        "mlm_probability": 0.15,
        "seed": 2701,
    }
    text_settings = {
        "max_batches": 125,
        "batch_size": 8,
        "sequence_length": 128,
        "text_length": 128,
        "mlm_probability": 0.15,
        "seed": 3701,
    }
    structure_settings = {
        "max_batches": 1000,
        "batch_size": 1,
        "sequence_length": 256,
        "geometry_pairs": 128,
        "mlm_probability": 0.15,
        "minimum_plddt": 70.0,
        "seed": 4701,
    }

    initializer_sequence = _validation(initializer_checkpoint, sequence_settings)
    initializer_sequence.update(
        {
            "data_manifest_sha256": "sequence-data",
            "model_config_sha256": "model",
            "tokenizer_sha256": "protein-tokenizer",
            "model_artifact": {"manifest_sha256": "artifact"},
            "validation_files": ["sequence.csv"],
            "batches": 1000,
            "masked_tokens": 38449,
            "base_esmc_loss": 1.91,
            "base_esmc_perplexity": 6.75,
            "bridge_loss": 1.81,
            "bridge_perplexity": 6.11,
            "bridge_minus_base_loss": -0.10,
        }
    )
    initializer_text = _validation(initializer_checkpoint, text_settings)
    initializer_text.update(
        {
            "data_manifest_sha256": "text-data",
            "model_config_sha256": "model",
            "protein_tokenizer_sha256": "protein-tokenizer",
            "text_tokenizer_sha256": "text-tokenizer",
            "model_artifact": {"manifest_sha256": "artifact"},
            "validation_files": ["text.csv"],
            "batches": 125,
            "pairs": 1000,
            "text_target_tokens": 100000,
            "protein_masked_tokens": 19000,
            "protein_mlm_loss": 1.8,
            "text_lm_loss": 3.7,
            "text_lm_perplexity": 40.4,
            "conditioning_control": {"paired_better": True},
            "retrieval": {"protein_to_text_recall_at_1": 0.01},
            "retrieval_encoding_policy": "fixed",
        }
    )
    initializer_structure = _validation(initializer_checkpoint, structure_settings)
    initializer_structure.update(
        {
            "data_manifest_sha256": "text-data",
            "structure_manifest_sha256": "structure-manifest",
            "structure_database_sha256": "structure-db",
            "model_config_sha256": "model",
            "protein_tokenizer_sha256": "protein-tokenizer",
            "model_artifact": {"manifest_sha256": "artifact"},
            "validation_files": ["text.csv"],
            "batches": 1000,
            "protein_masked_tokens": 19000,
            "geometry_targets": 127280,
            "structure_kind": "AlphaFold DB predicted",
            "distance_bins": 64,
            "minimum_plddt": 70.0,
            "uniform_reference_loss": 4.158883,
            "uniform_reference_accuracy": 0.015625,
            "protein_mlm_loss": 1.81,
            "protein_mlm_perplexity": 6.11,
            "geometry_loss": 4.15,
            "geometry_accuracy": 0.016,
        }
    )
    final_sequence = copy.deepcopy(initializer_sequence)
    final_text = copy.deepcopy(initializer_text)
    final_structure = copy.deepcopy(initializer_structure)
    for payload in (final_sequence, final_text, final_structure):
        payload["checkpoint"] = target_checkpoint
    final_structure["geometry_loss"] = 3.0
    final_structure["geometry_accuracy"] = 0.34
    for payload in (
        initializer_sequence,
        initializer_text,
        initializer_structure,
        final_sequence,
        final_text,
        final_structure,
    ):
        payload["evaluation_code_sha256"] = evaluator_hash

    paths = {}
    for name, payload in (
        ("initializer_sequence", initializer_sequence),
        ("initializer_text", initializer_text),
        ("initializer_structure", initializer_structure),
        ("final_sequence", final_sequence),
        ("final_text", final_text),
        ("final_structure", final_structure),
    ):
        paths[name] = _write(tmp_path / f"{name}.json", payload)

    audit = {
        "status": "passed",
        "evidence_level": "CHECKPOINT_AUDIT",
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "allowed_changed_prefixes": ["geometry_head."],
        "changed_tensors": [
            "geometry_head.classifier.bias",
            "geometry_head.classifier.weight",
            "geometry_head.left.weight",
            "geometry_head.right.weight",
        ],
        "changed_tensors_count": 4,
        "unchanged_tensors": 53,
        "source_checkpoint": initializer_checkpoint,
        "target_checkpoint": target_checkpoint,
        "changed_parameters": 1327168,
        "unchanged_parameters": 723566080,
    }
    audit_path = _write(tmp_path / "audit.json", audit)
    policy = {
        "schema_version": 1,
        "target_step": 1000,
        "allowed_changed_prefixes": ["geometry_head."],
        "expected_changed_parameters": 1327168,
        "expected_unchanged_parameters": 723566080,
        "expected_changed_tensors": [
            "geometry_head.classifier.bias",
            "geometry_head.classifier.weight",
            "geometry_head.left.weight",
            "geometry_head.right.weight",
        ],
        "expected_changed_tensors_count": 4,
        "expected_unchanged_tensors_count": 53,
        "minimum_geometry_loss_improvement": 0.5,
        "minimum_geometry_accuracy_improvement": 0.1,
        "required_upstream_selection": {
            "receipt_sha256": sha256_file(upstream_path),
            "selected_checkpoint": {
                key: initializer_checkpoint[key]
                for key in ("step", "manifest_sha256", "state_sha256")
            },
        },
        "required_train_receipt": {
            "status": "passed",
            "evidence_level": "PILOT",
            "task": "sequence_structure",
            "step": 1000,
            "source_tree_sha256": source_hash,
        },
        "required_train_provenance": target_provenance,
        "evaluator": {
            "source_snapshot": str(snapshot),
            "source_snapshot_sha256": source_hash,
            "evaluation_code_sha256": evaluator_hash,
            "script_sha256": {
                name: sha256_file(scripts / name)
                for name in (
                    "run_protected_geometry_gate.sbatch",
                    "validate_sequence_gate.sbatch",
                    "validate_text_gate.sbatch",
                    "validate_structure_gate.sbatch",
                    "audit_checkpoint_delta.sbatch",
                )
            },
            "sequence_settings": sequence_settings,
            "text_settings": text_settings,
            "structure_settings": structure_settings,
            "sequence_receipt_contract": {
                "data_manifest_sha256": "sequence-data",
                "model_config_sha256": "model",
                "tokenizer_sha256": "protein-tokenizer",
                "model_artifact_manifest_sha256": "artifact",
                "validation_files": ["sequence.csv"],
                "batches": 1000,
                "masked_tokens": 38449,
            },
            "text_receipt_contract": {
                "data_manifest_sha256": "text-data",
                "model_config_sha256": "model",
                "protein_tokenizer_sha256": "protein-tokenizer",
                "text_tokenizer_sha256": "text-tokenizer",
                "model_artifact_manifest_sha256": "artifact",
                "validation_files": ["text.csv"],
                "batches": 125,
                "pairs": 1000,
            },
            "structure_receipt_contract": {
                "data_manifest_sha256": "text-data",
                "structure_manifest_sha256": "structure-manifest",
                "structure_database_sha256": "structure-db",
                "model_config_sha256": "model",
                "protein_tokenizer_sha256": "protein-tokenizer",
                "model_artifact_manifest_sha256": "artifact",
                "validation_files": ["text.csv"],
                "batches": 1000,
                "geometry_targets": 127280,
                "structure_kind": "AlphaFold DB predicted",
            },
        },
        "test_data_policy": "forbidden",
        "proteingym_rows_opened": False,
    }
    policy_path = _write(tmp_path / "policy.json", policy)
    for root, directories, files in os.walk(snapshot, topdown=False):
        for name in files:
            os.chmod(Path(root) / name, 0o444)
        for name in directories:
            os.chmod(Path(root) / name, 0o555)
    os.chmod(snapshot, 0o555)
    return policy_path, upstream_path, train_path, audit_path, paths


def _finalize(tmp_path, fixture, output_name="result.json"):
    policy, upstream, train, audit, paths = fixture
    return finalize_protected_geometry(
        policy_path=str(policy),
        upstream_selection_receipt=str(upstream),
        train_receipt=str(train),
        initializer_sequence_receipt=str(paths["initializer_sequence"]),
        initializer_text_receipt=str(paths["initializer_text"]),
        initializer_structure_receipt=str(paths["initializer_structure"]),
        final_sequence_receipt=str(paths["final_sequence"]),
        final_text_receipt=str(paths["final_text"]),
        final_structure_receipt=str(paths["final_structure"]),
        checkpoint_audit_receipt=str(audit),
        output=str(tmp_path / output_name),
    )


def test_protected_geometry_finalizer_requires_improvement_and_bitwise_shared_path(tmp_path):
    result = _finalize(tmp_path, _fixture(tmp_path))
    assert result["status"] == "passed"
    assert result["selected_checkpoint"]["step"] == 1000
    assert result["shared_sequence_text_path_bitwise_preserved"] is True
    assert result["geometry_loss_improvement"] > 1.0
    assert result["proteingym_rows_opened"] is False


def test_protected_geometry_finalizer_rejects_shared_text_drift(tmp_path):
    fixture = _fixture(tmp_path)
    paths = fixture[-1]
    payload = json.loads(paths["final_text"].read_text())
    payload["text_lm_loss"] += 0.001
    paths["final_text"].write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="protected text text_lm_loss"):
        _finalize(tmp_path, fixture)
