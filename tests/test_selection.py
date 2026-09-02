import copy
import json
import os
from pathlib import Path

import pytest

from openprotein.policy_selection import (
    select_multimodal_checkpoint_with_policy,
    select_sequence_checkpoint_with_policy,
)
from openprotein.provenance import sha256_file, source_tree_sha256
from openprotein.selection import select_multimodal_checkpoint, select_sequence_checkpoint


def _validation(step, loss):
    return {
        "status": "passed",
        "evidence_level": "VALIDATION",
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "data_manifest_sha256": "data",
        "model_config_sha256": "config",
        "tokenizer_sha256": "tokenizer",
        "model_artifact": {"manifest_sha256": "artifact"},
        "validation_files": ["validation.csv"],
        "batches": 1000,
        "masked_tokens": 38449,
        "base_esmc_loss": 1.9,
        "base_esmc_perplexity": 6.7,
        "bridge_loss": loss,
        "bridge_perplexity": 6.0,
        "evaluation_code_sha256": "code",
        "evaluation_settings": {
            "max_batches": 1000,
            "sequence_length": 256,
            "mlm_probability": 0.15,
            "seed": 2701,
        },
        "checkpoint": {
            "step": step,
            "state_sha256": f"state-{step}",
            "training_provenance": {"train_config_sha256": "train"},
        },
    }


def test_sequence_selection_uses_minimum_loss_and_earliest_exact_tie(tmp_path):
    paths = []
    for step, loss in ((30000, 1.81), (10000, 1.82), (20000, 1.81)):
        path = tmp_path / f"{step}.json"
        path.write_text(json.dumps(_validation(step, loss)))
        paths.append(str(path))
    output = tmp_path / "selection.json"
    receipt = select_sequence_checkpoint(paths, str(output))
    assert receipt["selected"]["step"] == 20000
    assert [item["step"] for item in receipt["candidates"]] == [10000, 20000, 30000]
    assert receipt["test_data_accessed"] is False


def test_sequence_selection_rejects_comparability_drift_or_test_access(tmp_path):
    first = _validation(10000, 1.82)
    second = copy.deepcopy(_validation(20000, 1.81))
    second["masked_tokens"] += 1
    paths = []
    for index, payload in enumerate((first, second)):
        path = tmp_path / f"drift-{index}.json"
        path.write_text(json.dumps(payload))
        paths.append(str(path))
    with pytest.raises(RuntimeError, match="masked_tokens"):
        select_sequence_checkpoint(paths, str(tmp_path / "drift-selection.json"))

    second["masked_tokens"] = first["masked_tokens"]
    second["test_data_accessed"] = True
    PathLike = tmp_path / "test-access.json"
    PathLike.write_text(json.dumps(second))
    with pytest.raises(RuntimeError, match="test-data"):
        select_sequence_checkpoint(
            [str(paths[0]), str(PathLike)], str(tmp_path / "test-selection.json")
        )


def _policy_selection_fixture(tmp_path, steps=(10000, 20000, 30000)):
    snapshot = tmp_path / "source-snapshot"
    package = snapshot / "src" / "openprotein"
    scripts = snapshot / "scripts"
    package.mkdir(parents=True)
    scripts.mkdir()
    (package / "selection.py").write_text("# frozen selector\n")
    (package / "validation.py").write_text("# frozen evaluator\n")
    (scripts / "validate_sequence.sbatch").write_text("#!/usr/bin/env bash\n")

    validation_file = tmp_path / "validation.csv"
    validation_file.write_text("sequence\nACDE\n")
    data_manifest = tmp_path / "manifest.json"
    data_manifest.write_text("{}\n")
    artifact_manifest = tmp_path / "artifact.json"
    artifact_manifest.write_text("{}\n")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}\n")

    evaluator_hash = source_tree_sha256(package)
    policy = {
        "schema_version": 1,
        "candidate_steps": list(steps),
        "include_non_milestone_final_checkpoint": False,
        "required_pretrain_receipt": {
            "status": "passed",
            "evidence_level": "PRETRAIN",
            "target_tokens": 1_000_000_000,
            "token_budget_reached": True,
        },
        "evaluator": {
            "source_snapshot": str(snapshot),
            "source_snapshot_sha256": source_tree_sha256(snapshot),
            "evaluation_code_sha256": evaluator_hash,
            "validation_script_sha256": sha256_file(scripts / "validate_sequence.sbatch"),
            "selection_code_sha256": sha256_file(package / "selection.py"),
            "max_batches": 1000,
            "validation_glob": str(validation_file),
            "validation_file_sha256": sha256_file(validation_file),
            "data_manifest": str(data_manifest),
            "data_manifest_sha256": sha256_file(data_manifest),
            "model_artifact_manifest": str(artifact_manifest),
            "model_artifact_manifest_sha256": sha256_file(artifact_manifest),
            "protein_tokenizer": str(tokenizer),
            "protein_tokenizer_sha256": sha256_file(tokenizer),
        },
        "selection": {
            "primary_metric": "minimum_bridge_loss",
            "tie_break": "earliest_checkpoint_step",
            "exact_comparability_required": True,
        },
        "test_data_policy": "forbidden",
        "proteingym_rows_opened": False,
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    provenance = {"train_config_sha256": "packed-train"}
    pretrain = {
        "status": "passed",
        "evidence_level": "PRETRAIN",
        "target_tokens": 1_000_000_000,
        "token_budget_reached": True,
        "tokens": 1_000_000_123,
        "provenance": provenance,
    }
    pretrain_path = tmp_path / "pretrain.json"
    pretrain_path.write_text(json.dumps(pretrain))
    receipts = []
    for index, step in enumerate(steps):
        payload = _validation(step, 1.82 - index * 0.01)
        payload["checkpoint"]["training_provenance"] = provenance
        payload["evaluation_code_sha256"] = evaluator_hash
        payload["validation_files"] = [str(validation_file.resolve())]
        payload["data_manifest_sha256"] = sha256_file(data_manifest)
        payload["model_artifact"]["manifest_sha256"] = sha256_file(artifact_manifest)
        payload["tokenizer_sha256"] = sha256_file(tokenizer)
        path = tmp_path / f"validation-{step}.json"
        path.write_text(json.dumps(payload))
        receipts.append(str(path))
    for root, directories, files in os.walk(snapshot, topdown=False):
        for name in files:
            os.chmod(Path(root) / name, 0o444)
        for name in directories:
            os.chmod(Path(root) / name, 0o555)
    os.chmod(snapshot, 0o555)
    return policy_path, pretrain_path, receipts


def test_policy_bound_sequence_selection_requires_complete_candidate_set(tmp_path):
    policy, pretrain, receipts = _policy_selection_fixture(tmp_path)
    output = tmp_path / "policy-selection.json"
    selected = select_sequence_checkpoint_with_policy(
        receipts, str(pretrain), str(policy), str(output)
    )
    assert selected["selected"]["step"] == 30000
    assert selected["required_candidate_steps"] == [10000, 20000, 30000]
    assert selected["candidate_set_complete"] is True
    assert selected["pretrain_token_budget_verified"] is True
    assert selected["proteingym_rows_opened"] is False
    with pytest.raises(RuntimeError, match="complete candidate step set"):
        select_sequence_checkpoint_with_policy(
            receipts[:-1],
            str(pretrain),
            str(policy),
            str(tmp_path / "incomplete-selection.json"),
        )


def _text_validation(step, loss, ci_upper=-0.01):
    payload = _validation(step, 1.805 + step / 10_000_000)
    payload.update(
        {
            "protein_tokenizer_sha256": payload.pop("tokenizer_sha256"),
            "text_tokenizer_sha256": "text-tokenizer",
            "pairs": 1000,
            "text_target_tokens": 120000,
            "protein_masked_tokens": 19000,
            "retrieval_encoding_policy": "fixed",
            "text_lm_loss": loss,
            "text_lm_perplexity": 100.0,
            "conditioning_control": {
                "paired_better": ci_upper < 0,
                "batch_mean_normal_95_percent_interval": [-0.03, ci_upper],
                "paired_minus_shuffled_text_lm_loss": -0.02,
            },
            "evaluation_settings": {
                "max_batches": 125,
                "batch_size": 8,
                "sequence_length": 128,
                "text_length": 128,
                "mlm_probability": 0.15,
                "seed": 3701,
            },
        }
    )
    return payload


def test_multimodal_selection_applies_conditioning_and_sequence_gates(tmp_path):
    initializer = _validation(20000, 1.81)
    initializer_path = tmp_path / "initializer.json"
    initializer_path.write_text(json.dumps(initializer))
    provenance = {"train_config_sha256": "multimodal-train"}
    train = {
        "status": "passed",
        "task": "sequence_text",
        "provenance": provenance,
        "initialization_checkpoint": {
            key: initializer["checkpoint"][key]
            for key in ("step", "manifest_sha256", "state_sha256")
            if key in initializer["checkpoint"]
        },
    }
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps(train))
    policy = {
        "schema_version": 1,
        "candidate_steps": [1000, 5000, 10000],
        "primary_metric": "minimum_text_lm_loss",
        "max_sequence_loss_delta_from_initializer": 0.01,
        "require_conditioning_ci_upper_below_zero": True,
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    pairs = []
    for step, text_loss, ci_upper, sequence_loss in (
        (1000, 5.2, -0.01, 1.811),
        (5000, 4.9, 0.01, 1.812),
        (10000, 5.0, -0.01, 1.813),
    ):
        text_payload = _text_validation(step, text_loss, ci_upper)
        sequence_payload = _validation(step, sequence_loss)
        for payload in (text_payload, sequence_payload):
            payload["checkpoint"]["training_provenance"] = provenance
        text_path = tmp_path / f"text-{step}.json"
        sequence_path = tmp_path / f"sequence-{step}.json"
        text_path.write_text(json.dumps(text_payload))
        sequence_path.write_text(json.dumps(sequence_payload))
        pairs.append((str(text_path), str(sequence_path)))
    receipt = select_multimodal_checkpoint(
        pairs,
        str(train_path),
        str(initializer_path),
        str(policy_path),
        str(tmp_path / "multimodal-selection.json"),
    )
    assert receipt["selected"]["step"] == 10000
    failed = next(item for item in receipt["candidates"] if item["step"] == 5000)
    assert failed["ineligible_reasons"] == ["conditioning_ci_gate_failed"]


def _policy_bound_multimodal_fixture(tmp_path):
    snapshot = tmp_path / "source-snapshot"
    package = snapshot / "src" / "openprotein"
    scripts = snapshot / "scripts"
    package.mkdir(parents=True)
    scripts.mkdir()
    for name in ("selection.py", "validation.py"):
        (package / name).write_text(f"# frozen {name}\n")
    for name in ("validate_sequence_gate.sbatch", "validate_text_gate.sbatch"):
        (scripts / name).write_text("#!/usr/bin/env bash\n")

    sequence_validation_file = tmp_path / "sequence-validation.csv"
    sequence_validation_file.write_text("sequence\nACDE\n")
    text_validation_file = tmp_path / "text-validation.csv"
    text_validation_file.write_text("sequence,text\nACDE,example\n")
    sequence_manifest = tmp_path / "sequence-manifest.json"
    sequence_manifest.write_text("{}\n")
    text_manifest = tmp_path / "text-manifest.json"
    text_manifest.write_text("{}\n")
    artifact_manifest = tmp_path / "artifact.json"
    artifact_manifest.write_text("{}\n")
    protein_tokenizer = tmp_path / "protein-tokenizer.json"
    protein_tokenizer.write_text("{}\n")
    text_tokenizer = tmp_path / "text-tokenizer.model"
    text_tokenizer.write_text("tokenizer\n")

    upstream_checkpoint = {
        "step": 100000,
        "manifest_sha256": "upstream-manifest",
        "state_sha256": "upstream-state",
        "training_provenance": {"train_config_sha256": "packed"},
    }
    upstream = {
        "status": "passed",
        "evidence_level": "VALIDATION_SELECTION",
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "candidate_set_complete": True,
        "selected": {"checkpoint": upstream_checkpoint},
    }
    upstream_path = tmp_path / "upstream-selection.json"
    upstream_path.write_text(json.dumps(upstream))

    training_provenance = {
        "train_config_sha256": "multimodal-config",
        "training_code_sha256": source_tree_sha256(package),
    }
    train = {
        "status": "passed",
        "evidence_level": "PILOT",
        "task": "sequence_text",
        "step": 10000,
        "stopped_by": "step_limit",
        "source_tree_sha256": source_tree_sha256(snapshot),
        "provenance": training_provenance,
        "initialization_checkpoint": upstream_checkpoint,
    }
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps(train))

    evaluator_hash = source_tree_sha256(package)
    sequence_files = [str(sequence_validation_file.resolve())]
    text_files = [str(text_validation_file.resolve())]
    policy = {
        "schema_version": 2,
        "candidate_steps": [1000, 5000, 10000],
        "primary_metric": "minimum_text_lm_loss",
        "tie_break": "earliest_checkpoint_step",
        "max_sequence_loss_delta_from_initializer": 0.01,
        "require_conditioning_ci_upper_below_zero": True,
        "exact_comparability_required": True,
        "required_upstream_selection": {
            "receipt_sha256": sha256_file(upstream_path),
            "selected_checkpoint": {
                key: upstream_checkpoint[key]
                for key in ("step", "manifest_sha256", "state_sha256")
            },
        },
        "required_train_receipt": {
            "status": "passed",
            "evidence_level": "PILOT",
            "task": "sequence_text",
            "step": 10000,
            "stopped_by": "step_limit",
            "source_tree_sha256": source_tree_sha256(snapshot),
        },
        "required_train_provenance": training_provenance,
        "evaluator": {
            "source_snapshot": str(snapshot),
            "source_snapshot_sha256": source_tree_sha256(snapshot),
            "evaluation_code_sha256": evaluator_hash,
            "validation_script_sha256": {
                name: sha256_file(scripts / name)
                for name in ("validate_sequence_gate.sbatch", "validate_text_gate.sbatch")
            },
            "sequence_max_batches": 1000,
            "text_max_batches": 125,
            "sequence_evaluation_settings": {
                "max_batches": 1000,
                "sequence_length": 256,
                "mlm_probability": 0.15,
                "seed": 2701,
            },
            "text_evaluation_settings": {
                "max_batches": 125,
                "batch_size": 8,
                "sequence_length": 128,
                "text_length": 128,
                "mlm_probability": 0.15,
                "seed": 3701,
            },
            "sequence_validation_glob": str(sequence_validation_file),
            "sequence_validation_files": sequence_files,
            "sequence_validation_file_sha256": [sha256_file(sequence_validation_file)],
            "sequence_data_manifest": str(sequence_manifest),
            "sequence_data_manifest_sha256": sha256_file(sequence_manifest),
            "text_validation_glob": str(text_validation_file),
            "text_validation_files": text_files,
            "text_validation_file_sha256": [sha256_file(text_validation_file)],
            "text_data_manifest": str(text_manifest),
            "text_data_manifest_sha256": sha256_file(text_manifest),
            "model_artifact_manifest": str(artifact_manifest),
            "model_artifact_manifest_sha256": sha256_file(artifact_manifest),
            "protein_tokenizer": str(protein_tokenizer),
            "protein_tokenizer_sha256": sha256_file(protein_tokenizer),
            "text_tokenizer": str(text_tokenizer),
            "text_tokenizer_sha256": sha256_file(text_tokenizer),
        },
        "test_data_policy": "forbidden",
        "proteingym_rows_opened": False,
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))

    def bind_sequence(payload):
        payload["evaluation_code_sha256"] = evaluator_hash
        payload["validation_files"] = sequence_files
        payload["data_manifest_sha256"] = sha256_file(sequence_manifest)
        payload["model_artifact"]["manifest_sha256"] = sha256_file(artifact_manifest)
        payload["tokenizer_sha256"] = sha256_file(protein_tokenizer)
        return payload

    initializer = bind_sequence(_validation(100000, 1.81))
    initializer["checkpoint"] = upstream_checkpoint
    initializer_path = tmp_path / "initializer.json"
    initializer_path.write_text(json.dumps(initializer))

    pairs = []
    for step, text_loss in ((1000, 4.5), (5000, 4.0), (10000, 3.8)):
        sequence = bind_sequence(_validation(step, 1.811))
        sequence["checkpoint"]["training_provenance"] = training_provenance
        text = _text_validation(step, text_loss)
        text["checkpoint"]["training_provenance"] = training_provenance
        text["evaluation_code_sha256"] = evaluator_hash
        text["validation_files"] = text_files
        text["data_manifest_sha256"] = sha256_file(text_manifest)
        text["model_artifact"]["manifest_sha256"] = sha256_file(artifact_manifest)
        text["protein_tokenizer_sha256"] = sha256_file(protein_tokenizer)
        text["text_tokenizer_sha256"] = sha256_file(text_tokenizer)
        text["batches"] = 125
        text_path = tmp_path / f"text-{step}.json"
        sequence_path = tmp_path / f"sequence-{step}.json"
        text_path.write_text(json.dumps(text))
        sequence_path.write_text(json.dumps(sequence))
        pairs.append((str(text_path), str(sequence_path)))

    for root, directories, files in os.walk(snapshot, topdown=False):
        for name in files:
            os.chmod(Path(root) / name, 0o444)
        for name in directories:
            os.chmod(Path(root) / name, 0o555)
    os.chmod(snapshot, 0o555)
    return policy_path, train_path, initializer_path, upstream_path, pairs


def test_policy_bound_multimodal_selection_binds_upstream_and_evaluators(tmp_path):
    policy, train, initializer, upstream, pairs = _policy_bound_multimodal_fixture(tmp_path)
    output = tmp_path / "selection.json"
    receipt = select_multimodal_checkpoint_with_policy(
        pairs,
        str(train),
        str(initializer),
        str(upstream),
        str(policy),
        str(output),
    )
    assert receipt["selected"]["step"] == 10000
    assert receipt["candidate_set_complete"] is True
    assert receipt["exact_comparability_verified"] is True
    assert receipt["proteingym_rows_opened"] is False


def test_multimodal_selection_rejects_incomparable_initializer(tmp_path):
    initializer = _validation(20000, 1.81)
    initializer["masked_tokens"] += 1
    initializer_path = tmp_path / "initializer.json"
    initializer_path.write_text(json.dumps(initializer))
    provenance = {"train_config_sha256": "multimodal-train"}
    train = {
        "status": "passed",
        "task": "sequence_text",
        "provenance": provenance,
        "initialization_checkpoint": {
            key: initializer["checkpoint"][key]
            for key in ("step", "manifest_sha256", "state_sha256")
            if key in initializer["checkpoint"]
        },
    }
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps(train))
    policy = {
        "schema_version": 1,
        "candidate_steps": [1000],
        "primary_metric": "minimum_text_lm_loss",
        "max_sequence_loss_delta_from_initializer": 0.01,
        "require_conditioning_ci_upper_below_zero": True,
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    text = _text_validation(1000, 4.5)
    sequence = _validation(1000, 1.811)
    for payload in (text, sequence):
        payload["checkpoint"]["training_provenance"] = provenance
    text_path = tmp_path / "text.json"
    sequence_path = tmp_path / "sequence.json"
    text_path.write_text(json.dumps(text))
    sequence_path.write_text(json.dumps(sequence))
    with pytest.raises(RuntimeError, match="initializer and candidate"):
        select_multimodal_checkpoint(
            [(str(text_path), str(sequence_path))],
            str(train_path),
            str(initializer_path),
            str(policy_path),
            str(tmp_path / "selection.json"),
        )
