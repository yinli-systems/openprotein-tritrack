"""Policy-bound fail-closed checkpoint selection."""

from __future__ import annotations

import glob
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from openprotein.provenance import sha256_file, source_tree_sha256
from openprotein.selection import select_multimodal_checkpoint, select_sequence_checkpoint


def _atomic_json_no_overwrite(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite policy-bound selection receipt: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"policy mismatch for {label}: {actual!r} != {expected!r}")


def select_sequence_checkpoint_with_policy(
    validation_receipts: Iterable[str],
    pretrain_receipt: str,
    policy_path: str,
    output: str,
) -> Dict[str, Any]:
    """Require a complete predeclared candidate set before sequence selection."""

    policy_file = Path(policy_path).resolve()
    policy = json.loads(policy_file.read_text(encoding="utf-8"))
    if int(policy.get("schema_version", 0)) != 1:
        raise RuntimeError("unsupported sequence-selection policy schema")
    if policy.get("test_data_policy") != "forbidden":
        raise RuntimeError("sequence-selection policy must forbid test data")
    if policy.get("proteingym_rows_opened") is not False:
        raise RuntimeError("sequence-selection policy must bind unopened ProteinGym rows")
    if policy.get("include_non_milestone_final_checkpoint") is not False:
        raise RuntimeError("unsupported non-milestone final-checkpoint policy")

    expected_steps = [int(value) for value in policy.get("candidate_steps", [])]
    if not expected_steps or len(expected_steps) != len(set(expected_steps)):
        raise RuntimeError("policy candidate_steps must be unique and nonempty")
    if expected_steps != sorted(expected_steps):
        raise RuntimeError("policy candidate_steps must be strictly sorted")
    selection_policy = policy.get("selection", {})
    _require_equal(selection_policy.get("primary_metric"), "minimum_bridge_loss", "primary metric")
    _require_equal(selection_policy.get("tie_break"), "earliest_checkpoint_step", "tie break")
    _require_equal(
        selection_policy.get("exact_comparability_required"), True, "exact comparability"
    )

    train_path = Path(pretrain_receipt).resolve()
    train = json.loads(train_path.read_text(encoding="utf-8"))
    required_train = policy.get("required_pretrain_receipt", {})
    for key in ("status", "evidence_level", "target_tokens", "token_budget_reached"):
        _require_equal(train.get(key), required_train.get(key), f"pretrain receipt {key}")
    if not isinstance(train.get("provenance"), dict):
        raise RuntimeError("pretrain receipt has no structured provenance")
    if int(train.get("tokens", -1)) < int(required_train["target_tokens"]):
        raise RuntimeError("pretrain receipt token count is below the policy target")

    evaluator = policy.get("evaluator", {})
    source_snapshot = Path(evaluator["source_snapshot"]).resolve()
    if not source_snapshot.is_dir() or os.access(source_snapshot, os.W_OK):
        raise RuntimeError("policy evaluator source snapshot is missing or writable")
    _require_equal(
        source_tree_sha256(source_snapshot),
        evaluator.get("source_snapshot_sha256"),
        "evaluator source snapshot hash",
    )
    package_root = source_snapshot / "src" / "openprotein"
    _require_equal(
        source_tree_sha256(package_root),
        evaluator.get("evaluation_code_sha256"),
        "evaluation package hash",
    )
    _require_equal(
        sha256_file(source_snapshot / "scripts" / "validate_sequence.sbatch"),
        evaluator.get("validation_script_sha256"),
        "validation script hash",
    )
    _require_equal(
        sha256_file(source_snapshot / "src" / "openprotein" / "selection.py"),
        evaluator.get("selection_code_sha256"),
        "base selection code hash",
    )

    expected_validation_files = sorted(
        str(Path(value).resolve()) for value in glob.glob(evaluator["validation_glob"])
    )
    if len(expected_validation_files) != 1:
        raise RuntimeError("policy validation glob must resolve to exactly one file")
    _require_equal(
        sha256_file(Path(expected_validation_files[0])),
        evaluator.get("validation_file_sha256"),
        "validation file hash",
    )
    _require_equal(
        sha256_file(Path(evaluator["data_manifest"])),
        evaluator.get("data_manifest_sha256"),
        "validation data manifest hash",
    )
    _require_equal(
        sha256_file(Path(evaluator["model_artifact_manifest"])),
        evaluator.get("model_artifact_manifest_sha256"),
        "model artifact manifest hash",
    )
    _require_equal(
        sha256_file(Path(evaluator["protein_tokenizer"])),
        evaluator.get("protein_tokenizer_sha256"),
        "protein tokenizer hash",
    )

    paths = [Path(value).resolve() for value in validation_receipts]
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    actual_steps = sorted(int(record.get("checkpoint", {}).get("step", -1)) for record in records)
    _require_equal(actual_steps, expected_steps, "complete candidate step set")
    if len(paths) != len(set(paths)):
        raise RuntimeError("duplicate validation receipt paths")
    for path, record in zip(paths, records):
        if record.get("status") != "passed" or record.get("evidence_level") != "VALIDATION":
            raise RuntimeError(f"not a passed validation receipt: {path}")
        if record.get("test_data_accessed") is not False:
            raise RuntimeError(f"validation receipt accessed or omitted test-data status: {path}")
        checkpoint = record.get("checkpoint", {})
        _require_equal(
            checkpoint.get("training_provenance"), train["provenance"], f"{path} provenance"
        )
        _require_equal(
            record.get("evaluation_code_sha256"),
            evaluator["evaluation_code_sha256"],
            f"{path} evaluation code",
        )
        _require_equal(record.get("batches"), int(evaluator["max_batches"]), f"{path} batches")
        _require_equal(
            record.get("validation_files"), expected_validation_files, f"{path} validation files"
        )
        _require_equal(
            record.get("data_manifest_sha256"),
            evaluator["data_manifest_sha256"],
            f"{path} data manifest",
        )
        _require_equal(
            record.get("model_artifact", {}).get("manifest_sha256"),
            evaluator["model_artifact_manifest_sha256"],
            f"{path} model artifact",
        )
        _require_equal(
            record.get("tokenizer_sha256"),
            evaluator["protein_tokenizer_sha256"],
            f"{path} tokenizer",
        )

    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite policy-bound selection receipt: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as temporary_directory:
        base_output = Path(temporary_directory) / "base-selection.json"
        receipt = select_sequence_checkpoint([str(path) for path in paths], str(base_output))
    receipt.update(
        {
            "policy_path_at_selection": str(policy_file),
            "policy_sha256": sha256_file(policy_file),
            "pretrain_receipt_path_at_selection": str(train_path),
            "pretrain_receipt_sha256": sha256_file(train_path),
            "required_candidate_steps": expected_steps,
            "candidate_set_complete": True,
            "pretrain_token_budget_verified": True,
            "proteingym_rows_opened": False,
            "claim_boundary": (
                "Policy-bound validation-only checkpoint selection for the frozen-backbone "
                "adapter control; not scratch-7B or frozen-benchmark evidence."
            ),
        }
    )
    _atomic_json_no_overwrite(output_path, receipt)
    return receipt


def _checkpoint_identity(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: payload.get(key)
        for key in ("step", "manifest_sha256", "state_sha256")
    }


def _verify_validation_contract(
    payload: Dict[str, Any],
    path: Path,
    evaluator: Dict[str, Any],
    *,
    text: bool,
) -> None:
    if payload.get("status") != "passed" or payload.get("evidence_level") != "VALIDATION":
        raise RuntimeError(f"not a passed validation receipt: {path}")
    if payload.get("test_data_accessed") is not False:
        raise RuntimeError(f"validation receipt accessed or omitted test-data status: {path}")
    if payload.get("proteingym_rows_opened") is not False:
        raise RuntimeError(f"validation receipt opened or omitted ProteinGym status: {path}")
    prefix = "text" if text else "sequence"
    _require_equal(
        payload.get("evaluation_code_sha256"),
        evaluator["evaluation_code_sha256"],
        f"{path} evaluation code",
    )
    _require_equal(
        payload.get("batches"), int(evaluator[f"{prefix}_max_batches"]), f"{path} batches"
    )
    _require_equal(
        payload.get("evaluation_settings"),
        evaluator[f"{prefix}_evaluation_settings"],
        f"{path} evaluation settings",
    )
    _require_equal(
        payload.get("validation_files"),
        evaluator[f"{prefix}_validation_files"],
        f"{path} validation files",
    )
    _require_equal(
        payload.get("data_manifest_sha256"),
        evaluator[f"{prefix}_data_manifest_sha256"],
        f"{path} data manifest",
    )
    _require_equal(
        payload.get("model_artifact", {}).get("manifest_sha256"),
        evaluator["model_artifact_manifest_sha256"],
        f"{path} model artifact",
    )
    tokenizer_key = "protein_tokenizer_sha256" if text else "tokenizer_sha256"
    _require_equal(
        payload.get(tokenizer_key),
        evaluator["protein_tokenizer_sha256"],
        f"{path} protein tokenizer",
    )
    if text:
        _require_equal(
            payload.get("text_tokenizer_sha256"),
            evaluator["text_tokenizer_sha256"],
            f"{path} text tokenizer",
        )


def select_multimodal_checkpoint_with_policy(
    candidate_receipts: Iterable[Tuple[str, str]],
    train_receipt: str,
    initializer_validation_receipt: str,
    upstream_selection_receipt: str,
    policy_path: str,
    output: str,
) -> Dict[str, Any]:
    """Select a sequence-text checkpoint under a frozen end-to-end policy."""

    policy_file = Path(policy_path).resolve()
    policy = json.loads(policy_file.read_text(encoding="utf-8"))
    if int(policy.get("schema_version", 0)) != 2:
        raise RuntimeError("unsupported multimodal-selection policy schema")
    if policy.get("test_data_policy") != "forbidden":
        raise RuntimeError("multimodal-selection policy must forbid test data")
    if policy.get("proteingym_rows_opened") is not False:
        raise RuntimeError("multimodal-selection policy must bind unopened ProteinGym rows")

    expected_steps = [int(value) for value in policy.get("candidate_steps", [])]
    if not expected_steps or expected_steps != sorted(set(expected_steps)):
        raise RuntimeError("policy candidate_steps must be strictly sorted and unique")
    _require_equal(policy.get("primary_metric"), "minimum_text_lm_loss", "primary metric")
    _require_equal(policy.get("tie_break"), "earliest_checkpoint_step", "tie break")
    _require_equal(
        policy.get("require_conditioning_ci_upper_below_zero"), True, "conditioning gate"
    )
    _require_equal(
        policy.get("exact_comparability_required"), True, "exact comparability"
    )

    upstream_path = Path(upstream_selection_receipt).resolve()
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    required_upstream = policy["required_upstream_selection"]
    _require_equal(
        sha256_file(upstream_path), required_upstream["receipt_sha256"], "upstream receipt hash"
    )
    _require_equal(upstream.get("status"), "passed", "upstream status")
    _require_equal(upstream.get("evidence_level"), "VALIDATION_SELECTION", "upstream evidence")
    _require_equal(upstream.get("test_data_accessed"), False, "upstream test access")
    _require_equal(upstream.get("proteingym_rows_opened"), False, "upstream ProteinGym access")
    _require_equal(upstream.get("candidate_set_complete"), True, "upstream candidate set")
    upstream_checkpoint = upstream.get("selected", {}).get("checkpoint", {})
    _require_equal(
        _checkpoint_identity(upstream_checkpoint),
        required_upstream["selected_checkpoint"],
        "upstream selected checkpoint",
    )

    train_path = Path(train_receipt).resolve()
    train = json.loads(train_path.read_text(encoding="utf-8"))
    for key, expected in policy["required_train_receipt"].items():
        _require_equal(train.get(key), expected, f"train receipt {key}")
    if not isinstance(train.get("provenance"), dict):
        raise RuntimeError("sequence-text train receipt has no structured provenance")
    for key, expected in policy["required_train_provenance"].items():
        _require_equal(train["provenance"].get(key), expected, f"train provenance {key}")
    _require_equal(
        _checkpoint_identity(train.get("initialization_checkpoint", {})),
        required_upstream["selected_checkpoint"],
        "training initialization",
    )

    evaluator = policy["evaluator"]
    source_snapshot = Path(evaluator["source_snapshot"]).resolve()
    if not source_snapshot.is_dir() or os.access(source_snapshot, os.W_OK):
        raise RuntimeError("policy evaluator source snapshot is missing or writable")
    _require_equal(
        source_tree_sha256(source_snapshot),
        evaluator["source_snapshot_sha256"],
        "evaluator source snapshot hash",
    )
    package_root = source_snapshot / "src" / "openprotein"
    _require_equal(
        source_tree_sha256(package_root),
        evaluator["evaluation_code_sha256"],
        "evaluation package hash",
    )
    for script_name, expected_hash in evaluator["validation_script_sha256"].items():
        _require_equal(
            sha256_file(source_snapshot / "scripts" / script_name),
            expected_hash,
            f"validation script {script_name}",
        )

    artifact_paths = {
        "sequence_data_manifest_sha256": evaluator["sequence_data_manifest"],
        "text_data_manifest_sha256": evaluator["text_data_manifest"],
        "model_artifact_manifest_sha256": evaluator["model_artifact_manifest"],
        "protein_tokenizer_sha256": evaluator["protein_tokenizer"],
        "text_tokenizer_sha256": evaluator["text_tokenizer"],
    }
    for hash_key, artifact_path in artifact_paths.items():
        _require_equal(
            sha256_file(Path(artifact_path)), evaluator[hash_key], f"evaluator {hash_key}"
        )
    for prefix in ("sequence", "text"):
        paths = sorted(str(Path(value).resolve()) for value in glob.glob(evaluator[f"{prefix}_validation_glob"]))
        _require_equal(paths, evaluator[f"{prefix}_validation_files"], f"{prefix} validation files")
        expected_hashes = evaluator[f"{prefix}_validation_file_sha256"]
        _require_equal(
            [sha256_file(Path(path)) for path in paths],
            expected_hashes,
            f"{prefix} validation file hashes",
        )

    initializer_path = Path(initializer_validation_receipt).resolve()
    initializer = json.loads(initializer_path.read_text(encoding="utf-8"))
    _verify_validation_contract(initializer, initializer_path, evaluator, text=False)
    _require_equal(
        _checkpoint_identity(initializer.get("checkpoint", {})),
        required_upstream["selected_checkpoint"],
        "initializer validation checkpoint",
    )

    pairs = [(Path(text).resolve(), Path(sequence).resolve()) for text, sequence in candidate_receipts]
    if len(pairs) != len({(text, sequence) for text, sequence in pairs}):
        raise RuntimeError("duplicate multimodal candidate receipt paths")
    actual_steps = []
    for text_path, sequence_path in pairs:
        text_payload = json.loads(text_path.read_text(encoding="utf-8"))
        sequence_payload = json.loads(sequence_path.read_text(encoding="utf-8"))
        _verify_validation_contract(text_payload, text_path, evaluator, text=True)
        _verify_validation_contract(sequence_payload, sequence_path, evaluator, text=False)
        _require_equal(
            text_payload.get("checkpoint", {}).get("training_provenance"),
            train["provenance"],
            f"{text_path} provenance",
        )
        _require_equal(
            sequence_payload.get("checkpoint", {}).get("training_provenance"),
            train["provenance"],
            f"{sequence_path} provenance",
        )
        actual_steps.append(int(text_payload.get("checkpoint", {}).get("step", -1)))
    _require_equal(sorted(actual_steps), expected_steps, "complete multimodal candidate set")

    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite policy-bound selection receipt: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as temporary_directory:
        base_output = Path(temporary_directory) / "base-multimodal-selection.json"
        receipt = select_multimodal_checkpoint(
            [(str(text), str(sequence)) for text, sequence in pairs],
            str(train_path),
            str(initializer_path),
            str(policy_file),
            str(base_output),
        )
    receipt.update(
        {
            "policy_path_at_selection": str(policy_file),
            "policy_sha256": sha256_file(policy_file),
            "upstream_selection_receipt_path_at_selection": str(upstream_path),
            "upstream_selection_receipt_sha256": sha256_file(upstream_path),
            "train_receipt_path_at_selection": str(train_path),
            "train_receipt_sha256": sha256_file(train_path),
            "required_candidate_steps": expected_steps,
            "candidate_set_complete": True,
            "exact_comparability_verified": True,
            "proteingym_rows_opened": False,
            "claim_boundary": (
                "Policy-bound validation-only selection for the frozen-ESMC adapter-control "
                "sequence-text continuation; not scratch-7B or frozen-benchmark evidence."
            ),
        }
    )
    _atomic_json_no_overwrite(output_path, receipt)
    return receipt
