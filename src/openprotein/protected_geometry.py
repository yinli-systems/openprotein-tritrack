"""Fail-closed finalization of a protected geometry-head continuation."""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from openprotein.provenance import sha256_file, source_tree_sha256


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    return {key: checkpoint.get(key) for key in ("step", "manifest_sha256", "state_sha256")}


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"geometry policy mismatch for {label}: {actual!r} != {expected!r}")


def _require_validation(payload: Dict[str, Any], path: Path) -> None:
    if payload.get("status") != "passed" or payload.get("evidence_level") != "VALIDATION":
        raise RuntimeError(f"not a passed validation receipt: {path}")
    if payload.get("test_data_accessed") is not False:
        raise RuntimeError(f"validation accessed or omitted test-data status: {path}")
    if payload.get("proteingym_rows_opened") is not False:
        raise RuntimeError(f"validation opened or omitted ProteinGym status: {path}")


def _require_finite(payload: Dict[str, Any], keys: Iterable[str], label: str) -> None:
    for key in keys:
        value = payload.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"{label} has invalid {key}")


def _atomic_json_no_overwrite(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite protected-geometry receipt: {path}")
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


def finalize_protected_geometry(
    *,
    policy_path: str,
    upstream_selection_receipt: str,
    train_receipt: str,
    initializer_sequence_receipt: str,
    initializer_text_receipt: str,
    initializer_structure_receipt: str,
    final_sequence_receipt: str,
    final_text_receipt: str,
    final_structure_receipt: str,
    checkpoint_audit_receipt: str,
    output: str,
) -> Dict[str, Any]:
    """Require geometry improvement and bitwise protection of every shared tensor."""

    policy_file = Path(policy_path).resolve()
    policy = _load(policy_file)
    if int(policy.get("schema_version", 0)) != 1:
        raise RuntimeError("unsupported protected-geometry policy schema")
    _require_equal(policy.get("test_data_policy"), "forbidden", "test data policy")
    _require_equal(policy.get("proteingym_rows_opened"), False, "ProteinGym access")
    _require_equal(policy.get("allowed_changed_prefixes"), ["geometry_head."], "changed prefixes")

    upstream_path = Path(upstream_selection_receipt).resolve()
    upstream = _load(upstream_path)
    required_upstream = policy["required_upstream_selection"]
    _require_equal(sha256_file(upstream_path), required_upstream["receipt_sha256"], "upstream hash")
    _require_equal(upstream.get("status"), "passed", "upstream status")
    _require_equal(upstream.get("evidence_level"), "VALIDATION_SELECTION", "upstream evidence")
    _require_equal(upstream.get("candidate_set_complete"), True, "upstream candidate set")
    _require_equal(upstream.get("test_data_accessed"), False, "upstream test access")
    _require_equal(upstream.get("proteingym_rows_opened"), False, "upstream ProteinGym access")
    initializer_identity = _identity(upstream.get("selected", {}).get("checkpoint", {}))
    _require_equal(
        initializer_identity, required_upstream["selected_checkpoint"], "selected initializer"
    )

    train_path = Path(train_receipt).resolve()
    train = _load(train_path)
    for key, expected in policy["required_train_receipt"].items():
        _require_equal(train.get(key), expected, f"train receipt {key}")
    for key, expected in policy["required_train_provenance"].items():
        _require_equal(train.get("provenance", {}).get(key), expected, f"train provenance {key}")
    _require_equal(
        _identity(train.get("initialization_checkpoint", {})), initializer_identity, "train initializer"
    )

    receipt_paths = {
        "initializer_sequence": Path(initializer_sequence_receipt).resolve(),
        "initializer_text": Path(initializer_text_receipt).resolve(),
        "initializer_structure": Path(initializer_structure_receipt).resolve(),
        "final_sequence": Path(final_sequence_receipt).resolve(),
        "final_text": Path(final_text_receipt).resolve(),
        "final_structure": Path(final_structure_receipt).resolve(),
    }
    receipts = {name: _load(path) for name, path in receipt_paths.items()}
    for name, payload in receipts.items():
        _require_validation(payload, receipt_paths[name])

    for name in ("initializer_sequence", "initializer_text", "initializer_structure"):
        _require_equal(
            _identity(receipts[name].get("checkpoint", {})), initializer_identity, f"{name} checkpoint"
        )
    target_identity = _identity(receipts["final_structure"].get("checkpoint", {}))
    for name in ("final_sequence", "final_text"):
        _require_equal(
            _identity(receipts[name].get("checkpoint", {})), target_identity, f"{name} checkpoint"
        )
    _require_equal(target_identity["step"], int(policy["target_step"]), "target step")
    for name in ("final_sequence", "final_text", "final_structure"):
        _require_equal(
            receipts[name].get("checkpoint", {}).get("training_provenance"),
            train.get("provenance"),
            f"{name} training provenance",
        )

    evaluator = policy["evaluator"]
    source_snapshot = Path(evaluator["source_snapshot"]).resolve()
    if not source_snapshot.is_dir() or os.access(source_snapshot, os.W_OK):
        raise RuntimeError("geometry evaluator source snapshot is missing or writable")
    _require_equal(source_tree_sha256(source_snapshot), evaluator["source_snapshot_sha256"], "source hash")
    _require_equal(
        source_tree_sha256(source_snapshot / "src/openprotein"),
        evaluator["evaluation_code_sha256"],
        "evaluator package hash",
    )
    for script_name, expected_hash in evaluator["script_sha256"].items():
        _require_equal(
            sha256_file(source_snapshot / "scripts" / script_name),
            expected_hash,
            f"geometry script {script_name}",
        )
    for name, payload in receipts.items():
        _require_equal(payload.get("evaluation_code_sha256"), evaluator["evaluation_code_sha256"], f"{name} evaluator")
        kind = "structure" if "structure" in name else ("text" if "text" in name else "sequence")
        _require_equal(payload.get("evaluation_settings"), evaluator[f"{kind}_settings"], f"{name} settings")
        for key, expected in evaluator[f"{kind}_receipt_contract"].items():
            actual = (
                payload.get("model_artifact", {}).get("manifest_sha256")
                if key == "model_artifact_manifest_sha256"
                else payload.get(key)
            )
            _require_equal(actual, expected, f"{name} contract {key}")

    sequence_fields = (
        "data_manifest_sha256", "model_config_sha256", "tokenizer_sha256", "model_artifact",
        "validation_files", "batches", "masked_tokens", "base_esmc_loss",
        "base_esmc_perplexity", "bridge_loss", "bridge_perplexity", "bridge_minus_base_loss",
        "evaluation_settings",
    )
    text_fields = (
        "data_manifest_sha256", "model_config_sha256", "protein_tokenizer_sha256",
        "text_tokenizer_sha256", "model_artifact", "validation_files", "batches", "pairs",
        "text_target_tokens", "protein_masked_tokens", "protein_mlm_loss", "text_lm_loss",
        "text_lm_perplexity", "conditioning_control", "retrieval", "retrieval_encoding_policy",
        "evaluation_settings",
    )
    for key in sequence_fields:
        _require_equal(receipts["final_sequence"].get(key), receipts["initializer_sequence"].get(key), f"protected sequence {key}")
    for key in text_fields:
        _require_equal(receipts["final_text"].get(key), receipts["initializer_text"].get(key), f"protected text {key}")

    structure_comparability = (
        "data_manifest_sha256", "structure_manifest_sha256", "structure_database_sha256",
        "model_config_sha256", "protein_tokenizer_sha256", "model_artifact", "validation_files",
        "batches", "protein_masked_tokens", "geometry_targets", "distance_bins",
        "minimum_plddt", "uniform_reference_loss", "uniform_reference_accuracy",
        "protein_mlm_loss", "protein_mlm_perplexity", "evaluation_settings",
    )
    for key in structure_comparability:
        _require_equal(receipts["final_structure"].get(key), receipts["initializer_structure"].get(key), f"structure comparability {key}")
    for name in ("initializer_structure", "final_structure"):
        _require_finite(receipts[name], ("geometry_loss", "geometry_accuracy", "protein_mlm_loss"), name)
    baseline_structure = receipts["initializer_structure"]
    final_structure = receipts["final_structure"]
    loss_improvement = float(baseline_structure["geometry_loss"]) - float(final_structure["geometry_loss"])
    accuracy_improvement = float(final_structure["geometry_accuracy"]) - float(baseline_structure["geometry_accuracy"])
    if loss_improvement < float(policy["minimum_geometry_loss_improvement"]):
        raise RuntimeError("geometry loss improvement gate failed")
    if accuracy_improvement < float(policy["minimum_geometry_accuracy_improvement"]):
        raise RuntimeError("geometry accuracy improvement gate failed")
    if float(final_structure["geometry_loss"]) >= float(final_structure["uniform_reference_loss"]):
        raise RuntimeError("final geometry loss does not beat uniform reference")
    if float(final_structure["geometry_accuracy"]) <= float(final_structure["uniform_reference_accuracy"]):
        raise RuntimeError("final geometry accuracy does not beat uniform reference")

    audit_path = Path(checkpoint_audit_receipt).resolve()
    audit = _load(audit_path)
    _require_equal(audit.get("status"), "passed", "checkpoint audit status")
    _require_equal(audit.get("evidence_level"), "CHECKPOINT_AUDIT", "checkpoint audit evidence")
    _require_equal(audit.get("test_data_accessed"), False, "checkpoint audit test access")
    _require_equal(audit.get("proteingym_rows_opened"), False, "checkpoint audit ProteinGym access")
    _require_equal(audit.get("allowed_changed_prefixes"), ["geometry_head."], "audit prefixes")
    _require_equal(audit.get("changed_tensors"), policy["expected_changed_tensors"], "changed tensors")
    _require_equal(
        len(audit.get("changed_tensors", [])),
        int(policy["expected_changed_tensors_count"]),
        "changed tensor count",
    )
    _require_equal(audit.get("unchanged_tensors"), int(policy["expected_unchanged_tensors_count"]), "unchanged tensor count")
    _require_equal(_identity(audit.get("source_checkpoint", {})), initializer_identity, "audit source")
    _require_equal(_identity(audit.get("target_checkpoint", {})), target_identity, "audit target")
    _require_equal(audit.get("changed_parameters"), int(policy["expected_changed_parameters"]), "changed parameter count")
    _require_equal(audit.get("unchanged_parameters"), int(policy["expected_unchanged_parameters"]), "unchanged parameter count")

    output_path = Path(output).resolve()
    result = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": "PROTECTED_GEOMETRY_GATE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_path_at_finalization": str(policy_file),
        "policy_sha256": sha256_file(policy_file),
        "upstream_selection_receipt_sha256": sha256_file(upstream_path),
        "train_receipt_sha256": sha256_file(train_path),
        "validation_receipt_sha256": {
            name: sha256_file(path) for name, path in receipt_paths.items()
        },
        "checkpoint_audit_receipt_sha256": sha256_file(audit_path),
        "initializer_checkpoint": initializer_identity,
        "selected_checkpoint": target_identity,
        "geometry_loss": final_structure["geometry_loss"],
        "geometry_accuracy": final_structure["geometry_accuracy"],
        "geometry_loss_improvement": loss_improvement,
        "geometry_accuracy_improvement": accuracy_improvement,
        "changed_parameters": audit["changed_parameters"],
        "unchanged_parameters": audit["unchanged_parameters"],
        "shared_sequence_text_path_bitwise_preserved": True,
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "claim_boundary": (
            "Protected predicted-geometry head gate over a frozen sequence-text representation; "
            "not experimental structure validation, scratch-7B evidence, or a frozen benchmark."
        ),
    }
    _atomic_json_no_overwrite(output_path, result)
    return result
