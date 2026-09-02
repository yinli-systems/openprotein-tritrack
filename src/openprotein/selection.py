"""Fail-closed validation-only checkpoint selection."""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from openprotein.provenance import sha256_file, source_tree_sha256


_COMPARABILITY_FIELDS = (
    "data_manifest_sha256",
    "model_config_sha256",
    "tokenizer_sha256",
    "model_artifact",
    "validation_files",
    "batches",
    "masked_tokens",
    "base_esmc_loss",
    "base_esmc_perplexity",
    "evaluation_code_sha256",
    "evaluation_settings",
    "proteingym_rows_opened",
)


def _atomic_json_no_overwrite(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite selection receipt: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def select_sequence_checkpoint(
    validation_receipts: Iterable[str], output: str
) -> Dict[str, Any]:
    """Select minimum validation MLM loss under an exact comparability contract."""

    paths = [Path(value).resolve() for value in validation_receipts]
    if not paths:
        raise ValueError("at least one validation receipt is required")
    records = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "passed" or payload.get("evidence_level") != "VALIDATION":
            raise RuntimeError(f"not a passed validation receipt: {path}")
        if payload.get("test_data_accessed") is not False:
            raise RuntimeError(f"selection receipt accessed or omitted test-data status: {path}")
        checkpoint = payload.get("checkpoint")
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("training_provenance"), dict
        ):
            raise RuntimeError(f"validation receipt has no checkpoint provenance: {path}")
        for key in ("bridge_loss", "bridge_perplexity", "base_esmc_loss"):
            value = payload.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError(f"validation receipt has invalid {key}: {path}")
        records.append(
            {
                "path": path,
                "sha256": sha256_file(path),
                "payload": payload,
                "step": int(checkpoint["step"]),
            }
        )

    reference = records[0]["payload"]
    reference_provenance = reference["checkpoint"]["training_provenance"]
    for record in records[1:]:
        payload = record["payload"]
        drift = [key for key in _COMPARABILITY_FIELDS if payload.get(key) != reference.get(key)]
        if payload["checkpoint"]["training_provenance"] != reference_provenance:
            drift.append("checkpoint.training_provenance")
        if drift:
            raise RuntimeError(
                f"validation receipts are not exactly comparable: {record['path']}: "
                + ", ".join(drift)
            )
    steps = [record["step"] for record in records]
    if len(steps) != len(set(steps)):
        raise RuntimeError("validation receipts contain duplicate checkpoint steps")

    records.sort(key=lambda record: record["step"])
    selected = min(records, key=lambda record: (record["payload"]["bridge_loss"], record["step"]))
    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": "VALIDATION_SELECTION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_code_sha256": source_tree_sha256(Path(__file__).resolve().parent),
        "policy": "minimum bridge_loss; earliest checkpoint step breaks an exact tie",
        "comparability_fields": list(_COMPARABILITY_FIELDS)
        + ["checkpoint.training_provenance"],
        "test_data_accessed": False,
        "candidates": [
            {
                "step": record["step"],
                "bridge_loss": record["payload"]["bridge_loss"],
                "bridge_perplexity": record["payload"]["bridge_perplexity"],
                "base_esmc_loss": record["payload"]["base_esmc_loss"],
                "checkpoint": record["payload"]["checkpoint"],
                "validation_receipt_path_at_selection": str(record["path"]),
                "validation_receipt_sha256": record["sha256"],
            }
            for record in records
        ],
        "selected": {
            "step": selected["step"],
            "bridge_loss": selected["payload"]["bridge_loss"],
            "bridge_perplexity": selected["payload"]["bridge_perplexity"],
            "checkpoint": selected["payload"]["checkpoint"],
            "validation_receipt_sha256": selected["sha256"],
        },
        "claim_boundary": (
            "Validation-only checkpoint selection; this is not a frozen benchmark or a "
            "superiority claim."
        ),
    }
    _atomic_json_no_overwrite(Path(output).resolve(), receipt)
    return receipt


_TEXT_COMPARABILITY_FIELDS = (
    "data_manifest_sha256",
    "model_config_sha256",
    "protein_tokenizer_sha256",
    "text_tokenizer_sha256",
    "model_artifact",
    "validation_files",
    "batches",
    "pairs",
    "text_target_tokens",
    "protein_masked_tokens",
    "evaluation_code_sha256",
    "retrieval_encoding_policy",
    "evaluation_settings",
    "proteingym_rows_opened",
)


def _load_passed_validation(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed" or payload.get("evidence_level") != "VALIDATION":
        raise RuntimeError(f"not a passed validation receipt: {path}")
    if payload.get("test_data_accessed") is not False:
        raise RuntimeError(f"validation receipt accessed or omitted test-data status: {path}")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("training_provenance"), dict
    ):
        raise RuntimeError(f"validation receipt has no checkpoint provenance: {path}")
    return payload


def select_multimodal_checkpoint(
    candidate_receipts: Iterable[Tuple[str, str]],
    train_receipt: str,
    initializer_validation_receipt: str,
    policy_path: str,
    output: str,
) -> Dict[str, Any]:
    """Select a text checkpoint only after conditioning and sequence gates pass."""

    policy_file = Path(policy_path).resolve()
    policy = json.loads(policy_file.read_text(encoding="utf-8"))
    if int(policy.get("schema_version", 0)) not in {1, 2}:
        raise RuntimeError("unsupported multimodal selection policy schema")
    expected_steps = [int(value) for value in policy.get("candidate_steps", [])]
    if not expected_steps or len(expected_steps) != len(set(expected_steps)):
        raise RuntimeError("selection policy candidate_steps must be unique and nonempty")
    if policy.get("primary_metric") != "minimum_text_lm_loss":
        raise RuntimeError("unsupported multimodal primary metric")
    max_sequence_delta = float(policy["max_sequence_loss_delta_from_initializer"])
    if not math.isfinite(max_sequence_delta) or max_sequence_delta < 0:
        raise RuntimeError("invalid maximum sequence-loss delta")
    if policy.get("require_conditioning_ci_upper_below_zero") is not True:
        raise RuntimeError("multimodal policy must require a negative conditioning CI upper bound")

    train_path = Path(train_receipt).resolve()
    train = json.loads(train_path.read_text(encoding="utf-8"))
    if train.get("status") != "passed" or train.get("task") != "sequence_text":
        raise RuntimeError("train receipt is not a passed sequence-text run")
    initialization = train.get("initialization_checkpoint")
    if not isinstance(initialization, dict):
        raise RuntimeError("sequence-text train receipt has no initialization checkpoint")
    initializer_path = Path(initializer_validation_receipt).resolve()
    initializer = _load_passed_validation(initializer_path)
    initializer_checkpoint = initializer["checkpoint"]
    for key in ("step", "manifest_sha256", "state_sha256"):
        if initialization.get(key) != initializer_checkpoint.get(key):
            raise RuntimeError(
                f"initializer validation does not match training initialization: {key}"
            )
    initializer_loss = float(initializer["bridge_loss"])

    records = []
    for text_value, sequence_value in candidate_receipts:
        text_path = Path(text_value).resolve()
        sequence_path = Path(sequence_value).resolve()
        text_payload = _load_passed_validation(text_path)
        sequence_payload = _load_passed_validation(sequence_path)
        text_checkpoint = text_payload["checkpoint"]
        sequence_checkpoint = sequence_payload["checkpoint"]
        checkpoint_identity = ("step", "manifest_sha256", "state_sha256")
        if any(
            text_checkpoint.get(key) != sequence_checkpoint.get(key)
            for key in checkpoint_identity
        ):
            raise RuntimeError("text and sequence receipts evaluate different checkpoints")
        if text_checkpoint["training_provenance"] != train.get("provenance"):
            raise RuntimeError("candidate checkpoint provenance differs from train receipt")
        if sequence_checkpoint["training_provenance"] != train.get("provenance"):
            raise RuntimeError("sequence checkpoint provenance differs from train receipt")
        for key in ("text_lm_loss", "text_lm_perplexity"):
            value = text_payload.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError(f"text validation has invalid {key}: {text_path}")
        sequence_loss = sequence_payload.get("bridge_loss")
        if not isinstance(sequence_loss, (int, float)) or not math.isfinite(float(sequence_loss)):
            raise RuntimeError(f"sequence validation has invalid bridge_loss: {sequence_path}")
        conditioning = text_payload.get("conditioning_control")
        if not isinstance(conditioning, dict):
            raise RuntimeError(f"text validation has no conditioning control: {text_path}")
        interval = conditioning.get("batch_mean_normal_95_percent_interval")
        if not isinstance(interval, list) or len(interval) != 2:
            raise RuntimeError(f"conditioning control has no 95 percent interval: {text_path}")
        conditioning_upper = float(interval[1])
        conditioning_pass = (
            conditioning.get("paired_better") is True
            and math.isfinite(conditioning_upper)
            and conditioning_upper < 0.0
        )
        sequence_delta = float(sequence_loss) - initializer_loss
        sequence_pass = sequence_delta <= max_sequence_delta
        reasons = []
        if not conditioning_pass:
            reasons.append("conditioning_ci_gate_failed")
        if not sequence_pass:
            reasons.append("sequence_regression_gate_failed")
        records.append(
            {
                "step": int(text_checkpoint["step"]),
                "checkpoint": text_checkpoint,
                "text_lm_loss": float(text_payload["text_lm_loss"]),
                "text_lm_perplexity": float(text_payload["text_lm_perplexity"]),
                "sequence_bridge_loss": float(sequence_loss),
                "sequence_loss_delta_from_initializer": sequence_delta,
                "conditioning_ci95": [float(interval[0]), conditioning_upper],
                "conditioning_paired_minus_shuffled": conditioning.get(
                    "paired_minus_shuffled_text_lm_loss"
                ),
                "eligible": not reasons,
                "ineligible_reasons": reasons,
                "text_validation_receipt_sha256": sha256_file(text_path),
                "sequence_validation_receipt_sha256": sha256_file(sequence_path),
                "text_payload": text_payload,
                "sequence_payload": sequence_payload,
            }
        )
    if sorted(record["step"] for record in records) != sorted(expected_steps):
        raise RuntimeError("candidate receipt steps differ from the predeclared policy")
    if len(records) != len({record["step"] for record in records}):
        raise RuntimeError("candidate receipts contain duplicate checkpoint steps")

    text_reference = records[0]["text_payload"]
    sequence_reference = records[0]["sequence_payload"]
    initializer_drift = [
        key
        for key in _COMPARABILITY_FIELDS
        if initializer.get(key) != sequence_reference.get(key)
    ]
    if initializer_drift:
        raise RuntimeError(
            "initializer and candidate sequence validations are not exactly comparable: "
            + ", ".join(initializer_drift)
        )
    for record in records[1:]:
        text_drift = [
            key
            for key in _TEXT_COMPARABILITY_FIELDS
            if record["text_payload"].get(key) != text_reference.get(key)
        ]
        sequence_drift = [
            key
            for key in _COMPARABILITY_FIELDS
            if record["sequence_payload"].get(key) != sequence_reference.get(key)
        ]
        if text_drift or sequence_drift:
            raise RuntimeError(
                "multimodal validation receipts are not exactly comparable: "
                + ", ".join(text_drift + sequence_drift)
            )
    eligible = [record for record in records if record["eligible"]]
    if not eligible:
        raise RuntimeError("no multimodal checkpoint passes the predeclared gates")
    selected = min(eligible, key=lambda record: (record["text_lm_loss"], record["step"]))
    records.sort(key=lambda record: record["step"])

    public_fields = (
        "step",
        "checkpoint",
        "text_lm_loss",
        "text_lm_perplexity",
        "sequence_bridge_loss",
        "sequence_loss_delta_from_initializer",
        "conditioning_ci95",
        "conditioning_paired_minus_shuffled",
        "eligible",
        "ineligible_reasons",
        "text_validation_receipt_sha256",
        "sequence_validation_receipt_sha256",
    )
    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": "VALIDATION_SELECTION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_code_sha256": source_tree_sha256(Path(__file__).resolve().parent),
        "policy": policy,
        "policy_sha256": sha256_file(policy_file),
        "train_receipt_sha256": sha256_file(train_path),
        "initializer_validation_receipt_sha256": sha256_file(initializer_path),
        "initializer": {
            "checkpoint": initializer_checkpoint,
            "bridge_loss": initializer_loss,
        },
        "test_data_accessed": False,
        "candidates": [
            {key: record[key] for key in public_fields}
            for record in records
        ],
        "selected": {key: selected[key] for key in public_fields},
        "claim_boundary": (
            "Validation-only multimodal checkpoint selection; this is not a frozen benchmark "
            "or a superiority claim."
        ),
    }
    _atomic_json_no_overwrite(Path(output).resolve(), receipt)
    return receipt
