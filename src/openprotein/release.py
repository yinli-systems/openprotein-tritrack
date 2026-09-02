"""Optimizer-free adapter exports with immutable dependency receipts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from openprotein.artifacts import verify_artifact_manifest
from openprotein.config import load_model_config
from openprotein.model import OpenProteinModel
from openprotein.provenance import sha256_file

_EXTERNAL_BACKBONE_KINDS = {"esm2", "esmc"}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export_trainable_adapter(
    checkpoint_dir: str,
    model_config: str,
    tokenizer_json: str,
    model_artifact_manifest: str,
    output_dir: str,
    text_tokenizer_model: Optional[str] = None,
    pretrain_receipt: Optional[str] = None,
) -> Dict[str, Any]:
    """Export only trainable parameters; never bundle external backbone weights."""

    checkpoint = Path(checkpoint_dir).resolve()
    checkpoint_manifest_path = checkpoint / "manifest.json"
    state_path = checkpoint / "state.pt"
    config_path = Path(model_config).resolve()
    tokenizer_path = Path(tokenizer_json).resolve()
    artifact_manifest_path = Path(model_artifact_manifest).resolve()
    text_tokenizer_path = (
        Path(text_tokenizer_model).resolve() if text_tokenizer_model is not None else None
    )
    pretrain_receipt_path = (
        Path(pretrain_receipt).resolve() if pretrain_receipt is not None else None
    )
    final = Path(output_dir).resolve()
    if final.exists():
        raise FileExistsError(f"refusing to overwrite release directory: {final}")
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    if state_path.stat().st_size != int(checkpoint_manifest["state_bytes"]):
        raise RuntimeError("release checkpoint byte size differs from its manifest")
    state_sha256 = sha256_file(state_path)
    if state_sha256 != checkpoint_manifest["state_sha256"]:
        raise RuntimeError("release checkpoint SHA-256 differs from its manifest")
    config = load_model_config(str(config_path))
    artifact_receipt = verify_artifact_manifest(
        str(artifact_manifest_path),
        expected_root=(
            config.protein_backbone.model_path
            if config.protein_backbone.kind in _EXTERNAL_BACKBONE_KINDS
            else None
        ),
    )
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("release checkpoint has no structured provenance")
    expected = {
        "model_config_sha256": sha256_file(config_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "model_artifact_manifest_sha256": sha256_file(artifact_manifest_path),
    }
    mismatched = {key for key, value in expected.items() if provenance.get(key) != value}
    if mismatched:
        raise RuntimeError(
            "release checkpoint provenance mismatch: " + ", ".join(sorted(mismatched))
        )
    expected_text_tokenizer_sha256 = provenance.get("text_tokenizer_sha256")
    if expected_text_tokenizer_sha256 is not None:
        if text_tokenizer_path is None:
            raise RuntimeError(
                "multimodal checkpoint requires --text-tokenizer-model for release"
            )
        if sha256_file(text_tokenizer_path) != expected_text_tokenizer_sha256:
            raise RuntimeError("release text tokenizer SHA-256 differs from checkpoint provenance")
    elif text_tokenizer_path is not None:
        raise RuntimeError("checkpoint provenance does not bind a text tokenizer")
    trainable = payload.get("trainable_model")
    if not isinstance(trainable, dict) or not trainable:
        raise RuntimeError("release checkpoint contains no trainable model state")
    tensors = {
        str(name): tensor.detach().cpu().contiguous()
        for name, tensor in trainable.items()
        if isinstance(tensor, torch.Tensor)
    }
    if set(tensors) != set(trainable):
        raise RuntimeError("trainable model state contains a non-tensor value")
    trainable_parameters = sum(tensor.numel() for tensor in tensors.values())

    pretrain_payload = None
    if pretrain_receipt_path is not None:
        pretrain_payload = json.loads(pretrain_receipt_path.read_text(encoding="utf-8"))
        if pretrain_payload.get("status") != "passed":
            raise RuntimeError("pretraining receipt did not pass")
        if pretrain_payload.get("evidence_level") != "PRETRAIN":
            raise RuntimeError("release receipt is not PRETRAIN evidence")
        if pretrain_payload.get("token_budget_reached") is not True:
            raise RuntimeError("pretraining receipt did not reach its token budget")
        if int(pretrain_payload.get("tokens", -1)) < int(
            pretrain_payload.get("target_tokens", 0)
        ):
            raise RuntimeError("pretraining receipt token count is below target")
        if int(pretrain_payload.get("step", -1)) != int(payload["step"]):
            raise RuntimeError("pretraining receipt step differs from checkpoint")
        if pretrain_payload.get("provenance") != provenance:
            raise RuntimeError("pretraining receipt provenance differs from checkpoint")
        if int(pretrain_payload.get("trainable_parameters", -1)) != trainable_parameters:
            raise RuntimeError("pretraining receipt parameter count differs from checkpoint")

    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.tmp-", dir=final.parent))
    try:
        bundled_config_path = temporary / "model-config.json"
        shutil.copyfile(config_path, bundled_config_path)
        weights_path = temporary / "adapter.safetensors"
        save_file(
            tensors,
            str(weights_path),
            metadata={
                "format": "openprotein-trainable-adapter-v1",
                "checkpoint_state_sha256": state_sha256,
            },
        )
        bundled_text_tokenizer = None
        if text_tokenizer_path is not None:
            bundled_text_tokenizer = temporary / "text-tokenizer.model"
            shutil.copyfile(text_tokenizer_path, bundled_text_tokenizer)
        bundled_pretrain_receipt = None
        if pretrain_receipt_path is not None:
            bundled_pretrain_receipt = temporary / "pretrain-receipt.json"
            shutil.copyfile(pretrain_receipt_path, bundled_pretrain_receipt)
        manifest: Dict[str, Any] = {
            "schema_version": 2,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "format": (
                "OpenProtein trainable adapter; external protein backbone not included"
            ),
            "checkpoint": {
                "path_at_export": str(checkpoint),
                "step": int(payload["step"]),
                "manifest_sha256": sha256_file(checkpoint_manifest_path),
                "state_sha256": state_sha256,
            },
            "adapter": {
                "file": weights_path.name,
                "bytes": weights_path.stat().st_size,
                "sha256": sha256_file(weights_path),
                "tensors": len(tensors),
                "parameters": trainable_parameters,
                "dtypes": sorted({str(tensor.dtype) for tensor in tensors.values()}),
            },
            "model_config": {
                "file": bundled_config_path.name,
                "bytes": bundled_config_path.stat().st_size,
                "sha256": sha256_file(bundled_config_path),
            },
            "dependencies": {
                "model_config_sha256": expected["model_config_sha256"],
                "tokenizer_sha256": expected["tokenizer_sha256"],
                "model_artifact": artifact_receipt,
            },
            "training_provenance": provenance,
            "claim_boundary": (
                "This export is not evidence of frozen-benchmark superiority or biological "
                "efficacy."
            ),
        }
        if bundled_text_tokenizer is not None:
            manifest["text_tokenizer"] = {
                "file": bundled_text_tokenizer.name,
                "bytes": bundled_text_tokenizer.stat().st_size,
                "sha256": sha256_file(bundled_text_tokenizer),
            }
        if bundled_pretrain_receipt is not None and pretrain_payload is not None:
            manifest["pretrain_receipt"] = {
                "file": bundled_pretrain_receipt.name,
                "bytes": bundled_pretrain_receipt.stat().st_size,
                "sha256": sha256_file(bundled_pretrain_receipt),
                "evidence_level": pretrain_payload["evidence_level"],
                "step": int(pretrain_payload["step"]),
                "tokens": int(pretrain_payload["tokens"]),
                "target_tokens": int(pretrain_payload["target_tokens"]),
            }
        _write_json(temporary / "manifest.json", manifest)
        (temporary / "README.md").write_text(
            "# OpenProtein adapter\n\n"
            "This directory contains only the trainable OpenProtein bridge and task heads. "
            "It does not redistribute the external protein backbone. Fetch the exact "
            "backbone under its "
            "upstream license and verify it against `manifest.json` before loading. The exact "
            "model configuration and, for multimodal checkpoints, the trained text tokenizer "
            "are included. The external backbone may be relocated only through the verified "
            "reconstruction command.\n\n"
            "The release manifest records engineering provenance. Consult the separate frozen "
            "evaluation receipt before making any quality claim. Computational predictions do "
            "not replace wet-lab validation.\n",
            encoding="utf-8",
        )
        os.replace(temporary, final)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _release_member(release_root: Path, value: object, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise RuntimeError(f"{label} must be relative to the release directory")
    candidate = (release_root / relative).resolve()
    try:
        candidate.relative_to(release_root)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes the release directory") from error
    if not candidate.is_file():
        raise RuntimeError(f"{label} is missing")
    return candidate


def _verify_bundled_file(release_root: Path, item: object, label: str) -> Path:
    if not isinstance(item, dict):
        raise RuntimeError(f"release manifest has no {label} receipt")
    path = _release_member(release_root, item.get("file"), label)
    if path.stat().st_size != int(item.get("bytes", -1)):
        raise RuntimeError(f"{label} byte size differs from release manifest")
    if sha256_file(path) != item.get("sha256"):
        raise RuntimeError(f"{label} SHA-256 differs from release manifest")
    return path


def load_released_adapter(
    release_dir: str,
    tokenizer_json: str,
    model_artifact_manifest: str,
    *,
    model_config: Optional[str] = None,
    backbone_root: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[OpenProteinModel, Dict[str, Any]]:
    """Verify and reconstruct a released adapter over its external backbone."""

    release_root = Path(release_dir).resolve()
    manifest_path = release_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", 0)) != 2:
        raise RuntimeError("unsupported release manifest schema")

    bundled_config = _verify_bundled_file(
        release_root, manifest.get("model_config"), "model config"
    )
    config_path = Path(model_config).resolve() if model_config is not None else bundled_config
    dependencies = manifest.get("dependencies")
    if not isinstance(dependencies, dict):
        raise RuntimeError("release manifest has no dependency receipts")
    if sha256_file(config_path) != dependencies.get("model_config_sha256"):
        raise RuntimeError("model config SHA-256 differs from release manifest")
    tokenizer_path = Path(tokenizer_json).resolve()
    if sha256_file(tokenizer_path) != dependencies.get("tokenizer_sha256"):
        raise RuntimeError("protein tokenizer SHA-256 differs from release manifest")

    artifact_manifest_path = Path(model_artifact_manifest).resolve()
    artifact_dependency = dependencies.get("model_artifact")
    if not isinstance(artifact_dependency, dict):
        raise RuntimeError("release manifest has no model artifact receipt")
    if sha256_file(artifact_manifest_path) != artifact_dependency.get("manifest_sha256"):
        raise RuntimeError("model artifact manifest SHA-256 differs from release manifest")

    config = load_model_config(str(config_path))
    effective_backbone_root = backbone_root or config.protein_backbone.model_path
    if (
        config.protein_backbone.kind in _EXTERNAL_BACKBONE_KINDS
        and effective_backbone_root is None
    ):
        raise RuntimeError("reconstruction requires an external backbone root")
    artifact_receipt = verify_artifact_manifest(
        str(artifact_manifest_path),
        expected_root=(
            effective_backbone_root
            if config.protein_backbone.kind in _EXTERNAL_BACKBONE_KINDS
            else None
        ),
        artifact_root=backbone_root,
    )
    for key in ("file_count", "total_bytes"):
        if int(artifact_receipt[key]) != int(artifact_dependency.get(key, -1)):
            raise RuntimeError(f"model artifact {key} differs from release manifest")

    training_provenance = manifest.get("training_provenance")
    if not isinstance(training_provenance, dict):
        raise RuntimeError("release manifest has no training provenance")
    provenance_dependencies = {
        "model_config_sha256": dependencies.get("model_config_sha256"),
        "tokenizer_sha256": dependencies.get("tokenizer_sha256"),
        "model_artifact_manifest_sha256": artifact_dependency.get("manifest_sha256"),
    }
    mismatched = {
        key
        for key, value in provenance_dependencies.items()
        if training_provenance.get(key) != value
    }
    if mismatched:
        raise RuntimeError(
            "release training provenance mismatch: " + ", ".join(sorted(mismatched))
        )

    pretrain_receipt_dependency = manifest.get("pretrain_receipt")
    pretrain_payload = None
    pretrain_receipt_sha256 = None
    if pretrain_receipt_dependency is not None:
        pretrain_receipt_path = _verify_bundled_file(
            release_root, pretrain_receipt_dependency, "pretraining receipt"
        )
        pretrain_payload = json.loads(pretrain_receipt_path.read_text(encoding="utf-8"))
        if pretrain_payload.get("status") != "passed":
            raise RuntimeError("bundled pretraining receipt did not pass")
        if pretrain_payload.get("evidence_level") != "PRETRAIN":
            raise RuntimeError("bundled receipt is not PRETRAIN evidence")
        if pretrain_payload.get("token_budget_reached") is not True:
            raise RuntimeError("bundled pretraining receipt did not reach its token budget")
        if pretrain_payload.get("provenance") != training_provenance:
            raise RuntimeError("bundled pretraining receipt provenance differs")
        checkpoint_dependency = manifest.get("checkpoint")
        if not isinstance(checkpoint_dependency, dict):
            raise RuntimeError("release manifest has no checkpoint receipt")
        if int(pretrain_payload.get("step", -1)) != int(
            checkpoint_dependency.get("step", -2)
        ):
            raise RuntimeError("bundled pretraining receipt step differs from checkpoint")
        pretrain_receipt_sha256 = sha256_file(pretrain_receipt_path)

    text_tokenizer_receipt = manifest.get("text_tokenizer")
    expected_text_sha256 = training_provenance.get("text_tokenizer_sha256")
    if expected_text_sha256 is None and text_tokenizer_receipt is not None:
        raise RuntimeError("release contains an unbound text tokenizer")
    if expected_text_sha256 is not None:
        text_tokenizer_path = _verify_bundled_file(
            release_root, text_tokenizer_receipt, "text tokenizer"
        )
        if sha256_file(text_tokenizer_path) != expected_text_sha256:
            raise RuntimeError("text tokenizer SHA-256 differs from training provenance")

    adapter_path = _verify_bundled_file(
        release_root, manifest.get("adapter"), "adapter"
    )
    with safe_open(str(adapter_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("release manifest has no checkpoint receipt")
    if metadata.get("checkpoint_state_sha256") != checkpoint.get("state_sha256"):
        raise RuntimeError("adapter metadata is not bound to the released checkpoint")
    adapter_receipt = manifest["adapter"]
    tensor_parameters = sum(tensor.numel() for tensor in tensors.values())
    tensor_dtypes = sorted({str(tensor.dtype) for tensor in tensors.values()})
    if len(tensors) != int(adapter_receipt.get("tensors", -1)):
        raise RuntimeError("adapter tensor count differs from release manifest")
    if tensor_parameters != int(adapter_receipt.get("parameters", -1)):
        raise RuntimeError("adapter parameter count differs from release manifest")
    if pretrain_payload is not None and int(
        pretrain_payload.get("trainable_parameters", -1)
    ) != tensor_parameters:
        raise RuntimeError("bundled pretraining receipt parameter count differs")
    if tensor_dtypes != adapter_receipt.get("dtypes"):
        raise RuntimeError("adapter dtypes differ from release manifest")

    if (
        backbone_root is not None
        and config.protein_backbone.kind in _EXTERNAL_BACKBONE_KINDS
    ):
        config = replace(
            config,
            protein_backbone=replace(
                config.protein_backbone, model_path=str(Path(backbone_root).resolve())
            ),
        )
    device_object = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dtype is None:
        unique_dtypes = {tensor.dtype for tensor in tensors.values() if tensor.is_floating_point()}
        if len(unique_dtypes) != 1:
            raise RuntimeError(
                "adapter must have one floating-point dtype unless dtype is explicit"
            )
        dtype = next(iter(unique_dtypes))
    model = OpenProteinModel.from_config(config, device=device_object, dtype=dtype)
    model.to(device=device_object, dtype=dtype)
    expected_parameters = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(tensors) != set(expected_parameters):
        missing = sorted(set(expected_parameters) - set(tensors))
        extra = sorted(set(tensors) - set(expected_parameters))
        raise RuntimeError(f"adapter parameter names differ: missing={missing}, extra={extra}")
    with torch.no_grad():
        for name, parameter in expected_parameters.items():
            tensor = tensors[name]
            if tensor.shape != parameter.shape:
                raise RuntimeError(f"adapter parameter shape differs: {name}")
            if tensor.dtype != parameter.dtype:
                raise RuntimeError(f"adapter parameter dtype differs: {name}")
            parameter.copy_(tensor.to(device_object))
    model.eval()
    loaded_parameters = sum(parameter.numel() for parameter in model.parameters())
    if (
        config.protein_backbone.kind in _EXTERNAL_BACKBONE_KINDS
        and loaded_parameters != config.estimated_parameter_count
    ):
        raise RuntimeError("reconstructed model parameter count differs from model config")
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": "RECONSTRUCTION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_manifest_sha256": sha256_file(manifest_path),
        "adapter_sha256": sha256_file(adapter_path),
        "adapter_parameters": tensor_parameters,
        "loaded_model_parameters": loaded_parameters,
        "analytical_model_parameters": config.estimated_parameter_count,
        "device": str(device_object),
        "dtype": str(dtype),
        "artifact": artifact_receipt,
        "checkpoint": checkpoint,
        "pretrain_receipt_sha256": pretrain_receipt_sha256,
    }
    return model, receipt
