"""Fail-closed audits for cross-stage adapter checkpoint deltas."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch

from openprotein.provenance import sha256_file, source_tree_sha256


def _load_checkpoint(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    manifest_path = path / "manifest.json"
    state_path = path / "state.pt"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if state_path.stat().st_size != int(manifest["state_bytes"]):
        raise RuntimeError("checkpoint state byte count differs from its manifest")
    if sha256_file(state_path) != manifest["state_sha256"]:
        raise RuntimeError("checkpoint state SHA-256 differs from its manifest")
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(payload.get("trainable_model"), dict):
        raise RuntimeError("checkpoint has no trainable model state")
    return manifest, payload


def audit_checkpoint_delta(
    source_checkpoint: str,
    target_checkpoint: str,
    allowed_changed_prefixes: Sequence[str],
    output_path: str,
) -> Dict[str, Any]:
    """Require every changed adapter tensor to match a predeclared prefix."""

    prefixes = tuple(str(prefix) for prefix in allowed_changed_prefixes)
    if not prefixes or any(not prefix for prefix in prefixes):
        raise ValueError("at least one non-empty changed-parameter prefix is required")
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint audit: {output}")
    source_path = Path(source_checkpoint).resolve()
    target_path = Path(target_checkpoint).resolve()
    source_manifest, source = _load_checkpoint(source_path)
    target_manifest, target = _load_checkpoint(target_path)
    source_state = source["trainable_model"]
    target_state = target["trainable_model"]
    if set(source_state) != set(target_state):
        raise RuntimeError("checkpoint adapter parameter names differ")

    changed = []
    unchanged_tensors = 0
    unchanged_parameters = 0
    changed_parameters = 0
    for name in sorted(source_state):
        source_tensor = source_state[name]
        target_tensor = target_state[name]
        if source_tensor.shape != target_tensor.shape or source_tensor.dtype != target_tensor.dtype:
            raise RuntimeError(f"checkpoint tensor contract differs: {name}")
        if torch.equal(source_tensor, target_tensor):
            unchanged_tensors += 1
            unchanged_parameters += source_tensor.numel()
            continue
        if not name.startswith(prefixes):
            raise RuntimeError(f"checkpoint changed outside allowed prefixes: {name}")
        changed.append(name)
        changed_parameters += source_tensor.numel()
    if not changed:
        raise RuntimeError("checkpoint audit found no changed tensors")

    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": "CHECKPOINT_AUDIT",
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": {
            "path": str(source_path),
            "manifest_sha256": sha256_file(source_path / "manifest.json"),
            "state_sha256": source_manifest["state_sha256"],
            "step": source_manifest["step"],
        },
        "target_checkpoint": {
            "path": str(target_path),
            "manifest_sha256": sha256_file(target_path / "manifest.json"),
            "state_sha256": target_manifest["state_sha256"],
            "step": target_manifest["step"],
        },
        "allowed_changed_prefixes": list(prefixes),
        "changed_tensors": changed,
        "changed_parameters": changed_parameters,
        "unchanged_tensors": unchanged_tensors,
        "unchanged_parameters": unchanged_parameters,
        "audit_code_sha256": source_tree_sha256(Path(__file__).resolve().parent),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, output)
    return receipt
