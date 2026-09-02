"""Content-addressed manifests for immutable model/runtime artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openprotein.provenance import sha256_file


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build_artifact_manifest(
    root: str,
    output: str,
    patterns: Iterable[str],
    source: Dict[str, Any],
) -> Dict[str, Any]:
    artifact_root = Path(root).resolve()
    selected: List[Path] = []
    for pattern in patterns:
        selected.extend(path for path in artifact_root.glob(pattern) if path.is_file())
    selected = sorted(set(selected))
    if not selected:
        raise ValueError("artifact patterns matched no files")
    files = [
        {
            "path": path.relative_to(artifact_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in selected
    ]
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(artifact_root),
        "source": source,
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    index_path = artifact_root / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_files = set(index.get("weight_map", {}).values())
        present = {item["path"] for item in files}
        missing = sorted(weight_files - present)
        if missing:
            raise ValueError(f"manifest omits checkpoint shards: {missing}")
        manifest["checkpoint_metadata"] = index.get("metadata", {})
        manifest["checkpoint_tensors"] = len(index.get("weight_map", {}))
    _atomic_json(Path(output), manifest)
    return manifest


def verify_artifact_manifest(
    manifest_path: str,
    expected_root: Optional[str] = None,
    artifact_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Fail closed unless every artifact file matches its content receipt.

    ``artifact_root`` permits a byte-identical artifact set to be reconstructed
    at a different absolute path without rewriting the signed manifest.  The
    recorded root remains in the receipt so relocation is explicit.
    """

    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", 0)) != 1:
        raise RuntimeError("unsupported artifact manifest schema")
    recorded_root = Path(manifest["root"]).resolve()
    root = Path(artifact_root).resolve() if artifact_root is not None else recorded_root
    if expected_root is not None and root != Path(expected_root).resolve():
        raise RuntimeError("artifact manifest root does not match the configured model path")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("artifact manifest contains no files")

    total_bytes = 0
    seen = set()
    for item in files:
        relative = Path(item["path"])
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise RuntimeError("artifact manifest path escapes its root") from error
        if candidate in seen:
            raise RuntimeError("artifact manifest contains a duplicate file")
        seen.add(candidate)
        if not candidate.is_file():
            raise RuntimeError(f"artifact file is missing: {relative.as_posix()}")
        size = candidate.stat().st_size
        if size != int(item["bytes"]):
            raise RuntimeError(f"artifact file size differs: {relative.as_posix()}")
        if sha256_file(candidate) != item["sha256"]:
            raise RuntimeError(f"artifact file SHA-256 differs: {relative.as_posix()}")
        total_bytes += size
    if total_bytes != int(manifest["total_bytes"]):
        raise RuntimeError("artifact manifest total byte count differs")
    return {
        "manifest_sha256": sha256_file(path),
        "root": str(root),
        "recorded_root": str(recorded_root),
        "relocated": root != recorded_root,
        "file_count": len(files),
        "total_bytes": total_bytes,
    }
