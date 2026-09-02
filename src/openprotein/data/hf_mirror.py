"""Pinned, hash-verified retrieval of sharded Hugging Face datasets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHARD = re.compile(r"^shard-(\d{6})\.fasta\.zst$")


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _endpoint_url(endpoint: str) -> str:
    parsed = urlsplit(endpoint.rstrip("/"))
    if parsed.scheme != "https" or not parsed.netloc or parsed.path:
        raise ValueError("Hugging Face endpoint must be an HTTPS origin")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _next_link(link_header: Optional[str], endpoint: str) -> Optional[str]:
    if not link_header:
        return None
    for item in link_header.split(","):
        target, *parameters = item.split(";")
        if not any(parameter.strip() == 'rel="next"' for parameter in parameters):
            continue
        target = target.strip()
        if not target.startswith("<") or not target.endswith(">"):
            raise RuntimeError("Hugging Face pagination link is malformed")
        parsed = urlsplit(target[1:-1])
        base = urlsplit(endpoint)
        return urlunsplit((base.scheme, base.netloc, parsed.path, parsed.query, ""))
    return None


def build_pinned_hf_tree_manifest(
    source_definition_path: str,
    output_manifest_path: str,
    endpoint: str = "https://hf-mirror.com",
) -> Dict[str, Any]:
    """Resolve every shard at a pinned revision and record its LFS SHA-256."""

    definition_path = Path(source_definition_path).resolve()
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    transport = definition.get("mirror_transport", {})
    repo_id = transport.get("repo_id")
    revision = transport.get("revision")
    prefix = transport.get("path_prefix")
    if not all(isinstance(value, str) and value for value in (repo_id, revision, prefix)):
        raise ValueError("source definition has incomplete mirror transport metadata")
    if not _GIT_SHA.fullmatch(revision):
        raise ValueError("mirror revision must be a full commit SHA")
    base = _endpoint_url(endpoint)
    api_path = (
        f"/api/datasets/{quote(repo_id, safe='/')}/tree/{revision}/"
        f"{quote(prefix, safe='')}"
    )
    next_url: Optional[str] = f"{base}{api_path}?recursive=true&expand=true"
    entries: List[Dict[str, Any]] = []
    seen_paths = set()
    while next_url is not None:
        request = Request(next_url, headers={"User-Agent": "openprotein/0.1"})
        with urlopen(request, timeout=60) as response:
            page = json.load(response)
            link = response.headers.get("Link")
        if not isinstance(page, list):
            raise RuntimeError("Hugging Face tree API returned a non-list page")
        for item in page:
            if item.get("type") != "file":
                continue
            path = item.get("path")
            size = item.get("size")
            lfs = item.get("lfs")
            if not isinstance(path, str) or not path.startswith(f"{prefix}/"):
                raise RuntimeError("Hugging Face tree contains an unexpected path")
            match = _SHARD.fullmatch(Path(path).name)
            if match is None or path in seen_paths:
                raise RuntimeError("Hugging Face tree contains an invalid or duplicate shard")
            if not isinstance(size, int) or size <= 0 or not isinstance(lfs, dict):
                raise RuntimeError("Hugging Face shard metadata is incomplete")
            digest = lfs.get("oid")
            if int(lfs.get("size", -1)) != size or not isinstance(digest, str):
                raise RuntimeError("Hugging Face LFS metadata is inconsistent")
            if not _SHA256.fullmatch(digest):
                raise RuntimeError("Hugging Face LFS object ID is not SHA-256")
            seen_paths.add(path)
            entries.append(
                {
                    "path": path,
                    "local_path": Path(path).name,
                    "shard_index": int(match.group(1)),
                    "bytes": size,
                    "sha256": digest,
                }
            )
        next_url = _next_link(link, base)

    entries.sort(key=lambda item: int(item["shard_index"]))
    observed_indices = [int(item["shard_index"]) for item in entries]
    if observed_indices != list(range(1, len(entries) + 1)):
        raise RuntimeError("Hugging Face shard indices are not contiguous from one")
    observed_bytes = sum(int(item["bytes"]) for item in entries)
    expected_shards = int(transport["shards"])
    expected_bytes = int(transport["bytes"])
    if len(entries) != expected_shards or observed_bytes != expected_bytes:
        raise RuntimeError("Hugging Face tree differs from the source definition")

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "status": "PINNED_TREE_VERIFIED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_definition": str(definition_path),
        "source_definition_sha256": hashlib.sha256(definition_path.read_bytes()).hexdigest(),
        "endpoint": base,
        "repo_id": repo_id,
        "revision": revision,
        "path_prefix": prefix,
        "totals": {"shards": len(entries), "bytes": observed_bytes},
        "files": entries,
    }
    _atomic_json(Path(output_manifest_path).resolve(), payload)
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download_one(
    item: Dict[str, Any], output_root: Path, endpoint: str, repo_id: str, revision: str
) -> Dict[str, Any]:
    final = output_root / item["local_path"]
    partial = final.with_suffix(f"{final.suffix}.part")
    expected_bytes = int(item["bytes"])
    expected_sha256 = str(item["sha256"])
    if final.exists():
        if final.stat().st_size != expected_bytes or _sha256_file(final) != expected_sha256:
            raise RuntimeError(f"existing shard differs from pinned metadata: {final.name}")
        return {**item, "status": "already_verified"}
    if partial.exists() and partial.stat().st_size > expected_bytes:
        raise RuntimeError(f"partial shard exceeds expected size: {partial.name}")
    source_url = (
        f"{endpoint}/datasets/{quote(repo_id, safe='/')}/resolve/{revision}/"
        f"{quote(str(item['path']), safe='/')}?download=true"
    )
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "20",
            "--retry-delay",
            "5",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            source_url,
        ],
        check=True,
    )
    if partial.stat().st_size != expected_bytes:
        raise RuntimeError(f"downloaded shard has the wrong size: {partial.name}")
    if _sha256_file(partial) != expected_sha256:
        raise RuntimeError(f"downloaded shard has the wrong SHA-256: {partial.name}")
    os.replace(partial, final)
    return {**item, "status": "downloaded_and_verified"}


def download_pinned_hf_tree(
    tree_manifest_path: str,
    output_root: str,
    receipt_path: str,
    workers: int = 8,
) -> Dict[str, Any]:
    """Download a pinned tree concurrently and verify every LFS object hash."""

    if not 1 <= workers <= 32:
        raise ValueError("workers must be in [1, 32]")
    manifest_path = Path(tree_manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PINNED_TREE_VERIFIED":
        raise RuntimeError("Hugging Face tree manifest is not verified")
    endpoint = _endpoint_url(manifest["endpoint"])
    repo_id = str(manifest["repo_id"])
    revision = str(manifest["revision"])
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("Hugging Face tree manifest has no files")
    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    completed: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, item, destination, endpoint, repo_id, revision): item
            for item in files
        }
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(
                json.dumps(
                    {
                        "completed": len(completed),
                        "total": len(files),
                        "path": result["local_path"],
                        "status": result["status"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    completed.sort(key=lambda item: int(item["shard_index"]))
    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "status": "VERIFIED_SHARDED_INPUT",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "tree_manifest": str(manifest_path),
        "tree_manifest_sha256": _sha256_file(manifest_path),
        "dataset_root": str(destination),
        "endpoint": endpoint,
        "repo_id": repo_id,
        "revision": revision,
        "totals": {
            "shards": len(completed),
            "bytes": sum(int(item["bytes"]) for item in completed),
        },
        "files": [
            {key: item[key] for key in ("path", "local_path", "shard_index", "bytes", "sha256")}
            for item in completed
        ],
    }
    _atomic_json(Path(receipt_path).resolve(), receipt)
    return receipt
