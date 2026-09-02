import json
import shutil

import pytest

from openprotein.artifacts import build_artifact_manifest, verify_artifact_manifest


def test_model_manifest_requires_every_indexed_shard(tmp_path):
    (tmp_path / "model-00001.safetensors").write_bytes(b"one")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_parameters": 3, "total_size": 3},
                "weight_map": {"weight": "model-00001.safetensors"},
            }
        )
    )
    output = tmp_path / "manifest.json"
    manifest = build_artifact_manifest(
        str(tmp_path),
        str(output),
        ["*.json", "*.safetensors"],
        {"name": "fixture", "url": "https://example.invalid", "revision": "abc", "license": "MIT"},
    )
    assert manifest["checkpoint_metadata"]["total_parameters"] == 3
    assert manifest["checkpoint_tensors"] == 1
    assert manifest["total_bytes"] > 3
    receipt = verify_artifact_manifest(str(output), expected_root=str(tmp_path))
    assert receipt["file_count"] == 3

    relocated = tmp_path.parent / "relocated"
    relocated.mkdir()
    for source in tmp_path.iterdir():
        if source.is_file() and source != output:
            shutil.copyfile(source, relocated / source.name)
    relocated_receipt = verify_artifact_manifest(
        str(output), expected_root=str(relocated), artifact_root=str(relocated)
    )
    assert relocated_receipt["relocated"] is True
    assert relocated_receipt["recorded_root"] == str(tmp_path.resolve())

    (tmp_path / "model-00001.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="size differs"):
        verify_artifact_manifest(str(output), expected_root=str(tmp_path))
