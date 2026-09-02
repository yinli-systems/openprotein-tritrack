import hashlib
import json

import pytest
import torch

from openprotein.checkpoint_audit import audit_checkpoint_delta


def _checkpoint(path, state):
    path.mkdir()
    state_path = path / "state.pt"
    torch.save({"trainable_model": state}, state_path)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "step": 1,
                "state_bytes": state_path.stat().st_size,
                "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
            }
        )
    )


def test_checkpoint_delta_audit_rejects_changes_outside_allowed_prefix(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _checkpoint(
        source,
        {
            "bridge.weight": torch.tensor([1.0]),
            "geometry_head.weight": torch.tensor([2.0]),
        },
    )
    _checkpoint(
        target,
        {
            "bridge.weight": torch.tensor([1.0]),
            "geometry_head.weight": torch.tensor([3.0]),
        },
    )
    receipt = audit_checkpoint_delta(
        str(source),
        str(target),
        ["geometry_head."],
        str(tmp_path / "receipt.json"),
    )
    assert receipt["changed_tensors"] == ["geometry_head.weight"]
    assert receipt["unchanged_parameters"] == 1

    outside = tmp_path / "outside"
    _checkpoint(
        outside,
        {
            "bridge.weight": torch.tensor([4.0]),
            "geometry_head.weight": torch.tensor([3.0]),
        },
    )
    with pytest.raises(RuntimeError, match="outside allowed"):
        audit_checkpoint_delta(
            str(source),
            str(outside),
            ["geometry_head."],
            str(tmp_path / "rejected.json"),
        )
