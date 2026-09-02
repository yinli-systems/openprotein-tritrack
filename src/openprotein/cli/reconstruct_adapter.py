"""Verify and reconstruct an optimizer-free OpenProtein adapter release."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch

from openprotein.release import load_released_adapter


def _atomic_receipt(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite reconstruction receipt: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_dir")
    parser.add_argument("tokenizer_json")
    parser.add_argument("model_artifact_manifest")
    parser.add_argument("receipt")
    parser.add_argument("--model-config")
    parser.add_argument("--backbone-root")
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    args = parser.parse_args()
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}.get(args.dtype)
    _model, receipt = load_released_adapter(
        args.release_dir,
        args.tokenizer_json,
        args.model_artifact_manifest,
        model_config=args.model_config,
        backbone_root=args.backbone_root,
        device=args.device,
        dtype=dtype,
    )
    _atomic_receipt(Path(args.receipt).resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
