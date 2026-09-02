"""Predict coarse residue-pair distance bins from a verified release."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from openprotein.cli.generate_annotation import _atomic_json_no_overwrite
from openprotein.data.tokenizer import EsmcTokenizer
from openprotein.inference import predict_pair_distances, torch_dtype
from openprotein.release import load_released_adapter


def _pair(value: str) -> tuple[int, int]:
    try:
        left, right = value.split(":", 1)
        return int(left), int(right)
    except ValueError as error:
        raise argparse.ArgumentTypeError("pair must have one-based LEFT:RIGHT form") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_dir")
    parser.add_argument("protein_tokenizer_json")
    parser.add_argument("model_artifact_manifest")
    parser.add_argument("sequence")
    parser.add_argument("output")
    parser.add_argument("pairs", nargs="+", type=_pair)
    parser.add_argument("--backbone-root")
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, reconstruction = load_released_adapter(
        args.release_dir,
        args.protein_tokenizer_json,
        args.model_artifact_manifest,
        backbone_root=args.backbone_root,
        device=str(device),
        dtype=torch_dtype(args.dtype),
    )
    tokenizer = EsmcTokenizer.from_tokenizer_json(args.protein_tokenizer_json)
    geometry = predict_pair_distances(model, tokenizer, args.sequence, args.pairs, device)
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": "INFERENCE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reconstruction": reconstruction,
        "geometry": geometry,
        "claim_boundary": (
            "Coarse computational distances from a predicted-structure-trained decoder; "
            "not an experimental structure or biological efficacy result."
        ),
    }
    _atomic_json_no_overwrite(Path(args.output).resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
