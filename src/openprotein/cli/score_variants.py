"""Score variants with paired ESMC and OpenProtein masked marginals."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from openprotein.cli.generate_annotation import _atomic_json_no_overwrite
from openprotein.data.tokenizer import EsmcTokenizer
from openprotein.evaluation import paired_masked_marginal_scores
from openprotein.inference import torch_dtype
from openprotein.release import load_released_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_dir")
    parser.add_argument("protein_tokenizer_json")
    parser.add_argument("model_artifact_manifest")
    parser.add_argument("sequence")
    parser.add_argument("output")
    parser.add_argument("mutants", nargs="+")
    parser.add_argument("--positions-per-batch", type=int, default=8)
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
    sequence = args.sequence.strip().upper()
    scores = paired_masked_marginal_scores(
        model,
        tokenizer,
        sequence,
        args.mutants,
        device,
        positions_per_batch=args.positions_per_batch,
    )
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": "INFERENCE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reconstruction": reconstruction,
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "sequence_length": len(sequence),
        "mutants": args.mutants,
        "scores": scores,
        "score_definition": "summed mutant-minus-wildtype masked log odds",
        "claim_boundary": "Computational variant scores; not clinical or biological evidence.",
    }
    _atomic_json_no_overwrite(Path(args.output).resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
