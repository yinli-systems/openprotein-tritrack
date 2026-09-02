"""Generate a protein-conditioned annotation from a verified release."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from openprotein.data.text_tokenizer import SentencePieceTextTokenizer
from openprotein.data.tokenizer import EsmcTokenizer
from openprotein.inference import generate_annotation, torch_dtype
from openprotein.release import load_released_adapter


def _atomic_json_no_overwrite(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite inference receipt: {path}")
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
    parser.add_argument("protein_tokenizer_json")
    parser.add_argument("model_artifact_manifest")
    parser.add_argument("sequence")
    parser.add_argument("output")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--max-new-tokens", type=int, default=128)
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
    protein_tokenizer = EsmcTokenizer.from_tokenizer_json(args.protein_tokenizer_json)
    text_tokenizer = SentencePieceTextTokenizer(
        str(Path(args.release_dir).resolve() / "text-tokenizer.model")
    )
    generation = generate_annotation(
        model,
        protein_tokenizer,
        text_tokenizer,
        args.sequence,
        device,
        prefix=args.prefix,
        max_new_tokens=args.max_new_tokens,
    )
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": "INFERENCE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reconstruction": reconstruction,
        "generation": generation,
        "claim_boundary": "Computational annotation hypothesis; not biological evidence.",
    }
    _atomic_json_no_overwrite(Path(args.output).resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
