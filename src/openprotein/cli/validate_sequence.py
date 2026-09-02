"""Run validation-only MLM comparison against the exact ESMC path."""

from __future__ import annotations

import argparse
import json

from openprotein.validation import evaluate_sequence_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-json", required=True)
    parser.add_argument("--model-artifact-manifest", required=True)
    parser.add_argument("--validation-glob", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-batches", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--device")
    args = parser.parse_args()
    receipt = evaluate_sequence_validation(
        args.model_config,
        args.checkpoint,
        args.tokenizer_json,
        args.model_artifact_manifest,
        args.validation_glob,
        args.data_manifest,
        args.output,
        args.max_batches,
        args.sequence_length,
        device_name=args.device,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
