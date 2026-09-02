"""Run validation-only sequence and sampled-geometry metrics."""

from __future__ import annotations

import argparse
import json

from openprotein.validation import evaluate_structure_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protein-tokenizer-json", required=True)
    parser.add_argument("--model-artifact-manifest", required=True)
    parser.add_argument("--validation-glob", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--structure-database", required=True)
    parser.add_argument("--structure-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-batches", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--geometry-pairs", type=int, default=128)
    parser.add_argument("--minimum-plddt", type=float, default=70.0)
    parser.add_argument("--device")
    parser.add_argument("--record-id-allowlist")
    args = parser.parse_args()
    receipt = evaluate_structure_validation(
        args.model_config,
        args.checkpoint,
        args.protein_tokenizer_json,
        args.model_artifact_manifest,
        args.validation_glob,
        args.data_manifest,
        args.structure_database,
        args.structure_manifest,
        args.output,
        args.max_batches,
        args.batch_size,
        args.sequence_length,
        args.geometry_pairs,
        min_confidence=args.minimum_plddt,
        device_name=args.device,
        record_id_allowlist_path=args.record_id_allowlist,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
