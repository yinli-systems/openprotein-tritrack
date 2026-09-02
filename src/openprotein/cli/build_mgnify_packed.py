"""Build a deterministic packed subset of verified MGnify90 representatives."""

from __future__ import annotations

import argparse
import json

from openprotein.data.packed import build_selected_mgnify_packed_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_fasta")
    parser.add_argument("output_root")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--source-definition", required=True)
    parser.add_argument("--retrieval-receipt", required=True)
    parser.add_argument("--selection-basis-points", type=int, default=2500)
    parser.add_argument("--train-shards", type=int, default=128)
    parser.add_argument("--validation-shards", type=int, default=4)
    parser.add_argument("--test-shards", type=int, default=4)
    parser.add_argument("--minimum-train-residues", type=int, default=0)
    args = parser.parse_args()
    manifest = build_selected_mgnify_packed_dataset(
        args.source_fasta,
        args.output_root,
        args.tokenizer,
        args.source_definition,
        args.retrieval_receipt,
        selection_basis_points=args.selection_basis_points,
        shard_counts={
            "train": args.train_shards,
            "validation": args.validation_shards,
            "test": args.test_shards,
        },
        minimum_train_residues=args.minimum_train_residues,
    )
    print(
        json.dumps(
            {
                "selected_records": manifest["selected_records"],
                "selected_residues": manifest["selected_residues"],
                "split_totals": manifest["split_totals"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
