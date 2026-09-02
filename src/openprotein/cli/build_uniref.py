"""Build deterministic, verified CSV shards from an official UniRef FASTA."""

from __future__ import annotations

import argparse
import json

from openprotein.data.uniref import build_uniref_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_fasta")
    parser.add_argument("output_root")
    parser.add_argument("--source-definition", required=True)
    parser.add_argument("--train-shards", type=int, default=96)
    parser.add_argument("--validation-shards", type=int, default=1)
    parser.add_argument("--test-shards", type=int, default=1)
    parser.add_argument("--train-basis-points", type=int, default=9800)
    parser.add_argument("--validation-basis-points", type=int, default=100)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-length", type=int)
    args = parser.parse_args()
    manifest = build_uniref_csv(
        args.source_fasta,
        args.output_root,
        args.source_definition,
        shard_counts={
            "train": args.train_shards,
            "validation": args.validation_shards,
            "test": args.test_shards,
        },
        train_basis_points=args.train_basis_points,
        validation_basis_points=args.validation_basis_points,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    print(
        json.dumps(
            {
                "source_input": manifest["source_input"],
                "filters": manifest["filters"],
                "split_totals": manifest["split_totals"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
