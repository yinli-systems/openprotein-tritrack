"""Build a low-homology Swiss-Prot/AlphaFold multimodal corpus."""

from __future__ import annotations

import argparse
import json

from openprotein.data.clustered_multimodal import (
    build_clustered_multimodal_dataset,
    export_swissprot_fasta,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export-fasta")
    export.add_argument("--swissprot-database", required=True)
    export.add_argument("--swissprot-manifest", required=True)
    export.add_argument("--output-fasta", required=True)
    export.add_argument("--output-receipt", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--swissprot-database", required=True)
    build.add_argument("--swissprot-manifest", required=True)
    build.add_argument("--structure-database", required=True)
    build.add_argument("--structure-manifest", required=True)
    build.add_argument("--cluster-tsv", required=True)
    build.add_argument("--output-root", required=True)
    build.add_argument("--mmseqs-version", required=True)
    build.add_argument("--seed", type=int, default=260903)
    build.add_argument("--train-shards", type=int, default=96)
    build.add_argument("--mmseqs-command", nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    if args.command == "export-fasta":
        receipt = export_swissprot_fasta(
            args.swissprot_database,
            args.swissprot_manifest,
            args.output_fasta,
            args.output_receipt,
        )
    else:
        receipt = build_clustered_multimodal_dataset(
            args.swissprot_database,
            args.swissprot_manifest,
            args.structure_database,
            args.structure_manifest,
            args.cluster_tsv,
            args.output_root,
            args.mmseqs_version,
            args.mmseqs_command,
            seed=args.seed,
            train_shards=args.train_shards,
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
