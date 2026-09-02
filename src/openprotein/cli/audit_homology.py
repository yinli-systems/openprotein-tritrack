"""Prepare and summarize a manifest-bound MMseqs2 homology audit."""

from __future__ import annotations

import argparse
import json

from openprotein.homology import prepare_homology_inputs, summarize_homology_hits


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--train-manifest", required=True)
    prepare.add_argument("--validation-manifest", required=True)
    prepare.add_argument("--output-dir", required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--prepare", required=True)
    summarize.add_argument("--hits", required=True)
    summarize.add_argument("--output", required=True)
    summarize.add_argument("--clean-ids", required=True)
    summarize.add_argument("--excluded-ids", required=True)
    summarize.add_argument("--top-hits", required=True)
    summarize.add_argument("--mmseqs-version", required=True)
    summarize.add_argument("--search-slurm-job-id")
    summarize.add_argument("--mmseqs-command", nargs=argparse.REMAINDER, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        receipt = prepare_homology_inputs(
            args.train_manifest,
            args.validation_manifest,
            args.output_dir,
        )
    else:
        receipt = summarize_homology_hits(
            args.prepare,
            args.hits,
            args.output,
            args.clean_ids,
            args.excluded_ids,
            args.top_hits,
            args.mmseqs_version,
            args.mmseqs_command,
            args.search_slurm_job_id,
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
