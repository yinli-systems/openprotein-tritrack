"""Build hash-bound packed sequence shards from train-only CSV files."""

from __future__ import annotations

import argparse
import glob
import json

from openprotein.data.packed import build_packed_csv_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_glob")
    parser.add_argument("output_root")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()
    paths = sorted(glob.glob(args.input_glob))
    manifest = build_packed_csv_dataset(
        paths,
        args.output_root,
        args.tokenizer,
        args.source_manifest,
        args.source_root,
    )
    print(json.dumps(manifest["totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
