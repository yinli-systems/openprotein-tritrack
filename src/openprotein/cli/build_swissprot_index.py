"""Build a verified Swiss-Prot annotation index."""

from __future__ import annotations

import argparse
import json

from openprotein.data.swissprot import build_swissprot_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dat_gz")
    parser.add_argument("output_database")
    parser.add_argument("output_manifest")
    parser.add_argument("--source-definition", required=True)
    args = parser.parse_args()
    manifest = build_swissprot_index(
        args.source_dat_gz,
        args.output_database,
        args.output_manifest,
        args.source_definition,
    )
    print(json.dumps(manifest["database"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
