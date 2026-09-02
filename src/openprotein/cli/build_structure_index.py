"""Build a verified coordinate index directly from an AlphaFold DB tar."""

from __future__ import annotations

import argparse
import json

from openprotein.data.structure import build_alphafold_structure_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_tar")
    parser.add_argument("output_database")
    parser.add_argument("output_manifest")
    parser.add_argument("--source-definition", required=True)
    args = parser.parse_args()
    manifest = build_alphafold_structure_index(
        args.source_tar,
        args.output_database,
        args.output_manifest,
        args.source_definition,
    )
    print(json.dumps(manifest["database"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
