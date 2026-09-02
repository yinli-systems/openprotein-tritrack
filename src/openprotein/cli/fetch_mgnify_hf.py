"""Retrieve a pinned, sharded MGnify mirror with LFS hash verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openprotein.data.hf_mirror import build_pinned_hf_tree_manifest, download_pinned_hf_tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_definition")
    parser.add_argument("output_root")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    tree_manifest = output_root / "tree-manifest.json"
    receipt = output_root / "retrieval.json"
    build_pinned_hf_tree_manifest(
        args.source_definition, str(tree_manifest), endpoint=args.endpoint
    )
    result = download_pinned_hf_tree(
        str(tree_manifest), str(output_root / "shards"), str(receipt), workers=args.workers
    )
    print(json.dumps(result["totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
