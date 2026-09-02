"""Build a content-addressed CSV dataset manifest."""

from __future__ import annotations

import argparse
import json

from openprotein.data.manifest import build_csv_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument("output")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--license", required=True)
    args = parser.parse_args()
    manifest = build_csv_manifest(
        args.dataset_root,
        args.output,
        {
            "name": args.source_name,
            "url": args.source_url,
            "release": args.release,
            "license": args.license,
        },
    )
    print(json.dumps(manifest["split_totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
