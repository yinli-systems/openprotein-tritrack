"""Hash immutable model artifacts and write a JSON manifest."""

from __future__ import annotations

import argparse
import json

from openprotein.artifacts import build_artifact_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("output")
    parser.add_argument("--pattern", action="append", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--license", required=True)
    args = parser.parse_args()
    manifest = build_artifact_manifest(
        args.root,
        args.output,
        args.pattern,
        {
            "name": args.source_name,
            "url": args.source_url,
            "revision": args.revision,
            "license": args.license,
        },
    )
    print(
        json.dumps(
            {
                "files": len(manifest["files"]),
                "total_bytes": manifest["total_bytes"],
                "checkpoint_metadata": manifest.get("checkpoint_metadata"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
