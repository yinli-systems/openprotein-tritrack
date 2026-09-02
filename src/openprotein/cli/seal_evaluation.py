"""Create an immutable frozen-evaluation seal before benchmark access."""

from __future__ import annotations

import argparse
import json

from openprotein.evaluation import create_evaluation_seal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("benchmark_manifest")
    parser.add_argument("model_config")
    parser.add_argument("tokenizer_json")
    parser.add_argument("model_artifact_manifest")
    parser.add_argument("output")
    args = parser.parse_args()
    print(
        json.dumps(
            create_evaluation_seal(
                args.checkpoint_dir,
                args.benchmark_manifest,
                args.model_config,
                args.tokenizer_json,
                args.model_artifact_manifest,
                args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
