"""Build the verified TorchProtein EnzymeCommission sequence benchmark."""

from __future__ import annotations

import argparse
import json

from openprotein.ec_benchmark import build_ec_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", required=True)
    parser.add_argument("--source-definition", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    receipt = build_ec_benchmark(
        args.source_zip,
        args.source_definition,
        args.output_directory,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
