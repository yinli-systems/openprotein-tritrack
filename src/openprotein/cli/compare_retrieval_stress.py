"""Compare target and baseline on one receipt-bound retrieval stress benchmark."""

from __future__ import annotations

import argparse
import json

from openprotein.retrieval_comparison import compare_retrieval_stress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = compare_retrieval_stress(args.target, args.baseline, args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
