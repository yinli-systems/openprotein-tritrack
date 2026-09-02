"""Compare target and baseline retrieval receipts on an identical clean subset."""

from __future__ import annotations

import argparse
import json

from openprotein.retrieval_comparison import compare_decontaminated_retrieval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = compare_decontaminated_retrieval(args.target, args.baseline, args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
