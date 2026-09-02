"""Compare target and baseline frozen-encoder EC probe receipts."""

from __future__ import annotations

import argparse
import json

from openprotein.function_comparison import compare_ec_probes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = compare_ec_probes(args.target, args.baseline, args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
