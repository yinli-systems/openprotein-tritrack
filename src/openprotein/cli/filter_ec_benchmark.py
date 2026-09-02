"""Filter the EC sequence benchmark by a verified RDKit audit."""

from __future__ import annotations

import argparse
import json

from openprotein.ec_benchmark_filter import filter_ec_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-directory", required=True)
    parser.add_argument("--rdkit-audit-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    receipt = filter_ec_benchmark(
        args.benchmark_directory,
        args.rdkit_audit_directory,
        args.output_directory,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
