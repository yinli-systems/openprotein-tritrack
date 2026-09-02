"""Build a manifest-bound, protocol-inspired large retrieval benchmark."""

from __future__ import annotations

import argparse
import json

from openprotein.retrieval_stress import build_retrieval_stress_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--record-id-allowlist", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-receipt", required=True)
    parser.add_argument("--distractors", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=61_003)
    args = parser.parse_args()
    receipt = build_retrieval_stress_benchmark(
        args.data_manifest,
        args.record_id_allowlist,
        args.output_csv,
        args.output_receipt,
        args.distractors,
        args.seed,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
