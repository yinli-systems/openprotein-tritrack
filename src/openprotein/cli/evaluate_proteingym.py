"""Run a sealed ProteinGym substitution evaluation exactly once."""

from __future__ import annotations

import argparse
import json

from openprotein.evaluation import evaluate_frozen_proteingym


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("seal")
    parser.add_argument("output")
    parser.add_argument("--positions-per-batch", type=int, default=8)
    parser.add_argument("--device")
    args = parser.parse_args()
    receipt = evaluate_frozen_proteingym(
        args.seal,
        args.output,
        positions_per_batch=args.positions_per_batch,
        device_name=args.device,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
