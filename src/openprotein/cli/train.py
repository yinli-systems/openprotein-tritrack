"""Run the bounded single-rank pilot trainer."""

from __future__ import annotations

import argparse
import json

from openprotein.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    receipt = run_training(
        args.config,
        args.resume,
        args.max_steps,
        args.device,
        init_checkpoint=args.init_checkpoint,
    )
    if receipt.get("rank", 0) == 0:
        print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
