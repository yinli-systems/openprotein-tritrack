"""Run random-initialized protein model training."""

from __future__ import annotations

import argparse
import json

from openprotein.scratch.training import run_scratch_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--segment-end-step", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    receipt = run_scratch_training(
        args.config,
        resume=args.resume,
        max_steps_override=args.max_steps,
        segment_end_step=args.segment_end_step,
        device_name=args.device,
    )
    if receipt.get("rank", 0) == 0:
        print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
