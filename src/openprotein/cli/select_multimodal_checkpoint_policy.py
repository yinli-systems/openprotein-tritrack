"""Select a sequence-text checkpoint only under a complete frozen policy."""

from __future__ import annotations

import argparse
import json

from openprotein.policy_selection import select_multimodal_checkpoint_with_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--train-receipt", required=True)
    parser.add_argument("--initializer-validation-receipt", required=True)
    parser.add_argument("--upstream-selection-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=2,
        metavar=("TEXT_RECEIPT", "SEQUENCE_RECEIPT"),
        required=True,
    )
    args = parser.parse_args()
    receipt = select_multimodal_checkpoint_with_policy(
        args.candidate,
        args.train_receipt,
        args.initializer_validation_receipt,
        args.upstream_selection_receipt,
        args.policy,
        args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
