"""Select a sequence checkpoint only from a complete predeclared candidate set."""

from __future__ import annotations

import argparse
import json

from openprotein.policy_selection import select_sequence_checkpoint_with_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--pretrain-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("validation_receipts", nargs="+")
    args = parser.parse_args()
    receipt = select_sequence_checkpoint_with_policy(
        args.validation_receipts,
        args.pretrain_receipt,
        args.policy,
        args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
