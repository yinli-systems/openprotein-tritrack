"""Select a multimodal checkpoint under predeclared validation gates."""

from __future__ import annotations

import argparse
import json

from openprotein.selection import select_multimodal_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("policy")
    parser.add_argument("train_receipt")
    parser.add_argument("initializer_validation_receipt")
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=2,
        metavar=("TEXT_RECEIPT", "SEQUENCE_RECEIPT"),
        required=True,
    )
    args = parser.parse_args()
    receipt = select_multimodal_checkpoint(
        args.candidate,
        args.train_receipt,
        args.initializer_validation_receipt,
        args.policy,
        args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
