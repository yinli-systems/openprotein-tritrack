"""Select a sequence checkpoint from exactly comparable validation receipts."""

from __future__ import annotations

import argparse
import json

from openprotein.selection import select_sequence_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("validation_receipts", nargs="+")
    args = parser.parse_args()
    receipt = select_sequence_checkpoint(args.validation_receipts, args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
