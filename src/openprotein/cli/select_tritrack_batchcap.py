"""Select a TriTrack batch-cap result under a frozen efficiency policy."""

from __future__ import annotations

import argparse
import json

from openprotein.tritrack_selection import select_tritrack_batchcap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = select_tritrack_batchcap(args.policy, args.status, args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
