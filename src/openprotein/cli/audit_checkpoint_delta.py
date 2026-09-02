"""Audit that a staged checkpoint changed only predeclared parameter groups."""

from __future__ import annotations

import argparse
import json

from openprotein.checkpoint_audit import audit_checkpoint_delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_checkpoint")
    parser.add_argument("target_checkpoint")
    parser.add_argument("output")
    parser.add_argument("--allow-prefix", action="append", required=True)
    args = parser.parse_args()
    receipt = audit_checkpoint_delta(
        args.source_checkpoint,
        args.target_checkpoint,
        args.allow_prefix,
        args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
