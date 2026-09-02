"""Audit whether multimodal train and validation rows form a temporal split."""

from __future__ import annotations

import argparse
import json

from openprotein.temporal_audit import audit_swissprot_temporal_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--swissprot", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = audit_swissprot_temporal_split(
        args.data_manifest,
        args.swissprot,
        args.cutoff,
        args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
