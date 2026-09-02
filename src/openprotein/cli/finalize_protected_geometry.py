"""Finalize the protected geometry gate under a frozen policy."""

from __future__ import annotations

import argparse
import json

from openprotein.protected_geometry import finalize_protected_geometry


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "policy", "upstream-selection-receipt", "train-receipt",
        "initializer-sequence-receipt", "initializer-text-receipt",
        "initializer-structure-receipt", "final-sequence-receipt",
        "final-text-receipt", "final-structure-receipt", "checkpoint-audit-receipt",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    receipt = finalize_protected_geometry(
        policy_path=args.policy,
        upstream_selection_receipt=args.upstream_selection_receipt,
        train_receipt=args.train_receipt,
        initializer_sequence_receipt=args.initializer_sequence_receipt,
        initializer_text_receipt=args.initializer_text_receipt,
        initializer_structure_receipt=args.initializer_structure_receipt,
        final_sequence_receipt=args.final_sequence_receipt,
        final_text_receipt=args.final_text_receipt,
        final_structure_receipt=args.final_structure_receipt,
        checkpoint_audit_receipt=args.checkpoint_audit_receipt,
        output=args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
