"""Audit TorchProtein EC parse eligibility with a pinned RDKit version."""

from __future__ import annotations

import argparse
import json

from openprotein.ec_rdkit_audit import audit_ec_rdkit_cohort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", required=True)
    parser.add_argument("--source-definition", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--expected-rdkit-version")
    args = parser.parse_args()
    receipt = audit_ec_rdkit_cohort(
        args.source_zip,
        args.source_definition,
        args.output_directory,
        expected_rdkit_version=args.expected_rdkit_version,
        progress_callback=lambda split, total, valid, invalid: print(
            f"{split} total={total} valid={valid} invalid={invalid}",
            flush=True,
        ),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
