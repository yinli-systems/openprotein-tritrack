"""Join exact Swiss-Prot annotations onto split UniRef representatives."""

from __future__ import annotations

import argparse
import json

from openprotein.data.multimodal import join_uniref_swissprot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("uniref_root")
    parser.add_argument("swissprot_database")
    parser.add_argument("swissprot_manifest")
    parser.add_argument("output_root")
    args = parser.parse_args()
    manifest = join_uniref_swissprot(
        args.uniref_root,
        args.swissprot_database,
        args.swissprot_manifest,
        args.output_root,
    )
    print(
        json.dumps(
            {"join_counts": manifest["join_counts"], "split_totals": manifest["split_totals"]},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
