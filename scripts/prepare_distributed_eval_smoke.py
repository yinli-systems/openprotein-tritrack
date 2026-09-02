#!/usr/bin/env python3
"""Create a tiny synthetic opaque-archive benchmark for DDP evaluator smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite synthetic smoke fixture: {output}")
    output.mkdir(parents=True)

    reference = output / "reference.csv"
    reference.write_text(
        "DMS_id,DMS_filename,UniProt_ID,target_seq,coarse_selection_type\n"
        "SMOKE_A,smoke_a.csv,SYNTH_A,ACDEFGHIK,Activity\n"
        "SMOKE_B,smoke_b.csv,SYNTH_B,LMNPQRSTV,Binding\n",
        encoding="utf-8",
    )
    assays = output / "assays.zip"
    with zipfile.ZipFile(assays, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "assays/smoke_a.csv",
            "mutant,DMS_score\nA1C,-1.0\nC2A,1.0\n",
        )
        archive.writestr(
            "assays/smoke_b.csv",
            "mutant,DMS_score\nL1A,-1.0\nM2A,1.0\n",
        )
    scores = output / "scores.zip"
    with zipfile.ZipFile(scores, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "scores/ESM2/15B/SMOKE_A.csv",
            "mutant,esm2_t48_15B_UR50D\nA1C,-1.0\nC2A,1.0\n",
        )
        archive.writestr(
            "scores/ESM2/15B/SMOKE_B.csv",
            "mutant,esm2_t48_15B_UR50D\nL1A,-1.0\nM2A,1.0\n",
        )
    manifest = {
        "schema_version": 1,
        "name": "synthetic distributed evaluator smoke fixture",
        "synthetic": True,
        "archive_protocol": "proteingym-v1.3-opaque-zip",
        "dataset_root": str(output),
        "reference_file": artifact(reference),
        "assay_archive": artifact(assays),
        "score_archive": artifact(scores),
        "baselines": {
            "ESM2-15B": {
                "member_prefix": "ESM2/15B",
                "score_column": "esm2_t48_15B_UR50D",
                "directionality": 1.0,
            }
        },
        "claim_gate": {
            "required_baselines": ["ESM2-15B", "ESMC-6B-local"],
            "minimum_margin": 0.0,
            "paired_bootstrap_samples": 10000,
            "paired_bootstrap_seed": 1701,
            "minimum_paired_bootstrap_lower_bound": 0.0,
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
