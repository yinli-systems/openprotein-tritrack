#!/usr/bin/env bash
#SBATCH --job-name=sprot-index
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/sprot-index-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/sprot-index-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
release=2026_02
cd "$root/source"
"$root/runtime/venv/bin/python" -m openprotein.cli.build_swissprot_index \
  "$root/data/raw/uniprot/$release/uniprot_sprot.dat.gz" \
  "$root/data/processed/swissprot-$release/annotations.sqlite" \
  "$root/data/processed/swissprot-$release/manifest.json" \
  --source-definition "$root/source/data_sources/swissprot-$release.json"
