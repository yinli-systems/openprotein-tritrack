#!/usr/bin/env bash
#SBATCH --job-name=uniref-sprot-join
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/multimodal-join-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/multimodal-join-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
cd "$root/source"
"$root/runtime/venv/bin/python" -m openprotein.cli.join_multimodal \
  "$root/data/processed/uniref50-2026_02" \
  "$root/data/processed/swissprot-2026_02/annotations.sqlite" \
  "$root/data/processed/swissprot-2026_02/manifest.json" \
  "$root/data/processed/uniref50-swissprot-2026_02"
