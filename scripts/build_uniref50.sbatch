#!/usr/bin/env bash
#SBATCH --job-name=uniref50-build
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/uniref-build-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/uniref-build-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
cd "$root/source"
"$root/runtime/venv/bin/python" -m openprotein.cli.build_uniref \
  "$root/data/raw/uniref/2026_02/uniref50.fasta.gz" \
  "$root/data/processed/uniref50-2026_02" \
  --source-definition "$root/source/data_sources/uniref50-2026_02.json" \
  --train-shards 96 \
  --validation-shards 1 \
  --test-shards 1 \
  --train-basis-points 9800 \
  --validation-basis-points 100
