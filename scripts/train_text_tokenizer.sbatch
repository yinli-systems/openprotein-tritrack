#!/usr/bin/env bash
#SBATCH --job-name=protein-text-spm
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/text-tokenizer-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/text-tokenizer-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
dataset="$root/data/processed/uniref50-swissprot-2026_02"
cd "$root/source"
"$root/runtime/venv/bin/python" -m openprotein.cli.train_text_tokenizer \
  "$dataset/train/*.csv" \
  "$dataset/manifest.json" \
  "$root/data/processed/text-tokenizer-swissprot-2026_02" \
  --vocab-size 32768
