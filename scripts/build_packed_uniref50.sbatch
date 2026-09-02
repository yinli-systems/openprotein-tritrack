#!/usr/bin/env bash
#SBATCH --job-name=uniref50-pack
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/uniref-pack-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/uniref-pack-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
"$root/runtime/venv/bin/python" -m openprotein.cli.build_packed_sequences \
  "$root/data/processed/uniref50-2026_02/train/*.csv" \
  "$root/data/packed/uniref50-2026_02-esm2-33" \
  --tokenizer "$source_dir/configs/scratch/tokenizer/esm2-33.json" \
  --source-manifest "$root/data/processed/uniref50-2026_02/manifest.json" \
  --source-root "$root/data/processed/uniref50-2026_02"
