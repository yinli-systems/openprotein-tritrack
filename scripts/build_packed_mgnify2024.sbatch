#!/usr/bin/env bash
#SBATCH --job-name=mgnify24-pack
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-12:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/mgnify24-pack-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/mgnify24-pack-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
"$root/runtime/venv/bin/python" -m openprotein.cli.build_mgnify_packed \
  "$root/data/raw/mgnify/2024_04-litefold/shards" \
  "$root/data/packed/mgnify90-2024_04-litefold-50pct-esm2-33" \
  --tokenizer "$source_dir/configs/scratch/tokenizer/esm2-33.json" \
  --source-definition "$source_dir/data_sources/mgnify-proteins-2024_04-litefold.json" \
  --retrieval-receipt "$root/data/raw/mgnify/2024_04-litefold/retrieval.json" \
  --selection-basis-points 5000 \
  --train-shards 128 \
  --validation-shards 4 \
  --test-shards 4 \
  --minimum-train-residues 50000000000
