#!/usr/bin/env bash
#SBATCH --partition=gpu_5090
#SBATCH --time=1-12:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/scratch-scale-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/scratch-scale-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
config="${CONFIG:?CONFIG must name a scratch training config}"
gpu_count="${GPU_COUNT:?GPU_COUNT must match the requested Slurm GPU count}"
[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
[[ "$config" == "$source_dir/configs/scratch/train/"* ]]
[[ "$gpu_count" =~ ^[1-9][0-9]*$ ]]
cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
args=(--config "$config")
if [[ -n "${RESUME:-}" ]]; then
  args+=(--resume "$RESUME")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  args+=(--max-steps "$MAX_STEPS")
fi
"$root/runtime/venv/bin/torchrun" --standalone --nproc_per_node="$gpu_count" \
  -m openprotein.cli.train_scratch "${args[@]}"
