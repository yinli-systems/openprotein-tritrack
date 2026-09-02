#!/usr/bin/env bash
#SBATCH --job-name=protein7b-pilot
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/pilot-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/pilot-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
cd "$root/source"
export HF_HOME="$root/cache/hf"
export XDG_CACHE_HOME="$root/cache/xdg"
export TORCH_HOME="$root/cache/torch"
args=(
  --config "${TRAIN_CONFIG:-$root/source/configs/train/uniref50-pilot.json}"
)
if [[ -n "${MAX_STEPS:-}" ]]; then
  args+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "${RESUME:-}" ]]; then
  args+=(--resume "$RESUME")
fi
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  args+=(--init-checkpoint "$INIT_CHECKPOINT")
fi
"$root/runtime/venv/bin/python" -m openprotein.cli.train "${args[@]}"
