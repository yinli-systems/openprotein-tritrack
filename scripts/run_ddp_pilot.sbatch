#!/usr/bin/env bash
#SBATCH --job-name=protein7b-ddp4
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/ddp-pilot-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/ddp-pilot-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
cd "$root/source"
export HF_HOME="$root/cache/hf"
export XDG_CACHE_HOME="$root/cache/xdg"
export TORCH_HOME="$root/cache/torch"
args=(
  --config "${CONFIG:-$root/source/configs/train/uniref50-official-ddp4-pilot.json}"
)
if [[ -n "${MAX_STEPS:-}" ]]; then
  args+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "${RESUME:-}" ]]; then
  args+=(--resume "$RESUME")
fi
"$root/runtime/venv/bin/torchrun" --standalone --nproc_per_node=4 \
  -m openprotein.cli.train "${args[@]}"
