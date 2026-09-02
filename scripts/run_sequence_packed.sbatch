#!/usr/bin/env bash
#SBATCH --job-name=protein7b-seq-packed
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=1-12:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/sequence-packed-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/sequence-packed-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
config="${CONFIG:-$source_dir/configs/train/uniref50-official-ddp4-packed-1b.json}"
[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$root/cache/hf"
export XDG_CACHE_HOME="$root/cache/xdg"
export TORCH_HOME="$root/cache/torch"
args=(--config "$config")
if [[ -n "${MAX_STEPS:-}" ]]; then
  args+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "${RESUME:-}" ]]; then
  args+=(--resume "$RESUME")
fi
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  args+=(--init-checkpoint "$INIT_CHECKPOINT")
fi
"$root/runtime/venv/bin/torchrun" --standalone --nproc_per_node=4 \
  -m openprotein.cli.train "${args[@]}"
