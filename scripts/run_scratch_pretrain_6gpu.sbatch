#!/usr/bin/env bash
#SBATCH --job-name=scratch7b-pretrain-6g
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=48
#SBATCH --time=1-12:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/scratch-pretrain-6gpu-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/scratch-pretrain-6gpu-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
config="${CONFIG:?CONFIG must name the frozen scratch pretraining config}"
segment_end_step="${SEGMENT_END_STEP:?SEGMENT_END_STEP must be a checkpoint boundary}"
[[ "$source_dir" =~ ^$root/source-snapshots/[0-9a-f]{64}$ ]]
[[ ! -w "$source_dir" ]]
[[ "$config" == "$source_dir/configs/scratch/train/"* ]]
[[ "$segment_end_step" =~ ^[1-9][0-9]*$ ]]

cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
args=(--config "$config" --segment-end-step "$segment_end_step")
if [[ -n "${RESUME:-}" ]]; then
  args+=(--resume "$RESUME")
fi
"$root/runtime/venv/bin/torchrun" --standalone --nproc_per_node=6 \
  -m openprotein.cli.train_scratch "${args[@]}"
