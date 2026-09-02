#!/usr/bin/env bash
#SBATCH --job-name=mm1b-pretrain
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=7-00:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/mm1b-pretrain-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/mm1b-pretrain-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir=${SOURCE_DIR:?SOURCE_DIR must be a content-addressed snapshot}
[[ "$source_dir" =~ ^$root/source-snapshots/[0-9a-f]{64}$ ]]
[[ ! -w "$source_dir" ]]
mkdir -p "$root/logs"
task_tmp="$root/tmp/mm1b-formal-${SLURM_JOB_ID}"
mkdir -p "$task_tmp" "$root/cache/hf" "$root/cache/xdg" "$root/cache/torchinductor"
export PYTHONPATH="$source_dir/src"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="$task_tmp"
export HF_HOME="$root/cache/hf"
export XDG_CACHE_HOME="$root/cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="$root/cache/torchinductor"

cd "$source_dir"
args=(--config "$source_dir/configs/train/multimodal-1b-ddp4-1btoken.json")
if [[ -n "${RESUME:-}" ]]; then
  args+=(--resume "$RESUME")
fi
"$root/runtime/venv/bin/torchrun" \
  --standalone \
  --nproc_per_node=4 \
  -m openprotein.cli.train \
  "${args[@]}"
