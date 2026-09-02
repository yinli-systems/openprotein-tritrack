#!/usr/bin/env bash
#SBATCH --job-name=tritrack12-smoke
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:45:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/tritrack12-smoke-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/tritrack12-smoke-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir=${SOURCE_DIR:?SOURCE_DIR must be a content-addressed snapshot}
[[ "$source_dir" =~ ^$root/source-snapshots/[0-9a-f]{64}$ ]]
[[ ! -w "$source_dir" ]]
mkdir -p "$root/logs"
task_tmp="$root/tmp/tritrack12-smoke-${SLURM_JOB_ID}"
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
config="$source_dir/configs/train/multimodal-tritrack-1.2b-ddp4-smoke.json"
"$root/runtime/venv/bin/torchrun" --standalone --nproc_per_node=4 \
  -m openprotein.cli.train --config "$config" --max-steps 5

checkpoint="$root/runs/multimodal-tritrack-1.2b-ddp4-smoke/checkpoints/step-00000005"
[[ -f "$checkpoint/manifest.json" ]]
"$root/runtime/venv/bin/torchrun" --standalone --nproc_per_node=4 \
  -m openprotein.cli.train --config "$config" --resume "$checkpoint" --max-steps 10
