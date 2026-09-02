#!/usr/bin/env bash
#SBATCH --job-name=mm1b-batchcap
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/mm1b-batchcap-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/mm1b-batchcap-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir=${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}
config=${CONFIG:?CONFIG must name a frozen batch-cap config}
[[ "$source_dir" =~ ^$root/source-snapshots/[0-9a-f]{64}$ ]]
[[ ! -w "$source_dir" ]]
[[ "$config" == "$source_dir/configs/train/"*.json ]]
task_tmp="$root/tmp/mm1b-batchcap-${SLURM_JOB_ID}"
mkdir -p "$task_tmp" "$root/logs" "$root/cache/hf" "$root/cache/xdg" "$root/cache/torchinductor"
cd "$source_dir"
export PYTHONPATH="$source_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="$task_tmp"
export HF_HOME="$root/cache/hf"
export XDG_CACHE_HOME="$root/cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="$root/cache/torchinductor"

"$root/runtime/venv/bin/torchrun" --standalone --nproc_per_node=4 \
  -m openprotein.cli.train --config "$config"
