#!/usr/bin/env bash
#SBATCH --job-name=tritrack-hp6-cap
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=36
#SBATCH --time=01:30:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/tritrack-hp6-cap-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/tritrack-hp6-cap-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir=${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}
selection_output=${SELECTION_OUTPUT:?SELECTION_OUTPUT is required}
[[ "$source_dir" =~ ^$root/source-snapshots/[0-9a-f]{64}$ ]]
[[ ! -w "$source_dir" ]]
[[ "$selection_output" == "$root/receipts/tritrack-hp6-batchcap-"*.json ]]
test ! -e "$selection_output"

mkdir -p "$root/logs" "$root/receipts" "$root/tmp" \
  "$root/cache/hf" "$root/cache/xdg" "$root/cache/torchinductor"
task_tmp="$root/tmp/tritrack-hp6-cap-${SLURM_JOB_ID}"
status_tmp="$task_tmp/status.tsv"
status_final="$root/receipts/tritrack-hp6-batchcap-status-${SLURM_JOB_ID}.tsv"
mkdir "$task_tmp"
test ! -e "$status_final"
export PYTHONPATH="$source_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="$task_tmp"
export HF_HOME="$root/cache/hf"
export XDG_CACHE_HOME="$root/cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="$root/cache/torchinductor"
python="$root/runtime/venv/bin/python"
torchrun="$root/runtime/venv/bin/torchrun"
configs=(
  configs/train/tritrack-1.2b-hp6-seq512-mbs12-probe.json
  configs/train/tritrack-1.2b-hp6-seq512-mbs16-probe.json
  configs/train/tritrack-1.2b-hp6-seq512-mbs20-probe.json
)

cd "$source_dir"
for relative in "${configs[@]}"; do
  output_dir=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_dir"])' "$relative")
  test ! -e "$output_dir"
  set +e
  "$torchrun" --standalone --nproc_per_node=6 \
    -m openprotein.cli.train --config "$source_dir/$relative"
  code=$?
  set -e
  printf '%s\t%s\n' "$relative" "$code" >> "$status_tmp"
done

mv "$status_tmp" "$status_final"
"$python" -m openprotein.cli.select_tritrack_batchcap \
  --policy "$source_dir/configs/train/tritrack-1.2b-hp6-batchcap-policy.json" \
  --status "$status_final" \
  --output "$selection_output"
chmod a-w "$status_final" "$selection_output"
printf 'selection=%s\n' "$selection_output"
