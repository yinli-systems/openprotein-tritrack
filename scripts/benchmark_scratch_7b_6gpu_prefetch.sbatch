#!/usr/bin/env bash
#SBATCH --job-name=scratch7b-6gpu-pf
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/scratch7b-6gpu-pf-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/scratch7b-6gpu-pf-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
policy="${SELECTION_POLICY:?SELECTION_POLICY must name the read-only policy}"
policy_sha256="${SELECTION_POLICY_SHA256:?SELECTION_POLICY_SHA256 is required}"
[[ "$source_dir" =~ ^$root/source-snapshots/[0-9a-f]{64}$ ]]
[[ ! -w "$source_dir" ]]
[[ -f "$policy" ]]
[[ ! -w "$policy" ]]
printf '%s  %s\n' "$policy_sha256" "$policy" | sha256sum --check --status

cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"

configs=(
  configs/scratch/train/mgnify90-2024_04-7b-single-node-5090-6gpu-b14563-ga3-window4096-prefetch-maxb256-60.json
  configs/scratch/train/mgnify90-2024_04-7b-single-node-5090-6gpu-b14563-ga3-window4096-prefetch-maxb384-60.json
)

for config in "${configs[@]}"; do
  output_dir="$($root/runtime/venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_dir"])' "$config")"
  [[ ! -e "$output_dir" ]]
done

for config in "${configs[@]}"; do
  printf 'candidate_start=%s\n' "$config"
  set +e
  "$root/runtime/venv/bin/torchrun" --standalone --nproc_per_node=6 \
    -m openprotein.cli.train_scratch --config "$source_dir/$config"
  status=$?
  set -e
  printf 'candidate_exit=%s status=%d\n' "$config" "$status"
done

selection_output="/data/run01/scxi253/pretraining2-archive/receipts/scratch-7b-6gpu-prefetch-selection-job-${SLURM_JOB_ID}.json"
"$root/runtime/venv/bin/python" scripts/select_scratch_7b_6gpu_batchcap.py \
  --source-dir "$source_dir" \
  --policy "$policy" \
  --policy-sha256 "$policy_sha256" \
  --output "$selection_output"
printf 'selection_receipt=%s\n' "$selection_output"
