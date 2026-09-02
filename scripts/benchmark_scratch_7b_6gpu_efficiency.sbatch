#!/usr/bin/env bash
#SBATCH --job-name=scratch7b-6gpu-eff
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/scratch7b-6gpu-eff-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/scratch7b-6gpu-eff-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]

cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"

configs=(
  configs/scratch/train/mgnify90-2024_04-7b-single-node-5090-6gpu-b43690-ga1-window4096-40.json
  configs/scratch/train/mgnify90-2024_04-7b-single-node-5090-6gpu-b21845-ga2-window4096-40.json
  configs/scratch/train/mgnify90-2024_04-7b-single-node-5090-6gpu-b21845-ga2-window4096-noreshard-40.json
  configs/scratch/train/mgnify90-2024_04-7b-single-node-5090-6gpu-b14563-ga3-window4096-40.json
)

successes=0
for config in "${configs[@]}"; do
  printf 'candidate_start=%s\n' "$config"
  set +e
  "$root/runtime/venv/bin/torchrun" --standalone --nproc_per_node=6 \
    -m openprotein.cli.train_scratch --config "$source_dir/$config"
  status=$?
  set -e
  printf 'candidate_exit=%s status=%d\n' "$config" "$status"
  if [[ "$status" -eq 0 ]]; then
    successes=$((successes + 1))
  fi
done

printf 'successful_candidates=%d total_candidates=%d\n' "$successes" "${#configs[@]}"
[[ "$successes" -gt 0 ]]

migration_config="$source_dir/configs/scratch/train/mgnify90-2024_04-7b-single-node-5090-6gpu-topology-migration-probe-b14563-ga3-5010.json"
resume="$root/runs/scratch-7b-mgnify90-2024_04-formal-50b-global-lr-v2/checkpoints/step-00005000"
printf 'migration_probe_start=%s resume=%s\n' "$migration_config" "$resume"
"$root/runtime/venv/bin/torchrun" --standalone --nproc_per_node=6 \
  -m openprotein.cli.train_scratch --config "$migration_config" --resume "$resume"
printf 'migration_probe_exit=%s status=0\n' "$migration_config"
