#!/usr/bin/env bash
#SBATCH --job-name=protein7b-candidate
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/candidate-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/candidate-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR is required}"
train_config="${TRAIN_CONFIG:?TRAIN_CONFIG is required}"
case "$source_dir" in
  "$root"/source-candidates/*|"$root"/source-snapshots/*) ;;
  *)
    printf 'SOURCE_DIR must be under %s/source-candidates or source-snapshots: %s\n' \
      "$root" "$source_dir" >&2
    exit 1
    ;;
esac
cd "$source_dir"
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$root/cache/hf"
export XDG_CACHE_HOME="$root/cache/xdg"
export TORCH_HOME="$root/cache/torch"
args=(--config "$train_config")
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
