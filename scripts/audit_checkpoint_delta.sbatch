#!/usr/bin/env bash
#SBATCH --job-name=protein7b-ckpt-audit
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/checkpoint-audit-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/checkpoint-audit-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR is required}"
source_checkpoint="${SOURCE_CHECKPOINT:?SOURCE_CHECKPOINT is required}"
target_checkpoint="${TARGET_CHECKPOINT:?TARGET_CHECKPOINT is required}"
output="${OUTPUT:?OUTPUT is required}"
allowed_prefix="${ALLOWED_PREFIX:?ALLOWED_PREFIX is required}"
case "$source_dir" in
  "$root"/source-snapshots/*) ;;
  *)
    printf 'SOURCE_DIR must be an immutable source snapshot: %s\n' "$source_dir" >&2
    exit 1
    ;;
esac
cd "$source_dir"
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
"$root/runtime/venv/bin/python" -m openprotein.cli.audit_checkpoint_delta \
  "$source_checkpoint" "$target_checkpoint" "$output" \
  --allow-prefix "$allowed_prefix"
