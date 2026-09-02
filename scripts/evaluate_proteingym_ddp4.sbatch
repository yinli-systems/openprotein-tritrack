#!/usr/bin/env bash
#SBATCH --job-name=protein7b-pgym
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/proteingym-eval-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/proteingym-eval-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR is required}"
seal="${EVALUATION_SEAL:?EVALUATION_SEAL is required}"
output="${OUTPUT:?OUTPUT is required}"
case "$source_dir" in
  "$root"/source-snapshots/*) ;;
  *)
    printf 'SOURCE_DIR must be an immutable source snapshot: %s\n' "$source_dir" >&2
    exit 1
    ;;
esac
cd "$source_dir"
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$root/cache/hf"
export XDG_CACHE_HOME="$root/cache/xdg"
export TORCH_HOME="$root/cache/torch"
"$root/runtime/venv/bin/torchrun" --standalone --nproc-per-node=4 \
  -m openprotein.cli.evaluate_proteingym \
  "$seal" "$output" \
  --positions-per-batch "${POSITIONS_PER_BATCH:-8}"
