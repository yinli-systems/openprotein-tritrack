#!/usr/bin/env bash
#SBATCH --job-name=protein7b-pgym-smoke
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/proteingym-smoke-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/proteingym-smoke-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR is required}"
checkpoint="${CHECKPOINT:?CHECKPOINT is required}"
smoke_dir="${SMOKE_DIR:?SMOKE_DIR is required}"
case "$source_dir" in
  "$root"/source-snapshots/*) ;;
  *)
    printf 'SOURCE_DIR must be an immutable source snapshot: %s\n' "$source_dir" >&2
    exit 1
    ;;
esac
test ! -e "$smoke_dir"
cd "$source_dir"
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$root/cache/hf"
export XDG_CACHE_HOME="$root/cache/xdg"
export TORCH_HOME="$root/cache/torch"
"$root/runtime/venv/bin/python" scripts/prepare_distributed_eval_smoke.py "$smoke_dir"
"$root/runtime/venv/bin/python" -m openprotein.cli.seal_evaluation \
  "$checkpoint" \
  "$smoke_dir/manifest.json" \
  "$source_dir/configs/model/openprotein-7b.json" \
  /ssd/scxi253/esmfold2/models/Biohub_ESMC-6B/tokenizer.json \
  "$root/manifests/esmc-6b-local-snapshot.json" \
  "$smoke_dir/seal.json"
"$root/runtime/venv/bin/torchrun" --standalone --nproc-per-node=4 \
  -m openprotein.cli.evaluate_proteingym \
  "$smoke_dir/seal.json" "$smoke_dir/receipt.json" \
  --positions-per-batch 2
sha256sum "$smoke_dir/seal.json" "$smoke_dir/receipt.json"
