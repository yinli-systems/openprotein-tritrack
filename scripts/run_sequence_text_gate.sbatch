#!/usr/bin/env bash
#SBATCH --job-name=protein7b-seqtext-gate
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/seqtext-gate-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/seqtext-gate-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR is required}"
config="${TRAIN_CONFIG:?TRAIN_CONFIG is required}"
init_checkpoint="${INIT_CHECKPOINT:?INIT_CHECKPOINT is required}"
expected_step="${EXPECTED_INIT_STEP:?EXPECTED_INIT_STEP is required}"
expected_manifest_sha256="${EXPECTED_INIT_MANIFEST_SHA256:?EXPECTED_INIT_MANIFEST_SHA256 is required}"
expected_state_sha256="${EXPECTED_INIT_STATE_SHA256:?EXPECTED_INIT_STATE_SHA256 is required}"
case "$source_dir" in
  "$root"/source-snapshots/*) ;;
  *) printf 'SOURCE_DIR must be an immutable source snapshot: %s\n' "$source_dir" >&2; exit 1 ;;
esac
[[ ! -w "$source_dir" ]]
[[ "$config" == "$source_dir"/configs/train/* ]]
[[ -f "$init_checkpoint/manifest.json" && -f "$init_checkpoint/state.pt" ]]
actual_manifest_sha256="$(sha256sum "$init_checkpoint/manifest.json" | awk '{print $1}')"
[[ "$actual_manifest_sha256" == "$expected_manifest_sha256" ]]
read -r actual_step actual_state_sha256 < <(
  "$root/runtime/venv/bin/python" - "$init_checkpoint/manifest.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["step"], payload["state_sha256"])
PY
)
[[ "$actual_step" == "$expected_step" ]]
[[ "$actual_state_sha256" == "$expected_state_sha256" ]]
cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$root/cache/hf"
export XDG_CACHE_HOME="$root/cache/xdg"
export TORCH_HOME="$root/cache/torch"
"$root/runtime/venv/bin/python" -m openprotein.cli.train \
  --config "$config" \
  --init-checkpoint "$init_checkpoint"
