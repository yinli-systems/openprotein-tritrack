#!/usr/bin/env bash
#SBATCH --job-name=protein7b-release
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/release-reconstruct-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/release-reconstruct-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
checkpoint="${CHECKPOINT:?CHECKPOINT is required}"
release_dir="${RELEASE_DIR:?RELEASE_DIR is required}"
receipt="${RECONSTRUCTION_RECEIPT:?RECONSTRUCTION_RECEIPT is required}"
text_tokenizer="${TEXT_TOKENIZER_MODEL:?TEXT_TOKENIZER_MODEL is required}"
[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
test ! -e "$release_dir"
test ! -e "$receipt"
cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"

"$root/runtime/venv/bin/python" -m openprotein.cli.export_adapter \
  "$checkpoint" \
  "$source_dir/configs/model/openprotein-7b.json" \
  "/ssd/scxi253/esmfold2/models/Biohub_ESMC-6B/tokenizer.json" \
  "$root/manifests/esmc-6b-local-snapshot.json" \
  "$release_dir" \
  --text-tokenizer-model "$text_tokenizer"

"$root/runtime/venv/bin/python" -m openprotein.cli.reconstruct_adapter \
  "$release_dir" \
  "/ssd/scxi253/esmfold2/models/Biohub_ESMC-6B/tokenizer.json" \
  "$root/manifests/esmc-6b-local-snapshot.json" \
  "$receipt" \
  --backbone-root "/ssd/scxi253/esmfold2/models/Biohub_ESMC-6B" \
  --device cuda \
  --dtype bfloat16
