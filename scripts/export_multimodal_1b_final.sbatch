#!/usr/bin/env bash
#SBATCH --job-name=mm1b-final-export
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/mm1b-final-export-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/mm1b-final-export-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
checkpoint="${CHECKPOINT:?CHECKPOINT is required}"
pretrain_receipt="${PRETRAIN_RECEIPT:?PRETRAIN_RECEIPT is required}"
release_dir="${RELEASE_DIR:?RELEASE_DIR is required}"
reconstruction_receipt="${RECONSTRUCTION_RECEIPT:?RECONSTRUCTION_RECEIPT is required}"
[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
test ! -e "$release_dir"
test ! -e "$reconstruction_receipt"
cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
python="$root/runtime/venv/bin/python"
protein_tokenizer="$source_dir/configs/scratch/tokenizer/esm2-33.json"
artifact_manifest="$root/manifests/esm2-t33-650m-ur50d.json"

"$python" -m openprotein.cli.export_adapter \
  "$checkpoint" \
  "$source_dir/configs/model/openprotein-1b-multimodal.json" \
  "$protein_tokenizer" \
  "$artifact_manifest" \
  "$release_dir" \
  --text-tokenizer-model \
  "$root/data/processed/text-tokenizer-swissprot-2026_02/tokenizer.model" \
  --pretrain-receipt "$pretrain_receipt"

"$python" -m openprotein.cli.reconstruct_adapter \
  "$release_dir" \
  "$protein_tokenizer" \
  "$artifact_manifest" \
  "$reconstruction_receipt" \
  --device cuda \
  --dtype bfloat16

sha256sum \
  "$release_dir/adapter.safetensors" \
  "$release_dir/manifest.json" \
  "$release_dir/pretrain-receipt.json" \
  "$reconstruction_receipt"
