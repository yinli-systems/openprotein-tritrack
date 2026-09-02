#!/usr/bin/env bash
#SBATCH --job-name=mm1b-final-eval
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/mm1b-final-eval-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/mm1b-final-eval-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
checkpoint="${CHECKPOINT:?CHECKPOINT is required}"
pretrain_receipt="${PRETRAIN_RECEIPT:?PRETRAIN_RECEIPT is required}"
output_dir="${OUTPUT_DIR:?OUTPUT_DIR is required}"
[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
test ! -e "$output_dir"
test -f "$checkpoint/manifest.json"
test -f "$checkpoint/state.pt"
test -f "$pretrain_receipt"

staging="${output_dir}.staging-${SLURM_JOB_ID}"
test ! -e "$staging"
mkdir -p "$staging"
cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
python="$root/runtime/venv/bin/python"
model_config="$source_dir/configs/model/openprotein-1b-multimodal.json"
protein_tokenizer="$source_dir/configs/scratch/tokenizer/esm2-33.json"
text_tokenizer="$root/data/processed/text-tokenizer-swissprot-2026_02/tokenizer.model"
artifact_manifest="$root/manifests/esm2-t33-650m-ur50d.json"

"$python" -c '
import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
if not (
    receipt.get("status") == "passed"
    and receipt.get("evidence_level") == "PRETRAIN"
    and receipt.get("token_budget_reached") is True
    and int(receipt.get("tokens", -1)) >= int(receipt.get("target_tokens", 0))
):
    raise SystemExit("invalid PRETRAIN receipt")
' "$pretrain_receipt"

"$python" -m openprotein.cli.validate_sequence \
  --model-config "$model_config" \
  --checkpoint "$checkpoint" \
  --tokenizer-json "$protein_tokenizer" \
  --model-artifact-manifest "$artifact_manifest" \
  --validation-glob "$root/data/processed/uniref50-2026_02/validation/*.csv" \
  --data-manifest "$root/data/processed/uniref50-2026_02/manifest.json" \
  --output "$staging/sequence.json" \
  --max-batches 1000 \
  --sequence-length 256

"$python" -m openprotein.cli.validate_text \
  --model-config "$model_config" \
  --checkpoint "$checkpoint" \
  --protein-tokenizer-json "$protein_tokenizer" \
  --text-tokenizer-model "$text_tokenizer" \
  --model-artifact-manifest "$artifact_manifest" \
  --validation-glob "$root/data/processed/uniref50-swissprot-2026_02/validation/*.csv" \
  --data-manifest "$root/data/processed/uniref50-swissprot-2026_02/manifest.json" \
  --output "$staging/text.json" \
  --max-batches 125 \
  --batch-size 8 \
  --sequence-length 128 \
  --text-length 128

"$python" -m openprotein.cli.validate_structure \
  --model-config "$model_config" \
  --checkpoint "$checkpoint" \
  --protein-tokenizer-json "$protein_tokenizer" \
  --model-artifact-manifest "$artifact_manifest" \
  --validation-glob "$root/data/processed/uniref50-swissprot-2026_02/validation/*.csv" \
  --data-manifest "$root/data/processed/uniref50-swissprot-2026_02/manifest.json" \
  --structure-database "$root/data/processed/alphafold-swissprot-v6/structures.sqlite" \
  --structure-manifest "$root/data/processed/alphafold-swissprot-v6/manifest.json" \
  --output "$staging/structure.json" \
  --max-batches 1000 \
  --batch-size 1 \
  --sequence-length 256 \
  --geometry-pairs 128 \
  --minimum-plddt 70

(cd "$staging" && sha256sum sequence.json text.json structure.json > SHA256SUMS)
mv "$staging" "$output_dir"
printf 'evaluation_dir=%s\n' "$output_dir"
