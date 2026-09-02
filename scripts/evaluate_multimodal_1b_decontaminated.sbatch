#!/usr/bin/env bash
#SBATCH --job-name=mm1b-clean-ret
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/mm1b-clean-ret-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/mm1b-clean-ret-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
checkpoint="${CHECKPOINT:?CHECKPOINT is required}"
allowlist="${RECORD_ID_ALLOWLIST:?RECORD_ID_ALLOWLIST is required}"
output_dir="${OUTPUT_DIR:?OUTPUT_DIR is required}"
[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
[[ "$allowlist" == "$root/evaluations/"* ]]
[[ "$output_dir" == "$root/evaluations/"* ]]
test ! -e "$output_dir"
test -f "$checkpoint/manifest.json"
test -f "$checkpoint/state.pt"
test -f "$allowlist"

staging="${output_dir}.staging-${SLURM_JOB_ID}"
test ! -e "$staging"
mkdir -p "$staging"
cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
python="$root/runtime/venv/bin/python"

"$python" -m openprotein.cli.validate_text \
  --model-config "$source_dir/configs/model/openprotein-1b-multimodal.json" \
  --checkpoint "$checkpoint" \
  --protein-tokenizer-json "$source_dir/configs/scratch/tokenizer/esm2-33.json" \
  --text-tokenizer-model \
    "$root/data/processed/text-tokenizer-swissprot-2026_02/tokenizer.model" \
  --model-artifact-manifest "$root/manifests/esm2-t33-650m-ur50d.json" \
  --validation-glob \
    "$root/data/processed/uniref50-swissprot-2026_02/validation/*.csv" \
  --data-manifest \
    "$root/data/processed/uniref50-swissprot-2026_02/manifest.json" \
  --record-id-allowlist "$allowlist" \
  --output "$staging/retrieval.json" \
  --max-batches 1000 \
  --batch-size 8 \
  --sequence-length 256 \
  --text-length 256

(cd "$staging" && sha256sum retrieval.json > SHA256SUMS)
mv "$staging" "$output_dir"
chmod -R a-w "$output_dir"
printf 'evaluation_dir=%s\n' "$output_dir"
