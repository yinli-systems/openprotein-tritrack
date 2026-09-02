#!/usr/bin/env bash
#SBATCH --job-name=protrek650m-ret
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/protrek650m-ret-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/protrek650m-ret-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
weights_root="${PROTREK_WEIGHTS_ROOT:?PROTREK_WEIGHTS_ROOT is required}"
checkpoint="${PROTREK_CHECKPOINT:?PROTREK_CHECKPOINT is required}"
output_dir="${OUTPUT_DIR:?OUTPUT_DIR is required}"
repository_commit="${PROTREK_REPOSITORY_COMMIT:?PROTREK_REPOSITORY_COMMIT is required}"
model_revision="${PROTREK_MODEL_REVISION:?PROTREK_MODEL_REVISION is required}"
[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
test ! -e "$output_dir"
test -f "$checkpoint"
test -f "$weights_root/esm2_t33_650M_UR50D/config.json"
test -f "$weights_root/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/config.json"

staging="${output_dir}.staging-${SLURM_JOB_ID}"
test ! -e "$staging"
mkdir -p "$staging"
cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
python="${PROTREK_PYTHON:-$root/runtime/venv/bin/python}"

"$python" scripts/evaluate_protrek_retrieval.py \
  --weights-root "$weights_root" \
  --checkpoint "$checkpoint" \
  --validation-glob "$root/data/processed/uniref50-swissprot-2026_02/validation/*.csv" \
  --data-manifest "$root/data/processed/uniref50-swissprot-2026_02/manifest.json" \
  --output "$staging/retrieval.json" \
  --repository-commit "$repository_commit" \
  --model-revision "$model_revision" \
  --batch-size 8 \
  --sequence-length 256 \
  --text-length 256 \
  --seed 3701

(cd "$staging" && sha256sum retrieval.json > SHA256SUMS)
mv "$staging" "$output_dir"
chmod -R a-w "$output_dir"
printf 'evaluation_dir=%s\n' "$output_dir"
