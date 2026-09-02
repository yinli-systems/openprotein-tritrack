#!/usr/bin/env bash
#SBATCH --job-name=protrek-ret-stress
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/protrek-ret-stress-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/protrek-ret-stress-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
weights_root="${PROTREK_WEIGHTS_ROOT:?PROTREK_WEIGHTS_ROOT is required}"
checkpoint="${PROTREK_CHECKPOINT:?PROTREK_CHECKPOINT is required}"
benchmark_dir="${BENCHMARK_DIR:?BENCHMARK_DIR is required}"
output_dir="${OUTPUT_DIR:?OUTPUT_DIR is required}"
repository_commit="${PROTREK_REPOSITORY_COMMIT:?PROTREK_REPOSITORY_COMMIT is required}"
model_revision="${PROTREK_MODEL_REVISION:?PROTREK_MODEL_REVISION is required}"
python="${PROTREK_PYTHON:?PROTREK_PYTHON is required}"
[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
[[ "$benchmark_dir" == "$root/evaluations/"* ]]
[[ "$output_dir" == "$root/evaluations/"* ]]
test -f "$checkpoint"
test -x "$python"
test -f "$benchmark_dir/benchmark.csv"
test -f "$benchmark_dir/benchmark.json"
test ! -e "$output_dir"

staging="${output_dir}.staging-${SLURM_JOB_ID}"
test ! -e "$staging"
mkdir -p "$staging"
cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"

"$python" scripts/evaluate_protrek_retrieval_stress.py \
  --weights-root "$weights_root" \
  --checkpoint "$checkpoint" \
  --benchmark-csv "$benchmark_dir/benchmark.csv" \
  --benchmark-receipt "$benchmark_dir/benchmark.json" \
  --output "$staging/retrieval.json" \
  --repository-commit "$repository_commit" \
  --model-revision "$model_revision" \
  --batch-size 8 \
  --sequence-length 256 \
  --text-length 256 \
  --metric-chunk-size 16 \
  --seed 3701

(cd "$staging" && sha256sum retrieval.json > SHA256SUMS)
mv "$staging" "$output_dir"
chmod -R a-w "$output_dir"
printf 'evaluation_dir=%s\n' "$output_dir"
