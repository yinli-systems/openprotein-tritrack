#!/usr/bin/env bash
#SBATCH --job-name=mm1b-ret-stress
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/mm1b-ret-stress-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/mm1b-ret-stress-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
checkpoint="${CHECKPOINT:?CHECKPOINT is required}"
benchmark_dir="${BENCHMARK_DIR:?BENCHMARK_DIR is required}"
output_dir="${OUTPUT_DIR:?OUTPUT_DIR is required}"
[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
[[ "$benchmark_dir" == "$root/evaluations/"* ]]
[[ "$output_dir" == "$root/evaluations/"* ]]
test -f "$checkpoint/manifest.json"
test -f "$checkpoint/state.pt"
test -f "$benchmark_dir/benchmark.csv"
test -f "$benchmark_dir/benchmark.json"
test ! -e "$output_dir"

staging="${output_dir}.staging-${SLURM_JOB_ID}"
test ! -e "$staging"
mkdir -p "$staging"
cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
python="$root/runtime/venv/bin/python"

"$python" scripts/evaluate_multimodal_retrieval_stress.py \
  --model-config "$source_dir/configs/model/openprotein-1b-multimodal.json" \
  --checkpoint "$checkpoint" \
  --protein-tokenizer-json "$source_dir/configs/scratch/tokenizer/esm2-33.json" \
  --text-tokenizer-model \
    "$root/data/processed/text-tokenizer-swissprot-2026_02/tokenizer.model" \
  --model-artifact-manifest "$root/manifests/esm2-t33-650m-ur50d.json" \
  --benchmark-csv "$benchmark_dir/benchmark.csv" \
  --benchmark-receipt "$benchmark_dir/benchmark.json" \
  --output "$staging/retrieval.json" \
  --batch-size 8 \
  --sequence-length 256 \
  --text-length 256 \
  --metric-chunk-size 16 \
  --seed 3701

(cd "$staging" && sha256sum retrieval.json > SHA256SUMS)
mv "$staging" "$output_dir"
chmod -R a-w "$output_dir"
printf 'evaluation_dir=%s\n' "$output_dir"
