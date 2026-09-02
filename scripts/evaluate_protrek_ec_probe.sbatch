#!/usr/bin/env bash
#SBATCH --job-name=protrek-ec-probe
#SBATCH --partition=hp_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/protrek-ec-probe-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/protrek-ec-probe-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
benchmark_dir="${BENCHMARK_DIR:?BENCHMARK_DIR is required}"
output_dir="${OUTPUT_DIR:?OUTPUT_DIR is required}"
protocol_iteration_note="${PROTOCOL_ITERATION_NOTE:-The probe horizon was extended after v1 and v2 public-test access because all seeds still selected the terminal validation checkpoint; v3 is a convergence audit, not a first-look held-out evaluation.}"
weights_root="$root/baselines/protrek/hf-f2a9d59f333f191fd9268c0f8faf41a1100937c3"
checkpoint="$weights_root/ProTrek_650M.pt"
[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
[[ "$benchmark_dir" == "$root/data/processed/"* ]]
[[ "$output_dir" == "$root/evaluations/"* ]]
test ! -e "$output_dir"
test -f "$benchmark_dir/benchmark.csv"
test -f "$benchmark_dir/receipt.json"
test -f "$checkpoint"

staging="${output_dir}.staging-${SLURM_JOB_ID}"
test ! -e "$staging"
mkdir -p "$staging"
cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python="$root/runtime/protrek-venv/bin/python"

"$python" scripts/evaluate_protrek_ec_probe.py \
  --weights-root "$weights_root" \
  --checkpoint "$checkpoint" \
  --benchmark-csv "$benchmark_dir/benchmark.csv" \
  --benchmark-receipt "$benchmark_dir/receipt.json" \
  --output "$staging/ec-probe.json" \
  --repository-commit 48715468b8daa08236d05ff3310cd61db1896e94 \
  --model-revision f2a9d59f333f191fd9268c0f8faf41a1100937c3 \
  --batch-size 8 \
  --sequence-length 552 \
  --probe-steps 20000 \
  --probe-validation-interval 20 \
  --probe-minimum-steps 2000 \
  --probe-patience-validations 50 \
  --protocol-iteration-note "$protocol_iteration_note"

(cd "$staging" && sha256sum ec-probe.json > SHA256SUMS)
mv "$staging" "$output_dir"
chmod -R a-w "$output_dir"
printf 'evaluation_dir=%s\n' "$output_dir"
