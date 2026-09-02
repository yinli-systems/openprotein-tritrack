#!/usr/bin/env bash
#SBATCH --job-name=mm1b-ret-stress-build
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/mm1b-ret-stress-build-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/mm1b-ret-stress-build-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
allowlist="${RECORD_ID_ALLOWLIST:?RECORD_ID_ALLOWLIST is required}"
output_dir="${OUTPUT_DIR:?OUTPUT_DIR is required}"
[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
[[ "$allowlist" == "$root/evaluations/"* ]]
[[ "$output_dir" == "$root/evaluations/"* ]]
test -f "$allowlist"
test ! -e "$output_dir"

staging="${output_dir}.staging-${SLURM_JOB_ID}"
test ! -e "$staging"
mkdir -p "$staging"
cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
python="$root/runtime/venv/bin/python"

"$python" -m openprotein.cli.build_retrieval_stress \
  --data-manifest \
    "$root/data/processed/uniref50-swissprot-2026_02/manifest.json" \
  --record-id-allowlist "$allowlist" \
  --output-csv "$staging/benchmark.csv" \
  --output-receipt "$staging/benchmark.json" \
  --distractors 100000 \
  --seed 61003

(cd "$staging" && sha256sum benchmark.csv benchmark.json > SHA256SUMS)
mv "$staging" "$output_dir"
chmod -R a-w "$output_dir"
printf 'evaluation_dir=%s\n' "$output_dir"
