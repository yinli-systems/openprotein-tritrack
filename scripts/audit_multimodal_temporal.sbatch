#!/usr/bin/env bash
#SBATCH --job-name=mm1b-temporal
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/mm1b-temporal-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/mm1b-temporal-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
output_dir="${OUTPUT_DIR:?OUTPUT_DIR is required}"
[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
[[ "$output_dir" == "$root/evaluations/"* ]]
test ! -e "$output_dir"

staging="${output_dir}.staging-${SLURM_JOB_ID}"
test ! -e "$staging"
mkdir -p "$staging"
cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
python="$root/runtime/venv/bin/python"

"$python" -m openprotein.cli.audit_temporal \
  --data-manifest \
    "$root/data/processed/uniref50-swissprot-2026_02/manifest.json" \
  --swissprot "$root/data/raw/uniprot/2026_02/uniprot_sprot.dat.gz" \
  --cutoff 2024-01-01 \
  --output "$staging/temporal.json"

(cd "$staging" && sha256sum temporal.json > SHA256SUMS)
mv "$staging" "$output_dir"
chmod -R a-w "$output_dir"
printf 'evaluation_dir=%s\n' "$output_dir"
