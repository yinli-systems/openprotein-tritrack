#!/usr/bin/env bash
#SBATCH --job-name=protein7b-seq-gate-val
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/seq-gate-val-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/seq-gate-val-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR is required}"
checkpoint="${CHECKPOINT:?CHECKPOINT is required}"
output="${OUTPUT:?OUTPUT is required}"
case "$source_dir" in
  "$root"/source-snapshots/*) ;;
  *) printf 'SOURCE_DIR must be an immutable source snapshot: %s\n' "$source_dir" >&2; exit 1 ;;
esac
[[ ! -w "$source_dir" ]]
cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
"$root/runtime/venv/bin/python" -m openprotein.cli.validate_sequence \
  --model-config "$source_dir/configs/model/openprotein-7b.json" \
  --checkpoint "$checkpoint" \
  --tokenizer-json "/ssd/scxi253/esmfold2/models/Biohub_ESMC-6B/tokenizer.json" \
  --model-artifact-manifest "$root/manifests/esmc-6b-local-snapshot.json" \
  --validation-glob "$root/data/processed/uniref50-2026_02/validation/*.csv" \
  --data-manifest "$root/data/processed/uniref50-2026_02/manifest.json" \
  --output "$output" \
  --max-batches 1000 \
  --sequence-length 256
