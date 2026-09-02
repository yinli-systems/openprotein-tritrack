#!/usr/bin/env bash
#SBATCH --job-name=protein7b-structure-gate-val
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/structure-gate-val-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/structure-gate-val-%j.err

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
"$root/runtime/venv/bin/python" -m openprotein.cli.validate_structure \
  --model-config "$source_dir/configs/model/openprotein-7b.json" \
  --checkpoint "$checkpoint" \
  --protein-tokenizer-json "/ssd/scxi253/esmfold2/models/Biohub_ESMC-6B/tokenizer.json" \
  --model-artifact-manifest "$root/manifests/esmc-6b-local-snapshot.json" \
  --validation-glob "$root/data/processed/uniref50-swissprot-2026_02/validation/*.csv" \
  --data-manifest "$root/data/processed/uniref50-swissprot-2026_02/manifest.json" \
  --structure-database "$root/data/processed/alphafold-swissprot-v6/structures.sqlite" \
  --structure-manifest "$root/data/processed/alphafold-swissprot-v6/manifest.json" \
  --output "$output" \
  --max-batches 1000 --batch-size 1 --sequence-length 256 \
  --geometry-pairs 128 --minimum-plddt 70
