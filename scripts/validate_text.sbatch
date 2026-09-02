#!/usr/bin/env bash
#SBATCH --job-name=protein7b-textval
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/text-val-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/text-val-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:-$root/source}"
checkpoint="${CHECKPOINT:?CHECKPOINT is required}"
output="${OUTPUT:?OUTPUT is required}"
case "$source_dir" in
  "$root/source"|"$root"/source-candidates/*|"$root"/source-snapshots/*) ;;
  *)
    printf 'invalid SOURCE_DIR: %s\n' "$source_dir" >&2
    exit 1
    ;;
esac
cd "$source_dir"
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
"$root/runtime/venv/bin/python" -m openprotein.cli.validate_text \
  --model-config "$source_dir/configs/model/openprotein-7b.json" \
  --checkpoint "$checkpoint" \
  --protein-tokenizer-json "/ssd/scxi253/esmfold2/models/Biohub_ESMC-6B/tokenizer.json" \
  --text-tokenizer-model "$root/data/processed/text-tokenizer-swissprot-2026_02/tokenizer.model" \
  --model-artifact-manifest "$root/manifests/esmc-6b-local-snapshot.json" \
  --validation-glob "$root/data/processed/uniref50-swissprot-2026_02/validation/*.csv" \
  --data-manifest "$root/data/processed/uniref50-swissprot-2026_02/manifest.json" \
  --output "$output" \
  --max-batches "${MAX_BATCHES:-125}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --sequence-length 128 \
  --text-length 128
