#!/usr/bin/env bash
#SBATCH --job-name=protein7b-val
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/sequence-val-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/sequence-val-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:-$root/source}"
checkpoint="${CHECKPOINT:-$root/runs/uniref50-2026-02-official-pilot/checkpoints/step-00000100}"
output="${OUTPUT:-$root/runs/uniref50-2026-02-official-pilot/validation-1000.json}"
max_batches="${MAX_BATCHES:-1000}"
case "$source_dir" in
  "$root/source"|"$root"/source-candidates/*|"$root"/source-snapshots/*) ;;
  *)
    printf 'invalid SOURCE_DIR: %s\n' "$source_dir" >&2
    exit 1
    ;;
esac
cd "$source_dir"
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
"$root/runtime/venv/bin/python" -m openprotein.cli.validate_sequence \
  --model-config "$source_dir/configs/model/openprotein-7b.json" \
  --checkpoint "$checkpoint" \
  --tokenizer-json "/ssd/scxi253/esmfold2/models/Biohub_ESMC-6B/tokenizer.json" \
  --model-artifact-manifest "$root/manifests/esmc-6b-local-snapshot.json" \
  --validation-glob "$root/data/processed/uniref50-2026_02/validation/*.csv" \
  --data-manifest "$root/data/processed/uniref50-2026_02/manifest.json" \
  --output "$output" \
  --max-batches "$max_batches" \
  --sequence-length 256
