#!/usr/bin/env bash
#SBATCH --job-name=protein7b-infer
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/release-inference-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/release-inference-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
release_dir="${RELEASE_DIR:?RELEASE_DIR is required}"
output_root="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
test -f "$release_dir/manifest.json"
test ! -e "$output_root"
mkdir -p "$output_root"
cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
protein_tokenizer=/ssd/scxi253/esmfold2/models/Biohub_ESMC-6B/tokenizer.json
artifact_manifest="$root/manifests/esmc-6b-local-snapshot.json"
backbone_root=/ssd/scxi253/esmfold2/models/Biohub_ESMC-6B
sequence=ACDEFGHIKLMNPQRSTVWY

"$root/runtime/venv/bin/python" -m openprotein.cli.score_variants \
  "$release_dir" "$protein_tokenizer" "$artifact_manifest" "$sequence" \
  "$output_root/variant-scores.json" A1G C2A \
  --backbone-root "$backbone_root" --device cuda --dtype bfloat16

"$root/runtime/venv/bin/python" -m openprotein.cli.generate_annotation \
  "$release_dir" "$protein_tokenizer" "$artifact_manifest" "$sequence" \
  "$output_root/generated-annotation.json" \
  --max-new-tokens 16 --backbone-root "$backbone_root" --device cuda --dtype bfloat16

"$root/runtime/venv/bin/python" -m openprotein.cli.predict_geometry \
  "$release_dir" "$protein_tokenizer" "$artifact_manifest" "$sequence" \
  "$output_root/predicted-geometry.json" 1:10 2:15 \
  --backbone-root "$backbone_root" --device cuda --dtype bfloat16
