#!/usr/bin/env bash
#SBATCH --job-name=tritrack-final-val
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/tritrack-final-val-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/tritrack-final-val-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir=${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}
selection=${SELECTION_RECEIPT:?SELECTION_RECEIPT is required}
pretrain_job_id=${PRETRAIN_JOB_ID:?PRETRAIN_JOB_ID is required}
output_dir=${OUTPUT_DIR:?OUTPUT_DIR is required}
[[ "$source_dir" =~ ^$root/source-snapshots/[0-9a-f]{64}$ ]]
[[ ! -w "$source_dir" ]]
[[ "$output_dir" == "$root/evaluations/tritrack-"* ]]
test ! -e "$output_dir"

export PYTHONPATH="$source_dir/src"
export PYTHONDONTWRITEBYTECODE=1
python="$root/runtime/venv/bin/python"
readarray -t resolved < <(
  "$python" -c '
import glob, json, sys
from pathlib import Path
selection = json.load(open(sys.argv[1], encoding="utf-8"))
source = Path(sys.argv[2])
relative = selection["selected_formal_config"]
config_path = source / relative
config = json.load(open(config_path, encoding="utf-8"))
receipts = []
for path in glob.glob(str(Path(config["output_dir"]) / "receipt-step-*.json")):
    receipt = json.load(open(path, encoding="utf-8"))
    if receipt.get("status") == "passed" and receipt.get("evidence_level") == "PRETRAIN":
        receipts.append((path, receipt))
if len(receipts) != 1:
    raise SystemExit("expected exactly one passed PRETRAIN receipt")
receipt_path, receipt = receipts[0]
if receipt.get("slurm_job_id") != sys.argv[3] or int(receipt["tokens"]) < 3_000_000_000:
    raise SystemExit("PRETRAIN receipt job/token contract mismatch")
step = int(receipt["step"])
checkpoint = Path(config["output_dir"]) / "checkpoints" / f"step-{step:08d}"
if not (checkpoint / "manifest.json").is_file():
    raise SystemExit("final checkpoint is missing")
print(config_path)
print(receipt_path)
print(checkpoint)
' "$selection" "$source_dir" "$pretrain_job_id"
)
train_config=${resolved[0]}
pretrain_receipt=${resolved[1]}
checkpoint=${resolved[2]}
staging="${output_dir}.staging-${SLURM_JOB_ID}"
mkdir -p "$staging"
model_config="$source_dir/configs/model/openprotein-tritrack-1.2b.json"
protein_tokenizer="$source_dir/configs/scratch/tokenizer/esm2-33.json"
text_tokenizer="$root/data/processed/tritrack-text-tokenizer-v1/tokenizer.model"
artifact_manifest="$root/manifests/esm2-t33-650m-ur50d.json"
dataset="$root/data/processed/tritrack-clustered-swissprot-afdb-v1"
structures="$root/data/processed/alphafold-swissprot-v6"
cd "$source_dir"

"$python" -m openprotein.cli.validate_sequence \
  --model-config "$model_config" \
  --checkpoint "$checkpoint" \
  --tokenizer-json "$protein_tokenizer" \
  --model-artifact-manifest "$artifact_manifest" \
  --validation-glob "$root/data/processed/uniref50-2026_02/validation/*.csv" \
  --data-manifest "$root/data/processed/uniref50-2026_02/manifest.json" \
  --output "$staging/sequence.json" \
  --max-batches 1000 \
  --sequence-length 256

"$python" -m openprotein.cli.validate_text \
  --model-config "$model_config" \
  --checkpoint "$checkpoint" \
  --protein-tokenizer-json "$protein_tokenizer" \
  --text-tokenizer-model "$text_tokenizer" \
  --model-artifact-manifest "$artifact_manifest" \
  --validation-glob "$dataset/validation/*.csv" \
  --data-manifest "$dataset/manifest.json" \
  --output "$staging/text.json" \
  --max-batches 1000 \
  --batch-size 8 \
  --sequence-length 256 \
  --text-length 128

"$python" -m openprotein.cli.validate_structure \
  --model-config "$model_config" \
  --checkpoint "$checkpoint" \
  --protein-tokenizer-json "$protein_tokenizer" \
  --model-artifact-manifest "$artifact_manifest" \
  --validation-glob "$dataset/validation/*.csv" \
  --data-manifest "$dataset/manifest.json" \
  --structure-database "$structures/structures.sqlite" \
  --structure-manifest "$structures/manifest.json" \
  --output "$staging/structure.json" \
  --max-batches 3000 \
  --batch-size 2 \
  --sequence-length 256 \
  --geometry-pairs 128 \
  --minimum-plddt 70

cp "$selection" "$staging/batchcap-selection.json"
cp "$pretrain_receipt" "$staging/pretrain-receipt.json"
cp "$train_config" "$staging/train-config.json"
(cd "$staging" && sha256sum *.json > SHA256SUMS)
mv "$staging" "$output_dir"
chmod -R a-w "$output_dir"
printf 'evaluation_dir=%s\n' "$output_dir"
