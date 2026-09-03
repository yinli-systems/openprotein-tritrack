#!/usr/bin/env bash
#SBATCH --job-name=tritrack-data-v1
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/tritrack-data-v1-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/tritrack-data-v1-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir=${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}
mmseqs=${MMSEQS_BIN:?MMSEQS_BIN is required}
[[ "$source_dir" =~ ^$root/source-snapshots/[0-9a-f]{64}$ ]]
[[ ! -w "$source_dir" ]]
[[ "$mmseqs" == "$root/runtime/mmseqs2-18-8cc5c/bin/mmseqs" ]]
[[ "$($mmseqs version)" == 8cc5ce367b5638c4306c2d7cfc652dd099a4643f ]]

annotations="$root/data/processed/swissprot-2026_02"
structures="$root/data/processed/alphafold-swissprot-v6"
preparation="$root/data/processed/tritrack-clustering-v1"
dataset="$root/data/processed/tritrack-clustered-swissprot-afdb-v1"
tokenizer="$root/data/processed/tritrack-text-tokenizer-v1"
test ! -e "$dataset"
test ! -e "$tokenizer"

export PYTHONPATH="$source_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python="$root/runtime/venv/bin/python"
fasta="$preparation/swissprot.fasta"
cluster_prefix="$preparation/mmseqs30"
cluster_tsv="${cluster_prefix}_cluster.tsv"
mmseqs_tmp="$preparation/mmseqs-tmp"
resume_preparation=${RESUME_PREPARATION:-0}
[[ "$resume_preparation" == 0 || "$resume_preparation" == 1 ]]

mkdir -p "$root/logs"
if [[ "$resume_preparation" == 1 ]]; then
  test -d "$preparation"
  test -s "$fasta"
  test -s "$preparation/fasta-receipt.json"
  "$python" - "$preparation/fasta-receipt.json" "$fasta" <<'PY'
import json
import sys
from pathlib import Path

from openprotein.provenance import sha256_file

receipt_path = Path(sys.argv[1]).resolve()
fasta_path = Path(sys.argv[2]).resolve()
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
fasta = receipt.get("fasta", {})
source = receipt.get("source", {})
if receipt.get("status") != "passed":
    raise SystemExit("FASTA receipt did not pass")
if Path(fasta.get("path", "")).resolve() != fasta_path:
    raise SystemExit("FASTA receipt path mismatch")
if int(fasta.get("records", -1)) != 575_503:
    raise SystemExit("FASTA receipt record count mismatch")
if int(fasta.get("bytes", -1)) != fasta_path.stat().st_size:
    raise SystemExit("FASTA byte count mismatch")
if fasta.get("sha256") != sha256_file(fasta_path):
    raise SystemExit("FASTA digest mismatch")
for path_key, digest_key in (
    ("database_path", "database_sha256"),
    ("manifest_path", "manifest_sha256"),
):
    path = Path(source.get(path_key, "")).resolve()
    if not path.is_file() or source.get(digest_key) != sha256_file(path):
        raise SystemExit(f"source digest mismatch: {path_key}")
print("verified resumable FASTA and source receipts")
PY
else
  test ! -e "$preparation"
  mkdir "$preparation"
fi

cd "$source_dir"
if [[ "$resume_preparation" == 0 ]]; then
  "$python" -m openprotein.cli.build_clustered_multimodal export-fasta \
    --swissprot-database "$annotations/annotations.sqlite" \
    --swissprot-manifest "$annotations/manifest.json" \
    --output-fasta "$fasta" \
    --output-receipt "$preparation/fasta-receipt.json"
fi

# Keep this command stable so MMseqs can reuse completed content-addressed stages.
"$mmseqs" easy-cluster \
  "$fasta" \
  "$cluster_prefix" \
  "$mmseqs_tmp" \
  --min-seq-id 0.30 \
  -c 0.80 \
  --cov-mode 0 \
  --cluster-mode 1 \
  --max-iterations 1000 \
  --single-step-clustering 1 \
  --cluster-reassign 1 \
  --max-seqs 100000 \
  -s 7.5 \
  --threads "$SLURM_CPUS_PER_TASK" \
  --gpu 1 \
  --createdb-mode 2
test -s "$cluster_tsv"

"$python" -m openprotein.cli.build_clustered_multimodal build \
  --swissprot-database "$annotations/annotations.sqlite" \
  --swissprot-manifest "$annotations/manifest.json" \
  --structure-database "$structures/structures.sqlite" \
  --structure-manifest "$structures/manifest.json" \
  --cluster-tsv "$cluster_tsv" \
  --output-root "$dataset" \
  --mmseqs-version 8cc5ce367b5638c4306c2d7cfc652dd099a4643f \
  --seed 260903 \
  --train-shards 96 \
  --mmseqs-command easy-cluster swissprot.fasta mmseqs30 mmseqs-tmp \
    --min-seq-id 0.30 -c 0.80 --cov-mode 0 --cluster-mode 1 \
    --max-iterations 1000 --single-step-clustering 1 --cluster-reassign 1 \
    --max-seqs 100000 -s 7.5 --gpu 1 --createdb-mode 2

"$python" -m openprotein.cli.train_text_tokenizer \
  "$dataset/train/*.csv" \
  "$dataset/manifest.json" \
  "$tokenizer" \
  --vocab-size 32768

[[ "$mmseqs_tmp" == "$preparation/mmseqs-tmp" ]]
rm -rf -- "$mmseqs_tmp"
chmod -R a-w "$preparation" "$dataset" "$tokenizer"
printf 'dataset=%s\ntokenizer=%s\ncluster_tsv=%s\n' "$dataset" "$tokenizer" "$cluster_tsv"
