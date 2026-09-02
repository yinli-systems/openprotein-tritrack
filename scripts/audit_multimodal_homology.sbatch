#!/usr/bin/env bash
#SBATCH --job-name=mm1b-homology
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/mm1b-homology-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/mm1b-homology-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
output_dir="${OUTPUT_DIR:?OUTPUT_DIR is required}"
mmseqs="${MMSEQS_BIN:?MMSEQS_BIN is required}"
[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
output_dir="$(realpath -m "$output_dir")"
[[ "$output_dir" == "$root/evaluations/"* ]]
test ! -e "$output_dir"
test -x "$mmseqs"

staging="${output_dir}.staging-${SLURM_JOB_ID}"
[[ "$staging" == "$root/evaluations/"* ]]
test ! -e "$staging"
mkdir -p "$staging"
cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
python="$root/runtime/venv/bin/python"

"$python" -m openprotein.cli.audit_homology prepare \
  --train-manifest "$root/data/processed/uniref50-2026_02/manifest.json" \
  --validation-manifest "$root/data/processed/uniref50-swissprot-2026_02/manifest.json" \
  --output-dir "$staging"

format="query,target,pident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,qcov,tcov"
command=(
  "$mmseqs" easy-search
  "$staging/queries.fasta"
  "$staging/train.fasta"
  "$staging/hits.tsv"
  "$staging/mmseqs-tmp"
  --min-seq-id 0.20
  -c 0.50
  --cov-mode 0
  -e 1e-3
  -s 7.5
  --max-seqs 1000
  --threads "${SLURM_CPUS_PER_TASK}"
  --gpu 1
  --createdb-mode 2
  --format-output "$format"
)
"${command[@]}"
mmseqs_version="$($mmseqs version)"

"$python" -m openprotein.cli.audit_homology summarize \
  --prepare "$staging/prepare.json" \
  --hits "$staging/hits.tsv" \
  --output "$staging/homology.json" \
  --clean-ids "$staging/clean-query-ids.txt" \
  --excluded-ids "$staging/excluded-query-ids.txt" \
  --top-hits "$staging/top-hits.tsv" \
  --mmseqs-version "$mmseqs_version" \
  --search-slurm-job-id "$SLURM_JOB_ID" \
  --mmseqs-command "${command[@]}"

tmp_dir="$staging/mmseqs-tmp"
[[ "$tmp_dir" == "$staging/"* ]]
rm -rf -- "$tmp_dir"
(cd "$staging" && sha256sum \
  prepare.json hits.tsv homology.json clean-query-ids.txt excluded-query-ids.txt top-hits.tsv \
  > SHA256SUMS)
mv "$staging" "$output_dir"
chmod -R a-w "$output_dir"
printf 'evaluation_dir=%s\n' "$output_dir"
