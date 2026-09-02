#!/usr/bin/env bash
#SBATCH --job-name=mm1b-homology-finalize
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/mm1b-homology-finalize-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/mm1b-homology-finalize-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
staging="${HOMOLOGY_STAGING:?HOMOLOGY_STAGING is required}"
output_dir="${OUTPUT_DIR:?OUTPUT_DIR is required}"
search_job_id="${SEARCH_SLURM_JOB_ID:?SEARCH_SLURM_JOB_ID is required}"
mmseqs="${MMSEQS_BIN:?MMSEQS_BIN is required}"
[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
[[ "$staging" == "$root/evaluations/"*.staging-"$search_job_id" ]]
[[ "$output_dir" == "$root/evaluations/"* ]]
test -d "$staging"
test ! -e "$output_dir"
test -f "$staging/prepare.json"
test -f "$staging/hits.tsv"
test ! -e "$staging/homology.json"
test -x "$mmseqs"

cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
python="$root/runtime/venv/bin/python"
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
  --threads 6
  --gpu 1
  --createdb-mode 2
  --format-output "$format"
)

"$python" -m openprotein.cli.audit_homology summarize \
  --prepare "$staging/prepare.json" \
  --hits "$staging/hits.tsv" \
  --output "$staging/homology.json" \
  --clean-ids "$staging/clean-query-ids.txt" \
  --excluded-ids "$staging/excluded-query-ids.txt" \
  --top-hits "$staging/top-hits.tsv" \
  --mmseqs-version "$($mmseqs version)" \
  --search-slurm-job-id "$search_job_id" \
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
