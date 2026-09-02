#!/usr/bin/env bash
#SBATCH --job-name=mmseqs-gpu-smoke
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:10:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/mmseqs-gpu-smoke-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/mmseqs-gpu-smoke-%j.err

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

mkdir "$output_dir"
query="$source_dir/tests/fixtures/homology/query.fasta"
target="$source_dir/tests/fixtures/homology/target.fasta"
tmp_dir="$output_dir/mmseqs-tmp"
format="query,target,pident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,qcov,tcov"
"$mmseqs" easy-search \
  "$query" \
  "$target" \
  "$output_dir/hits.tsv" \
  "$tmp_dir" \
  --min-seq-id 0.20 \
  -c 0.50 \
  --cov-mode 0 \
  -e 1e-3 \
  -s 7.5 \
  --max-seqs 1000 \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --gpu 1 \
  --createdb-mode 2 \
  --format-output "$format"
test -s "$output_dir/hits.tsv"
grep -q query-identical "$output_dir/hits.tsv"
[[ "$tmp_dir" == "$output_dir/"* ]]
rm -rf -- "$tmp_dir"
(cd "$output_dir" && sha256sum hits.tsv > SHA256SUMS)
chmod -R a-w "$output_dir"
printf 'smoke_dir=%s\n' "$output_dir"
