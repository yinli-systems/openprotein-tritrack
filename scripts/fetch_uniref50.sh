#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIRECTORY" >&2
  exit 2
fi

output_directory=$1
url=https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniref/uniref50/uniref50.fasta.gz
expected_bytes=8770260598
expected_md5=3228886e9d749f050f60e9a0ce1f727d
partial_path="$output_directory/uniref50.fasta.gz.partial"
final_path="$output_directory/uniref50.fasta.gz"

mkdir -p "$output_directory"
if [[ ! -f "$final_path" ]]; then
  curl \
    --fail \
    --location \
    --retry 8 \
    --retry-all-errors \
    --connect-timeout 15 \
    --continue-at - \
    --output "$partial_path" \
    "$url"
  actual_bytes=$(stat -c %s "$partial_path")
  actual_md5=$(md5sum "$partial_path" | awk '{print $1}')
  [[ "$actual_bytes" == "$expected_bytes" ]]
  [[ "$actual_md5" == "$expected_md5" ]]
  mv "$partial_path" "$final_path"
fi

actual_bytes=$(stat -c %s "$final_path")
actual_md5=$(md5sum "$final_path" | awk '{print $1}')
[[ "$actual_bytes" == "$expected_bytes" ]]
[[ "$actual_md5" == "$expected_md5" ]]
printf 'verified bytes=%s md5=%s path=%s\n' "$actual_bytes" "$actual_md5" "$final_path"
