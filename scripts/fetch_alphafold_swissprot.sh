#!/usr/bin/env bash
set -euo pipefail

root=/ssd/scxi253/pretraining2
destination="$root/data/raw/alphafold/v6"
url=https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v6.tar
expected_bytes=28580050944
partial="$destination/swissprot_pdb_v6.tar.partial"
final="$destination/swissprot_pdb_v6.tar"

mkdir -p "$destination"
if [[ ! -f "$final" ]]; then
  curl --fail --location --retry 12 --retry-all-errors --continue-at - \
    --output "$partial" "$url"
  observed_bytes=$(stat -c '%s' "$partial")
  [[ "$observed_bytes" = "$expected_bytes" ]]
  tar -tf "$partial" >/dev/null
  observed_sha256=$(sha256sum "$partial" | awk '{print $1}')
  mv "$partial" "$final"
else
  observed_sha256=$(sha256sum "$final" | awk '{print $1}')
fi
printf 'bytes=%s sha256=%s tar_test=pass\n' \
  "$(stat -c '%s' "$final")" "$observed_sha256"
