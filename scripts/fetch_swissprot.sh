#!/usr/bin/env bash
set -euo pipefail

root=/ssd/scxi253/pretraining2
release=2026_02
destination="$root/data/raw/uniprot/$release"
url=https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz
expected_bytes=699031150
expected_md5=868b301a6ec93955f4e4355d579d8683
partial="$destination/uniprot_sprot.dat.gz.partial"
final="$destination/uniprot_sprot.dat.gz"

mkdir -p "$destination"
if [[ ! -f "$final" ]]; then
  curl --fail --location --retry 8 --retry-all-errors --continue-at - \
    --output "$partial" "$url"
  observed_bytes=$(stat -c '%s' "$partial")
  observed_md5=$(md5sum "$partial" | awk '{print $1}')
  [[ "$observed_bytes" = "$expected_bytes" ]]
  [[ "$observed_md5" = "$expected_md5" ]]
  mv "$partial" "$final"
fi
printf 'bytes=%s md5=%s\n' "$(stat -c '%s' "$final")" "$(md5sum "$final" | awk '{print $1}')"
