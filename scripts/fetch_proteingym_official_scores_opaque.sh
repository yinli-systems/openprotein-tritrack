#!/usr/bin/env bash
set -euo pipefail

root=/ssd/scxi253/pretraining2
destination="$root/data/raw/proteingym/v1.3"
url=https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/zero_shot_substitutions_scores.zip
expected_bytes=1911045703
partial="$destination/zero_shot_substitutions_scores.zip.partial"
final="$destination/zero_shot_substitutions_scores.zip"

mkdir -p "$destination"
if [[ ! -f "$final" ]]; then
  curl --fail --location --retry 12 --retry-all-errors --continue-at - \
    --output "$partial" "$url"
  observed_bytes=$(stat -c '%s' "$partial")
  [[ "$observed_bytes" = "$expected_bytes" ]]
  observed_sha256=$(sha256sum "$partial" | awk '{print $1}')
  mv "$partial" "$final"
else
  observed_bytes=$(stat -c '%s' "$final")
  [[ "$observed_bytes" = "$expected_bytes" ]]
  observed_sha256=$(sha256sum "$final" | awk '{print $1}')
fi
printf 'bytes=%s sha256=%s rows_accessed=false archive_listing_accessed=false\n' \
  "$observed_bytes" "$observed_sha256"
