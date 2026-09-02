#!/usr/bin/env bash
set -euo pipefail

root=/ssd/scxi253/pretraining2
destination="$root/data/raw/proteingym/v1.3"
url=https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/zero_shot_substitutions_scores.zip
expected_bytes=1911045703
chunk_bytes=$((16 * 1024 * 1024))
parallel_downloads="${PARALLEL_DOWNLOADS:-32}"
staging="$destination/.zero-shot-score-chunks-16m"
assembled="$destination/zero_shot_substitutions_scores.zip.segmented-partial"
final="$destination/zero_shot_substitutions_scores.zip"

if [[ -e "$final" ]]; then
  printf 'refusing to overwrite existing final artifact: %s\n' "$final" >&2
  exit 1
fi

mkdir -p "$destination" "$staging"
chunk_count=$(((expected_bytes + chunk_bytes - 1) / chunk_bytes))

download_chunk() {
  index="$1"
  start=$((index * chunk_bytes))
  end=$((start + chunk_bytes - 1))
  if ((end >= expected_bytes)); then
    end=$((expected_bytes - 1))
  fi
  expected_chunk_bytes=$((end - start + 1))
  chunk=$(printf '%s/chunk-%05d.part' "$staging" "$index")
  header=$(printf '%s/chunk-%05d.headers' "$staging" "$index")
  if [[ -f "$chunk" ]]; then
    observed=$(stat -c '%s' "$chunk")
    if [[ "$observed" = "$expected_chunk_bytes" ]]; then
      printf 'chunk=%s status=present bytes=%s\n' "$index" "$observed"
      return
    fi
    printf 'chunk=%s status=invalid-existing expected=%s observed=%s\n' \
      "$index" "$expected_chunk_bytes" "$observed" >&2
    return 1
  fi
  body_temporary="$chunk.partial"
  header_temporary="$header.partial"
  curl --silent --show-error --fail --location --retry 12 --retry-all-errors \
    --range "$start-$end" --dump-header "$header_temporary" \
    --output "$body_temporary" "$url"
  observed=$(stat -c '%s' "$body_temporary")
  [[ "$observed" = "$expected_chunk_bytes" ]]
  tr -d '\r' < "$header_temporary" | \
    grep -Fxi "content-range: bytes $start-$end/$expected_bytes" >/dev/null
  mv "$body_temporary" "$chunk"
  mv "$header_temporary" "$header"
  printf 'chunk=%s status=downloaded bytes=%s\n' "$index" "$observed"
}
export destination url expected_bytes chunk_bytes staging
export -f download_chunk

seq 0 $((chunk_count - 1)) | \
  xargs -n 1 -P "$parallel_downloads" bash -c 'download_chunk "$1"' _

for index in $(seq 0 $((chunk_count - 1))); do
  chunk=$(printf '%s/chunk-%05d.part' "$staging" "$index")
  start=$((index * chunk_bytes))
  end=$((start + chunk_bytes - 1))
  if ((end >= expected_bytes)); then
    end=$((expected_bytes - 1))
  fi
  [[ "$(stat -c '%s' "$chunk")" = "$((end - start + 1))" ]]
done

truncate -s "$expected_bytes" "$assembled"
for index in $(seq 0 $((chunk_count - 1))); do
  chunk=$(printf '%s/chunk-%05d.part' "$staging" "$index")
  dd if="$chunk" of="$assembled" bs=16M seek="$index" conv=notrunc status=none
done
[[ "$(stat -c '%s' "$assembled")" = "$expected_bytes" ]]
observed_sha256=$(sha256sum "$assembled" | awk '{print $1}')
mv "$assembled" "$final"
printf 'bytes=%s sha256=%s rows_accessed=false archive_listing_accessed=false chunks=%s\n' \
  "$expected_bytes" "$observed_sha256" "$chunk_count"
