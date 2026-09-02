#!/usr/bin/env bash

set -euo pipefail

url="${1:?URL is required}"
output="${2:?output path is required}"
expected_bytes="${3:?expected byte count is required}"
expected_md5="${4:?expected MD5 is required}"
expected_sha256="${5:?expected SHA-256 or - is required}"
parts_dir="${6:?parts directory is required}"
workers="${7:-8}"
max_parallel="${8:-$workers}"

[[ "$expected_bytes" =~ ^[1-9][0-9]*$ ]]
[[ "$workers" =~ ^[1-9][0-9]*$ ]]
[[ "$max_parallel" =~ ^[1-9][0-9]*$ ]]
((max_parallel <= workers))
test ! -e "$output"
staging="${output}.staging"
test ! -e "$staging"
mkdir -p "$parts_dir"

chunk=$(((expected_bytes + workers - 1) / workers))

download_part() {
  local index="$1"
  local start=$((index * chunk))
  local end=$(((index + 1) * chunk - 1))
  if ((end >= expected_bytes)); then
    end=$((expected_bytes - 1))
  fi
  local expected=$((end - start + 1))
  local part
  part="$(printf '%s/part-%04d' "$parts_dir" "$index")"
  touch "$part"
  local current
  current="$(wc -c < "$part" | tr -d ' ')"
  if ((current > expected)); then
    printf 'part %s is larger than its declared range: %s > %s\n' \
      "$index" "$current" "$expected" >&2
    return 1
  fi
  local stagnant=0
  while ((current < expected)); do
    local absolute_start=$((start + current))
    curl -fsSL \
      --max-time 120 \
      --range "${absolute_start}-${end}" \
      "$url" >> "$part" || true
    local updated
    updated="$(wc -c < "$part" | tr -d ' ')"
    if ((updated > expected)); then
      printf 'server violated requested range for part %s: %s > %s\n' \
        "$index" "$updated" "$expected" >&2
      return 1
    fi
    if ((updated == current)); then
      stagnant=$((stagnant + 1))
      if ((stagnant >= 20)); then
        printf 'part %s made no progress after 20 attempts\n' "$index" >&2
        return 1
      fi
      sleep 2
    else
      stagnant=0
    fi
    current="$updated"
  done
  printf 'part=%s bytes=%s range=%s-%s\n' "$index" "$current" "$start" "$end"
}

for ((batch_start = 0; batch_start < workers; batch_start += max_parallel)); do
  pids=()
  batch_end=$((batch_start + max_parallel))
  if ((batch_end > workers)); then
    batch_end="$workers"
  fi
  for ((index = batch_start; index < batch_end; index++)); do
    download_part "$index" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done

for ((index = 0; index < workers; index++)); do
  part="$(printf '%s/part-%04d' "$parts_dir" "$index")"
  cat "$part" >> "$staging"
done
observed_bytes="$(wc -c < "$staging" | tr -d ' ')"
test "$observed_bytes" = "$expected_bytes"

if command -v md5sum >/dev/null 2>&1; then
  observed_md5="$(md5sum "$staging" | awk '{print $1}')"
else
  observed_md5="$(md5 -q "$staging")"
fi
test "$observed_md5" = "$expected_md5"

if command -v sha256sum >/dev/null 2>&1; then
  observed_sha256="$(sha256sum "$staging" | awk '{print $1}')"
else
  observed_sha256="$(shasum -a 256 "$staging" | awk '{print $1}')"
fi
if [[ "$expected_sha256" != "-" ]]; then
  test "$observed_sha256" = "$expected_sha256"
fi

mv "$staging" "$output"
chmod a-w "$output"
printf 'output=%s bytes=%s md5=%s sha256=%s\n' \
  "$output" "$observed_bytes" "$observed_md5" "$observed_sha256"
