#!/usr/bin/env bash

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
definition="$source_dir/data_sources/mgnify-proteins-2024_04-litefold.json"
destination="$root/data/raw/mgnify/2024_04-litefold"

[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
mkdir -p "$destination" "$root/logs"
exec 9>"$destination/download.lock"
flock -n 9 || {
  printf 'another MGnify mirror download holds %s\n' "$destination/download.lock" >&2
  exit 1
}

cd "$source_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
"$root/runtime/venv/bin/python" -m openprotein.cli.fetch_mgnify_hf \
  "$definition" "$destination" --endpoint https://hf-mirror.com --workers 8
