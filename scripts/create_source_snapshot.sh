#!/usr/bin/env bash

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="$root/source"
python="$root/runtime/venv/bin/python"
cd "$source_dir"
digest="$($python -c 'from pathlib import Path; from openprotein.provenance import source_tree_sha256; print(source_tree_sha256(Path.cwd()))')"
snapshot_root="$root/source-snapshots"
final="$snapshot_root/$digest"

mkdir -p "$snapshot_root"
if [[ -d "$final" ]]; then
  observed="$(cd "$final" && "$python" -c 'from pathlib import Path; from openprotein.provenance import source_tree_sha256; print(source_tree_sha256(Path.cwd()))')"
  [[ "$observed" == "$digest" ]]
  printf '%s\n' "$final"
  exit 0
fi

temporary="$snapshot_root/.${digest}.tmp-$$"
mkdir "$temporary"
rsync -a \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  "$source_dir/" "$temporary/"
observed="$(cd "$temporary" && "$python" -c 'from pathlib import Path; from openprotein.provenance import source_tree_sha256; print(source_tree_sha256(Path.cwd()))')"
[[ "$observed" == "$digest" ]]
mv "$temporary" "$final"
chmod -R a-w "$final"
printf '%s\n' "$final"
