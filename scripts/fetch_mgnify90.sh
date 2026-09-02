#!/usr/bin/env bash

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
definition="$source_dir/data_sources/mgnify-proteins-2026_07.json"
destination="$root/data/raw/mgnify/2026_07"
final="$destination/mgy_clusters.fa.gz"
partial="$final.part"
receipt="$destination/retrieval.json"
url=https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/current_release/mgy_clusters.fa.gz

[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
mkdir -p "$destination" "$root/logs"
exec 9>"$destination/download.lock"
flock -n 9 || { printf 'another MGnify download holds %s\n' "$destination/download.lock" >&2; exit 1; }

expected_bytes="$($root/runtime/venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["fasta_bytes"])' "$definition")"
expected_md5="$($root/runtime/venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["fasta_md5"])' "$definition")"

if [[ ! -f "$final" ]]; then
  curl --fail --location --retry 20 --retry-delay 10 --continue-at - \
    --output "$partial" "$url"
  observed_bytes="$(stat -c %s "$partial")"
  [[ "$observed_bytes" == "$expected_bytes" ]]
  observed_md5="$(md5sum "$partial" | awk '{print $1}')"
  [[ "$observed_md5" == "$expected_md5" ]]
  mv "$partial" "$final"
fi

observed_bytes="$(stat -c %s "$final")"
[[ "$observed_bytes" == "$expected_bytes" ]]
observed_md5="$(md5sum "$final" | awk '{print $1}')"
[[ "$observed_md5" == "$expected_md5" ]]
observed_sha256="$(sha256sum "$final" | awk '{print $1}')"
gzip -t "$final"

"$root/runtime/venv/bin/python" - "$definition" "$final" "$receipt" "$observed_sha256" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

definition_path, fasta_path, receipt_path, sha256 = map(Path, sys.argv[1:])
definition = json.loads(definition_path.read_text(encoding="utf-8"))
payload = {
    "schema_version": 1,
    "status": "VERIFIED_RAW_INPUT",
    "source_definition": str(definition_path),
    "source_definition_release": definition["release"],
    "path": str(fasta_path),
    "bytes": fasta_path.stat().st_size,
    "md5": definition["fasta_md5"],
    "sha256": str(sha256),
    "gzip_test": "passed",
    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
}
target = Path(receipt_path)
with tempfile.NamedTemporaryFile(
    mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False
) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
os.replace(temporary, target)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
