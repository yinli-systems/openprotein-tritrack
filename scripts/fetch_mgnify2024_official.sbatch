#!/usr/bin/env bash
#SBATCH --job-name=mgnify24-official
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-12:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/mgnify24-official-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/mgnify24-official-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
definition="$source_dir/data_sources/mgnify-proteins-2024_04-litefold.json"
destination="$root/data/raw/mgnify/2024_04-official-prefix"
final="$destination/mgy_clusters.fa.gz"
receipt="$destination/retrieval.json"
ascp="$root/bootstrap/aspera/sdk-1.1.6/bin/ascp"
key="$root/bootstrap/aspera/fasp-public-rsa.key"
remote=fasp-public@fasp.ebi.ac.uk:/pub/databases/metagenomics/peptide_database/2024_04/mgy_clusters.fa.gz
url=https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2024_04/mgy_clusters.fa.gz

[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
[[ -x "$ascp" ]]
[[ -s "$key" ]]
mkdir -p "$destination" "$root/logs"
exec 9>"$destination/download.lock"
flock -n 9 || {
  printf 'another official MGnify 2024_04 download holds %s\n' \
    "$destination/download.lock" >&2
  exit 1
}

expected_bytes="$($root/runtime/venv/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["fasta_bytes"])' "$definition")"
expected_md5="$($root/runtime/venv/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["fasta_md5"])' "$definition")"

transport="EMBL-EBI Aspera fasp-public"
if [[ ! -s "$receipt" ]]; then
  existing_bytes=0
  if [[ -f "$final" ]]; then
    existing_bytes="$(stat -c %s "$final")"
  fi
  if [[ "$existing_bytes" == "$expected_bytes" ]]; then
    candidate_md5="$(md5sum "$final" | awk '{print $1}')"
    if [[ "$candidate_md5" == "$expected_md5" ]]; then
      transport="completed EMBL-EBI transfer candidate, independently reverified"
      printf 'candidate already matches official byte count and MD5; skipping transport only\n'
    elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
      printf 'exact-size candidate MD5 mismatch on compute node; login-node checksum recovery required\n' >&2
      exit 1
    else
      transport="EMBL-EBI Aspera full-checksum retransmit after MD5 mismatch"
      printf 'exact-size candidate MD5 mismatch; requesting full-checksum retransmit\n' >&2
      "$ascp" -T -k 3 --overwrite=diff -l 1g -P 33001 -i "$key" \
        "$remote" "$destination"
    fi
  elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # ParaCloud compute nodes currently reject the FASP SSH connection while
    # HTTPS remains routable. The login-node path uses the faster Aspera route.
    transport="EMBL-EBI HTTPS resumable range transfer"
    curl --fail --location --retry 20 --retry-delay 10 --retry-all-errors \
      --connect-timeout 30 --continue-at - --output "$final" "$url"
  else
    "$ascp" -T -k 3 --overwrite=diff -l 1g -P 33001 -i "$key" \
      "$remote" "$destination"
  fi
fi

observed_bytes="$(stat -c %s "$final")"
[[ "$observed_bytes" == "$expected_bytes" ]]
observed_md5="$(md5sum "$final" | awk '{print $1}')"
[[ "$observed_md5" == "$expected_md5" ]]
gzip -t "$final"
observed_sha256="$(sha256sum "$final" | awk '{print $1}')"

"$root/runtime/venv/bin/python" - \
  "$definition" "$final" "$receipt" "$observed_sha256" "$transport" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

definition_path, fasta_path, receipt_path = map(Path, sys.argv[1:4])
definition = json.loads(definition_path.read_text(encoding="utf-8"))
payload = {
    "schema_version": 1,
    "status": "VERIFIED_RAW_INPUT",
    "source_definition": str(definition_path),
    "source_definition_release": definition["release"],
    "source_url": definition["fasta_url"],
    "transport": sys.argv[5],
    "path": str(fasta_path),
    "bytes": fasta_path.stat().st_size,
    "md5": definition["fasta_md5"],
    "sha256": sys.argv[4],
    "gzip_test": "passed",
    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
}
with tempfile.NamedTemporaryFile(
    mode="w",
    encoding="utf-8",
    dir=receipt_path.parent,
    prefix=f".{receipt_path.name}.",
    delete=False,
) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
os.replace(temporary, receipt_path)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
