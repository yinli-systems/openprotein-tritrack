#!/usr/bin/env bash

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
verifier_pid="${VERIFY_PID:?VERIFY_PID must identify the official-gzip verifier}"
receipt="$root/data/raw/mgnify/2024_04-official-prefix/retrieval.json"
fasta="$root/data/raw/mgnify/2024_04-official-prefix/mgy_clusters.fa.gz"
output="$root/data/packed/mgnify90-2024_04-official-50pct-esm2-33"
definition="$source_dir/data_sources/mgnify-proteins-2024_04-litefold.json"

[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
[[ "$verifier_pid" =~ ^[1-9][0-9]*$ ]]
[[ ! -e "$output" ]]

while kill -0 "$verifier_pid" 2>/dev/null; do
  sleep 30
done

"$root/runtime/venv/bin/python" - "$receipt" "$fasta" "$definition" <<'PY'
import json
import sys
from pathlib import Path

receipt_path, fasta_path, definition_path = map(Path, sys.argv[1:])
if not receipt_path.is_file():
    raise RuntimeError("official MGnify verifier exited without a receipt")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
definition = json.loads(definition_path.read_text(encoding="utf-8"))
if receipt.get("status") != "VERIFIED_RAW_INPUT":
    raise RuntimeError("official MGnify receipt is not VERIFIED_RAW_INPUT")
if Path(receipt["path"]).resolve() != fasta_path.resolve():
    raise RuntimeError("official MGnify receipt points to another file")
if int(receipt["bytes"]) != int(definition["fasta_bytes"]):
    raise RuntimeError("official MGnify receipt byte count differs")
if receipt["md5"] != definition["fasta_md5"]:
    raise RuntimeError("official MGnify receipt MD5 differs")
if receipt.get("gzip_test") != "passed":
    raise RuntimeError("official MGnify receipt lacks a passed gzip test")
if not isinstance(receipt.get("sha256"), str) or len(receipt["sha256"]) != 64:
    raise RuntimeError("official MGnify receipt lacks a SHA-256")
PY

[[ ! -e "$output" ]]
if squeue -u scxi253 -h -n mgnify24-pack-official | grep -q .; then
  printf 'an official MGnify pack job is already active\n' >&2
  exit 1
fi

pack_job="$(
  sbatch --parsable --export=ALL,SOURCE_DIR="$source_dir" \
    "$source_dir/scripts/build_packed_mgnify2024_official.sbatch"
)"
printf 'PACK_JOB_ID=%s SUBMITTED_UTC=%s\n' "$pack_job" "$(date -u +%FT%TZ)"
