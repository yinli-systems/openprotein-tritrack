#!/usr/bin/env bash
#SBATCH --job-name=protein7b-runtime
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/runtime-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/runtime-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
python_bin="$root/runtime/venv/bin/python"
cd "$root/source"
export HF_HOME="$root/cache/hf"
export XDG_CACHE_HOME="$root/cache/xdg"
export TORCH_HOME="$root/cache/torch"

"$python_bin" -m pytest -q
"$python_bin" -m openprotein.cli.smoke \
  --config "$root/source/configs/model/openprotein-7b.json" \
  --receipt "$root/logs/openprotein7b-gpu-smoke-$SLURM_JOB_ID.json" \
  --steps 1 \
  --device cuda

"$python_bin" - <<'PY'
import json
import os
from pathlib import Path

root = Path("/ssd/scxi253/pretraining2")
receipt = root / "logs" / f"openprotein7b-gpu-smoke-{os.environ['SLURM_JOB_ID']}.json"
payload = json.loads(receipt.read_text())
assert payload["status"] == "passed"
assert payload["loaded_parameters"] == 7_076_898_432
marker = {
    "schema_version": 1,
    "job_id": os.environ["SLURM_JOB_ID"],
    "receipt": str(receipt),
    "gpu": payload["gpu_name"],
    "torch": payload["torch"],
    "cuda_runtime": payload["cuda_runtime"],
}
(root / "runtime" / ".gpu_verified").write_text(json.dumps(marker, indent=2) + "\n")
(root / "runtime" / ".prepared").write_text(json.dumps(marker, indent=2) + "\n")
PY
