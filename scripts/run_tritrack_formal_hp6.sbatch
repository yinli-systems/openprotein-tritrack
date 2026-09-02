#!/usr/bin/env bash
#SBATCH --job-name=tritrack-hp6-3b
#SBATCH --partition=hp_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=36
#SBATCH --time=7-00:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/tritrack-hp6-3b-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/tritrack-hp6-3b-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir=${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}
selection=${SELECTION_RECEIPT:?SELECTION_RECEIPT is required}
[[ "$source_dir" =~ ^$root/source-snapshots/[0-9a-f]{64}$ ]]
[[ ! -w "$source_dir" ]]
[[ "$selection" == "$root/receipts/tritrack-hp6-batchcap-"*.json ]]
test -f "$selection"

mkdir -p "$root/logs" "$root/tmp" "$root/cache/hf" \
  "$root/cache/xdg" "$root/cache/torchinductor"
task_tmp="$root/tmp/tritrack-hp6-formal-${SLURM_JOB_ID}"
mkdir "$task_tmp"
export PYTHONPATH="$source_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="$task_tmp"
export HF_HOME="$root/cache/hf"
export XDG_CACHE_HOME="$root/cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="$root/cache/torchinductor"
python="$root/runtime/venv/bin/python"

selected=$(
  "$python" -c '
import json, sys
from pathlib import Path
from openprotein.provenance import source_tree_sha256
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
source = Path(sys.argv[2]).resolve()
if receipt.get("status") != "passed":
    raise SystemExit("batch-cap selection did not pass")
if receipt.get("source_tree_sha256") != source_tree_sha256(source):
    raise SystemExit("selection/source snapshot mismatch")
selected = receipt.get("selected_formal_config")
if not isinstance(selected, str) or not selected.startswith("configs/train/tritrack-"):
    raise SystemExit("invalid selected formal config")
print(selected)
' "$selection" "$source_dir"
)
config="$source_dir/$selected"
test -f "$config"
args=(--config "$config")
if [[ -n "${RESUME:-}" ]]; then
  test -f "$RESUME/manifest.json"
  args+=(--resume "$RESUME")
fi
cd "$source_dir"
"$root/runtime/venv/bin/torchrun" --standalone --nproc_per_node=6 \
  -m openprotein.cli.train "${args[@]}"
