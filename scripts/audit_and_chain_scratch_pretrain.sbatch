#!/usr/bin/env bash
#SBATCH --job-name=scratch7b-audit-chain
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/scratch-audit-chain-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/scratch-audit-chain-%j.err

set -euo pipefail
ops_dir="${OPS_DIR:?OPS_DIR must name the immutable operations snapshot}"
ops_dir="$(readlink -f -- "$ops_dir")"
if [[ ! "$ops_dir" =~ ^/ssd/scxi253/pretraining2/source-snapshots/[0-9a-f]{64}$ ]]; then
  printf 'invalid OPS_DIR: %s\n' "$ops_dir" >&2
  exit 2
fi
if [[ ! -d "$ops_dir" || -w "$ops_dir" ]]; then
  printf 'OPS_DIR must exist and be read-only: %s\n' "$ops_dir" >&2
  exit 2
fi
trainer_source="${TRAINER_SOURCE:?TRAINER_SOURCE must name the immutable trainer snapshot}"
config="${CONFIG:?CONFIG must name the frozen scratch pretraining config}"
run_dir="${RUN_DIR:?RUN_DIR must name the formal scratch run}"
trainer_sbatch="${TRAINER_SBATCH:?TRAINER_SBATCH must name the frozen trainer sbatch}"
expected_world_size="${EXPECTED_WORLD_SIZE:?EXPECTED_WORLD_SIZE is required}"
chain_id="${CHAIN_ID:?CHAIN_ID is required}"
receipts=/data/run01/scxi253/pretraining2-archive/receipts
[[ "$trainer_source" == /ssd/scxi253/pretraining2/source-snapshots/[0-9a-f]* ]]
[[ ! -w "$trainer_source" ]]
[[ "$config" == "$trainer_source/configs/scratch/train/"* ]]
[[ "$trainer_sbatch" == "$trainer_source/scripts/"* ]]
[[ "$expected_world_size" =~ ^[1-9][0-9]*$ ]]
[[ "$chain_id" =~ ^[a-z0-9][a-z0-9-]*$ ]]

/ssd/scxi253/pretraining2/runtime/venv/bin/python \
  "$ops_dir/scripts/audit_and_chain_scratch_pretrain.py" \
  --run-dir "$run_dir" \
  --source-dir "$trainer_source" \
  --config "$config" \
  --completed-job-id "${COMPLETED_JOB_ID:?}" \
  --segment-start "${SEGMENT_START:?}" \
  --segment-end "${SEGMENT_END:?}" \
  --segment-size 5000 \
  --target-tokens 50000000000 \
  --lr-schedule-end-step 260000 \
  --expected-world-size "$expected_world_size" \
  --source-tree-sha256 "${SOURCE_TREE_SHA256:?}" \
  --training-code-sha256 "${TRAINING_CODE_SHA256:?}" \
  --train-config-sha256 "${TRAIN_CONFIG_SHA256:?}" \
  --data-manifest-sha256 "${DATA_MANIFEST_SHA256:?}" \
  --model-config-sha256 "${MODEL_CONFIG_SHA256:?}" \
  --tokenizer-sha256 "${TOKENIZER_SHA256:?}" \
  --trainer-sbatch "$trainer_sbatch" \
  --trainer-sbatch-sha256 "${TRAINER_SBATCH_SHA256:?}" \
  --audit-sbatch "$ops_dir/scripts/audit_and_chain_scratch_pretrain.sbatch" \
  --audit-sbatch-sha256 "${AUDIT_SBATCH_SHA256:?}" \
  --audit-script-sha256 "${AUDIT_SCRIPT_SHA256:?}" \
  --archive-receipts-dir "$receipts" \
  --current-audit-receipt-sha256 pending \
  --chain-policy-sha256 "${CHAIN_POLICY_SHA256:?}" \
  --chain-id "$chain_id"
