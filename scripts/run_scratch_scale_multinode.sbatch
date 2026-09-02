#!/usr/bin/env bash
#SBATCH --job-name=scratch7b-2x4-gate
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=1-12:00:00
#SBATCH --output=/ssd/scxi253/pretraining2/logs/scratch-scale-multinode-%j.out
#SBATCH --error=/ssd/scxi253/pretraining2/logs/scratch-scale-multinode-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
config="${CONFIG:?CONFIG must name a scratch training config}"
gpus_per_node="${GPUS_PER_NODE:?GPUS_PER_NODE must match the Slurm request}"
expected_world_size="${EXPECTED_WORLD_SIZE:?EXPECTED_WORLD_SIZE is required}"
[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
[[ "$config" == "$source_dir/configs/scratch/train/"* ]]
[[ "$gpus_per_node" =~ ^[1-9][0-9]*$ ]]
[[ "$expected_world_size" =~ ^[1-9][0-9]*$ ]]
[[ "$SLURM_NNODES" -ge 2 ]]
[[ $((SLURM_NNODES * gpus_per_node)) -eq "$expected_world_size" ]]

master_addr="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
master_port="$((20000 + SLURM_JOB_ID % 20000))"
hosts=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
[[ "${#hosts[@]}" -eq "$SLURM_NNODES" ]]
host0="${hosts[0]}"
host1="${hosts[1]}"
export root source_dir config gpus_per_node master_addr master_port host0 host1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

srun --kill-on-bad-exit=1 \
  --nodes="$SLURM_NNODES" \
  --ntasks="$SLURM_NNODES" \
  --ntasks-per-node=1 \
  --gpus-per-task="$gpus_per_node" \
  bash -c '
    set -euo pipefail
    cd "$source_dir"
    peer_host="$host0"
    if [[ "$SLURM_NODEID" -eq 0 ]]; then
      peer_host="$host1"
    fi
    source "$source_dir/scripts/select_distributed_interface.sh" "$peer_host"
    args=(--config "$config")
    if [[ -n "${RESUME:-}" ]]; then
      args+=(--resume "$RESUME")
    fi
    if [[ -n "${MAX_STEPS:-}" ]]; then
      args+=(--max-steps "$MAX_STEPS")
    fi
    exec "$root/runtime/venv/bin/torchrun" \
      --nnodes="$SLURM_NNODES" \
      --nproc-per-node="$gpus_per_node" \
      --node-rank="$SLURM_NODEID" \
      --master-addr="$master_addr" \
      --master-port="$master_port" \
      -m openprotein.cli.train_scratch "${args[@]}"
  '
