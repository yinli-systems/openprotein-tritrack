#!/usr/bin/env bash
#SBATCH --job-name=openprotein-net-probe
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=/data/run01/scxi253/pretraining2-archive/logs/network-probe-%j.out
#SBATCH --error=/data/run01/scxi253/pretraining2-archive/logs/network-probe-%j.err

set -euo pipefail
root=/ssd/scxi253/pretraining2
source_dir="${SOURCE_DIR:?SOURCE_DIR must name an immutable source snapshot}"
[[ "$source_dir" == "$root/source-snapshots/"* ]]
[[ ! -w "$source_dir" ]]
[[ "$SLURM_NNODES" -eq 2 ]]

mkdir -p /data/run01/scxi253/pretraining2-archive/logs \
  /data/run01/scxi253/pretraining2-archive/receipts
master_addr="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
master_port="$((30000 + SLURM_JOB_ID % 10000))"
hosts=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
[[ "${#hosts[@]}" -eq 2 ]]
host0="${hosts[0]}"
host1="${hosts[1]}"
receipt="/data/run01/scxi253/pretraining2-archive/receipts/network-probe-${SLURM_JOB_ID}.json"
export root source_dir master_addr master_port host0 host1 receipt
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

srun --kill-on-bad-exit=1 --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus-per-task=1 \
  bash -c '
    set -euo pipefail
    peer_host="$host0"
    if [[ "$SLURM_NODEID" -eq 0 ]]; then
      peer_host="$host1"
    fi
    source "$source_dir/scripts/select_distributed_interface.sh" "$peer_host"
    exec "$root/runtime/venv/bin/torchrun" \
      --nnodes=2 \
      --nproc-per-node=1 \
      --node-rank="$SLURM_NODEID" \
      --master-addr="$master_addr" \
      --master-port="$master_port" \
      -m openprotein.cli.probe_collective --receipt "$receipt"
  '
