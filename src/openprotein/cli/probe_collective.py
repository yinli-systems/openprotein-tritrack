"""Fail-closed NCCL collective probe for a Slurm multi-node allocation."""

from __future__ import annotations

import argparse
import json
import os
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(backend="nccl")
    try:
        value = torch.tensor(float(rank + 1), device=device)
        torch.distributed.all_reduce(value)
        observed_sum = float(value.item())
        expected_sum = float(world_size * (world_size + 1) // 2)
        if observed_sum != expected_sum:
            raise RuntimeError(
                f"collective sum mismatch: observed={observed_sum} expected={expected_sum}"
            )
        local = {
            "rank": rank,
            "host": socket.gethostname(),
            "interface": os.environ.get("OPENPROTEIN_DISTRIBUTED_INTERFACE"),
            "source_ip": os.environ.get("OPENPROTEIN_DISTRIBUTED_SOURCE_IP"),
            "peer_host": os.environ.get("OPENPROTEIN_DISTRIBUTED_PEER_HOST"),
            "peer_ip": os.environ.get("OPENPROTEIN_DISTRIBUTED_PEER_IP"),
            "gpu": torch.cuda.get_device_name(device),
        }
        gathered: list[dict[str, object] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, local)
        if rank == 0:
            payload: dict[str, object] = {
                "schema_version": 1,
                "status": "passed",
                "backend": "nccl",
                "world_size": world_size,
                "expected_sum": expected_sum,
                "observed_sum": observed_sum,
                "ranks": gathered,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_json(Path(args.receipt), payload)
            print(json.dumps(payload, sort_keys=True), flush=True)
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
