"""Bounded synthetic training smoke with an auditable JSON receipt."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from openprotein.config import load_model_config
from openprotein.model import OpenProteinModel
from openprotein.provenance import sha256_file, source_tree_sha256


def _git_revision(root: Path) -> Optional[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")

    config_path = Path(args.config).resolve()
    config = load_model_config(str(config_path))
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = OpenProteinModel.from_config(config, device=device, dtype=dtype)
    model.to(device=device, dtype=dtype)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)

    batch = 2 if config.protein_backbone.kind == "tiny" else 1
    protein_length = min(config.max_sequence_length, 32)
    text_length = min(config.max_text_length, 12)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    started = time.perf_counter()
    losses = []
    for _ in range(args.steps):
        protein_ids = torch.randint(
            0,
            config.protein_backbone.vocab_size,
            (batch, protein_length),
            generator=generator,
            device=device,
        )
        text_ids = torch.randint(
            0,
            config.text_vocab_size,
            (batch, text_length),
            generator=generator,
            device=device,
        )
        protein_mask = torch.ones_like(protein_ids)
        text_mask = torch.ones_like(text_ids)
        protein_labels = protein_ids.clone()
        protein_labels[:, 1::2] = -100
        text_labels = text_ids.clone()
        pair_indices = torch.tensor([[[0, 3], [2, 7]]] * batch, device=device)
        distance_labels = torch.tensor([[1, 2]] * batch, device=device)

        optimizer.zero_grad(set_to_none=True)
        output = model(
            protein_ids,
            protein_mask,
            text_ids,
            text_mask,
            protein_labels=protein_labels,
            text_labels=text_labels,
            pair_indices=pair_indices,
            distance_labels=distance_labels,
            attention_policy="bidirectional",
        )
        loss = output["loss"]
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non-finite smoke loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": "GPU_SMOKE" if device.type == "cuda" else "LOCAL_SMOKE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_tree_sha256": source_tree_sha256(Path.cwd()),
        "model_name": config.name,
        "analytical_parameters": config.estimated_parameter_count,
        "loaded_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "steps": args.steps,
        "seed": args.seed,
        "losses": losses,
        "elapsed_seconds": elapsed,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "git_revision": _git_revision(Path.cwd()),
    }
    if config.protein_backbone.model_path:
        index_path = Path(config.protein_backbone.model_path) / "model.safetensors.index.json"
        receipt["backbone_index_sha256"] = (
            sha256_file(index_path) if index_path.exists() else None
        )
    _atomic_json(Path(args.receipt), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
