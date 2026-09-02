"""FSDP2 training and distributed checkpoints for scratch protein encoders."""

from __future__ import annotations

import copy
import glob
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from openprotein.data.packed import PackedProteinBatcher, verify_packed_manifest
from openprotein.data.tokenizer import EsmcTokenizer
from openprotein.provenance import sha256_file, source_tree_sha256
from openprotein.scratch.config import load_scratch_model_config
from openprotein.scratch.model import ScratchProteinModel
from openprotein.training import RestartableBatcher


SCRATCH_TRAIN_SCHEMA_VERSION = 1
MINIMUM_SCRATCH_7B_PRETRAIN_TOKENS = 50_000_000_000
CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ScratchTrainConfig:
    schema_version: int
    name: str
    run_kind: str
    model_config: str
    tokenizer_json: str
    train_glob: str
    data_format: str
    output_dir: str
    sequence_length: int
    micro_batch_token_budget: int
    max_micro_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    target_tokens: Optional[int]
    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    adam_epsilon: float
    max_grad_norm: float
    mlm_probability: float
    seed: int
    log_every: int
    checkpoint_every: int
    checkpoint_at_start: bool
    precision: str
    activation_checkpointing: bool
    fsdp_shard_size: int
    data_manifest: Optional[str] = None
    checkpoint_keep_last: Optional[int] = None
    checkpoint_at_end: bool = True
    batch_lookahead: int = 1
    fsdp_reshard_after_forward: bool = True
    topology_migration: Optional[Dict[str, Any]] = None

    @classmethod
    def from_json(cls, path: str) -> "ScratchTrainConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("scratch train config must be a JSON object")
        expected = {field.name for field in fields(cls)}
        received = set(raw)
        missing = expected - received - {
            "data_manifest",
            "checkpoint_keep_last",
            "checkpoint_at_end",
            "batch_lookahead",
            "fsdp_reshard_after_forward",
            "topology_migration",
        }
        unknown = received - expected
        if missing:
            raise ValueError(f"scratch train config is missing fields: {sorted(missing)}")
        if unknown:
            raise ValueError(f"scratch train config has unknown fields: {sorted(unknown)}")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != SCRATCH_TRAIN_SCHEMA_VERSION:
            raise ValueError(f"scratch train schema must be {SCRATCH_TRAIN_SCHEMA_VERSION}")
        if self.run_kind not in {"scale_test", "pretrain"}:
            raise ValueError("run_kind must be scale_test or pretrain")
        if self.data_format not in {"csv", "packed_v1"}:
            raise ValueError("data_format must be csv or packed_v1")
        if self.precision != "bf16":
            raise ValueError("scratch distributed training currently requires bf16 compute")
        for name in (
            "sequence_length",
            "micro_batch_token_budget",
            "max_micro_batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "log_every",
            "checkpoint_every",
            "fsdp_shard_size",
            "batch_lookahead",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.micro_batch_token_budget < self.sequence_length:
            raise ValueError("micro_batch_token_budget must fit one maximum-length sequence")
        if self.data_format != "packed_v1" and self.batch_lookahead != 1:
            raise ValueError("batch_lookahead is supported only for packed_v1 data")
        if self.warmup_steps < 0 or self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be nonnegative and smaller than max_steps")
        if not 0.0 < self.learning_rate:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.min_learning_rate <= self.learning_rate:
            raise ValueError("min_learning_rate must be in [0, learning_rate]")
        if not 0.0 <= self.weight_decay:
            raise ValueError("weight_decay must be nonnegative")
        if not 0.0 < self.beta1 < 1.0 or not 0.0 < self.beta2 < 1.0:
            raise ValueError("Adam betas must be in (0, 1)")
        if self.adam_epsilon <= 0 or self.max_grad_norm <= 0:
            raise ValueError("Adam epsilon and max_grad_norm must be positive")
        if not 0.0 < self.mlm_probability <= 1.0:
            raise ValueError("mlm_probability must be in (0, 1]")
        if self.target_tokens is not None and self.target_tokens <= 0:
            raise ValueError("target_tokens must be positive when supplied")
        if self.checkpoint_keep_last is not None and self.checkpoint_keep_last <= 0:
            raise ValueError("checkpoint_keep_last must be positive when supplied")
        if self.topology_migration is not None:
            expected = {
                "from_world_size",
                "to_world_size",
                "from_step",
                "from_tokens",
                "checkpoint_manifest_sha256",
                "source_tree_sha256",
                "training_code_sha256",
                "train_config_sha256",
            }
            received = set(self.topology_migration)
            if received != expected:
                raise ValueError(
                    "topology_migration fields differ: "
                    f"missing={sorted(expected - received)}, "
                    f"unknown={sorted(received - expected)}"
                )
            for name in ("from_world_size", "to_world_size", "from_step", "from_tokens"):
                if int(self.topology_migration[name]) <= 0:
                    raise ValueError(f"topology_migration {name} must be positive")
            if int(self.topology_migration["from_world_size"]) == int(
                self.topology_migration["to_world_size"]
            ):
                raise ValueError("topology_migration must change world size")
            if int(self.topology_migration["to_world_size"]) != self.fsdp_shard_size:
                raise ValueError(
                    "topology_migration to_world_size must equal fsdp_shard_size"
                )
            for name in (
                "checkpoint_manifest_sha256",
                "source_tree_sha256",
                "training_code_sha256",
                "train_config_sha256",
            ):
                if not re.fullmatch(r"[0-9a-f]{64}", str(self.topology_migration[name])):
                    raise ValueError(f"topology_migration {name} must be SHA-256")
            if self.run_kind == "scale_test" and (
                self.target_tokens is not None
                or self.checkpoint_at_start
                or self.checkpoint_at_end
            ):
                raise ValueError(
                    "topology migration scale test must be checkpoint-free and token-unbound"
                )
        if self.run_kind == "pretrain":
            if self.target_tokens is None:
                raise ValueError("pretrain requires target_tokens")
            if self.target_tokens < MINIMUM_SCRATCH_7B_PRETRAIN_TOKENS:
                raise ValueError(
                    "7B scratch pretrain requires at least "
                    f"{MINIMUM_SCRATCH_7B_PRETRAIN_TOKENS:,} tokens"
                )
            if not self.data_manifest:
                raise ValueError("pretrain requires a hash-bound data_manifest")
            if self.data_format != "packed_v1":
                raise ValueError("formal 7B pretraining requires packed_v1 data")
            if not self.checkpoint_at_start:
                raise ValueError("pretrain requires a step-zero random seed checkpoint")
            if not self.checkpoint_at_end:
                raise ValueError("pretrain requires a final checkpoint")


def _find_project_root(config_path: Path) -> Path:
    for parent in (config_path.parent, *config_path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path(__file__).resolve().parents[3]


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _resolve_target_steps(
    train_config: ScratchTrainConfig,
    resumed_step: int,
    max_steps_override: Optional[int],
    segment_end_step: Optional[int],
) -> int:
    if max_steps_override is not None and segment_end_step is not None:
        raise ValueError("max_steps_override and segment_end_step are mutually exclusive")
    if train_config.run_kind == "pretrain" and max_steps_override is not None:
        raise ValueError("formal pretraining forbids max_steps_override")
    if segment_end_step is None:
        target_steps = max_steps_override or train_config.max_steps
    else:
        if train_config.run_kind != "pretrain":
            raise ValueError("segment_end_step is reserved for formal pretraining")
        if segment_end_step <= resumed_step:
            raise ValueError("segment_end_step must follow the current checkpoint step")
        if segment_end_step > train_config.max_steps:
            raise ValueError("segment_end_step exceeds the configured safety step limit")
        if segment_end_step % train_config.checkpoint_every:
            raise ValueError("segment_end_step must be a declared checkpoint boundary")
        target_steps = segment_end_step
    if target_steps < resumed_step:
        raise ValueError("target max steps precedes the resumed step")
    return target_steps


def _completion_fields(
    train_config: ScratchTrainConfig,
    target_reached: bool,
    step: int,
    target_steps: int,
    segment_end_step: Optional[int],
) -> Tuple[bool, str, str]:
    if train_config.run_kind != "pretrain":
        return step >= target_steps, "SCALE_TEST", "step_limit"
    if target_reached:
        return True, "SCRATCH_PRETRAIN", "token_budget"
    if segment_end_step is not None and step >= target_steps:
        return True, "SCRATCH_PRETRAIN_SEGMENT", "segment_checkpoint"
    return False, "SCALE_TEST", "step_limit"


def _validate_resume_metrics(
    metrics_path: Path,
    resumed_step: int,
    resumed_tokens: int,
) -> None:
    if not metrics_path.exists():
        return
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return
    steps = [int(row["step"]) for row in rows]
    if any(current <= previous for previous, current in zip(steps, steps[1:])):
        raise RuntimeError("existing scratch metrics steps are not strictly increasing")
    last = rows[-1]
    last_step = int(last["step"])
    last_tokens = int(last["tokens_seen"])
    if last_step > resumed_step:
        raise RuntimeError(
            "existing scratch metrics extend beyond the resume checkpoint; "
            "archive and rewind them before resuming"
        )
    if last_step == resumed_step and last_tokens != resumed_tokens:
        raise RuntimeError(
            "existing scratch metrics disagree with the resume checkpoint token count"
        )
    if last_tokens > resumed_tokens:
        raise RuntimeError(
            "existing scratch metrics tokens exceed the resume checkpoint"
        )


def _distributed_context() -> Tuple[int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not 0 <= rank < world_size:
        raise ValueError("RANK must be in [0, WORLD_SIZE)")
    return world_size, rank, local_rank, world_size > 1


def _learning_rate_schedule_end_step(
    config: ScratchTrainConfig, target_steps: int
) -> int:
    """Keep formal-pretraining LR independent of operational segment boundaries."""

    if config.run_kind == "pretrain":
        return config.max_steps
    return target_steps


def _learning_rate(config: ScratchTrainConfig, step: int, schedule_end_step: int) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * float(step + 1) / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(
        1, schedule_end_step - config.warmup_steps
    )
    return config.learning_rate + min(1.0, progress) * (
        config.min_learning_rate - config.learning_rate
    )


def _checkpoint_files_for_rank(path: Path, rank: int) -> List[Path]:
    pattern = re.compile(rf"^__{rank}_[0-9]+\.distcp$")
    return [candidate for candidate in sorted(path.iterdir()) if pattern.match(candidate.name)]


def _hash_files(paths: List[Path], root: Path) -> Dict[str, Dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    }


def _verify_file_entries(root: Path, entries: Dict[str, Dict[str, Any]]) -> None:
    for relative, expected in entries.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"checkpoint file is missing: {relative}")
        if path.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"checkpoint file size changed: {relative}")
        if sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"checkpoint file hash changed: {relative}")


def _validate_topology_migration_manifest(
    path: Path,
    config: ScratchTrainConfig,
    expected_provenance: Dict[str, Optional[str]],
    world_size: int,
) -> Dict[str, Any]:
    migration = config.topology_migration
    if migration is None:
        raise RuntimeError("topology migration policy is missing")
    manifest_path = path.resolve() / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest is missing: {manifest_path}")
    if sha256_file(manifest_path) != migration["checkpoint_manifest_sha256"]:
        raise RuntimeError("topology migration checkpoint manifest hash differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "world_size": int(migration["from_world_size"]),
        "step": int(migration["from_step"]),
        "tokens_seen": int(migration["from_tokens"]),
    }
    for name, value in expected.items():
        if int(manifest.get(name, -1)) != value:
            raise RuntimeError(f"topology migration checkpoint {name} differs")
    if world_size != int(migration["to_world_size"]):
        raise RuntimeError("topology migration runtime world size differs")
    provenance = manifest.get("provenance", {})
    for name in ("source_tree_sha256", "training_code_sha256", "train_config_sha256"):
        if provenance.get(name) != migration[name]:
            raise RuntimeError(f"topology migration source {name} differs")
    for name in ("data_manifest_sha256", "model_config_sha256", "tokenizer_sha256"):
        if provenance.get(name) != expected_provenance[name]:
            raise RuntimeError(f"topology migration stable {name} differs")
    initialization = manifest.get("initialization", {})
    if (
        initialization.get("type") != "random"
        or initialization.get("external_checkpoint_loaded") is not False
    ):
        raise RuntimeError("topology migration checkpoint is not scratch initialization")
    return manifest


def _topology_migration_seed(manifest_sha256: str, step: int) -> int:
    digest = hashlib.sha256(
        f"scratch-topology-migration:{manifest_sha256}:{step}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _topology_migration_required(
    manifest: Dict[str, Any],
    config: ScratchTrainConfig,
    world_size: int,
) -> bool:
    migration = config.topology_migration
    if migration is None:
        return False
    if world_size != int(migration["to_world_size"]):
        raise RuntimeError("topology migration runtime world size differs")
    checkpoint_world_size = int(manifest.get("world_size", -1))
    if checkpoint_world_size == int(migration["from_world_size"]):
        return True
    if checkpoint_world_size == int(migration["to_world_size"]):
        return False
    raise RuntimeError("checkpoint world size is outside the frozen topology lineage")


def _save_checkpoint(
    output_dir: Path,
    step: int,
    model: ScratchProteinModel,
    optimizer: torch.optim.Optimizer,
    batcher: Any,
    provenance: Dict[str, Optional[str]],
    initialization: Dict[str, Any],
    tokens_seen: int,
    elapsed_training_seconds: float,
    world_size: int,
    rank: int,
    distributed: bool,
) -> Path:
    root = output_dir / "checkpoints"
    final = root / f"step-{step:08d}"
    temporary = root / f".step-{step:08d}.tmp"
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        if final.exists() or temporary.exists():
            raise FileExistsError(f"checkpoint destination already exists for step {step}")
        temporary.mkdir()
    if distributed:
        torch.distributed.barrier()

    model_state, optimizer_state = get_state_dict(model, optimizer)
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=temporary,
        no_dist=not distributed,
    )
    rank_state = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "step": step,
        "tokens_seen": tokens_seen,
        "elapsed_training_seconds": elapsed_training_seconds,
        "world_size": world_size,
        "rank": rank,
        "batcher": batcher.state_dict(),
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": (
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        ),
    }
    rank_path = temporary / f"rank-{rank:05d}.pt"
    torch.save(rank_state, rank_path)
    if distributed:
        torch.distributed.barrier()

    owned = _checkpoint_files_for_rank(temporary, rank) + [rank_path]
    local_hashes = _hash_files(owned, temporary)
    gathered_hashes: List[Optional[Dict[str, Dict[str, Any]]]] = [None] * world_size
    if distributed:
        torch.distributed.all_gather_object(gathered_hashes, local_hashes)
    else:
        gathered_hashes = [local_hashes]

    if rank == 0:
        all_hashes: Dict[str, Dict[str, Any]] = {}
        for shard_hashes in gathered_hashes:
            if shard_hashes is None:
                raise RuntimeError("a rank did not return checkpoint hashes")
            overlap = set(all_hashes) & set(shard_hashes)
            if overlap:
                raise RuntimeError(f"checkpoint files have duplicate ownership: {sorted(overlap)}")
            all_hashes.update(shard_hashes)
        metadata = temporary / ".metadata"
        if not metadata.is_file():
            raise RuntimeError("distributed checkpoint metadata is missing")
        all_hashes.update(_hash_files([metadata], temporary))
        actual = {
            path.relative_to(temporary).as_posix()
            for path in temporary.iterdir()
            if path.is_file() and path.name != "manifest.json"
        }
        if actual != set(all_hashes):
            raise RuntimeError(
                "checkpoint manifest ownership is incomplete: "
                f"missing={sorted(actual - set(all_hashes))}, "
                f"extra={sorted(set(all_hashes) - actual)}"
            )
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "step": step,
            "tokens_seen": tokens_seen,
            "world_size": world_size,
            "parameter_count": model.config.parameter_count,
            "trainable_parameter_count": sum(p.numel() for p in model.parameters()),
            "initialization": initialization,
            "provenance": provenance,
            "files": dict(sorted(all_hashes.items())),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, final)
    if distributed:
        torch.distributed.barrier()
    return final


def _load_checkpoint(
    path: Path,
    model: ScratchProteinModel,
    optimizer: torch.optim.Optimizer,
    batcher: Any,
    expected_provenance: Dict[str, Optional[str]],
    world_size: int,
    rank: int,
    distributed: bool,
    topology_migration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    path = path.resolve()
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("checkpoint schema is unsupported")
    migrating = topology_migration is not None
    if migrating:
        _validate_topology_migration_manifest(
            path,
            topology_migration["config"],
            expected_provenance,
            world_size,
        )
    else:
        if int(manifest.get("world_size", -1)) != world_size:
            raise RuntimeError("checkpoint data/RNG state requires the original world size")
        if manifest.get("provenance") != expected_provenance:
            raise RuntimeError("checkpoint provenance differs from the current run")
    if int(manifest.get("parameter_count", -1)) != model.config.parameter_count:
        raise RuntimeError("checkpoint model parameter count differs from config")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("checkpoint manifest has no file map")
    if migrating:
        selected = {
            relative: files[relative]
            for index, relative in enumerate(sorted(files))
            if index % world_size == rank
        }
    else:
        selected = {
            relative: expected
            for relative, expected in files.items()
            if relative == ".metadata"
            and rank == 0
            or relative.startswith(f"__{rank}_")
            or relative == f"rank-{rank:05d}.pt"
        }
    verification_error: Optional[str] = None
    try:
        _verify_file_entries(path, selected)
    except Exception as exc:  # propagate one deterministic distributed failure
        verification_error = f"{type(exc).__name__}: {exc}"
    if distributed:
        errors: List[Optional[str]] = [None] * world_size
        torch.distributed.all_gather_object(errors, verification_error)
        failures = [error for error in errors if error]
        if failures:
            raise RuntimeError("checkpoint integrity verification failed: " + "; ".join(failures))
    elif verification_error:
        raise RuntimeError(verification_error)

    model_state, optimizer_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(state, checkpoint_id=path, no_dist=not distributed)
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
    )
    if migrating:
        old_world_size = int(manifest["world_size"])
        rank_states = [
            torch.load(
                path / f"rank-{old_rank:05d}.pt",
                map_location="cpu",
                weights_only=False,
            )
            for old_rank in range(old_world_size)
        ]
        first = rank_states[0]
        for old_rank, rank_state in enumerate(rank_states):
            if (
                int(rank_state.get("rank", -1)) != old_rank
                or int(rank_state.get("world_size", -1)) != old_world_size
                or int(rank_state.get("step", -1)) != int(first["step"])
                or int(rank_state.get("tokens_seen", -1)) != int(first["tokens_seen"])
            ):
                raise RuntimeError("topology migration rank state is inconsistent")
        migration_seed = _topology_migration_seed(
            topology_migration["manifest_sha256"], int(first["step"])
        )
        batcher.load_topology_migration_state(
            [rank_state["batcher"] for rank_state in rank_states],
            migration_seed,
        )
        random.seed(migration_seed + rank)
        torch.manual_seed(migration_seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(migration_seed + rank)
        rank_state = first
    else:
        rank_state = torch.load(
            path / f"rank-{rank:05d}.pt", map_location="cpu", weights_only=False
        )
        if int(rank_state["rank"]) != rank or int(rank_state["world_size"]) != world_size:
            raise RuntimeError("checkpoint rank state has invalid topology")
        batcher.load_state_dict(rank_state["batcher"])
        random.setstate(rank_state["python_random_state"])
        torch.set_rng_state(rank_state["torch_random_state"])
        if torch.cuda.is_available() and rank_state["cuda_random_state"] is not None:
            torch.cuda.set_rng_state(rank_state["cuda_random_state"])
    return {
        "step": int(rank_state["step"]),
        "tokens_seen": int(rank_state["tokens_seen"]),
        "elapsed_training_seconds": float(rank_state["elapsed_training_seconds"]),
        "manifest_sha256": sha256_file(manifest_path),
        "path": str(path),
        "initialization": manifest["initialization"],
        "topology_migration": (
            {
                "from_world_size": int(manifest["world_size"]),
                "to_world_size": world_size,
                "checkpoint_manifest_sha256": topology_migration["manifest_sha256"],
                "data_cursor_policy": "exact_unconsumed_path_cursor_repartition",
                "rng_policy": "sha256(checkpoint_manifest,step)+new_rank",
            }
            if migrating
            else None
        ),
    }


def _prune_checkpoints(output_dir: Path, keep_last: Optional[int]) -> List[Path]:
    """Keep only the newest requested number of restart checkpoints."""
    if keep_last is None:
        return []
    root = output_dir / "checkpoints"
    if not root.is_dir():
        return []
    checkpoints: List[Tuple[int, Path]] = []
    for candidate in root.iterdir():
        match = re.fullmatch(r"step-([0-9]{8})", candidate.name)
        if not match or not candidate.is_dir():
            continue
        step = int(match.group(1))
        manifest_path = candidate / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"checkpoint selected for retention has no manifest: {candidate}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("step", -1)) != step:
            raise RuntimeError(f"checkpoint retention found a mismatched manifest: {candidate}")
        checkpoints.append((step, candidate))
    checkpoints.sort()
    pruned = [path for _, path in checkpoints[:-keep_last]]
    for path in pruned:
        shutil.rmtree(path)
    return pruned


def _parallelize_fsdp2(
    model: ScratchProteinModel,
    world_size: int,
    rank: int,
    shard_size: int,
    activation_checkpointing: bool,
    reshard_after_forward: bool,
    device: torch.device,
    seed: int,
) -> Tuple[ScratchProteinModel, str, int]:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        checkpoint_wrapper,
    )
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    if world_size % shard_size:
        raise ValueError("WORLD_SIZE must be divisible by fsdp_shard_size")
    if world_size == shard_size:
        mesh = init_device_mesh("cuda", (shard_size,), mesh_dim_names=("shard",))
        strategy = "FSDP2_FULL_SHARD"
        shard_rank = rank
    else:
        mesh = init_device_mesh(
            "cuda",
            (world_size // shard_size, shard_size),
            mesh_dim_names=("replicate", "shard"),
        )
        strategy = "FSDP2_HYBRID_SHARD"
        shard_rank = rank % shard_size

    if activation_checkpointing:
        for index, layer in enumerate(model.layers):
            model.layers[index] = checkpoint_wrapper(
                layer,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                preserve_rng_state=True,
            )
    mixed_precision = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
    )
    for layer in model.layers:
        fully_shard(
            layer,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mixed_precision,
        )
    fully_shard(model, mesh=mesh, reshard_after_forward=False, mp_policy=mixed_precision)
    model.to_empty(device=device)
    torch.manual_seed(seed + shard_rank)
    torch.cuda.manual_seed(seed + shard_rank)
    model.reset_parameters()
    return model, strategy, shard_rank


def _optimizer(model: ScratchProteinModel, config: ScratchTrainConfig) -> torch.optim.AdamW:
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    for _, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    grouped_parameters = sum(p.numel() for p in decay) + sum(
        p.numel() for p in no_decay
    )
    if grouped_parameters != model.config.parameter_count:
        raise RuntimeError("optimizer groups do not cover every model parameter exactly once")
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.adam_epsilon,
        foreach=False,
    )


def _mlm_loss_scale(
    local_masked_tokens: int,
    global_masked_tokens: int,
    data_parallel_size: int,
) -> float:
    """Scale a local mean so averaged data-parallel gradients form a global mean."""

    if local_masked_tokens <= 0:
        raise ValueError("each MLM micro-batch must contain a masked token")
    if global_masked_tokens < local_masked_tokens:
        raise ValueError("global masked-token count is smaller than the local count")
    if data_parallel_size <= 0:
        raise ValueError("data-parallel size must be positive")
    return local_masked_tokens * data_parallel_size / global_masked_tokens


def run_scratch_training(
    train_config_path: str,
    resume: Optional[str] = None,
    max_steps_override: Optional[int] = None,
    segment_end_step: Optional[int] = None,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a scratch scale test or a hash-bound 7B pretraining job.

    There is intentionally no initialization-checkpoint argument. A resume may
    load only this trainer's own hash-bound distributed checkpoint lineage.
    """

    config_path = Path(train_config_path).resolve()
    train_config = ScratchTrainConfig.from_json(str(config_path))
    project_root = _find_project_root(config_path)

    def resolve(path: str) -> str:
        candidate = Path(path)
        return str(candidate if candidate.is_absolute() else project_root / candidate)

    model_config_path = Path(resolve(train_config.model_config))
    tokenizer_path = Path(resolve(train_config.tokenizer_json))
    model_config = load_scratch_model_config(str(model_config_path))
    tokenizer = EsmcTokenizer.from_tokenizer_json(str(tokenizer_path))
    vocabulary_size = max(tokenizer.vocabulary.values()) + 1
    if vocabulary_size != model_config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match scratch model")
    if tokenizer.pad_token_id != model_config.pad_token_id:
        raise ValueError("tokenizer pad id does not match scratch model")
    if tokenizer.mask_token_id != model_config.mask_token_id:
        raise ValueError("tokenizer mask id does not match scratch model")
    if train_config.sequence_length > model_config.max_sequence_length:
        raise ValueError("training sequence_length exceeds the model contract")

    paths = sorted(glob.glob(resolve(train_config.train_glob)))
    if not paths:
        raise ValueError(f"train_glob matched no files: {train_config.train_glob}")
    output_dir = Path(resolve(train_config.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    data_manifest_path = (
        Path(resolve(train_config.data_manifest)) if train_config.data_manifest else None
    )
    if data_manifest_path is not None and not data_manifest_path.is_file():
        raise FileNotFoundError(f"data manifest is missing: {data_manifest_path}")

    provenance: Dict[str, Optional[str]] = {
        "train_config_sha256": sha256_file(config_path),
        "model_config_sha256": sha256_file(model_config_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "data_manifest_sha256": (
            sha256_file(data_manifest_path) if data_manifest_path is not None else None
        ),
        "training_code_sha256": source_tree_sha256(Path(__file__).resolve().parent),
        "source_tree_sha256": source_tree_sha256(project_root),
    }

    world_size, rank, local_rank, distributed = _distributed_context()
    if model_config.is_7b_scale and (not distributed or world_size < 6):
        raise RuntimeError("scratch 7B requires at least six distributed CUDA ranks")
    if (
        model_config.is_7b_scale
        and train_config.run_kind == "pretrain"
        and world_size < 8
        and train_config.topology_migration is None
    ):
        raise RuntimeError(
            "scratch 7B pretraining below eight ranks requires a frozen topology migration"
        )
    if train_config.run_kind == "pretrain" and not model_config.is_7b_scale:
        raise RuntimeError("formal scratch pretraining requires the declared 7B architecture")
    if len(paths) < world_size:
        raise RuntimeError("training requires at least one input shard per distributed rank")
    if distributed:
        if device_name is not None or not torch.cuda.is_available():
            raise ValueError("distributed scratch training requires torchrun-managed CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl")
        collective_probe = torch.tensor(float(rank + 1), dtype=torch.float64, device=device)
        torch.distributed.all_reduce(collective_probe)
        expected_probe = float(world_size * (world_size + 1) // 2)
        if float(collective_probe.item()) != expected_probe:
            raise RuntimeError(
                "distributed preflight sum differs: "
                f"{float(collective_probe.item())} != {expected_probe}"
            )
        if rank == 0:
            print(
                json.dumps(
                    {
                        "distributed_preflight": "passed",
                        "world_size": world_size,
                        "collective_sum": expected_probe,
                        "socket_interface": os.environ.get("NCCL_SOCKET_IFNAME"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    else:
        device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

    if train_config.data_format == "packed_v1":
        if data_manifest_path is None:
            raise RuntimeError("packed_v1 data requires a manifest")
        verification_error: Optional[str] = None
        try:
            verify_packed_manifest(
                str(data_manifest_path),
                expected_index_paths=paths,
                expected_tokenizer_sha256=provenance["tokenizer_sha256"],
                content_shard_index=rank if distributed else 0,
                content_shard_count=world_size if distributed else 1,
            )
        except Exception as exc:
            verification_error = f"rank {rank}: {type(exc).__name__}: {exc}"
        if distributed:
            verification: List[Optional[str]] = [None] * world_size
            torch.distributed.all_gather_object(verification, verification_error)
            verification_error = next((item for item in verification if item), None)
        if verification_error:
            raise RuntimeError("packed data verification failed: " + verification_error)

    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_config.seed)
        torch.cuda.reset_peak_memory_stats(device)

    if distributed:
        with torch.device("meta"):
            model = ScratchProteinModel(model_config)
        model, parallelism, _ = _parallelize_fsdp2(
            model,
            world_size,
            rank,
            train_config.fsdp_shard_size,
            train_config.activation_checkpointing,
            train_config.fsdp_reshard_after_forward,
            device,
            train_config.seed,
        )
    else:
        model = ScratchProteinModel(model_config).to(device)
        parallelism = "NONE"
    model.train()
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    if trainable_parameters != model_config.parameter_count:
        raise RuntimeError("instantiated model parameter count differs from the config")
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("scratch model contains frozen parameters")
    optimizer = _optimizer(model, train_config)

    if train_config.data_format == "packed_v1":
        batcher: Any = PackedProteinBatcher(
            paths,
            tokenizer,
            train_config.sequence_length,
            train_config.micro_batch_token_budget,
            train_config.max_micro_batch_size,
            train_config.seed,
            train_config.mlm_probability,
            rank=rank,
            world_size=world_size,
            batch_lookahead=train_config.batch_lookahead,
        )
    else:
        batcher = RestartableBatcher(
            paths,
            tokenizer,
            train_config.sequence_length,
            batch_size=1,
            seed=train_config.seed,
            mlm_probability=train_config.mlm_probability,
            rank=rank,
            world_size=world_size,
            micro_batch_token_budget=train_config.micro_batch_token_budget,
            max_micro_batch_size=train_config.max_micro_batch_size,
        )
    migration_manifest: Optional[Dict[str, Any]] = None
    migration_required = False
    if train_config.topology_migration is not None:
        if resume is None:
            raise RuntimeError("topology migration requires an explicit resume checkpoint")
        resume_manifest_path = Path(resume).resolve() / "manifest.json"
        if not resume_manifest_path.is_file():
            raise FileNotFoundError(
                f"checkpoint manifest is missing: {resume_manifest_path}"
            )
        resume_manifest = json.loads(resume_manifest_path.read_text(encoding="utf-8"))
        migration_required = _topology_migration_required(
            resume_manifest, train_config, world_size
        )
        if migration_required:
            migration_manifest = _validate_topology_migration_manifest(
                Path(resume), train_config, provenance, world_size
            )
        else:
            migration_manifest = resume_manifest
        initialization = copy.deepcopy(migration_manifest["initialization"])
    else:
        initialization = {
            "type": "random",
            "seed": train_config.seed,
            "shard_seed_policy": "seed + intra_shard_coordinate",
            "shard_seed_cardinality": (
                train_config.fsdp_shard_size if distributed else 1
            ),
            "external_checkpoint_loaded": False,
            "pretrained_backbone_loaded": False,
            "all_parameters_trainable": True,
            "parameter_count": model_config.parameter_count,
            "model_config_sha256": provenance["model_config_sha256"],
            "topology_dependent_seed": distributed,
        }
    init_path = output_dir / "initialization.json"
    if rank == 0:
        if init_path.exists():
            existing = json.loads(init_path.read_text(encoding="utf-8"))
            if existing != initialization:
                raise RuntimeError("existing initialization receipt differs from this run")
        else:
            _atomic_json(init_path, initialization)
    if distributed:
        torch.distributed.barrier()

    step = 0
    tokens_seen = 0
    elapsed_offset = 0.0
    resumed_from: Optional[Dict[str, Any]] = None
    if resume:
        resumed_from = _load_checkpoint(
            Path(resume),
            model,
            optimizer,
            batcher,
            provenance,
            world_size,
            rank,
            distributed,
            (
                {
                    "config": train_config,
                    "manifest_sha256": train_config.topology_migration[
                        "checkpoint_manifest_sha256"
                    ],
                }
                if migration_required
                else None
            ),
        )
        step = resumed_from["step"]
        tokens_seen = resumed_from["tokens_seen"]
        elapsed_offset = resumed_from["elapsed_training_seconds"]
        if resumed_from["initialization"] != initialization:
            raise RuntimeError("checkpoint random initialization lineage differs")

    target_steps = _resolve_target_steps(
        train_config,
        step,
        max_steps_override,
        segment_end_step,
    )
    learning_rate_schedule_end_step = _learning_rate_schedule_end_step(
        train_config, target_steps
    )

    metrics_path = output_dir / "metrics.jsonl"
    if resume:
        _validate_resume_metrics(metrics_path, step, tokens_seen)

    if train_config.checkpoint_at_start and step == 0 and resume is None:
        checkpoint = _save_checkpoint(
            output_dir,
            0,
            model,
            optimizer,
            batcher,
            provenance,
            initialization,
            0,
            0.0,
            world_size,
            rank,
            distributed,
        )
        if rank == 0:
            print(f"checkpoint={checkpoint}", flush=True)

    started = time.perf_counter()
    run_tokens = 0
    optimizer.zero_grad(set_to_none=True)
    while step < target_steps and (
        train_config.target_tokens is None or tokens_seen < train_config.target_tokens
    ):
        step_started = time.perf_counter()
        raw_batches = []
        local_masked_tokens = 0
        step_tokens = 0
        step_padded_tokens = 0
        for _ in range(train_config.gradient_accumulation_steps):
            raw_batch = batcher.next()
            raw_batches.append(raw_batch)
            local_masked_tokens += int(raw_batch["protein_labels"].ne(-100).sum().item())
            step_tokens += int(raw_batch["protein_attention_mask"].sum().item())
            step_padded_tokens += int(raw_batch["protein_attention_mask"].numel())

        step_masked_tokens = local_masked_tokens
        if distributed:
            counts = torch.tensor(
                [
                    float(step_masked_tokens),
                    float(step_tokens),
                    float(step_padded_tokens),
                ],
                dtype=torch.float64,
                device=device,
            )
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
            step_masked_tokens = int(counts[0])
            step_tokens = int(counts[1])
            step_padded_tokens = int(counts[2])
        if step_masked_tokens <= 0:
            raise RuntimeError("scratch training step contains no masked MLM targets")

        local_loss_numerator = 0.0
        for raw_batch in raw_batches:
            input_ids = raw_batch["protein_input_ids"].to(device)
            attention_mask = raw_batch["protein_attention_mask"].to(device)
            labels = raw_batch["protein_labels"].to(device)
            micro_batch_masked_tokens = int(labels.ne(-100).sum().item())
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if not distributed and device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                loss = model(input_ids, attention_mask, labels)["loss"]
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("non-finite scratch training loss")
            loss_scale = _mlm_loss_scale(
                micro_batch_masked_tokens,
                step_masked_tokens,
                world_size if distributed else 1,
            )
            (loss * loss_scale).backward()
            local_loss_numerator += float(loss.detach()) * micro_batch_masked_tokens

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.max_grad_norm, error_if_nonfinite=True
        )
        learning_rate = _learning_rate(
            train_config, step, learning_rate_schedule_end_step
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - step_started

        if distributed:
            aggregate = torch.tensor(
                [
                    local_loss_numerator,
                    step_seconds,
                ],
                dtype=torch.float64,
                device=device,
            )
            torch.distributed.all_reduce(aggregate[:1], op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(aggregate[1:], op=torch.distributed.ReduceOp.MAX)
            loss_numerator = float(aggregate[0])
            step_seconds = float(aggregate[1])
        else:
            loss_numerator = local_loss_numerator
        tokens_seen += step_tokens
        run_tokens += step_tokens
        peak_memory_mib = (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        )
        if distributed and peak_memory_mib is not None:
            peak = torch.tensor(peak_memory_mib, dtype=torch.float64, device=device)
            torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
            peak_memory_mib = float(peak)
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()
        metric = {
            "step": step,
            "loss": loss_numerator / step_masked_tokens,
            "masked_tokens": step_masked_tokens,
            "learning_rate": learning_rate,
            "grad_norm": float(grad_norm.detach()),
            "tokens": step_tokens,
            "padded_tokens": step_padded_tokens,
            "packing_efficiency": step_tokens / step_padded_tokens,
            "tokens_seen": tokens_seen,
            "tokens_per_second": step_tokens / step_seconds,
            "step_seconds": step_seconds,
            "peak_memory_mib": peak_memory_mib,
            "run_kind": train_config.run_kind,
        }
        if rank == 0 and step % train_config.log_every == 0:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metric, sort_keys=True) + "\n")
            print(json.dumps(metric, sort_keys=True), flush=True)

        target_reached = (
            train_config.target_tokens is not None
            and tokens_seen >= train_config.target_tokens
        )
        if (
            step % train_config.checkpoint_every == 0
            or (step == target_steps and train_config.checkpoint_at_end)
            or target_reached
        ):
            checkpoint = _save_checkpoint(
                output_dir,
                step,
                model,
                optimizer,
                batcher,
                provenance,
                initialization,
                tokens_seen,
                elapsed_offset + (time.perf_counter() - started),
                world_size,
                rank,
                distributed,
            )
            if rank == 0:
                print(f"checkpoint={checkpoint}", flush=True)
                for pruned in _prune_checkpoints(
                    output_dir, train_config.checkpoint_keep_last
                ):
                    print(f"pruned_checkpoint={pruned}", flush=True)
            if distributed:
                torch.distributed.barrier()

    elapsed = time.perf_counter() - started
    target_reached = (
        train_config.target_tokens is not None
        and tokens_seen >= train_config.target_tokens
    )
    completed, evidence_level, stopped_by = _completion_fields(
        train_config,
        target_reached,
        step,
        target_steps,
        segment_end_step,
    )
    receipt = {
        "schema_version": 1,
        "status": "passed" if completed else "failed",
        "evidence_level": evidence_level,
        "name": train_config.name,
        "run_kind": train_config.run_kind,
        "data_format": train_config.data_format,
        "step": step,
        "tokens": tokens_seen,
        "target_tokens": train_config.target_tokens,
        "target_reached": target_reached,
        "segment_end_step": segment_end_step,
        "learning_rate_schedule_end_step": learning_rate_schedule_end_step,
        "stopped_by": stopped_by,
        "world_size": world_size,
        "rank": rank,
        "parallelism": parallelism,
        "fsdp_shard_size": train_config.fsdp_shard_size if distributed else None,
        "activation_checkpointing": train_config.activation_checkpointing,
        "checkpoint_at_end": train_config.checkpoint_at_end,
        "batch_lookahead": train_config.batch_lookahead,
        "loss_normalization": "global_masked_tokens",
        "fsdp_reshard_after_forward": train_config.fsdp_reshard_after_forward,
        "topology_migration": (
            train_config.topology_migration if migration_required else None
        ),
        "precision": train_config.precision if device.type == "cuda" else "fp32_test_only",
        "parameter_count": model_config.parameter_count,
        "trainable_parameter_count": trainable_parameters,
        "initialization": initialization,
        "resumed_from": resumed_from,
        "tokens_this_run": run_tokens,
        "elapsed_seconds_this_run": elapsed,
        "elapsed_seconds": elapsed_offset + elapsed,
        "average_tokens_per_second_this_run": run_tokens / elapsed if elapsed else None,
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "provenance": provenance,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if rank == 0:
        _atomic_json(output_dir / f"receipt-step-{step:08d}.json", receipt)
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if not completed:
        raise RuntimeError("scratch training stopped before its declared completion condition")
    return receipt
