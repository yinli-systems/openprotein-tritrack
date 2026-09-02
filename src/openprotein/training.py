"""Single-rank pilot trainer with deterministic restart receipts.

This module is intentionally the pilot path. Multi-GPU production pretraining
will use sharded optimizer/checkpoint state; the receipt labels below prevent a
single-rank run from being mistaken for that final evidence level.
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch

from openprotein.artifacts import verify_artifact_manifest
from openprotein.config import ModelConfig, load_model_config
from openprotein.data.streaming import (
    CSVProteinStream,
    JointMultimodalCSVStream,
    MultimodalCSVStream,
    ProteinMLMCollator,
    SequenceStructureTextCollator,
    SequenceTextCollator,
    SequenceStructureCollator,
    StructureCSVStream,
)
from openprotein.data.structure import STRUCTURE_FEATURE_DIMENSION, verify_structure_database
from openprotein.data.text_tokenizer import SentencePieceTextTokenizer
from openprotein.data.tokenizer import EsmcTokenizer
from openprotein.model import OpenProteinModel
from openprotein.provenance import sha256_file, source_tree_sha256


@dataclass(frozen=True)
class TrainConfig:
    name: str
    model_config: str
    tokenizer_json: str
    train_glob: str
    output_dir: str
    sequence_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    max_grad_norm: float
    mlm_probability: float
    seed: int
    log_every: int
    checkpoint_every: int
    precision: str
    model_artifact_manifest: Optional[str] = None
    data_manifest: Optional[str] = None
    task: str = "sequence_mlm"
    text_tokenizer_model: Optional[str] = None
    text_length: int = 128
    attention_policy: str = "bidirectional"
    loss_weights: Optional[Dict[str, float]] = None
    structure_database: Optional[str] = None
    structure_manifest: Optional[str] = None
    geometry_pairs: int = 128
    min_structure_confidence: float = 70.0
    detach_geometry_backbone: bool = False
    micro_batch_token_budget: Optional[int] = None
    max_micro_batch_size: Optional[int] = None
    max_tokens: Optional[int] = None
    structure_mask_probability: float = 0.15
    structure_dropout_probability: float = 0.0
    replay_train_glob: Optional[str] = None
    replay_data_manifest: Optional[str] = None
    replay_every: int = 4
    replay_loss_weight: float = 1.0
    warmup_tokens: Optional[int] = None

    @classmethod
    def from_json(cls, path: str) -> "TrainConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        config = cls(**raw)
        if config.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if not 0.0 < config.mlm_probability <= 1.0:
            raise ValueError("mlm_probability must be in (0, 1]")
        valid_tasks = {
            "sequence_mlm",
            "sequence_text",
            "sequence_structure",
            "sequence_structure_text",
        }
        if config.task not in valid_tasks:
            raise ValueError("unsupported training task")
        if config.task in {"sequence_text", "sequence_structure_text"} and not config.text_tokenizer_model:
            raise ValueError("text training requires text_tokenizer_model")
        if config.attention_policy not in {"bidirectional", "prefix_lm"}:
            raise ValueError("attention_policy must be bidirectional or prefix_lm")
        if config.task in {"sequence_structure", "sequence_structure_text"} and (
            not config.structure_database or not config.structure_manifest
        ):
            raise ValueError("structure training requires structure database and manifest")
        if config.detach_geometry_backbone and config.task != "sequence_structure":
            raise ValueError("detach_geometry_backbone is only valid for sequence_structure")
        if config.geometry_pairs <= 0:
            raise ValueError("geometry_pairs must be positive")
        if not 0.0 < config.structure_mask_probability <= 1.0:
            raise ValueError("structure_mask_probability must be in (0, 1]")
        if not 0.0 <= config.structure_dropout_probability < 1.0:
            raise ValueError("structure_dropout_probability must be in [0, 1)")
        if config.text_length < 2:
            raise ValueError("text_length must leave room for BOS and EOS")
        if config.micro_batch_token_budget is not None:
            if config.task != "sequence_mlm":
                raise ValueError("dynamic token batching is currently sequence_mlm-only")
            if config.micro_batch_token_budget < config.sequence_length:
                raise ValueError(
                    "micro_batch_token_budget must fit one maximum-length sequence"
                )
            if config.max_micro_batch_size is None or config.max_micro_batch_size <= 0:
                raise ValueError("dynamic token batching requires max_micro_batch_size")
        elif config.max_micro_batch_size is not None:
            raise ValueError("max_micro_batch_size requires micro_batch_token_budget")
        if config.max_tokens is not None:
            if config.max_tokens <= 0:
                raise ValueError("max_tokens must be positive")
        if config.warmup_tokens is not None:
            if config.max_tokens is None:
                raise ValueError("warmup_tokens requires max_tokens")
            if not 0 < config.warmup_tokens < config.max_tokens:
                raise ValueError("warmup_tokens must be in (0, max_tokens)")
        for field in (
            "sequence_length",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "log_every",
            "checkpoint_every",
        ):
            if getattr(config, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if (
            config.task in {"sequence_text", "sequence_structure_text"}
            and config.loss_weights
            and float(config.loss_weights.get("contrastive", 0.0)) != 0.0
            and config.micro_batch_size < 2
        ):
            raise ValueError("contrastive training requires micro_batch_size >= 2")
        if config.replay_train_glob is not None:
            if config.task != "sequence_structure_text":
                raise ValueError("sequence replay is only supported for the joint task")
            if config.replay_every < 2:
                raise ValueError("replay_every must leave at least one joint microbatch")
            if config.replay_loss_weight <= 0.0:
                raise ValueError("replay_loss_weight must be positive")
        elif config.replay_data_manifest is not None:
            raise ValueError("replay_data_manifest requires replay_train_glob")
        if config.loss_weights and not any(
            float(weight) != 0.0 for weight in config.loss_weights.values()
        ):
            raise ValueError("loss_weights must enable at least one objective")
        return config


class RestartableBatcher:
    """Rebuild a deterministic stream and skip raw records on restart."""

    def __init__(
        self,
        paths: List[str],
        tokenizer: EsmcTokenizer,
        sequence_length: int,
        batch_size: int,
        seed: int,
        mlm_probability: float,
        task: str = "sequence_mlm",
        text_tokenizer: Optional[SentencePieceTextTokenizer] = None,
        text_length: int = 128,
        structure_database: Optional[str] = None,
        geometry_pairs: int = 128,
        distance_bins: int = 64,
        min_structure_confidence: float = 70.0,
        structure_mask_probability: float = 0.15,
        structure_dropout_probability: float = 0.0,
        rank: int = 0,
        world_size: int = 1,
        micro_batch_token_budget: Optional[int] = None,
        max_micro_batch_size: Optional[int] = None,
    ) -> None:
        self.paths = paths
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.seed = seed
        self.mlm_probability = mlm_probability
        self.task = task
        self.text_tokenizer = text_tokenizer
        self.text_length = text_length
        self.structure_database = structure_database
        self.geometry_pairs = geometry_pairs
        self.distance_bins = distance_bins
        self.min_structure_confidence = min_structure_confidence
        self.structure_mask_probability = structure_mask_probability
        self.structure_dropout_probability = structure_dropout_probability
        self.rank = rank
        self.world_size = world_size
        self.micro_batch_token_budget = micro_batch_token_budget
        self.max_micro_batch_size = max_micro_batch_size
        self.epoch = 0
        self.records_consumed = 0
        collator_seed = seed + 104_729 * rank
        if task == "sequence_mlm":
            self.collator = ProteinMLMCollator(tokenizer, collator_seed, mlm_probability)
        elif task == "sequence_text" and text_tokenizer is not None:
            self.collator = SequenceTextCollator(
                tokenizer, text_tokenizer, collator_seed, mlm_probability
            )
        elif task == "sequence_structure" and structure_database is not None:
            self.collator = SequenceStructureCollator(
                tokenizer,
                collator_seed,
                geometry_pairs,
                distance_bins,
                mlm_probability,
                min_structure_confidence,
            )
        elif (
            task == "sequence_structure_text"
            and structure_database is not None
            and text_tokenizer is not None
        ):
            self.collator = SequenceStructureTextCollator(
                tokenizer,
                text_tokenizer,
                collator_seed,
                geometry_pairs,
                distance_bins,
                mlm_probability,
                structure_mask_probability,
                structure_dropout_probability,
                min_structure_confidence,
            )
        else:
            raise ValueError("task-specific batcher dependencies are missing")
        self._iterator: Optional[Iterator[Dict[str, object]]] = None
        self._pending: Optional[Dict[str, object]] = None

    def _new_iterator(self) -> Iterator[Dict[str, object]]:
        if self.task == "sequence_mlm":
            stream = CSVProteinStream(
                self.paths,
                self.tokenizer,
                max_length=self.sequence_length,
                seed=self.seed,
                rank=self.rank,
                world_size=self.world_size,
            )
        elif self.task == "sequence_text":
            if self.text_tokenizer is None:
                raise RuntimeError("text tokenizer disappeared from a multimodal batcher")
            stream = MultimodalCSVStream(
                self.paths,
                self.tokenizer,
                self.text_tokenizer,
                protein_max_length=self.sequence_length,
                text_max_length=self.text_length,
                seed=self.seed,
                rank=self.rank,
                world_size=self.world_size,
            )
        elif self.task == "sequence_structure_text":
            if self.text_tokenizer is None or self.structure_database is None:
                raise RuntimeError("joint multimodal batcher dependencies disappeared")
            stream = JointMultimodalCSVStream(
                self.paths,
                self.tokenizer,
                self.text_tokenizer,
                self.structure_database,
                protein_max_length=self.sequence_length,
                text_max_length=self.text_length,
                seed=self.seed,
                rank=self.rank,
                world_size=self.world_size,
                min_confidence=self.min_structure_confidence,
            )
        else:
            if self.structure_database is None:
                raise RuntimeError("structure database disappeared from a structure batcher")
            stream = StructureCSVStream(
                self.paths,
                self.tokenizer,
                self.structure_database,
                max_length=self.sequence_length,
                seed=self.seed,
                rank=self.rank,
                world_size=self.world_size,
                min_confidence=self.min_structure_confidence,
            )
        stream.set_epoch(self.epoch)
        iterator = iter(stream)
        for _ in range(self.records_consumed):
            try:
                next(iterator)
            except StopIteration as exc:
                raise RuntimeError("saved data position exceeds epoch length") from exc
        return iterator

    def _next_example(self) -> Dict[str, object]:
        if self._iterator is None:
            self._iterator = self._new_iterator()
        while True:
            try:
                example = next(self._iterator)
                self.records_consumed += 1
                return example
            except StopIteration:
                self.epoch += 1
                self.records_consumed = 0
                self._iterator = self._new_iterator()

    def next(self) -> Dict[str, torch.Tensor]:
        examples: List[Dict[str, object]] = []
        if self.micro_batch_token_budget is None:
            while len(examples) < self.batch_size:
                examples.append(self._next_example())
            return self.collator(examples)

        max_examples = int(self.max_micro_batch_size or 1)
        maximum_length = 0
        while len(examples) < max_examples:
            candidate = self._pending
            if candidate is not None:
                self._pending = None
            else:
                candidate = self._next_example()
            candidate_length = len(candidate["input_ids"])
            next_maximum = max(maximum_length, candidate_length)
            padded_tokens = next_maximum * (len(examples) + 1)
            if examples and padded_tokens > self.micro_batch_token_budget:
                self._pending = candidate
                break
            examples.append(candidate)
            maximum_length = next_maximum
        return self.collator(examples)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "rank": self.rank,
            "world_size": self.world_size,
            "epoch": self.epoch,
            "records_consumed": self.records_consumed,
            "pending": self._pending,
            "collator": self.collator.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("task", "sequence_mlm") != self.task:
            raise RuntimeError("checkpoint batcher task does not match the current task")
        if int(state.get("rank", 0)) != self.rank or int(state.get("world_size", 1)) != self.world_size:
            raise RuntimeError("checkpoint batcher rank topology does not match the current run")
        self.epoch = int(state["epoch"])
        self.records_consumed = int(state["records_consumed"])
        pending = state.get("pending")
        if pending is not None and not isinstance(pending, dict):
            raise RuntimeError("checkpoint batcher pending example is invalid")
        self._pending = pending
        self.collator.load_state_dict(state["collator"])
        self._iterator = None


class ReplayBatcher:
    """Deterministically interleave broad sequence replay into joint training."""

    def __init__(
        self,
        joint: RestartableBatcher,
        replay: RestartableBatcher,
        replay_every: int,
    ) -> None:
        self.joint = joint
        self.replay = replay
        self.replay_every = replay_every
        self.calls = 0

    @property
    def epoch(self) -> int:
        return self.joint.epoch

    @property
    def records_consumed(self) -> int:
        return self.joint.records_consumed

    def next(self) -> Dict[str, torch.Tensor]:
        is_replay = (self.calls + 1) % self.replay_every == 0
        self.calls += 1
        batch = self.replay.next() if is_replay else self.joint.next()
        batch["_objective_route"] = torch.tensor(1 if is_replay else 0, dtype=torch.uint8)
        return batch

    def state_dict(self) -> Dict[str, Any]:
        return {
            "kind": "sequence_replay",
            "replay_every": self.replay_every,
            "calls": self.calls,
            "joint": self.joint.state_dict(),
            "replay": self.replay.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("kind") != "sequence_replay":
            raise RuntimeError("checkpoint batcher kind does not match sequence replay")
        if int(state["replay_every"]) != self.replay_every:
            raise RuntimeError("checkpoint replay cadence differs from train config")
        self.calls = int(state["calls"])
        self.joint.load_state_dict(state["joint"])
        self.replay.load_state_dict(state["replay"])


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _learning_rate(
    config: TrainConfig,
    step: int,
    projected_tokens_seen: Optional[int] = None,
) -> float:
    if config.warmup_tokens is not None:
        if projected_tokens_seen is None:
            raise ValueError("token-based schedule requires projected_tokens_seen")
        if projected_tokens_seen < config.warmup_tokens:
            return config.learning_rate * projected_tokens_seen / config.warmup_tokens
        progress = (projected_tokens_seen - config.warmup_tokens) / max(
            1, int(config.max_tokens or 0) - config.warmup_tokens
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return config.min_learning_rate + cosine * (
            config.learning_rate - config.min_learning_rate
        )
    if step < config.warmup_steps:
        return config.learning_rate * float(step + 1) / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return config.min_learning_rate + cosine * (
        config.learning_rate - config.min_learning_rate
    )


def _trainable_state(model: OpenProteinModel) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _save_checkpoint(
    output_dir: Path,
    step: int,
    model: OpenProteinModel,
    optimizer: torch.optim.Optimizer,
    rank_states: List[Dict[str, Any]],
    provenance: Dict[str, Optional[str]],
    initialization_checkpoint: Optional[Dict[str, Any]],
    tokens_seen: int,
    elapsed_training_seconds: float,
) -> Path:
    checkpoint_root = output_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    final = checkpoint_root / f"step-{step:08d}"
    temporary = checkpoint_root / f".step-{step:08d}.tmp-{os.getpid()}"
    temporary.mkdir()
    payload = {
        "schema_version": 4,
        "step": step,
        "provenance": provenance,
        "initialization_checkpoint": initialization_checkpoint,
        "tokens_seen": tokens_seen,
        "elapsed_training_seconds": elapsed_training_seconds,
        "trainable_model": _trainable_state(model),
        "optimizer": optimizer.state_dict(),
        "rank_states": rank_states,
    }
    state_path = temporary / "state.pt"
    torch.save(payload, state_path)
    manifest = {
        "schema_version": 1,
        "step": step,
        "state_bytes": state_path.stat().st_size,
        "state_sha256": sha256_file(state_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(temporary / "manifest.json", manifest)
    os.replace(temporary, final)
    return final


def _load_checkpoint(
    path: Path,
    model: OpenProteinModel,
    optimizer: torch.optim.Optimizer,
    batcher: RestartableBatcher,
    expected_provenance: Dict[str, Optional[str]],
    rank: int,
    world_size: int,
) -> Dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text())
    state_path = path / "state.pt"
    if state_path.stat().st_size != manifest["state_bytes"]:
        raise RuntimeError("checkpoint byte size does not match manifest")
    if sha256_file(state_path) != manifest["state_sha256"]:
        raise RuntimeError("checkpoint SHA-256 does not match manifest")
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    schema_version = int(payload.get("schema_version", 1))
    if schema_version == 1:
        if payload["train_config_sha256"] != expected_provenance["train_config_sha256"]:
            raise RuntimeError("checkpoint was created by a different train config")
    elif schema_version in {2, 3, 4}:
        if payload["provenance"] != expected_provenance:
            raise RuntimeError("checkpoint provenance does not match the current run")
    else:
        raise RuntimeError(f"unsupported checkpoint schema version: {schema_version}")
    parameters = dict(model.named_parameters())
    expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if set(payload["trainable_model"]) != expected:
        raise RuntimeError("checkpoint trainable parameter names do not match model")
    with torch.no_grad():
        for name, value in payload["trainable_model"].items():
            parameters[name].copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))
    optimizer.load_state_dict(payload["optimizer"])
    if schema_version < 3:
        if world_size != 1:
            raise RuntimeError("legacy checkpoints can only resume with world_size=1")
        rank_state = {
            "batcher": payload["batcher"],
            "python_random_state": payload["python_random_state"],
            "torch_random_state": payload["torch_random_state"],
            "cuda_random_state": payload["cuda_random_state"],
        }
    else:
        rank_states = payload["rank_states"]
        if len(rank_states) != world_size:
            raise RuntimeError("checkpoint world size does not match the current run")
        rank_state = rank_states[rank]
    batcher.load_state_dict(rank_state["batcher"])
    random.setstate(rank_state["python_random_state"])
    torch.set_rng_state(rank_state["torch_random_state"])
    if torch.cuda.is_available() and rank_state["cuda_random_state"] is not None:
        if schema_version < 3:
            torch.cuda.set_rng_state_all(rank_state["cuda_random_state"])
        else:
            torch.cuda.set_rng_state(rank_state["cuda_random_state"])
    return {
        "step": int(payload["step"]),
        "tokens_seen": int(payload.get("tokens_seen", 0)),
        "elapsed_training_seconds": float(payload.get("elapsed_training_seconds", 0.0)),
        "checkpoint_schema_version": schema_version,
        "initialization_checkpoint": payload.get("initialization_checkpoint"),
    }


def _load_initialization_checkpoint(
    path: Path,
    model: OpenProteinModel,
    expected_provenance: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    """Load adapter weights across stages without optimizer, RNG, or data state."""

    manifest_path = path / "manifest.json"
    state_path = path / "state.pt"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if state_path.stat().st_size != int(manifest["state_bytes"]):
        raise RuntimeError("initialization checkpoint byte size does not match manifest")
    state_sha256 = sha256_file(state_path)
    if state_sha256 != manifest["state_sha256"]:
        raise RuntimeError("initialization checkpoint SHA-256 does not match manifest")
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    source_provenance = payload.get("provenance")
    if not isinstance(source_provenance, dict):
        raise RuntimeError("initialization checkpoint has no structured provenance")
    required_keys = (
        "model_config_sha256",
        "tokenizer_sha256",
        "model_artifact_manifest_sha256",
    )
    mismatched = {
        key
        for key in required_keys
        if source_provenance.get(key) != expected_provenance.get(key)
    }
    if mismatched:
        raise RuntimeError(
            "initialization checkpoint architecture provenance mismatch: "
            + ", ".join(sorted(mismatched))
        )
    parameters = dict(model.named_parameters())
    expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if set(payload["trainable_model"]) != expected:
        raise RuntimeError("initialization checkpoint trainable parameters do not match model")
    with torch.no_grad():
        for name, value in payload["trainable_model"].items():
            parameters[name].copy_(
                value.to(device=parameters[name].device, dtype=parameters[name].dtype)
            )
    return {
        "path": str(path.resolve()),
        "step": int(payload["step"]),
        "schema_version": int(payload.get("schema_version", 1)),
        "manifest_sha256": sha256_file(manifest_path),
        "state_sha256": state_sha256,
        "source_provenance": source_provenance,
        "optimizer_loaded": False,
        "data_state_loaded": False,
        "rng_state_loaded": False,
    }


def run_training(
    train_config_path: str,
    resume: Optional[str] = None,
    max_steps_override: Optional[int] = None,
    device_name: Optional[str] = None,
    init_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    if resume and init_checkpoint:
        raise ValueError("resume and init_checkpoint are mutually exclusive")
    config_path = Path(train_config_path).resolve()
    train_config = TrainConfig.from_json(str(config_path))
    project_root = config_path.parents[2]

    def resolve(path: str) -> str:
        candidate = Path(path)
        return str(candidate if candidate.is_absolute() else project_root / candidate)

    model_config: ModelConfig = load_model_config(resolve(train_config.model_config))
    tokenizer = EsmcTokenizer.from_tokenizer_json(resolve(train_config.tokenizer_json))
    text_tokenizer: Optional[SentencePieceTextTokenizer] = None
    if train_config.text_tokenizer_model is not None:
        text_tokenizer = SentencePieceTextTokenizer(resolve(train_config.text_tokenizer_model))
        if text_tokenizer.vocabulary_size != model_config.text_vocab_size:
            raise ValueError("text tokenizer vocabulary does not match the model config")
    paths = sorted(glob.glob(resolve(train_config.train_glob)))
    if not paths:
        raise ValueError(f"train_glob matched no files: {train_config.train_glob}")
    replay_paths = (
        sorted(glob.glob(resolve(train_config.replay_train_glob)))
        if train_config.replay_train_glob is not None
        else []
    )
    if train_config.replay_train_glob is not None and not replay_paths:
        raise ValueError(
            f"replay_train_glob matched no files: {train_config.replay_train_glob}"
        )
    if train_config.task == "sequence_structure_text" and (
        model_config.structure_feature_dimension != STRUCTURE_FEATURE_DIMENSION
    ):
        raise ValueError("joint task structure feature dimension differs from data pipeline")
    output_dir = Path(resolve(train_config.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    config_digest = sha256_file(config_path)
    source_digest = source_tree_sha256(project_root)
    training_code_digest = source_tree_sha256(Path(__file__).resolve().parent)
    model_config_digest = sha256_file(Path(resolve(train_config.model_config)))
    tokenizer_digest = sha256_file(Path(resolve(train_config.tokenizer_json)))
    text_tokenizer_digest = (
        sha256_file(Path(resolve(train_config.text_tokenizer_model)))
        if train_config.text_tokenizer_model is not None
        else None
    )

    def optional_manifest_digest(path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        manifest_path = Path(resolve(path))
        if not manifest_path.is_file():
            raise FileNotFoundError(f"required provenance manifest is missing: {manifest_path}")
        return sha256_file(manifest_path)

    model_artifact_digest = optional_manifest_digest(train_config.model_artifact_manifest)
    provisional_rank = int(os.environ.get("RANK", "0"))
    if train_config.model_artifact_manifest is not None and provisional_rank == 0:
        verify_artifact_manifest(
            resolve(train_config.model_artifact_manifest),
            expected_root=model_config.protein_backbone.model_path,
        )

    provenance = {
        "train_config_sha256": config_digest,
        "training_code_sha256": training_code_digest,
        "model_config_sha256": model_config_digest,
        "tokenizer_sha256": tokenizer_digest,
        "text_tokenizer_sha256": text_tokenizer_digest,
        "model_artifact_manifest_sha256": model_artifact_digest,
        "data_manifest_sha256": optional_manifest_digest(train_config.data_manifest),
        "structure_manifest_sha256": optional_manifest_digest(train_config.structure_manifest),
        "replay_data_manifest_sha256": optional_manifest_digest(
            train_config.replay_data_manifest
        ),
    }
    structure_database = (
        resolve(train_config.structure_database)
        if train_config.structure_database is not None
        else None
    )
    if structure_database is not None and not Path(structure_database).is_file():
        raise FileNotFoundError(f"structure database is missing: {structure_database}")
    if structure_database is not None:
        if train_config.structure_manifest is None:
            raise RuntimeError("structure training requires a structure manifest")
        verify_structure_database(
            structure_database,
            resolve(train_config.structure_manifest),
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if not 0 <= rank < world_size:
        raise ValueError("RANK must be in [0, WORLD_SIZE)")
    if distributed:
        if device_name is not None or not torch.cuda.is_available():
            raise ValueError("distributed training requires torchrun-managed CUDA devices")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl")
    else:
        device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if train_config.precision == "bf16" else torch.float32
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_config.seed)
        torch.cuda.reset_peak_memory_stats(device)

    model = OpenProteinModel.from_config(model_config, device=device, dtype=dtype)
    model.to(device=device, dtype=dtype)
    model.train()
    initialization_receipt = (
        _load_initialization_checkpoint(Path(init_checkpoint), model, provenance)
        if init_checkpoint
        else None
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        fused=device.type == "cuda",
    )
    joint_batcher = RestartableBatcher(
        paths,
        tokenizer,
        train_config.sequence_length,
        train_config.micro_batch_size,
        train_config.seed,
        train_config.mlm_probability,
        task=train_config.task,
        text_tokenizer=text_tokenizer,
        text_length=train_config.text_length,
        structure_database=structure_database,
        geometry_pairs=train_config.geometry_pairs,
        distance_bins=model_config.distance_bins,
        min_structure_confidence=train_config.min_structure_confidence,
        structure_mask_probability=train_config.structure_mask_probability,
        structure_dropout_probability=train_config.structure_dropout_probability,
        rank=rank,
        world_size=world_size,
        micro_batch_token_budget=train_config.micro_batch_token_budget,
        max_micro_batch_size=train_config.max_micro_batch_size,
    )
    batcher: Any = joint_batcher
    if replay_paths:
        replay_batcher = RestartableBatcher(
            replay_paths,
            tokenizer,
            train_config.sequence_length,
            train_config.micro_batch_size,
            train_config.seed + 8_000_021,
            train_config.mlm_probability,
            task="sequence_mlm",
            rank=rank,
            world_size=world_size,
        )
        batcher = ReplayBatcher(joint_batcher, replay_batcher, train_config.replay_every)

    step = 0
    tokens_seen = 0
    elapsed_offset = 0.0
    resumed_checkpoint_schema: Optional[int] = None
    if resume:
        resumed_state = _load_checkpoint(
            Path(resume),
            model,
            optimizer,
            batcher,
            expected_provenance=provenance,
            rank=rank,
            world_size=world_size,
        )
        step = int(resumed_state["step"])
        tokens_seen = int(resumed_state["tokens_seen"])
        elapsed_offset = float(resumed_state["elapsed_training_seconds"])
        resumed_checkpoint_schema = int(resumed_state["checkpoint_schema_version"])
        initialization_receipt = resumed_state["initialization_checkpoint"]
    target_steps = max_steps_override or train_config.max_steps
    if target_steps < step:
        raise ValueError("target max steps precedes resumed step")

    training_model: torch.nn.Module = model
    if distributed:
        training_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    metrics_path = output_dir / "metrics.jsonl"
    run_tokens = 0
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    while step < target_steps and (
        train_config.max_tokens is None or tokens_seen < train_config.max_tokens
    ):
        accumulated_loss = 0.0
        component_losses: Dict[str, float] = {}
        route_microbatches = {"joint": 0, "replay": 0}
        step_tokens = 0
        step_started = time.perf_counter()
        for accumulation_index in range(train_config.gradient_accumulation_steps):
            batch = {key: value.to(device) for key, value in batcher.next().items()}
            objective_route = int(batch.pop("_objective_route", torch.tensor(0)).item())
            is_replay = objective_route == 1
            route_microbatches["replay" if is_replay else "joint"] += 1
            active_loss_weights = (
                {"protein_mlm": train_config.replay_loss_weight}
                if is_replay
                else train_config.loss_weights
            )
            synchronization = (
                training_model.no_sync()
                if distributed
                and accumulation_index < train_config.gradient_accumulation_steps - 1
                else nullcontext()
            )
            with synchronization:
                output = training_model(
                    **batch,
                    attention_policy=train_config.attention_policy,
                    use_bridge=True,
                    loss_weights=active_loss_weights,
                    detach_geometry_backbone=train_config.detach_geometry_backbone,
                )
                loss = output["loss"]
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss")
                (loss / train_config.gradient_accumulation_steps).backward()
            accumulated_loss += float(loss.detach().cpu())
            for name, value in output["losses"].items():
                metric_name = f"replay_{name}" if is_replay else name
                component_losses[metric_name] = component_losses.get(metric_name, 0.0) + float(
                    value.detach().cpu()
                )
            step_tokens += int(batch["protein_attention_mask"].sum().item())
            if "text_attention_mask" in batch:
                step_tokens += int(batch["text_attention_mask"].sum().item())

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, train_config.max_grad_norm)
        learning_rate = _learning_rate(
            train_config,
            step,
            projected_tokens_seen=tokens_seen + step_tokens,
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
                [accumulated_loss, float(step_tokens), step_seconds],
                dtype=torch.float64,
                device=device,
            )
            torch.distributed.all_reduce(aggregate[:2], op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(aggregate[2:], op=torch.distributed.ReduceOp.MAX)
            accumulated_loss = float(aggregate[0])
            step_tokens = int(aggregate[1])
            step_seconds = float(aggregate[2])
            for name in sorted(component_losses):
                value = torch.tensor(component_losses[name], dtype=torch.float64, device=device)
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
                component_losses[name] = float(value)
        run_tokens += step_tokens
        tokens_seen += step_tokens
        peak_memory_mib = (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        )
        if distributed and peak_memory_mib is not None:
            peak_tensor = torch.tensor(peak_memory_mib, dtype=torch.float64, device=device)
            torch.distributed.all_reduce(peak_tensor, op=torch.distributed.ReduceOp.MAX)
            peak_memory_mib = float(peak_tensor)
        metric = {
            "step": step,
            "loss": accumulated_loss
            / (train_config.gradient_accumulation_steps * world_size),
            "losses": {
                name: value / (train_config.gradient_accumulation_steps * world_size)
                for name, value in sorted(component_losses.items())
            },
            "learning_rate": learning_rate,
            "grad_norm": float(grad_norm.detach().cpu()),
            "tokens": step_tokens,
            "tokens_seen": tokens_seen,
            "tokens_per_second": step_tokens / step_seconds,
            "step_seconds": step_seconds,
            "epoch": batcher.epoch,
            "records_consumed": batcher.records_consumed,
            "records_consumed_scope": "rank0" if distributed else "global",
            "peak_memory_mib": peak_memory_mib,
            "route_microbatches_per_rank": route_microbatches,
        }
        if rank == 0 and step % train_config.log_every == 0:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metric, sort_keys=True) + "\n")
            print(json.dumps(metric, sort_keys=True), flush=True)
        token_budget_reached = (
            train_config.max_tokens is not None
            and tokens_seen >= train_config.max_tokens
        )
        if (
            step % train_config.checkpoint_every == 0
            or step == target_steps
            or token_budget_reached
        ):
            local_state = {
                "batcher": batcher.state_dict(),
                "python_random_state": random.getstate(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_state": (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                ),
            }
            rank_states: List[Dict[str, Any]] = [local_state]
            if distributed:
                gathered: List[Optional[Dict[str, Any]]] = [None] * world_size
                torch.distributed.all_gather_object(gathered, local_state)
                rank_states = [state for state in gathered if state is not None]
            if rank == 0:
                checkpoint = _save_checkpoint(
                    output_dir,
                    step,
                    model,
                    optimizer,
                    rank_states,
                    provenance,
                    initialization_receipt,
                    tokens_seen,
                    elapsed_offset + (time.perf_counter() - started),
                )
                print(f"checkpoint={checkpoint}", flush=True)
            if distributed:
                torch.distributed.barrier()

    elapsed = time.perf_counter() - started
    receipt_peak_memory = (
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    )
    if distributed:
        final_timing = torch.tensor(
            [elapsed, float(receipt_peak_memory or 0.0)], dtype=torch.float64, device=device
        )
        torch.distributed.all_reduce(final_timing, op=torch.distributed.ReduceOp.MAX)
        elapsed = float(final_timing[0])
        receipt_peak_memory = float(final_timing[1])
    token_budget_reached = (
        train_config.max_tokens is not None
        and tokens_seen >= train_config.max_tokens
    )
    run_completed = train_config.max_tokens is None or token_budget_reached
    receipt = {
        "schema_version": 3,
        "status": "passed" if run_completed else "failed",
        "evidence_level": (
            "PRETRAIN"
            if token_budget_reached
            else ("PILOT" if train_config.max_tokens is None else "INCOMPLETE")
        ),
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "name": train_config.name,
        "task": train_config.task,
        "attention_policy": train_config.attention_policy,
        "detach_geometry_backbone": train_config.detach_geometry_backbone,
        "step": step,
        "tokens": tokens_seen,
        "tokens_this_run": run_tokens,
        "target_tokens": train_config.max_tokens,
        "token_budget_reached": token_budget_reached,
        "step_safety_limit": target_steps,
        "stopped_by": "token_budget" if token_budget_reached else "step_limit",
        "micro_batch_token_budget_per_rank": train_config.micro_batch_token_budget,
        "max_micro_batch_size_per_rank": train_config.max_micro_batch_size,
        "learning_rate_schedule": (
            "token_cosine" if train_config.warmup_tokens is not None else "step_cosine"
        ),
        "warmup_tokens": train_config.warmup_tokens,
        "elapsed_seconds": elapsed_offset + elapsed,
        "elapsed_seconds_this_run": elapsed,
        "average_tokens_per_second_this_run": run_tokens / elapsed if elapsed else None,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "precision": train_config.precision,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_memory_mib": receipt_peak_memory,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "resumed_checkpoint_schema": resumed_checkpoint_schema,
        "initialization_checkpoint": initialization_receipt,
        "source_tree_sha256": source_digest,
        "data_files": paths,
        "replay_data_files": replay_paths,
        "replay_every": train_config.replay_every if replay_paths else None,
    }
    if rank == 0:
        _atomic_json(output_dir / f"receipt-step-{step:08d}.json", receipt)
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if not run_completed:
        raise RuntimeError(
            f"token budget not reached: {tokens_seen} < {train_config.max_tokens} "
            f"at safety step limit {target_steps}"
        )
    return receipt
