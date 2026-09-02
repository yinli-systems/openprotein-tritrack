"""Validation-only sequence MLM comparison against the frozen ESMC path."""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.nn import functional as F

from openprotein.artifacts import verify_artifact_manifest
from openprotein.config import load_model_config
from openprotein.data.streaming import (
    CSVProteinStream,
    MultimodalCSVStream,
    ProteinMLMCollator,
    SequenceStructureCollator,
    SequenceTextCollator,
    StructureCSVStream,
)
from openprotein.evaluation_subset import (
    load_record_id_allowlist,
    verify_selected_record_ids,
)
from openprotein.data.structure import verify_structure_database
from openprotein.data.text_tokenizer import SentencePieceTextTokenizer
from openprotein.data.tokenizer import EsmcTokenizer
from openprotein.model import OpenProteinModel
from openprotein.provenance import sha256_file, source_tree_sha256


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_trainable_checkpoint(
    model: OpenProteinModel,
    checkpoint_dir: str,
    expected_provenance: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    checkpoint = Path(checkpoint_dir).resolve()
    manifest_path = checkpoint / "manifest.json"
    state_path = checkpoint / "state.pt"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if state_path.stat().st_size != int(manifest["state_bytes"]):
        raise RuntimeError("validation checkpoint byte size does not match its manifest")
    if sha256_file(state_path) != manifest["state_sha256"]:
        raise RuntimeError("validation checkpoint SHA-256 does not match its manifest")
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    provenance = payload.get("provenance")
    if expected_provenance is not None:
        if not isinstance(provenance, dict):
            raise RuntimeError("validation checkpoint has no structured provenance")
        mismatched = {
            key
            for key, value in expected_provenance.items()
            if provenance.get(key) != value
        }
        if mismatched:
            raise RuntimeError(
                "validation checkpoint provenance mismatch: " + ", ".join(sorted(mismatched))
            )
    parameters = dict(model.named_parameters())
    expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if set(payload["trainable_model"]) != expected:
        raise RuntimeError("validation checkpoint parameter names do not match the model")
    with torch.no_grad():
        for name, value in payload["trainable_model"].items():
            parameters[name].copy_(value.to(parameters[name]))
    return {
        "step": int(payload["step"]),
        "schema_version": int(payload.get("schema_version", 1)),
        "manifest_sha256": sha256_file(manifest_path),
        "state_sha256": manifest["state_sha256"],
        "training_provenance": provenance,
    }


@torch.no_grad()
def evaluate_sequence_validation(
    model_config_path: str,
    checkpoint_dir: str,
    tokenizer_json: str,
    model_artifact_manifest_path: str,
    validation_glob: str,
    data_manifest_path: str,
    output_path: str,
    max_batches: int,
    sequence_length: int = 256,
    mlm_probability: float = 0.15,
    seed: int = 2701,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare bridge and exact backbone logits on the same validation masks."""

    if max_batches <= 0:
        raise ValueError("max_batches must be positive")
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite validation receipt: {output}")
    manifest_path = Path(data_manifest_path).resolve()
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_root = Path(data_manifest["dataset_root"])
    expected_files = {
        str((dataset_root / item["path"]).resolve()): item
        for item in data_manifest["files"]
        if item["split"] == "validation"
    }
    paths = sorted(str(Path(path).resolve()) for path in glob.glob(validation_glob))
    if not paths or any(path not in expected_files for path in paths):
        raise ValueError("validation_glob is not a subset of the manifest validation split")
    for path in paths:
        if Path(path).stat().st_size != int(expected_files[path]["bytes"]):
            raise RuntimeError("validation shard byte size differs from its manifest")

    config = load_model_config(model_config_path)
    tokenizer = EsmcTokenizer.from_tokenizer_json(tokenizer_json)
    artifact_receipt = verify_artifact_manifest(
        model_artifact_manifest_path,
        expected_root=config.protein_backbone.model_path,
    )
    expected_provenance = {
        "model_config_sha256": sha256_file(Path(model_config_path)),
        "tokenizer_sha256": sha256_file(Path(tokenizer_json)),
        "model_artifact_manifest_sha256": sha256_file(Path(model_artifact_manifest_path)),
    }
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = OpenProteinModel.from_config(config, device=device, dtype=dtype)
    model.to(device=device, dtype=dtype).eval()
    checkpoint = load_trainable_checkpoint(model, checkpoint_dir, expected_provenance)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    stream = CSVProteinStream(paths, tokenizer, sequence_length, seed, shuffle_files=False)
    iterator = iter(stream)
    collator = ProteinMLMCollator(tokenizer, seed, mlm_probability)
    base_nll = 0.0
    bridge_nll = 0.0
    masked_tokens = 0
    batches = 0
    for _ in range(max_batches):
        try:
            example = next(iterator)
        except StopIteration:
            break
        batch = {key: value.to(device) for key, value in collator([example]).items()}
        labels = batch["protein_labels"]
        count = int(labels.ne(-100).sum())
        bridge = model(**batch, use_bridge=True, attention_policy="bidirectional")
        base_nll += float(
            F.cross_entropy(
                bridge["base_protein_logits"].reshape(
                    -1, bridge["base_protein_logits"].shape[-1]
                ),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        )
        bridge_nll += float(
            F.cross_entropy(
                bridge["protein_logits"].reshape(-1, bridge["protein_logits"].shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        )
        masked_tokens += count
        batches += 1
    if not masked_tokens:
        raise RuntimeError("validation produced no masked tokens")
    base_loss = base_nll / masked_tokens
    bridge_loss = bridge_nll / masked_tokens
    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": "VALIDATION",
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "model_config_sha256": sha256_file(Path(model_config_path)),
        "tokenizer_sha256": sha256_file(Path(tokenizer_json)),
        "model_artifact": artifact_receipt,
        "data_manifest_sha256": sha256_file(manifest_path),
        "validation_files": paths,
        "batches": batches,
        "masked_tokens": masked_tokens,
        "base_backbone_kind": config.protein_backbone.kind,
        "base_backbone_loss": base_loss,
        "base_backbone_perplexity": math.exp(base_loss),
        "base_esmc_loss": base_loss,
        "base_esmc_perplexity": math.exp(base_loss),
        "bridge_loss": bridge_loss,
        "bridge_perplexity": math.exp(bridge_loss),
        "bridge_minus_base_backbone_loss": bridge_loss - base_loss,
        "bridge_minus_base_loss": bridge_loss - base_loss,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "evaluation_code_sha256": source_tree_sha256(Path(__file__).resolve().parent),
        "evaluation_settings": {
            "max_batches": max_batches,
            "sequence_length": sequence_length,
            "mlm_probability": mlm_probability,
            "seed": seed,
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(output, receipt)
    return receipt


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> List[float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive sample count")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _retrieval_ranks(
    scores: torch.Tensor,
    match_group_ids: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    order = scores.argsort(dim=1, descending=True)
    if match_group_ids is None:
        targets = torch.arange(scores.shape[0], device=scores.device)
        return order.eq(targets[:, None]).nonzero(as_tuple=False)[:, 1].add(1).float()
    if len(match_group_ids) != scores.shape[0]:
        raise ValueError("retrieval group IDs must match the embedding count")
    group_members: Dict[str, List[int]] = {}
    for index, group_id in enumerate(match_group_ids):
        group_members.setdefault(group_id, []).append(index)
    inverse_order = torch.empty_like(order)
    positions = torch.arange(order.shape[1], device=order.device).expand_as(order)
    inverse_order.scatter_(1, order, positions)
    ranks = []
    for query, group_id in enumerate(match_group_ids):
        candidates = torch.tensor(group_members[group_id], device=scores.device)
        ranks.append(inverse_order[query].index_select(0, candidates).min().add(1))
    return torch.stack(ranks).float()


def _rank_metrics(prefix: str, ranks: torch.Tensor) -> Dict[str, Any]:
    total = int(ranks.numel())
    metrics: Dict[str, Any] = {
        f"{prefix}_mrr": float(ranks.reciprocal().mean()),
        f"{prefix}_median_rank": float(ranks.median()),
    }
    for cutoff in (1, 5, 10):
        successes = int(ranks.le(cutoff).sum())
        metrics[f"{prefix}_recall_at_{cutoff}"] = successes / total
        metrics[f"{prefix}_recall_at_{cutoff}_wilson_95_percent_interval"] = (
            _wilson_interval(successes, total)
        )
    return metrics


def retrieval_metrics(
    protein_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    match_group_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Paired retrieval metrics for independently encoded protein and text views."""

    if protein_embeddings.ndim != 2 or protein_embeddings.shape != text_embeddings.shape:
        raise ValueError("retrieval embeddings must have equal [examples, dimension] shapes")
    if protein_embeddings.shape[0] < 2:
        raise ValueError("retrieval metrics require at least two pairs")
    similarity = protein_embeddings.float() @ text_embeddings.float().transpose(0, 1)
    protein_to_text = _retrieval_ranks(similarity, match_group_ids)
    text_to_protein = _retrieval_ranks(similarity.transpose(0, 1), match_group_ids)
    candidates = int(similarity.shape[0])
    return {
        "candidate_count": candidates,
        "matching_policy": "exact_pair" if match_group_ids is None else "normalized_text_group",
        "random_reference_recall_at_1": 1.0 / candidates,
        "random_reference_recall_at_5": min(5.0 / candidates, 1.0),
        "random_reference_recall_at_10": min(10.0 / candidates, 1.0),
        **_rank_metrics("protein_to_text", protein_to_text),
        **_rank_metrics("text_to_protein", text_to_protein),
    }


def _difference_summary(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        raise ValueError("difference summary requires at least one value")
    mean = sum(values) / len(values)
    variance = (
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        if len(values) > 1
        else 0.0
    )
    standard_error = math.sqrt(variance / len(values))
    return {
        "batch_mean_paired_minus_shuffled": mean,
        "batch_mean_standard_error": standard_error,
        "batch_mean_normal_95_percent_interval": [
            mean - 1.96 * standard_error,
            mean + 1.96 * standard_error,
        ],
        "batches": len(values),
    }


_GEOMETRY_SEPARATION_BOUNDARIES = (6, 12, 24, 48, 96)


def _geometry_prior_diagnostics(
    calibration_counts: torch.Tensor,
    evaluation_buckets: torch.Tensor,
    evaluation_labels: torch.Tensor,
    model_evaluation_nll: float,
    model_evaluation_correct: int,
) -> Dict[str, Any]:
    if calibration_counts.ndim != 2 or calibration_counts.numel() == 0:
        raise ValueError("geometry calibration counts must have [buckets, classes] shape")
    if evaluation_labels.numel() == 0 or evaluation_labels.shape != evaluation_buckets.shape:
        raise ValueError("geometry prior evaluation requires aligned held-out targets")
    smoothed = calibration_counts.double().add(1.0)
    bucket_probabilities = smoothed / smoothed.sum(dim=1, keepdim=True)
    selected_probabilities = bucket_probabilities[
        evaluation_buckets.long(), evaluation_labels.long()
    ]
    separation_predictions = bucket_probabilities.argmax(dim=1)[evaluation_buckets.long()]

    global_counts = calibration_counts.sum(dim=0).double().add(1.0)
    global_probabilities = global_counts / global_counts.sum()
    global_predictions = global_probabilities.argmax().expand_as(evaluation_labels)
    total = int(evaluation_labels.numel())
    return {
        "policy": (
            "record-hash parity calibration/evaluation split; Laplace-smoothed priors are fit "
            "only on calibration records"
        ),
        "sequence_separation_boundaries": list(_GEOMETRY_SEPARATION_BOUNDARIES),
        "calibration_targets": int(calibration_counts.sum()),
        "evaluation_targets": total,
        "model_clean_sequence_loss_on_evaluation_half": model_evaluation_nll / total,
        "model_clean_sequence_accuracy_on_evaluation_half": model_evaluation_correct / total,
        "global_marginal_prior_loss": float(
            -global_probabilities[evaluation_labels.long()].log().mean()
        ),
        "global_marginal_prior_accuracy": float(
            global_predictions.eq(evaluation_labels).float().mean()
        ),
        "sequence_separation_prior_loss": float(-selected_probabilities.log().mean()),
        "sequence_separation_prior_accuracy": float(
            separation_predictions.eq(evaluation_labels).float().mean()
        ),
        "distance_class_calibration_counts": calibration_counts.tolist(),
    }


@torch.no_grad()
def evaluate_text_validation(
    model_config_path: str,
    checkpoint_dir: str,
    protein_tokenizer_json: str,
    text_tokenizer_model: str,
    model_artifact_manifest_path: str,
    validation_glob: str,
    data_manifest_path: str,
    output_path: str,
    max_batches: int,
    batch_size: int = 8,
    sequence_length: int = 128,
    text_length: int = 128,
    mlm_probability: float = 0.15,
    seed: int = 3701,
    device_name: Optional[str] = None,
    record_id_allowlist_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Validation-only language losses and leakage-safe paired retrieval."""

    if max_batches <= 0 or batch_size < 2:
        raise ValueError("text validation requires positive max_batches and batch_size >= 2")
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite validation receipt: {output}")
    manifest_path = Path(data_manifest_path).resolve()
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_root = Path(data_manifest["dataset_root"])
    expected_files = {
        str((dataset_root / item["path"]).resolve()): item
        for item in data_manifest["files"]
        if item["split"] == "validation"
    }
    paths = sorted(str(Path(path).resolve()) for path in glob.glob(validation_glob))
    if not paths or any(path not in expected_files for path in paths):
        raise ValueError("validation_glob is not a subset of the manifest validation split")
    for path in paths:
        if Path(path).stat().st_size != int(expected_files[path]["bytes"]):
            raise RuntimeError("validation shard byte size differs from its manifest")

    config = load_model_config(model_config_path)
    protein_tokenizer = EsmcTokenizer.from_tokenizer_json(protein_tokenizer_json)
    text_tokenizer = SentencePieceTextTokenizer(text_tokenizer_model)
    if text_tokenizer.vocabulary_size != config.text_vocab_size:
        raise ValueError("text tokenizer vocabulary does not match model config")
    artifact_receipt = verify_artifact_manifest(
        model_artifact_manifest_path,
        expected_root=config.protein_backbone.model_path,
    )
    expected_provenance = {
        "model_config_sha256": sha256_file(Path(model_config_path)),
        "tokenizer_sha256": sha256_file(Path(protein_tokenizer_json)),
        "text_tokenizer_sha256": sha256_file(Path(text_tokenizer_model)),
        "model_artifact_manifest_sha256": sha256_file(Path(model_artifact_manifest_path)),
    }
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = OpenProteinModel.from_config(config, device=device, dtype=dtype)
    model.to(device=device, dtype=dtype).eval()
    checkpoint = load_trainable_checkpoint(model, checkpoint_dir, expected_provenance)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    expected_record_ids = None
    record_id_set = None
    allowlist_receipt = None
    if record_id_allowlist_path is not None:
        expected_record_ids, record_id_set, allowlist_receipt = load_record_id_allowlist(
            record_id_allowlist_path
        )

    stream = MultimodalCSVStream(
        paths,
        protein_tokenizer,
        text_tokenizer,
        sequence_length,
        text_length,
        seed,
        shuffle_files=False,
        include_record_ids=record_id_set,
    )
    iterator = iter(stream)
    collator = SequenceTextCollator(
        protein_tokenizer,
        text_tokenizer,
        seed,
        mlm_probability,
    )
    protein_nll = 0.0
    text_nll = 0.0
    shuffled_protein_text_nll = 0.0
    clean_text_nll = 0.0
    shuffled_clean_protein_text_nll = 0.0
    conditioning_batch_differences = []
    clean_conditioning_batch_differences = []
    protein_tokens = 0
    text_tokens = 0
    batches = 0
    masked_protein_embeddings = []
    masked_text_embeddings = []
    clean_protein_embeddings = []
    clean_text_embeddings = []
    text_group_ids: List[str] = []
    sequence_lengths: List[int] = []
    selected_record_ids: List[str] = []
    for _ in range(max_batches):
        examples = []
        for _ in range(batch_size):
            try:
                examples.append(next(iterator))
            except StopIteration:
                break
        if not examples:
            break
        batch = {key: value.to(device) for key, value in collator(examples).items()}
        result = model(**batch, use_bridge=True, attention_policy="prefix_lm")
        required_losses = {"protein_mlm", "text_lm"}
        if len(examples) > 1:
            required_losses.add("contrastive")
        if set(result["losses"]) != required_losses:
            raise RuntimeError("text validation did not produce every declared objective")
        protein_count = int(batch["protein_labels"].ne(-100).sum())
        text_count = int(batch["text_labels"][:, 1:].ne(-100).sum())
        rotation = torch.roll(torch.arange(len(examples), device=device), shifts=1)
        shuffled_batch = {
            key: value
            for key, value in batch.items()
            if key != "protein_labels"
        }
        for key in ("protein_input_ids", "protein_attention_mask"):
            shuffled_batch[key] = shuffled_batch[key].index_select(0, rotation)
        shuffled_result = model(
            **shuffled_batch,
            use_bridge=True,
            attention_policy="prefix_lm",
        )
        clean_protein_input_ids = batch["protein_input_ids"].clone()
        masked_positions = batch["protein_labels"].ne(-100)
        clean_protein_input_ids[masked_positions] = batch["protein_labels"][masked_positions]
        clean_batch = {
            key: value
            for key, value in batch.items()
            if key != "protein_labels"
        }
        clean_batch["protein_input_ids"] = clean_protein_input_ids
        clean_result = model(
            **clean_batch,
            use_bridge=True,
            attention_policy="prefix_lm",
        )
        shuffled_clean_batch = dict(clean_batch)
        for key in ("protein_input_ids", "protein_attention_mask"):
            shuffled_clean_batch[key] = shuffled_clean_batch[key].index_select(0, rotation)
        shuffled_clean_result = model(
            **shuffled_clean_batch,
            use_bridge=True,
            attention_policy="prefix_lm",
        )
        protein_nll += float(result["losses"]["protein_mlm"]) * protein_count
        text_nll += float(result["losses"]["text_lm"]) * text_count
        shuffled_protein_text_nll += (
            float(shuffled_result["losses"]["text_lm"]) * text_count
        )
        clean_text_nll += float(clean_result["losses"]["text_lm"]) * text_count
        shuffled_clean_protein_text_nll += (
            float(shuffled_clean_result["losses"]["text_lm"]) * text_count
        )
        conditioning_batch_differences.append(
            float(result["losses"]["text_lm"])
            - float(shuffled_result["losses"]["text_lm"])
        )
        clean_conditioning_batch_differences.append(
            float(clean_result["losses"]["text_lm"])
            - float(shuffled_clean_result["losses"]["text_lm"])
        )
        protein_tokens += protein_count
        text_tokens += text_count
        masked_protein_embeddings.append(result["protein_embedding"].float().cpu())
        masked_text_embeddings.append(result["text_embedding"].float().cpu())
        clean_protein_embeddings.append(clean_result["protein_embedding"].float().cpu())
        clean_text_embeddings.append(clean_result["text_embedding"].float().cpu())
        text_group_ids.extend(str(example["text_group"]) for example in examples)
        sequence_lengths.extend(int(example["sequence_length"]) for example in examples)
        selected_record_ids.extend(str(example["record_id"]) for example in examples)
        batches += 1
    if not batches or not protein_tokens or not text_tokens:
        raise RuntimeError("text validation produced no complete evaluation batches")
    selected_subset = None
    if expected_record_ids is not None:
        selected_subset = verify_selected_record_ids(selected_record_ids, expected_record_ids)
    clean_protein_matrix = torch.cat(clean_protein_embeddings, dim=0)
    clean_text_matrix = torch.cat(clean_text_embeddings, dim=0)
    retrieval = retrieval_metrics(clean_protein_matrix, clean_text_matrix)
    group_aware_retrieval = retrieval_metrics(
        clean_protein_matrix,
        clean_text_matrix,
        match_group_ids=text_group_ids,
    )
    masked_retrieval_control = retrieval_metrics(
        torch.cat(masked_protein_embeddings, dim=0),
        torch.cat(masked_text_embeddings, dim=0),
    )
    protein_loss = protein_nll / protein_tokens
    text_loss = text_nll / text_tokens
    shuffled_protein_text_loss = shuffled_protein_text_nll / text_tokens
    clean_text_loss = clean_text_nll / text_tokens
    shuffled_clean_protein_text_loss = shuffled_clean_protein_text_nll / text_tokens
    text_group_counts: Dict[str, int] = {}
    for group_id in text_group_ids:
        text_group_counts[group_id] = text_group_counts.get(group_id, 0) + 1
    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": (
            "VALIDATION_DECONTAMINATED_SUBSET"
            if allowlist_receipt is not None
            else "VALIDATION"
        ),
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "model_config_sha256": sha256_file(Path(model_config_path)),
        "protein_tokenizer_sha256": sha256_file(Path(protein_tokenizer_json)),
        "text_tokenizer_sha256": sha256_file(Path(text_tokenizer_model)),
        "model_artifact": artifact_receipt,
        "data_manifest_sha256": sha256_file(manifest_path),
        "validation_files": paths,
        "batches": batches,
        "pairs": int(clean_protein_matrix.shape[0]),
        "record_id_subset": (
            {**allowlist_receipt, **selected_subset}
            if allowlist_receipt is not None and selected_subset is not None
            else None
        ),
        "protein_masked_tokens": protein_tokens,
        "text_target_tokens": text_tokens,
        "protein_mlm_loss": protein_loss,
        "protein_mlm_perplexity": math.exp(protein_loss),
        "text_lm_loss": text_loss,
        "text_lm_perplexity": math.exp(text_loss),
        "conditioning_control": {
            "policy": (
                "deterministic within-batch masked-protein rotation with text targets fixed"
            ),
            "shuffled_protein_text_lm_loss": shuffled_protein_text_loss,
            "paired_minus_shuffled_text_lm_loss": text_loss - shuffled_protein_text_loss,
            "paired_better": text_loss < shuffled_protein_text_loss,
            **_difference_summary(conditioning_batch_differences),
        },
        "clean_sequence_text_lm_loss": clean_text_loss,
        "clean_sequence_text_lm_perplexity": math.exp(clean_text_loss),
        "clean_sequence_conditioning_control": {
            "policy": "deterministic within-batch clean-protein rotation with text targets fixed",
            "shuffled_protein_text_lm_loss": shuffled_clean_protein_text_loss,
            "paired_minus_shuffled_text_lm_loss": (
                clean_text_loss - shuffled_clean_protein_text_loss
            ),
            "paired_better": clean_text_loss < shuffled_clean_protein_text_loss,
            **_difference_summary(clean_conditioning_batch_differences),
        },
        "retrieval": retrieval,
        "retrieval_group_aware": group_aware_retrieval,
        "retrieval_masked_protein_control": masked_retrieval_control,
        "retrieval_encoding_policy": (
            "primary retrieval restores all MLM-corrupted residues before encoding; protein prefix "
            "cannot attend text; text retrieval embedding is isolated from protein"
        ),
        "retrieval_duplicate_text_diagnostics": {
            "normalized_text_groups": len(text_group_counts),
            "duplicate_groups": sum(count > 1 for count in text_group_counts.values()),
            "pairs_in_duplicate_groups": sum(
                count for count in text_group_counts.values() if count > 1
            ),
            "largest_group": max(text_group_counts.values()),
        },
        "sequence_length_diagnostics": {
            "minimum": min(sequence_lengths),
            "maximum": max(sequence_lengths),
            "mean": sum(sequence_lengths) / len(sequence_lengths),
            "cropped_above_configured_length": sum(
                length > sequence_length - 2 for length in sequence_lengths
            ),
        },
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "evaluation_code_sha256": source_tree_sha256(Path(__file__).resolve().parent),
        "evaluation_settings": {
            "max_batches": max_batches,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "text_length": text_length,
            "mlm_probability": mlm_probability,
            "seed": seed,
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(output, receipt)
    return receipt


@torch.no_grad()
def evaluate_structure_validation(
    model_config_path: str,
    checkpoint_dir: str,
    protein_tokenizer_json: str,
    model_artifact_manifest_path: str,
    validation_glob: str,
    data_manifest_path: str,
    structure_database_path: str,
    structure_manifest_path: str,
    output_path: str,
    max_batches: int,
    batch_size: int = 1,
    sequence_length: int = 256,
    geometry_pairs: int = 128,
    mlm_probability: float = 0.15,
    min_confidence: float = 70.0,
    seed: int = 4701,
    device_name: Optional[str] = None,
    record_id_allowlist_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Validation-only MLM and sampled geometry metrics on predicted structures."""

    if max_batches <= 0 or batch_size <= 0:
        raise ValueError("structure validation batch counts must be positive")
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite validation receipt: {output}")
    manifest_path = Path(data_manifest_path).resolve()
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_root = Path(data_manifest["dataset_root"])
    expected_files = {
        str((dataset_root / item["path"]).resolve()): item
        for item in data_manifest["files"]
        if item["split"] == "validation"
    }
    paths = sorted(str(Path(path).resolve()) for path in glob.glob(validation_glob))
    if not paths or any(path not in expected_files for path in paths):
        raise ValueError("validation_glob is not a subset of the manifest validation split")
    for path in paths:
        if Path(path).stat().st_size != int(expected_files[path]["bytes"]):
            raise RuntimeError("validation shard byte size differs from its manifest")

    structure_path = Path(structure_database_path).resolve()
    structure_manifest_file = Path(structure_manifest_path).resolve()
    structure_receipt = verify_structure_database(
        str(structure_path),
        str(structure_manifest_file),
    )

    config = load_model_config(model_config_path)
    tokenizer = EsmcTokenizer.from_tokenizer_json(protein_tokenizer_json)
    artifact_receipt = verify_artifact_manifest(
        model_artifact_manifest_path,
        expected_root=config.protein_backbone.model_path,
    )
    expected_provenance = {
        "model_config_sha256": sha256_file(Path(model_config_path)),
        "tokenizer_sha256": sha256_file(Path(protein_tokenizer_json)),
        "model_artifact_manifest_sha256": sha256_file(Path(model_artifact_manifest_path)),
    }
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = OpenProteinModel.from_config(config, device=device, dtype=dtype)
    model.to(device=device, dtype=dtype).eval()
    checkpoint = load_trainable_checkpoint(model, checkpoint_dir, expected_provenance)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    expected_record_ids = None
    record_id_set = None
    allowlist_receipt = None
    if record_id_allowlist_path is not None:
        expected_record_ids, record_id_set, allowlist_receipt = load_record_id_allowlist(
            record_id_allowlist_path
        )
        if max_batches * batch_size < len(expected_record_ids):
            raise ValueError(
                "structure validation batch capacity is smaller than the record-ID allowlist"
            )

    stream = StructureCSVStream(
        paths,
        tokenizer,
        str(structure_path),
        sequence_length,
        seed,
        shuffle_files=False,
        min_confidence=min_confidence,
        include_record_ids=record_id_set,
    )
    iterator = iter(stream)
    collator = SequenceStructureCollator(
        tokenizer,
        seed,
        geometry_pairs,
        config.distance_bins,
        mlm_probability,
        min_confidence,
    )
    protein_nll = 0.0
    geometry_nll = 0.0
    clean_geometry_nll = 0.0
    protein_tokens = 0
    geometry_targets = 0
    geometry_correct = 0
    clean_geometry_correct = 0
    long_range_geometry_nll = 0.0
    long_range_geometry_targets = 0
    long_range_geometry_correct = 0
    distance_class_counts = torch.zeros(config.distance_bins, dtype=torch.long)
    separation_buckets = len(_GEOMETRY_SEPARATION_BOUNDARIES) + 1
    calibration_counts = torch.zeros(
        (separation_buckets, config.distance_bins), dtype=torch.long
    )
    prior_evaluation_buckets = []
    prior_evaluation_labels = []
    prior_model_nll = 0.0
    prior_model_correct = 0
    selected_record_ids: List[str] = []
    batches = 0
    for _ in range(max_batches):
        examples = []
        for _ in range(batch_size):
            try:
                examples.append(next(iterator))
            except StopIteration:
                break
        if not examples:
            break
        batch = {key: value.to(device) for key, value in collator(examples).items()}
        selected_record_ids.extend(str(example["record_id"]) for example in examples)
        result = model(**batch, use_bridge=True, attention_policy="bidirectional")
        if set(result["losses"]) != {"protein_mlm", "geometry"}:
            raise RuntimeError("structure validation did not produce every declared objective")
        protein_count = int(batch["protein_labels"].ne(-100).sum())
        geometry_mask = batch["distance_labels"].ne(-100)
        geometry_count = int(geometry_mask.sum())
        protein_nll += float(result["losses"]["protein_mlm"]) * protein_count
        geometry_nll += float(result["losses"]["geometry"]) * geometry_count
        predictions = result["distance_logits"].argmax(dim=-1)
        geometry_correct += int(
            predictions[geometry_mask].eq(batch["distance_labels"][geometry_mask]).sum()
        )
        clean_protein_input_ids = batch["protein_input_ids"].clone()
        masked_positions = batch["protein_labels"].ne(-100)
        clean_protein_input_ids[masked_positions] = batch["protein_labels"][masked_positions]
        clean_batch = {
            key: value
            for key, value in batch.items()
            if key != "protein_labels"
        }
        clean_batch["protein_input_ids"] = clean_protein_input_ids
        clean_result = model(
            **clean_batch,
            use_bridge=True,
            attention_policy="bidirectional",
        )
        clean_logits = clean_result["distance_logits"]
        clean_predictions = clean_logits.argmax(dim=-1)
        clean_geometry_nll += float(
            F.cross_entropy(
                clean_logits[geometry_mask].float(),
                batch["distance_labels"][geometry_mask],
                reduction="sum",
            )
        )
        clean_geometry_correct += int(
            clean_predictions[geometry_mask]
            .eq(batch["distance_labels"][geometry_mask])
            .sum()
        )
        separations = (batch["pair_indices"][..., 1] - batch["pair_indices"][..., 0]).abs()
        long_range_mask = geometry_mask & separations.ge(24)
        if long_range_mask.any():
            long_range_count = int(long_range_mask.sum())
            long_range_geometry_nll += float(
                F.cross_entropy(
                    clean_logits[long_range_mask].float(),
                    batch["distance_labels"][long_range_mask],
                    reduction="sum",
                )
            )
            long_range_geometry_targets += long_range_count
            long_range_geometry_correct += int(
                clean_predictions[long_range_mask]
                .eq(batch["distance_labels"][long_range_mask])
                .sum()
            )

        for example_index, example in enumerate(examples):
            example_mask = geometry_mask[example_index]
            labels = batch["distance_labels"][example_index][example_mask]
            example_separations = separations[example_index][example_mask]
            buckets = torch.bucketize(
                example_separations.cpu(),
                torch.tensor(_GEOMETRY_SEPARATION_BOUNDARIES),
                right=True,
            )
            labels_cpu = labels.cpu()
            distance_class_counts += torch.bincount(
                labels_cpu, minlength=config.distance_bins
            )
            split_byte = hashlib.sha256(str(example["record_id"]).encode("utf-8")).digest()[0]
            if split_byte % 2 == 0:
                flattened = buckets * config.distance_bins + labels_cpu
                calibration_counts += torch.bincount(
                    flattened,
                    minlength=separation_buckets * config.distance_bins,
                ).reshape(separation_buckets, config.distance_bins)
            else:
                prior_evaluation_buckets.append(buckets)
                prior_evaluation_labels.append(labels_cpu)
                example_logits = clean_logits[example_index][example_mask].float()
                prior_model_nll += float(
                    F.cross_entropy(example_logits, labels, reduction="sum")
                )
                prior_model_correct += int(
                    clean_predictions[example_index][example_mask].eq(labels).sum()
                )
        protein_tokens += protein_count
        geometry_targets += geometry_count
        batches += 1
    if not batches or not protein_tokens or not geometry_targets:
        raise RuntimeError("structure validation produced no supervised targets")
    if not long_range_geometry_targets:
        raise RuntimeError("structure validation produced no long-range geometry targets")
    if not int(calibration_counts.sum()) or not prior_evaluation_buckets:
        raise RuntimeError("structure validation could not form both prior splits")
    protein_loss = protein_nll / protein_tokens
    geometry_loss = geometry_nll / geometry_targets
    clean_geometry_loss = clean_geometry_nll / geometry_targets
    prior_diagnostics = _geometry_prior_diagnostics(
        calibration_counts,
        torch.cat(prior_evaluation_buckets),
        torch.cat(prior_evaluation_labels),
        prior_model_nll,
        prior_model_correct,
    )
    selected_subset = None
    if expected_record_ids is not None:
        if len(set(selected_record_ids)) != len(selected_record_ids):
            raise RuntimeError("structure evaluation selected duplicate record IDs")
        expected_set = set(expected_record_ids)
        selected_set = set(selected_record_ids)
        unexpected = sorted(selected_set - expected_set)
        if unexpected:
            raise RuntimeError(
                "structure evaluation selected IDs outside the allowlist: "
                + ", ".join(unexpected[:10])
            )
        missing = sorted(expected_set - selected_set)
        evaluation_order = "\n".join(selected_record_ids).encode("utf-8")
        missing_payload = "\n".join(missing).encode("utf-8")
        selected_subset = {
            "allowlisted_records": len(expected_record_ids),
            "evaluated_records": len(selected_record_ids),
            "structure_coverage": len(selected_record_ids) / len(expected_record_ids),
            "evaluation_order_sha256": hashlib.sha256(evaluation_order).hexdigest(),
            "selected_ids_are_allowlist_subset": True,
            "missing_records": len(missing),
            "missing_record_ids_sha256": hashlib.sha256(missing_payload).hexdigest(),
            "missing_policy": (
                "allowlisted records are omitted only when the exact AlphaFold sequence is "
                "absent or the deterministic crop lacks eligible pLDDT-gated geometry"
            ),
        }
    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": (
            "VALIDATION_DECONTAMINATED_SUBSET"
            if allowlist_receipt is not None
            else "VALIDATION"
        ),
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "structure_kind": "AlphaFold DB predicted",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "model_config_sha256": sha256_file(Path(model_config_path)),
        "protein_tokenizer_sha256": sha256_file(Path(protein_tokenizer_json)),
        "model_artifact": artifact_receipt,
        "data_manifest_sha256": sha256_file(manifest_path),
        "structure_manifest_sha256": sha256_file(structure_manifest_file),
        "structure_database_sha256": structure_receipt["database_sha256"],
        "validation_files": paths,
        "record_id_subset": (
            {**allowlist_receipt, **selected_subset}
            if allowlist_receipt is not None and selected_subset is not None
            else None
        ),
        "batches": batches,
        "protein_masked_tokens": protein_tokens,
        "geometry_targets": geometry_targets,
        "protein_mlm_loss": protein_loss,
        "protein_mlm_perplexity": math.exp(protein_loss),
        "geometry_loss": geometry_loss,
        "geometry_accuracy": geometry_correct / geometry_targets,
        "geometry_input_policy": "masked protein sequence retained for v2 compatibility",
        "clean_sequence_geometry_loss": clean_geometry_loss,
        "clean_sequence_geometry_accuracy": clean_geometry_correct / geometry_targets,
        "clean_sequence_long_range_geometry": {
            "minimum_sequence_separation": 24,
            "targets": long_range_geometry_targets,
            "loss": long_range_geometry_nll / long_range_geometry_targets,
            "accuracy": long_range_geometry_correct / long_range_geometry_targets,
        },
        "uniform_reference_loss": math.log(config.distance_bins),
        "uniform_reference_accuracy": 1.0 / config.distance_bins,
        "distance_class_counts": distance_class_counts.tolist(),
        "calibrated_prior_diagnostics": prior_diagnostics,
        "distance_bins": config.distance_bins,
        "minimum_plddt": min_confidence,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "evaluation_code_sha256": source_tree_sha256(Path(__file__).resolve().parent),
        "evaluation_settings": {
            "max_batches": max_batches,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "geometry_pairs": geometry_pairs,
            "mlm_probability": mlm_probability,
            "minimum_plddt": min_confidence,
            "seed": seed,
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(output, receipt)
    return receipt
