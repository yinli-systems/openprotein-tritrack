"""Evaluate the target multimodal checkpoint on a large retrieval candidate pool."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch

from openprotein.artifacts import verify_artifact_manifest
from openprotein.config import load_model_config
from openprotein.data.text_tokenizer import SentencePieceTextTokenizer
from openprotein.data.tokenizer import EsmcTokenizer
from openprotein.model import OpenProteinModel
from openprotein.provenance import sha256_file, source_tree_sha256
from openprotein.retrieval_stress import (
    large_pool_retrieval_metrics,
    load_retrieval_stress_rows,
)
from openprotein.validation import load_trainable_checkpoint


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _batches(values: Sequence[Dict[str, str]], batch_size: int) -> Iterable[Sequence[Dict[str, str]]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite retrieval stress receipt: {output}")
    rows, benchmark_receipt = load_retrieval_stress_rows(
        args.benchmark_csv, args.benchmark_receipt
    )
    if benchmark_receipt.get("test_data_accessed") is not False:
        raise RuntimeError("stress benchmark receipt does not prove test isolation")
    query_count = int(benchmark_receipt["benchmark"]["queries"])

    config = load_model_config(args.model_config)
    protein_tokenizer = EsmcTokenizer.from_tokenizer_json(args.protein_tokenizer_json)
    text_tokenizer = SentencePieceTextTokenizer(args.text_tokenizer_model)
    if text_tokenizer.vocabulary_size != config.text_vocab_size:
        raise ValueError("text tokenizer vocabulary does not match model config")
    artifact_receipt = verify_artifact_manifest(
        args.model_artifact_manifest,
        expected_root=config.protein_backbone.model_path,
    )
    expected_provenance = {
        "model_config_sha256": sha256_file(Path(args.model_config)),
        "tokenizer_sha256": sha256_file(Path(args.protein_tokenizer_json)),
        "text_tokenizer_sha256": sha256_file(Path(args.text_tokenizer_model)),
        "model_artifact_manifest_sha256": sha256_file(Path(args.model_artifact_manifest)),
    }
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = OpenProteinModel.from_config(config, device=device, dtype=dtype)
    model.to(device=device, dtype=dtype).eval()
    checkpoint = load_trainable_checkpoint(model, args.checkpoint, expected_provenance)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    crop_rng = random.Random(args.seed)
    protein_embeddings: List[torch.Tensor] = []
    text_embeddings: List[torch.Tensor] = []
    original_lengths: List[int] = []
    for batch_rows in _batches(rows, args.batch_size):
        protein_ids = []
        text_ids = []
        for row in batch_rows:
            sequence = row["sequence"]
            content_length = args.sequence_length - 2
            max_start = max(0, len(sequence) - content_length)
            crop_start = crop_rng.randint(0, max_start) if max_start else 0
            protein_ids.append(
                protein_tokenizer.encode(
                    sequence,
                    max_length=args.sequence_length,
                    crop_start=crop_start,
                )
            )
            text_ids.append(text_tokenizer.encode(row["text"], max_length=args.text_length))
            original_lengths.append(len(sequence))
        protein = protein_tokenizer.pad(protein_ids)
        text = text_tokenizer.pad(text_ids)
        result = model(
            protein_input_ids=protein["input_ids"].to(device),
            protein_attention_mask=protein["attention_mask"].to(device),
            text_input_ids=text["input_ids"].to(device),
            text_attention_mask=text["attention_mask"].to(device),
            use_bridge=True,
            attention_policy="prefix_lm",
        )
        if result["protein_embedding"] is None or result["text_embedding"] is None:
            raise RuntimeError("target model did not emit retrieval embeddings")
        protein_embeddings.append(result["protein_embedding"].to(torch.float16).cpu())
        text_embeddings.append(result["text_embedding"].to(torch.float16).cpu())

    protein_matrix = torch.cat(protein_embeddings)
    text_matrix = torch.cat(text_embeddings)
    peak_model_memory = (
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    )
    del model, protein_embeddings, text_embeddings, result, protein, text
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metrics = large_pool_retrieval_metrics(
        protein_matrix,
        text_matrix,
        query_count=query_count,
        device_name=str(device),
        chunk_size=args.metric_chunk_size,
    )
    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": "VALIDATION_PROTOCOL_INSPIRED_RETRIEVAL_STRESS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "official_protrek_benchmark_reproduction": False,
        "model": "OpenProtein multimodal target",
        "checkpoint": checkpoint,
        "model_config_sha256": sha256_file(Path(args.model_config)),
        "protein_tokenizer_sha256": sha256_file(Path(args.protein_tokenizer_json)),
        "text_tokenizer_sha256": sha256_file(Path(args.text_tokenizer_model)),
        "model_artifact": artifact_receipt,
        "benchmark_receipt": {
            "path": str(Path(args.benchmark_receipt).resolve()),
            "sha256": sha256_file(Path(args.benchmark_receipt)),
        },
        "metrics": metrics,
        "retrieval_encoding_policy": (
            "deterministic protein crop; protein prefix cannot attend text; text embedding is "
            "isolated from protein; exact-pair one-positive ranking over every candidate"
        ),
        "sequence_length_diagnostics": {
            "minimum": min(original_lengths),
            "maximum": max(original_lengths),
            "mean": sum(original_lengths) / len(original_lengths),
            "cropped_above_configured_length": sum(
                length > args.sequence_length - 2 for length in original_lengths
            ),
        },
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_model_memory_mib": peak_model_memory,
        "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
        "evaluation_source_tree_sha256": source_tree_sha256(Path.cwd()),
        "evaluation_settings": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "text_length": args.text_length,
            "seed": args.seed,
            "metric_chunk_size": args.metric_chunk_size,
            "embedding_storage_dtype": "float16",
            "metric_compute_dtype": "float32",
        },
        "comparability_limitations": benchmark_receipt["comparability_limitations"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protein-tokenizer-json", required=True)
    parser.add_argument("--text-tokenizer-model", required=True)
    parser.add_argument("--model-artifact-manifest", required=True)
    parser.add_argument("--benchmark-csv", required=True)
    parser.add_argument("--benchmark-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--text-length", type=int, default=256)
    parser.add_argument("--metric-chunk-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3701)
    parser.add_argument("--device")
    args = parser.parse_args()
    if min(
        args.batch_size,
        args.sequence_length - 2,
        args.text_length - 2,
        args.metric_chunk_size,
    ) <= 0:
        raise ValueError("invalid retrieval stress dimensions")
    print(json.dumps(evaluate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
