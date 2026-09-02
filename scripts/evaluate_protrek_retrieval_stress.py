"""Evaluate the pinned ProTrek-650M baseline on the local retrieval stress pool."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import huggingface_hub
import tokenizers
import torch
import transformers
from transformers import BertTokenizer, EsmTokenizer

from evaluate_protrek_retrieval import (
    ProteinEncoderCompat,
    TextEncoderCompat,
    _load_compatible_state,
    _official_encoder_states,
)
from openprotein.provenance import sha256_file, source_tree_sha256
from openprotein.retrieval_stress import (
    large_pool_retrieval_metrics,
    load_retrieval_stress_rows,
)


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


def _string_batches(values: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite ProTrek stress receipt: {output}")
    rows, benchmark_receipt = load_retrieval_stress_rows(
        args.benchmark_csv, args.benchmark_receipt
    )
    if benchmark_receipt.get("test_data_accessed") is not False:
        raise RuntimeError("stress benchmark receipt does not prove test isolation")
    query_count = int(benchmark_receipt["benchmark"]["queries"])

    weights_root = Path(args.weights_root).resolve()
    protein_config = weights_root / "esm2_t33_650M_UR50D"
    text_config = weights_root / "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    checkpoint_path = Path(args.checkpoint).resolve()
    protein_tokenizer = EsmTokenizer.from_pretrained(protein_config, local_files_only=True)
    text_tokenizer = BertTokenizer.from_pretrained(text_config, local_files_only=True)
    protein_model = ProteinEncoderCompat(protein_config)
    text_model = TextEncoderCompat(text_config)
    protein_state, text_state, state_diagnostics = _official_encoder_states(checkpoint_path)
    state_diagnostics["ignored_obsolete_protein_buffers"] = _load_compatible_state(
        protein_model,
        protein_state,
        ["model.esm.embeddings.position_ids"],
    )
    state_diagnostics["ignored_obsolete_text_buffers"] = _load_compatible_state(
        text_model,
        text_state,
        ["model.embeddings.position_ids"],
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    protein_model.to(device=device, dtype=dtype).eval()
    text_model.to(device=device, dtype=dtype).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    crop_rng = random.Random(args.seed)
    sequences: List[str] = []
    texts: List[str] = []
    original_lengths: List[int] = []
    content_length = args.sequence_length - 2
    for row in rows:
        sequence = row["sequence"]
        max_start = max(0, len(sequence) - content_length)
        crop_start = crop_rng.randint(0, max_start) if max_start else 0
        sequences.append(sequence[crop_start : crop_start + content_length])
        texts.append(row["text"])
        original_lengths.append(len(sequence))

    protein_embeddings: List[torch.Tensor] = []
    for batch_values in _string_batches(sequences, args.batch_size):
        encoded = protein_tokenizer.batch_encode_plus(
            batch_values, return_tensors="pt", padding=True
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        protein_embeddings.append(protein_model(encoded).to(torch.float16).cpu())
    del protein_model, protein_state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    text_embeddings: List[torch.Tensor] = []
    for batch_values in _string_batches(texts, args.batch_size):
        encoded = text_tokenizer.batch_encode_plus(
            batch_values,
            return_tensors="pt",
            truncation=True,
            max_length=args.text_length,
            padding=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        text_embeddings.append(text_model(encoded).to(torch.float16).cpu())
    peak_model_memory = (
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    )
    del text_model, text_state, encoded
    if device.type == "cuda":
        torch.cuda.empty_cache()

    protein_matrix = torch.cat(protein_embeddings)
    text_matrix = torch.cat(text_embeddings)
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
        "evidence_level": "VALIDATION_PROTOCOL_INSPIRED_RETRIEVAL_STRESS_PUBLIC_BASELINE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "official_protrek_benchmark_reproduction": False,
        "baseline": "westlake-repl/ProTrek_650M",
        "baseline_repository": "https://github.com/westlake-repl/ProTrek",
        "baseline_repository_commit": args.repository_commit,
        "baseline_model_revision": args.model_revision,
        "runtime_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
        },
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "protein_config_sha256": sha256_file(protein_config / "config.json"),
        "protein_vocabulary_sha256": sha256_file(protein_config / "vocab.txt"),
        "text_config_sha256": sha256_file(text_config / "config.json"),
        "text_vocabulary_sha256": sha256_file(text_config / "vocab.txt"),
        "state_load": state_diagnostics,
        "benchmark_receipt": {
            "path": str(Path(args.benchmark_receipt).resolve()),
            "sha256": sha256_file(Path(args.benchmark_receipt)),
        },
        "metrics": metrics,
        "retrieval_encoding_policy": (
            "same benchmark rows, order, deterministic protein crop, context lengths, exact-pair "
            "one-positive metric, and candidate pool as the target model; model-native tokenizers"
        ),
        "sequence_length_diagnostics": {
            "minimum": min(original_lengths),
            "maximum": max(original_lengths),
            "mean": sum(original_lengths) / len(original_lengths),
            "cropped_above_configured_length": sum(
                length > content_length for length in original_lengths
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
    parser.add_argument("--weights-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark-csv", required=True)
    parser.add_argument("--benchmark-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--model-revision", required=True)
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
        raise ValueError("invalid ProTrek retrieval stress dimensions")
    print(json.dumps(evaluate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
