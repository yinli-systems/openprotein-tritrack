"""Evaluate the target checkpoint with a frozen-encoder TorchProtein EC probe."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch

from openprotein.artifacts import verify_artifact_manifest
from openprotein.config import load_model_config
from openprotein.data.text_tokenizer import SentencePieceTextTokenizer
from openprotein.data.tokenizer import EsmcTokenizer
from openprotein.function_probe import load_ec_benchmark, run_frozen_ec_probe
from openprotein.model import OpenProteinModel
from openprotein.provenance import sha256_file, source_tree_sha256
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


def _batches(values: Sequence[Dict[str, str]], size: int) -> Iterable[Sequence[Dict[str, str]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


@torch.inference_mode()
def _encode(args: argparse.Namespace, rows: List[Dict[str, str]]) -> tuple[torch.Tensor, Dict[str, Any]]:
    config = load_model_config(args.model_config)
    protein_tokenizer = EsmcTokenizer.from_tokenizer_json(args.protein_tokenizer_json)
    text_tokenizer = SentencePieceTextTokenizer(args.text_tokenizer_model)
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

    dummy_text_ids = text_tokenizer.encode("")
    embeddings = []
    lengths = []
    content_length = args.sequence_length - 2
    for batch_rows in _batches(rows, args.batch_size):
        protein_ids = [
            protein_tokenizer.encode(
                row["sequence"],
                max_length=args.sequence_length,
                crop_start=0,
            )
            for row in batch_rows
        ]
        text_ids = [dummy_text_ids for _ in batch_rows]
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
        if result["protein_embedding"] is None:
            raise RuntimeError("target checkpoint did not emit a protein embedding")
        embeddings.append(result["protein_embedding"].to(torch.float16).cpu())
        lengths.extend(len(row["sequence"]) for row in batch_rows)
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    )
    matrix = torch.cat(embeddings)
    del model, embeddings, result, protein, text
    if device.type == "cuda":
        torch.cuda.empty_cache()
    provenance = {
        "model": "OpenProtein multimodal target",
        "checkpoint": checkpoint,
        "model_config_sha256": sha256_file(Path(args.model_config)),
        "protein_tokenizer_sha256": sha256_file(Path(args.protein_tokenizer_json)),
        "text_tokenizer_sha256": sha256_file(Path(args.text_tokenizer_model)),
        "model_artifact": artifact_receipt,
        "embedding_policy": (
            "N-terminal deterministic crop; target contrastive protein projection after the "
            "bridge; prefix-LM mask prevents protein tokens from attending the dummy text"
        ),
        "sequence_length_diagnostics": {
            "minimum": min(lengths),
            "maximum": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "cropped_above_550_residues": sum(length > content_length for length in lengths),
        },
        "peak_encoder_memory_mib": peak_memory,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    return matrix, provenance


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite EC probe receipt: {output}")
    rows, benchmark = load_ec_benchmark(args.benchmark_csv, args.benchmark_receipt)
    embeddings, provenance = _encode(args, rows)
    probe = run_frozen_ec_probe(
        embeddings,
        rows,
        label_count=int(benchmark["labels"]),
        device_name=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        steps=args.probe_steps,
        validation_interval=args.probe_validation_interval,
        learning_rate=args.probe_learning_rate,
        minimum_steps=args.probe_minimum_steps,
        patience_validations=args.probe_patience_validations,
    )
    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": "PUBLIC_BENCHMARK_FROZEN_ENCODER_TRANSFER",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_data_accessed": True,
        "proteingym_rows_opened": False,
        "benchmark": "TorchProtein EnzymeCommission",
        "benchmark_receipt": {
            "path": str(Path(args.benchmark_receipt).resolve()),
            "sha256": sha256_file(Path(args.benchmark_receipt)),
        },
        **provenance,
        "result": probe,
        "evaluation_settings": {
            "batch_size": args.batch_size,
            "sequence_length_with_special_tokens": args.sequence_length,
            "maximum_protein_residues": args.sequence_length - 2,
            "protocol_iteration_after_public_test_access": bool(
                args.protocol_iteration_note
            ),
            "protocol_iteration_note": args.protocol_iteration_note,
        },
        "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
        "evaluation_source_tree_sha256": source_tree_sha256(Path.cwd()),
        "comparability_limitations": [
            "This is frozen-encoder transfer, not full-model GearNet-style fine-tuning.",
            "Benchmark identity cutoffs do not clear overlap with model pretraining corpora.",
            "Only CA residues present in supplied PDB chains enter the sequence view.",
            *(
                [args.protocol_iteration_note]
                if args.protocol_iteration_note
                else []
            ),
        ],
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
    parser.add_argument("--sequence-length", type=int, default=552)
    parser.add_argument("--probe-steps", type=int, default=20000)
    parser.add_argument("--probe-validation-interval", type=int, default=20)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-2)
    parser.add_argument("--probe-minimum-steps", type=int, default=2000)
    parser.add_argument("--probe-patience-validations", type=int, default=50)
    parser.add_argument("--protocol-iteration-note")
    parser.add_argument("--device")
    args = parser.parse_args()
    if (
        args.batch_size <= 0
        or args.sequence_length < 4
        or min(
            args.probe_steps,
            args.probe_validation_interval,
            args.probe_minimum_steps,
            args.probe_patience_validations,
        )
        <= 0
        or args.probe_minimum_steps > args.probe_steps
        or args.probe_learning_rate <= 0
    ):
        raise ValueError("invalid EC probe encoding dimensions")
    print(json.dumps(evaluate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
