"""Evaluate ProTrek-650M with the same frozen-encoder TorchProtein EC probe."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import huggingface_hub
import tokenizers
import torch
import transformers
from transformers import EsmTokenizer

from evaluate_protrek_retrieval import (
    ProteinEncoderCompat,
    _load_compatible_state,
    _official_encoder_states,
)
from openprotein.function_probe import load_ec_benchmark, run_frozen_ec_probe
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


def _batches(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


@torch.inference_mode()
def _encode(args: argparse.Namespace, rows: List[Dict[str, str]]) -> tuple[torch.Tensor, Dict[str, Any]]:
    weights_root = Path(args.weights_root).resolve()
    protein_config = weights_root / "esm2_t33_650M_UR50D"
    checkpoint_path = Path(args.checkpoint).resolve()
    tokenizer = EsmTokenizer.from_pretrained(protein_config, local_files_only=True)
    model = ProteinEncoderCompat(protein_config)
    protein_state, _, state_diagnostics = _official_encoder_states(checkpoint_path)
    state_diagnostics["ignored_obsolete_protein_buffers"] = _load_compatible_state(
        model,
        protein_state,
        ["model.esm.embeddings.position_ids"],
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    content_length = args.sequence_length - 2
    sequences = [row["sequence"][:content_length] for row in rows]
    lengths = [len(row["sequence"]) for row in rows]
    embeddings = []
    for batch in _batches(sequences, args.batch_size):
        encoded = tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        embeddings.append(model(encoded).to(torch.float16).cpu())
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    )
    matrix = torch.cat(embeddings)
    del model, embeddings, encoded, protein_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    provenance = {
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
        "state_load": state_diagnostics,
        "embedding_policy": (
            "N-terminal deterministic crop and official normalized ProTrek protein projection"
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
        raise FileExistsError(f"refusing to overwrite ProTrek EC probe receipt: {output}")
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
        "evidence_level": "PUBLIC_BENCHMARK_FROZEN_ENCODER_TRANSFER_PUBLIC_BASELINE",
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
    parser.add_argument("--weights-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark-csv", required=True)
    parser.add_argument("--benchmark-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--model-revision", required=True)
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
        raise ValueError("invalid ProTrek EC probe encoding dimensions")
    print(json.dumps(evaluate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
