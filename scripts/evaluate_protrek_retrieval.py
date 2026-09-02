"""Evaluate the official ProTrek-650M checkpoint on the frozen validation pairs."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import huggingface_hub
import tokenizers
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
    EsmConfig,
    EsmForMaskedLM,
    EsmTokenizer,
)

from openprotein.provenance import sha256_file, source_tree_sha256
from openprotein.evaluation_subset import (
    load_record_id_allowlist,
    verify_selected_record_ids,
)
from openprotein.validation import retrieval_metrics


class ProteinEncoderCompat(nn.Module):
    """Inference-only equivalent of ProTrek's official ProteinEncoder."""

    def __init__(self, config_path: Path, representation_dimension: int = 1024) -> None:
        super().__init__()
        config = EsmConfig.from_pretrained(config_path, local_files_only=True)
        self.model = EsmForMaskedLM(config)
        self.model.esm.contact_head = None
        if config.position_embedding_type == "rotary":
            self.model.esm.embeddings.position_embeddings = None
        self.out = nn.Linear(config.hidden_size, representation_dimension)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.model.esm(**inputs).last_hidden_state[:, 0]
        return F.normalize(self.out(hidden), dim=-1)


class TextEncoderCompat(nn.Module):
    """Inference-only equivalent of ProTrek's official TextEncoder."""

    def __init__(self, config_path: Path, representation_dimension: int = 1024) -> None:
        super().__init__()
        config = BertConfig.from_pretrained(config_path, local_files_only=True)
        self.model = BertModel(config, add_pooling_layer=False)
        self.out = nn.Linear(config.hidden_size, representation_dimension)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.model(**inputs).last_hidden_state[:, 0]
        return F.normalize(self.out(hidden), dim=-1)


def _load_validation_pairs(
    paths: Iterable[str],
    sequence_length: int,
    seed: int,
    include_record_ids: set[str] | None = None,
) -> Tuple[List[str], List[str], List[str], List[str], List[int]]:
    """Reproduce MultimodalCSVStream's deterministic protein cropping policy."""

    crop_rng = random.Random(seed)
    sequences: List[str] = []
    texts: List[str] = []
    record_ids: List[str] = []
    text_groups: List[str] = []
    original_lengths: List[int] = []
    content_length = sequence_length - 2
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"id", "sequence", "text"}
            if reader.fieldnames is None or not required <= set(reader.fieldnames):
                raise ValueError(f"{path} lacks multimodal columns")
            for row_index, row in enumerate(reader):
                sequence = row["sequence"].strip().upper()
                text = row["text"].strip()
                if not sequence or not text:
                    raise ValueError(f"{path}:{row_index + 2} contains an empty modality")
                max_start = max(0, len(sequence) - content_length)
                crop_start = crop_rng.randint(0, max_start) if max_start else 0
                if include_record_ids is not None and row["id"] not in include_record_ids:
                    continue
                sequences.append(sequence[crop_start : crop_start + content_length])
                texts.append(text)
                record_ids.append(row["id"])
                text_groups.append(" ".join(text.casefold().split()))
                original_lengths.append(len(sequence))
    if len(sequences) < 2:
        raise RuntimeError("retrieval evaluation requires at least two validation pairs")
    return sequences, texts, record_ids, text_groups, original_lengths


def _official_encoder_states(
    checkpoint_path: Path,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, object]]:
    """Extract ParameterList slots 1 and 2 used by the official ProTrek code."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError("ProTrek checkpoint does not contain the official model state")
    state = payload["model"]
    protein = {key[2:]: value for key, value in state.items() if key.startswith("1.")}
    text = {key[2:]: value for key, value in state.items() if key.startswith("2.")}
    if not protein or not text:
        raise RuntimeError("ProTrek checkpoint lacks official protein/text encoder slots")
    diagnostics = {
        "checkpoint_tensors": len(state),
        "protein_encoder_tensors": len(protein),
        "text_encoder_tensors": len(text),
        "ignored_non_retrieval_tensors": len(state) - len(protein) - len(text),
    }
    return protein, text, diagnostics


def _load_compatible_state(
    module: nn.Module,
    state: Dict[str, torch.Tensor],
    allowed_obsolete_buffers: Sequence[str],
) -> List[str]:
    expected = set(module.state_dict())
    supplied = set(state)
    missing = sorted(expected - supplied)
    unexpected = sorted(supplied - expected)
    disallowed = sorted(set(unexpected) - set(allowed_obsolete_buffers))
    if missing or disallowed:
        raise RuntimeError(
            f"checkpoint compatibility failure: missing={missing}, unexpected={disallowed}"
        )
    filtered = {key: value for key, value in state.items() if key in expected}
    module.load_state_dict(filtered, strict=True, assign=True)
    return unexpected


def _batches(values: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite baseline receipt: {output}")
    manifest_path = Path(args.data_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_root = Path(manifest["dataset_root"])
    expected_files = {
        str((dataset_root / item["path"]).resolve()): item
        for item in manifest["files"]
        if item["split"] == "validation"
    }
    paths = sorted(str(Path(path).resolve()) for path in glob.glob(args.validation_glob))
    if not paths or any(path not in expected_files for path in paths):
        raise ValueError("validation_glob is not a subset of the manifest validation split")
    for path in paths:
        if Path(path).stat().st_size != int(expected_files[path]["bytes"]):
            raise RuntimeError("validation shard byte size differs from its manifest")

    expected_record_ids = None
    record_id_set = None
    allowlist_receipt = None
    if args.record_id_allowlist is not None:
        expected_record_ids, record_id_set, allowlist_receipt = load_record_id_allowlist(
            args.record_id_allowlist
        )
    sequences, texts, record_ids, text_groups, original_lengths = _load_validation_pairs(
        paths,
        args.sequence_length,
        args.seed,
        record_id_set,
    )
    selected_subset = None
    if expected_record_ids is not None:
        selected_subset = verify_selected_record_ids(record_ids, expected_record_ids)
    weights_root = Path(args.weights_root).resolve()
    protein_config = weights_root / "esm2_t33_650M_UR50D"
    text_config = weights_root / "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    checkpoint_path = Path(args.checkpoint).resolve()
    protein_tokenizer = EsmTokenizer.from_pretrained(
        protein_config, local_files_only=True
    )
    text_tokenizer = BertTokenizer.from_pretrained(text_config, local_files_only=True)
    protein_model = ProteinEncoderCompat(protein_config)
    text_model = TextEncoderCompat(text_config)
    protein_state, text_state, state_diagnostics = _official_encoder_states(
        checkpoint_path
    )
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
    parameter_counts = {
        "protein_encoder": sum(parameter.numel() for parameter in protein_model.parameters()),
        "text_encoder": sum(parameter.numel() for parameter in text_model.parameters()),
    }

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    protein_model.to(device=device, dtype=dtype).eval()
    text_model.to(device=device, dtype=dtype).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    protein_embeddings = []
    for batch in _batches(sequences, args.batch_size):
        encoded = protein_tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            padding=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        protein_embeddings.append(protein_model(encoded).float().cpu())
    del protein_model, protein_state

    text_embeddings = []
    for batch in _batches(texts, args.batch_size):
        encoded = text_tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=args.text_length,
            padding=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        text_embeddings.append(text_model(encoded).float().cpu())

    protein_matrix = torch.cat(protein_embeddings)
    text_matrix = torch.cat(text_embeddings)
    group_counts: Dict[str, int] = {}
    for group in text_groups:
        group_counts[group] = group_counts.get(group, 0) + 1
    receipt: Dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": (
            "VALIDATION_DECONTAMINATED_SUBSET_PUBLIC_BASELINE"
            if allowlist_receipt is not None
            else "VALIDATION_PUBLIC_BASELINE"
        ),
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
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
        "comparability_limitations": [
            (
                "The ProTrek model card names Swiss-Prot plus TrEMBL50 as training data, but "
                "does not expose enough record-level provenance to exclude overlap with the "
                "2026_02 validation proteins."
            ),
            (
                "This is a same-pool local validation comparison, not an official ProTrek "
                "benchmark result."
            ),
        ],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "protein_config_sha256": sha256_file(protein_config / "config.json"),
        "protein_vocabulary_sha256": sha256_file(protein_config / "vocab.txt"),
        "text_config_sha256": sha256_file(text_config / "config.json"),
        "text_vocabulary_sha256": sha256_file(text_config / "vocab.txt"),
        "state_load": state_diagnostics,
        "retrieval_parameter_counts": parameter_counts,
        "data_manifest_sha256": sha256_file(manifest_path),
        "validation_files": paths,
        "record_id_sha256": hashlib.sha256(
            "\n".join(record_ids).encode("utf-8")
        ).hexdigest(),
        "record_id_subset": (
            {**allowlist_receipt, **selected_subset}
            if allowlist_receipt is not None and selected_subset is not None
            else None
        ),
        "pairs": len(sequences),
        "retrieval": retrieval_metrics(protein_matrix, text_matrix),
        "retrieval_group_aware": retrieval_metrics(
            protein_matrix,
            text_matrix,
            match_group_ids=text_groups,
        ),
        "retrieval_encoding_policy": (
            "same raw validation rows, order, deterministic protein crop, 256-token protein "
            "context, and exact/group-aware candidate pool as the target model; ProTrek uses "
            "its model-native tokenizers with text capped at 256 tokens"
        ),
        "duplicate_text_diagnostics": {
            "normalized_text_groups": len(group_counts),
            "duplicate_groups": sum(count > 1 for count in group_counts.values()),
            "pairs_in_duplicate_groups": sum(
                count for count in group_counts.values() if count > 1
            ),
            "largest_group": max(group_counts.values()),
        },
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
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else None
        ),
        "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
        "evaluation_source_tree_sha256": source_tree_sha256(Path.cwd()),
        "evaluation_settings": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "text_length": args.text_length,
            "seed": args.seed,
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-glob", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--text-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3701)
    parser.add_argument("--device")
    parser.add_argument("--record-id-allowlist")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.sequence_length < 3 or args.text_length < 2:
        raise ValueError("invalid evaluation batch or context dimensions")
    print(json.dumps(evaluate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
