"""Deterministic public inference helpers for reconstructed releases."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from openprotein.data.text_tokenizer import SentencePieceTextTokenizer
from openprotein.data.tokenizer import EsmcTokenizer
from openprotein.model import OpenProteinModel


@torch.no_grad()
def generate_annotation(
    model: OpenProteinModel,
    protein_tokenizer: EsmcTokenizer,
    text_tokenizer: SentencePieceTextTokenizer,
    sequence: str,
    device: torch.device,
    *,
    prefix: str = "",
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """Greedily generate annotation text from a protein prefix-LM context."""

    normalized = sequence.strip().upper()
    if not normalized:
        raise ValueError("protein sequence must not be empty")
    if len(normalized) + 2 > model.config.max_sequence_length:
        raise ValueError("protein sequence exceeds the configured inference context")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    protein = protein_tokenizer.pad([protein_tokenizer.encode(normalized)])
    protein_ids = protein["input_ids"].to(device)
    protein_mask = protein["attention_mask"].to(device)
    if prefix:
        prefix_ids = text_tokenizer.encode(prefix)
        text_ids = prefix_ids[:-1]
    else:
        text_ids = [text_tokenizer.bos_token_id]
    if len(text_ids) >= model.config.max_text_length:
        raise ValueError("text prefix leaves no generation context")
    generated = []
    stopped_on_eos = False
    model.eval()
    for _ in range(min(max_new_tokens, model.config.max_text_length - len(text_ids))):
        text_tensor = torch.tensor([text_ids], dtype=torch.long, device=device)
        output = model(
            protein_input_ids=protein_ids,
            protein_attention_mask=protein_mask,
            text_input_ids=text_tensor,
            text_attention_mask=torch.ones_like(text_tensor),
            attention_policy="prefix_lm",
            use_bridge=True,
        )
        next_token = int(output["text_logits"][0, -1].float().argmax().item())
        if next_token == text_tokenizer.eos_token_id:
            stopped_on_eos = True
            break
        text_ids.append(next_token)
        generated.append(next_token)
    return {
        "sequence_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "sequence_length": len(normalized),
        "decoding": "greedy",
        "prefix": prefix,
        "max_new_tokens": max_new_tokens,
        "generated_tokens": generated,
        "stopped_on_eos": stopped_on_eos,
        "text": text_tokenizer.decode(text_ids),
    }


def torch_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if name is None:
        return None
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


@torch.no_grad()
def predict_pair_distances(
    model: OpenProteinModel,
    protein_tokenizer: EsmcTokenizer,
    sequence: str,
    residue_pairs: Sequence[Tuple[int, int]],
    device: torch.device,
) -> Dict[str, Any]:
    """Predict coarse C-alpha distance bins for one-based residue pairs."""

    normalized = sequence.strip().upper()
    if not normalized:
        raise ValueError("protein sequence must not be empty")
    if len(normalized) + 2 > model.config.max_sequence_length:
        raise ValueError("protein sequence exceeds the configured inference context")
    if not residue_pairs:
        raise ValueError("at least one residue pair is required")
    normalized_pairs = []
    for left, right in residue_pairs:
        left = int(left)
        right = int(right)
        if left == right or not (1 <= left <= len(normalized)) or not (
            1 <= right <= len(normalized)
        ):
            raise ValueError(f"invalid one-based residue pair: {left}:{right}")
        normalized_pairs.append((left, right))
    protein = protein_tokenizer.pad([protein_tokenizer.encode(normalized)])
    pair_indices = torch.tensor([normalized_pairs], dtype=torch.long, device=device)
    model.eval()
    output = model(
        protein_input_ids=protein["input_ids"].to(device),
        protein_attention_mask=protein["attention_mask"].to(device),
        pair_indices=pair_indices,
        attention_policy="bidirectional",
        use_bridge=True,
    )
    logits = output["distance_logits"][0].float()
    probabilities = torch.softmax(logits, dim=-1)
    predicted = probabilities.argmax(dim=-1)
    boundaries = torch.linspace(2.0, 32.0, steps=model.config.distance_bins - 1)
    predictions = []
    for pair, label, row in zip(normalized_pairs, predicted.tolist(), probabilities):
        predictions.append(
            {
                "residue_pair_one_based": list(pair),
                "distance_bin": int(label),
                "distance_lower_angstrom": (
                    None if label == 0 else float(boundaries[label - 1])
                ),
                "distance_upper_angstrom": (
                    None if label == model.config.distance_bins - 1 else float(boundaries[label])
                ),
                "probability": float(row[label]),
            }
        )
    return {
        "sequence_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "sequence_length": len(normalized),
        "distance_bins": model.config.distance_bins,
        "predictions": predictions,
        "structure_kind": "coarse pair-distance decoder trained on AlphaFold DB predictions",
    }
