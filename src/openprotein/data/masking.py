"""Deterministic ESM-style protein masking."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch


def mask_protein_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token_id: int,
    vocabulary_size: int,
    special_token_ids: Iterable[int],
    generator: torch.Generator,
    random_token_ids: Optional[Sequence[int]] = None,
    mlm_probability: float = 0.15,
    replace_with_mask_probability: float = 0.8,
    replace_with_random_probability: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return corrupted inputs and `-100`-masked labels.

    Sampling uses the supplied generator, making corruption reproducible across
    restarts when its state is checkpointed.
    """

    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have the same shape")
    if input_ids.ndim != 2:
        raise ValueError("input_ids and attention_mask must have shape [batch, length]")
    if not 0.0 <= mlm_probability <= 1.0:
        raise ValueError("mlm_probability must be in [0, 1]")
    if replace_with_mask_probability + replace_with_random_probability > 1.0:
        raise ValueError("replacement probabilities must sum to at most 1")

    eligible = attention_mask.to(dtype=torch.bool).clone()
    for token_id in special_token_ids:
        eligible &= input_ids.ne(int(token_id))

    sampling_scores = torch.rand(input_ids.shape, generator=generator, device=input_ids.device)
    selected = sampling_scores.lt(mlm_probability) & eligible
    if mlm_probability > 0.0:
        missing = eligible.any(dim=1) & ~selected.any(dim=1)
        if missing.any():
            fallback = sampling_scores.masked_fill(~eligible, 2.0).argmin(dim=1)
            rows = torch.arange(input_ids.shape[0], device=input_ids.device)[missing]
            selected[rows, fallback[missing]] = True
    labels = input_ids.clone()
    labels.masked_fill_(~selected, -100)

    decisions = torch.rand(input_ids.shape, generator=generator, device=input_ids.device)
    mask_positions = selected & decisions.lt(replace_with_mask_probability)
    random_positions = selected & decisions.ge(replace_with_mask_probability) & decisions.lt(
        replace_with_mask_probability + replace_with_random_probability
    )

    corrupted = input_ids.clone()
    corrupted[mask_positions] = int(mask_token_id)
    if random_token_ids is None:
        random_tokens = torch.randint(
            0,
            vocabulary_size,
            input_ids.shape,
            generator=generator,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
    else:
        candidates = torch.tensor(random_token_ids, dtype=input_ids.dtype, device=input_ids.device)
        if candidates.numel() == 0:
            raise ValueError("random_token_ids must not be empty")
        candidate_indices = torch.randint(
            0,
            candidates.numel(),
            input_ids.shape,
            generator=generator,
            device=input_ids.device,
        )
        random_tokens = candidates[candidate_indices]
    corrupted[random_positions] = random_tokens[random_positions]
    return corrupted, labels
