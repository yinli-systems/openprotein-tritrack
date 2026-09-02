"""Attention masks and bridge blocks for protein/text fusion."""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _as_bool_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, length]")
    return mask.to(dtype=torch.bool)


def build_bidirectional_mask(
    protein_mask: torch.Tensor, text_mask: torch.Tensor
) -> torch.Tensor:
    """Return an allowed-attention mask with shape ``[B, S, S]``."""

    protein_mask = _as_bool_mask(protein_mask, "protein_mask")
    text_mask = _as_bool_mask(text_mask, "text_mask")
    if protein_mask.shape[0] != text_mask.shape[0]:
        raise ValueError("protein and text masks must have the same batch size")
    valid = torch.cat([protein_mask, text_mask], dim=1)
    return valid[:, :, None] & valid[:, None, :]


def build_prefix_lm_mask(
    protein_mask: torch.Tensor, text_mask: torch.Tensor
) -> torch.Tensor:
    """Build a leak-free protein-prefix/text-causal attention mask.

    Protein queries see only valid protein keys. Text queries see every valid
    protein key and valid text keys up to their own text position.
    """

    protein_mask = _as_bool_mask(protein_mask, "protein_mask")
    text_mask = _as_bool_mask(text_mask, "text_mask")
    if protein_mask.shape[0] != text_mask.shape[0]:
        raise ValueError("protein and text masks must have the same batch size")

    batch, protein_length = protein_mask.shape
    text_length = text_mask.shape[1]
    total_length = protein_length + text_length
    allowed = torch.zeros(
        (batch, total_length, total_length), dtype=torch.bool, device=protein_mask.device
    )

    protein_pairs = protein_mask[:, :, None] & protein_mask[:, None, :]
    allowed[:, :protein_length, :protein_length] = protein_pairs

    if text_length:
        text_to_protein = text_mask[:, :, None] & protein_mask[:, None, :]
        allowed[:, protein_length:, :protein_length] = text_to_protein

        causal = torch.ones(
            (text_length, text_length), dtype=torch.bool, device=protein_mask.device
        ).tril()
        text_pairs = text_mask[:, :, None] & text_mask[:, None, :] & causal[None, :, :]
        allowed[:, protein_length:, protein_length:] = text_pairs

    return allowed


def _apply_rotary(q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to ``[B, H, S, D]`` query and key tensors."""

    head_dim = q.shape[-1]
    if head_dim % 2:
        raise ValueError("rotary head dimension must be even")
    sequence_length = q.shape[-2]
    positions = torch.arange(sequence_length, device=q.device, dtype=torch.float32)
    dimensions = torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32)
    inv_frequency = torch.exp(-math.log(10000.0) * dimensions / head_dim)
    angles = torch.outer(positions, inv_frequency)
    cos = angles.cos().to(dtype=q.dtype)[None, None, :, :]
    sin = angles.sin().to(dtype=q.dtype)[None, None, :, :]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    return rotate(q), rotate(k)


class BridgeBlock(nn.Module):
    """Pre-normalized RoPE attention plus SwiGLU feed-forward block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.residual_scale = residual_scale

        self.attention_norm = nn.RMSNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.attention_output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ffn_norm = nn.RMSNorm(hidden_size)
        self.gate_up = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        batch, sequence_length, _ = hidden.shape
        normalized = self.attention_norm(hidden)
        q, k, v = self.qkv(normalized).chunk(3, dim=-1)

        def heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(batch, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)
        q, k = _apply_rotary(q, k)
        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed[:, None, :, :],
            dropout_p=self.dropout if self.training else 0.0,
        )
        attention = attention.transpose(1, 2).reshape(batch, sequence_length, self.hidden_size)
        hidden = hidden + self.residual_scale * self.attention_output(attention)

        gate, up = self.gate_up(self.ffn_norm(hidden)).chunk(2, dim=-1)
        feed_forward = self.down(F.silu(gate) * up)
        hidden = hidden + self.residual_scale * feed_forward

        valid_queries = allowed.any(dim=-1).unsqueeze(-1)
        return hidden.masked_fill(~valid_queries, 0.0)
