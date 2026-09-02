"""ESM-style masked encoder implemented for random-initialized pretraining."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from openprotein.scratch.config import ScratchModelConfig


def _apply_rotary(q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE without constructing a quadratic sequence mask."""

    head_dim = q.shape[-1]
    sequence_length = q.shape[-2]
    positions = torch.arange(sequence_length, device=q.device, dtype=torch.float32)
    dimensions = torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32)
    inverse_frequency = torch.exp(-math.log(10_000.0) * dimensions / head_dim)
    angles = torch.outer(positions, inverse_frequency)
    cos = angles.cos().to(q.dtype)[None, None, :, :]
    sin = angles.sin().to(q.dtype)[None, None, :, :]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        ).flatten(-2)

    return rotate(q), rotate(k)


class ScratchSelfAttention(nn.Module):
    def __init__(self, config: ScratchModelConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dropout = config.attention_dropout
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, sequence_length, _ = hidden.shape
        q, k, v = self.qkv(hidden).chunk(3, dim=-1)

        def split_heads(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        q, k = _apply_rotary(q, k)
        # Bidirectional attention only needs a key-padding mask. Avoiding a
        # [batch, sequence, sequence] allocation is essential at length 1024.
        key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=key_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, sequence_length, self.hidden_size)
        return self.output(attended)


class ScratchEncoderLayer(nn.Module):
    """Pre-normalized ESM-style attention and GELU feed-forward block."""

    def __init__(self, config: ScratchModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.attention = ScratchSelfAttention(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_in = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.ffn_out = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.dropout = config.dropout

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention = self.attention(self.attention_norm(hidden), attention_mask)
        hidden = hidden + F.dropout(attention, p=self.dropout, training=self.training)
        feed_forward = self.ffn_out(F.gelu(self.ffn_in(self.ffn_norm(hidden))))
        hidden = hidden + F.dropout(feed_forward, p=self.dropout, training=self.training)
        return hidden.masked_fill(~attention_mask.to(dtype=torch.bool).unsqueeze(-1), 0.0)


class ScratchLMHead(nn.Module):
    def __init__(self, config: ScratchModelConfig, embedding_weight: nn.Parameter) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.embedding_weight = embedding_weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.layer_norm(F.gelu(self.dense(hidden)))
        return F.linear(hidden, self.embedding_weight, self.bias)


class ScratchProteinModel(nn.Module):
    """A full-parameter masked protein encoder with no external backbone."""

    def __init__(self, config: ScratchModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [ScratchEncoderLayer(config) for _ in range(config.num_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = ScratchLMHead(config, self.token_embedding.weight)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Scale residual branch outputs by depth to keep the random network
        # stable while retaining the ESM-style layer parameterization.
        residual_std = std / math.sqrt(2.0 * self.config.num_layers)
        for layer in (
            module for module in self.modules() if isinstance(module, ScratchEncoderLayer)
        ):
            nn.init.normal_(layer.attention.output.weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer.ffn_out.weight, mean=0.0, std=residual_std)
        nn.init.zeros_(self.lm_head.bias)
        with torch.no_grad():
            self.token_embedding.weight[self.config.pad_token_id].zero_()

    def _embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.token_embedding(input_ids)
        if self.config.token_dropout:
            masked = input_ids.eq(self.config.mask_token_id) & attention_mask.to(torch.bool)
            hidden = hidden.masked_fill(masked.unsqueeze(-1), 0.0)
            valid_tokens = attention_mask.sum(dim=-1).clamp_min(1).to(hidden.dtype)
            observed = masked.sum(dim=-1).to(hidden.dtype) / valid_tokens
            expected = 0.15 * 0.80
            scale = (1.0 - expected) / (1.0 - observed).clamp_min(1e-6)
            hidden = hidden * scale[:, None, None]
        return hidden.masked_fill(
            ~attention_mask.to(dtype=torch.bool).unsqueeze(-1), 0.0
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [batch, length]")
        if input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("input sequence exceeds max_sequence_length")
        if input_ids.numel() and (
            input_ids.min().item() < 0 or input_ids.max().item() >= self.config.vocab_size
        ):
            raise ValueError("input_ids contains a token outside the vocabulary")
        if not attention_mask.to(torch.bool).any(dim=-1).all():
            raise ValueError("every sequence must contain at least one valid token")

        hidden = self._embed(input_ids, attention_mask)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        hidden = self.final_layer_norm(hidden)
        logits = self.lm_head(hidden)
        loss: Optional[torch.Tensor] = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits.float().reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits, "hidden_states": hidden}
