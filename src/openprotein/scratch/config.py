"""Fail-closed configuration and exact accounting for scratch protein models."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict


SCRATCH_CONFIG_SCHEMA_VERSION = 1
SCRATCH_TARGET_MIN_PARAMETERS = 6_900_000_000
SCRATCH_TARGET_MAX_PARAMETERS = 7_300_000_000


@dataclass(frozen=True)
class ScratchModelConfig:
    """Architecture contract for a randomly initialized masked encoder.

    The deliberately narrow schema prevents a pretrained-backbone path or an
    initialization checkpoint from being smuggled into a scratch-model config.
    """

    schema_version: int
    name: str
    architecture: str
    initialization: str
    vocab_size: int
    pad_token_id: int
    mask_token_id: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    max_sequence_length: int
    position_embedding_type: str
    activation: str
    dropout: float
    attention_dropout: float
    token_dropout: bool
    tie_word_embeddings: bool
    initializer_range: float
    declared_parameter_count: int

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ScratchModelConfig":
        expected = {field.name for field in fields(cls)}
        received = set(raw)
        missing = expected - received
        unknown = received - expected
        if missing:
            raise ValueError(f"scratch model config is missing fields: {sorted(missing)}")
        if unknown:
            raise ValueError(f"scratch model config has unknown fields: {sorted(unknown)}")
        config = cls(
            schema_version=int(raw["schema_version"]),
            name=str(raw["name"]),
            architecture=str(raw["architecture"]),
            initialization=str(raw["initialization"]),
            vocab_size=int(raw["vocab_size"]),
            pad_token_id=int(raw["pad_token_id"]),
            mask_token_id=int(raw["mask_token_id"]),
            num_layers=int(raw["num_layers"]),
            hidden_size=int(raw["hidden_size"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            intermediate_size=int(raw["intermediate_size"]),
            max_sequence_length=int(raw["max_sequence_length"]),
            position_embedding_type=str(raw["position_embedding_type"]),
            activation=str(raw["activation"]),
            dropout=float(raw["dropout"]),
            attention_dropout=float(raw["attention_dropout"]),
            token_dropout=bool(raw["token_dropout"]),
            tie_word_embeddings=bool(raw["tie_word_embeddings"]),
            initializer_range=float(raw["initializer_range"]),
            declared_parameter_count=int(raw["declared_parameter_count"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != SCRATCH_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"scratch config schema must be {SCRATCH_CONFIG_SCHEMA_VERSION}"
            )
        if self.architecture != "masked_encoder":
            raise ValueError("scratch architecture must be masked_encoder")
        if self.initialization != "random":
            raise ValueError("scratch initialization must be random")
        if not self.tie_word_embeddings:
            raise ValueError("scratch model requires tied input/output embeddings")
        if self.position_embedding_type != "rotary":
            raise ValueError("scratch model requires rotary positions")
        if self.activation != "gelu":
            raise ValueError("scratch model requires GELU")
        for name in (
            "vocab_size",
            "num_layers",
            "hidden_size",
            "num_attention_heads",
            "intermediate_size",
            "max_sequence_length",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("attention head dimension must be even for rotary embeddings")
        if self.intermediate_size != 4 * self.hidden_size:
            raise ValueError("ESM-style scratch encoder requires intermediate_size=4*hidden_size")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id is outside the vocabulary")
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError("mask_token_id is outside the vocabulary")
        if self.pad_token_id == self.mask_token_id:
            raise ValueError("pad and mask token ids must differ")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive")
        if self.declared_parameter_count != self.parameter_count:
            raise ValueError(
                "declared scratch parameter count does not match the architecture: "
                f"{self.declared_parameter_count} != {self.parameter_count}"
            )

    def parameter_breakdown(self) -> Dict[str, int]:
        """Return the exact count for :class:`ScratchProteinModel`.

        Each encoder layer has biased Q/K/V/output projections, two biased
        LayerNorms, and a biased 4x GELU feed-forward network. The LM decoder
        weight is tied to the token embedding and is therefore not counted
        twice.
        """

        d = self.hidden_size
        i = self.intermediate_size
        per_layer = (
            3 * d * d
            + 3 * d
            + d * d
            + d
            + 4 * d
            + d * i
            + i
            + i * d
            + d
        )
        return {
            "token_embedding": self.vocab_size * d,
            "encoder_layers": self.num_layers * per_layer,
            "final_layer_norm": 2 * d,
            "lm_head_dense": d * d + d,
            "lm_head_layer_norm": 2 * d,
            "lm_head_bias": self.vocab_size,
        }

    @property
    def parameter_count(self) -> int:
        return sum(self.parameter_breakdown().values())

    @property
    def is_7b_scale(self) -> bool:
        return SCRATCH_TARGET_MIN_PARAMETERS <= self.parameter_count <= SCRATCH_TARGET_MAX_PARAMETERS


def load_scratch_model_config(path: str) -> ScratchModelConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("scratch model config must be a JSON object")
    return ScratchModelConfig.from_dict(raw)
