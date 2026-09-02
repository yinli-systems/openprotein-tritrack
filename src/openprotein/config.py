"""Typed model configuration and parameter accounting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


TARGET_MIN_PARAMETERS = 7_000_000_000
TARGET_MAX_PARAMETERS = 15_000_000_000


@dataclass(frozen=True)
class BackboneConfig:
    kind: str
    model_path: Optional[str]
    parameter_count: int
    vocab_size: int
    freeze: bool = True

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "BackboneConfig":
        return cls(
            kind=str(raw["kind"]),
            model_path=raw.get("model_path"),
            parameter_count=int(raw["parameter_count"]),
            vocab_size=int(raw["vocab_size"]),
            freeze=bool(raw.get("freeze", True)),
        )


@dataclass(frozen=True)
class ModelConfig:
    name: str
    protein_backbone: BackboneConfig
    hidden_size: int
    num_attention_heads: int
    num_bridge_layers: int
    intermediate_size: int
    text_vocab_size: int
    max_sequence_length: int
    max_text_length: int
    contrastive_dimension: int
    geometry_rank: int
    distance_bins: int
    dropout: float = 0.0
    residual_scale: float = 1.0
    tie_text_embeddings: bool = True
    structure_feature_dimension: int = 0
    structure_fusion: str = "global_gate"
    structure_mask_token: bool = False
    contrastive_text_layers: int = 0
    contrastive_temperature: float = 0.07
    multi_positive_contrastive: bool = False

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ModelConfig":
        return cls(
            name=str(raw["name"]),
            protein_backbone=BackboneConfig.from_dict(raw["protein_backbone"]),
            hidden_size=int(raw["hidden_size"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_bridge_layers=int(raw["num_bridge_layers"]),
            intermediate_size=int(raw["intermediate_size"]),
            text_vocab_size=int(raw["text_vocab_size"]),
            max_sequence_length=int(raw["max_sequence_length"]),
            max_text_length=int(raw["max_text_length"]),
            contrastive_dimension=int(raw["contrastive_dimension"]),
            geometry_rank=int(raw["geometry_rank"]),
            distance_bins=int(raw["distance_bins"]),
            dropout=float(raw.get("dropout", 0.0)),
            residual_scale=float(raw.get("residual_scale", 1.0)),
            tie_text_embeddings=bool(raw.get("tie_text_embeddings", True)),
            structure_feature_dimension=int(raw.get("structure_feature_dimension", 0)),
            structure_fusion=str(raw.get("structure_fusion", "global_gate")),
            structure_mask_token=bool(raw.get("structure_mask_token", False)),
            contrastive_text_layers=int(raw.get("contrastive_text_layers", 0)),
            contrastive_temperature=float(raw.get("contrastive_temperature", 0.07)),
            multi_positive_contrastive=bool(
                raw.get("multi_positive_contrastive", False)
            ),
        )

    def validate(self) -> None:
        if self.protein_backbone.kind not in {"esm2", "esmc", "tiny"}:
            raise ValueError("protein_backbone.kind must be 'esm2', 'esmc', or 'tiny'")
        if self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("hidden and intermediate sizes must be positive")
        if self.num_bridge_layers <= 0:
            raise ValueError("num_bridge_layers must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("attention head dimension must be even for rotary embeddings")
        if self.protein_backbone.vocab_size <= 0 or self.text_vocab_size <= 0:
            raise ValueError("vocabulary sizes must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        if self.distance_bins < 2:
            raise ValueError("distance_bins must be at least 2")
        if self.structure_feature_dimension < 0:
            raise ValueError("structure_feature_dimension must be non-negative")
        if self.structure_fusion not in {"global_gate", "residue_gate"}:
            raise ValueError("structure_fusion must be global_gate or residue_gate")
        if self.structure_feature_dimension == 0 and (
            self.structure_fusion != "global_gate" or self.structure_mask_token
        ):
            raise ValueError("structure fusion options require structure features")
        if not 0 <= self.contrastive_text_layers <= self.num_bridge_layers:
            raise ValueError("contrastive_text_layers must be in [0, num_bridge_layers]")
        if self.contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be positive")
        if self.multi_positive_contrastive and self.contrastive_text_layers == 0:
            raise ValueError(
                "multi-positive contrastive learning requires contextual text layers"
            )
        if self.protein_backbone.kind in {"esm2", "esmc"} and not self.protein_backbone.model_path:
            raise ValueError("an ESM2/ESMC backbone requires model_path")

    def parameter_breakdown(self) -> Dict[str, int]:
        """Return the exact count implied by this repository's modules.

        The external backbone count comes from its immutable checkpoint index.
        The formula is checked against instantiated tiny models in unit tests and
        against the loaded 7B model before any production run.
        """

        d = self.hidden_size
        i = self.intermediate_size
        per_bridge_layer = 4 * d * d + 3 * d * i + 2 * d
        text_output = 0 if self.tie_text_embeddings else self.text_vocab_size * d
        structure_adapter = 0
        if self.structure_feature_dimension:
            feature_dim = self.structure_feature_dimension
            structure_adapter = d * d + 2 * d * feature_dim + d + feature_dim
            if self.structure_fusion == "residue_gate":
                structure_adapter += d * feature_dim
            if self.structure_mask_token:
                structure_adapter += d
        return {
            "protein_backbone": self.protein_backbone.parameter_count,
            "bridge": self.num_bridge_layers * per_bridge_layer,
            "text_embedding": self.text_vocab_size * d,
            "text_output": text_output,
            "modality_embedding": 2 * d,
            "final_norm": d,
            "contrastive_heads": 2 * d * self.contrastive_dimension,
            "geometry_head": (
                2 * d * self.geometry_rank
                + self.geometry_rank * self.distance_bins
                + self.distance_bins
            ),
            "structure_adapter": structure_adapter,
            "contrastive_temperature": int(self.contrastive_text_layers > 0),
        }

    @property
    def estimated_parameter_count(self) -> int:
        return sum(self.parameter_breakdown().values())

    @property
    def is_target_scale(self) -> bool:
        return TARGET_MIN_PARAMETERS <= self.estimated_parameter_count <= TARGET_MAX_PARAMETERS


def load_model_config(path: str) -> ModelConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    config = ModelConfig.from_dict(raw)
    config.validate()
    return config
