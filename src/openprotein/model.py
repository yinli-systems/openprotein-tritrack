"""OpenProtein multimodal model and task heads."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from openprotein.attention import BridgeBlock, build_bidirectional_mask, build_prefix_lm_mask
from openprotein.backbone import ProteinBackbone, build_protein_backbone
from openprotein.config import ModelConfig


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denominator


class SampledGeometryHead(nn.Module):
    def __init__(self, hidden_size: int, rank: int, distance_bins: int) -> None:
        super().__init__()
        self.left = nn.Linear(hidden_size, rank, bias=False)
        self.right = nn.Linear(hidden_size, rank, bias=False)
        self.classifier = nn.Linear(rank, distance_bins)

    def forward(self, hidden: torch.Tensor, pair_indices: torch.Tensor) -> torch.Tensor:
        if pair_indices.ndim != 3 or pair_indices.shape[-1] != 2:
            raise ValueError("pair_indices must have shape [batch, pairs, 2]")
        batch, pairs, _ = pair_indices.shape
        if pair_indices.numel() and (
            pair_indices.min().item() < 0 or pair_indices.max().item() >= hidden.shape[1]
        ):
            raise ValueError("pair_indices contains an out-of-range residue index")
        batch_index = torch.arange(batch, device=hidden.device)[:, None].expand(batch, pairs)
        left_hidden = hidden[batch_index, pair_indices[..., 0]]
        right_hidden = hidden[batch_index, pair_indices[..., 1]]
        pair_features = self.left(left_hidden) * self.right(right_hidden)
        return self.classifier(pair_features)


class OpenProteinModel(nn.Module):
    def __init__(self, config: ModelConfig, protein_backbone: ProteinBackbone) -> None:
        super().__init__()
        config.validate()
        if protein_backbone.hidden_size != config.hidden_size:
            raise ValueError("protein backbone hidden size does not match bridge")
        if protein_backbone.vocab_size != config.protein_backbone.vocab_size:
            raise ValueError("protein backbone vocabulary does not match config")

        self.config = config
        self.protein_backbone = protein_backbone
        self.text_embedding = nn.Embedding(config.text_vocab_size, config.hidden_size)
        # nn.Embedding defaults to N(0, 1). Preserve that checkpoint
        # parameterization while applying the Transformer-scale factor at use
        # sites; this keeps tied output logits O(1) instead of O(sqrt(d)).
        self.text_embedding_scale = config.hidden_size**-0.5
        self.modality_embedding = nn.Embedding(2, config.hidden_size)
        self.bridge = nn.ModuleList(
            [
                BridgeBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    dropout=config.dropout,
                    residual_scale=config.residual_scale,
                )
                for _ in range(config.num_bridge_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(config.hidden_size)
        self.protein_projection = nn.Linear(
            config.hidden_size, config.contrastive_dimension, bias=False
        )
        self.text_projection = nn.Linear(
            config.hidden_size, config.contrastive_dimension, bias=False
        )
        self.geometry_head = SampledGeometryHead(
            config.hidden_size, config.geometry_rank, config.distance_bins
        )
        self.structure_norm: Optional[nn.RMSNorm]
        self.structure_projection: Optional[nn.Sequential]
        self.structure_gate: Optional[nn.Parameter]
        self.structure_gate_projection: Optional[nn.Linear]
        self.structure_mask_embedding: Optional[nn.Parameter]
        self.structure_output: Optional[nn.Linear]
        if config.structure_feature_dimension:
            feature_dim = config.structure_feature_dimension
            self.structure_norm = nn.RMSNorm(feature_dim)
            self.structure_projection = nn.Sequential(
                nn.Linear(feature_dim, config.hidden_size, bias=False),
                nn.SiLU(),
                nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            )
            if config.structure_fusion == "global_gate":
                self.structure_gate = nn.Parameter(torch.zeros(config.hidden_size))
                self.structure_gate_projection = None
            else:
                self.register_parameter("structure_gate", None)
                self.structure_gate_projection = nn.Linear(
                    feature_dim, config.hidden_size, bias=True
                )
                nn.init.zeros_(self.structure_gate_projection.weight)
                nn.init.zeros_(self.structure_gate_projection.bias)
            if config.structure_mask_token:
                self.structure_mask_embedding = nn.Parameter(
                    torch.zeros(config.hidden_size)
                )
            else:
                self.register_parameter("structure_mask_embedding", None)
            self.structure_output = nn.Linear(config.hidden_size, feature_dim, bias=False)
        else:
            self.structure_norm = None
            self.structure_projection = None
            self.register_parameter("structure_gate", None)
            self.structure_gate_projection = None
            self.register_parameter("structure_mask_embedding", None)
            self.structure_output = None
        if config.contrastive_text_layers:
            self.contrastive_logit_scale = nn.Parameter(
                torch.tensor(math.log(1.0 / config.contrastive_temperature))
            )
        else:
            self.register_parameter("contrastive_logit_scale", None)
        self.text_output: Optional[nn.Linear]
        if config.tie_text_embeddings:
            self.text_output = None
        else:
            self.text_output = nn.Linear(config.hidden_size, config.text_vocab_size, bias=False)

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "OpenProteinModel":
        backbone = build_protein_backbone(config, device=device, dtype=dtype)
        return cls(config, backbone)

    def _text_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.text_output is not None:
            return self.text_output(hidden)
        return F.linear(hidden, self.text_embedding.weight) * self.text_embedding_scale

    def _embed_text(self, text_input_ids: torch.Tensor) -> torch.Tensor:
        text_types = torch.ones_like(text_input_ids)
        return (
            self.text_embedding(text_input_ids) + self.modality_embedding(text_types)
        ) * self.text_embedding_scale

    def _contextualize_text(
        self, text_input_ids: torch.Tensor, text_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = self._embed_text(text_input_ids)
        allowed = text_mask[:, :, None] & text_mask[:, None, :]
        for block in self.bridge[: self.config.contrastive_text_layers]:
            hidden = block(hidden, allowed)
        return self.final_norm(hidden)

    def _fuse(
        self,
        protein_hidden: torch.Tensor,
        protein_mask: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_mask: torch.Tensor,
        attention_policy: str,
        structure_input_features: Optional[torch.Tensor] = None,
        structure_attention_mask: Optional[torch.Tensor] = None,
        structure_label_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, protein_length, _ = protein_hidden.shape
        if structure_input_features is not None:
            if self.structure_projection is None or self.structure_norm is None:
                raise ValueError("structure inputs require a configured structure adapter")
            if structure_input_features.shape[:2] != protein_hidden.shape[:2]:
                raise ValueError("structure features must align with protein tokens")
            if structure_input_features.shape[-1] != self.config.structure_feature_dimension:
                raise ValueError("structure feature dimension differs from model config")
            if structure_attention_mask is None:
                raise ValueError("structure inputs require structure_attention_mask")
            structure_mask = structure_attention_mask.to(dtype=torch.bool)
            if structure_mask.shape != protein_mask.shape:
                raise ValueError("structure mask must align with protein tokens")
            normalized_structure = self.structure_norm(
                structure_input_features.to(dtype=protein_hidden.dtype)
            )
            projected_structure = self.structure_projection(normalized_structure)
            projected_structure = projected_structure.masked_fill(
                ~structure_mask.unsqueeze(-1), 0.0
            )
            if self.structure_gate_projection is not None:
                structure_gate = torch.tanh(
                    self.structure_gate_projection(normalized_structure)
                )
            else:
                structure_gate = torch.tanh(self.structure_gate)
            structure_hidden = structure_gate * projected_structure
            if self.structure_mask_embedding is not None and structure_label_mask is not None:
                if structure_label_mask.shape != structure_mask.shape:
                    raise ValueError("structure label mask must align with protein tokens")
                masked_positions = structure_label_mask.to(dtype=torch.bool) & structure_mask
                structure_hidden = structure_hidden + (
                    masked_positions.unsqueeze(-1) * self.structure_mask_embedding
                )
            protein_hidden = protein_hidden + structure_hidden

        text_hidden = self._embed_text(text_input_ids)
        protein_types = torch.zeros(
            (batch, protein_length), dtype=torch.long, device=protein_hidden.device
        )
        protein_hidden = protein_hidden + self.modality_embedding(protein_types)
        hidden = torch.cat([protein_hidden, text_hidden], dim=1)

        if attention_policy == "bidirectional":
            allowed = build_bidirectional_mask(protein_mask, text_mask)
        elif attention_policy == "prefix_lm":
            allowed = build_prefix_lm_mask(protein_mask, text_mask)
        else:
            raise ValueError("attention_policy must be 'bidirectional' or 'prefix_lm'")

        for block in self.bridge:
            hidden = block(hidden, allowed)
        return self.final_norm(hidden)

    def forward(
        self,
        protein_input_ids: torch.Tensor,
        protein_attention_mask: torch.Tensor,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        protein_labels: Optional[torch.Tensor] = None,
        text_labels: Optional[torch.Tensor] = None,
        pair_indices: Optional[torch.Tensor] = None,
        distance_labels: Optional[torch.Tensor] = None,
        structure_input_features: Optional[torch.Tensor] = None,
        structure_attention_mask: Optional[torch.Tensor] = None,
        structure_labels: Optional[torch.Tensor] = None,
        structure_label_mask: Optional[torch.Tensor] = None,
        contrastive_group_ids: Optional[torch.Tensor] = None,
        attention_policy: str = "bidirectional",
        use_bridge: bool = True,
        loss_weights: Optional[Dict[str, float]] = None,
        detach_geometry_backbone: bool = False,
    ) -> Dict[str, object]:
        if protein_input_ids.ndim != 2:
            raise ValueError("protein_input_ids must have shape [batch, length]")
        protein_mask = protein_attention_mask.to(dtype=torch.bool)
        protein_hidden = self.protein_backbone.forward_hidden(
            protein_input_ids, protein_attention_mask
        )
        base_protein_logits = self.protein_backbone.project_logits(protein_hidden)

        batch = protein_input_ids.shape[0]
        if text_input_ids is None:
            text_input_ids = torch.empty(
                (batch, 0), dtype=torch.long, device=protein_input_ids.device
            )
        if text_attention_mask is None:
            text_attention_mask = torch.ones_like(text_input_ids, dtype=torch.bool)
        text_mask = text_attention_mask.to(dtype=torch.bool)

        if use_bridge:
            fused = self._fuse(
                protein_hidden,
                protein_mask,
                text_input_ids,
                text_mask,
                attention_policy,
                structure_input_features=structure_input_features,
                structure_attention_mask=structure_attention_mask,
                structure_label_mask=structure_label_mask,
            )
            protein_length = protein_input_ids.shape[1]
            bridged_protein = fused[:, :protein_length]
            bridged_text = fused[:, protein_length:]
            protein_logits = self.protein_backbone.project_logits(bridged_protein)
            text_logits = self._text_logits(bridged_text)
        else:
            if text_input_ids.shape[1]:
                raise ValueError("text inputs require use_bridge=True")
            bridged_protein = protein_hidden
            bridged_text = protein_hidden[:, :0]
            protein_logits = base_protein_logits
            text_logits = protein_hidden.new_empty((batch, 0, self.config.text_vocab_size))

        losses: Dict[str, torch.Tensor] = {}
        if protein_labels is not None:
            losses["protein_mlm"] = F.cross_entropy(
                protein_logits.reshape(-1, protein_logits.shape[-1]),
                protein_labels.reshape(-1),
                ignore_index=-100,
            )

        if text_labels is not None:
            if attention_policy != "prefix_lm":
                raise ValueError("text LM labels require prefix_lm to prevent target leakage")
            if text_logits.shape[1] < 2:
                raise ValueError("text LM loss requires at least two text tokens")
            losses["text_lm"] = F.cross_entropy(
                text_logits[:, :-1].reshape(-1, text_logits.shape[-1]),
                text_labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )

        contrastive_logits: Optional[torch.Tensor] = None
        protein_embedding: Optional[torch.Tensor] = None
        text_embedding: Optional[torch.Tensor] = None
        if bridged_text.shape[1] and attention_policy == "prefix_lm":
            protein_embedding = F.normalize(
                self.protein_projection(_masked_mean(bridged_protein, protein_mask)), dim=-1
            )
            isolated_text_hidden = (
                self._contextualize_text(text_input_ids, text_mask)
                if self.config.contrastive_text_layers
                else self._embed_text(text_input_ids)
            )
            text_embedding = F.normalize(
                self.text_projection(_masked_mean(isolated_text_hidden, text_mask)), dim=-1
            )
        if protein_embedding is not None and text_embedding is not None and batch > 1:
            all_protein = protein_embedding
            all_text = text_embedding
            target_offset = 0
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                from torch.distributed.nn.functional import all_gather

                all_protein = torch.cat(all_gather(protein_embedding), dim=0)
                all_text = torch.cat(all_gather(text_embedding), dim=0)
                target_offset = torch.distributed.get_rank() * batch
            scale = (
                self.contrastive_logit_scale.exp().clamp(max=100.0)
                if self.contrastive_logit_scale is not None
                else protein_embedding.new_tensor(1.0 / 0.07)
            )
            contrastive_logits = scale * protein_embedding @ all_text.transpose(0, 1)
            reverse_logits = scale * text_embedding @ all_protein.transpose(0, 1)
            if self.config.multi_positive_contrastive:
                if contrastive_group_ids is None or contrastive_group_ids.shape != (batch,):
                    raise ValueError(
                        "multi-positive contrastive learning requires one group id per example"
                    )
                local_groups = contrastive_group_ids.to(dtype=torch.long)
                all_groups = local_groups
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    gathered_groups = [torch.empty_like(local_groups) for _ in range(
                        torch.distributed.get_world_size()
                    )]
                    torch.distributed.all_gather(gathered_groups, local_groups)
                    all_groups = torch.cat(gathered_groups, dim=0)
                positive = local_groups[:, None].eq(all_groups[None, :])
                if not positive.any(dim=1).all():
                    raise RuntimeError("contrastive batch contains a query with no positive")

                def multi_positive_nce(logits: torch.Tensor) -> torch.Tensor:
                    positive_logits = logits.masked_fill(~positive, float("-inf"))
                    return -(
                        torch.logsumexp(positive_logits, dim=-1)
                        - torch.logsumexp(logits, dim=-1)
                    ).mean()

                losses["contrastive"] = 0.5 * (
                    multi_positive_nce(contrastive_logits)
                    + multi_positive_nce(reverse_logits)
                )
            else:
                targets = target_offset + torch.arange(
                    batch, device=contrastive_logits.device
                )
                losses["contrastive"] = 0.5 * (
                    F.cross_entropy(contrastive_logits, targets)
                    + F.cross_entropy(reverse_logits, targets)
                )

        distance_logits: Optional[torch.Tensor] = None
        if pair_indices is not None:
            geometry_hidden = (
                bridged_protein.detach() if detach_geometry_backbone else bridged_protein
            )
            distance_logits = self.geometry_head(geometry_hidden, pair_indices)
            if distance_labels is not None:
                losses["geometry"] = F.cross_entropy(
                    distance_logits.reshape(-1, distance_logits.shape[-1]),
                    distance_labels.reshape(-1),
                    ignore_index=-100,
                )

        structure_predictions: Optional[torch.Tensor] = None
        if structure_labels is not None:
            if self.structure_output is None:
                raise ValueError("structure labels require a configured structure adapter")
            if structure_labels.shape[:2] != bridged_protein.shape[:2]:
                raise ValueError("structure labels must align with protein tokens")
            if structure_label_mask is None:
                raise ValueError("structure labels require structure_label_mask")
            label_mask = structure_label_mask.to(dtype=torch.bool)
            if label_mask.shape != structure_labels.shape[:2] or not label_mask.any():
                raise ValueError("structure_label_mask must select at least one aligned token")
            structure_predictions = self.structure_output(bridged_protein)
            losses["structure_denoising"] = F.smooth_l1_loss(
                structure_predictions[label_mask],
                structure_labels[label_mask].to(dtype=structure_predictions.dtype),
            )

        weights = loss_weights or {}
        missing_weighted = {
            name for name, weight in weights.items() if float(weight) != 0.0 and name not in losses
        }
        if missing_weighted:
            raise ValueError(
                "requested weighted objectives were not produced: "
                + ", ".join(sorted(missing_weighted))
            )
        total_loss: Optional[torch.Tensor] = None
        for name, value in losses.items():
            weight = float(weights.get(name, 1.0))
            if weight == 0.0:
                continue
            weighted = weight * value
            total_loss = weighted if total_loss is None else total_loss + weighted

        return {
            "loss": total_loss,
            "losses": losses,
            "base_protein_logits": base_protein_logits,
            "protein_logits": protein_logits,
            "text_logits": text_logits,
            "contrastive_logits": contrastive_logits,
            "protein_embedding": protein_embedding,
            "text_embedding": text_embedding,
            "distance_logits": distance_logits,
            "structure_predictions": structure_predictions,
        }
