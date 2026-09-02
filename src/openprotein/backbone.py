"""Protein-backbone adapters.

The production adapter imports Biohub ESM lazily so tiny tests do not require
the 25.4 GB checkpoint or its full runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
import importlib.machinery
import sys
import types
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from openprotein.config import ModelConfig


class ProteinBackbone(nn.Module, ABC):
    hidden_size: int
    vocab_size: int

    @abstractmethod
    def forward_hidden(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TinyProteinBackbone(ProteinBackbone):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = config.protein_backbone.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.output = nn.Linear(self.hidden_size, self.vocab_size)

    def forward_hidden(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        return self.encoder(hidden, src_key_padding_mask=~attention_mask.to(torch.bool))

    def project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(hidden)


class EsmcProteinBackbone(ProteinBackbone):
    def __init__(
        self,
        config: ModelConfig,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if not config.protein_backbone.model_path:
            raise ValueError("ESMC model_path is required")
        try:
            EsmcForMaskedLM = _load_native_esmc_class()
        except ImportError as exc:
            raise RuntimeError(
                "The production ESMC adapter requires the pinned Biohub ESM runtime"
            ) from exc

        self.hidden_size = config.hidden_size
        self.vocab_size = config.protein_backbone.vocab_size
        load_device = device if device is not None else torch.device("cpu")
        self.model = EsmcForMaskedLM.from_pretrained(
            config.protein_backbone.model_path,
            device=load_device,
            dtype=dtype,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        if config.protein_backbone.freeze:
            self.model.requires_grad_(False)

    def forward_hidden(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        output = self.model.esmc(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            compute_sae=False,
        )
        return output.last_hidden_state

    def project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(hidden)


class Esm2ProteinBackbone(ProteinBackbone):
    """Load Meta's ESM-2 checkpoint through its official Transformers adapter.

    This deliberately avoids importing ``fair-esm`` because the production
    runtime already contains EvolutionaryScale's different top-level ``esm``
    package.  The immutable local directory is verified before model creation.
    """

    def __init__(
        self,
        config: ModelConfig,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if not config.protein_backbone.model_path:
            raise ValueError("ESM2 model_path is required")
        try:
            from transformers import EsmForMaskedLM
        except ImportError as exc:
            raise RuntimeError("the ESM2 adapter requires transformers") from exc

        load_device = device if device is not None else torch.device("cpu")
        self.model = EsmForMaskedLM.from_pretrained(
            config.protein_backbone.model_path,
            local_files_only=True,
            torch_dtype=dtype,
        )
        self.model.to(load_device)
        self.hidden_size = int(self.model.config.hidden_size)
        self.vocab_size = int(self.model.config.vocab_size)
        if self.hidden_size != config.hidden_size:
            raise ValueError("loaded ESM2 hidden size differs from model config")
        if self.vocab_size != config.protein_backbone.vocab_size:
            raise ValueError("loaded ESM2 vocabulary differs from model config")
        if config.protein_backbone.freeze:
            self.model.requires_grad_(False)
            self.model.eval()
        self.freeze = config.protein_backbone.freeze

    def train(self, mode: bool = True) -> "Esm2ProteinBackbone":
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward_hidden(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        output = self.model.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return output.last_hidden_state

    def project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(hidden)


def _load_native_esmc_class():
    """Import Biohub's native ESMC model without importing its full SDK.

    The upstream ``esm.models.esmc.__init__`` also imports compatibility APIs,
    protein-structure parsing, Pandas, and other inference-only surfaces. The
    pretraining adapter needs only the native model package. Registering a
    namespace package at the same pinned source path lets normal relative
    imports load ``model.py``, ``layers.py``, and ``kernels.py`` while avoiding
    those unrelated side effects. No model source or checkpoint key is changed.
    """

    import esm

    package_name = "esm.models.esmc"
    if package_name not in sys.modules:
        package_path = Path(esm.__file__).resolve().parent / "models" / "esmc"
        if not package_path.is_dir():
            raise ImportError(f"pinned ESM source is missing {package_path}")
        package = types.ModuleType(package_name)
        package.__file__ = str(package_path / "__init__.py")
        package.__package__ = package_name
        package.__path__ = [str(package_path)]
        package.__spec__ = importlib.machinery.ModuleSpec(
            package_name, loader=None, is_package=True
        )
        sys.modules[package_name] = package
    module = importlib.import_module("esm.models.esmc.model")
    return module.EsmcForMaskedLM


def build_protein_backbone(
    config: ModelConfig,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> ProteinBackbone:
    if config.protein_backbone.kind == "tiny":
        return TinyProteinBackbone(config)
    if config.protein_backbone.kind == "esm2":
        return Esm2ProteinBackbone(config, device=device, dtype=dtype)
    if config.protein_backbone.kind == "esmc":
        return EsmcProteinBackbone(config, device=device, dtype=dtype)
    raise ValueError(f"unsupported protein backbone: {config.protein_backbone.kind}")
