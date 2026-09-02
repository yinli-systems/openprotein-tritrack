"""Minimal, pinned ESMC protein tokenizer for high-throughput preprocessing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch


@dataclass(frozen=True)
class EsmcTokenizer:
    vocabulary: Dict[str, int]

    @classmethod
    def from_tokenizer_json(cls, path: str) -> "EsmcTokenizer":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        vocabulary = raw.get("model", {}).get("vocab")
        if not isinstance(vocabulary, dict):
            raise ValueError("tokenizer JSON is missing model.vocab")
        required = {"<cls>", "<pad>", "<eos>", "<unk>", "<mask>"}
        missing = required - set(vocabulary)
        if missing:
            raise ValueError(f"tokenizer is missing special tokens: {sorted(missing)}")
        return cls({str(token): int(index) for token, index in vocabulary.items()})

    @property
    def cls_token_id(self) -> int:
        return self.vocabulary["<cls>"]

    @property
    def pad_token_id(self) -> int:
        return self.vocabulary["<pad>"]

    @property
    def eos_token_id(self) -> int:
        return self.vocabulary["<eos>"]

    @property
    def unk_token_id(self) -> int:
        return self.vocabulary["<unk>"]

    @property
    def mask_token_id(self) -> int:
        return self.vocabulary["<mask>"]

    @property
    def special_token_ids(self) -> List[int]:
        return [
            index
            for token, index in self.vocabulary.items()
            if token.startswith("<") and token.endswith(">")
        ]

    @property
    def amino_acid_token_ids(self) -> List[int]:
        return [self.vocabulary[token] for token in "ACDEFGHIKLMNPQRSTVWY" if token in self.vocabulary]

    def encode(
        self,
        sequence: str,
        max_length: Optional[int] = None,
        crop_start: int = 0,
    ) -> List[int]:
        sequence = sequence.strip().upper()
        if max_length is not None:
            if max_length < 2:
                raise ValueError("max_length must leave room for <cls> and <eos>")
            content_length = max_length - 2
            if crop_start < 0 or crop_start > max(0, len(sequence) - content_length):
                raise ValueError("crop_start is outside the valid sequence range")
            sequence = sequence[crop_start : crop_start + content_length]
        ids = [self.cls_token_id]
        ids.extend(self.vocabulary.get(character, self.unk_token_id) for character in sequence)
        ids.append(self.eos_token_id)
        return ids

    def pad(self, examples: Iterable[List[int]]) -> Dict[str, torch.Tensor]:
        rows = list(examples)
        if not rows:
            raise ValueError("cannot pad an empty batch")
        width = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}
