"""Deterministic streaming readers for protein CSV pilot data."""

from __future__ import annotations

import csv
import hashlib
import random
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set

import torch
from torch.utils.data import IterableDataset, get_worker_info

from openprotein.data.masking import mask_protein_tokens
from openprotein.data.structure import (
    STRUCTURE_FEATURE_DIMENSION,
    ca_trace_features,
    sample_geometry_pairs,
)
from openprotein.data.text_tokenizer import SentencePieceTextTokenizer
from openprotein.data.tokenizer import EsmcTokenizer


def _contrastive_group_id(text_group: object) -> int:
    normalized = " ".join(str(text_group).casefold().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


class CSVProteinStream(IterableDataset):
    """Stream CSV records with deterministic file/rank/worker ownership."""

    def __init__(
        self,
        paths: Iterable[str],
        tokenizer: EsmcTokenizer,
        max_length: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle_files: bool = True,
    ) -> None:
        super().__init__()
        self.paths = [str(Path(path)) for path in paths]
        if not self.paths:
            raise ValueError("CSVProteinStream requires at least one path")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.shuffle_files = shuffle_files
        self.epoch = 0
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _owned_paths(self) -> List[str]:
        paths = list(self.paths)
        if self.shuffle_files:
            random.Random(self.seed + self.epoch).shuffle(paths)
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        owner = self.rank * workers + worker_id
        owners = self.world_size * workers
        return [path for index, path in enumerate(paths) if index % owners == owner]

    def __iter__(self) -> Iterator[Dict[str, object]]:
        crop_rng = random.Random(self.seed + 1_000_003 * self.epoch + self.rank)
        for path in self._owned_paths():
            with Path(path).open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or "text" not in reader.fieldnames:
                    raise ValueError(f"{path} does not contain a text column")
                for row_index, row in enumerate(reader):
                    sequence = row["text"].strip().upper()
                    if not sequence:
                        raise ValueError(f"{path}:{row_index + 2} contains an empty sequence")
                    content_length = self.max_length - 2
                    max_start = max(0, len(sequence) - content_length)
                    crop_start = crop_rng.randint(0, max_start) if max_start else 0
                    yield {
                        "input_ids": self.tokenizer.encode(
                            sequence, max_length=self.max_length, crop_start=crop_start
                        ),
                        "record_id": row.get("name") or row.get("id") or f"{path}:{row_index}",
                    }


class ProteinMLMCollator:
    def __init__(
        self,
        tokenizer: EsmcTokenizer,
        seed: int,
        mlm_probability: float = 0.15,
    ) -> None:
        self.tokenizer = tokenizer
        self.generator = torch.Generator().manual_seed(seed)
        self.mlm_probability = mlm_probability

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"generator_state": self.generator.get_state()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.generator.set_state(state["generator_state"])

    def __call__(self, examples: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        padded = self.tokenizer.pad([example["input_ids"] for example in examples])
        corrupted, labels = mask_protein_tokens(
            padded["input_ids"],
            padded["attention_mask"],
            mask_token_id=self.tokenizer.mask_token_id,
            vocabulary_size=max(self.tokenizer.vocabulary.values()) + 1,
            special_token_ids=self.tokenizer.special_token_ids,
            random_token_ids=self.tokenizer.amino_acid_token_ids,
            generator=self.generator,
            mlm_probability=self.mlm_probability,
        )
        if not labels.ne(-100).any():
            raise ValueError("batch contains no eligible protein tokens for MLM")
        return {
            "protein_input_ids": corrupted,
            "protein_attention_mask": padded["attention_mask"],
            "protein_labels": labels,
        }


class MultimodalCSVStream(IterableDataset):
    """Stream exact sequence-text pairs while preserving inherited shard ownership."""

    def __init__(
        self,
        paths: Iterable[str],
        protein_tokenizer: EsmcTokenizer,
        text_tokenizer: SentencePieceTextTokenizer,
        protein_max_length: int,
        text_max_length: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle_files: bool = True,
        include_record_ids: Optional[Set[str]] = None,
    ) -> None:
        super().__init__()
        self.paths = [str(Path(path)) for path in paths]
        if not self.paths:
            raise ValueError("MultimodalCSVStream requires at least one path")
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.protein_max_length = protein_max_length
        self.text_max_length = text_max_length
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.shuffle_files = shuffle_files
        self.include_record_ids = (
            frozenset(include_record_ids) if include_record_ids is not None else None
        )
        self.epoch = 0
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _owned_paths(self) -> List[str]:
        paths = list(self.paths)
        if self.shuffle_files:
            random.Random(self.seed + self.epoch).shuffle(paths)
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        owner = self.rank * workers + worker_id
        owners = self.world_size * workers
        return [path for index, path in enumerate(paths) if index % owners == owner]

    def __iter__(self) -> Iterator[Dict[str, object]]:
        crop_rng = random.Random(self.seed + 1_000_003 * self.epoch + self.rank)
        for path in self._owned_paths():
            with Path(path).open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"id", "sequence", "text"}
                if reader.fieldnames is None or not required <= set(reader.fieldnames):
                    raise ValueError(f"{path} lacks multimodal columns: {sorted(required)}")
                for row_index, row in enumerate(reader):
                    sequence = row["sequence"].strip().upper()
                    text = row["text"].strip()
                    if not sequence or not text:
                        raise ValueError(f"{path}:{row_index + 2} contains an empty modality")
                    content_length = self.protein_max_length - 2
                    max_start = max(0, len(sequence) - content_length)
                    crop_start = crop_rng.randint(0, max_start) if max_start else 0
                    if (
                        self.include_record_ids is not None
                        and row["id"] not in self.include_record_ids
                    ):
                        continue
                    yield {
                        "protein_input_ids": self.protein_tokenizer.encode(
                            sequence,
                            max_length=self.protein_max_length,
                            crop_start=crop_start,
                        ),
                        "text_input_ids": self.text_tokenizer.encode(
                            text, max_length=self.text_max_length
                        ),
                        "record_id": row["id"],
                        "sequence_length": len(sequence),
                        "text_group": " ".join(text.casefold().split()),
                    }


class SequenceTextCollator:
    def __init__(
        self,
        protein_tokenizer: EsmcTokenizer,
        text_tokenizer: SentencePieceTextTokenizer,
        seed: int,
        mlm_probability: float = 0.15,
    ) -> None:
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.generator = torch.Generator().manual_seed(seed)
        self.mlm_probability = mlm_probability

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"generator_state": self.generator.get_state()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.generator.set_state(state["generator_state"])

    def __call__(self, examples: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        protein = self.protein_tokenizer.pad(
            [example["protein_input_ids"] for example in examples]
        )
        corrupted, protein_labels = mask_protein_tokens(
            protein["input_ids"],
            protein["attention_mask"],
            mask_token_id=self.protein_tokenizer.mask_token_id,
            vocabulary_size=max(self.protein_tokenizer.vocabulary.values()) + 1,
            special_token_ids=self.protein_tokenizer.special_token_ids,
            random_token_ids=self.protein_tokenizer.amino_acid_token_ids,
            generator=self.generator,
            mlm_probability=self.mlm_probability,
        )
        text = self.text_tokenizer.pad([example["text_input_ids"] for example in examples])
        text_labels = text["input_ids"].clone()
        text_labels.masked_fill_(~text["attention_mask"].to(torch.bool), -100)
        group_ids = [_contrastive_group_id(example["text_group"]) for example in examples]
        return {
            "protein_input_ids": corrupted,
            "protein_attention_mask": protein["attention_mask"],
            "protein_labels": protein_labels,
            "text_input_ids": text["input_ids"],
            "text_attention_mask": text["attention_mask"],
            "text_labels": text_labels,
            "contrastive_group_ids": torch.tensor(group_ids, dtype=torch.long),
        }


class StructureCSVStream(IterableDataset):
    """Read inherited sequence splits and attach exact full-length AFDB coordinates."""

    def __init__(
        self,
        paths: Iterable[str],
        protein_tokenizer: EsmcTokenizer,
        structure_database: str,
        max_length: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle_files: bool = True,
        min_confidence: float = 70.0,
        include_record_ids: Optional[Set[str]] = None,
    ) -> None:
        super().__init__()
        self.paths = [str(Path(path)) for path in paths]
        if not self.paths:
            raise ValueError("StructureCSVStream requires at least one path")
        self.protein_tokenizer = protein_tokenizer
        self.structure_database = str(Path(structure_database).resolve())
        self.max_length = max_length
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.shuffle_files = shuffle_files
        self.min_confidence = min_confidence
        self.include_record_ids = (
            frozenset(include_record_ids) if include_record_ids is not None else None
        )
        self.epoch = 0
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _owned_paths(self) -> List[str]:
        paths = list(self.paths)
        if self.shuffle_files:
            random.Random(self.seed + self.epoch).shuffle(paths)
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        owner = self.rank * workers + worker_id
        owners = self.world_size * workers
        return [path for index, path in enumerate(paths) if index % owners == owner]

    def __iter__(self) -> Iterator[Dict[str, object]]:
        crop_rng = random.Random(self.seed + 1_000_003 * self.epoch + self.rank)
        uri = f"file:{self.structure_database}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            for path in self._owned_paths():
                with Path(path).open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    required = {"id", "accession", "sequence"}
                    if reader.fieldnames is None or not required <= set(reader.fieldnames):
                        raise ValueError(f"{path} lacks structure-join columns")
                    for row in reader:
                        structure = connection.execute(
                            """
                            SELECT sequence, coordinates_f32le, confidence_f32le
                            FROM structures WHERE accession = ? AND fragment = 1
                            """,
                            (row["accession"],),
                        ).fetchone()
                        if structure is None:
                            continue
                        expected_sequence, coordinate_blob, confidence_blob = structure
                        sequence = row["sequence"].strip().upper()
                        if sequence != expected_sequence:
                            continue
                        coordinates = torch.frombuffer(
                            bytearray(coordinate_blob), dtype=torch.float32
                        ).reshape(-1, 3)
                        confidence = torch.frombuffer(
                            bytearray(confidence_blob), dtype=torch.float32
                        )
                        if coordinates.shape[0] != len(sequence) or confidence.shape[0] != len(
                            sequence
                        ):
                            raise RuntimeError(
                                "structure index row has inconsistent residue counts"
                            )
                        content_length = self.max_length - 2
                        max_start = max(0, len(sequence) - content_length)
                        crop_start = crop_rng.randint(0, max_start) if max_start else 0
                        crop_stop = crop_start + content_length
                        cropped_coordinates = coordinates[crop_start:crop_stop].clone()
                        cropped_confidence = confidence[crop_start:crop_stop].clone()
                        eligible = torch.nonzero(
                            cropped_confidence.ge(self.min_confidence), as_tuple=False
                        ).flatten()
                        if eligible.numel() < 2 or int(eligible[-1] - eligible[0]) < 3:
                            continue
                        if (
                            self.include_record_ids is not None
                            and row["id"] not in self.include_record_ids
                        ):
                            continue
                        yield {
                            "protein_input_ids": self.protein_tokenizer.encode(
                                sequence,
                                max_length=self.max_length,
                                crop_start=crop_start,
                            ),
                            "coordinates": cropped_coordinates,
                            "confidence": cropped_confidence,
                            "record_id": row["id"],
                        }


class SequenceStructureCollator:
    def __init__(
        self,
        protein_tokenizer: EsmcTokenizer,
        seed: int,
        geometry_pairs: int,
        distance_bins: int,
        mlm_probability: float = 0.15,
        min_confidence: float = 70.0,
    ) -> None:
        self.protein_tokenizer = protein_tokenizer
        self.generator = torch.Generator().manual_seed(seed)
        self.geometry_pairs = geometry_pairs
        self.distance_bins = distance_bins
        self.mlm_probability = mlm_probability
        self.min_confidence = min_confidence

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"generator_state": self.generator.get_state()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.generator.set_state(state["generator_state"])

    def __call__(self, examples: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        protein = self.protein_tokenizer.pad(
            [example["protein_input_ids"] for example in examples]
        )
        corrupted, protein_labels = mask_protein_tokens(
            protein["input_ids"],
            protein["attention_mask"],
            mask_token_id=self.protein_tokenizer.mask_token_id,
            vocabulary_size=max(self.protein_tokenizer.vocabulary.values()) + 1,
            special_token_ids=self.protein_tokenizer.special_token_ids,
            random_token_ids=self.protein_tokenizer.amino_acid_token_ids,
            generator=self.generator,
            mlm_probability=self.mlm_probability,
        )
        batch_pairs = torch.zeros(
            (len(examples), self.geometry_pairs, 2), dtype=torch.long
        )
        batch_labels = torch.full(
            (len(examples), self.geometry_pairs), -100, dtype=torch.long
        )
        for index, example in enumerate(examples):
            pairs, labels = sample_geometry_pairs(
                example["coordinates"],
                example["confidence"],
                self.geometry_pairs,
                self.generator,
                min_confidence=self.min_confidence,
                distance_bins=self.distance_bins,
            )
            if not len(pairs):
                raise ValueError("structure example has no eligible high-confidence residue pairs")
            count = len(pairs)
            batch_pairs[index, :count] = pairs + 1  # account for the protein CLS token
            batch_labels[index, :count] = labels
        return {
            "protein_input_ids": corrupted,
            "protein_attention_mask": protein["attention_mask"],
            "protein_labels": protein_labels,
            "pair_indices": batch_pairs,
            "distance_labels": batch_labels,
        }


class JointMultimodalCSVStream(IterableDataset):
    """Attach exact sequence, curated text, and aligned AlphaFold geometry."""

    def __init__(
        self,
        paths: Iterable[str],
        protein_tokenizer: EsmcTokenizer,
        text_tokenizer: SentencePieceTextTokenizer,
        structure_database: str,
        protein_max_length: int,
        text_max_length: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle_files: bool = True,
        min_confidence: float = 70.0,
    ) -> None:
        super().__init__()
        self.paths = [str(Path(path)) for path in paths]
        if not self.paths:
            raise ValueError("JointMultimodalCSVStream requires at least one path")
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.structure_database = str(Path(structure_database).resolve())
        self.protein_max_length = protein_max_length
        self.text_max_length = text_max_length
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.shuffle_files = shuffle_files
        self.min_confidence = min_confidence
        self.epoch = 0
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _owned_paths(self) -> List[str]:
        paths = list(self.paths)
        if self.shuffle_files:
            random.Random(self.seed + self.epoch).shuffle(paths)
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        owner = self.rank * workers + worker_id
        owners = self.world_size * workers
        return [path for index, path in enumerate(paths) if index % owners == owner]

    def __iter__(self) -> Iterator[Dict[str, object]]:
        crop_rng = random.Random(self.seed + 1_000_003 * self.epoch + self.rank)
        uri = f"file:{self.structure_database}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            for path in self._owned_paths():
                with Path(path).open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    required = {"id", "accession", "sequence", "text"}
                    if reader.fieldnames is None or not required <= set(reader.fieldnames):
                        raise ValueError(f"{path} lacks joint multimodal columns")
                    for row in reader:
                        structure = connection.execute(
                            """
                            SELECT sequence, coordinates_f32le, confidence_f32le
                            FROM structures WHERE accession = ? AND fragment = 1
                            """,
                            (row["accession"],),
                        ).fetchone()
                        if structure is None:
                            continue
                        expected_sequence, coordinate_blob, confidence_blob = structure
                        sequence = row["sequence"].strip().upper()
                        text = row["text"].strip()
                        if sequence != expected_sequence or not text:
                            continue
                        coordinates = torch.frombuffer(
                            bytearray(coordinate_blob), dtype=torch.float32
                        ).reshape(-1, 3)
                        confidence = torch.frombuffer(
                            bytearray(confidence_blob), dtype=torch.float32
                        )
                        if coordinates.shape[0] != len(sequence) or confidence.shape[0] != len(
                            sequence
                        ):
                            raise RuntimeError(
                                "structure index row has inconsistent residue counts"
                            )
                        content_length = self.protein_max_length - 2
                        max_start = max(0, len(sequence) - content_length)
                        crop_start = crop_rng.randint(0, max_start) if max_start else 0
                        crop_stop = crop_start + content_length
                        cropped_coordinates = coordinates[crop_start:crop_stop].clone()
                        cropped_confidence = confidence[crop_start:crop_stop].clone()
                        _, structure_mask = ca_trace_features(
                            cropped_coordinates,
                            cropped_confidence,
                            min_confidence=self.min_confidence,
                        )
                        if int(structure_mask.sum()) < 4:
                            continue
                        yield {
                            "protein_input_ids": self.protein_tokenizer.encode(
                                sequence,
                                max_length=self.protein_max_length,
                                crop_start=crop_start,
                            ),
                            "text_input_ids": self.text_tokenizer.encode(
                                text, max_length=self.text_max_length
                            ),
                            "coordinates": cropped_coordinates,
                            "confidence": cropped_confidence,
                            "record_id": row["id"],
                            "text_group": " ".join(text.casefold().split()),
                        }


class SequenceStructureTextCollator:
    """Create leak-free MLM, text-LM, geometry, and structure-denoising targets."""

    def __init__(
        self,
        protein_tokenizer: EsmcTokenizer,
        text_tokenizer: SentencePieceTextTokenizer,
        seed: int,
        geometry_pairs: int,
        distance_bins: int,
        mlm_probability: float = 0.15,
        structure_mask_probability: float = 0.15,
        structure_dropout_probability: float = 0.0,
        min_confidence: float = 70.0,
    ) -> None:
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.generator = torch.Generator().manual_seed(seed)
        self.geometry_pairs = geometry_pairs
        self.distance_bins = distance_bins
        self.mlm_probability = mlm_probability
        self.structure_mask_probability = structure_mask_probability
        self.structure_dropout_probability = structure_dropout_probability
        self.min_confidence = min_confidence

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"generator_state": self.generator.get_state()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.generator.set_state(state["generator_state"])

    def __call__(self, examples: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        protein = self.protein_tokenizer.pad(
            [example["protein_input_ids"] for example in examples]
        )
        corrupted, protein_labels = mask_protein_tokens(
            protein["input_ids"],
            protein["attention_mask"],
            mask_token_id=self.protein_tokenizer.mask_token_id,
            vocabulary_size=max(self.protein_tokenizer.vocabulary.values()) + 1,
            special_token_ids=self.protein_tokenizer.special_token_ids,
            random_token_ids=self.protein_tokenizer.amino_acid_token_ids,
            generator=self.generator,
            mlm_probability=self.mlm_probability,
        )
        text = self.text_tokenizer.pad([example["text_input_ids"] for example in examples])
        text_labels = text["input_ids"].clone()
        text_labels.masked_fill_(~text["attention_mask"].to(torch.bool), -100)

        batch, token_length = protein["input_ids"].shape
        structure_labels = torch.zeros(
            (batch, token_length, STRUCTURE_FEATURE_DIMENSION), dtype=torch.float32
        )
        structure_attention_mask = torch.zeros((batch, token_length), dtype=torch.bool)
        batch_pairs = torch.zeros((batch, self.geometry_pairs, 2), dtype=torch.long)
        batch_distance_labels = torch.full(
            (batch, self.geometry_pairs), -100, dtype=torch.long
        )
        for index, example in enumerate(examples):
            features, valid = ca_trace_features(
                example["coordinates"],
                example["confidence"],
                min_confidence=self.min_confidence,
            )
            content_length = features.shape[0]
            structure_labels[index, 1 : 1 + content_length] = features
            structure_attention_mask[index, 1 : 1 + content_length] = valid
            pairs, labels = sample_geometry_pairs(
                example["coordinates"],
                example["confidence"],
                self.geometry_pairs,
                self.generator,
                min_confidence=self.min_confidence,
                distance_bins=self.distance_bins,
            )
            if not len(pairs):
                raise ValueError("joint example has no eligible high-confidence residue pairs")
            count = len(pairs)
            batch_pairs[index, :count] = pairs + 1
            batch_distance_labels[index, :count] = labels

        random_mask = torch.rand(
            structure_attention_mask.shape, generator=self.generator
        ).lt(self.structure_mask_probability)
        structure_label_mask = structure_attention_mask & random_mask
        if not structure_label_mask.any():
            first = torch.nonzero(structure_attention_mask, as_tuple=False)[0]
            structure_label_mask[first[0], first[1]] = True
        structure_input_features = structure_labels.clone()
        structure_input_features.masked_fill_(structure_label_mask.unsqueeze(-1), 0.0)
        if self.structure_dropout_probability:
            dropped = torch.rand((batch,), generator=self.generator).lt(
                self.structure_dropout_probability
            )
            if dropped.any():
                structure_label_mask[dropped] = structure_attention_mask[dropped]
                structure_input_features[dropped] = 0.0

        group_ids = [_contrastive_group_id(example["text_group"]) for example in examples]

        return {
            "protein_input_ids": corrupted,
            "protein_attention_mask": protein["attention_mask"],
            "protein_labels": protein_labels,
            "text_input_ids": text["input_ids"],
            "text_attention_mask": text["attention_mask"],
            "text_labels": text_labels,
            "structure_input_features": structure_input_features,
            "structure_attention_mask": structure_attention_mask,
            "structure_labels": structure_labels,
            "structure_label_mask": structure_label_mask,
            "pair_indices": batch_pairs,
            "distance_labels": batch_distance_labels,
            "contrastive_group_ids": torch.tensor(group_ids, dtype=torch.long),
        }
