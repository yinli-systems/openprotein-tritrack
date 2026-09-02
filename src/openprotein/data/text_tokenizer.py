"""Deterministic SentencePiece tooling for curated protein annotations."""

from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from openprotein.provenance import sha256_file


def _sentencepiece():
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("text tokenization requires sentencepiece") from exc
    return spm


def _iter_text(paths: List[str], counters: Dict[str, int]) -> Iterable[str]:
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "text" not in reader.fieldnames:
                raise ValueError(f"{path} has no text column")
            for row in reader:
                text = row["text"].strip()
                if not text:
                    continue
                counters["records"] += 1
                counters["utf8_bytes"] += len(text.encode("utf-8"))
                yield text


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def train_text_tokenizer(
    train_glob: str,
    data_manifest: str,
    output_root: str,
    vocab_size: int = 32_768,
) -> Dict[str, Any]:
    """Train a one-thread, no-shuffle unigram tokenizer from train text only."""

    paths = sorted(glob.glob(train_glob))
    if not paths:
        raise ValueError(f"text tokenizer glob matched no files: {train_glob}")
    manifest_path = Path(data_manifest).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"multimodal data manifest is missing: {manifest_path}")
    final_root = Path(output_root).resolve()
    if final_root.exists():
        raise FileExistsError(f"refusing to overwrite tokenizer output: {final_root}")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{final_root.name}.tmp-", dir=final_root.parent)
    )
    counters = {"records": 0, "utf8_bytes": 0}
    try:
        spm = _sentencepiece()
        spm.SentencePieceTrainer.train(
            sentence_iterator=_iter_text(paths, counters),
            model_prefix=str(temporary_root / "tokenizer"),
            vocab_size=vocab_size,
            model_type="unigram",
            character_coverage=1.0,
            byte_fallback=True,
            normalization_rule_name="nmt_nfkc",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<bos>",
            eos_piece="<eos>",
            user_defined_symbols=["<protein>", "<text>"],
            input_sentence_size=0,
            shuffle_input_sentence=False,
            num_threads=1,
            hard_vocab_limit=True,
        )
        model_path = temporary_root / "tokenizer.model"
        vocabulary_path = temporary_root / "tokenizer.vocab"
        processor = spm.SentencePieceProcessor(model_file=str(model_path))
        if processor.vocab_size() != vocab_size:
            raise RuntimeError("trained tokenizer vocabulary does not match the requested size")
        receipt: Dict[str, Any] = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "builder_code_sha256": sha256_file(Path(__file__).resolve()),
            "training_data": {
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "train_files": paths,
                "records": counters["records"],
                "utf8_bytes": counters["utf8_bytes"],
                "validation_and_test_used": False,
            },
            "trainer": {
                "model_type": "unigram",
                "vocab_size": vocab_size,
                "character_coverage": 1.0,
                "byte_fallback": True,
                "normalization_rule_name": "nmt_nfkc",
                "shuffle_input_sentence": False,
                "num_threads": 1,
            },
            "special_tokens": {
                "pad_id": processor.pad_id(),
                "unk_id": processor.unk_id(),
                "bos_id": processor.bos_id(),
                "eos_id": processor.eos_id(),
            },
            "artifacts": {
                "model": {
                    "path": "tokenizer.model",
                    "bytes": model_path.stat().st_size,
                    "sha256": sha256_file(model_path),
                },
                "vocabulary": {
                    "path": "tokenizer.vocab",
                    "bytes": vocabulary_path.stat().st_size,
                    "sha256": sha256_file(vocabulary_path),
                },
            },
        }
        _atomic_json(temporary_root / "manifest.json", receipt)
        os.replace(temporary_root, final_root)
        return receipt
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


class SentencePieceTextTokenizer:
    def __init__(self, model_path: str) -> None:
        spm = _sentencepiece()
        self.processor = spm.SentencePieceProcessor(model_file=model_path)
        if min(self.pad_token_id, self.unk_token_id, self.bos_token_id, self.eos_token_id) < 0:
            raise ValueError("SentencePiece model lacks required special tokens")

    @property
    def vocabulary_size(self) -> int:
        return int(self.processor.vocab_size())

    @property
    def pad_token_id(self) -> int:
        return int(self.processor.pad_id())

    @property
    def unk_token_id(self) -> int:
        return int(self.processor.unk_id())

    @property
    def bos_token_id(self) -> int:
        return int(self.processor.bos_id())

    @property
    def eos_token_id(self) -> int:
        return int(self.processor.eos_id())

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        pieces = list(self.processor.encode(text.strip(), out_type=int))
        if max_length is not None:
            if max_length < 2:
                raise ValueError("max_length must leave room for BOS and EOS")
            pieces = pieces[: max_length - 2]
        return [self.bos_token_id, *pieces, self.eos_token_id]

    def decode(self, token_ids: Iterable[int]) -> str:
        special = {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
        }
        content = [int(token) for token in token_ids if int(token) not in special]
        return str(self.processor.decode(content))

    def pad(self, examples: Iterable[List[int]]) -> Dict[str, torch.Tensor]:
        rows = list(examples)
        if not rows:
            raise ValueError("cannot pad empty text examples")
        width = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}
