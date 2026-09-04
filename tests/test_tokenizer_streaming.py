import csv
import json

import torch

from openprotein.data.streaming import (
    CSVProteinStream,
    MultimodalCSVStream,
    ProteinMLMCollator,
    SequenceTextCollator,
)
from openprotein.data.tokenizer import EsmcTokenizer


def _tokenizer(tmp_path):
    vocab = {
        "<cls>": 0,
        "<pad>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "A": 4,
        "C": 5,
        "D": 6,
        "E": 7,
        "F": 8,
        "G": 9,
        "H": 10,
        "I": 11,
        "K": 12,
        "L": 13,
        "M": 14,
        "N": 15,
        "P": 16,
        "Q": 17,
        "R": 18,
        "S": 19,
        "T": 20,
        "V": 21,
        "W": 22,
        "Y": 23,
        "<mask>": 24,
    }
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"model": {"vocab": vocab}}))
    return EsmcTokenizer.from_tokenizer_json(str(path))


def test_tokenizer_adds_boundaries_and_crops(tmp_path):
    tokenizer = _tokenizer(tmp_path)
    assert tokenizer.encode("ACDE", max_length=5, crop_start=1) == [0, 5, 6, 7, 2]


def test_stream_and_collator_are_reproducible(tmp_path):
    tokenizer = _tokenizer(tmp_path)
    path = tmp_path / "train.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "name", "text"])
        writer.writeheader()
        writer.writerow({"id": 1, "name": "one", "text": "ACDEFGHIKLMNPQRSTVWY"})
        writer.writerow({"id": 2, "name": "two", "text": "YYYYYAAAAACCCCC"})
    stream = CSVProteinStream([str(path)], tokenizer, max_length=10, seed=9)
    examples = list(stream)
    assert len(examples) == 2
    assert all(len(example["input_ids"]) == 10 for example in examples)

    first = ProteinMLMCollator(tokenizer, seed=77, mlm_probability=1.0)(examples)
    second = ProteinMLMCollator(tokenizer, seed=77, mlm_probability=1.0)(examples)
    assert torch.equal(first["protein_input_ids"], second["protein_input_ids"])
    assert torch.equal(first["protein_labels"], second["protein_labels"])
    random_replacements = first["protein_input_ids"]
    allowed = set(tokenizer.amino_acid_token_ids + [tokenizer.mask_token_id, 0, 2])
    assert set(random_replacements.flatten().tolist()) <= allowed


class _FakeTextTokenizer:
    pad_token_id = 0

    def encode(self, text, max_length=None):
        ids = [2, *[4 + (ord(character) % 10) for character in text], 3]
        return ids if max_length is None else ids[: max_length - 1] + [3]

    def pad(self, examples):
        rows = list(examples)
        width = max(map(len, rows))
        ids = torch.zeros((len(rows), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = torch.tensor(row)
            mask[index, : len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def test_multimodal_stream_and_collator_emit_both_supervision_paths(tmp_path):
    protein_tokenizer = _tokenizer(tmp_path)
    text_tokenizer = _FakeTextTokenizer()
    path = tmp_path / "pairs.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "sequence", "text"])
        writer.writeheader()
        writer.writerow({"id": "one", "sequence": "ACDEFG", "text": "kinase"})
        writer.writerow({"id": "two", "sequence": "MNPQRS", "text": "enzyme"})
    examples = list(
        MultimodalCSVStream(
            [str(path)], protein_tokenizer, text_tokenizer, 8, 8, seed=4
        )
    )
    batch = SequenceTextCollator(
        protein_tokenizer, text_tokenizer, seed=5, mlm_probability=0.15
    )(examples)
    assert set(batch) == {
        "protein_input_ids",
        "protein_attention_mask",
        "protein_labels",
        "text_input_ids",
        "text_attention_mask",
        "text_labels",
        "contrastive_group_ids",
    }
    assert batch["protein_labels"].ne(-100).sum(dim=1).ge(1).all()
    assert torch.equal(batch["text_labels"].ne(-100), batch["text_attention_mask"].bool())
    assert batch["contrastive_group_ids"].shape == (2,)
    assert batch["contrastive_group_ids"].unique().numel() == 2

    equivalent_examples = [dict(examples[0]), dict(examples[1])]
    equivalent_examples[1]["text_group"] = "  KINASE  "
    equivalent_batch = SequenceTextCollator(
        protein_tokenizer, text_tokenizer, seed=5, mlm_probability=0.15
    )(equivalent_examples)
    assert equivalent_batch["contrastive_group_ids"].unique().numel() == 1
