from types import SimpleNamespace

import torch

from openprotein.data.tokenizer import EsmcTokenizer
from openprotein.inference import generate_annotation, predict_pair_distances


class _FakeTextTokenizer:
    bos_token_id = 2
    eos_token_id = 3

    def encode(self, text):
        return [self.bos_token_id, 5, self.eos_token_id]

    def decode(self, token_ids):
        return " ".join(str(value) for value in token_ids if value not in {2, 3})


class _FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(max_sequence_length=16, max_text_length=8)

    def eval(self):
        return self

    def __call__(self, **kwargs):
        length = kwargs["text_input_ids"].shape[1]
        logits = torch.zeros((1, length, 8))
        logits[0, -1, 4 if length == 1 else 3] = 10
        return {"text_logits": logits}


def test_greedy_annotation_generation_stops_on_eos_and_hashes_sequence():
    protein_tokenizer = EsmcTokenizer(
        {"<cls>": 0, "<pad>": 1, "<eos>": 2, "<unk>": 3, "<mask>": 4, "A": 5}
    )
    result = generate_annotation(
        _FakeModel(),
        protein_tokenizer,
        _FakeTextTokenizer(),
        "AAAA",
        torch.device("cpu"),
        max_new_tokens=4,
    )
    assert result["generated_tokens"] == [4]
    assert result["stopped_on_eos"] is True
    assert result["text"] == "4"


def test_annotation_generation_rejects_overlength_sequence():
    protein_tokenizer = EsmcTokenizer(
        {"<cls>": 0, "<pad>": 1, "<eos>": 2, "<unk>": 3, "<mask>": 4, "A": 5}
    )
    try:
        generate_annotation(
            _FakeModel(),
            protein_tokenizer,
            _FakeTextTokenizer(),
            "A" * 15,
            torch.device("cpu"),
        )
    except ValueError as error:
        assert "exceeds" in str(error)
    else:
        raise AssertionError("overlength sequence was accepted")


class _FakeGeometryModel(_FakeModel):
    def __init__(self):
        self.config = SimpleNamespace(
            max_sequence_length=16, max_text_length=8, distance_bins=4
        )

    def __call__(self, **kwargs):
        pairs = kwargs["pair_indices"].shape[1]
        logits = torch.zeros((1, pairs, 4))
        logits[0, 0, 2] = 10
        logits[0, 1, 3] = 10
        return {"distance_logits": logits}


def test_pair_distance_prediction_uses_one_based_residue_positions():
    protein_tokenizer = EsmcTokenizer(
        {"<cls>": 0, "<pad>": 1, "<eos>": 2, "<unk>": 3, "<mask>": 4, "A": 5}
    )
    result = predict_pair_distances(
        _FakeGeometryModel(),
        protein_tokenizer,
        "AAAA",
        [(1, 4), (2, 4)],
        torch.device("cpu"),
    )
    assert [item["distance_bin"] for item in result["predictions"]] == [2, 3]
    assert result["predictions"][0]["distance_lower_angstrom"] == 17.0
    assert result["predictions"][0]["distance_upper_angstrom"] == 32.0
    assert result["predictions"][1]["distance_upper_angstrom"] is None
