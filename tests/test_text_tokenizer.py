import csv
import json

import pytest

from openprotein.data.text_tokenizer import SentencePieceTextTokenizer, train_text_tokenizer


sentencepiece = pytest.importorskip("sentencepiece")


def test_text_tokenizer_uses_train_split_only_and_has_fixed_special_ids(tmp_path):
    train = tmp_path / "data/train/train.csv"
    train.parent.mkdir(parents=True)
    with train.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "text"])
        writer.writeheader()
        for index in range(100):
            writer.writerow(
                {
                    "id": index,
                    "text": f"Protein kinase annotation number {index} with catalytic activity.",
                }
            )
    data_manifest = tmp_path / "data/manifest.json"
    data_manifest.write_text(json.dumps({"fixture": True}))
    output = tmp_path / "tokenizer"
    receipt = train_text_tokenizer(str(train), str(data_manifest), str(output), vocab_size=300)
    tokenizer = SentencePieceTextTokenizer(str(output / "tokenizer.model"))
    assert len(receipt["builder_code_sha256"]) == 64
    assert receipt["training_data"]["validation_and_test_used"] is False
    assert [
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
    ] == [0, 1, 2, 3]
    assert len(tokenizer.encode("Protein kinase", max_length=8)) <= 8
