"""Train the fixed text tokenizer from joined training annotations only."""

from __future__ import annotations

import argparse
import json

from openprotein.data.text_tokenizer import train_text_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_glob")
    parser.add_argument("data_manifest")
    parser.add_argument("output_root")
    parser.add_argument("--vocab-size", type=int, default=32768)
    args = parser.parse_args()
    receipt = train_text_tokenizer(
        args.train_glob, args.data_manifest, args.output_root, args.vocab_size
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
