"""Create an optimizer-free, content-addressed adapter release."""

from __future__ import annotations

import argparse
import json

from openprotein.release import export_trainable_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("model_config")
    parser.add_argument("tokenizer_json")
    parser.add_argument("model_artifact_manifest")
    parser.add_argument("output_dir")
    parser.add_argument("--text-tokenizer-model")
    parser.add_argument("--pretrain-receipt")
    args = parser.parse_args()
    receipt = export_trainable_adapter(
        args.checkpoint,
        args.model_config,
        args.tokenizer_json,
        args.model_artifact_manifest,
        args.output_dir,
        text_tokenizer_model=args.text_tokenizer_model,
        pretrain_receipt=args.pretrain_receipt,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
