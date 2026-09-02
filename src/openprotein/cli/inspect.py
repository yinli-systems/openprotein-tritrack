"""Inspect and validate a model configuration without loading its weights."""

from __future__ import annotations

import argparse
import json

from openprotein.config import load_model_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to a model JSON file")
    parser.add_argument(
        "--require-target-scale",
        action="store_true",
        help="Fail unless the analytical count is between 7B and 15B",
    )
    args = parser.parse_args()

    config = load_model_config(args.config)
    report = {
        "name": config.name,
        "estimated_parameter_count": config.estimated_parameter_count,
        "is_target_scale": config.is_target_scale,
        "breakdown": config.parameter_breakdown(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_target_scale and not config.is_target_scale:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
