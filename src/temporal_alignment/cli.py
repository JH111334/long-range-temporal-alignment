"""Command-line interface for reproducible smoke runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .pipeline import run_smoke_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run temporal-alignment Tier-0 diagnostics")
    parser.add_argument("--config", required=True, help="Path to a JSON experiment config")
    parser.add_argument("--output", required=True, help="Path for the JSON run artifact")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_smoke_pipeline(args.config, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
