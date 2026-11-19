"""Run the temporal SourceAPI boundary without a built-in network provider."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from SourceAPI.contracts import (  # noqa: E402
    SourceAPIError,
    SourceBatch,
    SourceClient,
    SourceConfig,
    SourceFailure,
    TemporalSourceAdapter,
)

SCHEMA_VERSION = "temporal-source-api-v1"


def contract_payload() -> dict[str, Any]:
    """Describe the fail-closed temporal source boundary."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "NO_PROVIDER_ENABLED",
        "flow": ["fetch", "validate", "normalize", "attach_provenance", "return_records"],
        "temporal_evidence": [
            "raw_temporal",
            "canonical_interval_optional",
            "source_timestamp",
            "event_time",
            "publication_time",
            "retrieved_at",
            "precision",
            "timezone",
            "calendar_system",
        ],
        "raw_payload_downstream": False,
        "network_default": False,
    }


def ingest_source(
    config: SourceConfig,
    client: SourceClient,
    adapter: TemporalSourceAdapter,
    *,
    sleeper: Callable[[float], None] = time.sleep,
) -> SourceBatch:
    """Fetch, retry bounded transient failures, and return temporal records."""
    attempts = 0
    while True:
        attempts += 1
        try:
            response = client.fetch(config)
            break
        except SourceAPIError as exc:
            if exc.retryable and attempts <= config.retry_budget:
                sleeper(config.retry_backoff_seconds * 2 ** (attempts - 1))
                continue
            return SourceBatch(
                records=(),
                failures=(SourceFailure(config.provider, exc.code, exc.detail, exc.retryable),),
                partial=False,
                attempts=attempts,
            )
    try:
        records = adapter.normalize(response, config)
    except SourceAPIError as exc:
        return SourceBatch(
            records=(),
            failures=(SourceFailure(config.provider, exc.code, exc.detail, exc.retryable),),
            partial=response.partial,
            attempts=attempts,
        )
    if not records:
        return SourceBatch(
            records=(),
            failures=(
                SourceFailure(
                    config.provider,
                    "schema_validation",
                    "provider response produced no canonical temporal records",
                    False,
                ),
            ),
            partial=response.partial,
            attempts=attempts,
        )
    failures = (
        (
            SourceFailure(
                config.provider,
                "partial_response",
                "provider marked the response as partial",
                True,
            ),
        )
        if response.partial
        else ()
    )
    return SourceBatch(
        records=records, failures=failures, partial=response.partial, attempts=attempts
    )


def build_parser() -> argparse.ArgumentParser:
    """Expose contract inspection only; providers remain explicit adapters."""
    parser = argparse.ArgumentParser(description="External temporal-source contract boundary.")
    parser.add_subparsers(dest="command", required=True).add_parser("contract")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print the fail-closed contract."""
    build_parser().parse_args(argv)
    print(json.dumps(contract_payload(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
