"""Expose the external-source boundary for temporal evidence ingestion."""

from SourceAPI.contracts import (
    ProviderResponse,
    SourceAPIError,
    SourceBatch,
    SourceClient,
    SourceConfig,
    SourceFailure,
    SourceProvenance,
    TemporalEnvelope,
    TemporalSourceAdapter,
    TemporalSourceRecord,
)
from SourceAPI.pipeline import ingest_source

__all__ = [
    "ProviderResponse",
    "SourceAPIError",
    "SourceBatch",
    "SourceClient",
    "SourceConfig",
    "SourceFailure",
    "SourceProvenance",
    "TemporalEnvelope",
    "TemporalSourceAdapter",
    "TemporalSourceRecord",
    "ingest_source",
]
