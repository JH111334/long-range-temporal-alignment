"""Define provider-neutral contracts for temporal source ingestion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from temporal_alignment.contracts import TemporalInterval

TemporalPrecision = Literal["unknown", "year", "month", "day", "instant"]
FailureCode = Literal[
    "timeout",
    "rate_limited",
    "authentication",
    "http_retryable",
    "http_non_retryable",
    "malformed_payload",
    "schema_validation",
    "partial_response",
]


def _required(name: str, value: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty trimmed string.")


def _iso_datetime(name: str, value: str) -> None:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 datetime.") from exc


@dataclass(frozen=True, slots=True)
class SourceAPIError(RuntimeError):
    """Expose a stable, visible provider failure."""

    code: FailureCode
    detail: str
    retryable: bool = False

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """Configure one provider without carrying a credential value."""

    provider: str
    base_url: str
    timeout_seconds: float = 30.0
    retry_budget: int = 1
    retry_backoff_seconds: float = 0.0
    credential_env: str | None = None

    def __post_init__(self) -> None:
        _required("provider", self.provider)
        if not self.base_url.startswith("https://"):
            raise ValueError("base_url must use HTTPS.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if not 0 <= self.retry_budget <= 5:
            raise ValueError("retry_budget must be between 0 and 5.")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must not be negative.")
        if self.credential_env and (
            not self.credential_env.isidentifier()
            or self.credential_env.upper() != self.credential_env
        ):
            raise ValueError("credential_env must be an uppercase environment-variable name.")


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """Bind one temporal record to provider, retrieval time and terms."""

    provider: str
    source_id: str
    source_uri: str
    retrieved_at: str
    license_or_terms: str
    revision: str | None = None
    checksum: str | None = None

    def __post_init__(self) -> None:
        for name in ("provider", "source_id", "source_uri", "retrieved_at", "license_or_terms"):
            _required(name, str(getattr(self, name)))
        _iso_datetime("retrieved_at", self.retrieved_at)
        if self.checksum and (
            len(self.checksum) != 64
            or any(character not in "0123456789abcdef" for character in self.checksum)
        ):
            raise ValueError("checksum must be 64 lowercase hexadecimal characters.")


@dataclass(frozen=True, slots=True)
class TemporalEnvelope:
    """Preserve raw temporal evidence beside an optional canonical interval."""

    raw_temporal: str
    precision: TemporalPrecision
    calendar_system: str
    canonical_interval: TemporalInterval | None = None
    source_timestamp: str | None = None
    event_time: str | None = None
    publication_time: str | None = None
    timezone: str | None = None

    def __post_init__(self) -> None:
        _required("raw_temporal", self.raw_temporal)
        _required("calendar_system", self.calendar_system)
        for name in ("source_timestamp", "event_time", "publication_time", "timezone"):
            value = getattr(self, name)
            if value is not None:
                _required(name, value)


@dataclass(frozen=True, slots=True)
class TemporalSourceRecord:
    """Represent normalized temporal evidence without carrying raw API JSON."""

    record_id: str
    text: str
    source_uri: str
    temporal: TemporalEnvelope
    attributes: Mapping[str, str]
    provenance: SourceProvenance

    def __post_init__(self) -> None:
        for name in ("record_id", "text", "source_uri"):
            _required(name, str(getattr(self, name)))
        if any(character.isspace() for character in self.record_id):
            raise ValueError("record_id must not contain whitespace.")
        if self.source_uri != self.provenance.source_uri:
            raise ValueError("record and provenance source_uri must match.")


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Carry raw provider data only to the temporal normalization adapter."""

    payload: Mapping[str, object]
    status_code: int
    partial: bool = False


@dataclass(frozen=True, slots=True)
class SourceFailure:
    """Expose a failed or partial source operation."""

    provider: str
    code: FailureCode
    detail: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class SourceBatch:
    """Return canonical temporal records plus visible failure state."""

    records: tuple[TemporalSourceRecord, ...]
    failures: tuple[SourceFailure, ...]
    partial: bool
    attempts: int


class SourceClient(Protocol):
    """Fetch one response from an explicitly configured provider."""

    def fetch(self, config: SourceConfig) -> ProviderResponse:
        """Return one provider response or raise SourceAPIError."""


class TemporalSourceAdapter(Protocol):
    """Normalize provider data without discarding raw temporal evidence."""

    def normalize(
        self, response: ProviderResponse, config: SourceConfig
    ) -> tuple[TemporalSourceRecord, ...]:
        """Validate and normalize one provider response."""
