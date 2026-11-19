"""Validate temporal SourceAPI behavior without network calls."""

from collections.abc import Mapping

from SourceAPI.contracts import (
    ProviderResponse,
    SourceAPIError,
    SourceConfig,
    SourceProvenance,
    TemporalEnvelope,
    TemporalSourceRecord,
)
from SourceAPI.pipeline import ingest_source
from temporal_alignment.contracts import HistoricalDate, HistoricalYear, TemporalInterval
from temporal_alignment.enums import Era


class FixtureClient:
    """Return one fixture response after an optional transient failure."""

    def __init__(self, *, fail_once: bool = False, partial: bool = False) -> None:
        self.calls = 0
        self.fail_once = fail_once
        self.partial = partial

    def fetch(self, config: SourceConfig) -> ProviderResponse:
        self.calls += 1
        if self.fail_once and self.calls == 1:
            raise SourceAPIError("rate_limited", "fixture 429", True)
        return ProviderResponse(
            payload={"items": [{"id": "event-1", "text": "Sự kiện diễn ra năm 1975."}]},
            status_code=200,
            partial=self.partial,
        )


class FixtureAdapter:
    """Normalize one fixture while preserving its raw temporal expression."""

    def normalize(
        self, response: ProviderResponse, config: SourceConfig
    ) -> tuple[TemporalSourceRecord, ...]:
        items = response.payload.get("items")
        if not isinstance(items, list) or not items or not isinstance(items[0], Mapping):
            raise SourceAPIError("malformed_payload", "items must be a non-empty list")
        item = items[0]
        source_uri = "provider://fixture/event-1"
        interval = TemporalInterval(
            HistoricalDate(HistoricalYear(Era.CE, 1975)),
            HistoricalDate(HistoricalYear(Era.CE, 1976)),
        )
        return (
            TemporalSourceRecord(
                record_id=str(item["id"]),
                text=str(item["text"]),
                source_uri=source_uri,
                temporal=TemporalEnvelope(
                    raw_temporal="năm 1975",
                    precision="year",
                    calendar_system="gregorian",
                    canonical_interval=interval,
                    event_time="1975",
                    publication_time="2026-09-01",
                    source_timestamp="2026-09-01T08:00:00+07:00",
                    timezone="Asia/Ho_Chi_Minh",
                ),
                attributes={"language": "vi"},
                provenance=SourceProvenance(
                    provider=config.provider,
                    source_id="event-1",
                    source_uri=source_uri,
                    retrieved_at="2026-09-06T09:00:00+00:00",
                    license_or_terms="fixture-only",
                    revision="fixture-v1",
                    checksum="0" * 64,
                ),
            ),
        )


def _config() -> SourceConfig:
    return SourceConfig(
        provider="fixture",
        base_url="https://example.test/api",
        retry_budget=1,
        retry_backoff_seconds=0,
        credential_env="FIXTURE_API_TOKEN",
    )


def test_source_api_preserves_raw_and_canonical_temporal_evidence() -> None:
    client = FixtureClient(fail_once=True)
    batch = ingest_source(_config(), client, FixtureAdapter(), sleeper=lambda _: None)
    assert client.calls == 2
    assert not batch.failures
    assert batch.records[0].temporal.raw_temporal == "năm 1975"
    assert batch.records[0].temporal.canonical_interval is not None
    assert not hasattr(batch.records[0], "payload")


def test_source_api_exposes_partial_response() -> None:
    batch = ingest_source(_config(), FixtureClient(partial=True), FixtureAdapter())
    assert batch.partial is True
    assert batch.failures[0].code == "partial_response"
