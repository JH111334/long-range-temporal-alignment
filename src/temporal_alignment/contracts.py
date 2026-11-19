"""Fail-closed contracts for temporal examples, assets, and experiment artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from .enums import AssetStatus, DataRole, Era, EvidenceClass, Granularity, TemporalKind
from .errors import ContractError, EvidenceGateError


def _required(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ContractError(f"{field_name} must not be empty")
    return normalized


@dataclass(frozen=True, slots=True)
class HistoricalYear:
    """A historical year without an ambiguous display year zero."""

    era: Era
    year: int

    def __post_init__(self) -> None:
        if self.year < 1:
            raise ContractError("historical year must be >= 1; year zero is not accepted")

    @property
    def astronomical(self) -> int:
        """Return astronomical numbering, where 1 BCE is 0 and 2 BCE is -1."""
        return self.year if self.era is Era.CE else 1 - self.year

    @classmethod
    def from_astronomical(cls, value: int) -> HistoricalYear:
        """Construct from astronomical numbering."""
        return cls(Era.CE, value) if value >= 1 else cls(Era.BCE, 1 - value)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HistoricalYear:
        return cls(era=Era(payload["era"]), year=int(payload["year"]))

    def display(self) -> str:
        return f"{self.year} {self.era.value}"


@dataclass(frozen=True, slots=True)
class HistoricalDate:
    """Precision-aware historical boundary without guessing missing fields."""

    year: HistoricalYear
    month: int | None = None
    day: int | None = None

    def __post_init__(self) -> None:
        if self.month is not None and not 1 <= self.month <= 12:
            raise ContractError("month must be in [1, 12]")
        if self.day is not None and self.month is None:
            raise ContractError("day requires month")
        if self.day is not None and not 1 <= self.day <= 31:
            raise ContractError("day must be in [1, 31]")

    @property
    def granularity(self) -> Granularity:
        """Infer stated precision; calendar validity remains a separate gate."""
        if self.day is not None:
            return Granularity.DAY
        if self.month is not None:
            return Granularity.MONTH
        return Granularity.YEAR

    @property
    def sort_key(self) -> tuple[int, int, int]:
        """Return a deterministic key without inventing a display date."""
        return (self.year.astronomical, self.month or 1, self.day or 1)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HistoricalDate:
        return cls(
            year=HistoricalYear.from_dict(payload),
            month=int(payload["month"]) if payload.get("month") is not None else None,
            day=int(payload["day"]) if payload.get("day") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class TemporalInterval:
    """Half-open interval `[start, end_exclusive)` with explicit endpoint precision."""

    start: HistoricalDate
    end_exclusive: HistoricalDate

    def __post_init__(self) -> None:
        if self.start.sort_key >= self.end_exclusive.sort_key:
            raise ContractError("temporal interval must have start < end_exclusive")

    @classmethod
    def one_year(cls, year: HistoricalYear) -> TemporalInterval:
        return cls(
            HistoricalDate(year),
            HistoricalDate(HistoricalYear.from_astronomical(year.astronomical + 1)),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TemporalInterval:
        return cls(
            start=HistoricalDate.from_dict(payload["start"]),
            end_exclusive=HistoricalDate.from_dict(payload["end_exclusive"]),
        )


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Minimum provenance attached to every example."""

    source_id: str
    source_url: str
    accessed_on: str
    evidence_class: EvidenceClass
    license_or_terms: str
    checksum: str

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "source_url",
            "accessed_on",
            "license_or_terms",
            "checksum",
        ):
            _required(getattr(self, name), name)
        try:
            date.fromisoformat(self.accessed_on)
        except ValueError as exc:
            raise ContractError("accessed_on must use YYYY-MM-DD") from exc

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProvenanceRecord:
        return cls(
            source_id=payload["source_id"],
            source_url=payload["source_url"],
            accessed_on=payload["accessed_on"],
            evidence_class=EvidenceClass(payload["evidence_class"]),
            license_or_terms=payload["license_or_terms"],
            checksum=payload["checksum"],
        )


@dataclass(frozen=True, slots=True)
class TemporalExample:
    """Canonical time-sensitive QA example."""

    example_id: str
    question: str
    answer: str
    event_time: TemporalInterval
    granularity: Granularity
    temporal_kind: TemporalKind
    language: str
    group_id: str
    provenance: ProvenanceRecord
    options: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("example_id", "question", "answer", "language", "group_id"):
            _required(getattr(self, name), name)
        if self.options and self.answer not in self.options:
            raise ContractError("answer must be one of options")
        endpoint_granularities = {
            self.event_time.start.granularity,
            self.event_time.end_exclusive.granularity,
        }
        if endpoint_granularities != {self.granularity}:
            raise ContractError("event_time endpoint precision must match granularity")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TemporalExample:
        return cls(
            example_id=payload["example_id"],
            question=payload["question"],
            answer=payload["answer"],
            event_time=TemporalInterval.from_dict(payload["event_time"]),
            granularity=Granularity(payload["granularity"]),
            temporal_kind=TemporalKind(payload["temporal_kind"]),
            language=payload["language"],
            group_id=payload["group_id"],
            provenance=ProvenanceRecord.from_dict(payload["provenance"]),
            options=tuple(payload.get("options", ())),
        )


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Governance record required before an external dataset can be ingested."""

    dataset_id: str
    official_url: str
    accessed_on: str
    license_or_terms: str
    reviewer: str
    revision: str
    sha256: str
    role: DataRole
    status: AssetStatus
    notes: str = ""

    def assert_ingestible(self) -> None:
        """Reject candidates that have not passed all evidence gates."""
        if self.status is not AssetStatus.APPROVED:
            raise EvidenceGateError(f"dataset {self.dataset_id} is not approved")
        required = {
            "official_url": self.official_url,
            "accessed_on": self.accessed_on,
            "license_or_terms": self.license_or_terms,
            "reviewer": self.reviewer,
            "revision": self.revision,
            "sha256": self.sha256,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise EvidenceGateError(f"dataset {self.dataset_id} missing: {', '.join(missing)}")


@dataclass(frozen=True, slots=True)
class RepresentationArtifact:
    """Provenance for one encoder output."""

    example_id: str
    encoder_id: str
    encoder_class: str
    values: tuple[float, ...]
    config_sha256: str

    def __post_init__(self) -> None:
        if not self.values:
            raise ContractError("representation values must not be empty")
        if not all(isinstance(value, (int, float)) for value in self.values):
            raise ContractError("representation values must be numeric")


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """Minimal immutable run record."""

    run_id: str
    evidence_class: EvidenceClass
    config_sha256: str
    data_manifest_sha256: str
    code_revision: str
    metrics: dict[str, float]
    resource: dict[str, float | str]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
