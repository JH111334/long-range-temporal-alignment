"""Define evidence-grounded input and output for temporal QA."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import TemporalInterval, _required
from ..enums import Granularity, TemporalKind
from ..errors import ContractError


@dataclass(frozen=True, slots=True)
class TemporalQARequest:
    """Represent one Vietnamese-first temporal QA request."""

    request_id: str
    question: str
    language: str
    target_time: TemporalInterval | None = None
    target_granularity: Granularity | None = None
    target_kind: TemporalKind | None = None

    def __post_init__(self) -> None:
        for name in ("request_id", "question", "language"):
            _required(getattr(self, name), name)
        temporal_fields = (self.target_time, self.target_granularity, self.target_kind)
        if any(value is not None for value in temporal_fields) and not all(
            value is not None for value in temporal_fields
        ):
            raise ContractError(
                "target_time, target_granularity and target_kind must appear together"
            )


@dataclass(frozen=True, slots=True)
class TemporalQAResponse:
    """Return an answer with evidence, or abstain with a reason."""

    request_id: str
    answer: str | None
    evidence_ids: tuple[str, ...]
    confidence: float
    canonical_time: TemporalInterval | None = None
    canonical_granularity: Granularity | None = None
    canonical_kind: TemporalKind | None = None
    abstention_reason: str | None = None

    def __post_init__(self) -> None:
        _required(self.request_id, "request_id")
        if not 0.0 <= self.confidence <= 1.0:
            raise ContractError("confidence must be between 0 and 1")
        answered = self.answer is not None and bool(self.answer.strip())
        abstained = self.abstention_reason is not None and bool(self.abstention_reason.strip())
        if answered == abstained:
            raise ContractError("response must contain either an answer or an abstention reason")
        if answered and not self.evidence_ids:
            raise ContractError("answered response requires at least one evidence_id")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ContractError("evidence_ids must be unique")
        temporal_fields = (
            self.canonical_time,
            self.canonical_granularity,
            self.canonical_kind,
        )
        if any(value is not None for value in temporal_fields) and not all(
            value is not None for value in temporal_fields
        ):
            raise ContractError(
                "canonical_time, canonical_granularity and canonical_kind must appear together"
            )
