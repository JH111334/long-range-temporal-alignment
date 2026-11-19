"""Define backend-neutral contracts for time-bounded evidence retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import TemporalInterval, _required
from ..errors import ContractError


@dataclass(frozen=True, slots=True)
class EvidencePassage:
    """Represent one versioned passage returned by retrieval."""

    evidence_id: str
    source_id: str
    text: str
    event_time: TemporalInterval
    group_id: str
    checksum: str

    def __post_init__(self) -> None:
        for name in ("evidence_id", "source_id", "text", "group_id", "checksum"):
            _required(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """Describe a query and its temporal filter."""

    question: str
    language: str
    target_time: TemporalInterval | None = None
    top_k: int = 10

    def __post_init__(self) -> None:
        _required(self.question, "question")
        _required(self.language, "language")
        if not 1 <= self.top_k <= 100:
            raise ContractError("top_k must be between 1 and 100")


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Return ranked evidence while preserving deterministic identifiers."""

    request: RetrievalRequest
    passages: tuple[EvidencePassage, ...]

    def __post_init__(self) -> None:
        identifiers = [passage.evidence_id for passage in self.passages]
        if len(identifiers) != len(set(identifiers)):
            raise ContractError("retrieval result contains duplicate evidence_id values")
        if len(self.passages) > self.request.top_k:
            raise ContractError("retrieval result exceeds requested top_k")
