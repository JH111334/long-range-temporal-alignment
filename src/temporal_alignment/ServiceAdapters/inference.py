"""Fail-closed contracts for optional remote inference probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..errors import ContractError, EvidenceGateError

ALLOWED_REMOTE_TASKS = frozenset({"feature-extraction", "text-classification"})
ALLOWED_PRIVACY_CLASSES = frozenset({"public", "restricted", "personal", "confidential"})
MUTABLE_REVISIONS = frozenset({"main", "master", "latest"})


@dataclass(frozen=True, slots=True)
class InferencePlan:
    """Describe one immutable, bounded remote inference probe."""

    provider: str
    provider_route: str
    model_id: str
    model_revision: str
    task: str
    input_schema: str
    output_schema: str
    privacy_class: str
    timeout_seconds: float
    retry_budget: int
    max_input_units: int
    normalization: str

    def __post_init__(self) -> None:
        text_fields = (
            "provider",
            "provider_route",
            "model_id",
            "model_revision",
            "task",
            "input_schema",
            "output_schema",
            "privacy_class",
            "normalization",
        )
        for field_name in text_fields:
            if not str(getattr(self, field_name)).strip():
                raise ContractError(f"{field_name} must be non-empty")
        if self.model_revision.casefold() in MUTABLE_REVISIONS:
            raise ContractError("remote inference model_revision must be immutable")
        if self.task not in ALLOWED_REMOTE_TASKS:
            raise ContractError(f"unsupported remote task: {self.task}")
        if self.privacy_class not in ALLOWED_PRIVACY_CLASSES:
            raise ContractError(f"unsupported privacy_class: {self.privacy_class}")
        if self.privacy_class != "public":
            raise EvidenceGateError("P1 remote inference accepts public inputs only")
        if not 0 < self.timeout_seconds <= 120:
            raise ContractError("timeout_seconds must be in (0, 120]")
        if not 0 <= self.retry_budget <= 3:
            raise ContractError("retry_budget must be between 0 and 3")
        if self.max_input_units <= 0:
            raise ContractError("max_input_units must be positive")


@dataclass(frozen=True, slots=True)
class InferenceArtifact:
    """Record validated provenance without storing payloads or secrets."""

    provider: str
    provider_route: str
    model_id: str
    resolved_revision: str
    request_sha256: str
    output_schema: str
    latency_ms: float
    evidence_class: str = "external_probe"


class RemoteInferenceAdapter(Protocol):
    """Define the boundary for a provider implementation."""

    def execute(self, plan: InferencePlan, payload: object) -> InferenceArtifact:
        """Execute only after an ADR and transport tests approve network access."""


class HuggingFaceInferenceAdapter:
    """Keep Hugging Face inference disabled until the network gate is approved."""

    provider = "huggingface"

    def validate_plan(self, plan: InferencePlan) -> None:
        """Reject provider mismatches before a future transport call."""
        if plan.provider.casefold() != self.provider:
            raise ContractError("inference provider must be huggingface")

    def execute(self, plan: InferencePlan, payload: object) -> InferenceArtifact:
        """Fail closed; this scaffold never sends payloads to a service."""
        self.validate_plan(plan)
        raise EvidenceGateError("remote inference is scaffold-only; execution is disabled")
