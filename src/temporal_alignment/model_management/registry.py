"""Define the minimum contract for a pinned model resource probe."""

from __future__ import annotations

from dataclasses import dataclass

from ..enums import AssetStatus
from ..errors import EvidenceGateError
from ..modeling import ResourceBudget


@dataclass(frozen=True, slots=True)
class ModelProbePlan:
    """Describe a model probe without loading model weights."""

    model_id: str
    revision: str
    tokenizer_revision: str
    license_or_terms: str
    status: AssetStatus
    precision: str
    quantization: str
    budget: ResourceBudget

    def assert_probe_ready(self) -> None:
        """Reject model loading until identity, terms, and resources are fixed."""
        if self.status is not AssetStatus.APPROVED:
            raise EvidenceGateError(f"model {self.model_id} is not approved")
        required = {
            "model_id": self.model_id,
            "revision": self.revision,
            "tokenizer_revision": self.tokenizer_revision,
            "license_or_terms": self.license_or_terms,
            "precision": self.precision,
            "quantization": self.quantization,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise EvidenceGateError(f"model probe missing: {', '.join(missing)}")
        if any(
            "unfrozen" in value.casefold() for value in (self.revision, self.tokenizer_revision)
        ):
            raise EvidenceGateError("model and tokenizer revisions must be pinned")
