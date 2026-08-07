"""Framework-neutral contracts for optional model and trainer integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .contracts import TemporalExample
from .errors import ContractError


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    """Hard resource ceiling that must be measured before model training."""

    ram_gib: float
    vram_gib: float
    max_sequence_length: int
    micro_batch_size: int

    def __post_init__(self) -> None:
        if self.ram_gib <= 0 or self.vram_gib < 0:
            raise ContractError("RAM must be positive and VRAM non-negative")
        if self.max_sequence_length < 1 or self.micro_batch_size < 1:
            raise ContractError("sequence length and micro-batch size must be positive")


@dataclass(frozen=True, slots=True)
class ResourceMeasurement:
    """Observed peak use from a resource probe."""

    peak_ram_gib: float
    peak_vram_gib: float
    device: str

    def fits(self, budget: ResourceBudget) -> bool:
        return self.peak_ram_gib <= budget.ram_gib and self.peak_vram_gib <= budget.vram_gib


class RepresentationModel(Protocol):
    """Adapter for a frozen LLM or another representation model."""

    model_id: str
    revision: str

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one fixed-width vector per text."""


class TemporalTrainer(Protocol):
    """Adapter for PEFT/full-training implementations."""

    trainer_id: str

    def fit(
        self,
        examples: Sequence[TemporalExample],
        *,
        budget: ResourceBudget,
    ) -> dict[str, str | float]:
        """Train under an explicit budget and return artifact metadata."""


def assert_resource_gate(
    measurement: ResourceMeasurement,
    budget: ResourceBudget,
) -> None:
    """Fail closed when a measured candidate exceeds the declared budget."""
    if not measurement.fits(budget):
        raise ContractError(
            "resource probe exceeded budget: "
            f"RAM {measurement.peak_ram_gib}/{budget.ram_gib} GiB, "
            f"VRAM {measurement.peak_vram_gib}/{budget.vram_gib} GiB"
        )
