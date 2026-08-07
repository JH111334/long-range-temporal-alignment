"""Dependency-free temporal representation baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .calendar import anchored_cycle_index, published_sexagenary_index
from .contracts import HistoricalYear
from .errors import ConfigurationError


class TimeEncoder(Protocol):
    """Contract implemented by every temporal encoder."""

    encoder_id: str
    evidence_class: str

    def encode(self, year: HistoricalYear) -> tuple[float, ...]:
        """Encode one historical year into a fixed-width vector."""


@dataclass(frozen=True, slots=True)
class GregorianScalarEncoder:
    """Simple monotonic baseline with a bias term that avoids zero vectors."""

    scale: float = 2_000.0
    encoder_id: str = "gregorian-scalar-v1"
    evidence_class: str = "baseline"

    def encode(self, year: HistoricalYear) -> tuple[float, ...]:
        if self.scale <= 0:
            raise ConfigurationError("scale must be positive")
        return (math.tanh(year.astronomical / self.scale), 1.0)


@dataclass(frozen=True, slots=True)
class FourierTimeEncoder:
    """Culture-neutral multi-period Fourier baseline."""

    periods: tuple[int, ...] = (10, 60, 100, 400)
    include_linear: bool = False
    scale: float = 2_000.0
    encoder_id: str = "fourier-time-v1"
    evidence_class: str = "baseline"

    def encode(self, year: HistoricalYear) -> tuple[float, ...]:
        if not self.periods or any(period < 2 for period in self.periods):
            raise ConfigurationError("periods must contain integers >= 2")
        values: list[float] = []
        if self.include_linear:
            values.append(math.tanh(year.astronomical / self.scale))
        for period in self.periods:
            angle = 2.0 * math.pi * year.astronomical / period
            values.extend((math.sin(angle), math.cos(angle)))
        return tuple(values)


@dataclass(frozen=True, slots=True)
class PublishedTicktackEncoder:
    """Operationalize Equations (1)–(3) as printed in arXiv:2503.04150v3.

    Constant alpha/beta make years in the same 60-year class identical. This
    behavior is retained deliberately as a testable published-equation baseline.
    """

    alpha: float = 0.75
    beta: float = 0.75
    encoder_id: str = "published-ticktack-eq1-3-v1"
    evidence_class: str = "published-equation"

    def encode(self, year: HistoricalYear) -> tuple[float, ...]:
        cycle_index = published_sexagenary_index(year)
        theta = 2.0 * math.pi * cycle_index / 60.0
        radius = self.alpha + self.beta * theta
        return (radius * math.cos(theta), radius * math.sin(theta))


@dataclass(frozen=True, slots=True)
class OperationalRadialTicktackEncoder:
    """Explicit radial assumption that distinguishes repeated 60-year classes."""

    anchor_year: int = 1984
    radius_base: float = 1.5
    radial_weight: float = 0.25
    time_scale: float = 1_000.0
    encoder_id: str = "operational-radial-ticktack-v1"
    evidence_class: str = "operational-assumption"

    def encode(self, year: HistoricalYear) -> tuple[float, ...]:
        if self.time_scale <= 0:
            raise ConfigurationError("time_scale must be positive")
        cycle_index = anchored_cycle_index(year, anchor_year=self.anchor_year)
        theta = 2.0 * math.pi * cycle_index / 60.0
        radius = self.radius_base + self.radial_weight * math.asinh(
            year.astronomical / self.time_scale
        )
        return (radius * math.cos(theta), radius * math.sin(theta))


@dataclass(frozen=True, slots=True)
class HybridCycleMonotonicEncoder:
    """Candidate representation combining a monotonic axis and cycle features."""

    periods: tuple[int, ...] = (10, 60, 100, 400)
    scale: float = 2_000.0
    encoder_id: str = "hybrid-cycle-monotonic-v1"
    evidence_class: str = "proposed-method"

    def encode(self, year: HistoricalYear) -> tuple[float, ...]:
        fourier = FourierTimeEncoder(
            periods=self.periods,
            include_linear=False,
            scale=self.scale,
        ).encode(year)
        linear = math.tanh(year.astronomical / self.scale)
        return (linear, *fourier)


def encoder_from_name(name: str) -> TimeEncoder:
    """Build a supported encoder from a stable configuration name."""
    factories = {
        "gregorian_scalar": GregorianScalarEncoder,
        "fourier": FourierTimeEncoder,
        "published_ticktack": PublishedTicktackEncoder,
        "operational_radial_ticktack": OperationalRadialTicktackEncoder,
        "hybrid": HybridCycleMonotonicEncoder,
    }
    try:
        return factories[name]()
    except KeyError as exc:
        raise ConfigurationError(f"unsupported encoder: {name}") from exc
