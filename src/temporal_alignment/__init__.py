"""Contracts and reproducible baselines for long-range temporal alignment."""

from .contracts import (
    DatasetManifest,
    HistoricalDate,
    HistoricalYear,
    TemporalExample,
    TemporalInterval,
)
from .encoders import (
    FourierTimeEncoder,
    GregorianScalarEncoder,
    HybridCycleMonotonicEncoder,
    OperationalRadialTicktackEncoder,
    PublishedTicktackEncoder,
)

__all__ = [
    "DatasetManifest",
    "FourierTimeEncoder",
    "GregorianScalarEncoder",
    "HistoricalDate",
    "HistoricalYear",
    "HybridCycleMonotonicEncoder",
    "OperationalRadialTicktackEncoder",
    "PublishedTicktackEncoder",
    "TemporalExample",
    "TemporalInterval",
]

__version__ = "0.1.0"
