"""TickTack-inspired 0.5B Baseline (Adapted Reimplementation).

Paper reference:
"Temporal Alignment of LLMs through Cycle Encoding for Long-Range Time Representations"
Xue Han et al., arXiv:2503.04150v3 (revised 2025-10-21).
"""

from __future__ import annotations

from .temporal_encoding import (
    compute_polar_coordinates,
    compute_sinusoidal_te,
    get_ticktack_temporal_embedding,
    inject_temporal_embedding,
    polar_to_cartesian,
    year_to_sexagenary_cycle,
)

__all__ = [
    "year_to_sexagenary_cycle",
    "compute_polar_coordinates",
    "polar_to_cartesian",
    "compute_sinusoidal_te",
    "get_ticktack_temporal_embedding",
    "inject_temporal_embedding",
]
