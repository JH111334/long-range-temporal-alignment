"""Immutable 40D FoNE x Can-Chi Feature Generator (FIXED_FONE_CANCHI_V1).

Contract Specification:
- FoNE 32D: 16 logarithmic periods from 10.0 to 10000.0, [cos(omega*y), sin(omega*y)]
- Can-Chi 8D: [x, y, cos(theta), sin(theta), radius, stem_norm, branch_norm, cycle_norm]
- Total dimension: 40
- Verified against golden vectors in artifacts/releases/FCLEARN_105/feature_manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Sequence


THIEN_CAN: tuple[str, ...] = (
    "Giáp", "Ất", "Bính", "Đinh", "Mậu",
    "Kỷ", "Canh", "Tân", "Nhâm", "Quý",
)

DIA_CHI: tuple[str, ...] = (
    "Tý", "Sửu", "Dần", "Mão", "Thìn", "Tỵ",
    "Ngọ", "Mùi", "Thân", "Dậu", "Tuất", "Hợi",
)


def compute_fone_32d(
    year: float,
    num_frequencies: int = 16,
    min_period: float = 10.0,
    max_period: float = 10000.0,
) -> list[float]:
    """Compute 32D Fourier Numerical Embedding (FoNE) for a given year."""
    if num_frequencies <= 0:
        raise ValueError(f"num_frequencies must be positive, got {num_frequencies}")
    if min_period <= 0 or max_period <= min_period:
        raise ValueError("Invalid min_period or max_period")

    ln_min = math.log(min_period)
    ln_max = math.log(max_period)
    step = (ln_max - ln_min) / (num_frequencies - 1) if num_frequencies > 1 else 0.0

    features: list[float] = []
    for k in range(num_frequencies):
        period = math.exp(ln_min + k * step)
        omega = 2.0 * math.pi / period
        features.append(math.cos(omega * year))
        features.append(math.sin(omega * year))
    return features


def compute_canchi_8d(
    year: int,
    t0: int = 1900,
) -> list[float]:
    """Compute 8D Sexagenary (Can-Chi) geometrical embedding for a given year."""
    offset = year - 4
    cycle_idx = offset % 60
    stem_idx = offset % 10
    branch_idx = offset % 12

    theta = 2.0 * math.pi * float(cycle_idx) / 60.0
    radius = 1.0 + float(year - t0) / 60.0

    x = radius * math.cos(theta)
    y = radius * math.sin(theta)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    stem_norm = float(stem_idx) / 10.0
    branch_norm = float(branch_idx) / 12.0
    cycle_norm = float(cycle_idx) / 60.0

    return [
        float(x),
        float(y),
        float(cos_theta),
        float(sin_theta),
        float(radius),
        float(stem_norm),
        float(branch_norm),
        float(cycle_norm),
    ]


def compute_fixed_fone_canchi_40d(year: int) -> list[float]:
    """Generate canonical 40D vector [FoNE 32D || Can-Chi 8D]."""
    fone = compute_fone_32d(float(year), num_frequencies=16, min_period=10.0, max_period=10000.0)
    canchi = compute_canchi_8d(year, t0=1900)
    return fone + canchi


def extract_canchi_angles_from_40d(vec40: Sequence[float]) -> tuple[float, float]:
    """Extract (theta, z) from canonical 40D vector for DualCycleCoreGenerator.
    
    In the 40D vector, indices 32..39 are Can-Chi 8D:
    - index 34: cos(theta)
    - index 35: sin(theta)
    - index 36: radius = 1.0 + (year - 1900)/60.0 -> z = radius - 1.0
    
    Returns:
        theta: float in radians [-pi, pi]
        z: float macro-epoch coordinate
    """
    cos_theta = vec40[34]
    sin_theta = vec40[35]
    radius = vec40[36]
    theta = math.atan2(sin_theta, cos_theta)
    z = radius - 1.0
    return theta, z


def verify_golden_manifest(manifest_path: str) -> bool:
    """Verify generated 40D vectors against the immutable manifest."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    goldens = data["frozen_representation_contract"]["golden_vectors"]
    for year_str, entry in goldens.items():
        year = int(year_str)
        computed = compute_fixed_fone_canchi_40d(year)
        expected = entry["vector"]
        if len(computed) != 40 or len(expected) != 40:
            return False
        for c, e in zip(computed, expected):
            if abs(c - e) > 1e-5:
                return False
    return True
