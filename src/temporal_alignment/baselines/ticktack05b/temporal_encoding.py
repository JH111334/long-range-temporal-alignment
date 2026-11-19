"""Temporal Encoding Implementation for TickTack v3 (arXiv:2503.04150v3).

Covers:
- Eq. 1: Gregorian year to Sexagenary cycle index C(y) in [0, 59].
- Eq. 2: Polar coordinate angle theta and radius r (literal vs adapted monotonic).
- Eq. 3: Cartesian conversion (x, y).
- Eq. 4: Sinusoidal Temporal Encoding TE(x), TE(y) in R^d.
- Eq. 5: Embedding injection h' = h + TE(x) + TE(y) at year-associated tokens.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple, Union

import torch


def year_to_sexagenary_cycle(year: int) -> int:
    """Map a historical year to sexagenary cycle index in [0, 59].

    Eq. 1 of Xue Han et al. (2025):
        C(y) = (y - 4) mod 60 for CE (y > 0).

    For BCE years (y < 0), astronomical convention applies (no year zero):
        1 BCE corresponds to astronomical year 0 (y_astro = y + 1).
        C(y) = (y + 1 - 4) mod 60 = (y - 3) mod 60.

    Args:
        year: Integer historical year (cannot be zero).

    Returns:
        Integer sexagenary index in [0, 59].
    """
    if year == 0:
        raise ValueError("Year zero does not exist in standard historical BCE/CE chronology.")
    if year > 0:
        return (year - 4) % 60
    return (year - 3) % 60


def compute_polar_coordinates(
    year: int,
    mode: str = "adapted_monotonic",
    alpha: float = 1.0,
    beta: float = 0.1,
    gamma: float = 0.5,
) -> Tuple[float, float]:
    """Compute polar angle theta and radius r.

    Eq. 2:
        theta = 2 * pi * C(y) / 60

    Radius Modes:
    - 'literal_eq2': r = alpha + beta * theta
        Direct literal formula from paper. Note: produces identical r for years
        spaced by multiples of 60 (e.g., 1965 and 2025).
    - 'adapted_monotonic': r = alpha + beta * theta + gamma * epoch_index
        Where epoch_index = floor((year - 4) / 60) for CE.
        Distinguishes distinct cycles across long-range temporal spans.

    Returns:
        Tuple of (radius, theta_radians).
    """
    cycle_idx = year_to_sexagenary_cycle(year)
    theta = 2.0 * math.pi * float(cycle_idx) / 60.0

    if mode == "literal_eq2":
        radius = alpha + beta * theta
    elif mode == "adapted_monotonic":
        if year > 0:
            epoch = math.floor((year - 4) / 60.0)
        else:
            epoch = math.floor((year - 3) / 60.0)
        radius = alpha + beta * theta + gamma * float(epoch)
    else:
        raise ValueError(f"Unknown radius mode: {mode}. Choose 'literal_eq2' or 'adapted_monotonic'.")

    return radius, theta


def polar_to_cartesian(radius: float, theta: float) -> Tuple[float, float]:
    """Convert polar coordinates (r, theta) to Cartesian (x, y) per Eq. 3."""
    x = radius * math.cos(theta)
    y = radius * math.sin(theta)
    return x, y


def compute_sinusoidal_te(
    val: float,
    dim: int = 896,
    max_timescale: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute sinusoidal temporal encoding vector in R^dim per Eq. 4.

    TE(v)_{2i}   = sin(v / 10000^(2i/dim))
    TE(v)_{2i+1} = cos(v / 10000^(2i/dim))

    Args:
        val: Coordinate scalar (x or y).
        dim: Hidden dimension size (defaults to 896 for Qwen2.5-0.5B).
        max_timescale: Base timescale constant (default 10000.0).
        device: Torch device.
        dtype: Output data type (defaults to torch.float32).

    Returns:
        Tensor of shape (dim,).
    """
    if dim % 2 != 0:
        raise ValueError(f"Dimension {dim} must be even for sinusoidal encoding.")

    half_dim = dim // 2
    i = torch.arange(half_dim, dtype=torch.float32, device=device)
    freqs = torch.exp(-math.log(max_timescale) * (2.0 * i / float(dim)))
    args = float(val) * freqs

    sin_part = torch.sin(args)
    cos_part = torch.cos(args)

    te = torch.empty(dim, dtype=dtype, device=device)
    te[0::2] = sin_part.to(dtype)
    te[1::2] = cos_part.to(dtype)
    return te


def get_ticktack_temporal_embedding(
    year: int,
    dim: int = 896,
    mode: str = "adapted_monotonic",
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute combined temporal vector TE(x) + TE(y) for a given year per Eq. 5.

    Returns:
        Tensor of shape (dim,).
    """
    radius, theta = compute_polar_coordinates(year, mode=mode)
    x, y = polar_to_cartesian(radius, theta)

    te_x = compute_sinusoidal_te(x, dim=dim, device=device, dtype=dtype)
    te_y = compute_sinusoidal_te(y, dim=dim, device=device, dtype=dtype)
    return te_x + te_y


def find_year_token_indices(
    input_ids: torch.Tensor,
    year: int,
    tokenizer: Any,
) -> List[int]:
    """Locate token indices in input_ids corresponding to the year substring.

    Args:
        input_ids: 1D Tensor of token IDs.
        year: Target year to search for.
        tokenizer: HuggingFace tokenizer.

    Returns:
        List of integer token positions where year token(s) occur.
    """
    year_str = str(year)
    ids_list = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)

    # Candidate tokenizations of the year (with or without leading space)
    candidate_tokens = [
        tokenizer.encode(year_str, add_special_tokens=False),
        tokenizer.encode(f" {year_str}", add_special_tokens=False),
    ]

    matched_indices: List[int] = []
    for cand in candidate_tokens:
        if not cand:
            continue
        c_len = len(cand)
        for idx in range(len(ids_list) - c_len + 1):
            if ids_list[idx : idx + c_len] == cand:
                matched_indices.extend(range(idx, idx + c_len))

    # Deduplicate while preserving order
    unique_indices = sorted(list(set(matched_indices)))
    return unique_indices


def inject_temporal_embedding(
    inputs_embeds: torch.Tensor,
    token_indices: List[int],
    te_vector: torch.Tensor,
) -> torch.Tensor:
    """Add temporal embedding vector to specific token positions in inputs_embeds per Eq. 5.

    h' = h + TE(x) + TE(y)

    Args:
        inputs_embeds: Tensor of shape (batch, seq_len, hidden_dim).
        token_indices: List of token positions to inject into.
        te_vector: Tensor of shape (hidden_dim,).

    Returns:
        Modified inputs_embeds tensor with temporal injection.
    """
    if not token_indices:
        return inputs_embeds

    # Clone to avoid in-place graph issues during backward pass
    modified = inputs_embeds.clone()
    te_reshaped = te_vector.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

    for idx in token_indices:
        if 0 <= idx < modified.shape[1]:
            modified[:, idx, :] = modified[:, idx, :] + te_reshaped

    return modified
