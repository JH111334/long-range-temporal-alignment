"""Unit tests for mechanistic Cosine Collapse reproduction and Scale Normalization recovery.

Verifies:
1. Raw sinusoidal TE has Euclidean norm significantly larger than typical word embeddings.
2. Broadcasting unscaled TE across word embeddings causes pairwise cosine to collapse to > 0.98.
3. Scale normalization (1/d) brings pairwise cosine back to baseline linguistic levels (< 0.20).
"""

from __future__ import annotations

import math
import torch
import pytest


def make_synthetic_ticktack_te(year: int, hidden_dim: int = 896) -> torch.Tensor:
    cycle_index = float((year - 3) % 60)
    theta = (cycle_index * 6.0) * (math.pi / 180.0)
    r = 0.5 + 0.5 * theta
    x = r * math.cos(theta)
    y = r * math.sin(theta)

    dim_indices = torch.arange(0, hidden_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (dim_indices / float(hidden_dim)))

    x_inp = x * inv_freq
    te_x = torch.stack([torch.sin(x_inp), torch.cos(x_inp)], dim=-1).flatten()

    y_inp = y * inv_freq
    te_y = torch.stack([torch.sin(y_inp), torch.cos(y_inp)], dim=-1).flatten()

    return te_x + te_y


def test_cosine_collapse_and_scale_fix():
    hidden_dim = 896
    num_tokens = 50

    # Synthetic word embeddings with typical norm around 0.5 - 1.0
    torch.manual_seed(42)
    embeddings = torch.randn(num_tokens, hidden_dim) * 0.03
    embed_norms = embeddings.norm(dim=-1)
    assert embed_norms.mean().item() < 2.0

    te_raw = make_synthetic_ticktack_te(1984, hidden_dim=hidden_dim)
    te_norm = te_raw.norm().item()
    # Theoretical norm is ~21-42
    assert te_norm > 20.0

    # 1. Baseline pairwise cosine among raw embeddings
    normed_base = embeddings / embeddings.norm(dim=-1, keepdim=True)
    cos_base = torch.mm(normed_base, normed_base.t())
    mask = ~torch.eye(num_tokens, dtype=torch.bool)
    mean_cos_base = cos_base[mask].mean().item()
    assert abs(mean_cos_base) < 0.20

    # 2. Under broadcast addition of unscaled TE
    broadcast_raw = embeddings + te_raw.unsqueeze(0)
    normed_raw = broadcast_raw / broadcast_raw.norm(dim=-1, keepdim=True)
    cos_raw = torch.mm(normed_raw, normed_raw.t())
    mean_cos_raw = cos_raw[mask].mean().item()
    # Must be collapsed to > 0.98
    assert mean_cos_raw > 0.98

    # 3. Under scale normalization (1/d)
    te_scaled = te_raw * (1.0 / float(hidden_dim))
    broadcast_scaled = embeddings + te_scaled.unsqueeze(0)
    normed_scaled = broadcast_scaled / broadcast_scaled.norm(dim=-1, keepdim=True)
    cos_scaled = torch.mm(normed_scaled, normed_scaled.t())
    mean_cos_scaled = cos_scaled[mask].mean().item()
    # Must recover to baseline levels (< 0.20)
    assert mean_cos_scaled < 0.20
