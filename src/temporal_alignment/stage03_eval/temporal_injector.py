"""Mathematical temporal feature generation and Seam embedding injection.

Implements the formal contract:
    E'_{b, j, :} = E_{b, j, :} + M_{b, j} * Delta_b
where:
    Delta_b = gate(alpha) * LayerNorm(Linear(v_40(t_b)))
and M is the temporal modulation mask (M=0 on all PAD/special tokens).

Supports both legacy sigmoid gate (torch.sigmoid(alpha)) and linear gate (alpha * norm)
with full autograd, device, dtype, and batch safety.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import TemporalPolicy, YearMention


def compute_fone_32d(year_f: float) -> List[float]:
    """Compute 32-dimensional Fourier Numerical Encoding (FoNE) across 16 dyadic periods."""
    periods = [10.0 * (1000.0 ** (i / 15.0)) for i in range(16)]
    feats: List[float] = []
    for p in periods:
        omega = 2.0 * math.pi / p
        feats.append(math.sin(omega * year_f))
        feats.append(math.cos(omega * year_f))
    return feats


def compute_fone_powers10_32d(year_f: float) -> List[float]:
    """Compute 32-dimensional FoNE across 16 log-spaced powers-of-10 periods (1 to 10,000 years)."""
    periods = [10.0 ** (4.0 * i / 15.0) for i in range(16)]
    feats: List[float] = []
    for p in periods:
        omega = 2.0 * math.pi / p
        feats.append(math.sin(omega * year_f))
        feats.append(math.cos(omega * year_f))
    return feats


def compute_canchi_8d(
    year_int: int,
    *,
    radius_mode: str = "linear",
    scale: float = 1000.0,
) -> List[float]:
    """Compute 8-dimensional Sexagenary (Can-Chi) cycle representation."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    offset = year_int - 4
    stem_idx = offset % 10
    branch_idx = offset % 12
    cycle_idx = offset % 60
    theta_stem = 2.0 * math.pi * float(stem_idx) / 10.0
    theta_branch = 2.0 * math.pi * float(branch_idx) / 12.0
    theta_cycle = 2.0 * math.pi * float(cycle_idx) / 60.0
    if radius_mode == "linear":
        radius = 1.0 + float(year_int - 2000) / 60.0
    elif radius_mode == "asinh":
        radius = 1.0 + 0.25 * math.asinh(float(year_int - 2000) / scale)
    else:
        raise ValueError(f"Unsupported radius_mode: {radius_mode}")
    return [
        math.sin(theta_stem),
        math.cos(theta_stem),
        math.sin(theta_branch),
        math.cos(theta_branch),
        math.sin(theta_cycle),
        math.cos(theta_cycle),
        float(radius),
        float(cycle_idx) / 60.0,
    ]


def compute_fixed_fone_canchi_40d(year: int) -> List[float]:
    """Concatenate 32D FoNE and 8D Can-Chi into canonical 40D temporal feature vector."""
    return compute_fone_32d(float(year)) + compute_canchi_8d(year)


def compute_temporal_40d(
    year: int,
    *,
    fone_mode: str = "powers10",
    radius_mode: str = "asinh",
    scale: float = 1000.0,
) -> List[float]:
    """Compute 40D temporal feature vector with configurable FoNE and radial compression."""
    if fone_mode == "dyadic":
        fone = compute_fone_32d(float(year))
    elif fone_mode == "powers10":
        fone = compute_fone_powers10_32d(float(year))
    else:
        raise ValueError(f"Unsupported fone_mode: {fone_mode}")
    canchi = compute_canchi_8d(year, radius_mode=radius_mode, scale=scale)
    return fone + canchi


def compute_theta_z(
    year: int,
    *,
    z_mode: str = "linear",
    scale: float = 1000.0,
    anchor_year: int = 1947,
) -> Tuple[float, float]:
    """Compute continuous phase angle theta and normalized epoch coordinate z for MaT-LoRA."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    offset = year - 3
    cycle = offset % 60
    theta = float(cycle) * 6.0 * (math.pi / 180.0)
    if z_mode == "linear":
        z = float(year - anchor_year) / 75.0
    elif z_mode == "asinh":
        z = math.asinh(float(year - anchor_year) / scale)
    else:
        raise ValueError(f"Unsupported z_mode: {z_mode}")
    return theta, z


class GatedResidualProjection(nn.Module):
    """Temporal projection layer mapping 40D time representation into transformer hidden space."""

    def __init__(
        self,
        in_features: int = 40,
        out_features: int = 2048,
        gate_mode: str = "linear",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gate_mode = gate_mode
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))

        # Kaiming/normal initialization matching training script
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, v_time: torch.Tensor) -> torch.Tensor:
        v_in = v_time.to(self.linear.weight.device, dtype=self.linear.weight.dtype)
        h = self.linear(v_in)
        h = self.norm(h)
        if self.gate_mode == "sigmoid":
            gate = torch.sigmoid(self.alpha)
            return gate * h
        elif self.gate_mode == "linear":
            return self.alpha * h
        else:
            raise ValueError(f"Unsupported gate_mode: {self.gate_mode}")

    def get_effective_gate(self) -> float:
        """Return numerical value of current gate scalar."""
        with torch.no_grad():
            if self.gate_mode == "sigmoid":
                return float(torch.sigmoid(self.alpha).item())
            return float(self.alpha.item())


def inject_temporal_features(
    inputs_embeds: torch.Tensor,
    seam_proj: nn.Module,
    v40_tensor: torch.Tensor,
    temporal_mask: torch.Tensor,
    policy: TemporalPolicy,
) -> torch.Tensor:
    """Inject projected temporal representation into inputs_embeds modulated by temporal_mask.

    Mathematical contract:
        E' = E + (M.unsqueeze(-1) * Delta.unsqueeze(1))
    Gradient-safe, non-in-place, device- and dtype-aligned.
    """
    if policy == TemporalPolicy.SEAM_OFF:
        return inputs_embeds

    B, S, D = inputs_embeds.shape
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype

    # Validate tensor compatibility
    if v40_tensor.shape[0] != B or v40_tensor.shape[1] != 40:
        raise ValueError(f"v40_tensor shape {v40_tensor.shape} != expected ({B}, 40).")
    if temporal_mask.shape != (B, S):
        raise ValueError(f"temporal_mask shape {temporal_mask.shape} != expected ({B}, {S}).")

    # Project v40 into hidden dimension D and match inputs_embeds device/dtype
    v40_dev = v40_tensor.to(device=device, dtype=dtype)
    delta = seam_proj(v40_dev).to(device=device, dtype=dtype)  # (B, D)

    # Modulate delta by temporal mask M
    mask_expanded = temporal_mask.to(device=device, dtype=dtype).unsqueeze(-1)  # (B, S, 1)
    delta_expanded = delta.unsqueeze(1)  # (B, 1, D)

    # Non-in-place residual addition preserving autograd graph
    injected_embeds = inputs_embeds + (mask_expanded * delta_expanded)
    return injected_embeds


def inject_multi_year_temporal_features(
    inputs_embeds: torch.Tensor,
    seam_proj: nn.Module,
    mentions_per_item: Sequence[Sequence[YearMention]],
    *,
    padding_side: str = "right",
    attention_mask: Optional[torch.Tensor] = None,
    fone_mode: str = "powers10",
    radius_mode: str = "asinh",
    scale: float = 1000.0,
) -> torch.Tensor:
    """Inject per-mention 40D temporal representations at every resolved year span.

    Each resolved YearMention m in batch item b gets its own v40(m.year) projected
    through seam_proj and added to its own token span [token_start, token_end) with
    sum-preserving weight 1.0 / span_len, strictly masked against PAD tokens.
    """
    B, S, D = inputs_embeds.shape
    if len(mentions_per_item) != B:
        raise ValueError(f"mentions_per_item length {len(mentions_per_item)} != batch size {B}.")

    device = inputs_embeds.device
    dtype = inputs_embeds.dtype

    all_mentions: List[Tuple[int, int, int, int]] = []
    v40_rows: List[List[float]] = []

    for b, mentions in enumerate(mentions_per_item):
        num_pads = 0
        if padding_side == "left" and attention_mask is not None:
            num_pads = int((attention_mask[b] == 0).sum().item())
        for m in mentions:
            if not m.is_resolved():
                continue
            t_start = max(0, min(S, int(m.token_start) + num_pads))
            t_end = max(0, min(S, int(m.token_end) + num_pads))
            if t_end > t_start:
                all_mentions.append((b, t_start, t_end, m.year))
                v40_rows.append(
                    compute_temporal_40d(
                        m.year,
                        fone_mode=fone_mode,
                        radius_mode=radius_mode,
                        scale=scale,
                    )
                )

    if not all_mentions:
        return inputs_embeds

    v40_tensor = torch.tensor(v40_rows, dtype=torch.float32, device=device)
    deltas = seam_proj(v40_tensor).to(device=device, dtype=dtype)  # (N_mentions, D)

    delta_grid = torch.zeros_like(inputs_embeds)
    for idx, (b, t_start, t_end, _) in enumerate(all_mentions):
        span_len = float(t_end - t_start)
        weight = 1.0 / span_len
        delta_grid[b, t_start:t_end, :] = delta_grid[b, t_start:t_end, :] + weight * deltas[idx].unsqueeze(0)

    if attention_mask is not None:
        valid_mask = attention_mask.to(device=device, dtype=dtype).unsqueeze(-1)
        delta_grid = delta_grid * valid_mask

    return inputs_embeds + delta_grid
