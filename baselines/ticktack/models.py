"""
Ticktack Baseline Architecture and Models (Han et al., 2025).

Implements:
- Sexagenary Can Chi cycle conversion (Eq. 1)
- Polar coordinates and Cartesian projections (Eq. 2-3)
- Sinusoidal Temporal Encoding in R^{B x 1 x hidden_dim} (Eq. 4)
- Scale normalization modes:
  * 'raw': Direct addition of sinusoidal TE (Eq. 5, causes cosine collapse)
  * 'scaled': 1/d energy-normalized addition preserving semantic variance
  * 'gated': Learnable/scalar gating factor dampening temporal amplitude
- LoRALinear: Low-Rank Adaptation module (Hu et al., 2021) for linear projection layers
- CONTROL_CONFIGS: 4-way disentanglement control matrix specifications (C1-C4)
"""

import math
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

CONTROL_CONFIGS: Dict[str, Dict[str, Union[str, float]]] = {
    'C1': {
        'name': 'C1_raw_broadcast',
        'te_mode': 'broadcast',
        'te_scale_mode': 'raw',
        'sigma': 0.5,
        'description': 'Raw Broadcast (Paper Formula, sigma = 0.5)'
    },
    'C2': {
        'name': 'C2_raw_date_token',
        'te_mode': 'date_token',
        'te_scale_mode': 'raw',
        'sigma': 0.5,
        'description': 'Raw Date-token only (Localized Injection, sigma = 0.5)'
    },
    'C3': {
        'name': 'C3_raw_sigma0',
        'te_mode': 'broadcast',
        'te_scale_mode': 'raw',
        'sigma': 0.0,
        'description': 'Raw with sigma = 0 (Pure NTP, no intra loss)'
    },
    'C4': {
        'name': 'C4_pt_intra_loss',
        'te_mode': 'none',
        'te_scale_mode': 'none',
        'sigma': 0.5,
        'description': 'PT + Intra loss (No TE, sigma = 0.5)'
    },
}

def compute_canchi_cycle(year: int) -> int:
    """
    Gregorian year to 60-year sexagenary cycle index (Eq. 1, Han et al., 2025).
    
    Index 0 corresponds to Giap Ty (Year 4 AD or 1984 AD is index 1; Year 3 AD is index 0).
    Properly handles BC years (year <= 0) and edge cases.
    """
    if year >= 4:
        return (year - 3) % 60
    elif year > 0:
        return (60 - abs(year - 3)) % 60
    else:
        return (60 - abs(year) - 2) % 60

def compute_batch_te(
    years: Union[torch.Tensor, List[int], int],
    hidden_dim: int = 896,
    scale_mode: str = 'raw',
    alpha: float = 0.5,
    beta: float = 0.5,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    gating_factor: float = 0.01,
) -> torch.Tensor:
    """
    Computes vectorized Sinusoidal Temporal Encoding in R^{B x 1 x hidden_dim} (Eq. 2, 3, 4).
    
    Args:
        years: 1D tensor, list, or scalar of Gregorian calendar years [B].
        hidden_dim: Embedding dimension of the transformer backbone (e.g. 896 for Qwen2.5-0.5B).
        scale_mode: 'raw' (unscaled, norm ~40-42), 'scaled' (scaled by 1/d), or 'gated' (scaled by gating_factor).
        alpha: Archimedean spiral offset parameter (default 0.5).
        beta: Archimedean spiral growth parameter (default 0.5).
        device: Target torch device. If None, inherits from years.device if tensor, else cpu.
        dtype: Output data type (default torch.float32).
        gating_factor: Multiplier used when scale_mode is 'gated' (default 0.01).
        
    Returns:
        Tensor of shape [B, 1, hidden_dim].
    """
    if isinstance(years, (int, float)):
        years = torch.tensor([years], device=device or "cpu")
    elif not isinstance(years, torch.Tensor):
        years = torch.tensor(years, device=device or "cpu")

    if device is None:
        device = years.device
        
    years = years.view(-1).to(device=device)
    cycle_indices = ((years.to(torch.int64) - 3) % 60).to(torch.float32)
    theta = (cycle_indices * 6.0) * (math.pi / 180.0)
    r = alpha + beta * theta
    x = (r * torch.cos(theta)).view(-1, 1).to(device)
    y = (r * torch.sin(theta)).view(-1, 1).to(device)
    
    dim_indices = torch.arange(0, hidden_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (dim_indices / float(hidden_dim))).unsqueeze(0)
    
    x_inp = x * inv_freq
    te_x = torch.stack([torch.sin(x_inp), torch.cos(x_inp)], dim=-1).flatten(start_dim=-2)
    
    y_inp = y * inv_freq
    te_y = torch.stack([torch.sin(y_inp), torch.cos(y_inp)], dim=-1).flatten(start_dim=-2)
    
    te_raw = (te_x + te_y).unsqueeze(1).to(dtype=dtype)  # [B, 1, hidden_dim]
    
    if scale_mode == 'raw':
        return te_raw
    elif scale_mode == 'scaled':
        scale = 1.0 / float(hidden_dim)
        return (te_raw * scale).to(dtype=dtype)
    elif scale_mode == 'gated':
        return (te_raw * gating_factor).to(dtype=dtype)
    else:
        raise ValueError(f"Unknown scale_mode: '{scale_mode}'. Expected 'raw', 'scaled', or 'gated'.")

def compute_te_vector(
    years: torch.Tensor,
    hidden_dim: int = 896,
    alpha: float = 0.5,
    beta: float = 0.5,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    scale_mode: str = "raw",
    gating_factor: float = 0.01,
) -> torch.Tensor:
    """Compatibility alias for compute_batch_te with explicit default device='cpu'."""
    return compute_batch_te(
        years=years,
        hidden_dim=hidden_dim,
        scale_mode=scale_mode,
        alpha=alpha,
        beta=beta,
        device=device,
        dtype=dtype,
        gating_factor=gating_factor,
    )

class LoRALinear(nn.Module):
    """
    Standard Low-Rank Adaptation (LoRA) module (Hu et al., 2021).
    Freezes base linear layer and introduces low-rank trainable matrices A and B.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank) if rank > 0 else 1.0
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        self.reset_parameters()

    def reset_parameters(self):
        if self.rank > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        mod = cls(linear.in_features, linear.out_features, rank=rank, alpha=alpha, bias=(linear.bias is not None))
        mod.base_layer.weight.data.copy_(linear.weight.data)
        mod.base_layer.weight.requires_grad = False
        if linear.bias is not None:
            mod.base_layer.bias.data.copy_(linear.bias.data)
            mod.base_layer.bias.requires_grad = False
        mod.to(device=linear.weight.device, dtype=linear.weight.dtype)
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.rank == 0:
            return base_out
        lora_out = F.linear(
            F.linear(x, self.lora_A.to(dtype=x.dtype, device=x.device)),
            self.lora_B.to(dtype=x.dtype, device=x.device)
        ) * self.scaling
        return base_out + lora_out

def apply_lora_to_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    rank: int = 8,
    alpha: float = 16.0,
    freeze_base: bool = True,
) -> Dict[str, LoRALinear]:
    """Replaces specified Linear layers in a model with LoRALinear adapters."""
    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj']
    adapted = {}
    candidates = []
    for name, module in model.named_modules():
        parts = name.split('.')
        leaf = parts[-1]
        if leaf in target_modules and isinstance(module, nn.Linear):
            candidates.append((name, parts, module))

    for name, parts, module in candidates:
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        layer_name = parts[-1]
        adapter = LoRALinear.from_linear(module, rank=rank, alpha=alpha)
        setattr(parent, layer_name, adapter)
        adapted[name] = adapter
    return adapted
