"""Periodic Dual-Cycle MaT-LoRA for Transformer Linear Modules.

Inherited Architecture:
- Yao et al. (arXiv:2602.11965): Matrix Time-LoRA (MaT-LoRA) Delta W(t) = B * F(t) * A.
- Dual-Cycle Variant: Micro-cycle core F_cycle(theta) + Macro-epoch core F_epoch(z).
- Base Model W_0 strictly frozen.
- Target: 72 MLP Linear layers (gate_proj, up_proj, down_proj across 24 layers in Qwen2.5-0.5B-Instruct).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _ModuleBase = nn.Module
except ImportError:
    torch = None
    nn = None
    F = None
    _ModuleBase = object


class DualCycleCoreGenerator(_ModuleBase):
    """Dynamic r x r core matrix generator F(theta, z) = F_base + F_cycle(theta) + F_epoch(z)."""

    def __init__(
        self,
        rank: int = 8,
        num_harmonics: int = 2,
        use_epoch_poly: bool = True,
    ) -> None:
        if nn is None:
            raise RuntimeError("PyTorch required")
        super().__init__()
        self.rank = rank
        self.num_harmonics = num_harmonics
        self.use_epoch_poly = use_epoch_poly

        # Static baseline identity component F_base = I_r (frozen)
        self.register_buffer("f_base", torch.eye(rank, dtype=torch.float32))

        # Fourier harmonic basis matrices for micro-cycle F_cycle(theta)
        std = 1.0 / math.sqrt(rank) if rank > 0 else 1.0
        self.cos_basis = nn.Parameter(torch.randn(num_harmonics, rank, rank) * (std * 0.1))
        self.sin_basis = nn.Parameter(torch.randn(num_harmonics, rank, rank) * (std * 0.1))

        # Macro-epoch progression matrices for F_epoch(z)
        self.epoch_linear = nn.Parameter(torch.randn(rank, rank) * (std * 0.1))
        if use_epoch_poly:
            self.epoch_quad = nn.Parameter(torch.randn(rank, rank) * (std * 0.01))
        else:
            self.register_parameter("epoch_quad", None)

    def forward(
        self,
        theta: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute core matrix F of shape (batch_size, rank, rank).
        
        Args:
            theta: Tensor of angles in radians, shape (batch_size,) or scalar
            z: Tensor of macro-epoch coordinates, shape (batch_size,) or scalar
            
        Returns:
            F of shape (batch_size, rank, rank)
        """
        if theta.dim() == 0:
            theta = theta.view(1)
        if z.dim() == 0:
            z = z.view(1)

        batch_size = theta.shape[0]
        device = self.cos_basis.device
        dtype = self.cos_basis.dtype

        theta = theta.to(device=device, dtype=dtype)
        z = z.to(device=device, dtype=dtype)

        # 1. Base identity component: (batch_size, rank, rank)
        f_core = self.f_base.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, self.rank, self.rank).clone()

        # 2. Micro-cycle Fourier harmonics
        for k in range(1, self.num_harmonics + 1):
            k_idx = k - 1
            cos_val = torch.cos(k * theta).view(batch_size, 1, 1)
            sin_val = torch.sin(k * theta).view(batch_size, 1, 1)
            c_k = self.cos_basis[k_idx].unsqueeze(0)
            s_k = self.sin_basis[k_idx].unsqueeze(0)
            f_core = f_core + (cos_val * c_k + sin_val * s_k)

        # 3. Macro-epoch linear drift
        z_val = z.view(batch_size, 1, 1)
        f_core = f_core + (z_val * self.epoch_linear.unsqueeze(0))

        # 4. Optional quadratic drift
        if self.epoch_quad is not None:
            z_sq = ((z / 10.0) ** 2).view(batch_size, 1, 1)
            f_core = f_core + (z_sq * self.epoch_quad.unsqueeze(0))

        return f_core


class DualCycleMaTLoRALinear(_ModuleBase):
    """Drop-in Linear replacement with Dual-Cycle Matrix Time-LoRA adaptation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
        num_harmonics: int = 2,
        use_epoch_poly: bool = True,
    ) -> None:
        if nn is None:
            raise RuntimeError("PyTorch required")
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank) if rank > 0 else 1.0

        # Frozen base linear layer W_0
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Trainable low-rank projection matrices A and B
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        self.reset_lora_parameters()

        # Dynamic Core Generator
        self.core_generator = DualCycleCoreGenerator(
            rank=rank,
            num_harmonics=num_harmonics,
            use_epoch_poly=use_epoch_poly,
        )

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Explicit bypass switch: when enabled=False, forward strictly returns base_layer(x)
        self.enabled: bool = True

        # Current temporal state (theta, z)
        self.current_theta: Optional[torch.Tensor] = None
        self.current_z: Optional[torch.Tensor] = None

    def reset_lora_parameters(self) -> None:
        """Initialize LoRA weights to strictly preserve W_0 at step 0 (Hu et al. 2021)."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initializing B to zero strictly guarantees Delta W = 0 at step 0
        nn.init.zeros_(self.lora_B)

    def set_enabled(self, enabled: bool) -> None:
        """Explicitly enable or bypass LoRA adaptation.
        
        When enabled=False, forward pass strictly computes W_0(x) with zero lora overhead.
        """
        self.enabled = bool(enabled)

    def clear_temporal_coordinates(self) -> None:
        """Clear active (theta, z) coordinates to prevent state leakage."""
        self.current_theta = None
        self.current_z = None

    def set_temporal_coordinates(
        self,
        theta: Optional[Union[float, torch.Tensor]],
        z: Optional[Union[float, torch.Tensor]],
    ) -> None:
        """Set active (theta, z) for the module."""
        if theta is None:
            self.current_theta = None
        elif isinstance(theta, torch.Tensor):
            self.current_theta = theta
        else:
            self.current_theta = torch.tensor([theta], dtype=torch.float32)

        if z is None:
            self.current_z = None
        elif isinstance(z, torch.Tensor):
            self.current_z = z
        else:
            self.current_z = torch.tensor([z], dtype=torch.float32)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        num_harmonics: int = 2,
        use_epoch_poly: bool = True,
    ) -> "DualCycleMaTLoRALinear":
        """Wrap an existing Linear (nn.Linear or bnb.nn.Linear4bit) module."""
        has_bias = getattr(linear, "bias", None) is not None
        mod = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=has_bias,
            num_harmonics=num_harmonics,
            use_epoch_poly=use_epoch_poly,
        )
        # Directly adopt linear layer to avoid shape mismatch on quantized or wrapped modules
        mod.base_layer = linear
        if hasattr(mod.base_layer, "weight") and mod.base_layer.weight is not None:
            mod.base_layer.weight.requires_grad = False
        if has_bias and hasattr(mod.base_layer, "bias") and mod.base_layer.bias is not None:
            mod.base_layer.bias.requires_grad = False
        # Match device only; keep trainable adapter weights strictly in float32
        device = getattr(linear.weight, "device", torch.device("cpu"))
        mod.to(device=device)
        return mod

    def forward(
        self,
        x: torch.Tensor,
        theta: Optional[Union[float, torch.Tensor]] = None,
        z: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass with time-dependent core transformation."""
        base_out = self.base_layer(x)
        if not self.enabled or self.rank == 0:
            return base_out

        orig_shape = x.shape
        batch_size = orig_shape[0]
        x_flat = x.view(batch_size, -1, self.in_features)

        th = theta if theta is not None else self.current_theta
        zv = z if z is not None else self.current_z

        if th is None or zv is None:
            th_val = torch.zeros(batch_size, device=x.device, dtype=torch.float32)
            zv_val = torch.zeros(batch_size, device=x.device, dtype=torch.float32)
        else:
            if not isinstance(th, torch.Tensor):
                th_val = torch.tensor([th], device=x.device, dtype=torch.float32).expand(batch_size)
            else:
                th_val = th.to(device=x.device, dtype=torch.float32)
                if th_val.dim() == 0:
                    th_val = th_val.view(1).expand(batch_size)
                elif th_val.shape[0] == 1 and batch_size > 1:
                    th_val = th_val.expand(batch_size)

            if not isinstance(zv, torch.Tensor):
                zv_val = torch.tensor([zv], device=x.device, dtype=torch.float32).expand(batch_size)
            else:
                zv_val = zv.to(device=x.device, dtype=torch.float32)
                if zv_val.dim() == 0:
                    zv_val = zv_val.view(1).expand(batch_size)
                elif zv_val.shape[0] == 1 and batch_size > 1:
                    zv_val = zv_val.expand(batch_size)

        # Low-rank projection in float32 for absolute numerical stability
        x_dropped = self.lora_dropout(x_flat)
        x_a = F.linear(x_dropped.float(), self.lora_A)

        # Dynamic Core F(theta, z): shape (batch_size, rank, rank) in float32
        f_core = self.core_generator(th_val, zv_val)

        # Core transformation: x_a @ F^T in float32
        x_af = torch.matmul(x_a, f_core.transpose(-1, -2))

        # Low-rank reconstruction: x_af @ B^T * scaling in float32
        lora_out = F.linear(x_af, self.lora_B) * self.scaling

        # Cast back to input dtype (e.g. float16) to add to base_out
        lora_out_reshaped = lora_out.view(*orig_shape[:-1], self.out_features).to(dtype=x.dtype)
        return base_out + lora_out_reshaped


def replace_mlp_with_dual_cycle_mat_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    num_harmonics: int = 2,
    use_epoch_poly: bool = True,
    target_suffixes: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj"),
) -> Dict[str, DualCycleMaTLoRALinear]:
    """Replace all MLP linear projections with DualCycleMaTLoRALinear."""
    replaced_modules: Dict[str, DualCycleMaTLoRALinear] = {}

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(name.endswith(suffix) for suffix in target_suffixes):
            # Locate parent module
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            child_name = parts[-1]

            adapted = DualCycleMaTLoRALinear.from_linear(
                linear=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                num_harmonics=num_harmonics,
                use_epoch_poly=use_epoch_poly,
            )
            setattr(parent, child_name, adapted)
            replaced_modules[name] = adapted

    return replaced_modules


def set_model_temporal_coordinates(
    adapted_modules: Dict[str, DualCycleMaTLoRALinear],
    theta: Optional[float],
    z: Optional[float],
) -> None:
    """Set active (theta, z) across all adapted DualCycleMaTLoRA modules."""
    for mod in adapted_modules.values():
        mod.set_temporal_coordinates(theta, z)


def clear_model_temporal_coordinates(
    adapted_modules: Dict[str, DualCycleMaTLoRALinear],
) -> None:
    """Clear active temporal coordinates across all adapted DualCycleMaTLoRA modules."""
    for mod in adapted_modules.values():
        mod.clear_temporal_coordinates()


def set_model_mat_lora_enabled(
    adapted_modules: Dict[str, DualCycleMaTLoRALinear],
    enabled: bool,
) -> None:
    """Set explicit enabled switch across all adapted DualCycleMaTLoRA modules."""
    for mod in adapted_modules.values():
        mod.set_enabled(enabled)

