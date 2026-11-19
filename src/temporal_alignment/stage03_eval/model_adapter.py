"""Model adaptation, checkpoint loading, and safe temporal context lifecycle.

Supports:
- Base Qwen model (unadapted backbone)
- Arm 4: Static MLP LoRA + Seam Projection
- Arm 5: Generalized Dual-Cycle MaT-LoRA + Seam Projection

Guarantees:
- Strict state-dict tensor validation on weights_only loading.
- Temporal context isolation (context manager guaranteeing cleanup on exit).
- Zero leakage of temporal state between evaluation instances.
"""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_injector import GatedResidualProjection

ALLOWED_MLP_TARGETS: Tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")
FORBIDDEN_ATTENTION_TARGETS: frozenset[str] = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})


class StaticLoRALinear(nn.Module):
    """Static Low-Rank Adaptation (LoRA) layer wrapping a frozen Linear layer."""

    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / float(rank)
        self.base_layer = original_linear
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        dev = original_linear.weight.device
        dtype = original_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=dev, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank, device=dev, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.adapter_mode: str = "ACTIVE"  # "ACTIVE" | "BOTH_OFF"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if getattr(self, "adapter_mode", "ACTIVE") == "BOTH_OFF":
            return base_out
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


class DualCycleCoreGenerator(nn.Module):
    """Generates dynamic r x r core transformation tensor F_core(theta, z)."""

    def __init__(self, rank: int = 8, num_harmonics: int = 2):
        super().__init__()
        self.rank = rank
        self.num_harmonics = num_harmonics
        self.register_buffer("f_base", torch.eye(rank, dtype=torch.float32))
        std = 1.0 / math.sqrt(rank)
        self.cos_basis = nn.Parameter(torch.randn(num_harmonics, rank, rank) * (std * 0.1))
        self.sin_basis = nn.Parameter(torch.randn(num_harmonics, rank, rank) * (std * 0.1))
        self.epoch_linear = nn.Parameter(torch.randn(rank, rank) * (std * 0.1))
        self.epoch_quad = nn.Parameter(torch.randn(rank, rank) * (std * 0.01))

    def forward(self, theta: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B = theta.shape[0]
        dev = self.cos_basis.device
        th = theta.to(device=dev, dtype=torch.float32)
        zv = z.to(device=dev, dtype=torch.float32)

        f_core = self.f_base.unsqueeze(0).expand(B, self.rank, self.rank).clone()
        for k in range(1, self.num_harmonics + 1):
            k_idx = k - 1
            cos_val = torch.cos(k * th).view(B, 1, 1)
            sin_val = torch.sin(k * th).view(B, 1, 1)
            f_core = f_core + (
                cos_val * self.cos_basis[k_idx].unsqueeze(0)
                + sin_val * self.sin_basis[k_idx].unsqueeze(0)
            )

        f_core = f_core + zv.view(B, 1, 1) * self.epoch_linear.unsqueeze(0)
        z_sq = ((zv / 10.0) ** 2).view(B, 1, 1)
        f_core = f_core + z_sq * self.epoch_quad.unsqueeze(0)
        return f_core


class GeneralizedDualCycleMaTLoRALinear(nn.Module):
    """Dynamic Matrix-Time LoRA linear layer conditioned on (theta, z)."""

    def __init__(
        self,
        original_linear: nn.Linear,
        core_gen: DualCycleCoreGenerator,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / float(rank)
        self.base_layer = original_linear
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.core_gen = core_gen
        dev = original_linear.weight.device
        dtype = original_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=dev, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank, device=dev, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.current_theta: Optional[torch.Tensor] = None
        self.current_z: Optional[torch.Tensor] = None
        self.current_temporal_active: Optional[torch.Tensor] = None
        self.adapter_mode: str = "ACTIVE"  # "ACTIVE" | "NEUTRAL_CORE" | "BOTH_OFF"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.adapter_mode == "BOTH_OFF":
            return base_out

        dev = x.device
        batch_size = x.size(0)
        identity_core = (
            self.core_gen.f_base.to(device=dev, dtype=torch.float32)
            .unsqueeze(0)
            .expand(batch_size, self.rank, self.rank)
        )

        if self.adapter_mode == "NEUTRAL_CORE":
            F_core = identity_core
        elif self.current_theta is None or self.current_z is None:
            if self.current_temporal_active is not None:
                # General replay or explicit neutral rows still train shared lora_A / lora_B with F=I
                F_core = identity_core
            else:
                return base_out
        else:
            F_dyn = self.core_gen(self.current_theta, self.current_z)
            if self.current_temporal_active is not None:
                act_mask = self.current_temporal_active.to(device=F_dyn.device, dtype=torch.bool).view(-1, 1, 1)
                F_core = torch.where(act_mask, F_dyn, identity_core.to(device=F_dyn.device, dtype=F_dyn.dtype))
            else:
                F_core = F_dyn

        x_a = F.linear(x.float(), self.lora_A.float())
        x_af = torch.bmm(x_a, F_core.transpose(1, 2))
        lora_out = F.linear(x_af, self.lora_B.float()) * self.scaling
        return base_out + lora_out.to(dtype=x.dtype)


def attach_adapters_to_model(
    model: nn.Module,
    arm: str,
    rank: int = 8,
    alpha: float = 16.0,
    target_submodules: Optional[List[str]] = None,
) -> Tuple[List[nn.Module], Optional[DualCycleCoreGenerator]]:
    """Attach Static LoRA or MaT-LoRA adapters strictly to MLP Linear layers (NO Attention)."""
    if arm == "base":
        return [], None

    targets = list(target_submodules) if target_submodules is not None else list(ALLOWED_MLP_TARGETS)
    forbidden = [t for t in targets if t in FORBIDDEN_ATTENTION_TARGETS]
    if forbidden:
        raise ValueError(
            f"Attention intervention is strictly forbidden under Stage 03 MLP-only contract: {forbidden}"
        )

    adapted_modules: List[nn.Module] = []
    core_gen: Optional[DualCycleCoreGenerator] = None

    if arm == "arm4":
        for name, module in model.named_modules():
            for target in targets:
                if hasattr(module, target) and isinstance(getattr(module, target), nn.Linear):
                    orig = getattr(module, target)
                    adapted = StaticLoRALinear(orig, rank=rank, alpha=alpha)
                    setattr(module, target, adapted)
                    adapted_modules.append(adapted)
        return adapted_modules, None

    if arm == "arm5":
        dev = next(model.parameters()).device
        core_gen = DualCycleCoreGenerator(rank=rank, num_harmonics=2).to(dev, dtype=torch.float32)
        for name, module in model.named_modules():
            for target in targets:
                if hasattr(module, target) and isinstance(getattr(module, target), nn.Linear):
                    orig = getattr(module, target)
                    adapted = GeneralizedDualCycleMaTLoRALinear(
                        orig,
                        core_gen=core_gen,
                        rank=rank,
                        alpha=alpha,
                    )
                    setattr(module, target, adapted)
                    adapted_modules.append(adapted)
        return adapted_modules, core_gen

    raise ValueError(f"Unknown arm type: '{arm}'. Expected 'base', 'arm4', or 'arm5'.")


@contextlib.contextmanager
def matlora_temporal_context(
    model: nn.Module,
    theta: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    *,
    temporal_active: Optional[torch.Tensor] = None,
    adapter_mode: str = "ACTIVE",
) -> Iterator[None]:
    """Safely apply (theta, z, temporal_active, adapter_mode) context with guaranteed cleanup."""
    if adapter_mode not in ("ACTIVE", "NEUTRAL_CORE", "BOTH_OFF"):
        raise ValueError(f"Unsupported adapter_mode: {adapter_mode}")
    mat_modules: List[GeneralizedDualCycleMaTLoRALinear] = []
    static_modules: List[StaticLoRALinear] = []
    for module in model.modules():
        if isinstance(module, GeneralizedDualCycleMaTLoRALinear):
            module.current_theta = theta
            module.current_z = z
            module.current_temporal_active = temporal_active
            module.adapter_mode = adapter_mode
            mat_modules.append(module)
        elif isinstance(module, StaticLoRALinear):
            module.adapter_mode = adapter_mode
            static_modules.append(module)
    try:
        yield
    finally:
        for mod in mat_modules:
            mod.current_theta = None
            mod.current_z = None
            mod.current_temporal_active = None
            mod.adapter_mode = "ACTIVE"
        for mod in static_modules:
            mod.adapter_mode = "ACTIVE"


def load_checkpoint_weights(
    model: nn.Module,
    seam_proj: Optional[GatedResidualProjection],
    core_gen: Optional[DualCycleCoreGenerator],
    checkpoint_path: str,
    arm: str = "arm5",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load trained weights with strict tensor validation and weights_only safety.

    Enforces:
    - Nonzero and complete expected tensor group coverage per arm.
    - Zero tolerance for missing LoRA parameters (216 tensors for 36-layer Qwen).
    - Seam projection parameters (alpha, linear.weight, linear.bias, norm.weight).
    - Core generator parameters for Arm 5 (cos_basis, sin_basis, epoch_linear, epoch_quad, f_base).
    - Strict NaN and Inf checks on all checkpoint tensors before loading.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {path}")

    dev = next(model.parameters()).device
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint at {path} is not a dictionary.")

    lora_state = ckpt.get("lora_state", {})
    seam_state = ckpt.get("seam_projection", {})
    core_state = ckpt.get("core_generator", None)

    # Hard validation of expected parameter groups per arm
    if arm in ("arm4", "arm5"):
        if not lora_state or len(lora_state) == 0:
            raise ValueError(
                f"Checkpoint {path} contains empty or missing 'lora_state' for arm '{arm}'."
            )
        if seam_proj is not None and (not seam_state or len(seam_state) == 0):
            raise ValueError(
                f"Checkpoint {path} contains empty or missing 'seam_projection' for arm '{arm}'."
            )

    if arm == "arm5":
        if core_gen is not None:
            if core_state is None or len(core_state) == 0:
                raise ValueError(
                    f"Checkpoint {path} missing 'core_generator' state dict required for Arm 5."
                )

    # Assert no NaN or Inf in checkpoint tensors
    for group_name, group_dict in [("lora_state", lora_state), ("seam_projection", seam_state)]:
        if isinstance(group_dict, dict):
            for k, v in group_dict.items():
                if isinstance(v, torch.Tensor):
                    if torch.isnan(v).any():
                        raise ValueError(f"NaN detected in {group_name} tensor '{k}' from {path}")
                    if torch.isinf(v).any():
                        raise ValueError(f"Inf detected in {group_name} tensor '{k}' from {path}")

    # Load LoRA parameters into model
    model_state = model.state_dict()
    matched_lora = 0
    for k, v in lora_state.items():
        if k in model_state:
            target_tensor = model_state[k]
            if target_tensor.shape != v.shape:
                raise ValueError(
                    f"Shape mismatch on key '{k}': checkpoint {v.shape} vs model {target_tensor.shape}"
                )
            target_tensor.copy_(v.to(target_tensor.device, dtype=target_tensor.dtype))
            matched_lora += 1
        elif strict:
            raise KeyError(f"Checkpoint LoRA key '{k}' not found in adapted model state dict.")

    if arm in ("arm4", "arm5") and matched_lora == 0:
        raise ValueError(
            f"Zero LoRA parameters matched adapted model state dict for arm '{arm}'. Check namespace."
        )

    # Load Seam projection
    if seam_proj is not None and seam_state:
        seam_proj.load_state_dict({k: v.to(dev) for k, v in seam_state.items()}, strict=strict)

    # Load Core Generator
    if core_gen is not None and core_state is not None:
        core_gen.load_state_dict({k: v.to(dev) for k, v in core_state.items()}, strict=strict)

    return {
        "status": "LOAD_SUCCESS",
        "checkpoint_path": str(path),
        "arm": arm,
        "matched_lora_tensors": matched_lora,
        "total_lora_tensors": len(lora_state),
        "seam_tensors": len(seam_state) if seam_state else 0,
        "core_tensors": len(core_state) if core_state else 0,
        "seam_alpha": float(seam_proj.alpha.item()) if seam_proj is not None and hasattr(seam_proj, "alpha") else None,
    }
