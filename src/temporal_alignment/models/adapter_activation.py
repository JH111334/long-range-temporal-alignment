"""Scoped, exception-safe adapter activation and bypass management.

Provides context managers and controllers for independently enabling/disabling:
1. MaT-LoRA adapters on MLP modules (active_lora=True/False)
2. Gated Seam continuous injection at token 14 (active_seam=True/False)
3. Specific temporal conditions: C_TRUE, C_WRONG_YEAR, C_INPUT_ZERO_FEATURE,
   C_OUTPUT_ZERO_SEAM, C_SEAM_OFF, C_LORA_OFF, C_BOTH_OFF.

Guarantees 100% exception-safe restoration of state and bitwise numerical parity
between C_BOTH_OFF and raw frozen Qwen.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .dual_cycle_mat_lora import DualCycleMaTLoRALinear
from .gated_seam import ConditioningContext, GatedResidualProjection


@contextmanager
def scoped_adapter_activation(
    adapted_modules: Dict[str, DualCycleMaTLoRALinear],
    cond_context: ConditioningContext,
    active_lora: bool = True,
    active_seam: bool = True,
    theta: Optional[Union[float, torch.Tensor]] = None,
    z: Optional[Union[float, torch.Tensor]] = None,
    seam_vector: Optional[torch.Tensor] = None,
    seam_token_index: int = 14,
) -> Iterator[Dict[str, Any]]:
    """Context manager for scoped adapter and seam activation with guaranteed cleanup.

    Args:
        adapted_modules: Dictionary of 72 DualCycleMaTLoRALinear modules.
        cond_context: ConditioningContext for gated seam injection.
        active_lora: If False, completely bypasses LoRA forward computation (returns base_layer(x)).
        active_seam: If False, clears seam injection vector.
        theta: Sexagenary phase angle coordinate.
        z: Century drift coordinate.
        seam_vector: Injected vector for token 14.
        seam_token_index: Seam token index (default 14).

    Yields:
        Dictionary with active state metadata.
    """
    # 1. Snapshot previous state across all 72 modules
    prev_lora_enabled: Dict[str, bool] = {}
    prev_theta: Dict[str, Optional[torch.Tensor]] = {}
    prev_z: Dict[str, Optional[torch.Tensor]] = {}

    for name, mod in adapted_modules.items():
        prev_lora_enabled[name] = getattr(mod, "enabled", True)
        prev_theta[name] = mod.current_theta
        prev_z[name] = mod.current_z

    prev_seam_enabled = cond_context.enabled
    prev_seam_vector = cond_context.current_vector
    prev_seam_idx = cond_context.injection_token_index

    try:
        # 2. Apply new LoRA state
        for mod in adapted_modules.values():
            mod.set_enabled(active_lora)
            if active_lora and (theta is not None or z is not None):
                mod.set_temporal_coordinates(theta=theta, z=z)
            else:
                mod.clear_temporal_coordinates()

        # 3. Apply new Seam state
        cond_context.injection_token_index = seam_token_index
        cond_context.enabled = active_seam
        if active_seam and seam_vector is not None:
            cond_context.set_vector(seam_vector)
        else:
            cond_context.clear()

        yield {
            "active_lora": active_lora,
            "active_seam": active_seam,
            "has_theta": theta is not None,
            "has_z": z is not None,
            "seam_vector_norm": seam_vector.norm().item() if seam_vector is not None else 0.0,
        }

    finally:
        # 4. Strictly restore all previous states in finally block
        for name, mod in adapted_modules.items():
            mod.set_enabled(prev_lora_enabled.get(name, True))
            mod.current_theta = prev_theta.get(name, None)
            mod.current_z = prev_z.get(name, None)

        cond_context.enabled = prev_seam_enabled
        cond_context.current_vector = prev_seam_vector
        cond_context.injection_token_index = prev_seam_idx


def configure_temporal_condition(
    condition_id: str,
    adapted_modules: Dict[str, DualCycleMaTLoRALinear],
    gated_proj: GatedResidualProjection,
    cond_context: ConditioningContext,
    coords_40d: Optional[List[float]],
    theta: Optional[float],
    z: Optional[float],
    device: torch.device,
    seam_token_index: int = 14,
) -> scoped_adapter_activation:
    """Return a scoped context manager configuring the specified Arm C condition.

    Conditions supported:
    - C_TRUE: Injects true 40D vector and true (theta, z) with LoRA active.
    - C_WRONG_YEAR: Injects perturbed (+30 yrs) 40D vector and (theta, z) with LoRA active.
    - C_INPUT_ZERO_FEATURE: Projects [0.0]*40 through gated_proj; (theta, z) = (0, 0); LoRA active.
    - C_OUTPUT_ZERO_SEAM: Injects post-projection all-zero tensor at seam; LoRA active.
    - C_SEAM_OFF: Seam cleared; true (theta, z) active on LoRA.
    - C_LORA_OFF: LoRA set_enabled(False); true seam vector injected.
    - C_BOTH_OFF: LoRA set_enabled(False) AND seam cleared (strictly base Qwen equivalent).
    """
    hidden_dim = gated_proj.out_features

    if condition_id == "C_TRUE":
        if coords_40d is not None:
            vec_t = torch.tensor(coords_40d, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                proj_v = gated_proj(vec_t).squeeze(0)
            return scoped_adapter_activation(
                adapted_modules, cond_context,
                active_lora=True, active_seam=True,
                theta=theta, z=z, seam_vector=proj_v,
                seam_token_index=seam_token_index,
            )
        else:
            return scoped_adapter_activation(
                adapted_modules, cond_context,
                active_lora=True, active_seam=False,
                theta=None, z=None, seam_vector=None,
            )

    elif condition_id == "C_WRONG_YEAR":
        if coords_40d is not None:
            vec_t = torch.tensor(coords_40d, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                proj_v = gated_proj(vec_t).squeeze(0)
            return scoped_adapter_activation(
                adapted_modules, cond_context,
                active_lora=True, active_seam=True,
                theta=theta, z=z, seam_vector=proj_v,
                seam_token_index=seam_token_index,
            )
        else:
            return scoped_adapter_activation(
                adapted_modules, cond_context,
                active_lora=True, active_seam=False,
                theta=None, z=None, seam_vector=None,
            )

    elif condition_id == "C_INPUT_ZERO_FEATURE":
        vec_zeros = torch.zeros((1, 40), dtype=torch.float32, device=device)
        with torch.no_grad():
            proj_v = gated_proj(vec_zeros).squeeze(0)
        return scoped_adapter_activation(
            adapted_modules, cond_context,
            active_lora=True, active_seam=True,
            theta=0.0, z=0.0, seam_vector=proj_v,
            seam_token_index=seam_token_index,
        )

    elif condition_id == "C_OUTPUT_ZERO_SEAM":
        zero_seam = torch.zeros(hidden_dim, dtype=torch.float32, device=device)
        return scoped_adapter_activation(
            adapted_modules, cond_context,
            active_lora=True, active_seam=True,
            theta=theta, z=z, seam_vector=zero_seam,
            seam_token_index=seam_token_index,
        )

    elif condition_id == "C_SEAM_OFF":
        return scoped_adapter_activation(
            adapted_modules, cond_context,
            active_lora=True, active_seam=False,
            theta=theta, z=z, seam_vector=None,
        )

    elif condition_id == "C_LORA_OFF":
        if coords_40d is not None:
            vec_t = torch.tensor(coords_40d, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                proj_v = gated_proj(vec_t).squeeze(0)
            return scoped_adapter_activation(
                adapted_modules, cond_context,
                active_lora=False, active_seam=True,
                theta=None, z=None, seam_vector=proj_v,
                seam_token_index=seam_token_index,
            )
        else:
            return scoped_adapter_activation(
                adapted_modules, cond_context,
                active_lora=False, active_seam=False,
                theta=None, z=None, seam_vector=None,
            )

    elif condition_id == "C_BOTH_OFF":
        return scoped_adapter_activation(
            adapted_modules, cond_context,
            active_lora=False, active_seam=False,
            theta=None, z=None, seam_vector=None,
        )

    else:
        raise ValueError(f"Unknown temporal condition ID: {condition_id}")
