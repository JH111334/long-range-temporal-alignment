"""Canonical Loader for M0-FC40 Checkpoints with Strict Key Validation and Integrity Asserts.

This module provides fail-closed loading of trained M0-FC40 weights into a Qwen backbone,
preventing silent fallbacks or bypassed adapters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from temporal_alignment.models.dual_cycle_mat_lora import (
    DualCycleMaTLoRALinear,
    replace_mlp_with_dual_cycle_mat_lora,
)
from temporal_alignment.models.gated_seam import (
    ConditioningContext,
    GatedResidualProjection,
    make_gated_embedding_hook,
)


def load_m0_fc40_for_inference(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    rank: int = 8,
    alpha: float = 16.0,
    num_harmonics: int = 2,
    injection_token_index: int = 14,
) -> Tuple[Dict[str, DualCycleMaTLoRALinear], GatedResidualProjection, ConditioningContext]:
    """Strict canonical loader for M0-FC40 checkpoints.

    Validates presence of 72 adapter modules and gated projection state.
    Asserts learned parameters are non-zero to guarantee active forward contributions.
    Fails closed (raises KeyError/ValueError) if the checkpoint format is invalid.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"M0-FC40 checkpoint not found at: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    if not isinstance(state, dict):
        raise TypeError(f"Expected dict state in checkpoint, got {type(state)}")

    # Strict key check: prevent legacy silent bypass
    if "adapters_state" not in state:
        raise KeyError(
            f"Mandatory key 'adapters_state' missing from checkpoint {ckpt_path}. "
            f"Keys found: {list(state.keys())}. "
            "Ensure checkpoint was saved by canonical M0-FC40 training pipeline."
        )
    if "gated_proj_state" not in state:
        raise KeyError(
            f"Mandatory key 'gated_proj_state' missing from checkpoint {ckpt_path}."
        )

    adapters_state = state["adapters_state"]
    if len(adapters_state) != 72:
        raise ValueError(
            f"Expected exactly 72 adapted MLP projections in checkpoint, found {len(adapters_state)}"
        )

    # 1. Replace MLP modules with DualCycleMaTLoRALinear
    adapted_modules = replace_mlp_with_dual_cycle_mat_lora(
        model=model,
        rank=rank,
        alpha=alpha,
        num_harmonics=num_harmonics,
    )
    if len(adapted_modules) != 72:
        raise ValueError(
            f"Expected 72 adapted modules constructed in backbone, got {len(adapted_modules)}"
        )

    # 2. Copy adapter weights
    total_b_norm = 0.0
    for name, mod in adapted_modules.items():
        if name not in adapters_state:
            raise KeyError(f"Module '{name}' missing from checkpoint adapters_state")
        mod_state = adapters_state[name]
        if "lora_A" not in mod_state or "lora_B" not in mod_state:
            raise KeyError(f"Module '{name}' in adapters_state is missing lora_A or lora_B keys: {list(mod_state.keys())}")

        mod.lora_A.data.copy_(mod_state["lora_A"].to(device))
        mod.lora_B.data.copy_(mod_state["lora_B"].to(device))

        if "core_state" in mod_state:
            mod.core_generator.load_state_dict(mod_state["core_state"])
        else:
            core_dict = {k.replace("core_generator.", ""): v for k, v in mod_state.items() if k.startswith("core_generator.")}
            if core_dict:
                mod.core_generator.load_state_dict(core_dict)
            else:
                raise KeyError(f"Module '{name}' in adapters_state is missing core generator weights")

        total_b_norm += mod.lora_B.norm().item()

    # Fail closed if weights were somehow all zero (which would mean no adaptation)
    if total_b_norm < 1e-4:
        raise ValueError(f"Total lora_B norm across all 72 modules is suspiciously near zero ({total_b_norm}).")

    # 3. Setup Gated Residual Projection
    hidden_dim = getattr(model.config, "hidden_size", 896)
    gated_proj = GatedResidualProjection(in_features=40, out_features=hidden_dim).to(device)
    gated_proj.load_state_dict(state["gated_proj_state"])

    # 4. Setup Conditioning Context and Register Forward Hook
    cond_context = ConditioningContext(injection_token_index=injection_token_index)
    hook = make_gated_embedding_hook(cond_context)
    embed_layer = model.get_input_embeddings()
    embed_layer.register_forward_hook(hook)

    model.eval()
    return adapted_modules, gated_proj, cond_context
