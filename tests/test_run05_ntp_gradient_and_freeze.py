"""Unit Test for Run 05 100% Backbone Freeze and Gradient Isolation."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from temporal_alignment.models.dual_cycle_mat_lora import replace_mlp_with_dual_cycle_mat_lora
from temporal_alignment.models.gated_seam import GatedResidualProjection


def test_backbone_100_percent_frozen():
    """Verify that all non-adapter parameters in the backbone have requires_grad=False."""
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    for p in model.parameters():
        p.requires_grad = False

    adapted_modules = replace_mlp_with_dual_cycle_mat_lora(model, rank=8, alpha=16)
    gated_proj = GatedResidualProjection(in_features=40, out_features=896)

    # Count trainable vs frozen parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_gated = [p for p in gated_proj.parameters() if p.requires_grad]

    assert len(adapted_modules) == 72, f"Expected 72 adapted MLP modules, got {len(adapted_modules)}"

    # All base transformer blocks must remain completely frozen
    for name, param in model.named_parameters():
        if "base_layer" in name or "self_attn" in name or "norm" in name or "embed" in name or "lm_head" in name:
            assert not param.requires_grad, f"Backbone parameter {name} leaked requires_grad=True!"

    # Verify adapter parameters are trainable
    assert len(trainable_params) > 0
    assert len(trainable_gated) > 0
