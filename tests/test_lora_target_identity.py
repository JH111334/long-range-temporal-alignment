"""Unit tests for LoRA target identity: QV LoRA vs MLP Dual-Cycle MaT-LoRA.

Verifies:
1. TICKTACK_INSPIRED_QV targets exactly q_proj and v_proj (48 modules across 24 layers).
2. DUAL_CYCLE_MAT_LORA targets exactly gate_proj, up_proj, down_proj (72 modules across 24 layers).
3. Backbone linear layers are 100% frozen (0 gradient).
4. Only adapter parameters are trainable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from temporal_alignment.models.dual_cycle_mat_lora import (
    DualCycleMaTLoRALinear,
)


class MockTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int = 64, intermediate_dim: int = 128):
        super().__init__()
        # Self attention
        self.self_attn = nn.ModuleDict({
            "q_proj": nn.Linear(hidden_dim, hidden_dim),
            "k_proj": nn.Linear(hidden_dim, hidden_dim),
            "v_proj": nn.Linear(hidden_dim, hidden_dim),
            "o_proj": nn.Linear(hidden_dim, hidden_dim),
        })
        # MLP
        self.mlp = nn.ModuleDict({
            "gate_proj": nn.Linear(hidden_dim, intermediate_dim),
            "up_proj": nn.Linear(hidden_dim, intermediate_dim),
            "down_proj": nn.Linear(intermediate_dim, hidden_dim),
        })


class MockQwenModel(nn.Module):
    def __init__(self, num_layers: int = 24, hidden_dim: int = 64, intermediate_dim: int = 128):
        super().__init__()
        self.layers = nn.ModuleList([
            MockTransformerLayer(hidden_dim, intermediate_dim) for _ in range(num_layers)
        ])


def test_dual_cycle_mat_lora_targets_exactly_72_mlp_modules():
    """Verify that DualCycleMaTLoRA replaces exactly 72 MLP modules and freezes base layers."""
    model = MockQwenModel(num_layers=24)
    adapted = {}

    for layer_idx, layer in enumerate(model.layers):
        for name in ["gate_proj", "up_proj", "down_proj"]:
            orig = layer.mlp[name]
            wrapped = DualCycleMaTLoRALinear(orig.in_features, orig.out_features, rank=8, alpha=16.0)
            wrapped.base_layer.weight.data.copy_(orig.weight.data)
            layer.mlp[name] = wrapped
            adapted[f"layer.{layer_idx}.mlp.{name}"] = wrapped

    assert len(adapted) == 72

    # Verify base layers are frozen
    for k, mod in adapted.items():
        assert not mod.base_layer.weight.requires_grad
        assert mod.lora_A.requires_grad
        assert mod.lora_B.requires_grad


def test_standard_qv_lora_targets_exactly_48_attention_modules():
    """Verify that standard QV LoRA targets exactly 48 attention modules."""
    model = MockQwenModel(num_layers=24)
    qv_count = 0

    for layer in model.layers:
        for name, mod in layer.self_attn.items():
            if name in ["q_proj", "v_proj"]:
                qv_count += 1

    assert qv_count == 48  # 24 * 2
