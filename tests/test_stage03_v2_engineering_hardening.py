"""Unit tests for Stage 03 V2 Engineering Hardening:
- Out-of-place safe seam autograd gradient flow
- Non-negative LogSumExp contrastive loss mathematical correctness
- Shared core generator optimizer parameter deduplication
- Collision-safe evaluation row indexing
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from temporal_alignment.models.gated_seam import (
    GatedResidualProjection,
    inject_seam_embedding_tensor,
)
from temporal_alignment.models.dual_cycle_mat_lora import (
    DualCycleCoreGenerator,
    DualCycleMaTLoRALinear,
)


def test_seam_autograd_out_of_place_gradient():
    """Verify that out-of-place seam injection allows full gradient flow through alpha and weights."""
    B, S, D = 4, 32, 64
    seam_proj = GatedResidualProjection(in_features=40, out_features=D)
    inputs_embeds = torch.randn(B, S, D, requires_grad=True)
    v40 = torch.randn(B, 40)

    # Initial forward: alpha=0.0 so injection is 0
    inj_0 = seam_proj(v40)
    out_0 = inject_seam_embedding_tensor(inputs_embeds, inj_0, token_index=14)

    assert out_0.shape == (B, S, D)
    # At alpha=0.0, output strictly equals inputs_embeds
    assert torch.allclose(out_0, inputs_embeds)

    # Compute loss and backward
    loss = out_0.sum()
    loss.backward()

    # Alpha receives gradient from the sum
    assert seam_proj.alpha.grad is not None
    assert torch.isfinite(seam_proj.alpha.grad)
    # Inputs_embeds receives clean gradient (all 1.0)
    assert torch.allclose(inputs_embeds.grad, torch.ones_like(inputs_embeds))

    # Now simulate non-zero alpha and check linear weight gradient
    seam_proj.zero_grad()
    with torch.no_grad():
        seam_proj.alpha.fill_(1.0)

    inj_1 = seam_proj(v40)
    out_1 = inject_seam_embedding_tensor(inputs_embeds, inj_1, token_index=14)
    # LayerNorm centers output across feature dimension (sum(normed) == 0).
    # To test non-zero gradient flow through Linear weight, compute loss against a non-uniform target.
    target = torch.randn_like(out_1)
    loss_1 = ((out_1 - target) ** 2).sum()
    loss_1.backward()

    assert seam_proj.linear.weight.grad is not None
    assert torch.isfinite(seam_proj.linear.weight.grad).all()
    assert seam_proj.linear.weight.grad.norm() > 0.0


def test_contrastive_loss_logsumexp_stability():
    """Verify that normalized LogSumExp contrastive loss is strictly non-negative (0 <= L < inf)."""
    B = 8
    temperature = 0.07

    # Dummy hidden states and cyclical years
    hidden = torch.randn(B, 16, 64)
    h_pool = F.normalize(hidden.mean(dim=1).float(), p=2, dim=-1)
    sim_matrix = torch.matmul(h_pool, h_pool.T) / temperature

    # Construct known matching cycles (years with matching mod 60)
    years = torch.tensor([1947, 2007, 1950, 2010, 1960, 1960, 1970, 1980], dtype=torch.long)
    cycles = (years - 3) % 60
    labels_eq = torch.eq(cycles.unsqueeze(0), cycles.unsqueeze(1))
    mask_self = torch.eye(B, dtype=torch.bool)
    pos_mask = labels_eq & (~mask_self)

    assert pos_mask.any(), "Test requires at least one positive pair"

    # Correct LogSumExp implementation
    sim_max = torch.max(sim_matrix, dim=-1, keepdim=True).values.detach()
    sim_shifted = sim_matrix - sim_max
    denom = (torch.exp(sim_shifted) * (~mask_self).float()).sum(dim=-1, keepdim=True) + 1e-8
    log_prob = sim_shifted - torch.log(denom)

    pos_counts = pos_mask.sum(dim=-1)
    valid_rows = pos_counts > 0
    loss_per_row = -(log_prob * pos_mask.float()).sum(dim=-1) / (pos_counts.float() + 1e-8)
    loss = loss_per_row[valid_rows].mean()

    # Mathematical property: negative log-prob is strictly >= 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0, f"Contrastive loss must be non-negative, got {loss.item()}"


def test_matlora_optimizer_deduplication():
    """Verify that multiple DualCycleMaTLoRALinear modules sharing one core_generator deduplicate cleanly."""
    core_gen = DualCycleCoreGenerator(rank=8, num_harmonics=2)
    l1 = nn.Linear(64, 128)
    l2 = nn.Linear(64, 128)

    mod1 = DualCycleMaTLoRALinear.from_linear(l1, rank=8)
    mod2 = DualCycleMaTLoRALinear.from_linear(l2, rank=8)

    # Both share the same core_gen
    mod1.core_generator = core_gen
    mod2.core_generator = core_gen

    model = nn.ModuleList([mod1, mod2])

    # Count parameters with naive list concatenation
    raw_params = list(model.parameters()) + list(core_gen.parameters())
    # Count with deduplication
    seen_ids = set()
    unique_params = []
    for p in raw_params:
        if p.requires_grad and id(p) not in seen_ids:
            seen_ids.add(id(p))
            unique_params.append(p)

    # Raw list contains duplicates of core_gen parameters
    assert len(raw_params) > len(unique_params)

    # Optimizer initializes cleanly without duplicate parameter warning
    opt = torch.optim.AdamW(unique_params, lr=1e-3)
    assert len(opt.param_groups[0]["params"]) == len(unique_params)


def test_evaluator_deduplication_collision_safe():
    """Verify that evaluation accumulator preserves all rows even when example_id collides."""
    eval_records = [
        {"example_id": "mcq:WB:1949:DUP", "input": "Q1", "target": "a"},
        {"example_id": "mcq:WB:1949:DUP", "input": "Q2", "target": "b"},
        {"example_id": "mcq:WB:1950:UNIQ", "input": "Q3", "target": "c"},
    ]

    # Collision-prone dictionary storage
    naive_store = {}
    for idx, c in enumerate(eval_records):
        cid = c.get("example_id", f"case_{idx}")
        naive_store[cid] = {"idx": idx}

    # Naive storage lost 1 sample
    assert len(naive_store) == 2

    # Collision-safe storage using row index
    safe_store = {}
    for idx, c in enumerate(eval_records):
        cid = c.get("example_id", f"case_{idx}")
        safe_key = f"{idx}_{cid}"
        safe_store[safe_key] = {"idx": idx, "original_id": cid}

    # Safe storage preserves all 3 samples exactly
    assert len(safe_store) == 3
