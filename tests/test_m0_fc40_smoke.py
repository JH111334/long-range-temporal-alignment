"""Unit and smoke tests for M0-FC40 components on CPU."""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from temporal_alignment.models.fone_canchi_40d import (
    compute_fixed_fone_canchi_40d,
    extract_canchi_angles_from_40d,
    verify_golden_manifest,
)
from temporal_alignment.models.gated_seam import (
    ConditioningContext,
    GatedResidualProjection,
    make_gated_embedding_hook,
)
from temporal_alignment.models.dual_cycle_mat_lora import (
    DualCycleCoreGenerator,
    DualCycleMaTLoRALinear,
    replace_mlp_with_dual_cycle_mat_lora,
    set_model_temporal_coordinates,
)
from temporal_alignment.models.tempun_dataset import (
    parse_strict_mcqa_answer,
)


def test_fone_canchi_40d_golden_manifest() -> None:
    manifest_path = "artifacts/releases/FCLEARN_105/feature_manifest.json"
    assert verify_golden_manifest(manifest_path) is True


def test_fone_canchi_40d_shapes_and_extraction() -> None:
    vec = compute_fixed_fone_canchi_40d(1947)
    assert len(vec) == 40
    theta, z = extract_canchi_angles_from_40d(vec)
    assert isinstance(theta, float)
    assert isinstance(z, float)
    assert -math.pi <= theta <= math.pi
    assert abs(z - (1947 - 1900) / 60.0) < 1e-5


def test_gated_residual_seam() -> None:
    proj = GatedResidualProjection(in_features=40, out_features=896)
    assert proj.alpha.item() == 0.0

    x = torch.randn(1, 40)
    out_zero = proj(x)
    assert torch.allclose(out_zero, torch.zeros_like(out_zero))

    # Gradient flows when alpha is active
    with torch.no_grad():
        proj.alpha.fill_(1.0)
    out_active = proj(x)
    assert not torch.allclose(out_active, torch.zeros_like(out_active))
    assert out_active.shape == (1, 896)

    loss = out_active.sum()
    loss.backward()
    assert proj.alpha.grad is not None and torch.isfinite(proj.alpha.grad).all()
    assert proj.linear.weight.grad is not None and torch.isfinite(proj.linear.weight.grad).all()


def test_embedding_hook_token14() -> None:
    ctx = ConditioningContext(injection_token_index=14)
    hook_fn = make_gated_embedding_hook(ctx)

    embed = nn.Embedding(100, 896)
    embed.register_forward_hook(hook_fn)

    # Context empty: unchanged
    inp = torch.randint(0, 100, (1, 20))
    out1 = embed(inp)
    assert out1.shape == (1, 20, 896)

    # Context active
    vec = torch.ones(896) * 2.5
    ctx.set_vector(vec)
    out2 = embed(inp)
    diff = out2 - out1
    # Check that position 14 is altered by exactly 2.5, and other positions 0
    assert torch.allclose(diff[0, 14, :], vec)
    diff_other = diff[0, :14, :]
    assert torch.allclose(diff_other, torch.zeros_like(diff_other))


def test_dual_cycle_core_generator() -> None:
    core_gen = DualCycleCoreGenerator(rank=8, num_harmonics=2)
    theta = torch.tensor([1.5])
    z = torch.tensor([0.78])
    f_core = core_gen(theta, z)
    assert f_core.shape == (1, 8, 8)
    assert torch.isfinite(f_core).all()

    loss = f_core.sum()
    loss.backward()
    assert core_gen.cos_basis.grad is not None and torch.isfinite(core_gen.cos_basis.grad).all()
    assert core_gen.sin_basis.grad is not None and torch.isfinite(core_gen.sin_basis.grad).all()
    assert core_gen.epoch_linear.grad is not None and torch.isfinite(core_gen.epoch_linear.grad).all()


def test_dual_cycle_mat_lora_linear_gradient() -> None:
    linear = nn.Linear(896, 4864, bias=False)
    adapted = DualCycleMaTLoRALinear.from_linear(linear, rank=8, alpha=16.0)

    # Base weight frozen
    assert adapted.base_layer.weight.requires_grad is False
    assert adapted.lora_A.requires_grad is True
    assert adapted.lora_B.requires_grad is True

    x = torch.randn(1, 16, 896)
    theta = torch.tensor([2.4])
    z = torch.tensor([0.78])
    out = adapted(x, theta=theta, z=z)
    assert out.shape == (1, 16, 4864)
    assert torch.isfinite(out).all()

    loss = out.sum()
    loss.backward()
    assert adapted.base_layer.weight.grad is None
    assert adapted.lora_A.grad is not None and torch.isfinite(adapted.lora_A.grad).all()
    assert adapted.lora_B.grad is not None and torch.isfinite(adapted.lora_B.grad).all()
    assert adapted.core_generator.cos_basis.grad is not None and torch.isfinite(adapted.core_generator.cos_basis.grad).all()


def test_strict_mcqa_answer_parser() -> None:
    assert parse_strict_mcqa_answer("A") == "a"
    assert parse_strict_mcqa_answer("(B)") == "b"
    assert parse_strict_mcqa_answer("C. Explanation here") == "c"
    assert parse_strict_mcqa_answer("a") == "a"
    assert parse_strict_mcqa_answer("(b)") == "b"
    assert parse_strict_mcqa_answer("Yes") == "Yes"
    assert parse_strict_mcqa_answer("No.") == "No"
    assert parse_strict_mcqa_answer("yes, definitely") == "Yes"
    assert parse_strict_mcqa_answer("Random invalid gibberish") is None


def test_parser_adversarial_fixtures() -> None:
    # Option prefixes
    assert parse_strict_mcqa_answer("Option: c") == "c"
    assert parse_strict_mcqa_answer("Answer: (d)") == "d"
    assert parse_strict_mcqa_answer("Choice (a)") == "a"
    assert parse_strict_mcqa_answer("Select: b") == "b"

    # Numeric and empty strings must fail
    assert parse_strict_mcqa_answer("") is None
    assert parse_strict_mcqa_answer("   \n\t  ") is None
    assert parse_strict_mcqa_answer("80000000") is None
    assert parse_strict_mcqa_answer("30800000") is None
    assert parse_strict_mcqa_answer("1947") is None

    # Long free-form text without matching prefix must fail
    assert parse_strict_mcqa_answer("This is a very long text explaining why the historical event happened in that decade") is None
    assert parse_strict_mcqa_answer("Neither option applies because the data is inconclusive") is None
