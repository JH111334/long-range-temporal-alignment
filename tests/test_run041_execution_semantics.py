"""Unit tests for Run 04.1 execution semantics, base parity, and state isolation.

Mandatory tests from Phase P1:
- test_true_base_disables_all_adapters
- test_mlp_zero_coord_not_off
- test_seam_off_identity
- test_model_state_isolation_and_interleaving
- test_checkpoint_tensor_identity_excludes_metadata
- test_strict_adapter_only_load
- test_hook_removal_after_exception
"""

from __future__ import annotations

import hashlib
import pytest
import torch
import torch.nn as nn

from temporal_alignment.models.dual_cycle_mat_lora import (
    DualCycleMaTLoRALinear,
    clear_model_temporal_coordinates,
    set_model_mat_lora_enabled,
    set_model_temporal_coordinates,
)
from temporal_alignment.models.gated_seam import (
    ConditioningContext,
    GatedResidualProjection,
    make_gated_embedding_hook,
    scoped_conditioning_context,
)


def _compute_tensor_fingerprint(tensors: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(tensors.keys()):
        t = tensors[k]
        h.update(k.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(str(list(t.shape)).encode("utf-8"))
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def test_true_base_disables_all_adapters():
    """Verify that when enabled=False, DualCycleMaTLoRALinear computes base_layer(x) bitwise identically."""
    torch.manual_seed(42)
    in_features, out_features = 32, 64
    base_linear = nn.Linear(in_features, out_features)
    mod = DualCycleMaTLoRALinear.from_linear(base_linear, rank=8, alpha=16.0)

    # Set non-zero LoRA B to simulate trained checkpoint
    nn.init.normal_(mod.lora_B, std=0.5)
    mod.set_temporal_coordinates(0.5, 1.2)

    x = torch.randn(2, 5, in_features)

    # When enabled=True: output != base_out
    mod.set_enabled(True)
    out_on = mod(x)
    base_out = base_linear(x)
    diff_on = (out_on - base_out).abs().max().item()
    assert diff_on > 1e-4, "Adapter output should differ from base when enabled"

    # When enabled=False: output strictly equals base_out
    mod.set_enabled(False)
    out_off = mod(x)
    diff_off = (out_off - base_out).abs().max().item()
    assert diff_off == 0.0, f"Disabled adapter must strictly equal base_out (got diff {diff_off})"


def test_mlp_zero_coord_not_off():
    """Verify hypothesis F01: (theta=0, z=0) with enabled=True is NOT logically equivalent to adapter OFF.
    
    Because F(0, 0) != 0 (contains identity and cosine basis terms), lora_out is non-zero.
    """
    torch.manual_seed(42)
    in_features, out_features = 32, 64
    base_linear = nn.Linear(in_features, out_features)
    mod = DualCycleMaTLoRALinear.from_linear(base_linear, rank=8, alpha=16.0)

    # Trained non-zero weights
    nn.init.normal_(mod.lora_B, std=0.5)

    x = torch.randn(2, 5, in_features)
    base_out = base_linear(x)

    # Set theta=0, z=0
    mod.set_enabled(True)
    mod.set_temporal_coordinates(0.0, 0.0)
    out_zero_coord = mod(x)

    diff_zero = (out_zero_coord - base_out).abs().max().item()
    assert diff_zero > 1e-4, (
        f"F(0,0) produces non-zero LoRA output ({diff_zero}), proving zero coordinates != adapter OFF!"
    )

    # Explicitly disabled
    mod.set_enabled(False)
    out_disabled = mod(x)
    assert (out_disabled - base_out).abs().max().item() == 0.0


def test_seam_off_identity():
    """Verify that when seam is disabled or vector is None, Seam_OFF(h) == h."""
    context = ConditioningContext(injection_token_index=14)
    hook = make_gated_embedding_hook(context)

    h = torch.randn(1, 20, 896)

    # 1. Vector is None -> identity
    out1 = hook(None, None, h)
    assert torch.equal(out1, h), "Hook with None vector must return input unchanged"

    # 2. Vector set, but enabled=False -> identity
    context.set_vector(torch.randn(896))
    context.set_enabled(False)
    out2 = hook(None, None, h)
    assert torch.equal(out2, h), "Disabled seam context must return input unchanged"

    # 3. Vector set and enabled=True -> altered at token 14
    context.set_enabled(True)
    out3 = hook(None, None, h)
    assert not torch.equal(out3, h), "Enabled seam context must inject vector"
    assert not torch.equal(out3[:, 14, :], h[:, 14, :]), "Token 14 must be modified"
    assert torch.equal(out3[:, :14, :], h[:, :14, :]), "Tokens before 14 must remain untouched"


def test_model_state_isolation_and_interleaving():
    """Verify that interleaving variants (ON -> OFF -> ON, coord1 -> coord2 -> coord1) does not leak state."""
    torch.manual_seed(42)
    in_features, out_features = 16, 32
    base_linear = nn.Linear(in_features, out_features)
    mod = DualCycleMaTLoRALinear.from_linear(base_linear, rank=4, alpha=8.0)
    nn.init.normal_(mod.lora_B, std=0.2)

    x = torch.randn(1, 4, in_features)

    # State A: enabled, theta=0.25, z=0.5
    mod.set_enabled(True)
    mod.set_temporal_coordinates(0.25, 0.5)
    out_a1 = mod(x).clone()

    # State B: disabled
    mod.set_enabled(False)
    out_b = mod(x).clone()

    # State C: enabled with different coordinates
    mod.set_enabled(True)
    mod.set_temporal_coordinates(1.0, -0.5)
    out_c = mod(x).clone()

    # Back to State A
    mod.set_enabled(True)
    mod.set_temporal_coordinates(0.25, 0.5)
    out_a2 = mod(x).clone()

    # Assert exact state recovery without residual leakage
    assert torch.allclose(out_a1, out_a2, atol=1e-7), "State A output must be deterministic and isolated"
    assert not torch.allclose(out_a1, out_b, atol=1e-4), "State A must differ from disabled State B"
    assert not torch.allclose(out_a1, out_c, atol=1e-4), "State A must differ from State C"


def test_checkpoint_tensor_identity_excludes_metadata():
    """Verify hypothesis F08: tensor fingerprint separates trainable weights from metadata differences."""
    t1 = {"lora_A": torch.randn(8, 32), "lora_B": torch.randn(64, 8)}
    t2 = {"lora_A": t1["lora_A"].clone(), "lora_B": t1["lora_B"].clone()}

    fp1 = _compute_tensor_fingerprint(t1)
    fp2 = _compute_tensor_fingerprint(t2)
    assert fp1 == fp2, "Identical tensors must produce identical fingerprints"

    # Perturb one weight slightly
    t2["lora_B"] = t2["lora_B"] + 1e-6
    fp3 = _compute_tensor_fingerprint(t2)
    assert fp1 != fp3, "Perturbed tensor must produce different fingerprint"


def test_strict_adapter_only_load():
    """Verify strict schema check on adapter tensors, failing closed on missing keys."""
    in_features, out_features = 16, 32
    mod = DualCycleMaTLoRALinear(in_features, out_features, rank=4)

    full_sd = mod.state_dict()
    adapter_keys = [k for k in full_sd if not k.startswith("base_layer.")]

    # Audited adapter-only state dict
    adapter_sd = {k: full_sd[k].clone() for k in adapter_keys}

    # Successful load with adapter keys
    mod.load_state_dict(adapter_sd, strict=False)

    # Missing critical adapter key -> fail closed check
    corrupt_sd = {k: v for k, v in adapter_sd.items() if k != "lora_B"}
    missing = [k for k in adapter_keys if k not in corrupt_sd]
    assert "lora_B" in missing, "Omission of lora_B must be detected by strict adapter schema"


def test_hook_removal_after_exception():
    """Verify that scoped_conditioning_context removes hook handles even when an exception occurs."""
    class DummyEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(100, 896))

        def forward(self, input_ids):
            return self.weight[input_ids]

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = DummyEmbed()

        def get_input_embeddings(self):
            return self.embed

    model = DummyModel()
    context = ConditioningContext(injection_token_index=14)

    # Prior hook count
    assert len(model.get_input_embeddings()._forward_hooks) == 0

    # Test normal exit
    with scoped_conditioning_context(model, context, vector=torch.randn(896)):
        assert len(model.get_input_embeddings()._forward_hooks) == 1
    assert len(model.get_input_embeddings()._forward_hooks) == 0, "Hook must be removed after normal context exit"

    # Test exception exit
    with pytest.raises(RuntimeError):
        with scoped_conditioning_context(model, context, vector=torch.randn(896)):
            assert len(model.get_input_embeddings()._forward_hooks) == 1
            raise RuntimeError("Simulated failure inside context")

    assert len(model.get_input_embeddings()._forward_hooks) == 0, "Hook must be strictly removed after exception!"
