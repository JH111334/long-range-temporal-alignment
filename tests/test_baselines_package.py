#!/usr/bin/env python3
"""
Unit tests for the centralized `baselines` package and `baselines.ticktack` modules.

Tests:
1. Package imports and public API exports.
2. Models (Eq 1-4, scale modes 'raw', 'scaled', 'gated', LoRALinear, apply_lora_to_model).
3. Contrastive intra loss (Eq 8, cycle pairing, empty/zero pair safety, compute_total_loss).
4. Date-token masking and localized TE injection.
5. Option permutation generator and debiasing evaluation structures.
6. Backward-compatibility of scripts.build_kaggle_pure_ticktack_baseline forwarder shim.
7. Verification of saved benchmark artifact schemas and notebook paths.
"""

import json
import math
import os
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import baselines
import baselines.ticktack as tt
from baselines.ticktack.models import (
    CONTROL_CONFIGS,
    compute_canchi_cycle,
    compute_batch_te,
    compute_te_vector,
    LoRALinear,
    apply_lora_to_model,
)
from baselines.ticktack.losses import (
    compute_intra_loss,
    compute_total_loss,
)
from baselines.ticktack.date_mask import (
    find_date_token_mask,
    inject_te_embeddings,
)
from baselines.ticktack.eval_debiasing import (
    generate_option_permutations,
)
from baselines.ticktack.build_kaggle_notebook import (
    CLEAN_TRAIN_PATH,
    CLEAN_EVAL_PATH,
    OUTPUT_DIR,
    NOTEBOOK_PATH,
    METADATA_PATH,
    _find_clean_path,
)
import scripts.build_kaggle_pure_ticktack_baseline as shim

class MockLinearBackbone(nn.Module):
    def __init__(self, in_features=64, out_features=64):
        super().__init__()
        self.q_proj = nn.Linear(in_features, out_features)
        self.k_proj = nn.Linear(in_features, out_features)
        self.v_proj = nn.Linear(in_features, out_features)
        self.config = type("Config", (), {"hidden_size": in_features})()
        self.dtype = torch.float32

class MockTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text.strip() == "1984":
            return [1984]
        elif text.strip() == "1947":
            return [1947]
        return [ord(c) for c in text]

    def decode(self, token_ids, **kwargs):
        return "".join([chr(tid) if 0 <= tid < 256 else str(tid) for tid in token_ids])

class TestBaselinesPackage:

    def test_package_structure_and_exports(self):
        """Verify that baselines and baselines.ticktack expose all required symbols."""
        assert hasattr(baselines, "__version__")
        expected_exports = [
            "CONTROL_CONFIGS",
            "compute_canchi_cycle",
            "compute_batch_te",
            "compute_te_vector",
            "LoRALinear",
            "apply_lora_to_model",
            "compute_intra_loss",
            "compute_total_loss",
            "find_date_token_mask",
            "inject_te_embeddings",
            "generate_option_permutations",
            "eval_choice_logit_single",
            "evaluate_dataset",
            "create_kaggle_notebook",
        ]
        for name in expected_exports:
            assert hasattr(tt, name), f"baselines.ticktack missing symbol: {name}"

    def test_backward_compatibility_shim(self):
        """Verify that scripts.build_kaggle_pure_ticktack_baseline re-exports identical symbols."""
        for name in [
            "CONTROL_CONFIGS",
            "compute_canchi_cycle",
            "compute_intra_loss",
            "find_date_token_mask",
            "inject_te_embeddings",
            "generate_option_permutations",
            "create_kaggle_notebook",
            "CLEAN_TRAIN_PATH",
            "CLEAN_EVAL_PATH",
        ]:
            assert hasattr(shim, name), f"Shim missing backward-compatible symbol: {name}"
            # Functional identity check
            assert getattr(shim, name) is getattr(tt, name) or getattr(shim, name) == getattr(tt, name)

    def test_canchi_cycle_conversions(self):
        """Verify Eq. 1 across standard years, historical landmarks, and BC boundaries."""
        assert compute_canchi_cycle(1984) == 1
        assert compute_canchi_cycle(1924) == 1
        assert compute_canchi_cycle(2044) == 1
        assert compute_canchi_cycle(1947) == 24
        assert compute_canchi_cycle(1887) == 24
        assert compute_canchi_cycle(2022) == 39
        assert compute_canchi_cycle(4) == 1
        assert compute_canchi_cycle(3) == 0
        assert compute_canchi_cycle(1) == 58
        assert compute_canchi_cycle(-1) == 57

    def test_sinusoidal_te_scale_modes(self):
        """Verify 'raw', 'scaled', and 'gated' scale modes for TE vectors."""
        years = torch.tensor([1984, 1947])
        d = 64
        te_raw = compute_batch_te(years, hidden_dim=d, scale_mode='raw')
        assert te_raw.shape == (2, 1, d)
        norm_raw = torch.norm(te_raw[0, 0, :], p=2).item()

        te_scaled = compute_batch_te(years, hidden_dim=d, scale_mode='scaled')
        assert te_scaled.shape == (2, 1, d)
        norm_scaled = torch.norm(te_scaled[0, 0, :], p=2).item()
        assert math.isclose(norm_scaled, norm_raw / float(d), rel_tol=1e-5)

        gating = 0.05
        te_gated = compute_batch_te(years, hidden_dim=d, scale_mode='gated', gating_factor=gating)
        norm_gated = torch.norm(te_gated[0, 0, :], p=2).item()
        assert math.isclose(norm_gated, norm_raw * gating, rel_tol=1e-5)

        with pytest.raises(ValueError, match="Unknown scale_mode"):
            compute_batch_te(years, hidden_dim=d, scale_mode='invalid_mode')

    def test_lora_linear_and_application(self):
        """Verify LoRALinear forward pass, rank=0 pass-through, and application to model."""
        backbone = MockLinearBackbone(in_features=32, out_features=32)
        adapted = apply_lora_to_model(backbone, target_modules=['q_proj', 'v_proj'], rank=4, alpha=8.0)
        assert 'q_proj' in adapted
        assert 'v_proj' in adapted
        assert isinstance(backbone.q_proj, LoRALinear)
        assert isinstance(backbone.v_proj, LoRALinear)
        assert not isinstance(backbone.k_proj, LoRALinear)

        x = torch.randn(2, 32)
        out = backbone.q_proj(x)
        assert out.shape == (2, 32)

    def test_intra_loss_and_total_loss(self):
        """Verify Eq. 8 contrastive clustering, zero pair safety, 3D tensor support, and total loss helper."""
        h = torch.randn(4, 32, requires_grad=True)
        # 1984 and 1924 share cycle 1
        years_with_pairs = torch.tensor([1984, 1924, 1947, 2022])
        loss = compute_intra_loss(h, years_with_pairs)
        assert loss.item() >= 0.0
        loss.backward()
        assert h.grad is not None

        # 3D Sequence tensor support [B, S, D]
        h_3d = torch.randn(4, 8, 32, requires_grad=True)
        loss_3d = compute_intra_loss(h_3d, [1984, 1924, 1947, 2022])
        assert loss_3d.item() >= 0.0
        loss_3d.backward()
        assert h_3d.grad is not None

        # All different cycles
        h2 = torch.randn(3, 32, requires_grad=True)
        years_no_pairs = torch.tensor([1984, 1985, 1986])
        loss_no_pairs = compute_intra_loss(h2, years_no_pairs)
        assert loss_no_pairs.item() == 0.0

        ntp_loss = torch.tensor(2.5)
        intra_l = torch.tensor(0.4)
        total_l = compute_total_loss(ntp_loss, intra_l, sigma=0.5)
        assert math.isclose(total_l.item(), 2.7, rel_tol=1e-5)

    def test_batch_te_and_date_mask_edge_cases(self):
        """Verify compute_batch_te and find_date_token_mask with list and scalar inputs."""
        # Python list input
        te_list = compute_batch_te([1984, 1947], hidden_dim=32)
        assert te_list.shape == (2, 1, 32)

        # Scalar int input
        te_scalar = compute_batch_te(1984, hidden_dim=32)
        assert te_scalar.shape == (1, 1, 32)

        # 0D tensor input
        te_0d = compute_batch_te(torch.tensor(1984), hidden_dim=32)
        assert te_0d.shape == (1, 1, 32)

        # Date token mask with 0D tensor and scalar int
        tokenizer = MockTokenizer()
        input_ids = torch.tensor([[10, 1984, 20, 30]])
        mask_0d = find_date_token_mask(input_ids, torch.tensor(1984), tokenizer)
        assert mask_0d.shape == (1, 4, 1)
        assert mask_0d[0, 1, 0].item() is True

        mask_int = find_date_token_mask(input_ids, 1984, tokenizer)
        assert mask_int.shape == (1, 4, 1)
        assert mask_int[0, 1, 0].item() is True

    def test_lora_rank_zero_and_leaf_matching(self):
        """Verify LoRALinear handles rank=0 without warnings and apply_lora matches exact leaves."""
        lora_zero = LoRALinear(in_features=16, out_features=16, rank=0)
        x = torch.randn(2, 16)
        out_zero = lora_zero(x)
        assert out_zero.shape == (2, 16)

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Module()
                self.layer.q_proj = nn.Linear(16, 16)
                self.layer.sq_proj = nn.Linear(16, 16)

        nest = NestedModel()
        adapted = apply_lora_to_model(nest, target_modules=['q_proj'], rank=2)
        assert 'layer.q_proj' in adapted
        assert 'layer.sq_proj' not in adapted
        assert isinstance(nest.layer.q_proj, LoRALinear)
        assert not isinstance(nest.layer.sq_proj, LoRALinear)

    def test_eval_choice_logit_with_mock_model(self):
        """Verify eval_choice_logit_single works safely on small vocabulary mock models."""
        class MockSmallLM(nn.Module):
            def __init__(self, vocab_size=100):
                super().__init__()
                self.vocab_size = vocab_size
                self.param = nn.Parameter(torch.zeros(1))
            def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
                b = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
                s = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
                logits = torch.randn(b, s, self.vocab_size)
                # Ensure option 'b' has highest logit
                logits[0, -1, ord('b')] = 10.0
                return type("Output", (), {"logits": logits})()

        model = MockSmallLM(vocab_size=256)
        tok = MockTokenizer()
        prompt = "Question: Test? Options: (a) A (b) B (c) C (d) D\nAnswer:"
        res = tt.eval_choice_logit_single(model, tok, prompt, "b")
        assert res["is_valid"] == 1
        assert res["prediction"] == "b"
        assert res["is_correct"] == 1

    def test_subpackages_exposure(self):
        """Verify baselines.fone and baselines.discrete subpackages are properly exposed."""
        import baselines.fone as fone_pkg
        import baselines.discrete as disc_pkg
        assert hasattr(baselines, "fone")
        assert hasattr(baselines, "discrete")
        assert "fone" in baselines.__all__
        assert "discrete" in baselines.__all__

    def test_date_token_mask_and_localized_injection(self):
        """Verify date token mask identification and localized injection."""
        tokenizer = MockTokenizer()
        input_ids = torch.tensor([[10, 1984, 20, 30], [10, 1947, 20, 30]])
        years = torch.tensor([1984, 1947])
        mask = find_date_token_mask(input_ids, years, tokenizer)
        assert mask.shape == (2, 4, 1)
        assert mask[0, 1, 0].item() is True
        assert mask[0, 0, 0].item() is False

        model = nn.Module()
        model.config = type("Config", (), {"hidden_size": 32})()
        model.dtype = torch.float32
        model.model = nn.Module()
        model.model.embed_tokens = nn.Embedding(3000, 32)

        # Broadcast vs Date-token
        embed_none = inject_te_embeddings(model, tokenizer, input_ids, years, te_mode='none')
        embed_bc = inject_te_embeddings(model, tokenizer, input_ids, years, te_mode='broadcast')
        embed_dt = inject_te_embeddings(model, tokenizer, input_ids, years, te_mode='date_token')

        assert torch.allclose(embed_none[0, 0, :], embed_dt[0, 0, :])
        assert not torch.allclose(embed_none[0, 1, :], embed_dt[0, 1, :])
        assert not torch.allclose(embed_none[0, 0, :], embed_bc[0, 0, :])

    def test_option_permutations(self):
        """Verify 4-fold cyclic rotation for MCQA and 2-orderings for Yes/No."""
        prompt = "Question: What occurred in 1984? Options: (a) OptA (b) OptB (c) OptC (d) OptD\nAnswer:"
        perms = generate_option_permutations(prompt, "c")
        assert len(perms) == 4
        assert perms[0]["target"] == "c"
        assert perms[1]["target"] == "b"
        assert perms[2]["target"] == "a"
        assert perms[3]["target"] == "d"
        assert perms[0]["rot_name"] == "ABCD"
        assert perms[1]["rot_name"] == "BCDA"
        assert perms[2]["rot_name"] == "CDAB"
        assert perms[3]["rot_name"] == "DABC"

        yn_prompt = "Question: Was 1984 significant? Answer as Yes or No\nAnswer:"
        yn_perms = generate_option_permutations(yn_prompt, "yes")
        assert len(yn_perms) == 2
        assert "Yes or No" in yn_perms[0]["prompt"]
        assert "No or Yes" in yn_perms[1]["prompt"]

    def test_saved_artifacts_presence_and_integrity(self):
        """Verify presence and valid schema of notebook and output JSON artifacts."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        nb_path = os.path.join(repo_root, "baselines", "ticktack", "notebooks", "ticktack_paper_rebuild.ipynb")
        results_path = os.path.join(repo_root, "baselines", "ticktack", "outputs", "TICKTACK_CONTROL_MATRIX_RESULTS.json")
        rebuild_path = os.path.join(repo_root, "baselines", "ticktack", "outputs", "TICKTACK_REBUILD_BENCHMARK_RESULTS.json")
        log_path = os.path.join(repo_root, "baselines", "ticktack", "outputs", "ticktack-paper-rebuild-baseline.log")

        assert os.path.exists(nb_path), f"Notebook missing at {nb_path}"
        assert os.path.exists(results_path), f"Results missing at {results_path}"
        assert os.path.exists(rebuild_path), f"Rebuild results missing at {rebuild_path}"
        assert os.path.exists(log_path), f"Log missing at {log_path}"

        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "summary" in data
        assert "base" in data["summary"]
        assert "c1_raw_broadcast" in data["summary"]
        assert "c2_raw_date_token" in data["summary"]
        assert "c3_raw_sigma0" in data["summary"]
        assert "c4_pt_intra_loss" in data["summary"]
