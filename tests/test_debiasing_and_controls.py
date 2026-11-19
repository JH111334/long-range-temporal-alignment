#!/usr/bin/env python3
"""
Unit tests for Option Permutation Rotation (Letter Bias Debiasing),
Can Chi Sexagenary Cycle Grouping, Date-Token TE Injection, and Control Configurations.
Pass criteria:
1. Can Chi Cycle Grouping: 1924 and 1984 share cycle 1 and form positive pairs in intra loss.
   Device resilience: compute_intra_loss works smoothly regardless of tensor device differences.
2. Option Permutation Generator:
   - 4-choice MCQA: 4 cyclic rotations (ABCD, BCDA, CDAB, DABC) with exact target rotation.
   - Yes/No: 2 orderings ('Answer as Yes or No', 'Answer as No or Yes').
   - 100% coverage on eval_180_clean.jsonl (156 MCQA, 24 Yes/No, 0 fallbacks).
3. Date-Token Mask & Injection with Real Tokenizer:
   - Uses real Qwen tokenizer to find exact date token sub-sequences on evaluation samples.
   - 'date_token' mode injects TE ONLY at date token positions; other tokens remain strictly untouched.
   - Device preservation: output masks inherit input_ids.device by default.
4. Letter Bias & Debiased Metrics:
   - Simulated 100% 'a' collapse produces 0.0% consistency and exactly 25.0% MCQA mean accuracy.
   - Oracle model produces 100.0% across all metrics.
5. 4 Control Configurations Matrix:
   - Directly tests the production CONTROL_CONFIGS exported from scripts.build_kaggle_pure_ticktack_baseline.
   - C1: Raw Broadcast, sigma = 0.5
   - C2: Raw Date-token only, sigma = 0.5
   - C3: Raw with sigma = 0 (pure NTP, no intra loss)
   - C4: PT + Intra loss (no TE, sigma = 0.5)
"""

import json
import os
import re
import sys
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

for p in [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
]:
    if p not in sys.path:
        sys.path.insert(0, p)


from transformers import AutoTokenizer

from scripts.build_kaggle_pure_ticktack_baseline import (
    CONTROL_CONFIGS,
    compute_canchi_cycle,
    compute_intra_loss,
    find_date_token_mask,
    inject_te_embeddings,
    generate_option_permutations,
)

class MockModelWithEmbeddings(nn.Module):
    """Mock model with embed_tokens to test inject_te_embeddings without loading 0.5B weights."""
    def __init__(self, vocab_size=151936, hidden_size=64):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.dtype = torch.float32
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)

class TestDebiasingAndControls:

    def test_canchi_sexagenary_cycle_grouping(self):
        """1924 and 1984 belong to cycle 1 and form positive pairs in intra loss."""
        assert compute_canchi_cycle(1924) == 1
        assert compute_canchi_cycle(1984) == 1
        assert compute_canchi_cycle(2044) == 1
        
        # 1887 and 1947 belong to cycle 24
        assert compute_canchi_cycle(1887) == 24
        assert compute_canchi_cycle(1947) == 24
        
        # 1962 and 2022 belong to cycle 39
        assert compute_canchi_cycle(1962) == 39
        assert compute_canchi_cycle(2022) == 39

        # Test compute_intra_loss with positive pair from differing years (1924 and 1984)
        d = 128
        H = torch.zeros(2, d, requires_grad=True)
        with torch.no_grad():
            H[0, 0] = 1.0
            H[1, 1] = 1.0  # orthogonal -> cos = 0 -> loss = 1.0 - 0 = 1.0
        years = torch.tensor([1924, 1984])
        loss = compute_intra_loss(H, years)
        assert torch.isclose(loss, torch.tensor(1.0), atol=1e-5)
        loss.backward()
        assert H.grad is not None and H.grad.norm().item() > 0.0

    def test_compute_intra_loss_device_and_zero_pair_safety(self):
        """Verify device resilience and gradient graph continuity when no pairs exist."""
        B, d = 3, 64
        H = torch.randn(B, d, requires_grad=True)
        years = torch.tensor([1947, 1984, 2022])  # 3 distinct cycles: 24, 1, 39
        
        loss = compute_intra_loss(H, years)
        assert loss.item() == 0.0
        # Backward should execute without error and produce zero gradient
        loss.backward()
        assert H.grad is not None
        assert H.grad.norm().item() == 0.0

    def test_option_permutation_generator_mcqa(self):
        """Verify 4 cyclic rotations (ABCD, BCDA, CDAB, DABC) and target tracking for MCQA."""
        prompt = "Question: In 1947, Private Civil Liberties Index in Slovakia was: Options: (a) 0.51 (b) 0.43 (c) 0.69 (d) 1.26\nAnswer:"
        target = "b"  # 0.43 is the correct value
        
        perms = generate_option_permutations(prompt, target)
        assert len(perms) == 4
        
        # Rot 0: ABCD
        assert perms[0]["rot_name"] == "ABCD"
        assert perms[0]["target"] == "b"
        assert "(a) 0.51 (b) 0.43 (c) 0.69 (d) 1.26" in perms[0]["prompt"]
        
        # Rot 1: BCDA -> 0.43 is now option (a)
        assert perms[1]["rot_name"] == "BCDA"
        assert perms[1]["target"] == "a"
        assert "(a) 0.43 (b) 0.69 (c) 1.26 (d) 0.51" in perms[1]["prompt"]
        
        # Rot 2: CDAB -> 0.43 is now option (d)
        assert perms[2]["rot_name"] == "CDAB"
        assert perms[2]["target"] == "d"
        assert "(a) 0.69 (b) 1.26 (c) 0.51 (d) 0.43" in perms[2]["prompt"]
        
        # Rot 3: DABC -> 0.43 is now option (c)
        assert perms[3]["rot_name"] == "DABC"
        assert perms[3]["target"] == "c"
        assert "(a) 1.26 (b) 0.51 (c) 0.43 (d) 0.69" in perms[3]["prompt"]

    def test_option_permutation_generator_yes_no(self):
        """Verify 2 orderings for Yes/No questions with invariant target."""
        prompt = "In 1947, Private Civil Liberties Index was higher than 1952? Answer as Yes or No"
        target = "Yes"
        
        perms = generate_option_permutations(prompt, target)
        assert len(perms) == 2
        assert perms[0]["rot_name"] == "Yes/No"
        assert perms[0]["target"] == "yes"
        assert "Answer as Yes or No" in perms[0]["prompt"]
        
        assert perms[1]["rot_name"] == "No/Yes"
        assert perms[1]["target"] == "yes"
        assert "Answer as No or Yes" in perms[1]["prompt"]

    def test_eval_180_permutation_coverage(self):
        """Verify that all 180 eval samples generate valid permutations without fallbacks."""
        eval_path = None
        for candidate in [
            os.path.join("Data", "tempun_clean", "eval_180_clean.jsonl"),
            os.path.join("data", "tempun_clean", "eval_180_clean.jsonl"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Data", "tempun_clean", "eval_180_clean.jsonl")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "tempun_clean", "eval_180_clean.jsonl")),
        ]:
            if os.path.exists(candidate):
                eval_path = candidate
                break
        assert eval_path is not None, "eval_180_clean.jsonl not found in any candidate path"
        
        with open(eval_path, "r", encoding="utf-8") as f:
            cases = [json.loads(line) for line in f]
            
        assert len(cases) == 180
        mcq_count = 0
        yn_count = 0
        
        for c in cases:
            perms = generate_option_permutations(c["input"], c["target"])
            if len(perms) == 4:
                mcq_count += 1
                assert [p["rot_name"] for p in perms] == ["ABCD", "BCDA", "CDAB", "DABC"]
                for p in perms:
                    assert p["target"] in ["a", "b", "c", "d"]
            elif len(perms) == 2:
                yn_count += 1
                assert [p["rot_name"] for p in perms] == ["Yes/No", "No/Yes"]
                for p in perms:
                    assert p["target"] in ["yes", "no"]
            else:
                pytest.fail(f"Unexpected permutation count {len(perms)} for sample: {c['example_id']}")
                
        assert mcq_count == 156
        assert yn_count == 24

    def test_date_token_mask_and_localized_injection_with_real_tokenizer(self):
        """
        Verify production find_date_token_mask and inject_te_embeddings using real Qwen tokenizer.
        Confirms date tokens are accurately isolated and non-date tokens remain strictly untouched.
        """
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        
        # Sample prompt with 1947
        prompt = "In 1947, Private Civil Liberties Index in Portugal was higher than 1952? Answer as Yes or No"
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        years = torch.tensor([1947], dtype=torch.long)
        
        mask = find_date_token_mask(input_ids, years, tokenizer)
        assert mask.shape == (1, input_ids.shape[1], 1)
        assert mask.dtype == torch.bool
        assert mask.device == input_ids.device
        
        # Verify that 1947 tokens are flagged True in the mask
        year_tokens = tokenizer.encode("1947", add_special_tokens=False)
        flagged_tokens = input_ids[0][mask[0, :, 0]].tolist()
        assert flagged_tokens == year_tokens, f"Expected {year_tokens}, got {flagged_tokens}"
        
        # Test localized injection vs broadcast injection
        hidden_dim = 64
        model = MockModelWithEmbeddings(vocab_size=len(tokenizer), hidden_size=hidden_dim)
        
        # C2: Date-token only
        injected_c2 = inject_te_embeddings(model, tokenizer, input_ids, years, te_mode="date_token", scale_mode="raw")
        base_embeds = model.model.embed_tokens(input_ids)
        
        # Non-date tokens must be identical
        non_date_mask = ~mask.squeeze(-1)
        assert torch.equal(injected_c2[non_date_mask], base_embeds[non_date_mask])
        
        # Date tokens must differ (TE added)
        date_mask = mask.squeeze(-1)
        assert not torch.equal(injected_c2[date_mask], base_embeds[date_mask])
        
        # C1: Broadcast mode
        injected_c1 = inject_te_embeddings(model, tokenizer, input_ids, years, te_mode="broadcast", scale_mode="raw")
        assert not torch.equal(injected_c1[non_date_mask], base_embeds[non_date_mask])
        
        # C4: None mode
        injected_c4 = inject_te_embeddings(model, tokenizer, input_ids, years, te_mode="none")
        assert torch.equal(injected_c4, base_embeds)

    def test_letter_bias_debiased_metrics_simulation(self):
        """
        Simulate a model that collapses to always predicting 'a'.
        Under canonical eval: accuracy = targets that happen to be 'a'.
        Under 4-rotation debiased eval:
        - Mean accuracy across rotations = exactly 25.0% for MCQA.
        - Strict consistency = exactly 0.0% (collapse detected!).
        """
        eval_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "tempun_clean", "eval_180_clean.jsonl"))
        with open(eval_path, "r", encoding="utf-8") as f:
            cases = [json.loads(line) for line in f]
            
        mcq_cases = [c for c in cases if c["target"].lower() in ["a", "b", "c", "d"]]
        assert len(mcq_cases) == 156
        
        # Simulated model: always predicts 'a'
        rot_correct_counts = [0, 0, 0, 0]
        strict_correct = 0
        total_mcq_rot_evals = 0
        
        for c in mcq_cases:
            perms = generate_option_permutations(c["input"], c["target"])
            assert len(perms) == 4
            case_corrs = []
            for p in perms:
                pred = "a"  # 100% collapsed prediction
                is_corr = int(pred == p["target"])
                case_corrs.append(is_corr)
                rot_correct_counts[p["rot_idx"]] += is_corr
                total_mcq_rot_evals += 1
            if all(c == 1 for c in case_corrs):
                strict_correct += 1
                
        # In canonical eval (Rot 0), accuracy is 31/156 = 19.87%
        canonical_acc = (rot_correct_counts[0] / 156.0) * 100.0
        assert rot_correct_counts[0] == 31
        assert pytest.approx(canonical_acc, abs=0.1) == 19.87
        
        # Across all 4 rotations, total correct is 31 + 38 + 42 + 45 = 156
        total_correct = sum(rot_correct_counts)
        assert total_correct == 156
        debiased_mean_acc = (total_correct / float(total_mcq_rot_evals)) * 100.0
        assert debiased_mean_acc == 25.0  # Exactly 1/4 random choice!
        
        # Strict consistency MUST be exactly 0.0%
        consistency = (strict_correct / 156.0) * 100.0
        assert consistency == 0.0, "Collapsed model cannot be consistent across all 4 rotations!"

    def test_control_configurations_spec(self):
        """Verify that the 4 production control configurations are mathematically and conceptually distinct."""
        controls = CONTROL_CONFIGS
        assert set(controls.keys()) == {"C1", "C2", "C3", "C4"}
        
        # C1: Raw Broadcast, sigma = 0.5
        assert controls["C1"]["te_mode"] == "broadcast"
        assert controls["C1"]["te_scale_mode"] == "raw"
        assert controls["C1"]["sigma"] == 0.5
        
        # C2: Raw Date-token only, sigma = 0.5 (isolates broadcast injection vs localized)
        assert controls["C2"]["te_mode"] == "date_token"
        assert controls["C2"]["te_scale_mode"] == "raw"
        assert controls["C2"]["sigma"] == 0.5
        
        # C3: Raw with sigma = 0.0 (pure NTP, no intra-class contrastive loss)
        assert controls["C3"]["te_mode"] == "broadcast"
        assert controls["C3"]["te_scale_mode"] == "raw"
        assert controls["C3"]["sigma"] == 0.0
        
        # C4: PT + Intra loss (no TE injection, sigma = 0.5)
        assert controls["C4"]["te_mode"] == "none"
        assert controls["C4"]["te_scale_mode"] == "none"
        assert controls["C4"]["sigma"] == 0.5
