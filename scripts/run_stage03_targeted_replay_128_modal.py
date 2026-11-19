"""
STAGE 03 GPU REPLAY & AUDIT ON 128 FROZEN SAMPLES (MODAL GPU A10G)

Authorized on: 2026-11-05
Budget Cap: $0.35 USD (Hard Timeout: 600s)
Hardware: 1x NVIDIA A10G (24GB VRAM)
Volume: 'stage03-artifacts' mounted at /vol

Executes:
1. S0_HISTORICAL_LETTER: Exact replication of frozen evaluation logic (batch size 4, left padding,
   token_idx=14, unnormalized multi-token max extraction). Matches historical anchors.
2. S1_AUDITED: Evaluator defect correction (batch size 1 unpadded, dynamic year injection via
   locate_year_tokens, clean space-prefixed candidate tokens ' A'/' B'/' C'/' D' and ' Yes'/' No').
3. Paired delta analysis (S1 vs S0) isolating evaluator bug impact on Arm 4 and Arm 5.
"""

from __future__ import annotations

import argparse
import io
import gzip
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

app = modal.App("stage03-qwen3b-targeted-replay-128")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "numpy>=1.26.0",
        "scipy",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

volume = modal.Volume.from_name("stage03-artifacts", create_if_missing=True)


# =========================================================================
# MODAL REMOTE WORKER FUNCTION
# =========================================================================

@app.function(image=image, gpu="A10G", timeout=600, volumes={"/vol": volume})
def run_arm_replay_and_audit(
    arm_type: str,  # 'static_mlp' or 'matlora'
    samples_json_str: str,
    rank: int = 8,
    alpha: float = 16.0,
) -> Dict[str, Any]:
    import gc
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    device = "cuda:0"
    print("\n=======================================================")
    print(f"STARTING BOUNDED 128-SAMPLE REPLAY & AUDIT: ARM [{arm_type.upper()}]")
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=======================================================\n")

    # Parse 128 samples
    eval_cases = json.loads(samples_json_str)
    print(f"Loaded {len(eval_cases)} frozen replay samples.")

    # --- 1. Math & Feature Functions ---
    def compute_fone_32d(year_f: float) -> List[float]:
        periods = [10.0 * (1000.0 ** (i / 15.0)) for i in range(16)]
        feats = []
        for p in periods:
            omega = 2.0 * math.pi / p
            feats.append(math.sin(omega * year_f))
            feats.append(math.cos(omega * year_f))
        return feats

    def compute_canchi_8d(year_int: int) -> List[float]:
        offset = year_int - 4
        stem_idx = offset % 10
        branch_idx = offset % 12
        cycle_idx = offset % 60
        theta_stem = 2.0 * math.pi * float(stem_idx) / 10.0
        theta_branch = 2.0 * math.pi * float(branch_idx) / 12.0
        theta_cycle = 2.0 * math.pi * float(cycle_idx) / 60.0
        radius = 1.0 + float(year_int - 2000) / 60.0
        return [
            math.sin(theta_stem), math.cos(theta_stem),
            math.sin(theta_branch), math.cos(theta_branch),
            math.sin(theta_cycle), math.cos(theta_cycle),
            float(radius), float(cycle_idx) / 60.0,
        ]

    def compute_fixed_fone_canchi_40d(year: int) -> List[float]:
        return compute_fone_32d(float(year)) + compute_canchi_8d(year)

    def compute_theta_z(year: int) -> Tuple[float, float]:
        offset = year - 3
        cycle = offset % 60
        theta = float(cycle) * 6.0 * (math.pi / 180.0)
        z = float(year - 1947) / 75.0
        return theta, z

    # --- 2. Architecture Modules ---
    class GatedResidualProjection(nn.Module):
        def __init__(self, in_features: int = 40, out_features: int = 2048):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
            self.norm = nn.LayerNorm(out_features, bias=False)
            self.alpha = nn.Parameter(torch.zeros(1))

        def forward(self, v_time: torch.Tensor) -> torch.Tensor:
            h = self.linear(v_time)
            h = self.norm(h)
            gate = torch.sigmoid(self.alpha)
            return gate * h

    def inject_seam_prefill_safe(model, seam_proj, input_ids: torch.Tensor, v40: torch.Tensor, token_idx: int = 14):
        embed_tokens = model.get_input_embeddings()
        inputs_embeds = embed_tokens(input_ids)
        B, S, D = inputs_embeds.shape
        if S > token_idx:
            inj = seam_proj(v40.to(inputs_embeds.device, dtype=inputs_embeds.dtype))
            delta = torch.zeros_like(inputs_embeds)
            delta[:, token_idx, :] = inj
            inputs_embeds = inputs_embeds + delta
        return inputs_embeds

    class StaticLoRALinear(nn.Module):
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

            device = original_linear.weight.device
            dtype = original_linear.weight.dtype
            self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=dtype))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, rank, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base_out = self.base_layer(x)
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            return base_out + lora_out

    class DualCycleCoreGenerator(nn.Module):
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
                f_core = f_core + (cos_val * self.cos_basis[k_idx].unsqueeze(0) + sin_val * self.sin_basis[k_idx].unsqueeze(0))

            f_core = f_core + zv.view(B, 1, 1) * self.epoch_linear.unsqueeze(0)
            z_sq = ((zv / 10.0) ** 2).view(B, 1, 1)
            f_core = f_core + z_sq * self.epoch_quad.unsqueeze(0)
            return f_core

    class GeneralizedDualCycleMaTLoRALinear(nn.Module):
        def __init__(self, original_linear: nn.Linear, core_gen: DualCycleCoreGenerator, rank: int = 8, alpha: float = 16.0):
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base_out = self.base_layer(x)
            if self.current_theta is None or self.current_z is None:
                return base_out

            x_a = F.linear(x.float(), self.lora_A.float())
            F_core = self.core_gen(self.current_theta, self.current_z)
            x_af = torch.bmm(x_a, F_core.transpose(1, 2))
            lora_out = F.linear(x_af, self.lora_B.float()) * self.scaling
            return base_out + lora_out.to(dtype=x.dtype)

    # --- 3. Load Base Model ---
    model_id = "Qwen/Qwen2.5-3B"
    print(f"Loading Base model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    for p in model.parameters():
        p.requires_grad = False

    seam_projection = GatedResidualProjection(in_features=40, out_features=2048).to(device, dtype=torch.bfloat16)
    target_submodules = ["gate_proj", "up_proj", "down_proj"]
    mlp_modules = []
    core_gen = None

    if arm_type == "static_mlp":
        for name, module in model.named_modules():
            for target in target_submodules:
                if hasattr(module, target) and isinstance(getattr(module, target), nn.Linear):
                    orig = getattr(module, target)
                    adapted = StaticLoRALinear(orig, rank=rank, alpha=alpha)
                    setattr(module, target, adapted)
                    mlp_modules.append(adapted)
        print(f"Attached StaticLoRALinear to {len(mlp_modules)} MLP modules.")
    elif arm_type == "matlora":
        core_gen = DualCycleCoreGenerator(rank=rank, num_harmonics=2).to(device, dtype=torch.float32)
        for name, module in model.named_modules():
            for target in target_submodules:
                if hasattr(module, target) and isinstance(getattr(module, target), nn.Linear):
                    orig = getattr(module, target)
                    adapted = GeneralizedDualCycleMaTLoRALinear(orig, core_gen=core_gen, rank=rank, alpha=alpha)
                    setattr(module, target, adapted)
                    mlp_modules.append(adapted)
        print(f"Attached GeneralizedDualCycleMaTLoRALinear to {len(mlp_modules)} MLP modules.")

    # --- 4. Load Trained Weights from Persistent Volume ---
    volume.reload()
    prefix = f"stage03_3b_targeted_{arm_type}"
    weights_path = Path(f"/vol/{prefix}_5epochs_weights.pt")
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")

    print(f"Loading weights from {weights_path} ({weights_path.stat().st_size:,} bytes)...")
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)

    lora_state = ckpt.get("lora_state", {})
    model_state = model.state_dict()
    loaded_lora = 0
    for k, v in lora_state.items():
        if k in model_state:
            model_state[k].copy_(v.to(model_state[k].device, dtype=model_state[k].dtype))
            loaded_lora += 1
    print(f"Loaded {loaded_lora}/{len(lora_state)} LoRA tensors.")

    seam_state = ckpt.get("seam_projection", {})
    seam_projection.load_state_dict({k: v.to(device, dtype=torch.bfloat16) for k, v in seam_state.items()})
    print(f"Loaded Seam Projection. Final alpha: {seam_projection.alpha.item():.5f}")

    if core_gen is not None and "core_generator" in ckpt and ckpt["core_generator"] is not None:
        core_gen.load_state_dict({k: v.to(device, dtype=torch.float32) for k, v in ckpt["core_generator"].items()})
        print("Loaded CoreGenerator weights.")

    model.eval()
    seam_projection.eval()
    if core_gen is not None:
        core_gen.eval()

    # --- 5. Permutation Logic ---
    def generate_option_permutations(prompt_text: str, target_text: str):
        tgt = target_text.strip().lower()
        if tgt in ["yes", "no"]:
            p0 = prompt_text
            p1 = re.sub(r"Answer as Yes or No", "Answer as No or Yes", prompt_text, flags=re.IGNORECASE)
            if p1 == p0:
                p1 = re.sub(r"Yes\s*(or|/)\s*No", "No or Yes", prompt_text, flags=re.IGNORECASE)
            return [{"prompt": p0, "target": tgt}, {"prompt": p1, "target": tgt}]
        pattern = re.compile(
            r"^(.*Options:\s*)\(a\)\s*(.*?)\s*\(b\)\s*(.*?)\s*\(c\)\s*(.*?)\s*\(d\)\s*(.*?)$",
            re.DOTALL,
        )
        suffix = ""
        m_suffix = re.search(r"(\s*\n?\s*Answer:\s*)$", prompt_text, re.IGNORECASE)
        if m_suffix:
            suffix = m_suffix.group(1)
            base_prompt = prompt_text[: len(prompt_text) - len(suffix)]
        else:
            base_prompt = prompt_text
        m = pattern.search(base_prompt)
        if not m:
            return [{"prompt": prompt_text, "target": tgt}]
        prefix_str = m.group(1)
        opts = [m.group(2).strip(), m.group(3).strip(), m.group(4).strip(), m.group(5).strip()]
        letters = ["a", "b", "c", "d"]
        if tgt not in letters:
            return [{"prompt": prompt_text, "target": tgt}]
        t_idx = letters.index(tgt)
        perms = []
        for k in range(4):
            r_opts = [opts[(j + k) % 4] for j in range(4)]
            r_tgt = letters[(t_idx - k) % 4]
            perms.append({
                "prompt": f"{prefix_str}(a) {r_opts[0]} (b) {r_opts[1]} (c) {r_opts[2]} (d) {r_opts[3]}{suffix}",
                "target": r_tgt,
            })
        return perms

    # =========================================================================
    # PROTOCOL S0: HISTORICAL LETTER (Frozen legacy replay)
    # =========================================================================
    print("\n--- Running Protocol S0: HISTORICAL LETTER ---")
    tokenizer.padding_side = "left"

    def _extract_ids(words):
        ids = []
        for w in words:
            for prefix_ch in ["", " "]:
                enc = tokenizer.encode(prefix_ch + w, add_special_tokens=False)
                if enc and len(enc) == 1:
                    ids.append(enc[0])
        return list(dict.fromkeys(ids))

    s0_yes_ids = _extract_ids(["yes", "Yes", "YES"])
    s0_no_ids = _extract_ids(["no", "No", "NO"])
    s0_choice_ids = {k: _extract_ids([k, k.upper()]) for k in ["a", "b", "c", "d"]}

    def score_s0(lg, gold_str: str) -> Tuple[str, str, int]:
        t_norm = gold_str.strip().lower()
        if t_norm in ["yes", "no"]:
            s_yes = max(lg[idx].item() for idx in s0_yes_ids) if s0_yes_ids else -1e9
            s_no = max(lg[idx].item() for idx in s0_no_ids) if s0_no_ids else -1e9
            pred = "yes" if s_yes >= s_no else "no"
        else:
            scores = {k: (max(lg[idx].item() for idx in c_ids) if c_ids else -1e9) for k, c_ids in s0_choice_ids.items()}
            pred = max(scores, key=scores.get)
        return pred, t_norm, int(pred == t_norm)

    s0_case_results = {}
    s0_can_correct = 0
    s0_strict_correct = 0
    s0_deb_sum = 0.0

    eval_batch_size = 4
    for b_start in range(0, len(eval_cases), eval_batch_size):
        b_cases = eval_cases[b_start : b_start + eval_batch_size]
        all_prompts, all_targets, all_years, case_perm_slices = [], [], [], []

        for case in b_cases:
            yr = case.get("year", 2000)
            perms = generate_option_permutations(case["input"], case["target"])
            s_idx = len(all_prompts)
            for p in perms:
                all_prompts.append(p["prompt"])
                all_targets.append(p["target"])
                all_years.append(yr)
            e_idx = len(all_prompts)
            case_perm_slices.append((case, (s_idx, e_idx)))

        formatted = [f"Question: {p.strip()}\nAnswer:" if not p.strip().endswith("Answer:") else p.strip() for p in all_prompts]
        inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        v40_eval = torch.tensor([compute_fixed_fone_canchi_40d(y) for y in all_years], dtype=torch.float32, device=device)

        if arm_type == "matlora":
            th_list, z_list = [], []
            for y in all_years:
                th_i, z_i = compute_theta_z(y)
                th_list.append(th_i)
                z_list.append(z_i)
            th_eval = torch.tensor(th_list, dtype=torch.float32, device=device)
            z_eval = torch.tensor(z_list, dtype=torch.float32, device=device)
            for mod in mlp_modules:
                mod.current_theta = th_eval
                mod.current_z = z_eval

        with torch.no_grad():
            inputs_embeds = inject_seam_prefill_safe(model, seam_projection, inputs["input_ids"], v40_eval, token_idx=14)
            transformer_out = model.model(inputs_embeds=inputs_embeds, attention_mask=inputs["attention_mask"])
            last_hidden_token = transformer_out.last_hidden_state[:, -1, :]
            logits = model.lm_head(last_hidden_token)

        b_results = []
        for idx, (p_text, gold) in enumerate(zip(all_prompts, all_targets)):
            pred, t_norm, is_corr = score_s0(logits[idx], gold)
            b_results.append({"pred": pred, "gold": t_norm, "is_correct": is_corr})

        for case, (s_idx, e_idx) in case_perm_slices:
            cid = case.get("example_id")
            outcomes = [r["is_correct"] for r in b_results[s_idx:e_idx]]
            is_can = outcomes[0]
            d_score = float(np.mean(outcomes))
            is_st = all(outcomes)
            if is_can:
                s0_can_correct += 1
            if is_st:
                s0_strict_correct += 1
            s0_deb_sum += d_score
            s0_case_results[cid] = {
                "canonical_correct": bool(is_can),
                "strict_correct": bool(is_st),
                "debiased_score": d_score,
                "outcomes": outcomes,
            }

    print(f"S0 Replay Results: Canonical = {s0_can_correct}/{len(eval_cases)} ({s0_can_correct/len(eval_cases)*100:.2f}%), Strict = {s0_strict_correct}/{len(eval_cases)} ({s0_strict_correct/len(eval_cases)*100:.2f}%)")

    # =========================================================================
    # PROTOCOL S1: AUDITED (batch_size=1, dynamic year token, clean label IDs)
    # =========================================================================
    print("\n--- Running Protocol S1: AUDITED SCORER ---")

    def locate_year_token_indices(tok, prompt_str: str, yr_int: int) -> int:
        yr_str = str(yr_int)
        enc = tok(prompt_str, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
        c_idx = prompt_str.find(yr_str)
        if c_idx == -1:
            return 14
        yr_end = c_idx + len(yr_str)
        matching = []
        for t_i, (st, en) in enumerate(offsets):
            if max(st, c_idx) < min(en, yr_end):
                matching.append(t_i)
        return matching[0] if matching else 14

    # Clean target token IDs (single space continuation)
    # ' A': 362, ' B': 425, ' C': 356, ' D': 422
    # ' Yes': 7414, ' No': 2308
    tid_A = tokenizer.encode(" A", add_special_tokens=False)[0]
    tid_B = tokenizer.encode(" B", add_special_tokens=False)[0]
    tid_C = tokenizer.encode(" C", add_special_tokens=False)[0]
    tid_D = tokenizer.encode(" D", add_special_tokens=False)[0]
    tid_Yes = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    tid_No = tokenizer.encode(" No", add_special_tokens=False)[0]

    s1_choice_map = {"a": tid_A, "b": tid_B, "c": tid_C, "d": tid_D}

    s1_case_results = {}
    s1_can_correct = 0
    s1_strict_correct = 0
    s1_deb_sum = 0.0

    for case in eval_cases:
        cid = case.get("example_id")
        yr = case.get("year", 2000)
        perms = generate_option_permutations(case["input"], case["target"])
        perm_outcomes = []
        perm_preds = []

        for p in perms:
            p_text = p["prompt"]
            gold = p["target"].strip().lower()
            formatted_prompt = f"Question: {p_text.strip()}\nAnswer:" if not p_text.strip().endswith("Answer:") else p_text.strip()

            inp_single = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)
            v40_single = torch.tensor([compute_fixed_fone_canchi_40d(yr)], dtype=torch.float32, device=device)

            yr_token_idx = locate_year_token_indices(tokenizer, formatted_prompt, yr)

            if arm_type == "matlora":
                th_val, z_val = compute_theta_z(yr)
                th_t = torch.tensor([th_val], dtype=torch.float32, device=device)
                z_t = torch.tensor([z_val], dtype=torch.float32, device=device)
                for mod in mlp_modules:
                    mod.current_theta = th_t
                    mod.current_z = z_t

            with torch.no_grad():
                inputs_embeds = inject_seam_prefill_safe(model, seam_projection, inp_single["input_ids"], v40_single, token_idx=yr_token_idx)
                transformer_out = model.model(inputs_embeds=inputs_embeds)
                last_hidden_single = transformer_out.last_hidden_state[:, -1, :]
                logits_single = model.lm_head(last_hidden_single)[0]

            if gold in ["yes", "no"]:
                score_yes = logits_single[tid_Yes].item()
                score_no = logits_single[tid_No].item()
                pred = "yes" if score_yes >= score_no else "no"
            else:
                scores = {k: logits_single[tid].item() for k, tid in s1_choice_map.items()}
                pred = max(scores, key=scores.get)

            is_corr = int(pred == gold)
            perm_outcomes.append(is_corr)
            perm_preds.append(pred)

        is_can = perm_outcomes[0]
        d_score = float(np.mean(perm_outcomes))
        is_st = all(perm_outcomes)

        if is_can:
            s1_can_correct += 1
        if is_st:
            s1_strict_correct += 1
        s1_deb_sum += d_score

        s1_case_results[cid] = {
            "canonical_correct": bool(is_can),
            "strict_correct": bool(is_st),
            "debiased_score": d_score,
            "outcomes": perm_outcomes,
            "predictions": perm_preds,
        }

    print(f"S1 Audited Results: Canonical = {s1_can_correct}/{len(eval_cases)} ({s1_can_correct/len(eval_cases)*100:.2f}%), Strict = {s1_strict_correct}/{len(eval_cases)} ({s1_strict_correct/len(eval_cases)*100:.2f}%)")

    total_time = time.time() - t_start
    print(f"\nArm [{arm_type.upper()}] Replay & Audit finished in {total_time:.2f} seconds.")

    return {
        "arm_type": arm_type,
        "runtime_seconds": round(total_time, 2),
        "total_samples": len(eval_cases),
        "s0_results": {
            "canonical_correct": s0_can_correct,
            "canonical_pct": round(s0_can_correct / len(eval_cases) * 100, 2),
            "strict_correct": s0_strict_correct,
            "strict_pct": round(s0_strict_correct / len(eval_cases) * 100, 2),
            "debiased_mean_pct": round(s0_deb_sum / len(eval_cases) * 100, 2),
            "cases": s0_case_results,
        },
        "s1_results": {
            "canonical_correct": s1_can_correct,
            "canonical_pct": round(s1_can_correct / len(eval_cases) * 100, 2),
            "strict_correct": s1_strict_correct,
            "strict_pct": round(s1_strict_correct / len(eval_cases) * 100, 2),
            "debiased_mean_pct": round(s1_deb_sum / len(eval_cases) * 100, 2),
            "cases": s1_case_results,
        },
        "delta_s1_minus_s0": {
            "canonical_diff": s1_can_correct - s0_can_correct,
            "canonical_diff_pct": round((s1_can_correct - s0_can_correct) / len(eval_cases) * 100, 2),
            "strict_diff": s1_strict_correct - s0_strict_correct,
            "strict_diff_pct": round((s1_strict_correct - s0_strict_correct) / len(eval_cases) * 100, 2),
            "debiased_mean_diff_pct": round((s1_deb_sum - s0_deb_sum) / len(eval_cases) * 100, 2),
        },
    }


# =========================================================================
# LOCAL ENTRYPOINT
# =========================================================================

@app.local_entrypoint()
def main(
    manifest_path: str = "Reports/6-11Report/_POST_FORENSIC_OPEN_GATES_20261105/REPLAY_SAMPLE_MANIFEST.json",
    eval_benchmark_path: str = "data/tempun_full_clean/eval_5292_clean.jsonl.gz",
    out_dir: str = "Reports/6-11Report/_PHASE1_G1_BOTTLENECK_20261105",
):
    print("=== Launching GPU Replay on Modal (A10G) for Arm 4 & Arm 5 ===")
    t0 = time.time()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    samples = manifest["samples"]

    eval_cases_all = []
    with gzip.open(eval_benchmark_path, "rt", encoding="utf-8") as f:
        for line in f:
            eval_cases_all.append(json.loads(line))

    # Build 128 full case objects
    full_sample_records = []
    for s in samples:
        idx = s["original_eval_index"]
        ev = eval_cases_all[idx]
        full_sample_records.append(ev)

    samples_json_str = json.dumps(full_sample_records)
    print(f"Serialized {len(full_sample_records)} samples ({len(samples_json_str):,} chars).")

    # Run Arm 4 (static_mlp) and Arm 5 (matlora)
    print("\n--- Dispatching Arm 4 (Static MLP LoRA) to Modal A10G ---")
    arm4_res = run_arm_replay_and_audit.remote("static_mlp", samples_json_str)

    print("\n--- Dispatching Arm 5 (MaT-LoRA) to Modal A10G ---")
    arm5_res = run_arm_replay_and_audit.remote("matlora", samples_json_str)

    total_wall_time = time.time() - t0
    # A10G price: ~$1.10/hr = $0.000306/sec
    gpu_seconds_total = arm4_res["runtime_seconds"] + arm5_res["runtime_seconds"]
    gross_cost_usd = round(gpu_seconds_total * (1.10 / 3600.0), 4)

    summary_output = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "experiment_name": "GPU_REPLAY_128_S0_VS_S1_EVALUATION",
        "timestamp": datetime.now().isoformat(),
        "gross_cost_usd": gross_cost_usd,
        "budget_cap_usd": 0.35,
        "budget_respected": gross_cost_usd <= 0.35,
        "total_wall_time_seconds": round(total_wall_time, 2),
        "total_gpu_compute_seconds": gpu_seconds_total,
        "historical_anchors_128": {
            "arm4_canonical": 37,
            "arm4_strict": 9,
            "arm5_canonical": 43,
            "arm5_strict": 13,
        },
        "arm4_results": arm4_res,
        "arm5_results": arm5_res,
    }

    out_file = Path(out_dir) / "GPU_REPLAY_128_S0_S1_RESULTS.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved Replay & Audit Results to {out_file}")
    print(f"Total Gross Spend: ${gross_cost_usd:.4f} USD (Budget Cap: $0.35 USD).")
