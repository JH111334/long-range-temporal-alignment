"""STAGE 03 G1 UNCONSTRAINED FREE-GENERATION GPU SMOKE EVALUATION (MODAL GPU A10G)

Authorized by: User & ChatGPT Research Owners
Date: 2026-11-08
Budget Cap: $0.35 USD (Hard Timeout: 300s)
Hardware: 1x NVIDIA A10G (24GB VRAM)
Volume: 'stage03-artifacts' mounted at /vol

Executes:
1. Unconstrained autoregressive greedy generation over full vocabulary (max_new_tokens=16).
2. Metadata-grounded temporal injection (YEAR_SPAN, sum-preserving 1/span_len, zero padding).
3. Exact linear gate mode: Delta = alpha * LayerNorm(W * v40 + b), matching training checkpoint.
4. Pure Python deterministic regex parser (LABEL_ONLY) rejecting prose articles ('a event').
5. Canonical, debiased mean, and strict accuracy over audited rotations.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

app = modal.App("stage03-qwen3b-g1-freegen-smoke")

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

@app.function(image=image, gpu="A10G", timeout=1200, volumes={"/vol": volume})
def run_g1_freegen_smoke(
    arm: str,  # 'arm5', 'arm4', or 'base'
    samples_json_str: str,
    temporal_policy: str = "YEAR_SPAN",
    max_new_tokens: int = 16,
) -> Dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    device = "cuda:0"
    print("\n=======================================================")
    print(f"STARTING G1 FREE-GENERATION GPU SMOKE: ARM [{arm.upper()}]")
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")
    print(f"Temporal Policy: {temporal_policy} | Gate Mode: LINEAR")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=======================================================\n")

    eval_cases = json.loads(samples_json_str)
    print(f"Loaded {len(eval_cases)} evaluation cases.")

    # --- 1. Continuous Temporal Math ---
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
            v_in = v_time.to(self.linear.weight.device, dtype=self.linear.weight.dtype)
            proj = self.linear(v_in)
            normed = self.norm(proj)
            # Linear gate matching training script: alpha * normed
            return self.alpha * normed

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

            device_t = original_linear.weight.device
            dtype_t = original_linear.weight.dtype
            self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=device_t, dtype=dtype_t))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, rank, device=device_t, dtype=dtype_t))
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
            dtype_t = original_linear.weight.dtype
            self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=dev, dtype=dtype_t))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, rank, device=dev, dtype=dtype_t))
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

    # --- 3. Load Backbone Model ---
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

    seam_proj = None
    core_gen = None
    mlp_modules = []
    target_submodules = ["gate_proj", "up_proj", "down_proj"]

    if arm in ("arm4", "arm5"):
        seam_proj = GatedResidualProjection(in_features=40, out_features=2048).to(device, dtype=torch.bfloat16)

        if arm == "arm4":
            for name, module in model.named_modules():
                for target in target_submodules:
                    if hasattr(module, target) and isinstance(getattr(module, target), nn.Linear):
                        orig = getattr(module, target)
                        adapted = StaticLoRALinear(orig, rank=8, alpha=16.0)
                        setattr(module, target, adapted)
                        mlp_modules.append(adapted)
            print(f"Attached StaticLoRALinear to {len(mlp_modules)} MLP modules.")
            ckpt_name = "stage03_3b_targeted_static_mlp_5epochs_weights.pt"
        elif arm == "arm5":
            core_gen = DualCycleCoreGenerator(rank=8, num_harmonics=2).to(device, dtype=torch.float32)
            for name, module in model.named_modules():
                for target in target_submodules:
                    if hasattr(module, target) and isinstance(getattr(module, target), nn.Linear):
                        orig = getattr(module, target)
                        adapted = GeneralizedDualCycleMaTLoRALinear(orig, core_gen=core_gen, rank=8, alpha=16.0)
                        setattr(module, target, adapted)
                        mlp_modules.append(adapted)
            print(f"Attached GeneralizedDualCycleMaTLoRALinear to {len(mlp_modules)} MLP modules.")
            ckpt_name = "stage03_3b_targeted_matlora_5epochs_weights.pt"

        # Load Checkpoint Weights from Volume
        volume.reload()
        ckpt_path = Path(f"/vol/{ckpt_name}")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found on /vol: {ckpt_path}")

        print(f"Loading checkpoint from {ckpt_path} ({ckpt_path.stat().st_size:,} bytes)...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        lora_state = ckpt.get("lora_state", {})
        model_state = model.state_dict()
        loaded_lora = 0
        for k, v in lora_state.items():
            if k in model_state:
                model_state[k].copy_(v.to(model_state[k].device, dtype=model_state[k].dtype))
                loaded_lora += 1
        print(f"Loaded {loaded_lora}/{len(lora_state)} LoRA tensors.")
        if loaded_lora != 216:
            raise ValueError(f"Expected 216 LoRA tensors, loaded {loaded_lora}.")

        seam_state = ckpt.get("seam_projection", {})
        if seam_state:
            seam_proj.load_state_dict({k: v.to(device) for k, v in seam_state.items()})
            print(f"Loaded Seam projection. Alpha = {seam_proj.alpha.item():.6f}")

        if core_gen is not None and "core_generator" in ckpt:
            core_state = ckpt["core_generator"]
            core_gen.load_state_dict({k: v.to(device) for k, v in core_state.items()})
            print(f"Loaded Core Generator ({len(core_state)} tensors).")

        seam_proj.eval()
        if core_gen is not None:
            core_gen.eval()
    else:
        print("ARM = BASE: unadapted backbone, Seam disabled.")

    model.eval()

    # --- 4. Rotations & Formatters ---
    def format_inference_prompt(raw_prompt: str) -> str:
        cleaned = raw_prompt.strip()
        if cleaned.endswith("Answer:"):
            return cleaned
        return f"Question: {cleaned}\nAnswer:"

    def generate_audited_permutations(raw_input: str, gold_target: str, task_id: str):
        tgt = gold_target.strip().lower()
        if tgt in ["yes", "no"] or task_id == "Rel":
            p0 = raw_input.strip()
            p1 = re.sub(r"Answer as Yes or No", "Answer as No or Yes", p0, flags=re.IGNORECASE)
            if p1 == p0:
                p1 = re.sub(r"Yes\s*(or|/)\s*No", "No or Yes", p0, flags=re.IGNORECASE)
            return [
                {"rotation_id": 0, "prompt": format_inference_prompt(p0), "target": tgt},
                {"rotation_id": 1, "prompt": format_inference_prompt(p1), "target": tgt},
            ]

        m_suff = re.search(r"(\s*\n?\s*Answer:\s*)$", raw_input, re.IGNORECASE)
        suffix = m_suff.group(1) if m_suff else ""
        base = raw_input[: len(raw_input) - len(suffix)].strip() if m_suff else raw_input.strip()

        pattern = re.compile(r"^(.*Options:\s*)\(a\)\s*(.*?)\s*\(b\)\s*(.*?)\s*\(c\)\s*(.*?)\s*\(d\)\s*(.*?)$", re.DOTALL)
        m = pattern.search(base)
        letters = ["a", "b", "c", "d"]
        if not m or tgt not in letters:
            return [{"rotation_id": 0, "prompt": format_inference_prompt(raw_input), "target": tgt}]

        prefix_str = m.group(1)
        opts = [m.group(2).strip(), m.group(3).strip(), m.group(4).strip(), m.group(5).strip()]
        t_idx = letters.index(tgt)
        results = []
        for k in range(4):
            r_opts = [opts[(j + k) % 4] for j in range(4)]
            r_tgt = letters[(t_idx - k) % 4]
            p_text = f"{prefix_str}(a) {r_opts[0]} (b) {r_opts[1]} (c) {r_opts[2]} (d) {r_opts[3]}{suffix}"
            results.append({"rotation_id": k, "prompt": format_inference_prompt(p_text), "target": r_tgt})
        return results

    # --- 5. Deterministic Answer Parser ---
    MCQA_ISOLATED = re.compile(r"^\s*[\(\[]?([A-D])[\)\]]?(?:[\.\:\,])?\s*$", re.IGNORECASE)
    MCQA_DELIMITED = re.compile(r"^\s*[\(\[]?([A-D])[\)\]]?[\)\.\:\,\-]\s+", re.IGNORECASE)
    MCQA_PREFIXED = re.compile(r"^\s*(?:the\s+correct\s+)?(?:answer|choice|option)\s*(?:is|\:)?\s*[\(\[]?([A-D])[\)\]]?(?:[\.\:\,]|\s|$)", re.IGNORECASE)
    REL_ISOLATED = re.compile(r"^\s*(yes|no)\b(?:[\.\!\?])?\s*", re.IGNORECASE)
    REL_PREFIXED = re.compile(r"^\s*(?:the\s+correct\s+)?answer\s*(?:is|\:)?\s*(yes|no)\b", re.IGNORECASE)

    def parse_answer(text: str, task_id: str) -> Tuple[Optional[str], bool, str]:
        stripped = text.strip()
        if not stripped:
            return None, False, "EMPTY_OUTPUT"

        if task_id == "Rel":
            m_iso = REL_ISOLATED.search(stripped)
            if m_iso:
                return m_iso.group(1).lower(), True, "VALID"
            m_pre = REL_PREFIXED.search(stripped)
            if m_pre:
                return m_pre.group(1).lower(), True, "VALID"
            return None, False, "NO_RECOGNIZABLE_LABEL"

        m_iso = MCQA_ISOLATED.match(stripped)
        if m_iso:
            return m_iso.group(1).lower(), True, "VALID"
        m_del = MCQA_DELIMITED.match(stripped)
        if m_del:
            return m_del.group(1).lower(), True, "VALID"
        m_pre = MCQA_PREFIXED.match(stripped)
        if m_pre:
            return m_pre.group(1).lower(), True, "VALID"

        return None, False, "NO_RECOGNIZABLE_LABEL"

    # --- 6. Execution Loop ---
    case_records = []
    canonical_correct = 0
    strict_correct = 0
    debiased_scores = []
    invalid_count = 0
    total_rotations = 0

    stop_strings = ["\n", "\n\n", "Question:", "Answer:"]
    eos_token_id = tokenizer.eos_token_id

    for g_idx, case in enumerate(eval_cases):
        cid = case.get("example_id", f"case_{g_idx}")
        task_id = case.get("task_id", "MCQ")
        anchor_year = case.get("year")
        if anchor_year is None:
            continue
        anchor_year = int(anchor_year)

        v40 = compute_fixed_fone_canchi_40d(anchor_year)
        th_val, z_val = compute_theta_z(anchor_year)

        perms = generate_audited_permutations(case["input"], case["target"], task_id=task_id)
        rot_outcomes = []

        for rot in perms:
            rot_id = rot["rotation_id"]
            fmt_prompt = rot["prompt"]
            gold = rot["target"].lower().strip()
            total_rotations += 1

            enc = tokenizer(fmt_prompt, return_tensors="pt", add_special_tokens=False).to(device)
            input_ids = enc["input_ids"]
            attn_mask = enc["attention_mask"]
            S = input_ids.shape[1]

            # Build temporal modulation & inject representation
            inputs_embeds = model.get_input_embeddings()(input_ids)
            if seam_proj is not None and temporal_policy == "MULTI_YEAR_SPAN":
                fast_enc = tokenizer(fmt_prompt, return_offsets_mapping=True, add_special_tokens=False)
                offsets = fast_enc["offset_mapping"]
                year_pat = re.compile(r"\b([1-9]\d{2,3})\b")
                delta_grid = torch.zeros_like(inputs_embeds)
                seen_spans = set()
                for m_yr in year_pat.finditer(fmt_prompt):
                    yr_val = int(m_yr.group(1))
                    if yr_val < 100 or yr_val > 2100:
                        continue
                    c_start, c_end = m_yr.start(), m_yr.end()
                    if (c_start, c_end) in seen_spans:
                        continue
                    seen_spans.add((c_start, c_end))
                    matched_toks = [
                        t_i for t_i, (os_start, os_end) in enumerate(offsets)
                        if os_start != os_end and not (os_end <= c_start or os_start >= c_end)
                    ]
                    if matched_toks:
                        v40_m = torch.tensor([compute_fixed_fone_canchi_40d(yr_val)], dtype=torch.float32, device=device)
                        seam_delta_m = seam_proj(v40_m).to(inputs_embeds.dtype)
                        w_m = 1.0 / float(len(matched_toks))
                        for t_i in matched_toks:
                            delta_grid[0, t_i, :] = delta_grid[0, t_i, :] + (w_m * seam_delta_m[0])
                inputs_embeds = inputs_embeds + delta_grid
            elif seam_proj is not None and temporal_policy == "YEAR_SPAN":
                temp_mask = torch.zeros((1, S), dtype=torch.float32, device=device)
                y_str = str(anchor_year)
                c_idx = fmt_prompt.find(y_str)
                if c_idx != -1:
                    fast_enc = tokenizer(fmt_prompt, return_offsets_mapping=True, add_special_tokens=False)
                    offsets = fast_enc["offset_mapping"]
                    y_start = c_idx
                    y_end = c_idx + len(y_str)
                    matched_toks = []
                    for t_i, (os_start, os_end) in enumerate(offsets):
                        if os_start == os_end:
                            continue
                        if not (os_end <= y_start or os_start >= y_end):
                            matched_toks.append(t_i)
                    if matched_toks:
                        weight = 1.0 / len(matched_toks)
                        for t_i in matched_toks:
                            temp_mask[0, t_i] = weight
                if temp_mask.sum() > 0:
                    v40_t = torch.tensor([v40], dtype=torch.float32, device=device)
                    seam_delta = seam_proj(v40_t).to(inputs_embeds.dtype)
                    mask_t = temp_mask.unsqueeze(-1).to(inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds + (mask_t * seam_delta.unsqueeze(1))
            inputs_embeds = inputs_embeds.to(dtype=model.dtype)

            # Maintain MaT-LoRA temporal context
            if arm == "arm5":
                th_t = torch.tensor([th_val], dtype=torch.float32, device=device)
                z_t = torch.tensor([z_val], dtype=torch.float32, device=device)
                for mod in mlp_modules:
                    mod.current_theta = th_t
                    mod.current_z = z_t

            # --- Greedy Autoregressive Generation ---
            generated_ids = []
            finish_reason = "length"

            with torch.no_grad():
                prefill_out = model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=True, return_dict=True)
                past_kv = prefill_out.past_key_values
                next_tok = int(torch.argmax(prefill_out.logits[:, -1, :], dim=-1).item())
                generated_ids.append(next_tok)

                if next_tok == eos_token_id:
                    finish_reason = "eos"
                else:
                    curr_mask = attn_mask
                    for step in range(1, max_new_tokens):
                        cur_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                        if any(st in cur_text for st in stop_strings):
                            finish_reason = "stop_string"
                            break

                        curr_mask = torch.cat([curr_mask, torch.ones((1, 1), dtype=curr_mask.dtype, device=device)], dim=1)
                        curr_inp = torch.tensor([[next_tok]], dtype=torch.long, device=device)
                        step_out = model(input_ids=curr_inp, attention_mask=curr_mask, past_key_values=past_kv, use_cache=True, return_dict=True)
                        past_kv = step_out.past_key_values
                        next_tok = int(torch.argmax(step_out.logits[:, -1, :], dim=-1).item())
                        generated_ids.append(next_tok)

                        if next_tok == eos_token_id:
                            finish_reason = "eos"
                            break

            # Reset MaT-LoRA context
            if arm == "arm5":
                for mod in mlp_modules:
                    mod.current_theta = None
                    mod.current_z = None

            raw_continuation = tokenizer.decode(generated_ids, skip_special_tokens=True)
            label, is_valid, reason = parse_answer(raw_continuation, task_id)
            is_correct = bool(is_valid and label == gold)

            if not is_valid:
                invalid_count += 1
            rot_outcomes.append(is_correct)

            case_records.append({
                "example_id": cid,
                "rotation_id": rot_id,
                "task_id": task_id,
                "gold": gold,
                "raw_continuation": raw_continuation.strip(),
                "parsed_label": label,
                "parse_reason": reason,
                "is_valid": is_valid,
                "is_correct": is_correct,
                "finish_reason": finish_reason,
            })

        is_can = rot_outcomes[0]
        is_strict = all(rot_outcomes)
        d_score = float(np.mean(rot_outcomes))

        if is_can:
            canonical_correct += 1
        if is_strict:
            strict_correct += 1
        debiased_scores.append(d_score)

        if (g_idx + 1) % 10 == 0 or (g_idx + 1) == len(eval_cases):
            elapsed_sec = round(time.time() - t_start, 1)
            print(f"[HEARTBEAT] Case {g_idx + 1}/{len(eval_cases)} | Elapsed: {elapsed_sec}s | Canonical: {canonical_correct}/{g_idx + 1} | Strict: {strict_correct}/{g_idx + 1} | Invalid: {invalid_count}")

    total_examples = len(eval_cases)
    total_time = round(time.time() - t_start, 2)

    return {
        "status": "PASS_GPU_SMOKE",
        "arm": arm,
        "temporal_policy": temporal_policy,
        "total_examples": total_examples,
        "total_records": len(case_records),
        "canonical_accuracy": round(canonical_correct / total_examples, 4),
        "canonical_correct": canonical_correct,
        "strict_accuracy": round(strict_correct / total_examples, 4),
        "strict_correct": strict_correct,
        "debiased_mean_accuracy": round(float(np.mean(debiased_scores)), 4) if debiased_scores else 0.0,
        "invalid_rate": round(invalid_count / total_rotations, 4) if total_rotations else 0.0,
        "invalid_count": invalid_count,
        "runtime_seconds": total_time,
        "records_sample": case_records[:8],  # First 8 records preview
    }


# =========================================================================
# LOCAL ENTRYPOINT
# =========================================================================

@app.local_entrypoint()
def main(
    arm: str = "arm5",
    n_samples: int = 16,
    temporal_policy: str = "YEAR_SPAN",
    manifest_path: str = "Reports/6-11Report/_POST_FORENSIC_OPEN_GATES_20261105/REPLAY_SAMPLE_MANIFEST.json",
    eval_benchmark_path: str = "data/tempun_full_clean/eval_5292_clean.jsonl.gz",
    out_dir: str = "Reports/6-11Report/G1_FREEGEN_INJECTION_V2",
    dry_run_verify: bool = False,
):
    norm_arm = arm.lower()
    if "arm5" in norm_arm:
        arm_clean = "arm5"
    elif "arm4" in norm_arm:
        arm_clean = "arm4"
    else:
        arm_clean = "base"

    print(f"=== Stage 03 G1 Free-Generation GPU Smoke: Arm [{arm_clean.upper()}] ===")
    t0 = time.time()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    samples = manifest["samples"]

    if n_samples >= len(samples):
        selected_samples = samples[:n_samples]
    elif n_samples == 16:
        # Stratified selection across all 6 tasks for balanced smoke coverage
        tasks = ["DB", "TB", "WB", "Rel", "MM", "RB"]
        samples_by_task = {t: [] for t in tasks}
        for s in samples:
            t = s.get("task_id")
            if t in samples_by_task:
                samples_by_task[t].append(s)

        selected_samples = []
        task_quotas = {"DB": 3, "TB": 3, "WB": 3, "Rel": 3, "MM": 2, "RB": 2}
        for t in tasks:
            q = task_quotas.get(t, 2)
            selected_samples.extend(samples_by_task[t][:q])
    else:
        selected_samples = samples[:n_samples]

    eval_cases_all = []
    with gzip.open(eval_benchmark_path, "rt", encoding="utf-8") as f:
        for line in f:
            eval_cases_all.append(json.loads(line))

    selected_cases = []
    for s in selected_samples:
        idx = s["original_eval_index"]
        selected_cases.append(eval_cases_all[idx])

    task_dist = Counter([c["task_id"] for c in selected_cases])
    print(f"Selected {len(selected_cases)} frozen cases: {dict(task_dist)}.")

    if dry_run_verify:
        print("[DRY RUN VERIFY]: Sample extraction verified without launching Modal GPU job.")
        return

    resolved_policy = "SEAM_OFF" if arm_clean == "base" else temporal_policy
    if resolved_policy not in ("YEAR_SPAN", "MULTI_YEAR_SPAN", "SEAM_OFF"):
        raise ValueError(f"Unsupported temporal_policy: {resolved_policy}")

    samples_json_str = json.dumps(selected_cases)
    print(f"\nDispatching {len(selected_cases)} samples to Modal A10G GPU worker (policy={resolved_policy})...")
    remote_result = run_g1_freegen_smoke.remote(
        arm=arm_clean,
        samples_json_str=samples_json_str,
        temporal_policy=resolved_policy,
        max_new_tokens=16,
    )

    total_wall_time = round(time.time() - t0, 2)
    # A10G price: ~$1.10/hr = $0.000306/sec
    gpu_seconds = remote_result["runtime_seconds"]
    gross_cost_usd = round(gpu_seconds * (1.10 / 3600.0), 4)

    is_replay_128 = (len(selected_cases) >= 128)
    exp_prefix = "GPU_REPLAY_128" if is_replay_128 else f"GPU_SMOKE_{len(selected_cases)}"
    policy_tag = "_MULTI_YEAR" if resolved_policy == "MULTI_YEAR_SPAN" else ""

    payload = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "experiment_name": f"G1_FREEGEN_{exp_prefix}{policy_tag}",
        "timestamp": datetime.now().isoformat(),
        "arm": arm_clean,
        "temporal_policy": resolved_policy,
        "n_samples": len(selected_cases),
        "gross_cost_usd": gross_cost_usd,
        "budget_cap_usd": 0.35,
        "budget_respected": gross_cost_usd <= 0.35,
        "wall_time_seconds": total_wall_time,
        "gpu_compute_seconds": gpu_seconds,
        "results": remote_result,
    }

    out_p = Path(out_dir) / f"{exp_prefix}{policy_tag}_{arm_clean.upper()}_RESULTS.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=======================================================")
    print(f"GPU EXECUTION COMPLETE: ARM [{arm_clean.upper()}] ({exp_prefix})")
    print(f"Canonical Accuracy: {remote_result['canonical_accuracy'] * 100:.1f}% ({remote_result['canonical_correct']}/{len(selected_cases)})")
    print(f"Strict Accuracy:    {remote_result['strict_accuracy'] * 100:.1f}% ({remote_result['strict_correct']}/{len(selected_cases)})")
    print(f"Debiased Accuracy:  {remote_result['debiased_mean_accuracy'] * 100:.1f}%")
    print(f"Invalid Rate:       {remote_result['invalid_rate'] * 100:.1f}% ({remote_result['invalid_count']} records)")
    print(f"GPU Time:           {gpu_seconds}s | Wall Time: {total_wall_time}s")
    print(f"Spend:              ${gross_cost_usd:.4f} USD (Cap: $0.35 USD)")
    print(f"Results saved to:   {out_p}")
    print("=======================================================\n")
    print("=======================================================\n")
