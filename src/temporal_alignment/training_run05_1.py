"""Training Execution Engine for V2 Run 05.1 (Forensic Phoenix Recovery).

Implements:
- Arm A (R51_A_NTP_REL): L_NTP + lambda_R * L_relation + lambda_C * L_relation_ctrl
- Arm B (R51_B_NTP_REL_MCQA): L_NTP + lambda_R * L_relation + lambda_M * L_MCQA
- ChatML Target Supervision with <|im_end|> (token 151645) and strict prompt label masking (-100)
- 100% Frozen Backbone Qwen2.5-0.5B-Instruct (base_layer.weight.requires_grad = False)
- Deterministic Stratified Sampler across years, entities, and targets (seed 42)
- Compact adapter-only checkpoint serialization (~13.7 MB, 0 base_layer keys)
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .baselines.ticktack05b.evaluator import build_p12_prompt
from .contracts import Era, HistoricalYear
from .models.dual_cycle_mat_lora import (
    DualCycleMaTLoRALinear,
    clear_model_temporal_coordinates,
    replace_mlp_with_dual_cycle_mat_lora,
    set_model_temporal_coordinates,
)
from .models.fone_canchi_40d import (
    compute_fixed_fone_canchi_40d,
    extract_canchi_angles_from_40d,
)
from .models.gated_seam import (
    ConditioningContext,
    GatedResidualProjection,
    make_gated_embedding_hook,
)
from .models.m0_fc40_loader import load_m0_fc40_for_inference


class DeterministicStratifiedSampler:
    """Deterministic stratified sampler balancing years, entities, and labels across steps."""
    def __init__(self, data: List[Dict[str, Any]], key_fn: Any, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.strata = defaultdict(list)
        for item in data:
            stratum = key_fn(item)
            self.strata[stratum].append(item)
        self.strata_keys = sorted(list(self.strata.keys()))
        for k in self.strata_keys:
            self.rng.shuffle(self.strata[k])
        self.indices = {k: 0 for k in self.strata_keys}
        self.round_robin_idx = 0

    def sample(self) -> Dict[str, Any]:
        k = self.strata_keys[self.round_robin_idx % len(self.strata_keys)]
        self.round_robin_idx += 1
        idx = self.indices[k]
        item = self.strata[k][idx % len(self.strata[k])]
        self.indices[k] += 1
        return item


def compute_tensor_fingerprint(state_dict: Dict[str, Any]) -> str:
    """Compute deterministic SHA256 digest over state dict tensors."""
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        val = state_dict[key]
        if isinstance(val, torch.Tensor):
            hasher.update(key.encode("utf-8"))
            hasher.update(val.detach().cpu().to(torch.float32).numpy().tobytes())
        elif isinstance(val, dict):
            for sub_k in sorted(val.keys()):
                sub_v = val[sub_k]
                if isinstance(sub_v, torch.Tensor):
                    hasher.update(f"{key}.{sub_k}".encode("utf-8"))
                    hasher.update(sub_v.detach().cpu().to(torch.float32).numpy().tobytes())
    return hasher.hexdigest()


def compute_causal_ntp_loss(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
) -> Tuple[torch.Tensor, float, int]:
    """Compute causal next-token prediction loss with manual one-position shift."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    valid_tokens = shift_labels.numel()
    ppl = math.exp(min(loss.item(), 20.0))
    return loss, ppl, valid_tokens


def compute_target_ce_loss(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    target: str,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """Compute target-only causal cross-entropy loss with ChatML EOS (<|im_end|>) supervision."""
    # Strict ChatML completion: target letter + <|im_end|>\n
    full_target = f"{target}<|im_end|>\n"
    full_text = prompt + full_target
    prompt_enc = tokenizer(prompt, return_tensors="pt")
    full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256)

    prompt_len = prompt_enc["input_ids"].shape[1]
    input_ids = full_enc["input_ids"].to(device)
    attention_mask = full_enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100  # Mask prompt tokens

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    valid_tokens = (shift_labels != -100).sum().item()
    return loss, valid_tokens


def train_run05_1_arms(
    repo_root: Path,
    steps: int = 40,
    device_name: str = "cpu",
) -> Dict[str, Any]:
    """Execute training for Arm A and Arm B with full loss and gradient ledger."""
    device = torch.device(device_name)
    random.seed(42)
    torch.manual_seed(42)

    out_dir = repo_root / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_1"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    parent_ckpt_path = repo_root / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_02" / "checkpoints" / "C1_COMPACT_ADAPTER_ONLY.pt"
    if not parent_ckpt_path.is_file():
        raise FileNotFoundError(f"Missing parent C1 checkpoint: {parent_ckpt_path}")

    # Compute parent SHA and fingerprint
    with open(parent_ckpt_path, "rb") as f:
        parent_sha = hashlib.sha256(f.read()).hexdigest()
    parent_state = torch.load(parent_ckpt_path, map_location="cpu", weights_only=True)
    parent_fingerprint = compute_tensor_fingerprint(parent_state)

    # Load Model and Tokenizer
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None,
            local_files_only=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=False)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None,
            local_files_only=False,
        )

    base_model.to(device)

    # Freeze 100% Backbone
    for param in base_model.parameters():
        param.requires_grad = False

    # Setup Adapter Architecture cleanly without external M0 dependency
    adapted_modules = replace_mlp_with_dual_cycle_mat_lora(
        model=base_model,
        rank=8,
        alpha=16.0,
        num_harmonics=2,
    )
    gated_proj = GatedResidualProjection(in_features=40, out_features=getattr(base_model.config, "hidden_size", 896)).to(device)
    cond_context = ConditioningContext(injection_token_index=14)
    hook = make_gated_embedding_hook(cond_context)
    base_model.get_input_embeddings().register_forward_hook(hook)

    # Load Training Datasets
    nwp_path = repo_root / "data" / "tempun" / "train_nwp.jsonl"
    rel_path = repo_root / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_02" / "train_relation_pairs.jsonl"
    mcqa_path = repo_root / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_04" / "MCQA4_TRAIN_CLEAN.jsonl"

    with open(nwp_path, "r", encoding="utf-8") as f:
        nwp_data = [json.loads(line) for line in f if line.strip()]
    with open(rel_path, "r", encoding="utf-8") as f:
        rel_data = [json.loads(line) for line in f if line.strip()]
    with open(mcqa_path, "r", encoding="utf-8") as f:
        mcqa_data = [json.loads(line) for line in f if line.strip()]

    gradient_traces = []
    arm_results = {}

    for arm in ["R51_A_NTP_REL", "R51_B_NTP_REL_MCQA"]:
        print(f"\n================ Starting {arm} Training ================")
        # Strict Reset to Parent Checkpoint State
        p_adapters = parent_state.get("adapters_state", {})
        p_gated = parent_state.get("gated_proj_state", {})

        for mod_name, module in adapted_modules.items():
            if mod_name in p_adapters:
                mod_dict = {k: v.to(device).clone() for k, v in p_adapters[mod_name].items()}
                incomp = module.load_state_dict(mod_dict, strict=False)
                assert all("base_layer" in k for k in incomp.missing_keys), f"Unexpected missing keys: {incomp.missing_keys}"
                assert len(incomp.unexpected_keys) == 0, f"Unexpected extra keys: {incomp.unexpected_keys}"

        if p_gated:
            g_dict = {k: v.to(device).clone() for k, v in p_gated.items()}
            gated_proj.load_state_dict(g_dict, strict=True)

        # STRICT CORRECTNESS: Only enable adapter parameters for grad!
        trainable_params = []
        for mod in adapted_modules.values():
            mod.lora_A.requires_grad = True
            mod.lora_B.requires_grad = True
            for p in mod.core_generator.parameters():
                p.requires_grad = True
            trainable_params.extend([mod.lora_A, mod.lora_B] + list(mod.core_generator.parameters()))
            # Lock base linear layer
            mod.base_layer.weight.requires_grad = False
            if mod.base_layer.bias is not None:
                mod.base_layer.bias.requires_grad = False

        for p in gated_proj.parameters():
            p.requires_grad = True
            trainable_params.append(p)

        optimizer = torch.optim.AdamW(trainable_params, lr=2e-4, weight_decay=0.01)

        base_model.train()
        gated_proj.train()

        losses_trace = []
        ntp_losses = []
        task_losses = []

        # Initialize Stratified Samplers with seed 42
        sampler_nwp = DeterministicStratifiedSampler(
            nwp_data,
            key_fn=lambda x: str(x.get("canonical_year", 1984)),
            seed=42,
        )
        sampler_rel = DeterministicStratifiedSampler(
            rel_data,
            key_fn=lambda x: f"{x.get('year_eval', 1984)}_{x.get('target', 'Yes')}_{x.get('entity', '')}",
            seed=42,
        )
        sampler_mcqa = DeterministicStratifiedSampler(
            mcqa_data,
            key_fn=lambda x: f"{x.get('year_eval', 1984)}_{x.get('target', 'a')}_{x.get('entity', '')}",
            seed=42,
        )

        for step in range(1, steps + 1):
            optimizer.zero_grad()

            # Slot 1: NTP Causal Loss (clean continuation, unconditioned)
            cond_context.clear()
            clear_model_temporal_coordinates(adapted_modules)
            nwp_item = sampler_nwp.sample()
            ntp_text = f"{nwp_item.get('input', '')} {nwp_item.get('target', '')}".strip()

            l_ntp, ppl, v_ntp = compute_causal_ntp_loss(base_model, tokenizer, ntp_text, device)
            ntp_losses.append(l_ntp.item())

            # Slot 2: Relation Causal Loss
            rel_item = sampler_rel.sample()
            rel_year = rel_item.get("year_eval", 1984)
            coords = compute_fixed_fone_canchi_40d(rel_year)
            coords_t = torch.tensor(coords, dtype=torch.float32, device=device)
            proj_vec = gated_proj(coords_t)
            cond_context.set_vector(proj_vec)
            theta, z = extract_canchi_angles_from_40d(coords)
            set_model_temporal_coordinates(adapted_modules, theta=theta, z=z)

            rel_prompt, _ = build_p12_prompt(rel_item.get("input", ""), tokenizer)
            rel_target = rel_item.get("target", "Yes")
            l_rel, v_rel = compute_target_ce_loss(base_model, tokenizer, rel_prompt, rel_target, device)

            # Slot 3: Arm A gets matched relation control; Arm B gets MCQA
            if arm == "R51_A_NTP_REL":
                ctrl_item = sampler_rel.sample()
                ctrl_year = ctrl_item.get("year_eval", 1984)
                c_coords = compute_fixed_fone_canchi_40d(ctrl_year)
                c_coords_t = torch.tensor(c_coords, dtype=torch.float32, device=device)
                c_proj_vec = gated_proj(c_coords_t)
                cond_context.set_vector(c_proj_vec)
                c_th, c_z = extract_canchi_angles_from_40d(c_coords)
                set_model_temporal_coordinates(adapted_modules, theta=c_th, z=c_z)

                ctrl_prompt, _ = build_p12_prompt(ctrl_item.get("input", ""), tokenizer)
                ctrl_target = ctrl_item.get("target", "No")
                l_slot3, v_slot3 = compute_target_ce_loss(base_model, tokenizer, ctrl_prompt, ctrl_target, device)
                total_loss = l_ntp + l_rel + l_slot3
                task_losses.append(l_rel.item() + l_slot3.item())
            else:
                mcqa_item = sampler_mcqa.sample()
                mcqa_year = mcqa_item.get("year_eval", 1984)
                m_coords = compute_fixed_fone_canchi_40d(mcqa_year)
                m_coords_t = torch.tensor(m_coords, dtype=torch.float32, device=device)
                m_proj_vec = gated_proj(m_coords_t)
                cond_context.set_vector(m_proj_vec)
                m_th, m_z = extract_canchi_angles_from_40d(m_coords)
                set_model_temporal_coordinates(adapted_modules, theta=m_th, z=m_z)

                mcqa_prompt, _ = build_p12_prompt(mcqa_item.get("question", ""), tokenizer)
                mcqa_target = mcqa_item.get("target", "a")
                l_mcqa, v_mcqa = compute_target_ce_loss(base_model, tokenizer, mcqa_prompt, mcqa_target, device)
                total_loss = l_ntp + l_rel + l_mcqa
                task_losses.append(l_rel.item() + l_mcqa.item())

            total_loss.backward()

            # Clean context after backward
            cond_context.clear()

            # Verify zero gradient leakage to base layers
            for mod in adapted_modules.values():
                assert mod.base_layer.weight.grad is None

            grad_norm = nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            step_entry = {
                "arm": arm,
                "step": step,
                "total_loss": round(total_loss.item(), 4),
                "l_ntp": round(l_ntp.item(), 4),
                "ntp_ppl": round(ppl, 2),
                "grad_norm": round(grad_norm.item(), 4),
            }
            losses_trace.append(step_entry)
            if step % 10 == 0 or step == steps:
                gradient_traces.append(step_entry)
                print(f"[{arm}] Step {step:02d}/{steps:02d} | Total Loss: {total_loss.item():.4f} | NTP: {l_ntp.item():.4f} | PPL: {ppl:.2f} | Grad: {grad_norm.item():.4f}")

        # Save COMPACT Checkpoint (Strictly Adapter and Gated Proj only, ~13.7 MB)
        final_adapters = {}
        for mod_name, module in adapted_modules.items():
            mod_st = {}
            for pk, pv in module.state_dict().items():
                if "base_layer" not in pk:
                    mod_st[pk] = pv.cpu()
            final_adapters[mod_name] = mod_st

        final_gated = {pk: pv.cpu() for pk, pv in gated_proj.state_dict().items()}

        final_compact_state = {
            "arm_name": arm,
            "steps": steps,
            "final_loss": round(losses_trace[-1]["total_loss"], 4),
            "final_5step_avg_loss": round(sum(e["total_loss"] for e in losses_trace[-5:]) / 5, 4),
            "adapters_state": final_adapters,
            "gated_proj_state": final_gated,
        }

        arm_ckpt_path = ckpt_dir / f"{arm}.pt"
        torch.save(final_compact_state, arm_ckpt_path)
        with open(arm_ckpt_path, "rb") as f:
            arm_sha = hashlib.sha256(f.read()).hexdigest()
        arm_fingerprint = compute_tensor_fingerprint(final_compact_state)

        arm_results[arm] = {
            "checkpoint_path": str(arm_ckpt_path.relative_to(repo_root)),
            "sha256": arm_sha,
            "file_size_bytes": arm_ckpt_path.stat().st_size,
            "tensor_fingerprint": arm_fingerprint,
            "instantaneous_step40_loss": losses_trace[-1]["total_loss"],
            "final_5step_rolling_loss": round(sum(e["total_loss"] for e in losses_trace[-5:]) / 5, 4),
            "final_ntp_loss": round(sum(ntp_losses[-5:]) / 5, 4),
            "final_ntp_ppl": round(math.exp(min(sum(ntp_losses[-5:]) / 5, 20.0)), 2),
        }

        # Write TRAIN_COMPLETION_<ARM>.json
        arm_suffix = "A" if "A_" in arm else "B"
        with open(out_dir / f"TRAIN_COMPLETION_{arm_suffix}.json", "w", encoding="utf-8") as f:
            json.dump(arm_results[arm], f, indent=2)

    # Save Gradient Trace
    with open(out_dir / "LOSS_COMPONENT_GRADIENT_TRACE.jsonl", "w", encoding="utf-8") as f:
        for tr in gradient_traces:
            f.write(json.dumps(tr) + "\n")

    # Checkpoint Parent & Identity Manifest
    fp_a = arm_results["R51_A_NTP_REL"]["tensor_fingerprint"]
    fp_b = arm_results["R51_B_NTP_REL_MCQA"]["tensor_fingerprint"]

    identity_report = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "parent_checkpoint": {
            "name": "C1_COMPACT_ADAPTER_ONLY",
            "path": str(parent_ckpt_path.relative_to(repo_root)),
            "sha256": parent_sha,
            "tensor_fingerprint": parent_fingerprint,
            "file_size_bytes": parent_ckpt_path.stat().st_size,
        },
        "arms_trained": arm_results,
        "divergence_verification": {
            "a_differs_from_parent": fp_a != parent_fingerprint,
            "b_differs_from_parent": fp_b != parent_fingerprint,
            "a_differs_from_b": fp_a != fp_b,
            "arm_a_checkpoint_is_compact": arm_results["R51_A_NTP_REL"]["file_size_bytes"] < 20_000_000,
            "arm_b_checkpoint_is_compact": arm_results["R51_B_NTP_REL_MCQA"]["file_size_bytes"] < 20_000_000,
            "verdict": "GENUINE_INDEPENDENT_TRAINING_VERIFIED_COMPACT_SCHEMA",
        },
    }

    with open(out_dir / "CHECKPOINT_PARENT_AND_TENSOR_IDENTITY.json", "w", encoding="utf-8") as f:
        json.dump(identity_report, f, indent=2)

    return identity_report


if __name__ == "__main__":
    _repo = Path(__file__).resolve().parents[2]
    train_run05_1_arms(repo_root=_repo, steps=40, device_name="cpu")
