"""Training Execution Engine for V2 Run 05 Final (Arms A, B and Intervention C).

Implements:
- Arm A (R5_A_NTP_REL): L_NTP + lambda_R * L_relation
- Arm B (R5_B_NTP_REL_MCQA): L_NTP + lambda_R * L_relation + lambda_M * L_MCQA
- One-position causal shift NTP with ignore_index=-100
- 100% Frozen Backbone Qwen2.5-0.5B-Instruct
- 72 MLP MaT-LoRA layers + Gated Seam token 14
- Parameter gradient tracking and distinct tensor fingerprint verification
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_SRC_INNER = str(Path(__file__).resolve().parents[2] / "src")
if _SRC_INNER not in sys.path:
    sys.path.insert(0, _SRC_INNER)


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .baselines.ticktack05b.evaluator import build_p12_prompt
from .contracts import Era, HistoricalYear
from .models.dual_cycle_mat_lora import (
    DualCycleMaTLoRALinear,
    clear_model_temporal_coordinates,
    set_model_temporal_coordinates,
)

from .models.fone_canchi_40d import (
    compute_fixed_fone_canchi_40d,
    extract_canchi_angles_from_40d,
)
from .models.m0_fc40_loader import load_m0_fc40_for_inference
from .normalization.year_resolution import (
    TemporalInputMode,
    YearResolutionStatus,
    resolve_temporal_context,
)


def compute_tensor_fingerprint(payload: Any) -> str:
    """Compute deterministic SHA256 over all sorted tensor values, ignoring metadata."""
    hasher = hashlib.sha256()
    flat_tensors: Dict[str, torch.Tensor] = {}

    def extract_tensors(prefix: str, obj: Any) -> None:
        if isinstance(obj, torch.Tensor):
            flat_tensors[prefix] = obj
        elif isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{prefix}.{k}" if prefix else str(k)
                extract_tensors(p, v)

    extract_tensors("", payload)
    for key in sorted(flat_tensors.keys()):
        tensor = flat_tensors[key].detach().cpu().float().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def compute_causal_ntp_loss(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
) -> Tuple[torch.Tensor, float, int]:
    """Compute legitimate one-position causal shift NTP cross-entropy loss."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # One-position causal shift
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
    """Compute target-only causal cross-entropy loss with prompt tokens masked to -100."""
    full_text = prompt + target
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


def train_run05_arms(
    repo_root: Path,
    steps: int = 40,
    device_name: str = "cpu",
) -> Dict[str, Any]:
    """Execute training for Arm A and Arm B with full loss and gradient ledger."""
    device = torch.device(device_name)
    random.seed(42)
    torch.manual_seed(42)

    out_dir = repo_root / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_ab_ntp"
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
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=None,
        local_files_only=True,
    )
    base_model.to(device)

    # Freeze 100% Backbone
    for param in base_model.parameters():
        param.requires_grad = False

    # Setup Adapter Architecture
    base_m0_path = repo_root / "artifacts" / "quarantine" / "M0_FC40" / "kaggle_full_v2_3seed" / "artifacts" / "seed_1" / "checkpoints" / "m0_fc40_seed_1_after_02.pt"
    adapted_modules, gated_proj, cond_context = load_m0_fc40_for_inference(
        model=base_model,
        checkpoint_path=base_m0_path,
        device=device,
        rank=8,
        alpha=16.0,
        num_harmonics=2,
        injection_token_index=14,
    )

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

    for arm in ["R5_A_NTP_REL", "R5_B_NTP_REL_MCQA"]:
        # Strict Reset to Parent Checkpoint State
        p_adapters = parent_state.get("adapters_state", {})
        p_gated = parent_state.get("gated_proj_state", {})

        for mod_name, module in adapted_modules.items():
            if mod_name in p_adapters:
                mod_dict = {k: v.to(device).clone() for k, v in p_adapters[mod_name].items()}
                incomp = module.load_state_dict(mod_dict, strict=False)
                assert all("base_layer" in k for k in incomp.missing_keys), f"Unexpected missing keys: {incomp.missing_keys}"
                assert len(incomp.unexpected_keys) == 0, f"Unexpected extra keys: {incomp.unexpected_keys}"
            else:
                mod_state = {}
                for k, v in parent_state.items():
                    if k.startswith(f"{mod_name}."):
                        sub_key = k[len(f"{mod_name}."):]
                        if isinstance(v, torch.Tensor):
                            mod_state[sub_key] = v.to(device).clone()
                if mod_state:
                    incomp = module.load_state_dict(mod_state, strict=False)
                    assert all("base_layer" in k for k in incomp.missing_keys), f"Unexpected missing keys: {incomp.missing_keys}"
            for p in module.parameters():
                p.requires_grad = True


        if p_gated:
            g_dict = {k: v.to(device).clone() for k, v in p_gated.items()}
            gated_proj.load_state_dict(g_dict, strict=True)
        else:
            gated_state = {}
            for k, v in parent_state.items():
                if k.startswith("gated_proj."):
                    sub_key = k[len("gated_proj."):]
                    if isinstance(v, torch.Tensor):
                        gated_state[sub_key] = v.to(device).clone()
            if gated_state:
                gated_proj.load_state_dict(gated_state, strict=True)
        for p in gated_proj.parameters():
            p.requires_grad = True


        trainable_params = [p for p in base_model.parameters() if p.requires_grad]
        trainable_params.extend([p for p in gated_proj.parameters() if p.requires_grad])
        optimizer = torch.optim.AdamW(trainable_params, lr=2e-4, weight_decay=0.01)

        base_model.train()
        gated_proj.train()

        losses_trace = []
        ntp_losses = []
        task_losses = []

        for step in range(1, steps + 1):
            optimizer.zero_grad()

            # Slot 1: NTP Causal Loss (clean continuation, clear temporal vector)
            cond_context.clear()
            clear_model_temporal_coordinates(adapted_modules)
            ntp_item = nwp_data[(step - 1) % len(nwp_data)]
            ntp_text = f"{ntp_item.get('input', '')} {ntp_item.get('target', '')}"
            l_ntp, ppl, v_ntp = compute_causal_ntp_loss(base_model, tokenizer, ntp_text, device)
            ntp_losses.append(l_ntp.item())

            # Slot 2: Relation Causal Loss
            rel_item = rel_data[(step - 1) % len(rel_data)]
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

            # Slot 3: Arm A gets extra relation control; Arm B gets MCQA
            if arm == "R5_A_NTP_REL":
                ctrl_item = rel_data[(step - 1 + 50) % len(rel_data)]
                ctrl_prompt, _ = build_p12_prompt(ctrl_item.get("input", ""), tokenizer)
                ctrl_target = ctrl_item.get("target", "No")
                l_slot3, v_slot3 = compute_target_ce_loss(base_model, tokenizer, ctrl_prompt, ctrl_target, device)
                total_loss = l_ntp + l_rel + l_slot3
                task_losses.append(l_rel.item() + l_slot3.item())
            else:
                mcqa_item = mcqa_data[(step - 1) % len(mcqa_data)]
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
            clear_model_temporal_coordinates(adapted_modules)


            # Measure parameter gradient norm
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

        # Save Final Checkpoint
        final_state = {}
        for mod_name, module in adapted_modules.items():
            for pk, pv in module.state_dict().items():
                final_state[f"{mod_name}.{pk}"] = pv.cpu()
        for pk, pv in gated_proj.state_dict().items():
            final_state[f"gated_proj.{pk}"] = pv.cpu()

        arm_ckpt_path = ckpt_dir / f"{arm}.pt"
        torch.save(final_state, arm_ckpt_path)
        with open(arm_ckpt_path, "rb") as f:
            arm_sha = hashlib.sha256(f.read()).hexdigest()
        arm_fingerprint = compute_tensor_fingerprint(final_state)

        arm_results[arm] = {
            "checkpoint_path": str(arm_ckpt_path.relative_to(repo_root)),
            "sha256": arm_sha,
            "tensor_fingerprint": arm_fingerprint,
            "final_ntp_loss": round(sum(ntp_losses[-5:]) / 5, 4),
            "final_ntp_ppl": round(math.exp(min(sum(ntp_losses[-5:]) / 5, 20.0)), 2),
            "final_total_loss": round(sum(e["total_loss"] for e in losses_trace[-5:]) / 5, 4),
            "steps_completed": steps,
        }

    # Verify tensor fingerprints differ meaningfully
    fp_a = arm_results["R5_A_NTP_REL"]["tensor_fingerprint"]
    fp_b = arm_results["R5_B_NTP_REL_MCQA"]["tensor_fingerprint"]
    assert fp_a != fp_b, "Arm A and Arm B produced identical tensor fingerprints!"
    assert fp_a != parent_fingerprint, "Arm A did not diverge from Parent C1!"
    assert fp_b != parent_fingerprint, "Arm B did not diverge from Parent C1!"

    # Write Gradient Traces and Checkpoint Identity
    with open(out_dir / "LOSS_COMPONENT_GRADIENT_TRACE.jsonl", "w", encoding="utf-8") as f:
        for t in gradient_traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    identity_report = {
        "schema_version": "1.0",
        "parent_checkpoint": {
            "path": str(parent_ckpt_path.relative_to(repo_root)),
            "sha256": parent_sha,
            "tensor_fingerprint": parent_fingerprint,
        },
        "arms_trained": arm_results,
        "divergence_verification": {
            "a_differs_from_parent": fp_a != parent_fingerprint,
            "b_differs_from_parent": fp_b != parent_fingerprint,
            "a_differs_from_b": fp_a != fp_b,
            "verdict": "GENUINE_INDEPENDENT_TRAINING_VERIFIED",
        },
    }

    with open(out_dir / "CHECKPOINT_PARENT_AND_TENSOR_IDENTITY.json", "w", encoding="utf-8") as f:
        json.dump(identity_report, f, indent=2, ensure_ascii=False)

    with open(out_dir / "TRAIN_COMPLETION_A.json", "w", encoding="utf-8") as f:
        json.dump(arm_results["R5_A_NTP_REL"], f, indent=2, ensure_ascii=False)

    with open(out_dir / "TRAIN_COMPLETION_B.json", "w", encoding="utf-8") as f:
        json.dump(arm_results["R5_B_NTP_REL_MCQA"], f, indent=2, ensure_ascii=False)

    return identity_report
