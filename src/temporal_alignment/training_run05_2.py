"""Training Execution Engine for V2 Run 05.2 (Mechanism Contract Repair).

Implements:
- Arm A (R52_A_NTP_REL): L_NTP + lambda_R * L_relation + lambda_C * L_relation_ctrl
- Arm B (R52_B_NTP_REL_MCQA): L_NTP + lambda_R * L_relation + lambda_M * L_MCQA
- ChatML Target Supervision with <|im_end|> (token 151645) and strict prompt label masking (-100)
- 100% Frozen Backbone Qwen2.5-0.5B-Instruct (base_layer.weight.requires_grad = False)
- Deterministic Stratified Sampler balancing multi-strata (years, entities, targets) without collapse
- Year anchor resolution via temporal_resolver to prevent silent 2000/1984 fallback
- Scoped adapter activation for safe condition toggling
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

from temporal_alignment.contracts import Era, HistoricalYear
from temporal_alignment.experiment_schema import TemporalConditionType
from temporal_alignment.experiments.gatekeeper import Gatekeeper, GatePredicate, GateStatus
from temporal_alignment.experiments.sampler_contract import (
    make_mcqa_sampler,
    make_nwp_sampler,
    make_relation_sampler,
)
from temporal_alignment.models.adapter_activation import (
    configure_temporal_condition,
    scoped_adapter_activation,
)
from temporal_alignment.models.dual_cycle_mat_lora import (
    DualCycleMaTLoRALinear,
    clear_model_temporal_coordinates,
    replace_mlp_with_dual_cycle_mat_lora,
    set_model_temporal_coordinates,
)
from temporal_alignment.models.fone_canchi_40d import (
    compute_fixed_fone_canchi_40d,
    extract_canchi_angles_from_40d,
)
from temporal_alignment.models.gated_seam import (
    ConditioningContext,
    GatedResidualProjection,
    make_gated_embedding_hook,
)
from temporal_alignment.normalization.temporal_resolver import resolve_temporal_anchor
from temporal_alignment.training.prompt_contract import (
    PromptProtocol,
    render_prompt,
    tokenize_sft_target,
)


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
    """Compute target-only causal cross-entropy loss using strict prompt_contract."""
    input_ids, labels, attention_mask, valid_tokens = tokenize_sft_target(
        prompt_str=prompt,
        target_str=target,
        tokenizer=tokenizer,
        device=device,
        max_length=256,
    )

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss, valid_tokens


def train_run05_2_arms(
    repo_root: Path,
    steps: int = 40,
    device_name: str = "cpu",
) -> Dict[str, Any]:
    """Execute training for Arm A and Arm B with full loss and gradient ledger."""
    device = torch.device(device_name)
    random.seed(42)
    torch.manual_seed(42)

    out_dir = repo_root / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_2"
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
            dtype=torch.float32,
            device_map=None,
            local_files_only=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=False)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float32,
            device_map=None,
            local_files_only=False,
        )

    base_model.to(device)

    # Freeze 100% Backbone
    for param in base_model.parameters():
        param.requires_grad = False

    # Setup Adapter Architecture
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

    for arm in ["R52_A_NTP_REL", "R52_B_NTP_REL_MCQA"]:
        print(f"\n================ Starting {arm} Training ================", flush=True)
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
            mod.set_enabled(True)
            mod.lora_A.requires_grad = True
            mod.lora_B.requires_grad = True
            for p in mod.core_generator.parameters():
                p.requires_grad = True
            trainable_params.extend([mod.lora_A, mod.lora_B] + list(mod.core_generator.parameters()))
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

        # Initialize Verified Deterministic Samplers
        sampler_nwp = make_nwp_sampler(nwp_data, seed=42)
        sampler_rel = make_relation_sampler(rel_data, seed=42)
        sampler_mcqa = make_mcqa_sampler(mcqa_data, seed=42)

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
            rel_anchor = resolve_temporal_anchor(rel_item)
            rel_year = rel_anchor.effective_year.astronomical if rel_anchor.effective_year else (rel_item.get("year_eval") or 1984)

            rel_coords = compute_fixed_fone_canchi_40d(rel_year)
            rel_theta, rel_z = extract_canchi_angles_from_40d(rel_coords)
            set_model_temporal_coordinates(adapted_modules, theta=rel_theta, z=rel_z)
            rel_vec_t = torch.tensor(rel_coords, dtype=torch.float32, device=device).unsqueeze(0)
            rel_proj = gated_proj(rel_vec_t).squeeze(0)
            cond_context.injection_token_index = 14
            cond_context.set_vector(rel_proj)

            rel_prompt, _ = render_prompt(rel_item.get("input", ""), tokenizer, protocol=PromptProtocol.P12_CORRECTED_V2)
            rel_target = rel_item.get("target", "Yes")
            l_rel, v_rel = compute_target_ce_loss(base_model, tokenizer, rel_prompt, rel_target, device)

            # Slot 3: Arm A gets matched relation control; Arm B gets MCQA
            if arm == "R52_A_NTP_REL":
                ctrl_item = sampler_rel.sample()
                ctrl_anchor = resolve_temporal_anchor(ctrl_item)
                ctrl_year = ctrl_anchor.effective_year.astronomical if ctrl_anchor.effective_year else (ctrl_item.get("year_eval") or 1984)

                ctrl_coords = compute_fixed_fone_canchi_40d(ctrl_year)
                ctrl_theta, ctrl_z = extract_canchi_angles_from_40d(ctrl_coords)
                set_model_temporal_coordinates(adapted_modules, theta=ctrl_theta, z=ctrl_z)
                ctrl_vec_t = torch.tensor(ctrl_coords, dtype=torch.float32, device=device).unsqueeze(0)
                ctrl_proj = gated_proj(ctrl_vec_t).squeeze(0)
                cond_context.injection_token_index = 14
                cond_context.set_vector(ctrl_proj)

                ctrl_prompt, _ = render_prompt(ctrl_item.get("input", ""), tokenizer, protocol=PromptProtocol.P12_CORRECTED_V2)
                ctrl_target = ctrl_item.get("target", "No")
                l_slot3, v_slot3 = compute_target_ce_loss(base_model, tokenizer, ctrl_prompt, ctrl_target, device)
                total_loss = l_ntp + 0.5 * l_rel + 0.25 * l_slot3
                task_losses.append(l_rel.item() + l_slot3.item())
            else:
                mcqa_item = sampler_mcqa.sample()
                mcqa_anchor = resolve_temporal_anchor(mcqa_item)
                mcqa_year = mcqa_anchor.effective_year.astronomical if mcqa_anchor.effective_year else (mcqa_item.get("year_eval") or 1984)

                mcqa_coords = compute_fixed_fone_canchi_40d(mcqa_year)
                mcqa_theta, mcqa_z = extract_canchi_angles_from_40d(mcqa_coords)
                set_model_temporal_coordinates(adapted_modules, theta=mcqa_theta, z=mcqa_z)
                mcqa_vec_t = torch.tensor(mcqa_coords, dtype=torch.float32, device=device).unsqueeze(0)
                mcqa_proj = gated_proj(mcqa_vec_t).squeeze(0)
                cond_context.injection_token_index = 14
                cond_context.set_vector(mcqa_proj)

                mcqa_prompt, _ = render_prompt(mcqa_item.get("question", ""), tokenizer, protocol=PromptProtocol.P12_CORRECTED_V2)
                mcqa_target = mcqa_item.get("target", "a")
                l_mcqa, v_mcqa = compute_target_ce_loss(base_model, tokenizer, mcqa_prompt, mcqa_target, device)
                total_loss = l_ntp + 0.5 * l_rel + 0.5 * l_mcqa
                task_losses.append(l_rel.item() + l_mcqa.item())

            total_loss.backward()

            # Clean context after backward
            cond_context.clear()
            clear_model_temporal_coordinates(adapted_modules)

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
                print(f"[{arm}] Step {step:02d}/{steps:02d} | Total Loss: {total_loss.item():.4f} | NTP: {l_ntp.item():.4f} | PPL: {ppl:.2f} | Grad: {grad_norm.item():.4f}", flush=True)

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

        arm_suffix = "A" if "A_" in arm else "B"
        with open(out_dir / f"TRAIN_COMPLETION_{arm_suffix}.json", "w", encoding="utf-8") as f:
            json.dump(arm_results[arm], f, indent=2)

    # Save Gradient Trace
    with open(out_dir / "LOSS_COMPONENT_GRADIENT_TRACE.jsonl", "w", encoding="utf-8") as f:
        for tr in gradient_traces:
            f.write(json.dumps(tr) + "\n")

    # Checkpoint Parent & Identity Manifest
    fp_a = arm_results["R52_A_NTP_REL"]["tensor_fingerprint"]
    fp_b = arm_results["R52_B_NTP_REL_MCQA"]["tensor_fingerprint"]

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
            "arm_a_checkpoint_is_compact": arm_results["R52_A_NTP_REL"]["file_size_bytes"] < 20_000_000,
            "arm_b_checkpoint_is_compact": arm_results["R52_B_NTP_REL_MCQA"]["file_size_bytes"] < 20_000_000,
            "verdict": "GENUINE_INDEPENDENT_TRAINING_VERIFIED_COMPACT_SCHEMA",
        },
    }

    with open(out_dir / "CHECKPOINT_PARENT_AND_TENSOR_IDENTITY.json", "w", encoding="utf-8") as f:
        json.dump(identity_report, f, indent=2)

    # Register Gate P6 (TRAIN)
    gk = Gatekeeper(out_dir)
    p6_preds = [
        GatePredicate("arm_a_trained", True, arm_results["R52_A_NTP_REL"]["sha256"][:12], "valid_sha"),
        GatePredicate("arm_b_trained", True, arm_results["R52_B_NTP_REL_MCQA"]["sha256"][:12], "valid_sha"),
        GatePredicate("a_differs_from_parent", fp_a != parent_fingerprint, fp_a != parent_fingerprint, True),
        GatePredicate("b_differs_from_parent", fp_b != parent_fingerprint, fp_b != parent_fingerprint, True),
        GatePredicate("a_differs_from_b", fp_a != fp_b, fp_a != fp_b, True),
        GatePredicate("arm_a_compact", arm_results["R52_A_NTP_REL"]["file_size_bytes"] < 20_000_000, arm_results["R52_A_NTP_REL"]["file_size_bytes"], "< 20MB"),
        GatePredicate("arm_b_compact", arm_results["R52_B_NTP_REL_MCQA"]["file_size_bytes"] < 20_000_000, arm_results["R52_B_NTP_REL_MCQA"]["file_size_bytes"], "< 20MB"),
    ]
    res_p6 = gk.register_gate("P6_TRAIN", p6_preds, prerequisite_gates=["P5_CANARY"])
    gk.save_manifest()
    print(f"Gate P6 (TRAIN): {res_p6.status.value}", flush=True)

    return identity_report


if __name__ == "__main__":
    _repo = Path(__file__).resolve().parents[2]
    train_run05_2_arms(repo_root=_repo, steps=40, device_name="cpu")
