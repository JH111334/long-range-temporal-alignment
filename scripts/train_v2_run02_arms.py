"""Execute Real Equal-Budget Training for 4 Matched Experimental Arms.

Enforces Gate 2 constraints:
- Identical Starting M0 Checkpoint (seed 1 after_02)
- 100% Frozen Qwen2.5-0.5B-Instruct Backbone (zero backbone gradients)
- Exactly 30 Optimizer Steps per Arm with AdamW (lr=2e-4, weight_decay=0.01, batch_size=2)
- Independent Checkpoint Output per Arm with SHA256 and parameter delta proof
- Full gradient norm tracking and execution ledger
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from temporal_alignment.baselines.ticktack05b.evaluator import (
    build_p12_prompt,
)
from temporal_alignment.models.dual_cycle_mat_lora import (
    DualCycleMaTLoRALinear,
    set_model_temporal_coordinates,
)
from temporal_alignment.models.fone_canchi_40d import (
    compute_fixed_fone_canchi_40d,
    extract_canchi_angles_from_40d,
)
from temporal_alignment.models.m0_fc40_loader import (
    load_m0_fc40_for_inference,
)


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def train_run02():
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cpu")
    print(f"Training on device: {device}")

    out_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_02"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    base_ckpt_path = _REPO_ROOT / "artifacts" / "quarantine" / "M0_FC40" / "kaggle_full_v2_3seed" / "artifacts" / "seed_1" / "checkpoints" / "m0_fc40_seed_1_after_02.pt"
    if not base_ckpt_path.is_file():
        raise FileNotFoundError(f"Missing base M0 checkpoint: {base_ckpt_path}")

    # 1. Load Tokenizer and Base Model
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=None,
        local_files_only=True,
    )
    base_model.to(device)

    # 2. Strict Freeze of All Backbone Parameters
    for name, param in base_model.named_parameters():
        param.requires_grad = False

    # 3. Setup M0 Adapter Architecture
    adapted_modules, gated_proj, cond_context = load_m0_fc40_for_inference(
        model=base_model,
        checkpoint_path=base_ckpt_path,
        device=device,
        rank=8,
        alpha=16.0,
        num_harmonics=2,
        injection_token_index=14,
    )

    # Cache initial weights for exact reset and parameter delta measurement
    initial_adapter_state = {k: {pk: pv.cpu().clone() for pk, pv in v.state_dict().items()} for k, v in adapted_modules.items()}
    initial_gated_state = {pk: pv.cpu().clone() for pk, pv in gated_proj.state_dict().items()}

    # 4. Load Training Data Splits
    train_nwp_path = _REPO_ROOT / "data" / "tempun" / "train_nwp.jsonl"
    train_pairs_path = out_dir / "train_relation_pairs.jsonl"

    with open(train_nwp_path, "r", encoding="utf-8") as f:
        nwp_train_records = [json.loads(line) for line in f if line.strip()]

    with open(train_pairs_path, "r", encoding="utf-8") as f:
        relation_train_records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(nwp_train_records)} NWP train records and {len(relation_train_records)} relation pairs.")

    arms = [
        "C0_CONTINUED_MASKED",
        "C1_TRUE_TEMPORAL_RELATION",
        "C2_SHUFFLED_CONDITIONING",
        "C3_FORMAT_ONLY",
    ]

    steps_per_arm = 30
    batch_size = 2
    lr = 0.0002
    weight_decay = 0.01

    ledger_records = []
    checkpoint_identities = {}
    gradient_audits = {}

    all_years_pool = [1947, 1952, 1960, 1970, 1980, 1984, 1990, 2000, 2010, 2020, 2022]

    for arm in arms:
        print(f"\n==========================================")
        print(f"Starting Training for Arm: {arm}")
        print(f"==========================================")

        # Exact state reset to base M0 checkpoint
        for k, v in adapted_modules.items():
            v.load_state_dict(initial_adapter_state[k])
            # Strict freeze of underlying backbone projection
            if hasattr(v, "base_layer"):
                for p in v.base_layer.parameters():
                    p.requires_grad = False
            # Only enable adapter parameters
            v.lora_A.requires_grad = True
            v.lora_B.requires_grad = True
            for p in v.core_generator.parameters():
                p.requires_grad = True

        gated_proj.load_state_dict(initial_gated_state)
        for p in gated_proj.parameters():
            p.requires_grad = True

        trainable_params = []
        for v in adapted_modules.values():
            trainable_params.append(v.lora_A)
            trainable_params.append(v.lora_B)
            trainable_params.extend([p for p in v.core_generator.parameters() if p.requires_grad])
        trainable_params.extend([p for p in gated_proj.parameters() if p.requires_grad])

        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

        arm_losses = []
        arm_grad_norms = []
        t_arm_start = time.time()

        base_model.train()

        # Arm data selection
        if arm == "C0_CONTINUED_MASKED":
            data_source = nwp_train_records
        else:
            data_source = relation_train_records

        rng = random.Random(42)
        shuffled_indices = list(range(len(data_source)))
        rng.shuffle(shuffled_indices)

        for step in range(steps_per_arm):
            optimizer.zero_grad()
            batch_loss = 0.0

            # Batch samples
            batch_idxs = [shuffled_indices[(step * batch_size + b) % len(data_source)] for b in range(batch_size)]

            for b_idx in batch_idxs:
                item = data_source[b_idx]

                if arm == "C0_CONTINUED_MASKED":
                    prompt_text = item["input"]
                    target_text = item["target"]
                    year = item["canonical_year"]
                elif arm == "C1_TRUE_TEMPORAL_RELATION":
                    prompt_text, _ = build_p12_prompt(item["question"], tokenizer)
                    target_text = item["target"]
                    year = item["year_eval"]
                elif arm == "C2_SHUFFLED_CONDITIONING":
                    prompt_text, _ = build_p12_prompt(item["question"], tokenizer)
                    target_text = item["target"]
                    # Shuffled conditioning: choose year not equal to eval year
                    cand_years = [y for y in all_years_pool if y != item["year_eval"]]
                    year = rng.choice(cand_years)
                elif arm == "C3_FORMAT_ONLY":
                    # Generic format prompt without temporal logic
                    generic_q = "Answer directly with Yes or No."
                    prompt_text, _ = build_p12_prompt(generic_q, tokenizer)
                    target_text = item["target"]
                    year = 1984

                # Tokenize prompt and target
                prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
                target_ids = tokenizer(" " + target_text, add_special_tokens=False)["input_ids"]
                eos_id = tokenizer.eos_token_id

                input_ids = prompt_ids + target_ids + [eos_id]
                labels = [-100] * len(prompt_ids) + target_ids + [eos_id]

                inp_t = torch.tensor([input_ids], dtype=torch.long, device=device)
                lbl_t = torch.tensor([labels], dtype=torch.long, device=device)

                # Temporal feature conditioning
                vec40 = compute_fixed_fone_canchi_40d(year)
                theta, z = extract_canchi_angles_from_40d(vec40)
                set_model_temporal_coordinates(adapted_modules, theta, z)

                vec_t = torch.tensor(vec40, dtype=torch.float32, device=device).unsqueeze(0)
                injected_v = gated_proj(vec_t).squeeze(0)
                cond_context.injection_token_index = 14
                cond_context.set_vector(injected_v)

                out = base_model(input_ids=inp_t, labels=lbl_t)
                loss = out.loss / batch_size
                loss.backward()
                batch_loss += loss.item() * batch_size

                cond_context.clear()

            # Fail-fast boundary assert
            if math.isnan(batch_loss) or math.isinf(batch_loss):
                raise FloatingPointError(f"Non-finite loss detected in arm {arm} at step {step}: {batch_loss}")

            # Check backbone parameters have zero grad
            for name, param in base_model.named_parameters():
                if "lora_" not in name and "core_generator" not in name and "gated_proj" not in name:
                    if param.grad is not None and param.grad.abs().sum().item() > 0:
                        raise ValueError(f"CRITICAL: Backbone parameter {name} received non-zero gradient!")

            # Track trainable grad norm
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()
            optimizer.step()

            arm_losses.append(batch_loss)
            arm_grad_norms.append(grad_norm)

            ledger_records.append({
                "arm": arm,
                "step": step + 1,
                "loss": round(batch_loss, 5),
                "grad_norm": round(grad_norm, 5),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+07:00"),
            })

            if (step + 1) % 10 == 0 or (step + 1) == steps_per_arm:
                print(f"[{arm}] Step {step + 1}/{steps_per_arm} - Loss: {batch_loss:.4f} - Grad Norm: {grad_norm:.4f}")

        # Compute parameter delta norm from initial state
        param_delta_sq = 0.0
        total_p_norm = 0.0
        for k, mod in adapted_modules.items():
            init_b = initial_adapter_state[k]["lora_B"]
            curr_b = mod.lora_B.data.cpu()
            diff_b = curr_b - init_b
            param_delta_sq += (diff_b ** 2).sum().item()
            total_p_norm += (curr_b ** 2).sum().item()

        param_delta_norm = math.sqrt(param_delta_sq)
        print(f"[{arm}] Training complete in {time.time() - t_arm_start:.1f}s. Param Delta Norm: {param_delta_norm:.6f}")

        # 5. Save Checkpoint for this arm
        save_ckpt_path = ckpt_dir / f"{arm}.pt"
        saved_state = {
            "arm_name": arm,
            "steps": steps_per_arm,
            "final_loss": arm_losses[-1],
            "param_delta_norm": param_delta_norm,
            "adapters_state": {k: {pk: pv.cpu() for pk, pv in v.state_dict().items()} for k, v in adapted_modules.items()},
            "gated_proj_state": {pk: pv.cpu() for pk, pv in gated_proj.state_dict().items()},
        }
        torch.save(saved_state, save_ckpt_path)
        ckpt_sha = compute_sha256(save_ckpt_path)

        checkpoint_identities[arm] = {
            "checkpoint_path": str(save_ckpt_path),
            "sha256": ckpt_sha,
            "bytes": save_ckpt_path.stat().st_size,
            "param_delta_from_m0": round(param_delta_norm, 6),
            "final_step_loss": round(arm_losses[-1], 5),
            "optimizer_steps_executed": steps_per_arm,
        }

        gradient_audits[arm] = {
            "mean_loss": round(sum(arm_losses) / len(arm_losses), 5),
            "min_loss": round(min(arm_losses), 5),
            "max_loss": round(max(arm_losses), 5),
            "mean_grad_norm": round(sum(arm_grad_norms) / len(arm_grad_norms), 5),
            "nonzero_gradient_verified": all(g > 0 for g in arm_grad_norms),
            "backbone_leakage_gradient": 0.0,
            "numerical_instability_detected": False,
        }

    # 6. Save Ledger and Audits
    ledger_path = out_dir / "TRAINING_ARM_EXECUTION_LEDGER.jsonl"
    with open(ledger_path, "w", encoding="utf-8") as f:
        for r in ledger_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_dir / "CHECKPOINT_IDENTITY.json", "w", encoding="utf-8") as f:
        json.dump(checkpoint_identities, f, ensure_ascii=False, indent=2)

    with open(out_dir / "GRADIENT_AND_NUMERIC_AUDIT.json", "w", encoding="utf-8") as f:
        json.dump(gradient_audits, f, ensure_ascii=False, indent=2)

    print("\nGate 2 training complete! Checkpoints and execution ledgers written.")


if __name__ == "__main__":
    train_run02()
