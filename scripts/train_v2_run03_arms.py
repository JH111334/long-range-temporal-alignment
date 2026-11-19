"""Execute Real Equal-Budget Training for 4 Matched Experimental Arms in V2 Run 03.

Enforces Gate 2 constraints:
- Identical Starting Checkpoint: Run 02 C1 (SHA256: 09acf723ea97401924a070b14775d09584d0a7a565c68d83c71218e814f72cb9)
- 100% Frozen Qwen2.5-0.5B-Instruct Backbone (zero backbone gradients enforced)
- Exactly 30 Optimizer Steps per Arm with AdamW (lr=2e-4, weight_decay=0.01, batch_size=2)
- Multi-Task Arm Design:
    * R3_A_RELATION_ONLY: 1.0 Relation, 0.0 MCQA (True FoNE-CanChi)
    * R3_B_MCQA_ONLY: 0.0 Relation, 1.0 MCQA (True FoNE-CanChi)
    * R3_C_JOINT_TRUE: 0.5 Relation + 0.5 MCQA (True FoNE-CanChi - PRIMARY ARM)
    * R3_D_JOINT_SHUFFLED: 0.5 Relation + 0.5 MCQA (Shuffled FoNE-CanChi - NEGATIVE CONTROL)
- Independent Checkpoint Output per Arm with SHA256 and parameter delta proof
- Full gradient norm tracking and execution ledgers
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


def train_run03():
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cpu")
    print(f"Training V2 Run 03 on device: {device}")

    run02_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_02"
    run03_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_03"
    ckpt_dir = run03_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    c1_ckpt_path = run02_dir / "checkpoints" / "C1_TRUE_TEMPORAL_RELATION.pt"
    if not c1_ckpt_path.is_file():
        raise FileNotFoundError(f"Missing starting C1 checkpoint: {c1_ckpt_path}")

    c1_sha256 = compute_sha256(c1_ckpt_path)
    expected_c1_sha = "09acf723ea97401924a070b14775d09584d0a7a565c68d83c71218e814f72cb9"
    if c1_sha256 != expected_c1_sha:
        raise ValueError(f"C1 Checkpoint SHA256 mismatch! Expected {expected_c1_sha}, got {c1_sha256}")
    print(f"Verified Starting Checkpoint C1 SHA256: {c1_sha256}")

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

    # 3. Setup M0 Adapter Architecture and Load C1 Checkpoint Weights
    base_m0_path = _REPO_ROOT / "artifacts" / "quarantine" / "M0_FC40" / "kaggle_full_v2_3seed" / "artifacts" / "seed_1" / "checkpoints" / "m0_fc40_seed_1_after_02.pt"
    adapted_modules, gated_proj, cond_context = load_m0_fc40_for_inference(
        model=base_model,
        checkpoint_path=base_m0_path,
        device=device,
        rank=8,
        alpha=16.0,
        num_harmonics=2,
        injection_token_index=14,
    )

    # Load C1 weights into adapter and gated_proj
    c1_saved = torch.load(c1_ckpt_path, map_location=device, weights_only=False)
    for k, v in adapted_modules.items():
        v.load_state_dict(c1_saved["adapters_state"][k])
    gated_proj.load_state_dict(c1_saved["gated_proj_state"])
    print(f"Successfully initialized model with Run 02 C1 weights.")

    # Cache initial C1 state for exact reset and parameter delta measurement
    c1_adapter_state = {k: {pk: pv.cpu().clone() for pk, pv in v.state_dict().items()} for k, v in adapted_modules.items()}
    c1_gated_state = {pk: pv.cpu().clone() for pk, pv in gated_proj.state_dict().items()}

    # 4. Load Training Data Splits
    relation_train_path = run02_dir / "train_relation_pairs.jsonl"
    mcqa_train_path = run03_dir / "MCQA4_TRAIN.jsonl"

    with open(relation_train_path, "r", encoding="utf-8") as f:
        relation_train_records = [json.loads(line) for line in f if line.strip()]

    with open(mcqa_train_path, "r", encoding="utf-8") as f:
        mcqa_train_records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(relation_train_records)} relation pairs and {len(mcqa_train_records)} MCQA4 train records.")

    arms = [
        "R3_A_RELATION_ONLY",
        "R3_B_MCQA_ONLY",
        "R3_C_JOINT_TRUE",
        "R3_D_JOINT_SHUFFLED",
    ]

    steps_per_arm = 30
    batch_size = 2
    lr = 0.0002
    weight_decay = 0.01

    ledger_records = []
    checkpoint_identities = {}
    mixing_ledger = []

    all_years_pool = [1947, 1952, 1960, 1970, 1980, 1984, 1990, 2000, 2010, 2020, 2022]

    # Pre-shuffle datasets deterministically
    rng_rel = random.Random(42)
    rel_indices = list(range(len(relation_train_records)))
    rng_rel.shuffle(rel_indices)

    rng_mcqa = random.Random(42)
    mcqa_indices = list(range(len(mcqa_train_records)))
    rng_mcqa.shuffle(mcqa_indices)

    rng_shuff = random.Random(42)

    for arm in arms:
        print(f"\n==========================================")
        print(f"Starting Training for Arm: {arm}")
        print(f"==========================================")

        # Exact state reset to C1 checkpoint
        for k, v in adapted_modules.items():
            v.load_state_dict(c1_adapter_state[k])
            # Strict freeze of underlying backbone projection
            if hasattr(v, "base_layer"):
                for p in v.base_layer.parameters():
                    p.requires_grad = False
            # Only enable adapter parameters
            v.lora_A.requires_grad = True
            v.lora_B.requires_grad = True
            for p in v.core_generator.parameters():
                p.requires_grad = True

        gated_proj.load_state_dict(c1_gated_state)
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
        arm_rel_losses = []
        arm_mcqa_losses = []
        arm_grad_norms = []
        t_arm_start = time.time()

        base_model.train()

        for step in range(steps_per_arm):
            optimizer.zero_grad()
            batch_total_loss = 0.0
            step_rel_loss = 0.0
            step_mcqa_loss = 0.0

            # Determine task composition for this step
            if arm == "R3_A_RELATION_ONLY":
                # 2 relation samples
                step_items = [
                    ("relation", relation_train_records[rel_indices[(step * 2 + 0) % len(relation_train_records)]]),
                    ("relation", relation_train_records[rel_indices[(step * 2 + 1) % len(relation_train_records)]]),
                ]
            elif arm == "R3_B_MCQA_ONLY":
                # 2 MCQA samples
                step_items = [
                    ("mcqa", mcqa_train_records[mcqa_indices[(step * 2 + 0) % len(mcqa_train_records)]]),
                    ("mcqa", mcqa_train_records[mcqa_indices[(step * 2 + 1) % len(mcqa_train_records)]]),
                ]
            elif arm in ("R3_C_JOINT_TRUE", "R3_D_JOINT_SHUFFLED"):
                # Balanced joint: 1 relation + 1 MCQA
                step_items = [
                    ("relation", relation_train_records[rel_indices[step % len(relation_train_records)]]),
                    ("mcqa", mcqa_train_records[mcqa_indices[step % len(mcqa_train_records)]]),
                ]

            for task_type, item in step_items:
                prompt_text, _ = build_p12_prompt(item["question"], tokenizer)
                target_text = item["target"]
                year_eval = item["year_eval"]

                if arm == "R3_D_JOINT_SHUFFLED":
                    # Shuffled conditioning: choose year not equal to eval year
                    cand_years = [y for y in all_years_pool if y != year_eval]
                    year_cond = rng_shuff.choice(cand_years)
                else:
                    year_cond = year_eval

                # Tokenize prompt and target
                prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
                target_ids = tokenizer(" " + target_text, add_special_tokens=False)["input_ids"]
                eos_id = tokenizer.eos_token_id

                input_ids = prompt_ids + target_ids + [eos_id]
                labels = [-100] * len(prompt_ids) + target_ids + [eos_id]

                inp_t = torch.tensor([input_ids], dtype=torch.long, device=device)
                lbl_t = torch.tensor([labels], dtype=torch.long, device=device)

                # Temporal feature conditioning
                vec40 = compute_fixed_fone_canchi_40d(year_cond)
                theta, z = extract_canchi_angles_from_40d(vec40)
                set_model_temporal_coordinates(adapted_modules, theta, z)

                vec_t = torch.tensor(vec40, dtype=torch.float32, device=device).unsqueeze(0)
                injected_v = gated_proj(vec_t).squeeze(0)
                cond_context.injection_token_index = 14
                cond_context.set_vector(injected_v)

                out = base_model(input_ids=inp_t, labels=lbl_t)
                loss_i = out.loss / batch_size
                loss_i.backward()

                sub_loss_val = loss_i.item() * batch_size
                batch_total_loss += sub_loss_val
                if task_type == "relation":
                    step_rel_loss += sub_loss_val
                else:
                    step_mcqa_loss += sub_loss_val

                cond_context.clear()

            # Fail-fast boundary assert
            if math.isnan(batch_total_loss) or math.isinf(batch_total_loss):
                raise FloatingPointError(f"Non-finite loss detected in arm {arm} at step {step}: {batch_total_loss}")

            # Check backbone parameters have zero grad
            for name, param in base_model.named_parameters():
                if "lora_" not in name and "core_generator" not in name and "gated_proj" not in name:
                    if param.grad is not None and param.grad.abs().sum().item() > 0:
                        raise ValueError(f"CRITICAL: Backbone parameter {name} received non-zero gradient!")

            # Track trainable grad norm
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()
            optimizer.step()

            arm_losses.append(batch_total_loss)
            arm_rel_losses.append(step_rel_loss)
            arm_mcqa_losses.append(step_mcqa_loss)
            arm_grad_norms.append(grad_norm)

            step_entry = {
                "arm": arm,
                "step": step + 1,
                "total_loss": round(batch_total_loss, 5),
                "rel_loss": round(step_rel_loss, 5),
                "mcqa_loss": round(step_mcqa_loss, 5),
                "grad_norm": round(grad_norm, 5),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+07:00"),
            }
            ledger_records.append(step_entry)
            mixing_ledger.append(step_entry)

            if (step + 1) % 10 == 0 or (step + 1) == steps_per_arm:
                print(f"[{arm}] Step {step + 1}/{steps_per_arm} - Total Loss: {batch_total_loss:.4f} (Rel: {step_rel_loss:.4f}, MCQA: {step_mcqa_loss:.4f}) - Grad Norm: {grad_norm:.4f}")

        # Compute parameter delta norm from C1 starting state
        param_delta_sq = 0.0
        total_p_norm = 0.0
        for k, mod in adapted_modules.items():
            c1_b = c1_adapter_state[k]["lora_B"]
            curr_b = mod.lora_B.data.cpu()
            diff_b = curr_b - c1_b
            param_delta_sq += (diff_b ** 2).sum().item()
            total_p_norm += (curr_b ** 2).sum().item()

        param_delta_norm = math.sqrt(param_delta_sq)
        print(f"[{arm}] Training complete in {time.time() - t_arm_start:.1f}s. Param Delta Norm from C1: {param_delta_norm:.6f}")

        # 5. Save Checkpoint for this arm
        save_ckpt_path = ckpt_dir / f"{arm}.pt"
        saved_state = {
            "arm_name": arm,
            "steps": steps_per_arm,
            "starting_checkpoint": "C1_TRUE_TEMPORAL_RELATION",
            "starting_c1_sha256": c1_sha256,
            "final_loss": arm_losses[-1],
            "param_delta_from_c1_norm": param_delta_norm,
            "adapters_state": {k: {pk: pv.cpu() for pk, pv in v.state_dict().items()} for k, v in adapted_modules.items()},
            "gated_proj_state": {pk: pv.cpu() for pk, pv in gated_proj.state_dict().items()},
        }
        torch.save(saved_state, save_ckpt_path)
        ckpt_sha = compute_sha256(save_ckpt_path)

        checkpoint_identities[arm] = {
            "checkpoint_path": str(save_ckpt_path),
            "sha256": ckpt_sha,
            "bytes": save_ckpt_path.stat().st_size,
            "param_delta_from_c1": round(param_delta_norm, 6),
            "final_step_loss": round(arm_losses[-1], 5),
            "final_rel_loss": round(arm_rel_losses[-1], 5),
            "final_mcqa_loss": round(arm_mcqa_losses[-1], 5),
            "optimizer_steps_executed": steps_per_arm,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "grad_clip_max_norm": 1.0,
            "batch_size": batch_size,
        }

    # 6. Save Training Execution Ledger and Checkpoint Identity
    ledger_path = run03_dir / "TRAINING_EXECUTION.jsonl"
    with open(ledger_path, "w", encoding="utf-8") as f:
        for rec in ledger_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved execution ledger to {ledger_path}")

    mixing_ledger_path = run03_dir / "TASK_MIXING_AND_GRADIENT_LEDGER.jsonl"
    with open(mixing_ledger_path, "w", encoding="utf-8") as f:
        for rec in mixing_ledger:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved task mixing ledger to {mixing_ledger_path}")

    identities_path = run03_dir / "CHECKPOINT_IDENTITY.json"
    with open(identities_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_identities, f, indent=2)
    print(f"Saved checkpoint identities to {identities_path}")


if __name__ == "__main__":
    train_run03()
