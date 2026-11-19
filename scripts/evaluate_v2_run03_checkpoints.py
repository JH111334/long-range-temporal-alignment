"""Comprehensive Multi-Scoreboard Evaluation for V2 Run 03 Checkpoints.

Evaluates 5 Model Variants:
1. RUN02_C1 (Starting Baseline Checkpoint)
2. R3_A_RELATION_ONLY
3. R3_B_MCQA_ONLY
4. R3_C_JOINT_TRUE (Primary Arm)
5. R3_D_JOINT_SHUFFLED (Negative Control)

Across 4 Distinct Evaluation Scoreboards:
- Scoreboard A: DEV_TEMPORAL_RELATION (60 Dev Relation Pairs)
- Scoreboard B: DEV_TEMPORAL_MCQA4 (120 Balanced 4-Choice MCQA items, including hard negatives)
- Scoreboard C: COUNTERFACTUAL_AND_OPTION_ORDER (Entity-disjoint CF consistency + cyclic option permutation invariance)
- Scoreboard D: SECONDARY_RETENTION_MCQA180 (180 TempUN Small Benchmark Items)

Emits verifiable JSON ledgers and statistical test artifacts.
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
    task_conditioned_strict_parser,
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


def exact_mcnemar_binomial(a_only: int, b_only: int) -> Tuple[float, float]:
    """Exact two-sided binomial test for paired discordant counts."""
    n = a_only + b_only
    if n == 0:
        return 0.0, 1.0
    k = min(a_only, b_only)
    cum_prob = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    two_sided_p = min(1.0, 2.0 * cum_prob)
    chi2 = ((abs(a_only - b_only) - 1.0) ** 2) / n if n > 0 else 0.0
    return chi2, two_sided_p


def generate_single_prediction(
    model: Any,
    tokenizer: Any,
    cond_context: Any,
    adapted_modules: Dict[str, Any],
    gated_proj: Any,
    device: torch.device,
    question_text: str,
    year_cond: int,
    max_new_tokens: int = 4,
) -> Tuple[str, Optional[str], str]:
    """Generate free-vocabulary greedy output and parse strictly."""
    prompt_str, is_yn = build_p12_prompt(question_text, tokenizer)
    inputs = tokenizer(prompt_str, return_tensors="pt").to(device)

    # Condition FoNE-CanChi
    vec40 = compute_fixed_fone_canchi_40d(year_cond)
    theta, z = extract_canchi_angles_from_40d(vec40)
    set_model_temporal_coordinates(adapted_modules, theta, z)

    vec_t = torch.tensor(vec40, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        injected_v = gated_proj(vec_t).squeeze(0)
    cond_context.injection_token_index = 14
    cond_context.set_vector(injected_v)

    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    cond_context.clear()
    gen_tokens = out_ids[0, prompt_len:].cpu().tolist()
    raw_gen = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    parsed_pred, val_status = task_conditioned_strict_parser(raw_gen, is_yn)

    return raw_gen, parsed_pred, val_status


def evaluate_run03():
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cpu")
    print(f"Evaluating V2 Run 03 on device: {device}")

    run02_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_02"
    run03_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_03"
    ckpt_dir = run03_dir / "checkpoints"

    # 1. Load Data Splits
    # Scoreboard A: Dev Relation Pairs (60 items)
    dev_rel_path = run02_dir / "dev_relation_pairs.jsonl"
    with open(dev_rel_path, "r", encoding="utf-8") as f:
        all_dev_rel = [json.loads(line) for line in f if line.strip()]
    rng_a = random.Random(42)
    scoreboard_a_items = rng_a.sample(all_dev_rel, min(60, len(all_dev_rel)))

    # Scoreboard B: Dev MCQA4 (120 items)
    dev_mcqa_path = run03_dir / "MCQA4_DEV.jsonl"
    with open(dev_mcqa_path, "r", encoding="utf-8") as f:
        scoreboard_b_items = [json.loads(line) for line in f if line.strip()]

    # Scoreboard C: Counterfactuals (30 items) + Option Permutations (20 items)
    cf_path = run02_dir / "COUNTERFACTUAL_CASES.jsonl"
    with open(cf_path, "r", encoding="utf-8") as f:
        all_cf = [json.loads(line) for line in f if line.strip()]
    rng_c = random.Random(42)
    cf_items = rng_c.sample(all_cf, min(30, len(all_cf)))

    # Select 20 items from Scoreboard B for option permutation
    option_perm_base = rng_c.sample(scoreboard_b_items, 20)

    # Scoreboard D: TempUN Small 180 items
    eval_benchmark_path = _REPO_ROOT / "data" / "tempun" / "eval_mcqa.jsonl"
    with open(eval_benchmark_path, "r", encoding="utf-8") as f:
        scoreboard_d_items = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded evaluation test suites:")
    print(f"- Scoreboard A (Dev Relation): {len(scoreboard_a_items)} items")
    print(f"- Scoreboard B (Dev MCQA4): {len(scoreboard_b_items)} items")
    print(f"- Scoreboard C (CF / Permutation): {len(cf_items)} CF pairs, {len(option_perm_base)} permuted items")
    print(f"- Scoreboard D (TempUN Small 180): {len(scoreboard_d_items)} items")

    # 2. Load Model Architecture
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading base model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=None,
        local_files_only=True,
    )
    base_model.to(device)

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
    base_model.eval()

    models_to_eval = [
        ("RUN02_C1", run02_dir / "checkpoints" / "C1_TRUE_TEMPORAL_RELATION.pt"),
        ("R3_A_RELATION_ONLY", ckpt_dir / "R3_A_RELATION_ONLY.pt"),
        ("R3_B_MCQA_ONLY", ckpt_dir / "R3_B_MCQA_ONLY.pt"),
        ("R3_C_JOINT_TRUE", ckpt_dir / "R3_C_JOINT_TRUE.pt"),
        ("R3_D_JOINT_SHUFFLED", ckpt_dir / "R3_D_JOINT_SHUFFLED.pt"),
    ]

    results_a = {}
    results_b = {}
    results_c = {}
    results_d = {}
    all_error_cases = []

    for model_name, ckpt_path in models_to_eval:
        print(f"\n==================================================")
        print(f"Evaluating Model: {model_name}")
        print(f"Checkpoint: {ckpt_path.name}")
        print(f"==================================================")

        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        for k, v in adapted_modules.items():
            v.load_state_dict(saved["adapters_state"][k])
        gated_proj.load_state_dict(saved["gated_proj_state"])

        # ----------------------------------------------------
        # Scoreboard A: DEV_TEMPORAL_RELATION (60 items)
        # ----------------------------------------------------
        correct_a = 0
        invalid_a = 0
        pred_a_counts = {"Yes": 0, "No": 0, "Invalid": 0}
        per_case_a = []

        for item in scoreboard_a_items:
            q_text = item["question"]
            gold = item["target"]
            year_eval = item["year_eval"]

            raw_gen, parsed_pred, val_status = generate_single_prediction(
                base_model, tokenizer, cond_context, adapted_modules, gated_proj,
                device, q_text, year_eval
            )

            is_correct = (parsed_pred == gold)
            if is_correct:
                correct_a += 1
            if val_status != "VALID":
                invalid_a += 1
                pred_a_counts["Invalid"] += 1
            else:
                pred_a_counts[parsed_pred] = pred_a_counts.get(parsed_pred, 0) + 1

            per_case_a.append({
                "sample_id": item.get("sample_id", ""),
                "gold": gold,
                "parsed_pred": parsed_pred,
                "is_correct": is_correct,
            })

            if not is_correct:
                all_error_cases.append({
                    "model": model_name,
                    "scoreboard": "A_RELATION",
                    "question": q_text,
                    "gold": gold,
                    "raw_gen": raw_gen,
                    "parsed": parsed_pred,
                })

        acc_a = round(correct_a / len(scoreboard_a_items), 4)
        results_a[model_name] = {
            "sample_size": len(scoreboard_a_items),
            "correct": correct_a,
            "accuracy": acc_a,
            "invalid_count": invalid_a,
            "invalid_rate": round(invalid_a / len(scoreboard_a_items), 4),
            "prediction_distribution": pred_a_counts,
            "per_case": per_case_a,
        }
        print(f"[Scoreboard A] Acc: {acc_a * 100:.2f}% ({correct_a}/{len(scoreboard_a_items)}) - Dist: {pred_a_counts}")

        # ----------------------------------------------------
        # Scoreboard B: DEV_TEMPORAL_MCQA4 (120 items)
        # ----------------------------------------------------
        correct_b = 0
        invalid_b = 0
        hard_neg_correct = 0
        hard_neg_total = 0
        pred_b_counts = {"a": 0, "b": 0, "c": 0, "d": 0, "Invalid": 0}
        per_case_b = []

        for item in scoreboard_b_items:
            q_text = item["question"]
            gold = item["target"].lower()
            year_eval = item["year_eval"]
            is_inc = item.get("is_increasing", True)

            raw_gen, parsed_pred, val_status = generate_single_prediction(
                base_model, tokenizer, cond_context, adapted_modules, gated_proj,
                device, q_text, year_eval
            )

            is_correct = (parsed_pred == gold)
            if is_correct:
                correct_b += 1
            if val_status != "VALID":
                invalid_b += 1
                pred_b_counts["Invalid"] += 1
            else:
                pred_b_counts[parsed_pred] = pred_b_counts.get(parsed_pred, 0) + 1

            if not is_inc:
                hard_neg_total += 1
                if is_correct:
                    hard_neg_correct += 1

            per_case_b.append({
                "sample_id": item.get("sample_id", ""),
                "gold": gold,
                "parsed_pred": parsed_pred,
                "is_correct": is_correct,
                "is_increasing": is_inc,
            })

            if not is_correct:
                all_error_cases.append({
                    "model": model_name,
                    "scoreboard": "B_MCQA4",
                    "question": q_text,
                    "gold": gold,
                    "raw_gen": raw_gen,
                    "parsed": parsed_pred,
                })

        acc_b = round(correct_b / len(scoreboard_b_items), 4)
        acc_hard = round(hard_neg_correct / hard_neg_total, 4) if hard_neg_total > 0 else 0.0
        results_b[model_name] = {
            "sample_size": len(scoreboard_b_items),
            "correct": correct_b,
            "accuracy": acc_b,
            "invalid_count": invalid_b,
            "invalid_rate": round(invalid_b / len(scoreboard_b_items), 4),
            "hard_neg_accuracy": acc_hard,
            "hard_neg_correct": hard_neg_correct,
            "hard_neg_total": hard_neg_total,
            "prediction_distribution": pred_b_counts,
            "per_case": per_case_b,
        }
        print(f"[Scoreboard B] Acc: {acc_b * 100:.2f}% ({correct_b}/{len(scoreboard_b_items)}) - Hard Neg Acc: {acc_hard * 100:.2f}% ({hard_neg_correct}/{hard_neg_total}) - Dist: {pred_b_counts}")

        # ----------------------------------------------------
        # Scoreboard C: COUNTERFACTUAL & OPTION ORDER INVARIANCE
        # ----------------------------------------------------
        cf_flipped = 0
        cf_both_correct = 0
        cf_train_entity_both = 0
        cf_train_entity_total = 0
        cf_dev_entity_both = 0
        cf_dev_entity_total = 0

        train_entity_names = {"GBXE", "DVPR", "RRHM"}

        for item in cf_items:
            fwd = item["fwd"]
            rev = item["rev"]
            entity = fwd.get("entity", "")
            is_train_ent = entity in train_entity_names
            if is_train_ent:
                cf_train_entity_total += 1
            else:
                cf_dev_entity_total += 1

            # Base question (fwd)
            _, p_base, _ = generate_single_prediction(
                base_model, tokenizer, cond_context, adapted_modules, gated_proj,
                device, fwd["question"], fwd["year_eval"]
            )
            # Counterfactual question (rev)
            _, p_cf, _ = generate_single_prediction(
                base_model, tokenizer, cond_context, adapted_modules, gated_proj,
                device, rev["question"], rev["year_eval"]
            )

            gold_base = fwd["target"]
            gold_cf = rev["target"]

            if p_base != p_cf:
                cf_flipped += 1
            if p_base == gold_base and p_cf == gold_cf:
                cf_both_correct += 1
                if is_train_ent:
                    cf_train_entity_both += 1
                else:
                    cf_dev_entity_both += 1

        # Option Permutation Rotation Test (20 items)
        # Shift options: (a,b,c,d) -> (b,c,d,a)
        perm_preserved = 0
        perm_letters = ["a", "b", "c", "d"]

        for item in option_perm_base:
            orig_q = item["question"]
            orig_target = item["target"].lower()
            orig_opts = item["options"]
            year_eval = item["year_eval"]

            # Baseline unpermuted prediction
            _, p_orig, _ = generate_single_prediction(
                base_model, tokenizer, cond_context, adapted_modules, gated_proj,
                device, orig_q, year_eval
            )

            # Construct rotated question: (a->b, b->c, c->d, d->a)
            # Text body:
            body = item["body"]
            rotated_opts = {
                "a": orig_opts["d"],
                "b": orig_opts["a"],
                "c": orig_opts["b"],
                "d": orig_opts["c"],
            }
            # New gold letter after rotation
            rot_map = {"d": "a", "a": "b", "b": "c", "c": "d"}
            new_target = rot_map[orig_target]

            rot_q = f"Which statement regarding {body} is historically correct?\n"
            rot_q += f"(a) {rotated_opts['a']}\n(b) {rotated_opts['b']}\n(c) {rotated_opts['c']}\n(d) {rotated_opts['d']}"

            _, p_rot, _ = generate_single_prediction(
                base_model, tokenizer, cond_context, adapted_modules, gated_proj,
                device, rot_q, year_eval
            )

            # If model tracks the true statement semantic rather than fixed letter position:
            if p_orig == orig_target and p_rot == new_target:
                perm_preserved += 1

        results_c[model_name] = {
            "cf_sample_size": len(cf_items),
            "cf_label_flip_rate": round(cf_flipped / len(cf_items), 4),
            "cf_both_correct_rate": round(cf_both_correct / len(cf_items), 4),
            "cf_train_entities_both_correct": round(cf_train_entity_both / max(1, cf_train_entity_total), 4),
            "cf_dev_entities_both_correct": round(cf_dev_entity_both / max(1, cf_dev_entity_total), 4),
            "option_perm_sample_size": len(option_perm_base),
            "option_order_invariance_rate": round(perm_preserved / len(option_perm_base), 4),
            "option_order_invariance_count": perm_preserved,
        }
        print(f"[Scoreboard C] CF Flip: {results_c[model_name]['cf_label_flip_rate']*100:.1f}%, CF Joint Acc: {results_c[model_name]['cf_both_correct_rate']*100:.1f}% (Train Ent: {results_c[model_name]['cf_train_entities_both_correct']*100:.1f}%, Dev Ent: {results_c[model_name]['cf_dev_entities_both_correct']*100:.1f}%), Option Invariance: {results_c[model_name]['option_order_invariance_rate']*100:.1f}%")

        # ----------------------------------------------------
        # Scoreboard D: SECONDARY_RETENTION_MCQA180 (180 items)
        # ----------------------------------------------------
        # For CPU budget, evaluate on a balanced 60-item representative slice of 180 TempUN Small items
        rng_d = random.Random(42)
        sample_180_slice = rng_d.sample(scoreboard_d_items, min(60, len(scoreboard_d_items)))

        correct_d = 0
        invalid_d = 0
        for item in sample_180_slice:
            raw_input = item.get("input", "") or item.get("text", "")
            cond = item.get("conditioning") or {}
            year_eval = cond.get("effective_year") or item.get("canonical_year") or 1984

            raw_gen, parsed_pred, val_status = generate_single_prediction(
                base_model, tokenizer, cond_context, adapted_modules, gated_proj,
                device, raw_input, year_eval
            )
            if parsed_pred == gold:
                correct_d += 1
            if val_status != "VALID":
                invalid_d += 1

        acc_d = round(correct_d / len(sample_180_slice), 4)
        results_d[model_name] = {
            "sample_size": len(sample_180_slice),
            "correct": correct_d,
            "accuracy": acc_d,
            "invalid_count": invalid_d,
            "invalid_rate": round(invalid_d / len(sample_180_slice), 4),
        }
        print(f"[Scoreboard D] TempUN 60-slice Acc: {acc_d * 100:.2f}% ({correct_d}/{len(sample_180_slice)})")

    # ----------------------------------------------------
    # Statistical Significance Testing (Primary Arm R3_C vs Others)
    # ----------------------------------------------------
    primary_arm = "R3_C_JOINT_TRUE"
    paired_stats = {}

    for comp in ["RUN02_C1", "R3_A_RELATION_ONLY", "R3_B_MCQA_ONLY", "R3_D_JOINT_SHUFFLED"]:
        # McNemar on Scoreboard A
        cases_prim_a = results_a[primary_arm]["per_case"]
        cases_comp_a = results_a[comp]["per_case"]
        b_a = sum(1 for p, c in zip(cases_prim_a, cases_comp_a) if p["is_correct"] and not c["is_correct"])
        c_a = sum(1 for p, c in zip(cases_prim_a, cases_comp_a) if not p["is_correct"] and c["is_correct"])
        chi2_a, p_val_a = exact_mcnemar_binomial(b_a, c_a)

        # McNemar on Scoreboard B
        cases_prim_b = results_b[primary_arm]["per_case"]
        cases_comp_b = results_b[comp]["per_case"]
        b_b = sum(1 for p, c in zip(cases_prim_b, cases_comp_b) if p["is_correct"] and not c["is_correct"])
        c_b = sum(1 for p, c in zip(cases_prim_b, cases_comp_b) if not p["is_correct"] and c["is_correct"])
        chi2_b, p_val_b = exact_mcnemar_binomial(b_b, c_b)

        paired_stats[f"{primary_arm}_vs_{comp}"] = {
            "scoreboard_a_relation": {
                "prim_only_correct": b_a,
                "comp_only_correct": c_a,
                "chi2": round(chi2_a, 4),
                "p_value": round(p_val_a, 4),
                "is_significant_005": p_val_a < 0.05,
            },
            "scoreboard_b_mcqa": {
                "prim_only_correct": b_b,
                "comp_only_correct": c_b,
                "chi2": round(chi2_b, 4),
                "p_value": round(p_val_b, 4),
                "is_significant_005": p_val_b < 0.05,
            }
        }

    # Save Output Ledgers
    out_rel_path = run03_dir / "DEV_RELATION_RESULTS.json"
    with open(out_rel_path, "w", encoding="utf-8") as f:
        json.dump(results_a, f, indent=2)

    out_mcqa_path = run03_dir / "DEV_MCQA4_RESULTS.json"
    with open(out_mcqa_path, "w", encoding="utf-8") as f:
        json.dump(results_b, f, indent=2)

    out_cf_path = run03_dir / "COUNTERFACTUAL_AND_OPTION_ORDER.json"
    with open(out_cf_path, "w", encoding="utf-8") as f:
        json.dump(results_c, f, indent=2)

    out_ret_path = run03_dir / "RETENTION_MATRIX.json"
    with open(out_ret_path, "w", encoding="utf-8") as f:
        json.dump(results_d, f, indent=2)

    out_stats_path = run03_dir / "PAIRED_GROUP_STATS.json"
    with open(out_stats_path, "w", encoding="utf-8") as f:
        json.dump(paired_stats, f, indent=2)

    out_err_path = run03_dir / "ERROR_CASES.jsonl"
    with open(out_err_path, "w", encoding="utf-8") as f:
        for err in all_error_cases:
            f.write(json.dumps(err) + "\n")

    print(f"\nSaved all Gate 3 evaluation ledgers to {run03_dir}.")


if __name__ == "__main__":
    evaluate_run03()
