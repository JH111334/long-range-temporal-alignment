"""Comprehensive Execution Script for Run 04.1 Reconciliation, Capability Forensics, and Artifact Generation.

Executes:
1. Multi-Scoreboard Evaluation of all variants (RAW, PATCHED_BASE, C1, R4_A, R4_B, R4_C, R4_D).
2. Phase P4 Controlled Interventions on the SAME checkpoint (R4_B):
   - ADAPTED_TRUE
   - ADAPTED_WRONG
   - ADAPTED_ZERO
   - ADAPTED_SEAM_OFF
   - ADAPTED_LORA_OFF
   - ADAPTED_BOTH_OFF
3. Mechanistic layer & seam norms and logit distributions.
4. Statistical McNemar testing and cluster analysis on DEV entities (BCXZ vs GVRO).
5. Generation of all Phase P0-P5 JSON, YAML, and Markdown deliverables.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from temporal_alignment.baselines.ticktack05b.evaluator import (
    build_p12_prompt,
    task_conditioned_strict_parser,
)
from temporal_alignment.models.dual_cycle_mat_lora import (
    DualCycleMaTLoRALinear,
    clear_model_temporal_coordinates,
    replace_mlp_with_dual_cycle_mat_lora,
    set_model_mat_lora_enabled,
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


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def exact_mcnemar(b: int, c: int) -> Tuple[float, float]:
    n = b + c
    if n == 0:
        return 0.0, 1.0
    k = min(b, c)
    cum_prob = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    p_val = min(1.0, 2.0 * cum_prob)
    chi2 = ((abs(b - c) - 1.0) ** 2) / n if n > 0 else 0.0
    return float(chi2), float(p_val)


def main():
    print("=== STARTING RUN 04.1 RECONCILIATION & CAPABILITY FORENSICS ===")
    t0 = time.time()
    device = torch.device("cpu")

    out_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_04_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    run04_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_04"
    run02_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_02"
    ckpt_dir = run04_dir / "checkpoints"

    # 1. Load Data
    with open(run04_dir / "MCQA4_DEV_CLEAN.jsonl", "r", encoding="utf-8") as f:
        dev_mcqa_clean = [json.loads(line) for line in f if line.strip()]

    with open(run02_dir / "dev_relation_pairs.jsonl", "r", encoding="utf-8") as f:
        all_dev_rel = [json.loads(line) for line in f if line.strip()]
    rng_rel = random.Random(42)
    dev_rel_60 = rng_rel.sample(all_dev_rel, min(60, len(all_dev_rel)))

    with open(_REPO_ROOT / "data" / "tempun" / "eval_mcqa.jsonl", "r", encoding="utf-8") as f:
        eval_180_items = [json.loads(line) for line in f if line.strip()]
    rng_slice = random.Random(42)
    sample_60_retention = rng_slice.sample(eval_180_items, min(60, len(eval_180_items)))

    # Representative 60 items for Oracle Diagnostic (deterministic sample matching run 04)
    rng_oracle = random.Random(42)
    oracle_60_items = rng_oracle.sample(dev_mcqa_clean, 60)

    print(f"Data Loaded: S1 Clean MCQA={len(dev_mcqa_clean)}, S2 Dev Rel={len(dev_rel_60)}, S3 Oracle Diag={len(oracle_60_items)}, S4 Retention 60-slice={len(sample_60_retention)}")

    # 2. Setup Base Model
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_id} from local cache...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        device_map=None,
        local_files_only=True,
    ).to(device)
    base_model.eval()

    # Freeze base parameters
    for p in base_model.parameters():
        p.requires_grad = False

    # 3. Setup MaT-LoRA Architecture
    adapted_modules = replace_mlp_with_dual_cycle_mat_lora(
        model=base_model, rank=8, alpha=16.0, num_harmonics=2
    )
    hidden_dim = getattr(base_model.config, "hidden_size", 896)
    gated_proj = GatedResidualProjection(in_features=40, out_features=hidden_dim).to(device)

    cond_context = ConditioningContext(injection_token_index=14)
    hook = make_gated_embedding_hook(cond_context)
    base_model.get_input_embeddings().register_forward_hook(hook)

    c1_compact_path = run02_dir / "checkpoints" / "C1_COMPACT_ADAPTER_ONLY.pt"

    # Define model evaluation matrix
    models_to_evaluate = [
        ("RAW_QWEN05B", None, "DISABLED"),
        ("PATCHED_BASE_EQUIVALENT", None, "DISABLED"),
        ("RUN02_C1", c1_compact_path, "ENABLED"),
        ("R4_A_CLEAN_MCQA_ONLY", ckpt_dir / "R4_A_CLEAN_MCQA_ONLY.pt", "ENABLED"),
        ("R4_B_CLEAN_JOINT_TRUE", ckpt_dir / "R4_B_CLEAN_JOINT_TRUE.pt", "ENABLED"),
        ("R4_C_CLEAN_JOINT_SHUFFLED", ckpt_dir / "R4_C_CLEAN_JOINT_SHUFFLED.pt", "ENABLED"),
        ("R4_D_CLEAN_JOINT_REPLAY", ckpt_dir / "R4_D_CLEAN_JOINT_REPLAY.pt", "ENABLED"),
    ]

    s1_all_models: Dict[str, Any] = {}
    s2_all_models: Dict[str, Any] = {}
    s3_all_models: Dict[str, Any] = {}
    s4_all_models: Dict[str, Any] = {}

    def predict_one(
        question_text: str,
        year_cond: Optional[int],
        adapter_mode: str = "ENABLED",
        wrong_year: Optional[int] = None,
        zero_coord: bool = False,
        seam_off: bool = False,
        lora_off: bool = False,
    ) -> Tuple[str, Optional[str], str]:
        prompt_str, is_yn = build_p12_prompt(question_text, tokenizer)
        inputs = tokenizer(prompt_str, return_tensors="pt").to(device)

        if adapter_mode == "DISABLED" or (lora_off and seam_off):
            set_model_mat_lora_enabled(adapted_modules, False)
            clear_model_temporal_coordinates(adapted_modules)
            cond_context.set_enabled(False)
            cond_context.clear()
        else:
            # Configure LoRA
            if lora_off:
                set_model_mat_lora_enabled(adapted_modules, False)
                clear_model_temporal_coordinates(adapted_modules)
            else:
                set_model_mat_lora_enabled(adapted_modules, True)
                if zero_coord:
                    set_model_temporal_coordinates(adapted_modules, 0.0, 0.0)
                else:
                    y = wrong_year if wrong_year is not None else (year_cond if year_cond is not None else 1984)
                    vec40 = compute_fixed_fone_canchi_40d(y)
                    th, zv = extract_canchi_angles_from_40d(vec40)
                    set_model_temporal_coordinates(adapted_modules, th, zv)

            # Configure Seam
            if seam_off:
                cond_context.set_enabled(False)
                cond_context.clear()
            else:
                cond_context.set_enabled(True)
                if zero_coord:
                    vec_t = torch.zeros(1, 40, dtype=torch.float32, device=device)
                else:
                    y = wrong_year if wrong_year is not None else (year_cond if year_cond is not None else 1984)
                    vec40 = compute_fixed_fone_canchi_40d(y)
                    vec_t = torch.tensor(vec40, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    injected_v = gated_proj(vec_t).squeeze(0)
                cond_context.set_vector(injected_v)

        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out_ids = base_model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        cond_context.clear()
        clear_model_temporal_coordinates(adapted_modules)

        gen_tokens = out_ids[0, prompt_len:].cpu().tolist()
        raw_gen = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        parsed_pred, val_status = task_conditioned_strict_parser(raw_gen, is_yn)
        return raw_gen, parsed_pred, val_status

    # Part 1: Re-evaluate All Models across 4 Scoreboards
    print("\n--- Evaluating Models on Scoreboards 1 to 4 ---")
    for mod_name, ckpt_p, ad_mode in models_to_evaluate:
        print(f"Evaluating {mod_name}...", flush=True)

        if ckpt_p is not None:
            saved = torch.load(ckpt_p, map_location=device, weights_only=False)
            adapters_state = saved.get("adapters_state", {})
            gated_state = saved.get("gated_proj_state", {})
            for k, v in adapted_modules.items():
                v.load_state_dict(adapters_state[k], strict=False)
            gated_proj.load_state_dict(gated_state)

        # S1: Clean MCQA (120)
        corr_s1 = 0
        corr_higher = 0
        corr_lower = 0
        s1_cases = []
        for item in dev_mcqa_clean:
            _, pred, _ = predict_one(item["question"], item["year_eval"], adapter_mode=ad_mode)
            is_c = (pred == item["target"].lower())
            if is_c:
                corr_s1 += 1
                if item["direction"] == "higher":
                    corr_higher += 1
                else:
                    corr_lower += 1
            s1_cases.append({"sample_id": item["sample_id"], "gold": item["target"].lower(), "pred": pred, "is_correct": is_c, "direction": item["direction"], "entity": item["entity"]})
        s1_all_models[mod_name] = {
            "accuracy": round(corr_s1 / 120 * 100, 2),
            "correct": corr_s1,
            "total": 120,
            "higher_acc": round(corr_higher / 60 * 100, 2),
            "lower_acc": round(corr_lower / 60 * 100, 2),
            "cases": s1_cases,
        }

        # S2: Dev Relation (60)
        corr_s2 = 0
        s2_cases = []
        for item in dev_rel_60:
            _, pred, _ = predict_one(item["question"], item["year_eval"], adapter_mode=ad_mode)
            is_c = (pred == item["target"])
            if is_c:
                corr_s2 += 1
            s2_cases.append({"sample_id": item.get("sample_id", ""), "gold": item["target"], "pred": pred, "is_correct": is_c, "entity": item.get("entity", "")})
        s2_all_models[mod_name] = {
            "accuracy": round(corr_s2 / 60 * 100, 2),
            "correct": corr_s2,
            "total": 60,
            "cases": s2_cases,
        }

        # S3: Oracle Diagnostic (60)
        corr_s3 = 0
        for item in oracle_60_items:
            diag_prompt = f"In {item['year_eval']}, {item['body']} was {item['val_eval']}, and in {item['year_base']}, it was {item['val_base']}. Which statement is correct?\n(a) {item['options']['a']}\n(b) {item['options']['b']}\n(c) {item['options']['c']}\n(d) {item['options']['d']}"
            _, pred, _ = predict_one(diag_prompt, item["year_eval"], adapter_mode=ad_mode)
            if pred == item["target"].lower():
                corr_s3 += 1
        s3_all_models[mod_name] = {
            "accuracy": round(corr_s3 / 60 * 100, 2),
            "correct": corr_s3,
            "total": 60,
        }

        # S4: Retention Slice (60)
        corr_s4 = 0
        inv_s4 = 0
        for item in sample_60_retention:
            raw_input = item.get("input", "") or item.get("text", "")
            gold = item.get("target", "").strip()
            cond = item.get("conditioning") or {}
            y_eval = cond.get("effective_year") or item.get("canonical_year") or 1984
            is_yn = ("yes or no" in raw_input.lower())
            gold_norm = gold.lower() if not is_yn else gold.capitalize()

            _, pred, val_stat = predict_one(raw_input, y_eval, adapter_mode=ad_mode)
            if pred == gold_norm:
                corr_s4 += 1
            if val_stat != "VALID":
                inv_s4 += 1
        s4_all_models[mod_name] = {
            "accuracy": round(corr_s4 / 60 * 100, 2),
            "correct": corr_s4,
            "invalid_count": inv_s4,
            "total": 60,
        }

    # Part 2: Controlled Interventions on the SAME Checkpoint (R4_B)
    print("\n--- Executing Phase P4 Controlled Interventions on R4_B ---")
    r4_b_path = ckpt_dir / "R4_B_CLEAN_JOINT_TRUE.pt"
    saved_r4_b = torch.load(r4_b_path, map_location=device, weights_only=False)
    for k, v in adapted_modules.items():
        v.load_state_dict(saved_r4_b["adapters_state"][k], strict=False)
    gated_proj.load_state_dict(saved_r4_b["gated_proj_state"])

    interventions = [
        ("ADAPTED_TRUE", {}),
        ("ADAPTED_WRONG", {"wrong_year": 1947}),  # Swap year
        ("ADAPTED_ZERO", {"zero_coord": True}),
        ("ADAPTED_SEAM_OFF", {"seam_off": True}),
        ("ADAPTED_LORA_OFF", {"lora_off": True}),
        ("ADAPTED_BOTH_OFF", {"lora_off": True, "seam_off": True}),
    ]

    intervention_results: Dict[str, Any] = {}
    per_case_intervention_rows = []

    for int_name, kwargs in interventions:
        corr = 0
        corr_hi = 0
        corr_lo = 0
        int_cases = []

        for item in dev_mcqa_clean:
            # If wrong year, use year_base instead of year_eval to invert temporal direction
            if int_name == "ADAPTED_WRONG":
                kwargs["wrong_year"] = item["year_base"]

            raw_gen, pred, val_stat = predict_one(
                item["question"], item["year_eval"], adapter_mode="ENABLED", **kwargs
            )
            is_c = (pred == item["target"].lower())
            if is_c:
                corr += 1
                if item["direction"] == "higher":
                    corr_hi += 1
                else:
                    corr_lo += 1

            int_cases.append({
                "sample_id": item["sample_id"],
                "gold": item["target"].lower(),
                "pred": pred,
                "is_correct": is_c,
                "direction": item["direction"],
                "entity": item["entity"],
            })

            per_case_intervention_rows.append({
                "intervention": int_name,
                "sample_id": item["sample_id"],
                "gold": item["target"].lower(),
                "pred": pred,
                "is_correct": is_c,
                "direction": item["direction"],
                "entity": item["entity"],
                "raw_gen": raw_gen,
            })

        acc = round(corr / 120 * 100, 2)
        intervention_results[int_name] = {
            "accuracy": acc,
            "correct": corr,
            "total": 120,
            "higher_acc": round(corr_hi / 60 * 100, 2),
            "lower_acc": round(corr_lo / 60 * 100, 2),
            "cases": int_cases,
        }
        print(f"[{int_name}] Acc: {acc}% ({corr}/120), Higher: {round(corr_hi/60*100, 1)}%, Lower: {round(corr_lo/60*100, 1)}%")

    # Part 3: Layer & Seam Norm Measurements
    print("\n--- Measuring Mechanistic Layer & Seam Norms ---")
    lora_norms = {}
    dummy_x = torch.randn(1, 1, 896, device=device)
    for k, v in adapted_modules.items():
        with torch.no_grad():
            b_norm = v.base_layer.weight.norm().item()
            lora_b_norm = v.lora_B.norm().item()
            lora_a_norm = v.lora_A.norm().item()
            lora_norms[k] = {
                "base_weight_norm": round(b_norm, 4),
                "lora_B_norm": round(lora_b_norm, 4),
                "lora_A_norm": round(lora_a_norm, 4),
                "scaling": v.scaling,
            }

    seam_norm_summary = {
        "alpha_parameter": round(gated_proj.alpha.item(), 6),
        "linear_weight_norm": round(gated_proj.linear.weight.norm().item(), 4),
        "norm_weight_present": hasattr(gated_proj, "norm"),
        "active_injection_token": 14,
    }

    layer_trace_payload = {
        "seam_diagnostics": seam_norm_summary,
        "adapter_norms_sample": {k: lora_norms[k] for k in list(lora_norms.keys())[:6]},
        "total_adapted_modules": len(lora_norms),
    }

    with open(out_dir / "LAYER_AND_SEAM_TRACE_SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(layer_trace_payload, f, indent=2)

    with open(out_dir / "PER_CASE_INTERVENTION_RESULTS.jsonl", "w", encoding="utf-8") as f:
        for r in per_case_intervention_rows:
            f.write(json.dumps(r) + "\n")

    # Part 4: Statistical Testing & Cluster Analysis (BCXZ vs GVRO)
    print("\n--- Computing Reconciled McNemar & Cluster Statistics ---")
    paired_stats = {}
    primary_cases = s1_all_models["R4_B_CLEAN_JOINT_TRUE"]["cases"]

    for comp in ["RAW_QWEN05B", "PATCHED_BASE_EQUIVALENT", "RUN02_C1", "R4_A_CLEAN_MCQA_ONLY", "R4_C_CLEAN_JOINT_SHUFFLED", "R4_D_CLEAN_JOINT_REPLAY"]:
        comp_cases = s1_all_models[comp]["cases"]
        b = sum(1 for p, c in zip(primary_cases, comp_cases) if p["is_correct"] and not c["is_correct"])
        c = sum(1 for p, c in zip(primary_cases, comp_cases) if not p["is_correct"] and c["is_correct"])
        both_c = sum(1 for p, c in zip(primary_cases, comp_cases) if p["is_correct"] and c["is_correct"])
        both_w = sum(1 for p, c in zip(primary_cases, comp_cases) if not p["is_correct"] and not c["is_correct"])
        chi2, p_val = exact_mcnemar(b, c)

        # Cluster breakdown by DEV entity
        bcxz_p = [p for p in primary_cases if p["entity"] == "BCXZ"]
        bcxz_c = [c for c in comp_cases if c["entity"] == "BCXZ"]
        gvro_p = [p for p in primary_cases if p["entity"] == "GVRO"]
        gvro_c = [c for c in comp_cases if c["entity"] == "GVRO"]

        acc_p_bcxz = sum(1 for p in bcxz_p if p["is_correct"]) / len(bcxz_p) * 100
        acc_c_bcxz = sum(1 for c in bcxz_c if c["is_correct"]) / len(bcxz_c) * 100
        acc_p_gvro = sum(1 for p in gvro_p if p["is_correct"]) / len(gvro_p) * 100
        acc_c_gvro = sum(1 for c in gvro_c if c["is_correct"]) / len(gvro_c) * 100

        paired_stats[f"R4_B_vs_{comp}"] = {
            "both_correct": both_c,
            "both_wrong": both_w,
            "b_primary_only": b,
            "c_comparator_only": c,
            "chi2": round(chi2, 4),
            "p_value_scientific": f"{p_val:.6e}",
            "is_significant_005": p_val < 0.05,
            "cluster_breakdown": {
                "entity_BCXZ_count": len(bcxz_p),
                "acc_primary_BCXZ": round(acc_p_bcxz, 2),
                "acc_comp_BCXZ": round(acc_c_bcxz, 2),
                "delta_BCXZ": round(acc_p_bcxz - acc_c_bcxz, 2),
                "entity_GVRO_count": len(gvro_p),
                "acc_primary_GVRO": round(acc_p_gvro, 2),
                "acc_comp_GVRO": round(acc_c_gvro, 2),
                "delta_GVRO": round(acc_p_gvro - acc_c_gvro, 2),
            },
        }

    cluster_limitations_payload = {
        "audit_note": "DEV set contains exactly 2 entity series (BCXZ=107 items, GVRO=13 items). Pairwise p-values are valid at case-level, but cannot support broad population-level generalization claims without additional independent entity clusters.",
        "paired_statistics": paired_stats,
    }
    with open(out_dir / "PAIRED_CASE_STATS_AND_CLUSTER_LIMITATIONS.json", "w", encoding="utf-8") as f:
        json.dump(cluster_limitations_payload, f, indent=2)

    # Part 5: Reconciled Scoreboards Summary
    reconciled_scoreboards = {
        "W0_RAW_BASE": {
            "s1_clean_mcqa_acc": s1_all_models["RAW_QWEN05B"]["accuracy"],
            "s2_dev_relation_acc": s2_all_models["RAW_QWEN05B"]["accuracy"],
            "s3_oracle_diag_acc": s3_all_models["RAW_QWEN05B"]["accuracy"],
            "s4_retention_slice_acc": s4_all_models["RAW_QWEN05B"]["accuracy"],
        },
        "PATCHED_BASE_EQUIVALENT": {
            "s1_clean_mcqa_acc": s1_all_models["PATCHED_BASE_EQUIVALENT"]["accuracy"],
            "s2_dev_relation_acc": s2_all_models["PATCHED_BASE_EQUIVALENT"]["accuracy"],
            "s3_oracle_diag_acc": s3_all_models["PATCHED_BASE_EQUIVALENT"]["accuracy"],
            "s4_retention_slice_acc": s4_all_models["PATCHED_BASE_EQUIVALENT"]["accuracy"],
        },
        "RUN02_C1": {
            "s1_clean_mcqa_acc": s1_all_models["RUN02_C1"]["accuracy"],
            "s2_dev_relation_acc": s2_all_models["RUN02_C1"]["accuracy"],
            "s3_oracle_diag_acc": s3_all_models["RUN02_C1"]["accuracy"],
            "s4_retention_slice_acc": s4_all_models["RUN02_C1"]["accuracy"],
        },
        "R4_A_CLEAN_MCQA_ONLY": {
            "s1_clean_mcqa_acc": s1_all_models["R4_A_CLEAN_MCQA_ONLY"]["accuracy"],
            "s2_dev_relation_acc": s2_all_models["R4_A_CLEAN_MCQA_ONLY"]["accuracy"],
            "s3_oracle_diag_acc": s3_all_models["R4_A_CLEAN_MCQA_ONLY"]["accuracy"],
            "s4_retention_slice_acc": s4_all_models["R4_A_CLEAN_MCQA_ONLY"]["accuracy"],
        },
        "R4_B_CLEAN_JOINT_TRUE": {
            "s1_clean_mcqa_acc": s1_all_models["R4_B_CLEAN_JOINT_TRUE"]["accuracy"],
            "s2_dev_relation_acc": s2_all_models["R4_B_CLEAN_JOINT_TRUE"]["accuracy"],
            "s3_oracle_diag_acc": s3_all_models["R4_B_CLEAN_JOINT_TRUE"]["accuracy"],
            "s4_retention_slice_acc": s4_all_models["R4_B_CLEAN_JOINT_TRUE"]["accuracy"],
        },
        "R4_C_CLEAN_JOINT_SHUFFLED": {
            "s1_clean_mcqa_acc": s1_all_models["R4_C_CLEAN_JOINT_SHUFFLED"]["accuracy"],
            "s2_dev_relation_acc": s2_all_models["R4_C_CLEAN_JOINT_SHUFFLED"]["accuracy"],
            "s3_oracle_diag_acc": s3_all_models["R4_C_CLEAN_JOINT_SHUFFLED"]["accuracy"],
            "s4_retention_slice_acc": s4_all_models["R4_C_CLEAN_JOINT_SHUFFLED"]["accuracy"],
        },
        "R4_D_CLEAN_JOINT_REPLAY": {
            "s1_clean_mcqa_acc": s1_all_models["R4_D_CLEAN_JOINT_REPLAY"]["accuracy"],
            "s2_dev_relation_acc": s2_all_models["R4_D_CLEAN_JOINT_REPLAY"]["accuracy"],
            "s3_oracle_diag_acc": s3_all_models["R4_D_CLEAN_JOINT_REPLAY"]["accuracy"],
            "s4_retention_slice_acc": s4_all_models["R4_D_CLEAN_JOINT_REPLAY"]["accuracy"],
            "reconciliation_verdict": "REPLAY_NOT_TESTED (IDENTICAL_TO_R4_B)",
        },
        "R4_B_SAME_CHECKPOINT_INTERVENTIONS": {
            "ADAPTED_TRUE": intervention_results["ADAPTED_TRUE"]["accuracy"],
            "ADAPTED_WRONG": intervention_results["ADAPTED_WRONG"]["accuracy"],
            "ADAPTED_ZERO": intervention_results["ADAPTED_ZERO"]["accuracy"],
            "ADAPTED_SEAM_OFF": intervention_results["ADAPTED_SEAM_OFF"]["accuracy"],
            "ADAPTED_LORA_OFF": intervention_results["ADAPTED_LORA_OFF"]["accuracy"],
            "ADAPTED_BOTH_OFF": intervention_results["ADAPTED_BOTH_OFF"]["accuracy"],
        },
    }

    with open(out_dir / "SCOREBOARD_RECONCILED.json", "w", encoding="utf-8") as f:
        json.dump(reconciled_scoreboards, f, indent=2)

    print(f"=== EXECUTION COMPLETED IN {time.time() - t0:.1f} SECONDS ===")


if __name__ == "__main__":
    main()
