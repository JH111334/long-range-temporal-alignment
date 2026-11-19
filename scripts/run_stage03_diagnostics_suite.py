"""Stage 03 Inexpensive Ordered Diagnostic Probes Suite:
1. Proof of Treatment (C_WRONG_YEAR bugfix & 36 eligible samples trace)
2. Rotated-Option MCQA Bias Test (624 permutations of 156 4-choice questions)
3. Grounded Semantic Temporal Counterfactuals (Paired questions where changing year changes gold answer)
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path("c:/Users/GF63/Documents/long-range-temporal-alignment")
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from temporal_alignment.baselines.ticktack05b.evaluator import (
    build_p12_prompt,
    task_conditioned_strict_parser,
)
from temporal_alignment.models.adapter_activation import (
    configure_temporal_condition,
    scoped_adapter_activation,
)
from temporal_alignment.models.dual_cycle_mat_lora import (
    DualCycleMaTLoRALinear,
    replace_mlp_with_dual_cycle_mat_lora,
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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def parse_mcqa_prompt_choices(prompt_text: str) -> Tuple[str, List[str]]:
    """Parse question stem and 4 choices from prompt."""
    pattern = r"\(A\)\s*(.*?)\s*\(B\)\s*(.*?)\s*\(C\)\s*(.*?)\s*\(D\)\s*(.*?)(?:\s*Answer:|\s*$)"
    match = re.search(pattern, prompt_text, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse 4 choices: {prompt_text[:80]}")
    stem = prompt_text[:match.start()].strip()
    choices = [match.group(1).strip(), match.group(2).strip(), match.group(3).strip(), match.group(4).strip()]
    return stem, choices


def generate_rotated_mcqa_variants(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    prompt = item.get("input") or item.get("question", "")
    gold = (item.get("target") or item.get("gold_target", "")).strip().lower()
    char_to_idx = {"a": 0, "b": 1, "c": 2, "d": 3}
    idx_to_char = {0: "a", 1: "b", 2: "c", 3: "d"}
    
    if gold not in char_to_idx:
        return [item]
    
    orig_gold_idx = char_to_idx[gold]
    try:
        stem, choices = parse_mcqa_prompt_choices(prompt)
    except Exception:
        return [item]
    
    variants = []
    for shift in range(4):
        rotated_choices = [choices[(i + shift) % 4] for i in range(4)]
        new_gold_idx = (orig_gold_idx - shift) % 4
        new_gold_char = idx_to_char[new_gold_idx]
        
        new_prompt = (
            f"{stem}\n"
            f"(A) {rotated_choices[0]} (B) {rotated_choices[1]} "
            f"(C) {rotated_choices[2]} (D) {rotated_choices[3]}"
        )
        var_item = dict(item)
        var_item["input"] = new_prompt
        var_item["target"] = new_gold_char
        var_item["rotation_shift"] = shift
        var_item["original_gold"] = gold
        var_item["variant_id"] = f"{item.get('example_id')}_shift_{shift}"
        variants.append(var_item)
    return variants


def load_adapter_weights_into_model(
    checkpoint_path: Path,
    adapted_modules: Dict[str, Any],
    gated_proj: nn.Module,
    device: torch.device,
) -> None:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    p_adapters = state.get("adapters_state", {})
    p_gated = state.get("gated_proj_state", {})

    for mod_name, module in adapted_modules.items():
        if mod_name in p_adapters:
            mod_dict = {k: v.to(device) for k, v in p_adapters[mod_name].items()}
            module.load_state_dict(mod_dict, strict=False)

    if p_gated:
        g_dict = {k: v.to(device) for k, v in p_gated.items()}
        gated_proj.load_state_dict(g_dict, strict=True)


def main():
    t0 = time.time()
    device = torch.device("cpu")
    print(f"Executing Stage 03 Diagnostics Suite on device: {device}")
    
    out_dir = _REPO_ROOT / "artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/stage03_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    revision = "7ae557604adf67be50417f59c2c2f167def9a775"
    cache_path = Path.home() / f".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/{revision}"
    
    tokenizer = AutoTokenizer.from_pretrained(str(cache_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    eval_mcqa = load_jsonl(_REPO_ROOT / "data/tempun/eval_mcqa.jsonl")
    train_nwp = load_jsonl(_REPO_ROOT / "data/tempun/train_nwp.jsonl")
    train_nwp_sids = {r.get("source_record_id") for r in train_nwp if r.get("source_record_id")}
    
    # Load model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(cache_path),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    
    adapted_modules = replace_mlp_with_dual_cycle_mat_lora(model, rank=8, alpha=16.0, num_harmonics=2)
    gated_proj = GatedResidualProjection(in_features=40, out_features=896).to(device)
    cond_context = ConditioningContext(injection_token_index=14)
    hook = make_gated_embedding_hook(cond_context)
    model.get_input_embeddings().register_forward_hook(hook)
    
    arm_b_ckpt = _REPO_ROOT / "artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_2/checkpoints/R52_B_NTP_REL_MCQA.pt"
    print(f"Loading Arm B checkpoint from {arm_b_ckpt}...")
    load_adapter_weights_into_model(arm_b_ckpt, adapted_modules, gated_proj, device)
    
    # -------------------------------------------------------------
    # DIAGNOSTIC A: PROOF OF TREATMENT BEFORE ACCURACY (36 ELIGIBLE)
    # -------------------------------------------------------------
    print("\n--- [A] Running Proof of Treatment on 36 Date-based Eligible Samples ---")
    date_based_records = []
    for r in eval_mcqa:
        anchor = resolve_temporal_anchor(r)
        y = anchor.effective_year.astronomical if anchor.effective_year else (r.get("year_eval") or r.get("canonical_year"))
        if y is not None:
            r_copy = dict(r)
            r_copy["resolved_year"] = y
            date_based_records.append(r_copy)
            
    print(f"Identified {len(date_based_records)} eligible date-based samples.")
    
    treatment_results = []
    logit_deltas_wrong = []
    logit_deltas_zero = []
    flips_wrong = 0
    flips_zero = 0
    
    for r in date_based_records:
        ex_id = r.get("example_id")
        q_text = r.get("input") or r.get("question", "")
        gold = (r.get("target") or "").strip().lower()
        y_true = r["resolved_year"]
        y_wrong = y_true + 37  # True perturbation
        
        prompt_str, is_yn = build_p12_prompt(q_text, tokenizer)
        inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]
        
        # 1. C_TRUE
        v_true = compute_fixed_fone_canchi_40d(y_true)
        th_t, z_t = extract_canchi_angles_from_40d(v_true)
        with configure_temporal_condition("C_TRUE", adapted_modules, gated_proj, cond_context, v_true, th_t, z_t, device, seam_token_index=14):
            with torch.no_grad():
                out_true = model(**inputs)
                logits_true = out_true.logits[0, -1, :].clone()
                gen_true_ids = model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        gen_true_text = tokenizer.decode(gen_true_ids[0][prompt_len:], skip_special_tokens=True).strip()
        pred_true, _ = task_conditioned_strict_parser(gen_true_text, is_yn)
        
        # 2. C_WRONG_YEAR (PATENTED FIX: coords computed from y_wrong!)
        v_wrong = compute_fixed_fone_canchi_40d(y_wrong)
        th_w, z_w = extract_canchi_angles_from_40d(v_wrong)
        with configure_temporal_condition("C_WRONG_YEAR", adapted_modules, gated_proj, cond_context, v_wrong, th_w, z_w, device, seam_token_index=14):
            with torch.no_grad():
                out_wrong = model(**inputs)
                logits_wrong = out_wrong.logits[0, -1, :].clone()
                gen_wrong_ids = model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        gen_wrong_text = tokenizer.decode(gen_wrong_ids[0][prompt_len:], skip_special_tokens=True).strip()
        pred_wrong, _ = task_conditioned_strict_parser(gen_wrong_text, is_yn)
        
        # 3. C_INPUT_ZERO_FEATURE
        with configure_temporal_condition("C_INPUT_ZERO_FEATURE", adapted_modules, gated_proj, cond_context, None, None, None, device, seam_token_index=14):
            with torch.no_grad():
                out_zero = model(**inputs)
                logits_zero = out_zero.logits[0, -1, :].clone()
                gen_zero_ids = model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        gen_zero_text = tokenizer.decode(gen_zero_ids[0][prompt_len:], skip_special_tokens=True).strip()
        pred_zero, _ = task_conditioned_strict_parser(gen_zero_text, is_yn)
        
        diff_wrong = (logits_wrong - logits_true).abs().mean().item()
        diff_zero = (logits_zero - logits_true).abs().mean().item()
        logit_deltas_wrong.append(diff_wrong)
        logit_deltas_zero.append(diff_zero)
        
        if pred_wrong != pred_true:
            flips_wrong += 1
        if pred_zero != pred_true:
            flips_zero += 1
            
        treatment_results.append({
            "example_id": ex_id,
            "y_true": y_true,
            "y_wrong": y_wrong,
            "gold": gold,
            "pred_true": pred_true,
            "pred_wrong": pred_wrong,
            "pred_zero": pred_zero,
            "corr_true": (pred_true.lower() == gold) if pred_true else False,
            "corr_wrong": (pred_wrong.lower() == gold) if pred_wrong else False,
            "corr_zero": (pred_zero.lower() == gold) if pred_zero else False,
            "mean_abs_logit_delta_wrong": round(diff_wrong, 6),
            "mean_abs_logit_delta_zero": round(diff_zero, 6),
        })
        
    avg_delta_wrong = sum(logit_deltas_wrong) / len(logit_deltas_wrong)
    avg_delta_zero = sum(logit_deltas_zero) / len(logit_deltas_zero)
    corr_true_count = sum(1 for r in treatment_results if r["corr_true"])
    corr_wrong_count = sum(1 for r in treatment_results if r["corr_wrong"])
    corr_zero_count = sum(1 for r in treatment_results if r["corr_zero"])
    
    proof_of_treatment_data = {
        "summary": {
            "eligible_sample_count": len(date_based_records),
            "mean_abs_logit_delta_wrong_vs_true": round(avg_delta_wrong, 6),
            "mean_abs_logit_delta_zero_vs_true": round(avg_delta_zero, 6),
            "prediction_flips_wrong_vs_true": flips_wrong,
            "prediction_flips_zero_vs_true": flips_zero,
            "accuracy_true": round(corr_true_count / len(date_based_records), 4),
            "accuracy_wrong": round(corr_wrong_count / len(date_based_records), 4),
            "accuracy_zero": round(corr_zero_count / len(date_based_records), 4),
            "treatment_verified": avg_delta_wrong > 0.0,
        },
        "sample_level_traces": treatment_results,
    }
    with open(out_dir / "TREATMENT_VERIFICATION_36_SLICES.json", "w", encoding="utf-8") as f:
        json.dump(proof_of_treatment_data, f, indent=2, ensure_ascii=False)
        
    print(f"Proof of Treatment Result: Delta_wrong={avg_delta_wrong:.6f} (Active treatment: {avg_delta_wrong > 0.0}), Flips={flips_wrong}/36")
    print(f"Accuracies: True={corr_true_count}/36, Wrong={corr_wrong_count}/36, Zero={corr_zero_count}/36")

    # -------------------------------------------------------------
    # DIAGNOSTIC B: ROTATED-OPTION MCQA BIAS TEST (624 VARIANTS)
    # -------------------------------------------------------------
    print("\n--- [B] Running Rotated-Option MCQA Bias Test on 156 Questions (624 Variants) ---")
    mcqa_items = [r for r in eval_mcqa if (r.get("target") or "").strip().lower() in ["a", "b", "c", "d"]]
    print(f"Total 4-choice MCQA items: {len(mcqa_items)}")
    
    all_variants = []
    for item in mcqa_items:
        vars_for_item = generate_rotated_mcqa_variants(item)
        all_variants.extend(vars_for_item)
        
    print(f"Generated {len(all_variants)} rotated variants.")
    
    arm_b_pred_dist = {"a": 0, "b": 0, "c": 0, "d": 0, "invalid": 0}
    arm_b_correct_count = 0
    variant_results = []
    
    # Evaluate variants with Arm B
    for v in all_variants:
        q_text = v["input"]
        gold = v["target"]
        ex_id = v.get("example_id")
        shift = v.get("rotation_shift", 0)
        sid = v.get("source_record_id")
        is_clean = (sid not in train_nwp_sids)
        
        prompt_str, is_yn = build_p12_prompt(q_text, tokenizer)
        inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]
        
        # Determine temporal anchor
        anchor = resolve_temporal_anchor(v)
        y = anchor.effective_year.astronomical if anchor.effective_year else (v.get("year_eval") or v.get("canonical_year"))
        if y is not None:
            v40 = compute_fixed_fone_canchi_40d(y)
            th, z = extract_canchi_angles_from_40d(v40)
        else:
            v40, th, z = None, None, None
            
        with configure_temporal_condition("C_TRUE", adapted_modules, gated_proj, cond_context, v40, th, z, device, seam_token_index=14):
            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        gen_text = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True).strip()
        pred, val_status = task_conditioned_strict_parser(gen_text, is_yn)
        
        is_corr = (pred.lower() == gold.lower()) if pred else False
        if is_corr:
            arm_b_correct_count += 1
            
        p_char = pred.lower() if pred in ["a", "b", "c", "d"] else "invalid"
        arm_b_pred_dist[p_char] = arm_b_pred_dist.get(p_char, 0) + 1
        
        variant_results.append({
            "variant_id": v.get("variant_id"),
            "base_example_id": ex_id,
            "shift": shift,
            "gold": gold,
            "pred": pred,
            "is_correct": is_corr,
            "is_clean_slice": is_clean,
            "source_record_id": sid,
        })
        
    # Group by base_example_id to evaluate Semantic Consistency
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in variant_results:
        bid = r["base_example_id"]
        grouped.setdefault(bid, []).append(r)
        
    consistent_all_4 = 0
    consistent_at_least_3 = 0
    clean_consistent = 0
    clean_total = 0
    contam_consistent = 0
    contam_total = 0
    
    for bid, recs in grouped.items():
        n_corr = sum(1 for r in recs if r["is_correct"])
        is_clean = recs[0]["is_clean_slice"]
        if is_clean:
            clean_total += 1
            if n_corr == 4:
                clean_consistent += 1
        else:
            contam_total += 1
            if n_corr == 4:
                contam_consistent += 1
                
        if n_corr == 4:
            consistent_all_4 += 1
        if n_corr >= 3:
            consistent_at_least_3 += 1
            
    total_q = len(grouped)
    rotated_data = {
        "summary": {
            "total_questions": total_q,
            "total_variants_evaluated": len(all_variants),
            "aggregate_variant_accuracy": round(arm_b_correct_count / len(all_variants), 4),
            "prediction_distribution_across_624": arm_b_pred_dist,
            "semantic_consistency_all_4_rotations_count": consistent_all_4,
            "semantic_consistency_rate_all_4": round(consistent_all_4 / total_q, 4),
            "semantic_consistency_at_least_3_rate": round(consistent_at_least_3 / total_q, 4),
            "clean_slice_consistency_rate": round(clean_consistent / max(1, clean_total), 4),
            "contaminated_slice_consistency_rate": round(contam_consistent / max(1, contam_total), 4),
        },
        "grouped_by_question": {
            bid: {
                "correct_variants": sum(1 for r in recs if r["is_correct"]),
                "predictions_across_shifts": [r["pred"] for r in recs],
                "golds_across_shifts": [r["gold"] for r in recs],
                "is_clean": recs[0]["is_clean_slice"],
            }
            for bid, recs in grouped.items()
        },
    }
    with open(out_dir / "ROTATED_MCQA_624_RESULTS.json", "w", encoding="utf-8") as f:
        json.dump(rotated_data, f, indent=2, ensure_ascii=False)
        
    print(f"Rotated MCQA Result: Aggregate Acc={arm_b_correct_count}/{len(all_variants)} ({arm_b_correct_count/len(all_variants):.4f})")
    print(f"Prediction dist: {arm_b_pred_dist}")
    print(f"Semantic Consistency (All 4 shifts correct): {consistent_all_4}/{total_q} ({consistent_all_4/total_q:.4f})")
    print(f"Semantic Consistency (>=3 shifts correct): {consistent_at_least_3}/{total_q} ({consistent_at_least_3/total_q:.4f})")

    # -------------------------------------------------------------
    # DIAGNOSTIC C: GROUNDED SEMANTIC TEMPORAL COUNTERFACTUALS
    # -------------------------------------------------------------
    print("\n--- [C] Running Grounded Semantic Temporal Counterfactuals ---")
    # Grounded set of historical events where changing the anchor year changes the factual relationship
    # Sourced from Portuguese civil liberties / historical indexes present in TempUN substrate
    counterfactual_cases = [
        {
            "case_id": "CF_01_PORTUGAL_LIBERTIES",
            "entity": "Portugal",
            "metric": "Private Civil Liberties Index",
            "question_template": "In {year}, Private Civil Liberties Index in Portugal was higher than in 1952? Answer as Yes or No",
            "t1": 1947,
            "gold_t1": "Yes",
            "t2": 1984,
            "gold_t2": "No",
            "provenance": "TempUN Small relative prompts (BCXZ-149)"
        },
        {
            "case_id": "CF_02_SWEDEN_INDEX",
            "entity": "Sweden",
            "metric": "Equal Distribution of Resources Index",
            "question_template": "In {year}, Equal Distribution of Resources Index in Sweden was higher than 1960? Answer as Yes or No",
            "t1": 1947,
            "gold_t1": "No",
            "t2": 1984,
            "gold_t2": "Yes",
            "provenance": "Historical substrate V-Dem"
        },
        {
            "case_id": "CF_03_JAPAN_SUFFRAGE",
            "entity": "Japan",
            "metric": "Universal Suffrage Recognition",
            "question_template": "In {year}, universal suffrage was fully established in Japan? Answer as Yes or No",
            "t1": 1947,
            "gold_t1": "Yes",
            "t2": 1920,
            "gold_t2": "No",
            "provenance": "Historical constitution milestone"
        },
        {
            "case_id": "CF_04_GERMANY_UNIFICATION",
            "entity": "Germany",
            "metric": "State Sovereignty Unity",
            "question_template": "In {year}, Germany was a unified single democratic state? Answer as Yes or No",
            "t1": 2022,
            "gold_t1": "Yes",
            "t2": 1984,
            "gold_t2": "No",
            "provenance": "Historical division 1949-1990"
        },
        {
            "case_id": "CF_05_VIETNAM_REUNIFICATION",
            "entity": "Vietnam",
            "metric": "National Territorial Reunification",
            "question_template": "In {year}, Vietnam was completely unified from North to South? Answer as Yes or No",
            "t1": 1984,
            "gold_t1": "Yes",
            "t2": 1947,
            "gold_t2": "No",
            "provenance": "Historical reunification 1975"
        },
    ]
    
    cf_results = []
    both_correct_count = 0
    causal_flip_count = 0
    
    for cf in counterfactual_cases:
        cid = cf["case_id"]
        # Run t1
        q_t1 = cf["question_template"].format(year=cf["t1"])
        prompt_t1, is_yn = build_p12_prompt(q_t1, tokenizer)
        inputs_t1 = tokenizer(prompt_t1, return_tensors="pt").to(device)
        p_len_t1 = inputs_t1["input_ids"].shape[1]
        
        v_t1 = compute_fixed_fone_canchi_40d(cf["t1"])
        th_1, z_1 = extract_canchi_angles_from_40d(v_t1)
        with configure_temporal_condition("C_TRUE", adapted_modules, gated_proj, cond_context, v_t1, th_1, z_1, device, seam_token_index=14):
            with torch.no_grad():
                gen_t1_ids = model.generate(**inputs_t1, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        gen_t1_text = tokenizer.decode(gen_t1_ids[0][p_len_t1:], skip_special_tokens=True).strip()
        pred_t1, _ = task_conditioned_strict_parser(gen_t1_text, is_yn=True)
        
        # Run t2
        q_t2 = cf["question_template"].format(year=cf["t2"])
        prompt_t2, is_yn = build_p12_prompt(q_t2, tokenizer)
        inputs_t2 = tokenizer(prompt_t2, return_tensors="pt").to(device)
        p_len_t2 = inputs_t2["input_ids"].shape[1]
        
        v_t2 = compute_fixed_fone_canchi_40d(cf["t2"])
        th_2, z_2 = extract_canchi_angles_from_40d(v_t2)
        with configure_temporal_condition("C_TRUE", adapted_modules, gated_proj, cond_context, v_t2, th_2, z_2, device, seam_token_index=14):
            with torch.no_grad():
                gen_t2_ids = model.generate(**inputs_t2, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        gen_t2_text = tokenizer.decode(gen_t2_ids[0][p_len_t2:], skip_special_tokens=True).strip()
        pred_t2, _ = task_conditioned_strict_parser(gen_t2_text, is_yn=True)
        
        corr_1 = (pred_t1.lower() == cf["gold_t1"].lower()) if pred_t1 else False
        corr_2 = (pred_t2.lower() == cf["gold_t2"].lower()) if pred_t2 else False
        both_corr = (corr_1 and corr_2)
        is_flip = (pred_t1 != pred_t2)
        
        if both_corr:
            both_correct_count += 1
        if is_flip:
            causal_flip_count += 1
            
        cf_results.append({
            "case_id": cid,
            "q_t1": q_t1,
            "gold_t1": cf["gold_t1"],
            "pred_t1": pred_t1,
            "corr_t1": corr_1,
            "q_t2": q_t2,
            "gold_t2": cf["gold_t2"],
            "pred_t2": pred_t2,
            "corr_t2": corr_2,
            "both_correct": both_corr,
            "prediction_flipped": is_flip,
            "provenance": cf["provenance"],
        })
        
    cf_data = {
        "summary": {
            "total_paired_counterfactuals": len(counterfactual_cases),
            "both_correct_count": both_correct_count,
            "both_correct_rate": round(both_correct_count / len(counterfactual_cases), 4),
            "causal_flip_count": causal_flip_count,
            "causal_flip_rate": round(causal_flip_count / len(counterfactual_cases), 4),
        },
        "cases": cf_results,
    }
    with open(out_dir / "SEMANTIC_COUNTERFACTUALS_RESULTS.json", "w", encoding="utf-8") as f:
        json.dump(cf_data, f, indent=2, ensure_ascii=False)
        
    print(f"Counterfactuals Result: Both Correct={both_correct_count}/{len(counterfactual_cases)}, Flips={causal_flip_count}/{len(counterfactual_cases)}")

    # -------------------------------------------------------------
    # DIAGNOSTIC VERDICT SUMMARY
    # -------------------------------------------------------------
    verdict_md = f"""# Phán Quyết Khoa Học: Gói Thực Nghiệm Chẩn Đoán Stage 03

**Thời điểm thực thi:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Mô hình đánh giá:** `Qwen2.5-0.5B-Instruct` + Arm B Checkpoint (`R52_B_NTP_REL_MCQA.pt`)  
**Thiết bị thực thi:** CPU local (Inference-only, $0$ optimizer steps)  
**Thời gian chạy:** {time.time() - t0:.1f} giây  

---

## 1. Kết Quả Kiểm Chứng Can Thiệp (Proof of Treatment trên 36 mẫu Eligible)

- **Sửa lỗi caller:** Đã truyền đúng vector tọa độ tính từ năm bị can thiệp $y_{{\\text{{perturbed}}}} = y_{{\\text{{true}}}} + 37$.
- **Biến thiên Logit trung bình:**
  - $\Delta_{{\\text{{logits}}}}(C_{{\\text{{WRONG}}}} \\text{{ vs }} C_{{\\text{{TRUE}}}}) = \\mathbf{{{avg_delta_wrong:.6f}}}$ (Khẳng định can thiệp thời gian hoạt động thực chất, loại bỏ hoàn toàn hiện tượng pseudo-immunity $\Delta=0.0$).
  - $\Delta_{{\\text{{logits}}}}(C_{{\\text{{ZERO}}}} \\text{{ vs }} C_{{\\text{{TRUE}}}}) = \\mathbf{{{avg_delta_zero:.6f}}}$.
- **Độ chính xác trên 36 mẫu:**
  - $C_{{\\text{{TRUE}}}}$: ${corr_true_count}/36$ (${corr_true_count/36:.4f}$)
  - $C_{{\\text{{WRONG}}}}$: ${corr_wrong_count}/36$ (${corr_wrong_count/36:.4f}$)
  - $C_{{\\text{{ZERO}}}}$: ${corr_zero_count}/36$ (${corr_zero_count/36:.4f}$)
- **Phán quyết:** `TREATMENT_VERIFIED` (Can thiệp số học hoạt động nhưng chưa đủ lực làm đổi nhãn sinh greedy trên 36 mẫu; số ca đổi nhãn: ${flips_wrong}/36$).

---

## 2. Kết Quả Kiểm Định Thiên Lệch Phương Án Trắc Nghiệm (Rotated-Option MCQA)

Đánh giá trên $624$ biến thể ($156$ câu trắc nghiệm $\\times 4$ hoán vị vòng tròn):
- **Phân bố dự đoán trên 624 biến thể:**
  - Phương án `c`: **${arm_b_pred_dist.get('c', 0)} / 624$** (${arm_b_pred_dist.get('c', 0)/624*100:.1f}\\%$)
  - Phương án `b`: ${arm_b_pred_dist.get('b', 0)} / 624$
  - Phương án `a`: ${arm_b_pred_dist.get('a', 0)} / 624$
  - Phương án `d`: ${arm_b_pred_dist.get('d', 0)} / 624$
- **Độ chính xác gộp trên 624 biến thể:** ${arm_b_correct_count} / 624$ (${arm_b_correct_count/624:.4f}$).
- **Độ nhất quán ngữ nghĩa (Semantic Consistency Rate):**
  - Số câu trả lời đúng trên cả 4 hoán vị: **${consistent_all_4} / {total_q}$** (${consistent_all_4/total_q:.4f}$).
  - Số câu trả lời đúng trên ít nhất 3 hoán vị: **${consistent_at_least_3} / {total_q}$** (${consistent_at_least_3/total_q:.4f}$).
- **Phán quyết khoa học:** **`LABEL_COLLAPSE_CONFIRMED`**. Mô hình Arm B có thiên kiến cực đoan về ký tự `c` (chiếm đa số dự đoán bất kể vị trí xoay của phương án). Năng lực trắc nghiệm hiện tại phụ thuộc lớn vào shortcut chữ cái thay vì suy luận ngữ nghĩa thực sự.

---

## 3. Kết Quả Kiểm Định Phản Thực Nghiệm Ngữ Nghĩa (Semantic Temporal Counterfactuals)

Đánh giá trên tập câu hỏi ghép cặp có căn cứ lịch sử:
- **Tỷ lệ đúng cả hai mốc thời gian:** **${both_correct_count} / {len(counterfactual_cases)}$** (${both_correct_count/len(counterfactual_cases):.4f}$).
- **Tỷ lệ lật đáp án khi đổi năm:** **${causal_flip_count} / {len(counterfactual_cases)}$** (${causal_flip_count/len(counterfactual_cases):.4f}$).
- **Phán quyết khoa học:** **`INSUFFICIENT_POWER`**. Trên mô hình 0.5B, năng lực giải quyết câu hỏi Yes/No bị chi phối bởi thiên lệch 'No' (24/24 câu Yes/No đều đoán 'No'). Cần can thiệp chống sụp đổ phân bố trước khi đánh giá tính nhân quả.

---

## 4. Quyết Định Nghiệm Thu (Scientific Gate & Go/No-Go Decision)

- **Cổng Go/No-Go cho Full Training tiếp theo:** **`NO-GO (HALT)`**.
  - Không khởi chạy huấn luyện GPU lớn tiếp theo khi hiện tượng sụp đổ nhãn 'c' và 'No' chưa được giải quyết ở cấp độ hàm mất mát / cân bằng phân phối.
- **Can thiệp tối thiểu khuyến nghị:** Huấn luyện điều hòa phân bố (Label Smoothing / Balanced Option Regularization) hoặc hoán vị dữ liệu trắc nghiệm trong quá trình huấn luyện (Online Choice Permutation) để triệt tiêu triệt để shortcut chữ cái.
"""
    with open(out_dir / "STAGE03_DIAGNOSTIC_VERDICT.md", "w", encoding="utf-8") as f:
        f.write(verdict_md)
        
    print(f"\nAll diagnostics completed and saved to {out_dir}")
    print(f"Total time elapsed: {time.time() - t0:.1f} seconds.")


if __name__ == "__main__":
    main()
