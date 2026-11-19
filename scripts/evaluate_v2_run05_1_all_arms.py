"""Comprehensive Independent Evaluation for V2 Run 05.1 (Forensic Phoenix Recovery).

Evaluates:
- Arm A (R51_A_NTP_REL)
- Arm B (R51_B_NTP_REL_MCQA)
- Arm C (R51_C_B_CONDITIONING_INTERVENTIONS on frozen Arm B):
  * C_TRUE: Normal temporal coordinates via YearResolution V2
  * C_WRONG_YEAR: Deterministic mismatched year (+30 years shift)
  * C_ZERO_VECTOR: Zero coordinate vector (dim=40)
  * C_SEAM_OFF: Seam injection bypassed, MLP LoRA active
  * C_LORA_OFF: MLP LoRA angles cleared, Seam injection active
  * C_BOTH_OFF: Both Seam and LoRA bypassed (Base Equivalent)

Benchmarks:
- TempUN Small 180 (data/tempun/eval_mcqa.jsonl, n=180)
- DEV MCQA Clean (artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04/MCQA4_DEV_CLEAN.jsonl, n=120)
- DEV Relation (artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_02/dev_relation_pairs.jsonl, n=60)
- Held-out NTP Perplexity (data/tempun/train_nwp.jsonl, unseen slice n=100)

Produces:
- TEMPUN180_RAW_GENERATIONS_A.jsonl
- TEMPUN180_RAW_GENERATIONS_B.jsonl
- TEMPUN180_RAW_GENERATIONS_C_<condition>.jsonl (x6)
- C_ABLATION_ELIGIBILITY_AND_PATHWAY_TRACE.jsonl
- SCOREBOARD_A_B_C_BY_TASK_AND_PROTOCOL.json
- PAIRED_STATS_AND_EFFECTIVE_TREATMENTS.json
- DEV_DIAGNOSTICS_BY_ARM.jsonl
- HELDOUT_NTP_CE_PPL_BY_ARM.json
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from temporal_alignment.baselines.ticktack05b.evaluator import (
    build_p12_prompt,
    task_conditioned_strict_parser,
)
from temporal_alignment.contracts import Era, HistoricalYear
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
from temporal_alignment.normalization.year_resolution import (
    TemporalInputMode,
    YearResolution,
    YearResolutionStatus,
    resolve_temporal_context,
)


def exact_mcnemar_binomial(b: int, c: int) -> Tuple[float, float]:
    """Exact two-sided McNemar binomial test with Edwards continuity correction."""
    n = b + c
    if n == 0:
        return 0.0, 1.0
    chi2 = ((abs(b - c) - 1.0) ** 2) / float(n)
    from math import comb
    k = min(b, c)
    p_val = 2.0 * sum(comb(n, i) * (0.5 ** n) for i in range(k + 1))
    p_val = min(1.0, p_val)
    return chi2, p_val


def compute_heldout_ntp_metrics(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    corpus: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    """Compute held-out next-token cross-entropy and perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for item in corpus:
            text = f"{item.get('input', '')} {item.get('target', '')}".strip()
            if not text:
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] <= 1:
                continue

            outputs = model(input_ids=input_ids)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            n_tokens = shift_labels.numel()
            total_loss += loss.item()
            total_tokens += n_tokens

    avg_ce = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_ce, 20.0))
    return {
        "heldout_ce_loss": round(avg_ce, 4),
        "heldout_perplexity": round(ppl, 2),
        "total_eval_tokens": total_tokens,
    }


def load_adapter_weights_into_model(
    checkpoint_path: Path,
    adapted_modules: Dict[str, Any],
    gated_proj: nn.Module,
    device: torch.device,
) -> None:
    """Load compact adapter checkpoint tensors cleanly."""
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


def evaluate_single_sample(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    gated_proj: nn.Module,
    cond_context: ConditioningContext,
    adapted_modules: Dict[str, DualCycleMaTLoRALinear],
    item: Dict[str, Any],
    condition_id: str,
    device: torch.device,
    max_new_tokens: int = 4,
) -> Dict[str, Any]:
    """Evaluate a single question under a given conditioning condition."""
    example_id = item.get("example_id") or str(item.get("id", ""))
    question_text = item.get("question") or item.get("input", "")
    gold_target = item.get("target", "")

    # Year Resolution
    year_res = resolve_temporal_context(item)
    has_single_year = (year_res.status == YearResolutionStatus.SINGLE and year_res.effective_year is not None)
    eligibility = "ELIGIBLE_FOR_TREATMENT" if has_single_year else "INELIGIBLE_UNTREATED"

    # Setup Condition
    pathway_trace = "UNKNOWN"

    if condition_id == "C_TRUE":
        if has_single_year:
            year_val = year_res.effective_year.astronomical
            vec40 = compute_fixed_fone_canchi_40d(year_val)
            theta, z = extract_canchi_angles_from_40d(vec40)
            set_model_temporal_coordinates(adapted_modules, theta=theta, z=z)

            vec_t = torch.tensor(vec40, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                proj_v = gated_proj(vec_t).squeeze(0)
            cond_context.injection_token_index = 14
            cond_context.set_vector(proj_v)
            pathway_trace = f"YEAR_ACTIVE_{year_val}"
        else:
            cond_context.clear()
            clear_model_temporal_coordinates(adapted_modules)
            pathway_trace = "BYPASS_NO_SINGLE_YEAR"

    elif condition_id == "C_WRONG_YEAR":
        if has_single_year:
            year_val = year_res.effective_year.astronomical + 30
            vec40 = compute_fixed_fone_canchi_40d(year_val)
            theta, z = extract_canchi_angles_from_40d(vec40)
            set_model_temporal_coordinates(adapted_modules, theta=theta, z=z)

            vec_t = torch.tensor(vec40, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                proj_v = gated_proj(vec_t).squeeze(0)
            cond_context.injection_token_index = 14
            cond_context.set_vector(proj_v)
            pathway_trace = f"WRONG_YEAR_ACTIVE_{year_val}"
        else:
            cond_context.clear()
            clear_model_temporal_coordinates(adapted_modules)
            pathway_trace = "BYPASS_NO_SINGLE_YEAR"

    elif condition_id == "C_ZERO_VECTOR":
        if has_single_year:
            vec40 = [0.0] * 40
            set_model_temporal_coordinates(adapted_modules, theta=0.0, z=0.0)
            vec_t = torch.tensor(vec40, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                proj_v = gated_proj(vec_t).squeeze(0)
            cond_context.injection_token_index = 14
            cond_context.set_vector(proj_v)
            pathway_trace = "ZERO_VECTOR_ACTIVE"
        else:
            cond_context.clear()
            clear_model_temporal_coordinates(adapted_modules)
            pathway_trace = "BYPASS_NO_SINGLE_YEAR"

    elif condition_id == "C_SEAM_OFF":
        if has_single_year:
            year_val = year_res.effective_year.astronomical
            vec40 = compute_fixed_fone_canchi_40d(year_val)
            theta, z = extract_canchi_angles_from_40d(vec40)
            set_model_temporal_coordinates(adapted_modules, theta=theta, z=z)
            cond_context.clear()
            pathway_trace = f"SEAM_OFF_LORA_ACTIVE_{year_val}"
        else:
            cond_context.clear()
            clear_model_temporal_coordinates(adapted_modules)
            pathway_trace = "BYPASS_NO_SINGLE_YEAR"

    elif condition_id == "C_LORA_OFF":
        if has_single_year:
            year_val = year_res.effective_year.astronomical
            vec40 = compute_fixed_fone_canchi_40d(year_val)
            clear_model_temporal_coordinates(adapted_modules)

            vec_t = torch.tensor(vec40, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                proj_v = gated_proj(vec_t).squeeze(0)
            cond_context.injection_token_index = 14
            cond_context.set_vector(proj_v)
            pathway_trace = f"LORA_OFF_SEAM_ACTIVE_{year_val}"
        else:
            cond_context.clear()
            clear_model_temporal_coordinates(adapted_modules)
            pathway_trace = "BYPASS_NO_SINGLE_YEAR"

    elif condition_id == "C_BOTH_OFF":
        cond_context.clear()
        clear_model_temporal_coordinates(adapted_modules)
        pathway_trace = "BOTH_OFF_BASE_EQUIVALENT"

    # Greedy Free-Vocabulary Generation
    prompt_str, is_yn = build_p12_prompt(question_text, tokenizer)
    inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=151645,
            pad_token_id=151643,
        )

    # Post-generation cleanup
    cond_context.clear()
    clear_model_temporal_coordinates(adapted_modules)

    gen_tokens = out_ids[0, prompt_len:].cpu().tolist()
    raw_gen = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    parsed_pred, val_status = task_conditioned_strict_parser(raw_gen, is_yn)

    is_valid = (val_status == "VALID")
    is_correct = False
    if is_valid and parsed_pred is not None:
        is_correct = (parsed_pred.strip().lower() == gold_target.strip().lower())

    return {
        "example_id": example_id,
        "is_yn": is_yn,
        "gold_target": gold_target,
        "raw_generation": raw_gen,
        "parsed_pred": parsed_pred,
        "validity_status": val_status,
        "is_valid": is_valid,
        "is_correct": is_correct,
        "year_resolution_status": year_res.status.value,
        "effective_year": year_res.effective_year.astronomical if year_res.effective_year else None,
        "eligibility": eligibility,
        "pathway_trace": pathway_trace,
    }


def run_full_evaluation(device_name: Optional[str] = None, repo_root: Optional[Path] = None):
    root = repo_root or _REPO_ROOT
    device = torch.device(device_name) if device_name else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting V2 Run 05.1 Full Evaluation on {device} ===", flush=True)

    run051_dir = root / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_1"
    ckpt_dir = run051_dir / "checkpoints"
    out_dir = run051_dir

    ckpt_a = ckpt_dir / "R51_A_NTP_REL.pt"
    ckpt_b = ckpt_dir / "R51_B_NTP_REL_MCQA.pt"

    assert ckpt_a.exists(), f"Missing Checkpoint A: {ckpt_a}"
    assert ckpt_b.exists(), f"Missing Checkpoint B: {ckpt_b}"

    # Load Foundation Model
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print("Loading tokenizer and base model...", flush=True)
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

    base_model = base_model.to(device)
    for p in base_model.parameters():
        p.requires_grad = False

    # Setup Architecture
    adapted_modules = replace_mlp_with_dual_cycle_mat_lora(
        model=base_model,
        rank=8,
        alpha=16.0,
        num_harmonics=2,
    )
    gated_proj = GatedResidualProjection(in_features=40, out_features=896).to(device)
    cond_context = ConditioningContext(injection_token_index=14)
    hook = make_gated_embedding_hook(cond_context)
    base_model.get_input_embeddings().register_forward_hook(hook)
    base_model.eval()

    # Load Evaluation Datasets
    print("Loading test datasets...", flush=True)
    tempun_path = root / "data" / "tempun" / "eval_mcqa.jsonl"
    dev_mcqa_path = root / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_04" / "MCQA4_DEV_CLEAN.jsonl"
    dev_rel_path = root / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_02" / "dev_relation_pairs.jsonl"
    nwp_path = root / "data" / "tempun" / "train_nwp.jsonl"

    with open(tempun_path, "r", encoding="utf-8") as f:
        tempun180 = [json.loads(line) for line in f if line.strip()]
    with open(dev_mcqa_path, "r", encoding="utf-8") as f:
        dev_mcqa = [json.loads(line) for line in f if line.strip()]
    with open(dev_rel_path, "r", encoding="utf-8") as f:
        dev_rel = [json.loads(line) for line in f if line.strip()]
    with open(nwp_path, "r", encoding="utf-8") as f:
        nwp_data = [json.loads(line) for line in f if line.strip()]

    # Unseen NTP held-out corpus
    heldout_ntp_corpus = nwp_data[-100:]
    dev_rel60 = dev_rel[:60]

    print(f"Datasets: TempUN={len(tempun180)}, Dev_MCQA={len(dev_mcqa)}, Dev_Rel={len(dev_rel60)}, NTP_Heldout={len(heldout_ntp_corpus)}", flush=True)

    # 1. Evaluate Arm A
    print("\n--- Evaluating Arm A (R51_A_NTP_REL) ---", flush=True)
    load_adapter_weights_into_model(ckpt_a, adapted_modules, gated_proj, device)

    ntp_metrics_a = compute_heldout_ntp_metrics(base_model, tokenizer, heldout_ntp_corpus, device)
    print(f"Arm A Held-out NTP: CE={ntp_metrics_a['heldout_ce_loss']}, PPL={ntp_metrics_a['heldout_perplexity']}", flush=True)

    tempun_gens_a = []
    for item in tempun180:
        res = evaluate_single_sample(base_model, tokenizer, gated_proj, cond_context, adapted_modules, item, condition_id="C_TRUE", device=device)
        tempun_gens_a.append(res)

    dev_mcqa_gens_a = []
    for item in dev_mcqa:
        res = evaluate_single_sample(base_model, tokenizer, gated_proj, cond_context, adapted_modules, item, condition_id="C_TRUE", device=device)
        dev_mcqa_gens_a.append(res)

    dev_rel_gens_a = []
    for item in dev_rel60:
        res = evaluate_single_sample(base_model, tokenizer, gated_proj, cond_context, adapted_modules, item, condition_id="C_TRUE", device=device)
        dev_rel_gens_a.append(res)

    with open(out_dir / "TEMPUN180_RAW_GENERATIONS_A.jsonl", "w", encoding="utf-8") as f:
        for r in tempun_gens_a:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 2. Evaluate Arm B
    print("\n--- Evaluating Arm B (R51_B_NTP_REL_MCQA) ---", flush=True)
    load_adapter_weights_into_model(ckpt_b, adapted_modules, gated_proj, device)

    ntp_metrics_b = compute_heldout_ntp_metrics(base_model, tokenizer, heldout_ntp_corpus, device)
    print(f"Arm B Held-out NTP: CE={ntp_metrics_b['heldout_ce_loss']}, PPL={ntp_metrics_b['heldout_perplexity']}", flush=True)

    tempun_gens_b = []
    for item in tempun180:
        res = evaluate_single_sample(base_model, tokenizer, gated_proj, cond_context, adapted_modules, item, condition_id="C_TRUE", device=device)
        tempun_gens_b.append(res)

    dev_mcqa_gens_b = []
    for item in dev_mcqa:
        res = evaluate_single_sample(base_model, tokenizer, gated_proj, cond_context, adapted_modules, item, condition_id="C_TRUE", device=device)
        dev_mcqa_gens_b.append(res)

    dev_rel_gens_b = []
    for item in dev_rel60:
        res = evaluate_single_sample(base_model, tokenizer, gated_proj, cond_context, adapted_modules, item, condition_id="C_TRUE", device=device)
        dev_rel_gens_b.append(res)

    with open(out_dir / "TEMPUN180_RAW_GENERATIONS_B.jsonl", "w", encoding="utf-8") as f:
        for r in tempun_gens_b:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 3. Evaluate Arm C (6 Interventions on Arm B)
    arm_c_conditions = ["C_TRUE", "C_WRONG_YEAR", "C_ZERO_VECTOR", "C_SEAM_OFF", "C_LORA_OFF", "C_BOTH_OFF"]
    tempun_gens_c = {}

    print("\n--- Evaluating Arm C Interventions on Frozen Arm B ---", flush=True)
    for cond_id in arm_c_conditions:
        print(f"Evaluating Condition: {cond_id} on TempUN 180...", flush=True)
        cond_gens = []
        for item in tempun180:
            res = evaluate_single_sample(base_model, tokenizer, gated_proj, cond_context, adapted_modules, item, condition_id=cond_id, device=device)
            cond_gens.append(res)
        tempun_gens_c[cond_id] = cond_gens

        with open(out_dir / f"TEMPUN180_RAW_GENERATIONS_{cond_id}.jsonl", "w", encoding="utf-8") as f:
            for r in cond_gens:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 4. Write C Pathway Trace
    trace_records = []
    for i, item in enumerate(tempun180):
        c_true_res = tempun_gens_c["C_TRUE"][i]
        trace_records.append({
            "example_id": c_true_res["example_id"],
            "eligibility": c_true_res["eligibility"],
            "year_resolution_status": c_true_res["year_resolution_status"],
            "effective_year": c_true_res["effective_year"],
            "c_true_pathway": c_true_res["pathway_trace"],
            "c_true_pred": c_true_res["parsed_pred"],
            "c_wrong_year_pred": tempun_gens_c["C_WRONG_YEAR"][i]["parsed_pred"],
            "c_zero_vector_pred": tempun_gens_c["C_ZERO_VECTOR"][i]["parsed_pred"],
            "c_seam_off_pred": tempun_gens_c["C_SEAM_OFF"][i]["parsed_pred"],
            "c_lora_off_pred": tempun_gens_c["C_LORA_OFF"][i]["parsed_pred"],
            "c_both_off_pred": tempun_gens_c["C_BOTH_OFF"][i]["parsed_pred"],
        })
    with open(out_dir / "C_ABLATION_ELIGIBILITY_AND_PATHWAY_TRACE.jsonl", "w", encoding="utf-8") as f:
        for tr in trace_records:
            f.write(json.dumps(tr, ensure_ascii=False) + "\n")

    # 5. Compute Metrics Helper
    def summarize_results(gens: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(gens)
        valid = sum(1 for r in gens if r["is_valid"])
        correct = sum(1 for r in gens if r["is_correct"])
        invalid = total - valid
        return {
            "total_items": total,
            "valid_items": valid,
            "invalid_items": invalid,
            "correct_items": correct,
            "valid_rate": round(valid / max(1, total), 4),
            "raw_accuracy": round(correct / max(1, total), 4),
            "conditional_accuracy": round(correct / max(1, valid), 4) if valid > 0 else 0.0,
        }

    # 6. Scoreboard Construction
    scoreboard = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "study_id": "V2_RUN05_1_PHOENIX_RECOVERY",
        "timestamp": "2026-10-16T13:25:00+07:00",
        "protocol": "P12_CHATML_TASK_AWARE_GREEDY",
        "arms": {
            "R51_A_NTP_REL": {
                "checkpoint": str(ckpt_a.relative_to(_REPO_ROOT)),
                "ntp_metrics": ntp_metrics_a,
                "tempun180_overall": summarize_results(tempun_gens_a),
                "tempun180_date_based_36": summarize_results([r for r in tempun_gens_a if r["eligibility"] == "ELIGIBLE_FOR_TREATMENT"]),
                "tempun180_untreated_144": summarize_results([r for r in tempun_gens_a if r["eligibility"] == "INELIGIBLE_UNTREATED"]),
                "dev_mcqa_clean_120": summarize_results(dev_mcqa_gens_a),
                "dev_relation_60": summarize_results(dev_rel_gens_a),
            },
            "R51_B_NTP_REL_MCQA": {
                "checkpoint": str(ckpt_b.relative_to(_REPO_ROOT)),
                "ntp_metrics": ntp_metrics_b,
                "tempun180_overall": summarize_results(tempun_gens_b),
                "tempun180_date_based_36": summarize_results([r for r in tempun_gens_b if r["eligibility"] == "ELIGIBLE_FOR_TREATMENT"]),
                "tempun180_untreated_144": summarize_results([r for r in tempun_gens_b if r["eligibility"] == "INELIGIBLE_UNTREATED"]),
                "dev_mcqa_clean_120": summarize_results(dev_mcqa_gens_b),
                "dev_relation_60": summarize_results(dev_rel_gens_b),
            },
            "R51_C_B_INTERVENTIONS": {
                cond_id: {
                    "tempun180_overall": summarize_results(tempun_gens_c[cond_id]),
                    "tempun180_date_based_36": summarize_results([r for r in tempun_gens_c[cond_id] if r["eligibility"] == "ELIGIBLE_FOR_TREATMENT"]),
                    "tempun180_untreated_144": summarize_results([r for r in tempun_gens_c[cond_id] if r["eligibility"] == "INELIGIBLE_UNTREATED"]),
                }
                for cond_id in arm_c_conditions
            },
        },
    }

    with open(out_dir / "SCOREBOARD_A_B_C_BY_TASK_AND_PROTOCOL.json", "w", encoding="utf-8") as f:
        json.dump(scoreboard, f, indent=2, ensure_ascii=False)

    # 7. Paired McNemar Statistics
    def compare_paired(list_1: List[Dict[str, Any]], list_2: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(list_1) == len(list_2)
        both_correct = 0
        only_1 = 0
        only_2 = 0
        both_incorrect = 0
        for r1, r2 in zip(list_1, list_2):
            c1, c2 = r1["is_correct"], r2["is_correct"]
            if c1 and c2:
                both_correct += 1
            elif c1 and not c2:
                only_1 += 1
            elif not c1 and c2:
                only_2 += 1
            else:
                both_incorrect += 1
        chi2, p_val = exact_mcnemar_binomial(only_1, only_2)
        return {
            "total_pairs": len(list_1),
            "both_correct": both_correct,
            "only_arm_1_correct": only_1,
            "only_arm_2_correct": only_2,
            "both_incorrect": both_incorrect,
            "delta_correct": only_1 - only_2,
            "mcnemar_chi2": round(chi2, 4),
            "mcnemar_p_value": round(p_val, 4),
            "statistically_significant_p05": p_val < 0.05,
        }

    paired_stats = {
        "arm_a_vs_arm_b": {
            "tempun180_overall": compare_paired(tempun_gens_a, tempun_gens_b),
            "tempun180_date_based_36": compare_paired(
                [r for r in tempun_gens_a if r["eligibility"] == "ELIGIBLE_FOR_TREATMENT"],
                [r for r in tempun_gens_b if r["eligibility"] == "ELIGIBLE_FOR_TREATMENT"],
            ),
            "dev_mcqa_clean_120": compare_paired(dev_mcqa_gens_a, dev_mcqa_gens_b),
            "dev_relation_60": compare_paired(dev_rel_gens_a, dev_rel_gens_b),
        },
        "arm_c_interventions_vs_c_true_on_36_date_based": {
            cond_id: compare_paired(
                [r for r in tempun_gens_c["C_TRUE"] if r["eligibility"] == "ELIGIBLE_FOR_TREATMENT"],
                [r for r in tempun_gens_c[cond_id] if r["eligibility"] == "ELIGIBLE_FOR_TREATMENT"],
            )
            for cond_id in arm_c_conditions if cond_id != "C_TRUE"
        },
    }

    with open(out_dir / "PAIRED_STATS_AND_EFFECTIVE_TREATMENTS.json", "w", encoding="utf-8") as f:
        json.dump(paired_stats, f, indent=2, ensure_ascii=False)

    # 8. DEV Diagnostics
    dev_diag = []
    for i in range(len(dev_mcqa)):
        ra, rb = dev_mcqa_gens_a[i], dev_mcqa_gens_b[i]
        dev_diag.append({
            "task": "DEV_MCQA",
            "example_id": dev_mcqa[i].get("example_id", f"dev_mcqa_{i}"),
            "gold_target": ra["gold_target"],
            "pred_a": ra["parsed_pred"],
            "pred_b": rb["parsed_pred"],
            "a_correct": ra["is_correct"],
            "b_correct": rb["is_correct"],
        })
    for i in range(len(dev_rel60)):
        ra, rb = dev_rel_gens_a[i], dev_rel_gens_b[i]
        dev_diag.append({
            "task": "DEV_RELATION",
            "example_id": dev_rel60[i].get("example_id", f"dev_rel_{i}"),
            "gold_target": ra["gold_target"],
            "pred_a": ra["parsed_pred"],
            "pred_b": rb["parsed_pred"],
            "a_correct": ra["is_correct"],
            "b_correct": rb["is_correct"],
        })

    with open(out_dir / "DEV_DIAGNOSTICS_BY_ARM.jsonl", "w", encoding="utf-8") as f:
        for d in dev_diag:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # 9. Held-out NTP Summary
    heldout_ntp_report = {
        "R51_A_NTP_REL": ntp_metrics_a,
        "R51_B_NTP_REL_MCQA": ntp_metrics_b,
    }
    with open(out_dir / "HELDOUT_NTP_CE_PPL_BY_ARM.json", "w", encoding="utf-8") as f:
        json.dump(heldout_ntp_report, f, indent=2, ensure_ascii=False)

    print("\n=== V2 Run 05.1 Full Evaluation Completed Successfully! ===", flush=True)
    return scoreboard


if __name__ == "__main__":
    run_full_evaluation()
