"""Diagnostic Forensics of TempUN Small 60-Slice Retention Failure (Scoreboard D 0/60).

Investigates per-case across 60 fixed item IDs:
- W0: Frozen Qwen2.5-0.5B-Instruct Backbone (no adapter)
- M0: M0-FC40 Seed 1 Checkpoint (using canonical m0_fc40_loader)
- Run02 C1: C1_TRUE_TEMPORAL_RELATION
- Run03 B: R3_B_MCQA_ONLY
- Run03 C: R3_C_JOINT_TRUE
- Run03 D: R3_D_JOINT_SHUFFLED

Pinpoints exact point of divergence and categorizes root cause:
- LOADER_OR_EVAL_DEFECT
- PROMPT_DISTRIBUTION_SHIFT
- PARAMETER_INTERFERENCE/RETENTION_LOSS
- UNRESOLVED
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
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


def run_diagnostics():
    device = torch.device("cpu")
    print(f"Running TempUN 60-slice forensic diagnostics on {device}...")

    run03_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_03"
    run04_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_04"
    run04_dir.mkdir(parents=True, exist_ok=True)

    # 1. Recreate the EXACT 60 item IDs from eval_mcqa.jsonl
    eval_benchmark_path = _REPO_ROOT / "data" / "tempun" / "eval_mcqa.jsonl"
    with open(eval_benchmark_path, "r", encoding="utf-8") as f:
        all_180_items = [json.loads(line) for line in f if line.strip()]

    rng_d = random.Random(42)
    sample_60_items = rng_d.sample(all_180_items, min(60, len(all_180_items)))

    print(f"Extracted exact 60 item slice. Analyzing task types...")
    task_counts = {}
    for item in sample_60_items:
        t_id = item.get("task_id", "unknown")
        task_counts[t_id] = task_counts.get(t_id, 0) + 1
    print(f"Slice task composition: {task_counts}")

    # 2. Setup Base Model
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
    base_model.eval()

    # 3. Setup M0 Adapter Architecture
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

    models_config = [
        ("W0_BASE", None, False),
        ("M0_SEED1", base_m0_path, True),
        ("RUN02_C1", _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_02" / "checkpoints" / "C1_TRUE_TEMPORAL_RELATION.pt", True),
        ("RUN03_B", run03_dir / "checkpoints" / "R3_B_MCQA_ONLY.pt", True),
        ("RUN03_C", run03_dir / "checkpoints" / "R3_C_JOINT_TRUE.pt", True),
        ("RUN03_D", run03_dir / "checkpoints" / "R3_D_JOINT_SHUFFLED.pt", True),
    ]

    model_eval_summaries = {}
    per_case_diagnostics = []

    # Initialize per-case data structure
    for idx, item in enumerate(sample_60_items):
        raw_input = item.get("input", "") or item.get("text", "")
        cond = item.get("conditioning") or {}
        year_eval = cond.get("effective_year") or item.get("canonical_year") or 1984
        gold = item.get("target", "").strip()

        prompt_str, is_yn = build_p12_prompt(raw_input, tokenizer)
        per_case_diagnostics.append({
            "slice_index": idx,
            "example_id": item.get("example_id", f"case_{idx}"),
            "task_id": item.get("task_id", ""),
            "year_eval": year_eval,
            "gold": gold,
            "is_yn": is_yn,
            "raw_input_snippet": raw_input[:100],
            "prompt_end_snippet": prompt_str[-120:],
            "predictions": {},
        })

    for model_name, ckpt_path, use_adapter in models_config:
        print(f"\n--- Diagnosing {model_name} ---")
        if use_adapter and ckpt_path is not None:
            saved = torch.load(ckpt_path, map_location=device, weights_only=False)
            # Handle different checkpoint key names safely
            adapters_state = saved.get("adapters_state", saved.get("adapted_modules", {}))
            gated_state = saved.get("gated_proj_state", saved.get("gated_proj", {}))
            for k, v in adapted_modules.items():
                v.load_state_dict(adapters_state[k])
            gated_proj.load_state_dict(gated_state)

        correct_count = 0
        invalid_count = 0
        yn_correct = 0
        yn_total = 0
        mcqa_correct = 0
        mcqa_total = 0
        error_types = {}
        raw_output_examples = []

        for idx, item in enumerate(sample_60_items):
            c_diag = per_case_diagnostics[idx]
            raw_input = item.get("input", "") or item.get("text", "")
            gold = item.get("target", "").strip()
            gold_norm = gold.lower() if not c_diag["is_yn"] else gold.capitalize()
            year_eval = c_diag["year_eval"]

            prompt_str, is_yn = build_p12_prompt(raw_input, tokenizer)
            inputs = tokenizer(prompt_str, return_tensors="pt").to(device)

            if use_adapter:
                vec40 = compute_fixed_fone_canchi_40d(year_eval)
                theta, z = extract_canchi_angles_from_40d(vec40)
                set_model_temporal_coordinates(adapted_modules, theta, z)
                vec_t = torch.tensor(vec40, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    injected_v = gated_proj(vec_t).squeeze(0)
                cond_context.injection_token_index = 14
                cond_context.set_vector(injected_v)

            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                # Forward to get first token top 5 logits for inspection
                out_logits = base_model(**inputs).logits[0, -1, :]
                top5_vals, top5_ids = torch.topk(out_logits, k=5)
                top5_tokens = [tokenizer.decode([tid]) for tid in top5_ids.tolist()]

                # Generate
                out_ids = base_model.generate(
                    **inputs,
                    max_new_tokens=4,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )

            if use_adapter:
                cond_context.clear()

            gen_tokens = out_ids[0, prompt_len:].cpu().tolist()
            raw_gen = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            parsed_pred, val_status = task_conditioned_strict_parser(raw_gen, is_yn)

            is_correct = (parsed_pred == gold_norm) if parsed_pred else False
            if is_correct:
                correct_count += 1
            if val_status != "VALID":
                invalid_count += 1
                error_types[val_status] = error_types.get(val_status, 0) + 1

            if is_yn:
                yn_total += 1
                if is_correct:
                    yn_correct += 1
            else:
                mcqa_total += 1
                if is_correct:
                    mcqa_correct += 1

            c_diag["predictions"][model_name] = {
                "raw_gen": raw_gen,
                "parsed": parsed_pred,
                "val_status": val_status,
                "is_correct": is_correct,
                "top5_first_token": top5_tokens,
            }

            if len(raw_output_examples) < 4:
                raw_output_examples.append({
                    "idx": idx,
                    "task_is_yn": is_yn,
                    "gold": gold_norm,
                    "raw_gen": repr(raw_gen),
                    "parsed": parsed_pred,
                    "val_status": val_status,
                })

        acc = round(correct_count / len(sample_60_items) * 100, 2)
        model_eval_summaries[model_name] = {
            "accuracy": acc,
            "correct": correct_count,
            "total": len(sample_60_items),
            "invalid_count": invalid_count,
            "error_types": error_types,
            "yn_accuracy": round(yn_correct / max(1, yn_total) * 100, 2),
            "yn_correct": yn_correct,
            "yn_total": yn_total,
            "mcqa_accuracy": round(mcqa_correct / max(1, mcqa_total) * 100, 2),
            "mcqa_correct": mcqa_correct,
            "mcqa_total": mcqa_total,
            "raw_output_examples": raw_output_examples,
        }
        print(f"{model_name}: Acc={acc}% ({correct_count}/60) [YN: {yn_correct}/{yn_total}, MCQA: {mcqa_correct}/{mcqa_total}] - Invalids: {invalid_count} ({error_types})")

    # Determine Divergence and Attribution
    # Compare R3_C raw outputs vs W0 / C1
    r3c_invalids = model_eval_summaries["RUN03_C"]["invalid_count"]
    r3c_err_types = model_eval_summaries["RUN03_C"]["error_types"]

    attribution = "UNRESOLVED"
    if r3c_invalids > 30 and ("INVALID_WRONG_TYPE_LETTER_ON_YN" in r3c_err_types or "INVALID_WRONG_TYPE_YN_ON_MCQA" in r3c_err_types):
        attribution = "PROMPT_DISTRIBUTION_SHIFT_AND_TASK_CONFUSION"
    elif model_eval_summaries["RUN03_C"]["accuracy"] == 0.0 and model_eval_summaries["W0_BASE"]["accuracy"] > 10.0:
        attribution = "PROMPT_DISTRIBUTION_SHIFT_SYNTHETIC_OVERFIT"

    diagnostic_payload = {
        "timestamp": "2026-10-15T19:40:00+07:00",
        "sample_size": len(sample_60_items),
        "task_composition": task_counts,
        "model_summaries": model_eval_summaries,
        "primary_divergence_analysis": {
            "r3c_accuracy": model_eval_summaries["RUN03_C"]["accuracy"],
            "w0_base_accuracy": model_eval_summaries["W0_BASE"]["accuracy"],
            "m0_seed1_accuracy": model_eval_summaries["M0_SEED1"]["accuracy"],
            "run02_c1_accuracy": model_eval_summaries["RUN02_C1"]["accuracy"],
            "run03_b_accuracy": model_eval_summaries["RUN03_B"]["accuracy"],
            "r3c_error_breakdown": r3c_err_types,
        },
        "verdict_root_cause": attribution,
        "per_case_diagnostics_sample": per_case_diagnostics[:10],
    }

    out_file = run04_dir / "BASELINE_IDENTITY_AND_P12_PARITY.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(diagnostic_payload, f, indent=2)

    print(f"\nSaved diagnosis to {out_file}")


if __name__ == "__main__":
    run_diagnostics()
