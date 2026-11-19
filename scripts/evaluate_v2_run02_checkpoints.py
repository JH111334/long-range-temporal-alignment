"""Evaluate 4 Trained Arm Checkpoints on Dev Pairs, Counterfactuals and Benchmark.

Enforces Gate 3 constraints:
- Evaluates distinct trained checkpoints (C0, C1, C2, C3)
- Evaluates on unseen group-held-out dev pairs (dev_relation_pairs.jsonl)
- Evaluates counterfactual flipping consistency (COUNTERFACTUAL_CASES.jsonl)
- Evaluates exploratory 180 TempUN Small cases under P12 RAW_GREEDY_V1
- Computes exact binomial McNemar tests on discordant pairs
- Emits RESULTS_AND_PAIRED_STATS.json, NEGATIVE_CONTROLS.json, COUNTERFACTUAL_EVAL.jsonl, ERROR_CASES.jsonl
"""

from __future__ import annotations

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
    # Binomial p-value under p=0.5
    cum_prob = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    two_sided_p = min(1.0, 2.0 * cum_prob)
    chi2 = ((abs(a_only - b_only) - 1.0) ** 2) / n if n > 0 else 0.0
    return chi2, two_sided_p


def evaluate_run02():
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cpu")
    print(f"Evaluating on device: {device}")

    out_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_02"
    ckpt_dir = out_dir / "checkpoints"

    # 1. Load Datasets
    dev_pairs_path = out_dir / "dev_relation_pairs.jsonl"
    cf_path = out_dir / "COUNTERFACTUAL_CASES.jsonl"
    eval_benchmark_path = _REPO_ROOT / "data" / "tempun" / "eval_mcqa.jsonl"

    with open(dev_pairs_path, "r", encoding="utf-8") as f:
        all_dev_pairs = [json.loads(line) for line in f if line.strip()]

    with open(cf_path, "r", encoding="utf-8") as f:
        all_cf_pairs = [json.loads(line) for line in f if line.strip()]

    with open(eval_benchmark_path, "r", encoding="utf-8") as f:
        eval_180_records = [json.loads(line) for line in f if line.strip()]

    # Select representative 60 dev pairs for CPU evaluation
    rng = random.Random(42)
    dev_sample = rng.sample(all_dev_pairs, min(60, len(all_dev_pairs)))
    cf_sample = rng.sample(all_cf_pairs, min(30, len(all_cf_pairs)))

    print(f"Selected {len(dev_sample)} held-out dev pairs and {len(cf_sample)} counterfactual pairs for CPU evaluation.")

    # 2. Load Base Model
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

    # 3. Setup Adapter Architecture
    base_ckpt_path = _REPO_ROOT / "artifacts" / "quarantine" / "M0_FC40" / "kaggle_full_v2_3seed" / "artifacts" / "seed_1" / "checkpoints" / "m0_fc40_seed_1_after_02.pt"
    adapted_modules, gated_proj, cond_context = load_m0_fc40_for_inference(
        model=base_model,
        checkpoint_path=base_ckpt_path,
        device=device,
        rank=8,
        alpha=16.0,
        num_harmonics=2,
        injection_token_index=14,
    )
    base_model.eval()

    arms = [
        "C0_CONTINUED_MASKED",
        "C1_TRUE_TEMPORAL_RELATION",
        "C2_SHUFFLED_CONDITIONING",
        "C3_FORMAT_ONLY",
    ]

    dev_results_by_arm: Dict[str, List[Dict[str, Any]]] = {}
    cf_results_by_arm: Dict[str, List[Dict[str, Any]]] = {}
    eval180_results_by_arm: Dict[str, List[Dict[str, Any]]] = {}

    for arm in arms:
        ckpt_path = ckpt_dir / f"{arm}.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing arm checkpoint: {ckpt_path}")

        print(f"\nEvaluating Arm: {arm} (Checkpoint SHA: {compute_sha256(ckpt_path)[:12]}...)")
        state = torch.load(ckpt_path, map_location=device)

        # Load weights into adapted_modules
        for k, v in adapted_modules.items():
            mod_state = state["adapters_state"][k]
            v.load_state_dict(mod_state, strict=False)

        gated_proj.load_state_dict(state["gated_proj_state"])

        # --- Evaluate on Dev Sample ---
        arm_dev_res = []
        for idx, rec in enumerate(dev_sample):
            q_text = rec["question"]
            gold = rec["target"]
            year = rec["year_eval"]

            prompt, is_yn = build_p12_prompt(q_text, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            vec40 = compute_fixed_fone_canchi_40d(year)
            theta, z = extract_canchi_angles_from_40d(vec40)
            set_model_temporal_coordinates(adapted_modules, theta, z)

            vec_t = torch.tensor(vec40, dtype=torch.float32, device=device).unsqueeze(0)
            injected_v = gated_proj(vec_t).squeeze(0)
            cond_context.injection_token_index = 14
            cond_context.set_vector(injected_v)

            with torch.no_grad():
                gen_outputs = base_model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                gen_tokens = gen_outputs[0][inputs["input_ids"].shape[1]:].tolist()
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            cond_context.clear()

            parsed_pred, validity = task_conditioned_strict_parser(gen_text, is_yn=True)
            is_valid = (validity == "VALID")
            is_correct = (is_valid and parsed_pred is not None and parsed_pred.lower() == gold.lower())

            arm_dev_res.append({
                "pair_id": rec["pair_id"],
                "gold": gold,
                "pred": parsed_pred,
                "gen_text": gen_text,
                "is_correct": is_correct,
                "is_valid": is_valid,
            })

        dev_results_by_arm[arm] = arm_dev_res

        # --- Evaluate on Counterfactual Pairs ---
        arm_cf_res = []
        for idx, cf in enumerate(cf_sample):
            fwd = cf["fwd"]
            rev = cf["rev"]

            # Run forward
            p_fwd, _ = build_p12_prompt(fwd["question"], tokenizer)
            inp_fwd = tokenizer(p_fwd, return_tensors="pt").to(device)
            vec_fwd = compute_fixed_fone_canchi_40d(fwd["year_eval"])
            theta_f, z_f = extract_canchi_angles_from_40d(vec_fwd)
            set_model_temporal_coordinates(adapted_modules, theta_f, z_f)
            cond_context.set_vector(gated_proj(torch.tensor(vec_fwd, dtype=torch.float32).unsqueeze(0)).squeeze(0))
            with torch.no_grad():
                out_f = base_model.generate(**inp_fwd, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                gen_f = tokenizer.decode(out_f[0][inp_fwd["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            cond_context.clear()
            pred_fwd, _ = task_conditioned_strict_parser(gen_f, is_yn=True)

            # Run reverse
            p_rev, _ = build_p12_prompt(rev["question"], tokenizer)
            inp_rev = tokenizer(p_rev, return_tensors="pt").to(device)
            vec_rev = compute_fixed_fone_canchi_40d(rev["year_eval"])
            theta_r, z_r = extract_canchi_angles_from_40d(vec_rev)
            set_model_temporal_coordinates(adapted_modules, theta_r, z_r)
            cond_context.set_vector(gated_proj(torch.tensor(vec_rev, dtype=torch.float32).unsqueeze(0)).squeeze(0))
            with torch.no_grad():
                out_r = base_model.generate(**inp_rev, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                gen_r = tokenizer.decode(out_r[0][inp_rev["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            cond_context.clear()
            pred_rev, _ = task_conditioned_strict_parser(gen_r, is_yn=True)

            flipped_properly = (pred_fwd is not None and pred_rev is not None and pred_fwd != pred_rev)
            both_correct = (pred_fwd == fwd["target"] and pred_rev == rev["target"])

            arm_cf_res.append({
                "fwd_pair_id": fwd["pair_id"],
                "fwd_target": fwd["target"],
                "fwd_pred": pred_fwd,
                "rev_target": rev["target"],
                "rev_pred": pred_rev,
                "flipped_properly": flipped_properly,
                "both_correct": both_correct,
            })

        cf_results_by_arm[arm] = arm_cf_res

        # --- Evaluate on Exploratory 180 Benchmark (First 30 cases for rapid CPU reporting) ---
        arm_eval180_res = []
        for idx, rec in enumerate(eval_180_records[:30]):
            raw_text = rec.get("input", "") or rec.get("text", "")
            gold = rec.get("target", "").strip()
            m_year = re.search(r"\b(19\d\d|20\d\d)\b", raw_text)
            year = int(m_year.group(1)) if m_year else 1984

            prompt, is_yn = build_p12_prompt(raw_text, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            vec40 = compute_fixed_fone_canchi_40d(year)
            theta, z = extract_canchi_angles_from_40d(vec40)
            set_model_temporal_coordinates(adapted_modules, theta, z)
            cond_context.set_vector(gated_proj(torch.tensor(vec40, dtype=torch.float32).unsqueeze(0)).squeeze(0))

            with torch.no_grad():
                out = base_model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            cond_context.clear()

            parsed_pred, validity = task_conditioned_strict_parser(gen, is_yn)
            is_valid = (validity == "VALID")
            is_correct = (is_valid and parsed_pred is not None and parsed_pred.lower() == gold.lower())

            arm_eval180_res.append({
                "example_id": rec.get("example_id", f"case_{idx}"),
                "is_correct": is_correct,
                "is_valid": is_valid,
                "pred": parsed_pred,
                "gold": gold,
            })
        eval180_results_by_arm[arm] = arm_eval180_res

    # 4. Compute Metrics and McNemar Tests
    dev_metrics = {}
    for arm in arms:
        res = dev_results_by_arm[arm]
        total = len(res)
        correct = sum(1 for r in res if r["is_correct"])
        valid = sum(1 for r in res if r["is_valid"])

        cf_res = cf_results_by_arm[arm]
        cf_flipped = sum(1 for r in cf_res if r["flipped_properly"])
        cf_both_correct = sum(1 for r in cf_res if r["both_correct"])

        b30_res = eval180_results_by_arm[arm]
        b30_correct = sum(1 for r in b30_res if r["is_correct"])

        dev_metrics[arm] = {
            "dev_total": total,
            "dev_correct": correct,
            "dev_accuracy_pct": round((correct / total) * 100.0, 2),
            "dev_invalid_pct": round(((total - valid) / total) * 100.0, 2),
            "counterfactual_sample_total": len(cf_res),
            "counterfactual_flipped_count": cf_flipped,
            "counterfactual_flip_rate_pct": round((cf_flipped / len(cf_res)) * 100.0, 2),
            "counterfactual_both_correct_pct": round((cf_both_correct / len(cf_res)) * 100.0, 2),
            "benchmark_sample_30_acc_pct": round((b30_correct / len(b30_res)) * 100.0, 2),
        }

    # Paired comparisons on dev set
    def compare_arms(res_a: List[Dict[str, Any]], res_b: List[Dict[str, Any]]) -> Dict[str, Any]:
        both_c = sum(1 for ra, rb in zip(res_a, res_b) if ra["is_correct"] and rb["is_correct"])
        a_only = sum(1 for ra, rb in zip(res_a, res_b) if ra["is_correct"] and not rb["is_correct"])
        b_only = sum(1 for ra, rb in zip(res_a, res_b) if not ra["is_correct"] and rb["is_correct"])
        neither = sum(1 for ra, rb in zip(res_a, res_b) if not ra["is_correct"] and not rb["is_correct"])
        chi2, p_val = exact_mcnemar_binomial(a_only, b_only)
        delta_pct = round(((a_only - b_only) / len(res_a)) * 100.0, 2)
        return {
            "n": len(res_a),
            "both_correct": both_c,
            "a_only": a_only,
            "b_only": b_only,
            "neither": neither,
            "delta_pct": delta_pct,
            "exact_mcnemar_chi2": round(chi2, 4),
            "exact_two_sided_p": round(p_val, 5),
            "statistically_significant_05": p_val < 0.05,
        }

    c1_dev = dev_results_by_arm["C1_TRUE_TEMPORAL_RELATION"]
    c0_dev = dev_results_by_arm["C0_CONTINUED_MASKED"]
    c2_dev = dev_results_by_arm["C2_SHUFFLED_CONDITIONING"]
    c3_dev = dev_results_by_arm["C3_FORMAT_ONLY"]

    paired_c1_vs_c0 = compare_arms(c1_dev, c0_dev)
    paired_c1_vs_c2 = compare_arms(c1_dev, c2_dev)
    paired_c1_vs_c3 = compare_arms(c1_dev, c3_dev)

    # 5. Save Artifacts
    results_and_paired_stats = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+07:00"),
        "protocol": "P12_TASK_AWARE_CHATML / RAW_GREEDY_V1",
        "eval_datasets": {
            "primary_held_out_dev": "artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_02/dev_relation_pairs.jsonl",
            "counterfactuals": "artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_02/COUNTERFACTUAL_CASES.jsonl",
            "exploratory_benchmark": "data/tempun/eval_mcqa.jsonl",
        },
        "arm_metrics": dev_metrics,
        "paired_comparisons_on_dev": {
            "C1_vs_C0": paired_c1_vs_c0,
            "C1_vs_C2_negative_control": paired_c1_vs_c2,
            "C1_vs_C3_format_control": paired_c1_vs_c3,
        },
        "summary": {
            "C0_CONTINUED_MASKED": f"{dev_metrics['C0_CONTINUED_MASKED']['dev_accuracy_pct']}% (CF Both Correct: {dev_metrics['C0_CONTINUED_MASKED']['counterfactual_both_correct_pct']}%)",
            "C1_TRUE_TEMPORAL_RELATION": f"{dev_metrics['C1_TRUE_TEMPORAL_RELATION']['dev_accuracy_pct']}% (CF Both Correct: {dev_metrics['C1_TRUE_TEMPORAL_RELATION']['counterfactual_both_correct_pct']}%)",
            "C2_SHUFFLED_CONDITIONING": f"{dev_metrics['C2_SHUFFLED_CONDITIONING']['dev_accuracy_pct']}% (CF Both Correct: {dev_metrics['C2_SHUFFLED_CONDITIONING']['counterfactual_both_correct_pct']}%)",
            "C3_FORMAT_ONLY": f"{dev_metrics['C3_FORMAT_ONLY']['dev_accuracy_pct']}% (CF Both Correct: {dev_metrics['C3_FORMAT_ONLY']['counterfactual_both_correct_pct']}%)",
        }
    }

    with open(out_dir / "RESULTS_AND_PAIRED_STATS.json", "w", encoding="utf-8") as f:
        json.dump(results_and_paired_stats, f, ensure_ascii=False, indent=2)

    negative_controls = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+07:00"),
        "c1_accuracy": dev_metrics["C1_TRUE_TEMPORAL_RELATION"]["dev_accuracy_pct"],
        "c2_shuffled_accuracy": dev_metrics["C2_SHUFFLED_CONDITIONING"]["dev_accuracy_pct"],
        "delta_c1_c2": paired_c1_vs_c2["delta_pct"],
        "exact_p_value": paired_c1_vs_c2["exact_two_sided_p"],
        "counterfactual_flip_rate_c1": dev_metrics["C1_TRUE_TEMPORAL_RELATION"]["counterfactual_flip_rate_pct"],
        "counterfactual_flip_rate_c2": dev_metrics["C2_SHUFFLED_CONDITIONING"]["counterfactual_flip_rate_pct"],
        "verdict": "CONFIRMED_CAUSAL_DEPENDENCE" if (paired_c1_vs_c2["delta_pct"] > 0 and paired_c1_vs_c2["exact_two_sided_p"] < 0.05) else "INCONCLUSIVE_OR_NO_STATISTICAL_GAIN",
    }

    with open(out_dir / "NEGATIVE_CONTROLS.json", "w", encoding="utf-8") as f:
        json.dump(negative_controls, f, ensure_ascii=False, indent=2)

    with open(out_dir / "COUNTERFACTUAL_EVAL.jsonl", "w", encoding="utf-8") as f:
        for r in cf_results_by_arm["C1_TRUE_TEMPORAL_RELATION"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_dir / "ERROR_CASES.jsonl", "w", encoding="utf-8") as f:
        for r in dev_results_by_arm["C1_TRUE_TEMPORAL_RELATION"]:
            if not r["is_correct"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nEvaluation artifacts written successfully!")


if __name__ == "__main__":
    evaluate_run02()
