"""Full 180-case Evaluation for M0-FC40 3-Seed Checkpoints under P12 Protocol.

Evaluates:
- Seed 1 after_02 checkpoint
- Seed 2 after_02 checkpoint
- Seed 3 after_02 checkpoint
- Base Qwen2.5-0.5B-Instruct model (zero-adapter control)
Across all 180 cases of TempUN Small eval_mcqa.jsonl using RAW_GREEDY_V1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

_PKG_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from temporal_alignment.baselines.ticktack05b.evaluator import (
    build_p12_prompt,
    task_conditioned_strict_parser,
)
from temporal_alignment.models.dual_cycle_mat_lora import (
    DualCycleMaTLoRALinear,
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


def load_eval_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def evaluate_checkpoint_on_180(
    model: nn.Module,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    device: torch.device,
    adapted_modules: Optional[Dict[str, DualCycleMaTLoRALinear]] = None,
    gated_proj: Optional[GatedResidualProjection] = None,
    cond_context: Optional[ConditioningContext] = None,
    seam_token_index: int = 14,
) -> Dict[str, Any]:
    model.eval()
    results = []
    correct_total = 0
    correct_mcqa = 0
    correct_yn = 0
    invalid_total = 0

    mcqa_count = 0
    yn_count = 0

    for rec in records:
        raw_text = rec.get("input", "") or rec.get("text", "")
        gold_target = rec.get("target", "").strip()

        # Year determination
        year = rec.get("year")
        if year is None:
            cond = rec.get("conditioning", {})
            year = cond.get("effective_year", 1947)
        year = int(year)

        # Build prompt and determine task type strictly from question text
        prompt, is_yn = build_p12_prompt(raw_text, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        # Temporal conditioning for M0-FC40 if active
        if adapted_modules is not None:
            vec40 = compute_fixed_fone_canchi_40d(year)
            theta, z = extract_canchi_angles_from_40d(vec40)
            set_model_temporal_coordinates(adapted_modules, theta, z)

            if gated_proj is not None and cond_context is not None:
                cond_context.injection_token_index = seam_token_index
                vec40_t = torch.tensor(vec40, dtype=torch.float32, device=device).unsqueeze(0)
                injected_vec = gated_proj(vec40_t).squeeze(0)
                cond_context.set_vector(injected_vec)

        # Free-vocabulary greedy generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            # Slice newly generated tokens
            gen_ids = outputs[0][input_ids.shape[1]:].tolist()
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        parsed_pred, validity_status = task_conditioned_strict_parser(gen_text, is_yn)

        is_valid = (validity_status == "VALID")
        is_correct = False
        if is_valid and parsed_pred is not None:
            is_correct = (parsed_pred.lower() == gold_target.lower())

        if is_yn:
            yn_count += 1
            if is_correct: correct_yn += 1
        else:
            mcqa_count += 1
            if is_correct: correct_mcqa += 1

        if not is_valid: invalid_total += 1
        if is_correct: correct_total += 1

        results.append({
            "example_id": rec.get("example_id", ""),
            "year": year,
            "is_yn": is_yn,
            "gold_target": gold_target,
            "generated_tokens": gen_ids,
            "generated_text": gen_text,
            "parsed_pred": parsed_pred,
            "validity_status": validity_status,
            "is_valid": is_valid,
            "is_correct": is_correct,
        })

    n_total = len(records)
    strict_acc = (correct_total / n_total) * 100.0 if n_total > 0 else 0.0
    mcqa_acc = (correct_mcqa / mcqa_count) * 100.0 if mcqa_count > 0 else 0.0
    yn_acc = (correct_yn / yn_count) * 100.0 if yn_count > 0 else 0.0
    inv_rate = (invalid_total / n_total) * 100.0 if n_total > 0 else 0.0

    return {
        "n_samples": n_total,
        "strict_accuracy": round(strict_acc, 2),
        "mcqa_accuracy": round(mcqa_acc, 2),
        "yn_accuracy": round(yn_acc, 2),
        "invalid_rate": round(inv_rate, 2),
        "correct_count": correct_total,
        "invalid_count": invalid_total,
        "records": results,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running M0-FC40 Full 180 Evaluation on device: {device}")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    revision = "7ae557604adf67be50417f59c2c2f167def9a775"
    eval_path = Path("data/tempun/eval_mcqa.jsonl")
    records = load_eval_records(eval_path)
    print(f"Loaded {len(records)} eval records.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_dir = Path("data/Output/TICKTACK_05B_MATCHED_TEMPUN_SMALL/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cache_path = out_dir / "raw_baseline_180.json"
    if base_cache_path.exists():
        print(f"\n--- Loading Cached Base Model Control ({base_cache_path}) ---")
        with open(base_cache_path, "r", encoding="utf-8") as f:
            base_eval = json.load(f)
        print(f"[B_RAW_QWEN05B] Strict Acc: {base_eval['strict_accuracy']}% | MCQA: {base_eval['mcqa_accuracy']}% | YN: {base_eval['yn_accuracy']}% | Invalid: {base_eval['invalid_rate']}%")
    else:
        print("\n--- Evaluating Base Model Control (B_RAW_QWEN05B) ---")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
            trust_remote_code=True,
        ).to(device)

        base_eval = evaluate_checkpoint_on_180(base_model, tokenizer, records, device)
        print(f"[B_RAW_QWEN05B] Strict Acc: {base_eval['strict_accuracy']}% | MCQA: {base_eval['mcqa_accuracy']}% | YN: {base_eval['yn_accuracy']}% | Invalid: {base_eval['invalid_rate']}%")

        with open(base_cache_path, "w", encoding="utf-8") as f:
            json.dump(base_eval, f, indent=2, ensure_ascii=False)

    # 2. Evaluate M0-FC40 Checkpoints across 3 seeds (after_02)
    m0_root = Path("artifacts/quarantine/M0_FC40/kaggle_full_v2_3seed/artifacts")
    seeds = [1, 2, 3]
    m0_results = {}

    for s in seeds:
        print(f"\n--- Evaluating M0-FC40 Seed {s} after_02 ---")
        ckpt_path = m0_root / f"seed_{s}/checkpoints/m0_fc40_seed_{s}_after_02.pt"
        if not ckpt_path.exists():
            print(f"WARNING: Checkpoint {ckpt_path} not found!")
            continue

        # Setup fresh model with DualCycleMaTLoRA and Seam
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
            trust_remote_code=True,
        ).to(device)

        adapted_modules = replace_mlp_with_dual_cycle_mat_lora(model, rank=8, alpha=16.0, num_harmonics=2)
        gated_proj = GatedResidualProjection(in_features=40, out_features=896).to(device)
        cond_context = ConditioningContext(injection_token_index=14)
        hook = make_gated_embedding_hook(cond_context)
        model.get_input_embeddings().register_forward_hook(hook)

        # Load weights
        state = torch.load(ckpt_path, map_location=device)
        if "adapted_modules" in state:
            for m_name, m_state in state["adapted_modules"].items():
                if m_name in adapted_modules:
                    adapted_modules[m_name].load_state_dict(m_state, strict=False)
        if "gated_proj" in state:
            gated_proj.load_state_dict(state["gated_proj"], strict=False)

        res = evaluate_checkpoint_on_180(
            model=model,
            tokenizer=tokenizer,
            records=records,
            device=device,
            adapted_modules=adapted_modules,
            gated_proj=gated_proj,
            cond_context=cond_context,
            seam_token_index=14,
        )
        print(f"[M0-FC40 Seed {s} after_02] Strict Acc: {res['strict_accuracy']}% | MCQA: {res['mcqa_accuracy']}% | YN: {res['yn_accuracy']}% | Invalid: {res['invalid_rate']}%")
        m0_results[f"seed_{s}"] = res

    # Save all M0-FC40 180-case predictions
    with open(out_dir / "m0_fc40_3seed_180_eval.json", "w", encoding="utf-8") as f:
        json.dump(m0_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll M0-FC40 evaluations saved to {out_dir}")


if __name__ == "__main__":
    main()
