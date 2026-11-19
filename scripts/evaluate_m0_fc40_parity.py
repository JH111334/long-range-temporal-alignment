"""Factored Protocol Re-Evaluation Runner for M0-FC40 Checkpoints & Base Model.

Features:
- Evaluates across P00, P01, P10, P11 defined in EVAL_PROTOCOL_MANIFEST_V2.json.
- Supports Seam modes:
  * on: Standard contract frozen seam at token_index = 14 (Option S0).
  * off: Seam disabled (continuous vector = None), Dual-Cycle MaT-LoRA active.
  * s1: Repositioned seam injected at user prompt token boundary (Option S1).
- Uses exact partner canonical regex parser from ticktack_v2.
- Captures token telemetry: first_token_id, first_token_literal, stop_reason, token length.
- Supports locked diagnostic subset (24 cases) or full 180 cases.
- Operates on Base Qwen (no adapter) or loads specified checkpoint state dict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_PKG_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from temporal_alignment.models.fone_canchi_40d import (
    compute_fixed_fone_canchi_40d,
    extract_canchi_angles_from_40d,
)
from temporal_alignment.models.dual_cycle_mat_lora import (
    DualCycleMaTLoRALinear,
    replace_mlp_with_dual_cycle_mat_lora,
    set_model_temporal_coordinates,
)
from temporal_alignment.models.gated_seam import (
    ConditioningContext,
    GatedResidualProjection,
    make_gated_embedding_hook,
)
from temporal_alignment.models.tempun_dataset import (
    TempUNExample,
    load_tempun_eval_records,
)


def partner_canonical_parser(gen_text: str, target: str) -> Tuple[Optional[str], str]:
    """Exact canonical parser from partner's ticktack_v2/eval/eval_mcqa_generative.py."""
    cleaned = gen_text.strip()
    is_yn = target.strip() in ["Yes", "No"]
    if is_yn:
        m = re.search(r"\b(Yes|No)\b", cleaned, re.IGNORECASE)
        if m:
            return m.group(1).capitalize(), "VALID"
        return None, "INVALID"
    else:
        m = re.search(r"\(([a-d])\)", cleaned, re.IGNORECASE)
        if m:
            return m.group(1).lower(), "VALID"
        m = re.search(r"(?:option|answer|choice|select)\s*[:\-]?\s*\(?([a-d])\)?\b", cleaned, re.IGNORECASE)
        if m:
            return m.group(1).lower(), "VALID"
        m = re.match(r"^([a-d])\b", cleaned, re.IGNORECASE)
        if m:
            return m.group(1).lower(), "VALID"
        m = re.search(r"\b([a-d])\b", cleaned, re.IGNORECASE)
        if m and len(cleaned.split()) <= 5:
            return m.group(1).lower(), "VALID"
        return None, "INVALID"


def build_prompt_by_protocol(prompt_raw: str, protocol_id: str, tokenizer: Any) -> Tuple[str, int]:
    """Format prompt according to EVAL_PROTOCOL_MANIFEST_V2 and compute user token offset."""
    raw = prompt_raw.strip()
    instruction = "\nAnswer directly with only the option letter (a, b, c, d) or Yes/No."

    if protocol_id == "P00":
        return raw, 14
    elif protocol_id == "P01":
        return raw + instruction, 14
    elif protocol_id == "P10":
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": raw}
        ]
        # Sys header length
        sys_prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        sys_toks = tokenizer.encode(sys_prefix, add_special_tokens=False)
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), len(sys_toks)
    elif protocol_id == "P11":
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Please answer the question by selecting the correct option."},
            {"role": "user", "content": raw + instruction}
        ]
        sys_prefix = "<|im_start|>system\nYou are a helpful assistant. Please answer the question by selecting the correct option.<|im_end|>\n<|im_start|>user\n"
        sys_toks = tokenizer.encode(sys_prefix, add_special_tokens=False)
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), len(sys_toks)
    else:
        raise ValueError(f"Unknown protocol_id: {protocol_id}")


def select_diagnostic_subset(records: List[TempUNExample], n_cases: int = 24) -> List[TempUNExample]:
    """Select locked deterministic 24-case diagnostic subset balancing Yes/No and 4-choice."""
    yn_cases = [r for r in records if r.target in ("Yes", "No")]
    mc_cases = [r for r in records if r.target in ("a", "b", "c", "d")]
    selected = yn_cases[:8] + mc_cases[:16]
    return selected[:n_cases]


def evaluate_model_on_records(
    model: nn.Module,
    tokenizer: Any,
    records: List[TempUNExample],
    protocol_id: str,
    device: torch.device,
    seam_mode: str = "on",
    cond_context: Optional[ConditioningContext] = None,
    gated_proj: Optional[GatedResidualProjection] = None,
    adapted_modules: Optional[Dict[str, DualCycleMaTLoRALinear]] = None,
    max_new_tokens: int = 8,
) -> Dict[str, Any]:
    """Evaluate model with instrumented token-level telemetry."""
    model.eval()
    results = []
    correct_count = 0
    valid_count = 0
    invalid_count = 0
    token_type_counts = Counter()

    for rec in records:
        year = rec.year
        gold = rec.target

        # Build prompt & compute user token offset
        formatted_prompt, user_token_offset = build_prompt_by_protocol(rec.prompt, protocol_id, tokenizer)
        tok_in = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        prompt_len = tok_in["input_ids"].shape[1]

        # Conditioning
        if adapted_modules is not None:
            vec40 = compute_fixed_fone_canchi_40d(year)
            theta, z = extract_canchi_angles_from_40d(vec40)
            set_model_temporal_coordinates(adapted_modules, theta, z)

            if gated_proj is not None and cond_context is not None:
                if seam_mode == "off":
                    cond_context.clear()
                elif seam_mode == "s1":
                    cond_context.injection_token_index = user_token_offset
                    vec40_t = torch.tensor(vec40, dtype=torch.float32, device=device).unsqueeze(0)
                    injected_vec = gated_proj(vec40_t).squeeze(0)
                    cond_context.set_vector(injected_vec)
                else:  # "on" (Option S0)
                    cond_context.injection_token_index = 14
                    vec40_t = torch.tensor(vec40, dtype=torch.float32, device=device).unsqueeze(0)
                    injected_vec = gated_proj(vec40_t).squeeze(0)
                    cond_context.set_vector(injected_vec)

        with torch.no_grad():
            gen_out = model.generate(
                **tok_in,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )

        seq = gen_out.sequences[0]
        gen_tokens = seq[prompt_len:].tolist()
        raw_completion = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Token telemetry
        first_token_id = gen_tokens[0] if gen_tokens else None
        first_token_lit = repr(tokenizer.decode([first_token_id])) if first_token_id is not None else "NONE"
        stop_reason = "EOS" if (tokenizer.eos_token_id in gen_tokens) else "MAX_TOKENS"

        # Parse with canonical partner parser
        parsed_ans, status = partner_canonical_parser(raw_completion, gold)
        is_valid = (status == "VALID")
        is_correct = (parsed_ans == gold) if is_valid else False

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
        if is_correct:
            correct_count += 1

        if not raw_completion:
            token_type_counts["EMPTY"] += 1
        elif raw_completion.isdigit():
            token_type_counts["NUMERIC"] += 1
        elif is_valid:
            token_type_counts["VALID_OPTION"] += 1
        else:
            token_type_counts["OTHER_INVALID"] += 1

        results.append({
            "example_id": rec.example_id,
            "stage_id": rec.stage_id,
            "year": rec.year,
            "gold_target": gold,
            "protocol_id": protocol_id,
            "seam_mode": seam_mode,
            "raw_completion": raw_completion,
            "first_token_id": first_token_id,
            "first_token_literal": first_token_lit,
            "stop_reason": stop_reason,
            "gen_length": len(gen_tokens),
            "parsed_answer": parsed_ans,
            "is_valid": is_valid,
            "is_correct": is_correct,
        })

        if cond_context is not None:
            cond_context.clear()
        if adapted_modules is not None:
            set_model_temporal_coordinates(adapted_modules, None, None)

    total = len(records)
    return {
        "protocol_id": protocol_id,
        "seam_mode": seam_mode,
        "total_cases": total,
        "correct": correct_count,
        "valid": valid_count,
        "invalid": invalid_count,
        "strict_acc": (correct_count / total * 100.0) if total > 0 else 0.0,
        "valid_acc": (correct_count / valid_count * 100.0) if valid_count > 0 else 0.0,
        "invalid_rate": (invalid_count / total * 100.0) if total > 0 else 0.0,
        "token_type_breakdown": dict(token_type_counts),
        "predictions": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate M0-FC40 protocols with telemetry")
    parser.add_argument("--eval_file", type=str, default="data/tempun/eval_mcqa.jsonl")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to M0_FC40 checkpoint .pt (if None, evaluates base model)")
    parser.add_argument("--protocols", type=str, default="P00,P01,P10,P11")
    parser.add_argument("--seam_mode", type=str, default="on", choices=["on", "off", "s1"], help="Seam conditioning mode")
    parser.add_argument("--diagnostic_subset", action="store_true", help="Run on locked 24-case subset")
    parser.add_argument("--output_file", type=str, default="artifacts/audits/M0_FC40_PROMPT_FACTORIAL_RESULT.json")
    args = parser.parse_args()

    protocols = [p.strip() for p in args.protocols.split(",")]

    eval_path = Path(args.eval_file)
    if not eval_path.exists():
        matches = list(Path(".").glob("**/eval_mcqa.jsonl"))
        if matches:
            eval_path = matches[0]
        else:
            raise FileNotFoundError(f"Cannot find eval file at {args.eval_file}")

    records = load_tempun_eval_records(eval_path)
    if args.diagnostic_subset:
        records = select_diagnostic_subset(records, 24)
        print(f"Running on DIAGNOSTIC locked subset: {len(records)} cases")
    else:
        print(f"Running on FULL eval set: {len(records)} cases")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer & config
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    revision = "7ae557604adf67be50417f59c2c2f167def9a775"
    cache_path = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision

    if cache_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(cache_path), local_files_only=True)
        config = AutoConfig.from_pretrained(str(cache_path), local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        config = AutoConfig.from_pretrained(model_id, revision=revision)

    print(f"Loading model ({model_id} @ {revision[:8]})...")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if cache_path.exists() and (cache_path / "model.safetensors").exists():
        model = AutoModelForCausalLM.from_pretrained(
            str(cache_path),
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=dtype)

    model.to(device)

    cond_context = None
    gated_proj = None
    adapted_modules = None

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        print(f"Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        adapted_modules = replace_mlp_with_dual_cycle_mat_lora(model, rank=8, alpha=16.0, num_harmonics=2)
        adapters_state = state["adapters_state"]
        for name, mod in adapted_modules.items():
            if name in adapters_state:
                mod.lora_A.data.copy_(adapters_state[name]["lora_A"].to(device))
                mod.lora_B.data.copy_(adapters_state[name]["lora_B"].to(device))
                mod.core_generator.load_state_dict(adapters_state[name]["core_state"])

        hidden_dim = model.config.hidden_size
        gated_proj = GatedResidualProjection(in_features=40, out_features=hidden_dim).to(device)
        gated_proj.load_state_dict(state["gated_proj_state"])
        cond_context = ConditioningContext(injection_token_index=14)
        make_gated_embedding_hook(cond_context)

        # Register hook on embedding module
        embed_module = model.get_input_embeddings()
        embed_module.register_forward_hook(make_gated_embedding_hook(cond_context))
        print(f"Loaded checkpoint with seam_mode={args.seam_mode}")

    results_by_protocol = {}
    for proto in protocols:
        print(f"\n--- Evaluating Protocol: {proto} (Seam={args.seam_mode}) ---")
        t0 = time.time()
        res = evaluate_model_on_records(
            model=model,
            tokenizer=tokenizer,
            records=records,
            protocol_id=proto,
            device=device,
            seam_mode=args.seam_mode,
            cond_context=cond_context,
            gated_proj=gated_proj,
            adapted_modules=adapted_modules,
        )
        print(f"Protocol {proto}: Strict ACC = {res['strict_acc']:.2f}%, Invalid Rate = {res['invalid_rate']:.2f}%, Valid ACC = {res['valid_acc']:.2f}% (elapsed={time.time()-t0:.1f}s)")
        print(f"  Token Types: {res['token_type_breakdown']}")
        results_by_protocol[proto] = res

    out_p = Path(args.output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "model": model_id,
        "revision": revision,
        "checkpoint": str(args.checkpoint) if args.checkpoint else "BASE_MODEL_FROZEN",
        "seam_mode": args.seam_mode,
        "diagnostic_subset": args.diagnostic_subset,
        "total_cases": len(records),
        "protocol_summary": {
            p: {k: v for k, v in res.items() if k != "predictions"}
            for p, res in results_by_protocol.items()
        },
        "sample_predictions": {
            p: res["predictions"] for p, res in results_by_protocol.items()
        }
    }
    out_p.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved protocol factorial results to {out_p}")


if __name__ == "__main__":
    main()
