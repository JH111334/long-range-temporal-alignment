"""Zero-Shot Baseline Probe for Qwen2.5-0.5B-Instruct on TempUN 180 MCQA.

Compares:
1. Mode A: Raw prompt (no chat template, identical to current M0-FC40 runner).
2. Mode B: ChatML prompt (system + instruction, identical to partner Ticktack runner).
Evaluated with both Current Strict Parser and Partner Regex Parser.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_strict_mcqa_answer_current(prediction: str) -> Optional[str]:
    """Current runner parser from tempun_dataset.py."""
    cleaned = prediction.strip().strip(".()[]: \n\t")
    if cleaned in ("A", "B", "C", "D"):
        return cleaned
    lower = cleaned.lower()
    if lower in ("yes", "y"):
        return "Yes"
    if lower in ("no", "n"):
        return "No"
    match_letter = re.match(r"^([A-D])[\s\.\:\,\)]", cleaned)
    if match_letter:
        return match_letter.group(1)
    match_yn = re.match(r"^(yes|no)\b", lower)
    if match_yn:
        return "Yes" if match_yn.group(1) == "yes" else "No"
    return None


def parse_prediction_partner(gen_text: str, target: str) -> Tuple[str, str]:
    """Partner regex parser from ticktack_v2."""
    cleaned = gen_text.strip()
    is_yn = target.strip() in ["Yes", "No"]
    if is_yn:
        m = re.search(r"\b(Yes|No)\b", cleaned, re.IGNORECASE)
        if m:
            return m.group(1).capitalize(), "VALID"
        return cleaned, "INVALID"
    else:
        # Match parenthesized letter
        m = re.search(r"\(([a-d])\)", cleaned, re.IGNORECASE)
        if m:
            return m.group(1).lower(), "VALID"
        # Match option prefixes
        m = re.search(r"(?:option|answer|choice|select)\s*[:\-]?\s*\(?([a-d])\)?\b", cleaned, re.IGNORECASE)
        if m:
            return m.group(1).lower(), "VALID"
        # Match text starting with single letter e.g. 'a.', 'b:'
        m = re.match(r"^([a-d])\b", cleaned, re.IGNORECASE)
        if m:
            return m.group(1).lower(), "VALID"
        # Match isolated letter in short response
        m = re.search(r"\b([a-d])\b", cleaned, re.IGNORECASE)
        if m and len(cleaned.split()) <= 5:
            return m.group(1).lower(), "VALID"
        return cleaned, "INVALID"


def run_probe(sample_limit: Optional[int] = None):
    print("=" * 80)
    print("QWEN2.5-0.5B-INSTRUCT ZERO-SHOT FORENSIC PROBE")
    print("=" * 80)

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    revision = "7ae557604adf67be50417f59c2c2f167def9a775"

    # Find eval dataset
    matches = list(Path(".").glob("**/eval_mcqa.jsonl"))
    if not matches:
        raise FileNotFoundError("eval_mcqa.jsonl not found")
    eval_path = matches[0]
    print(f"Loading eval dataset from: {eval_path}")
    lines = eval_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(l) for l in lines if l.strip()]
    if sample_limit:
        records = records[:sample_limit]
    print(f"Total evaluation cases: {len(records)}")

    print(f"\nLoading model and tokenizer ({model_id} @ {revision[:8]})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    device = torch.device("cpu")
    model.to(device)
    print("Model loaded successfully on CPU.")

    results = {}

    for mode_name, use_chat_template in [("Mode_A_RawPrompt", False), ("Mode_B_ChatMLPrompt", True)]:
        print(f"\n--- Running Evaluation: {mode_name} ---")
        t0 = time.time()

        predictions = []
        cur_correct = 0
        cur_valid = 0
        cur_invalid = 0

        part_correct = 0
        part_valid = 0
        part_invalid = 0

        for idx, rec in enumerate(records):
            raw_prompt = rec["input"]
            target = rec["target"]

            if use_chat_template:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Please answer the question by selecting the correct option."},
                    {"role": "user", "content": raw_prompt.strip() + "\nAnswer directly with only the option letter (a, b, c, d) or Yes/No."}
                ]
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted = raw_prompt

            tok_in = tokenizer(formatted, return_tensors="pt").to(device)
            prompt_len = tok_in["input_ids"].shape[1]

            with torch.no_grad():
                gen_out = model.generate(
                    **tok_in,
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = gen_out[0, prompt_len:]
            raw_gen = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # 1. Parse with current strict parser
            cur_p = parse_strict_mcqa_answer_current(raw_gen)
            if cur_p is not None:
                cur_valid += 1
                if cur_p == target:
                    cur_correct += 1
            else:
                cur_invalid += 1

            # 2. Parse with partner parser
            part_p, part_status = parse_prediction_partner(raw_gen, target)
            if part_status == "VALID":
                part_valid += 1
                if part_p == target:
                    part_correct += 1
            else:
                part_invalid += 1

            predictions.append({
                "example_id": rec.get("example_id", f"case_{idx}"),
                "gold_target": target,
                "raw_completion": raw_gen,
                "cur_parsed": cur_p,
                "cur_correct": (cur_p == target) if cur_p else False,
                "part_parsed": part_p if part_status == "VALID" else None,
                "part_correct": (part_p == target) if part_status == "VALID" else False,
            })

            if (idx + 1) % 30 == 0 or (idx + 1) == len(records):
                print(f"  Processed {idx + 1}/{len(records)} cases... elapsed={time.time() - t0:.1f}s")

        n = len(records)
        mode_res = {
            "mode": mode_name,
            "total": n,
            "current_parser": {
                "correct": cur_correct,
                "valid": cur_valid,
                "invalid": cur_invalid,
                "strict_acc": cur_correct / n * 100.0,
                "valid_acc": (cur_correct / cur_valid * 100.0) if cur_valid > 0 else 0.0,
                "invalid_rate": cur_invalid / n * 100.0,
            },
            "partner_parser": {
                "correct": part_correct,
                "valid": part_valid,
                "invalid": part_invalid,
                "strict_acc": part_correct / n * 100.0,
                "valid_acc": (part_correct / part_valid * 100.0) if part_valid > 0 else 0.0,
                "invalid_rate": part_invalid / n * 100.0,
            },
            "sample_predictions": predictions[:5],
        }
        results[mode_name] = mode_res

        print(f"\nResults for {mode_name}:")
        print(f"  Current Strict Parser : Strict ACC = {cur_correct/n*100:.2f}% ({cur_correct}/{n}), Invalid Rate = {cur_invalid/n*100:.2f}% ({cur_invalid}/{n})")
        print(f"  Partner Regex Parser  : Strict ACC = {part_correct/n*100:.2f}% ({part_correct}/{n}), Invalid Rate = {part_invalid/n*100:.2f}% ({part_invalid}/{n})")

    out_file = Path("artifacts/audits/QWEN_BASE_ZEROSHOT_PROBE_RECEIPT.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote full probe receipt to {out_file}")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_probe(sample_limit=limit)
