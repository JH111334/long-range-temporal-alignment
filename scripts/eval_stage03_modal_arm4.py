"""
Stage 03 Evaluation on Modal: Arm 4 (Static MLP LoRA on Qwen2.5-3B Base)
Loads weights from modal.Volume('stage03-artifacts') at /vol/stage03_3b_static_mlp_lora_weights.pt
Runs:
1. TempUN Clean (5,292 MCQA cases) with 4-fold cyclic option debiasing (21,168 inferences)
2. 54 Historical Counterfactual Gold Pairs from grounded_counterfactual_benchmark.jsonl
Saves results to Volume and local workspace.
"""

from __future__ import annotations

import io
import gzip
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import modal

_REPO_ROOT = Path(__file__).resolve().parent.parent

cuda_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "numpy>=1.26.0",
        "safetensors",
    )
)

volume = modal.Volume.from_name("stage03-artifacts", create_if_missing=True)
app = modal.App("stage03-3b-arm4-eval", image=cuda_image)


@app.function(
    gpu="A10G",
    timeout=2400,
    volumes={"/vol": volume},
)
def evaluate_static_mlp_lora(
    eval_bytes: bytes,
    cf_bytes: bytes,
    rank: int = 8,
    alpha: float = 16.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[{datetime.now().isoformat()}] Starting Arm 4 Evaluation on Modal GPU...")
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # --- 1. Load Data ---
    with gzip.GzipFile(fileobj=io.BytesIO(eval_bytes), mode="rb") as f:
        eval_cases = [json.loads(line) for line in f if line.strip()]

    cf_raw = [json.loads(line) for line in cf_bytes.decode("utf-8").splitlines() if line.strip()]
    pairs_dict = defaultdict(list)
    for r in cf_raw:
        pairs_dict[r.get("pair_id", r.get("id", "unknown"))].append(r)
    cf_pairs = list(pairs_dict.values())

    print(f"Loaded {len(eval_cases):,} eval samples, {len(cf_pairs)} CF pairs (from {len(cf_raw)} raw items).")

    # --- 2. Temporal Representation (FoNE 32D + Can-Chi 8D = 40D) ---
    def compute_fone_32d(year: float, num_frequencies: int = 16, min_period: float = 10.0, max_period: float = 10000.0) -> List[float]:
        ln_min = math.log(min_period)
        ln_max = math.log(max_period)
        features = []
        for i in range(num_frequencies):
            period_i = math.exp(ln_min + (ln_max - ln_min) * (i / max(1, num_frequencies - 1)))
            omega = 2.0 * math.pi / period_i
            features.append(math.cos(omega * year))
            features.append(math.sin(omega * year))
        return features

    def compute_canchi_8d(year: int, t0: int = 2000) -> List[float]:
        offset = year - 4
        cycle_idx = offset % 60
        stem_idx = offset % 10
        branch_idx = offset % 12
        theta = 2.0 * math.pi * float(cycle_idx) / 60.0
        radius = 1.0 + float(year - t0) / 60.0
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        return [
            float(x), float(y), float(math.cos(theta)), float(math.sin(theta)),
            float(radius), float(stem_idx) / 10.0, float(branch_idx) / 12.0, float(cycle_idx) / 60.0
        ]

    def compute_fixed_fone_canchi_40d(year: int) -> List[float]:
        return compute_fone_32d(float(year)) + compute_canchi_8d(year)

    # --- 3. Model Architecture Modules ---
    class GatedResidualProjection(nn.Module):
        def __init__(self, in_features: int = 40, out_features: int = 2048):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
            self.norm = nn.LayerNorm(out_features, bias=False)
            self.alpha = nn.Parameter(torch.zeros(1))
            nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.linear.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            proj = self.linear(x)
            normed = self.norm(proj)
            return self.alpha * normed

    def inject_seam_prefill_safe(model, seam_proj, input_ids: torch.Tensor, v40: torch.Tensor, token_idx: int = 14):
        embed_tokens = model.get_input_embeddings()
        inputs_embeds = embed_tokens(input_ids)
        B, S, D = inputs_embeds.shape
        if S > token_idx:
            inj = seam_proj(v40.to(inputs_embeds.device, dtype=inputs_embeds.dtype))
            delta = torch.zeros_like(inputs_embeds)
            delta[:, token_idx, :] = inj
            inputs_embeds = inputs_embeds + delta
        return inputs_embeds

    class StaticLoRALinear(nn.Module):
        def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
            super().__init__()
            self.in_features = original_linear.in_features
            self.out_features = original_linear.out_features
            self.rank = rank
            self.scaling = alpha / float(rank)
            self.base_layer = original_linear
            self.base_layer.weight.requires_grad = False
            if self.base_layer.bias is not None:
                self.base_layer.bias.requires_grad = False

            device = original_linear.weight.device
            dtype = original_linear.weight.dtype

            self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=dtype))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, rank, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

            self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
            self.enabled: bool = True

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base_out = self.base_layer(x)
            if not self.enabled:
                return base_out
            x_a = F.linear(x, self.lora_A)
            lora_out = F.linear(self.dropout(x_a), self.lora_B) * self.scaling
            return base_out + lora_out

    # --- 4. Initialize Base Model & Attach Static LoRA to 108 MLP Modules ---
    MODEL_ID = "Qwen/Qwen2.5-3B"
    print(f"Loading Base Model: {MODEL_ID} in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    for p in model.parameters():
        p.requires_grad = False

    seam_projection = GatedResidualProjection(in_features=40, out_features=2048).to(model.device, dtype=torch.bfloat16)

    target_suffixes = ("gate_proj", "up_proj", "down_proj")
    mlp_modules = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(name.endswith(suffix) for suffix in target_suffixes):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            c_name = parts[-1]
            adapted = StaticLoRALinear(module, rank=rank, alpha=alpha, dropout=0.0)
            setattr(parent, c_name, adapted)
            mlp_modules.append(adapted)

    print(f"Attached Static LoRA to {len(mlp_modules)} MLP modules.")

    # --- 5. Load Trained Weights from Persistent Volume ---
    weights_path = Path("/vol/stage03_3b_static_mlp_lora_weights.pt")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found at {weights_path}")

    print(f"Loading checkpoint from {weights_path} ({weights_path.stat().st_size:,} bytes)...")
    ckpt = torch.load(weights_path, map_location="cpu")

    # Load LoRA parameters
    lora_state = ckpt.get("lora_state", {})
    model_state = model.state_dict()
    loaded_lora_keys = 0
    for k, v in lora_state.items():
        if k in model_state:
            model_state[k].copy_(v.to(model_state[k].device, dtype=model_state[k].dtype))
            loaded_lora_keys += 1
        else:
            print(f"Warning: Key {k} in ckpt not in model!")
    print(f"Loaded {loaded_lora_keys}/{len(lora_state)} LoRA parameter tensors.")

    # Load Seam Projection parameters
    seam_state = ckpt.get("seam_projection", {})
    seam_projection.load_state_dict({k: v.to(model.device, dtype=torch.bfloat16) for k, v in seam_state.items()})
    print(f"Loaded Seam Projection. Final alpha: {seam_projection.alpha.item():.5f}")

    train_metadata = ckpt.get("metadata", {})
    print(f"Training metadata: {train_metadata}")

    # --- 6. Decoupled Memory-Safe Evaluation ---
    print("\n========== STARTING EVALUATION ON TEMPUN CLEAN (5,292 CASES) ==========")
    model.eval()
    seam_projection.eval()
    tokenizer.padding_side = "left"

    def extract_token_ids(tok, words):
        ids = []
        for w in words:
            for prefix in ["", " "]:
                enc = tok.encode(prefix + w, add_special_tokens=False)
                if enc and len(enc) == 1:
                    ids.append(enc[0])
        return list(dict.fromkeys(ids))

    yes_ids = extract_token_ids(tokenizer, ["yes", "Yes", "YES"])
    no_ids = extract_token_ids(tokenizer, ["no", "No", "NO"])
    choice_ids = {k: extract_token_ids(tokenizer, [k, k.upper()]) for k in ["a", "b", "c", "d"]}

    def score_prediction_row(logits_row, target_text, prior_bias_rel=0.0):
        t_norm = target_text.strip().lower()
        if t_norm in ["yes", "no"]:
            s_yes = (max(logits_row[idx].item() for idx in yes_ids) if yes_ids else -1e9) + prior_bias_rel
            s_no = max(logits_row[idx].item() for idx in no_ids) if no_ids else -1e9
            pred = "yes" if s_yes >= s_no else "no"
        else:
            s = {k: (max(logits_row[idx].item() for idx in c_ids) if c_ids else -1e9) for k, c_ids in choice_ids.items()}
            pred = max(s, key=s.get)
        return pred, t_norm, int(pred == t_norm)

    def generate_option_permutations(prompt_text: str, target_text: str):
        tgt = target_text.strip().lower()
        if tgt in ["yes", "no"]:
            p0 = prompt_text
            p1 = re.sub(r"Answer as Yes or No", "Answer as No or Yes", prompt_text, flags=re.IGNORECASE)
            if p1 == p0:
                p1 = re.sub(r"Yes\s*(or|/)\s*No", "No or Yes", prompt_text, flags=re.IGNORECASE)
            return [{"prompt": p0, "target": tgt, "rotation": 0}, {"prompt": p1, "target": tgt, "rotation": 1}]
        suffix = ""
        m_suffix = re.search(r"(\s*\n?\s*Answer:\s*)$", prompt_text, re.IGNORECASE)
        if m_suffix:
            suffix = m_suffix.group(1)
            base_prompt = prompt_text[:len(prompt_text) - len(suffix)]
        else:
            base_prompt = prompt_text
        pattern = re.compile(r"^(.*Options:\s*)\(a\)\s*(.*?)\s*\(b\)\s*(.*?)\s*\(c\)\s*(.*?)\s*\(d\)\s*(.*?)$", re.DOTALL)
        m = pattern.search(base_prompt)
        if not m:
            return [{"prompt": prompt_text, "target": tgt, "rotation": 0}]
        prefix = m.group(1)
        opts = [m.group(2).strip(), m.group(3).strip(), m.group(4).strip(), m.group(5).strip()]
        letters = ["a", "b", "c", "d"]
        if tgt not in letters:
            return [{"prompt": prompt_text, "target": tgt, "rotation": 0}]
        t_idx = letters.index(tgt)
        perms = []
        for k in range(4):
            r_opts = [opts[(j + k) % 4] for j in range(4)]
            r_tgt = letters[(t_idx - k) % 4]
            perms.append({"prompt": f"{prefix}(a) {r_opts[0]} (b) {r_opts[1]} (c) {r_opts[2]} (d) {r_opts[3]}{suffix}", "target": r_tgt, "rotation": k})
        return perms

    eval_batch_size = 4
    total_eval = len(eval_cases)
    case_results = {}
    can_correct, strict_correct, deb_sum = 0, 0, 0.0
    task_scores = defaultdict(lambda: {"canonical": 0, "debiased_sum": 0.0, "total": 0})
    t_eval_start = time.time()

    for b_start in range(0, total_eval, eval_batch_size):
        b_cases = eval_cases[b_start:b_start + eval_batch_size]
        all_prompts, all_targets, all_years, case_perm_slices = [], [], [], []
        for case in b_cases:
            yr = case.get("year")
            perms = generate_option_permutations(case["input"], case["target"])
            s_idx = len(all_prompts)
            for p in perms:
                all_prompts.append(p["prompt"])
                all_targets.append(p["target"])
                all_years.append(yr)
            e_idx = len(all_prompts)
            case_perm_slices.append((case, (s_idx, e_idx)))

        formatted = [f"Question: {p.strip()}\nAnswer:" if not p.strip().endswith("Answer:") else p.strip() for p in all_prompts]
        inputs = tokenizer(formatted, return_tensors="pt", padding=True).to(model.device)
        v40_list = [compute_fixed_fone_canchi_40d(int(y) if y is not None else 2000) for y in all_years]
        v40_t = torch.tensor(v40_list, dtype=torch.float32, device=model.device)

        with torch.no_grad():
            inputs_embeds = inject_seam_prefill_safe(model, seam_projection, inputs["input_ids"], v40_t, token_idx=14)
            out = model(inputs_embeds=inputs_embeds, attention_mask=inputs["attention_mask"])
            logits = out.logits[:, -1, :]

        b_results = []
        for idx, (p_text, gold) in enumerate(zip(all_prompts, all_targets)):
            pred, t_norm, is_corr = score_prediction_row(logits[idx], gold)
            b_results.append({"pred": pred, "gold": t_norm, "is_correct": is_corr})

        for case, (s_idx, e_idx) in case_perm_slices:
            cid = case.get("example_id", f"case_{len(case_results)}")
            task_id = case.get("task_id", "Unknown")
            outcomes = [r["is_correct"] for r in b_results[s_idx:e_idx]]
            is_can = outcomes[0]
            d_score = float(np.mean(outcomes))
            is_st = all(outcomes)
            if is_can:
                can_correct += 1
            if is_st:
                strict_correct += 1
            deb_sum += d_score
            task_scores[task_id]["canonical"] += int(is_can)
            task_scores[task_id]["debiased_sum"] += d_score
            task_scores[task_id]["total"] += 1
            safe_key = f"{len(case_results)}_{cid}"
            case_results[safe_key] = {"cid": cid, "is_correct": is_can, "debiased_score": d_score, "is_strict": is_st, "task_id": task_id}

        done = min(b_start + eval_batch_size, total_eval)
        if done % 500 < eval_batch_size or done == total_eval:
            elapsed = time.time() - t_eval_start
            print(f"Eval: {done}/{total_eval} ({done/elapsed:.1f} cases/s)...")
            torch.cuda.empty_cache()

    task_summary = {}
    for t, v in sorted(task_scores.items()):
        c = v["total"]
        task_summary[t] = {
            "canonical_acc": round(v["canonical"] / c * 100, 2),
            "debiased_mean_acc": round(v["debiased_sum"] / c * 100, 2),
            "count": c,
        }

    # Evaluate 54 Counterfactual Pairs
    print("\n========== STARTING EVALUATION ON 54 COUNTERFACTUAL PAIRS ==========")
    both_c4 = 0
    cf_details = []
    for pair in cf_pairs:
        if len(pair) >= 2:
            q1, q2 = pair[0], pair[1]
            f1 = [f"Question: {q1['input'].strip()}\nAnswer:"]
            f2 = [f"Question: {q2['input'].strip()}\nAnswer:"]
            i1 = tokenizer(f1, return_tensors="pt").to(model.device)
            i2 = tokenizer(f2, return_tensors="pt").to(model.device)
            v1 = torch.tensor([compute_fixed_fone_canchi_40d(int(q1.get("year", 2000)))], dtype=torch.float32, device=model.device)
            v2 = torch.tensor([compute_fixed_fone_canchi_40d(int(q2.get("year", 2000)))], dtype=torch.float32, device=model.device)
            with torch.no_grad():
                emb1 = inject_seam_prefill_safe(model, seam_projection, i1["input_ids"], v1, token_idx=14)
                emb2 = inject_seam_prefill_safe(model, seam_projection, i2["input_ids"], v2, token_idx=14)
                o1 = model(inputs_embeds=emb1, attention_mask=i1["attention_mask"]).logits[:, -1, :]
                o2 = model(inputs_embeds=emb2, attention_mask=i2["attention_mask"]).logits[:, -1, :]
                p1, g1, r1 = score_prediction_row(o1[0], q1["target"])
                p2, g2, r2 = score_prediction_row(o2[0], q2["target"])
            is_both = bool(r1 and r2)
            if is_both:
                both_c4 += 1
            cf_details.append({
                "pair_id": q1.get("pair_id", "unknown"),
                "q1_pred": p1, "q1_gold": g1, "q1_correct": bool(r1),
                "q2_pred": p2, "q2_gold": g2, "q2_correct": bool(r2),
                "both_correct": is_both,
            })

    cf_rate = round(both_c4 / len(cf_pairs) * 100, 2) if cf_pairs else 0.0
    print(f"Arm 4 Counterfactual 54P BothCorrect: {both_c4}/{len(cf_pairs)} ({cf_rate}%)")

    results_dict = {
        "arm_name": "Arm 4: Static MLP LoRA (Qwen2.5-3B Base)",
        "canonical_acc": round(can_correct / total_eval * 100.0, 2),
        "debiased_mean_acc": round(deb_sum / total_eval * 100.0, 2),
        "strict_acc": round(strict_correct / total_eval * 100.0, 2),
        "total": total_eval,
        "task_summary": task_summary,
        "cf_54p": {
            "both_correct": both_c4,
            "total": len(cf_pairs),
            "rate": cf_rate,
        },
        "case_results": case_results,
    }

    summary_dict = {
        "timestamp": datetime.now().isoformat(),
        "arm_name": "Arm 4: Static MLP LoRA (Qwen2.5-3B Base)",
        "model_id": MODEL_ID,
        "train_metadata": train_metadata,
        "canonical_acc": results_dict["canonical_acc"],
        "debiased_mean_acc": results_dict["debiased_mean_acc"],
        "strict_acc": results_dict["strict_acc"],
        "cf_54p_both_correct": both_c4,
        "cf_54p_rate": cf_rate,
        "task_summary": task_summary,
    }

    print("\n========== FINAL ARM 4 METRICS SUMMARY ==========")
    print(f"- Canonical Accuracy: {results_dict['canonical_acc']}%")
    print(f"- Debiased Mean Accuracy: {results_dict['debiased_mean_acc']}%")
    print(f"- Strict Accuracy (4-fold): {results_dict['strict_acc']}%")
    print(f"- Counterfactual 54P BothCorrect: {both_c4}/{len(cf_pairs)} ({cf_rate}%)")
    print(f"- Task Breakdown: {task_summary}")

    # Save to Volume
    try:
        vol_res = Path("/vol/stage03_3b_static_mlp_lora_results.json")
        vol_sum = Path("/vol/stage03_3b_static_mlp_lora_summary.json")
        vol_res.write_text(json.dumps(results_dict, indent=2), encoding="utf-8")
        vol_sum.write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
        volume.commit()
        print("Committed results and summary to Modal Volume.")
    except Exception as e:
        print("Warning: Volume write error:", e)

    return results_dict, summary_dict


@app.local_entrypoint()
def main():
    print(f"[{datetime.now().isoformat()}] Launching Stage 03 Arm 4 Evaluation on Modal GPU...")

    eval_path = _REPO_ROOT / "data" / "tempun_full_clean" / "eval_5292_clean.jsonl.gz"
    cf_path = _REPO_ROOT / "data" / "counterfactual" / "grounded_counterfactual_benchmark.jsonl"

    if not eval_path.exists() or not cf_path.exists():
        print(f"Error: Missing local dataset files: eval={eval_path.exists()}, cf={cf_path.exists()}")
        sys.exit(1)

    eval_bytes = eval_path.read_bytes()
    cf_bytes = cf_path.read_bytes()
    print(f"Loaded eval ({len(eval_bytes):,} bytes), CF ({len(cf_bytes):,} bytes).")

    res, summary = evaluate_static_mlp_lora.remote(
        eval_bytes=eval_bytes,
        cf_bytes=cf_bytes,
        rank=8,
        alpha=16.0,
    )

    out_dir_1 = _REPO_ROOT / "artifacts" / "kaggle_outputs" / "job4_static_mlp_lora"
    out_dir_2 = _REPO_ROOT / "Documents" / "Documenting" / "Results" / "STAGE03_3B_V2_20261026"
    out_dir_1.mkdir(parents=True, exist_ok=True)
    out_dir_2.mkdir(parents=True, exist_ok=True)

    # Save results
    (out_dir_1 / "stage03_3b_static_mlp_lora_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out_dir_1 / "stage03_3b_static_mlp_lora_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir_2 / "stage03_3b_static_mlp_lora_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out_dir_2 / "stage03_3b_static_mlp_lora_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Successfully saved all Arm 4 results and summary locally!")
    print(f"Canonical Acc: {res['canonical_acc']}% | Debiased: {res['debiased_mean_acc']}% | Strict: {res['strict_acc']}%")
    print(f"CF 54P: {res['cf_54p']['both_correct']}/{res['cf_54p']['total']} ({res['cf_54p']['rate']}%)")
