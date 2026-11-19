"""Run Stage 03 Arm 4 (Static MLP LoRA) on Modal GPU (NVIDIA A10G).

Features:
- Single-Variable Ablation: Standard Static LoRA on all 108 MLP modules
  (gate_proj, up_proj, down_proj on all 36 layers, r=8, alpha=16.0)
  + Safe Gated Seam 40D at prefill token 14, NO dynamic core generator F_core(theta, z).
- Fixed Batch Size: micro_batch=16, grad_accum=1 (fixed effective batch size 16), 1 epoch.
- Persistent Modal Volume: Saves weights, metrics, and summary to /vol/ directly with volume.commit().
- Decoupled Memory-Safe Evaluation: TempUN Clean 5,292 MCQA (Canonical, 4-fold Debiased,
  Strict Acc, 6-task breakdown) + 54 Counterfactual Pairs.
- Returns weights, full evaluation JSON, and summary, saving directly to local workspace.
"""

from __future__ import annotations

import io
import gzip
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

_REPO_ROOT = Path(__file__).resolve().parents[1]

app = modal.App("stage03-qwen3b-static-mlp-lora")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "scipy", "numpy")
)

volume = modal.Volume.from_name("stage03-artifacts", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=7200, volumes={"/vol": volume})
def train_and_evaluate_static_mlp_lora(
    train_bytes: bytes,
    eval_bytes: bytes,
    cf_bytes: bytes,
    epochs: int = 1,
    lr: float = 2e-4,
    micro_batch: int = 16,
    grad_accum: int = 1,
    max_len: int = 128,
    rank: int = 8,
    alpha: float = 16.0,
) -> Tuple[Dict[str, Any], Dict[str, Any], bytes]:
    import gc
    import random
    import re
    from collections import defaultdict
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForCausalLM

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    set_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[{datetime.now().isoformat()}] Modal Container Started on: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")

    # --- 1. Decompress & Load Datasets in Memory ---
    print("Decompressing datasets in container memory...")
    with gzip.GzipFile(fileobj=io.BytesIO(train_bytes), mode="rb") as f:
        train_cases = [json.loads(line) for line in f if line.strip()]
    with gzip.GzipFile(fileobj=io.BytesIO(eval_bytes), mode="rb") as f:
        eval_cases = [json.loads(line) for line in f if line.strip()]
    cf_pairs = [json.loads(line) for line in cf_bytes.decode("utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(train_cases):,} train samples, {len(eval_cases):,} eval samples, {len(cf_pairs)} CF pairs.")

    # --- 2. FoNE x Can-Chi 40D Feature Generator ---
    def compute_fone_32d(year: float, num_frequencies: int = 16, min_period: float = 10.0, max_period: float = 10000.0) -> List[float]:
        ln_min = math.log(min_period)
        ln_max = math.log(max_period)
        step = (ln_max - ln_min) / (num_frequencies - 1) if num_frequencies > 1 else 0.0
        features = []
        for k in range(num_frequencies):
            period = math.exp(ln_min + k * step)
            omega = 2.0 * math.pi / period
            features.append(math.cos(omega * year))
            features.append(math.sin(omega * year))
        return features

    def compute_canchi_8d(year: int, t0: int = 1900) -> List[float]:
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

    # --- 3. Architecture Modules ---
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
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    seen_ids = set()
    trainable_params = []
    for p in (list(model.parameters()) + list(seam_projection.parameters())):
        if p.requires_grad and id(p) not in seen_ids:
            seen_ids.add(id(p))
            trainable_params.append(p)

    n_train = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {n_train:,} ({n_train/3.09e9*100:.3f}% of total 3.09B model).")

    # --- 5. Training Dataset & Dataloader ---
    class Stage03StaticMLPDataset(Dataset):
        def __init__(self, samples, tok, max_l=128):
            self.examples = []
            for s in samples:
                query = str(s.get("input", ""))
                ans = str(s.get("target", ""))
                year = int(s.get("year", 2000))
                task_id = str(s.get("task_id", "Unknown"))

                prompt_text = f"Question: {query.strip()}\nAnswer:"
                full_text = f"{prompt_text} {ans.strip()}"
                enc = tok(full_text, truncation=True, max_length=max_l, return_tensors="pt")
                input_ids = enc["input_ids"].squeeze(0)

                prefix_enc = tok(prompt_text, truncation=True, max_length=max_l, return_tensors="pt")
                prefix_len = prefix_enc["input_ids"].size(1)

                labels = input_ids.clone()
                labels[:min(prefix_len, len(labels))] = -100

                v40 = compute_fixed_fone_canchi_40d(year)
                is_rel = int(task_id == "Rel" or ans.strip().lower() in ["yes", "no"])

                self.examples.append({
                    "input_ids": input_ids,
                    "labels": labels,
                    "v40": v40,
                    "year": year,
                    "is_rel": is_rel,
                })

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            return self.examples[idx]

    def collate_fn(batch, pad_token_id):
        max_l = max(len(x["input_ids"]) for x in batch)
        b_in, b_lbl, b_v40, b_rel = [], [], [], []
        for x in batch:
            pad_len = max_l - len(x["input_ids"])
            inp = torch.cat([x["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            lbl = torch.cat([x["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            b_in.append(inp)
            b_lbl.append(lbl)
            b_v40.append(x["v40"])
            b_rel.append(x["is_rel"])
        return {
            "input_ids": torch.stack(b_in),
            "labels": torch.stack(b_lbl),
            "v40": torch.tensor(b_v40, dtype=torch.float32),
            "is_rel": torch.tensor(b_rel, dtype=torch.float32),
        }

    # --- 6. Training Loop (Fixed Batch Size 16, 1 Epoch) ---
    print("\n========== STARTING ARM 4 TRAINING (STATIC MLP LORA, 3B) ==========")
    model.train()
    seam_projection.train()
    tokenizer.padding_side = "right"

    train_ds = Stage03StaticMLPDataset(train_cases, tokenizer, max_l=max_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=micro_batch,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    step = 0
    accum_loss = 0.0
    t_start = time.time()
    total_steps = len(train_loader) // grad_accum

    for epoch in range(epochs):
        for b_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            v40 = batch["v40"].to(model.device)

            inputs_embeds = inject_seam_prefill_safe(model, seam_projection, input_ids, v40, token_idx=14)

            outputs = model(inputs_embeds=inputs_embeds, labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()
            accum_loss += loss.item() * grad_accum

            if (b_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                if step % 250 == 0 or step == total_steps:
                    elapsed = time.time() - t_start
                    steps_per_sec = step / elapsed if elapsed > 0 else 0
                    print(f"Epoch {epoch+1} | Step {step}/{total_steps} | Loss: {accum_loss/250.0:.4f} | Seam alpha: {seam_projection.alpha.item():.5f} | Speed: {steps_per_sec:.1f} steps/s")
                    accum_loss = 0.0
                if step % 500 == 0:
                    torch.cuda.empty_cache()

    train_time = time.time() - t_start
    print(f"Training completed in {train_time:.1f}s ({train_time/60.0:.2f} minutes).")

    # Serialize weights
    ckpt = {
        "lora_state": {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k},
        "seam_projection": {k: v.cpu() for k, v in seam_projection.state_dict().items()},
        "metadata": {
            "architecture": "StaticLoRA_MLP108_Seam40D",
            "rank": rank,
            "alpha": alpha,
            "steps": step,
            "final_alpha": float(seam_projection.alpha.item()),
            "training_time_s": train_time,
            "device": torch.cuda.get_device_name(0),
        }
    }
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    weights_bytes = buf.getvalue()

    # Save directly to Modal Volume for persistence
    try:
        vol_weights_path = Path("/vol/stage03_3b_static_mlp_lora_weights.pt")
        vol_weights_path.write_bytes(weights_bytes)
        volume.commit()
        print(f"Committed weights to Modal Volume: {vol_weights_path} ({len(weights_bytes):,} bytes)")
    except Exception as e:
        print("Warning: Volume write error:", e)

    # Deallocate optimizer before evaluation (Strict Zero OOM)
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    # --- 7. Decoupled Memory-Safe Evaluation ---
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
    for pair in cf_pairs:
        if len(pair) >= 2:
            q1, q2 = pair[0], pair[1]
            f1 = [f"Question: {q1['input'].strip()}\nAnswer:"]
            f2 = [f"Question: {q2['input'].strip()}\nAnswer:"]
            i1 = tokenizer(f1, return_tensors="pt").to(model.device)
            i2 = tokenizer(f2, return_tensors="pt").to(model.device)
            v1 = torch.tensor([compute_fixed_fone_canchi_40d(q1.get("year", 2000))], dtype=torch.float32, device=model.device)
            v2 = torch.tensor([compute_fixed_fone_canchi_40d(q2.get("year", 2000))], dtype=torch.float32, device=model.device)
            with torch.no_grad():
                emb1 = inject_seam_prefill_safe(model, seam_projection, i1["input_ids"], v1, token_idx=14)
                emb2 = inject_seam_prefill_safe(model, seam_projection, i2["input_ids"], v2, token_idx=14)
                o1 = model(inputs_embeds=emb1, attention_mask=i1["attention_mask"]).logits[:, -1, :]
                o2 = model(inputs_embeds=emb2, attention_mask=i2["attention_mask"]).logits[:, -1, :]
                r1 = score_prediction_row(o1[0], q1["target"])[2]
                r2 = score_prediction_row(o2[0], q2["target"])[2]
            if r1 and r2:
                both_c4 += 1

    cf_rate = round(both_c4 / len(cf_pairs) * 100, 2) if cf_pairs else 0.0
    print(f"Arm 4 Counterfactual 54P BothCorrect: {both_c4}/{len(cf_pairs)} ({cf_rate}%)")

    results_dict = {
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
        "arm": "Arm 4: Static MLP LoRA (108 modules) + FoNE 40D Safe Seam",
        "canonical_acc": results_dict["canonical_acc"],
        "debiased_mean_acc": results_dict["debiased_mean_acc"],
        "strict_acc": results_dict["strict_acc"],
        "cf_54p_both_correct": both_c4,
        "cf_54p_rate": cf_rate,
        "task_breakdown": task_summary,
        "train_metadata": ckpt["metadata"],
    }

    # Save to Modal Volume
    try:
        vol_res_path = Path("/vol/stage03_3b_static_mlp_lora_results.json")
        with open(vol_res_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2)
        vol_sum_path = Path("/vol/training_summary.json")
        with open(vol_sum_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)
        volume.commit()
        print("Committed results and summary to Modal Volume.")
    except Exception as e:
        print("Warning: Volume write results error:", e)

    print("\n========== FINAL ARM 4 METRICS SUMMARY ==========")
    print(f"- Canonical Accuracy: {results_dict['canonical_acc']}%")
    print(f"- Debiased Mean Accuracy: {results_dict['debiased_mean_acc']}%")
    print(f"- Strict Accuracy (4-fold): {results_dict['strict_acc']}%")
    print(f"- Counterfactual 54P BothCorrect: {both_c4}/{len(cf_pairs)} ({cf_rate}%)")
    print(f"- Task Breakdown: {task_summary}")

    return results_dict, summary_dict, weights_bytes


@app.local_entrypoint()
def main():
    print(f"[{datetime.now().isoformat()}] Launching Stage 03 Arm 4 (Static MLP LoRA) on Modal GPU...")

    train_path = _REPO_ROOT / "data" / "tempun_full_clean" / "train_104k_clean.jsonl.gz"
    eval_path = _REPO_ROOT / "data" / "tempun_full_clean" / "eval_5292_clean.jsonl.gz"
    cf_path = _REPO_ROOT / "data" / "counterfactual" / "COUNTERFACTUAL_GOLD_AUDIT.jsonl"

    if not train_path.exists() or not eval_path.exists() or not cf_path.exists():
        print("Error: Missing local dataset files!")
        sys.exit(1)

    train_bytes = train_path.read_bytes()
    eval_bytes = eval_path.read_bytes()
    cf_bytes = cf_path.read_bytes()
    print(f"Loaded train ({len(train_bytes):,} bytes), eval ({len(eval_bytes):,} bytes), CF ({len(cf_bytes):,} bytes).")

    out_dir = _REPO_ROOT / "artifacts" / "kaggle_outputs" / "job4_static_mlp_lora"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = _REPO_ROOT / "Documents" / "Documenting" / "Results" / "STAGE03_3B_V2_20261026"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Executing remote training & evaluation on Modal A10G...")
    t0 = time.time()
    results_dict, summary_dict, weights_bytes = train_and_evaluate_static_mlp_lora.remote(
        train_bytes=train_bytes,
        eval_bytes=eval_bytes,
        cf_bytes=cf_bytes,
        epochs=1,
        lr=2e-4,
        micro_batch=16,
        grad_accum=1,
        max_len=128,
        rank=8,
        alpha=16.0,
    )
    total_runtime = time.time() - t0
    print(f"Remote execution completed in {total_runtime:.1f}s ({total_runtime/60.0:.2f} mins).")

    # Save artifacts locally
    weights_path = out_dir / "stage03_3b_static_mlp_lora_weights.pt"
    weights_path.write_bytes(weights_bytes)
    print(f"Saved weights: {weights_path} ({len(weights_bytes):,} bytes)")

    results_path = out_dir / "stage03_3b_static_mlp_lora_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Saved results: {results_path} ({results_path.stat().st_size:,} bytes)")

    summary_path = out_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Saved summary: {summary_path}")

    # Also copy to Documenting/Results/
    dest_results = results_dir / "stage03_3b_static_mlp_lora_results.json"
    with open(dest_results, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Copied to Documenting/Results: {dest_results}")

    # Run four-arm McNemar paired analysis
    print("\nTriggering four-arm McNemar analysis...")
    import subprocess
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "audit_stage03_3b_four_arm_comparison.py"),
        "--job4_file", str(results_path),
    ]
    subprocess.run(cmd, check=True)
    print("Four-arm McNemar analysis completed successfully.")


if __name__ == "__main__":
    main()
