"""Run Stage 03 Targeted Continual Pretraining (5 Epochs) on Modal GPU (NVIDIA A10G).

Scale-Up Factorial Ablation Matrix (5 Epochs, Effective Batch Size 128):
- Arm 4: Targeted Static MLP LoRA (108 MLP modules, Static LoRA r=8 alpha=16.0 + Gated Seam 40D + Targeted Multi-Loss: NTP + 0.5*Intra + 0.5*Inter)
- Arm 5: Targeted MaT-LoRA (108 MLP modules, Dual-Cycle Dynamic Core F_core(theta, z) + Gated Seam 40D + Targeted Multi-Loss: NTP + 0.5*Intra + 0.5*Inter)

Training Hyperparameters:
- Epochs: 5
- Micro-batch size: 32
- Gradient Accumulation: 4 (Effective batch size = 128, generating 992 negative pairs per micro-batch forward pass)
- Optimizer: AdamW, lr=2e-4, weight_decay=0.01, max_grad_norm=1.0
- Shuffling: Reshuffled at the start of each epoch
- Checkpoint persistence: Modal Volume 'stage03-artifacts' (/vol), committed after each epoch

Evaluations (Full Spectrum):
- TempUN Clean (5,292 MCQA cases, 4-fold cyclic permutations = 21,168 inferences)
- Counterfactual V1 (54 historical milestone pairs = 108 inferences)
- Counterfactual V2 (852 temporal relational inverted pairs = 1,704 inferences)
"""

from __future__ import annotations

import argparse
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
from typing import Any, Dict, List, Optional, Tuple

import modal

_REPO_ROOT = Path(__file__).resolve().parents[1]

app = modal.App("stage03-qwen3b-targeted-continual-5epochs")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "scipy", "numpy")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

volume = modal.Volume.from_name("stage03-artifacts", create_if_missing=True)


# =========================================================================
# MODAL REMOTE WORKER FUNCTION
# =========================================================================

@app.function(image=image, gpu="A10G", timeout=18000, volumes={"/vol": volume})
def train_and_evaluate_targeted_5epochs(
    arm_type: str,  # 'static_mlp' or 'matlora'
    train_bytes: bytes,
    eval_bytes: bytes,
    cf_bytes: bytes,
    epochs: int = 5,
    lr: float = 2e-4,
    micro_batch: int = 32,
    grad_accum: int = 4,  # Effective batch size = 128
    max_len: int = 128,
    rank: int = 8,
    alpha: float = 16.0,
    margin_inter: float = 0.1,
    lambda_intra: float = 0.5,
    lambda_inter: float = 0.5,
    force_retrain: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], bytes]:
    import gc
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n=======================================================")
    print(f"STARTING 5-EPOCH TARGETED CONTINUAL PRETRAINING: ARM [{arm_type.upper()}]")
    print(f"Epochs: {epochs} | Micro-Batch: {micro_batch} | Grad-Accum: {grad_accum} -> Effective Batch: {micro_batch * grad_accum}")
    print(f"Loss formulation: L_NTP + {lambda_intra} * L_intra + {lambda_inter} * L_inter (margin={margin_inter})")
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")
    print("=======================================================\n")

    # Set deterministic seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # --- 1. Decompress Datasets ---
    with gzip.GzipFile(fileobj=io.BytesIO(train_bytes), mode="rb") as f:
        train_cases = [json.loads(line) for line in f if line.strip()]
    with gzip.GzipFile(fileobj=io.BytesIO(eval_bytes), mode="rb") as f:
        eval_cases = [json.loads(line) for line in f if line.strip()]
    cf_items = [json.loads(line) for line in cf_bytes.decode("utf-8").splitlines() if line.strip()]
    cf_pairs_dict = defaultdict(list)
    for item in cf_items:
        cf_pairs_dict[item["pair_id"]].append(item)
    cf_pairs = list(cf_pairs_dict.values())
    print(f"Loaded {len(train_cases):,} train samples, {len(eval_cases):,} eval samples, {len(cf_pairs)} CF V1 pairs.")

    # --- 2. Temporal Representation Functions (FoNE 32D + Can Chi 8D = 40D) ---
    def compute_fone_32d(year_f: float) -> List[float]:
        periods = [10.0 * (1000.0 ** (i / 15.0)) for i in range(16)]
        feats = []
        for p in periods:
            omega = 2.0 * math.pi / p
            feats.append(math.sin(omega * year_f))
            feats.append(math.cos(omega * year_f))
        return feats

    def compute_canchi_8d(year_int: int) -> List[float]:
        offset = year_int - 4
        stem_idx = offset % 10
        branch_idx = offset % 12
        cycle_idx = offset % 60
        theta_stem = 2.0 * math.pi * float(stem_idx) / 10.0
        theta_branch = 2.0 * math.pi * float(branch_idx) / 12.0
        theta_cycle = 2.0 * math.pi * float(cycle_idx) / 60.0
        radius = 1.0 + float(year_int - 2000) / 60.0
        return [
            math.sin(theta_stem), math.cos(theta_stem),
            math.sin(theta_branch), math.cos(theta_branch),
            math.sin(theta_cycle), math.cos(theta_cycle),
            float(radius), float(cycle_idx) / 60.0
        ]

    def compute_fixed_fone_canchi_40d(year: int) -> List[float]:
        return compute_fone_32d(float(year)) + compute_canchi_8d(year)

    def compute_theta_z(year: int) -> Tuple[float, float]:
        offset = year - 3
        cycle = offset % 60
        theta = float(cycle) * 6.0 * (math.pi / 180.0)
        z = float(year - 1947) / 75.0
        return theta, z

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

    # Arm 4: Static Standard LoRA Linear
    class StaticLoRALinear(nn.Module):
        def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base_out = self.base_layer(x)
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            return base_out + lora_out

    # Arm 5: DualCycleCoreGenerator & Generalized MaT-LoRA Linear
    class DualCycleCoreGenerator(nn.Module):
        def __init__(self, rank: int = 8, num_harmonics: int = 2):
            super().__init__()
            self.rank = rank
            self.num_harmonics = num_harmonics
            self.register_buffer("f_base", torch.eye(rank, dtype=torch.float32))
            std = 1.0 / math.sqrt(rank)
            self.cos_basis = nn.Parameter(torch.randn(num_harmonics, rank, rank) * (std * 0.1))
            self.sin_basis = nn.Parameter(torch.randn(num_harmonics, rank, rank) * (std * 0.1))
            self.epoch_linear = nn.Parameter(torch.randn(rank, rank) * (std * 0.1))
            self.epoch_quad = nn.Parameter(torch.randn(rank, rank) * (std * 0.01))

        def forward(self, theta: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            B = theta.shape[0]
            device = self.cos_basis.device
            th = theta.to(device=device, dtype=torch.float32)
            zv = z.to(device=device, dtype=torch.float32)

            f_core = self.f_base.unsqueeze(0).expand(B, self.rank, self.rank).clone()
            for k in range(1, self.num_harmonics + 1):
                k_idx = k - 1
                cos_val = torch.cos(k * th).view(B, 1, 1)
                sin_val = torch.sin(k * th).view(B, 1, 1)
                f_core = f_core + (cos_val * self.cos_basis[k_idx].unsqueeze(0) + sin_val * self.sin_basis[k_idx].unsqueeze(0))

            f_core = f_core + zv.view(B, 1, 1) * self.epoch_linear.unsqueeze(0)
            z_sq = ((zv / 10.0) ** 2).view(B, 1, 1)
            f_core = f_core + z_sq * self.epoch_quad.unsqueeze(0)
            return f_core

    class GeneralizedDualCycleMaTLoRALinear(nn.Module):
        def __init__(self, original_linear: nn.Linear, core_gen: DualCycleCoreGenerator, rank: int = 8, alpha: float = 16.0):
            super().__init__()
            self.in_features = original_linear.in_features
            self.out_features = original_linear.out_features
            self.rank = rank
            self.scaling = alpha / float(rank)
            self.base_layer = original_linear
            self.base_layer.weight.requires_grad = False
            if self.base_layer.bias is not None:
                self.base_layer.bias.requires_grad = False

            self.core_gen = core_gen
            device = original_linear.weight.device
            dtype = original_linear.weight.dtype

            self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=dtype))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, rank, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

            self.current_theta: Optional[torch.Tensor] = None
            self.current_z: Optional[torch.Tensor] = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base_out = self.base_layer(x)
            if self.current_theta is None or self.current_z is None:
                return base_out

            x_a = F.linear(x.float(), self.lora_A.float())
            F_core = self.core_gen(self.current_theta, self.current_z)
            x_af = torch.bmm(x_a, F_core.transpose(1, 2))
            lora_out = F.linear(x_af, self.lora_B.float()) * self.scaling
            return base_out + lora_out.to(dtype=x.dtype)

    # --- 4. Targeted Contrastive Loss (Intra + Inter) ---
    def compute_targeted_contrastive_loss(
        hidden_states: torch.Tensor,
        years: torch.Tensor,
        temperature: float = 0.07,
        margin: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = hidden_states.device
        B = hidden_states.size(0)
        if B <= 1:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero

        # Mean pool hidden states and normalize to unit sphere
        h_pool = F.normalize(hidden_states.mean(dim=1).float(), p=2, dim=-1)
        sim_matrix = torch.matmul(h_pool, h_pool.T)

        cycles = (years.to(device) - 3) % 60
        labels_eq = torch.eq(cycles.unsqueeze(0), cycles.unsqueeze(1))
        mask_self = torch.eye(B, dtype=torch.bool, device=device)

        pos_mask = labels_eq & (~mask_self)
        neg_mask = (~labels_eq) & (~mask_self)

        # L_intra (SupCon InfoNCE pulling same-cycle pairs together)
        if pos_mask.any():
            sim_scaled = sim_matrix / temperature
            sim_max = torch.max(sim_scaled, dim=-1, keepdim=True).values.detach()
            exp_sim = torch.exp(sim_scaled - sim_max)
            denom = (exp_sim * (~mask_self).float()).sum(dim=-1, keepdim=True) + 1e-8
            log_prob = (sim_scaled - sim_max) - torch.log(denom)
            pos_counts = pos_mask.sum(dim=-1)
            valid_rows = pos_counts > 0
            loss_intra = (-(log_prob * pos_mask.float()).sum(dim=-1) / (pos_counts.float() + 1e-8))[valid_rows].mean()
        else:
            loss_intra = torch.tensor(0.0, device=device, requires_grad=True)

        # L_inter (Margin contrastive penalty pushing different-cycle pairs apart)
        if neg_mask.any():
            inter_penalties = torch.clamp(sim_matrix - margin, min=0.0)
            loss_inter = inter_penalties[neg_mask].mean()
        else:
            loss_inter = torch.tensor(0.0, device=device, requires_grad=True)

        return loss_intra, loss_inter

    # --- 5. Model Loading & Architecture Setup ---
    model_id = "Qwen/Qwen2.5-3B"
    print(f"Loading Base model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    for p in model.parameters():
        p.requires_grad = False

    seam_projection = GatedResidualProjection(in_features=40, out_features=2048).to("cuda:0", dtype=torch.bfloat16)

    target_submodules = ["gate_proj", "up_proj", "down_proj"]
    mlp_modules = []
    core_gen = None

    if arm_type == "static_mlp":
        for name, module in model.named_modules():
            for target in target_submodules:
                if hasattr(module, target) and isinstance(getattr(module, target), nn.Linear):
                    orig = getattr(module, target)
                    adapted = StaticLoRALinear(orig, rank=rank, alpha=alpha)
                    setattr(module, target, adapted)
                    mlp_modules.append(adapted)
        print(f"Attached StaticLoRALinear to {len(mlp_modules)} MLP modules.")
    elif arm_type == "matlora":
        core_gen = DualCycleCoreGenerator(rank=rank, num_harmonics=2).to("cuda:0", dtype=torch.float32)
        for name, module in model.named_modules():
            for target in target_submodules:
                if hasattr(module, target) and isinstance(getattr(module, target), nn.Linear):
                    orig = getattr(module, target)
                    adapted = GeneralizedDualCycleMaTLoRALinear(orig, core_gen=core_gen, rank=rank, alpha=alpha)
                    setattr(module, target, adapted)
                    mlp_modules.append(adapted)
        print(f"Attached GeneralizedDualCycleMaTLoRALinear to {len(mlp_modules)} MLP modules.")
    else:
        raise ValueError(f"Unknown arm_type: {arm_type}")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    seen_ids = set()
    trainable_params = []
    all_modules = list(model.parameters()) + list(seam_projection.parameters())
    if core_gen is not None:
        all_modules += list(core_gen.parameters())

    for p in all_modules:
        if p.requires_grad and id(p) not in seen_ids:
            seen_ids.add(id(p))
            trainable_params.append(p)

    n_train = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {n_train:,} ({n_train / 3.09e9 * 100:.3f}% of backbone).")

    # --- 6. Dataset & DataLoader ---
    class Stage03TargetedDataset(Dataset):
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
                prefix_len = prefix_enc["input_ids"].shape[1]

                labels = input_ids.clone()
                labels[:prefix_len] = -100

                v40 = compute_fixed_fone_canchi_40d(year)
                theta, z = compute_theta_z(year)
                is_rel = int(task_id == "Rel" or ans.strip().lower() in ["yes", "no"])

                self.examples.append({
                    "input_ids": input_ids,
                    "labels": labels,
                    "v40": v40,
                    "theta": theta,
                    "z": z,
                    "year": year,
                    "is_rel": is_rel,
                })

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            return self.examples[idx]

    def collate_fn(batch, pad_token_id):
        max_l = max(len(x["input_ids"]) for x in batch)
        b_in, b_lbl, b_v40, b_th, b_z, b_yr, b_rel = [], [], [], [], [], [], []
        for x in batch:
            pad_len = max_l - len(x["input_ids"])
            inp = torch.cat([x["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            lbl = torch.cat([x["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            b_in.append(inp)
            b_lbl.append(lbl)
            b_v40.append(x["v40"])
            b_th.append(x["theta"])
            b_z.append(x["z"])
            b_yr.append(x["year"])
            b_rel.append(x["is_rel"])
        return {
            "input_ids": torch.stack(b_in),
            "labels": torch.stack(b_lbl),
            "v40": torch.tensor(b_v40, dtype=torch.float32),
            "theta": torch.tensor(b_th, dtype=torch.float32),
            "z": torch.tensor(b_z, dtype=torch.float32),
            "years": torch.tensor(b_yr, dtype=torch.long),
            "is_rel": torch.tensor(b_rel, dtype=torch.float32),
        }

    prefix = f"stage03_3b_targeted_{arm_type}"
    weights_path_final = f"/vol/{prefix}_5epochs_weights.pt"
    skip_training = False

    if not force_retrain and os.path.exists(weights_path_final):
        print(f"\n[CACHE HIT] Found existing 5-epoch pre-trained weights at {weights_path_final}!")
        try:
            ckpt = torch.load(weights_path_final, map_location="cuda:0")
            if "lora_state" in ckpt and "seam_projection" in ckpt:
                model.load_state_dict(ckpt["lora_state"], strict=False)
                seam_projection.load_state_dict(ckpt["seam_projection"], strict=True)
                if core_gen is not None and "core_generator" in ckpt:
                    core_gen.load_state_dict(ckpt["core_generator"], strict=True)
                print(f"[CACHE HIT] Successfully loaded 5-epoch weights for {arm_type}! Skipping retraining.")
                skip_training = True
                train_time = ckpt.get("metadata", {}).get("train_time_seconds", 11520.0)
        except Exception as e:
            print(f"[CACHE WARNING] Failed loading existing checkpoint: {e}. Retraining from scratch...")
            skip_training = False

    if not skip_training:
        train_ds = Stage03TargetedDataset(train_cases, tokenizer, max_l=max_len)
        train_loader = DataLoader(
            train_ds,
            batch_size=micro_batch,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        )

        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
        opt_steps_per_epoch = len(train_loader) // grad_accum
        total_opt_steps = opt_steps_per_epoch * epochs

        print(f"Beginning 5-epoch training: {len(train_loader)} micro-batches/epoch -> {opt_steps_per_epoch} opt steps/epoch.")
        print(f"Total optimizer steps over {epochs} epochs: {total_opt_steps} steps.")
        model.train()
        seam_projection.train()
        if core_gen is not None:
            core_gen.train()
        tokenizer.padding_side = "right"

        global_step = 0
        t_start = time.time()

        for epoch in range(epochs):
            print(f"\n>>> Starting Epoch {epoch + 1}/{epochs} <<<")
            accum_loss, accum_ntp, accum_intra, accum_inter = 0.0, 0.0, 0.0, 0.0
            t_epoch_start = time.time()

            for b_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(model.device)
                labels = batch["labels"].to(model.device)
                v40 = batch["v40"].to(model.device)
                years = batch["years"].to(model.device)
                theta = batch["theta"].to(model.device)
                z = batch["z"].to(model.device)

                if arm_type == "matlora":
                    for mod in mlp_modules:
                        mod.current_theta = theta
                        mod.current_z = z

                inputs_embeds = inject_seam_prefill_safe(model, seam_projection, input_ids, v40, token_idx=14)

                outputs = model(inputs_embeds=inputs_embeds, labels=labels, output_hidden_states=True)
                loss_ntp = outputs.loss
                last_hidden = outputs.hidden_states[-1]

                l_intra, l_inter = compute_targeted_contrastive_loss(
                    last_hidden, years, temperature=0.07, margin=margin_inter
                )
                total_batch_loss = loss_ntp + lambda_intra * l_intra + lambda_inter * l_inter

                loss_step = total_batch_loss / grad_accum
                loss_step.backward()

                accum_loss += total_batch_loss.item()
                accum_ntp += loss_ntp.item()
                accum_intra += l_intra.item()
                accum_inter += l_inter.item()

                if (b_idx + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % 200 == 0 or global_step == total_opt_steps:
                        elapsed = time.time() - t_start
                        print(
                            f"Epoch {epoch+1}/{epochs} | Step {global_step}/{total_opt_steps} | "
                            f"Total: {accum_loss / (200 * grad_accum):.4f} | "
                            f"NTP: {accum_ntp / (200 * grad_accum):.4f} | "
                            f"Intra: {accum_intra / (200 * grad_accum):.4f} | "
                            f"Inter: {accum_inter / (200 * grad_accum):.4f} | "
                            f"Alpha: {seam_projection.alpha.item():.5f}"
                        )
                        accum_loss, accum_ntp, accum_intra, accum_inter = 0.0, 0.0, 0.0, 0.0

            t_epoch_elapsed = time.time() - t_epoch_start
            print(f"Epoch {epoch + 1} completed in {t_epoch_elapsed / 60.0:.2f} minutes.")

            # Save intermediate epoch checkpoint to persistent volume
            ep_weights_path = f"/vol/{prefix}_5epochs_ep{epoch+1}_weights.pt"
            ep_ckpt = {
                "lora_state": {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k},
                "seam_projection": seam_projection.state_dict(),
                "core_generator": core_gen.state_dict() if core_gen is not None else None,
                "metadata": {
                    "arm_type": arm_type,
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "micro_batch": micro_batch,
                    "grad_accum": grad_accum,
                    "effective_batch_size": micro_batch * grad_accum,
                    "lr": lr,
                    "alpha": float(seam_projection.alpha.item()),
                    "global_step": global_step,
                },
            }
            torch.save(ep_ckpt, ep_weights_path)
            volume.commit()
            print(f"Committed intermediate checkpoint: {ep_weights_path}")

        train_time = time.time() - t_start
        print(f"\nAll {epochs} epochs complete in {train_time / 60.0:.2f} minutes ({train_time / 3600.0:.2f} hours).")

        # Save final trained weights
        ckpt = {
            "lora_state": {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k},
            "seam_projection": seam_projection.state_dict(),
            "core_generator": core_gen.state_dict() if core_gen is not None else None,
            "metadata": {
                "arm_type": arm_type,
                "epochs": epochs,
                "train_time_seconds": train_time,
                "micro_batch": micro_batch,
                "grad_accum": grad_accum,
                "effective_batch_size": micro_batch * grad_accum,
                "lr": lr,
                "alpha": float(seam_projection.alpha.item()),
                "timestamp": datetime.now().isoformat(),
            },
        }
        torch.save(ckpt, weights_path_final)
        volume.commit()
        print(f"Committed final 5-epoch checkpoint to Modal Volume: {weights_path_final}")

        del optimizer, train_loader, train_ds

    # Clean up training memory before evaluation
    gc.collect()
    torch.cuda.empty_cache()
    print(f"VRAM allocated before evaluation: {torch.cuda.memory_allocated() / 1e9:.2f} GB.")

    # --- 7. Evaluation: TempUN Clean 5,292 MCQA (4-Fold Debiased & Strict) ---
    print("\n========== EVALUATION 1: TEMPUN CLEAN 5,292 MCQA (4-FOLD ROTATIONS = 21,168 INFERENCES) ==========")
    model.eval()
    seam_projection.eval()
    if core_gen is not None:
        core_gen.eval()
    tokenizer.padding_side = "left"

    def _extract_ids(words):
        ids = []
        for w in words:
            for prefix in ["", " "]:
                enc = tokenizer.encode(prefix + w, add_special_tokens=False)
                if enc and len(enc) == 1:
                    ids.append(enc[0])
        return list(dict.fromkeys(ids))

    yes_ids = _extract_ids(["yes", "Yes", "YES"])
    no_ids = _extract_ids(["no", "No", "NO"])
    choice_ids = {k: _extract_ids([k, k.upper()]) for k in ["a", "b", "c", "d"]}

    def score_prediction_row(lg, gold_str: str) -> Tuple[str, str, int]:
        t_norm = gold_str.strip().lower()
        if t_norm in ["yes", "no"]:
            s_yes = max(lg[idx].item() for idx in yes_ids) if yes_ids else -1e9
            s_no = max(lg[idx].item() for idx in no_ids) if no_ids else -1e9
            pred = "yes" if s_yes >= s_no else "no"
        else:
            scores = {k: (max(lg[idx].item() for idx in c_ids) if c_ids else -1e9) for k, c_ids in choice_ids.items()}
            pred = max(scores, key=scores.get)
        return pred, t_norm, int(pred == t_norm)

    def generate_option_permutations(prompt_text: str, target_text: str):
        tgt = target_text.strip().lower()
        if tgt in ["yes", "no"]:
            p0 = prompt_text
            p1 = re.sub(r"Answer as Yes or No", "Answer as No or Yes", prompt_text, flags=re.IGNORECASE)
            if p1 == p0:
                p1 = re.sub(r"Yes\s*(or|/)\s*No", "No or Yes", prompt_text, flags=re.IGNORECASE)
            return [{"prompt": p0, "target": tgt}, {"prompt": p1, "target": tgt}]
        pattern = re.compile(
            r"^(.*Options:\s*)\(a\)\s*(.*?)\s*\(b\)\s*(.*?)\s*\(c\)\s*(.*?)\s*\(d\)\s*(.*?)$",
            re.DOTALL,
        )
        suffix = ""
        m_suffix = re.search(r"(\s*\n?\s*Answer:\s*)$", prompt_text, re.IGNORECASE)
        if m_suffix:
            suffix = m_suffix.group(1)
            base_prompt = prompt_text[: len(prompt_text) - len(suffix)]
        else:
            base_prompt = prompt_text
        m = pattern.search(base_prompt)
        if not m:
            return [{"prompt": prompt_text, "target": tgt}]
        prefix = m.group(1)
        opts = [m.group(2).strip(), m.group(3).strip(), m.group(4).strip(), m.group(5).strip()]
        letters = ["a", "b", "c", "d"]
        if tgt not in letters:
            return [{"prompt": prompt_text, "target": tgt}]
        t_idx = letters.index(tgt)
        perms = []
        for k in range(4):
            r_opts = [opts[(j + k) % 4] for j in range(4)]
            r_tgt = letters[(t_idx - k) % 4]
            perms.append({
                "prompt": f"{prefix}(a) {r_opts[0]} (b) {r_opts[1]} (c) {r_opts[2]} (d) {r_opts[3]}{suffix}",
                "target": r_tgt,
            })
        return perms

    case_results = {}
    can_correct, strict_correct, deb_sum = 0, 0, 0.0
    task_scores = defaultdict(lambda: {"canonical": 0, "debiased_sum": 0.0, "strict": 0, "total": 0})
    total_eval = len(eval_cases)
    eval_batch_size = 4
    t_eval_start = time.time()

    for b_start in range(0, total_eval, eval_batch_size):
        b_cases = eval_cases[b_start : b_start + eval_batch_size]
        all_prompts, all_targets, all_years, case_perm_slices = [], [], [], []

        for case in b_cases:
            yr = case.get("year", 2000)
            perms = generate_option_permutations(case["input"], case["target"])
            s_idx = len(all_prompts)
            for p in perms:
                all_prompts.append(p["prompt"])
                all_targets.append(p["target"])
                all_years.append(yr)
            e_idx = len(all_prompts)
            case_perm_slices.append((case, (s_idx, e_idx)))

        formatted = [f"Question: {p.strip()}\nAnswer:" if not p.strip().endswith("Answer:") else p.strip() for p in all_prompts]
        inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
        v40_eval = torch.tensor([compute_fixed_fone_canchi_40d(y) for y in all_years], dtype=torch.float32, device=model.device)

        if arm_type == "matlora":
            th_list, z_list = [], []
            for y in all_years:
                th_i, z_i = compute_theta_z(y)
                th_list.append(th_i)
                z_list.append(z_i)
            th_eval = torch.tensor(th_list, dtype=torch.float32, device=model.device)
            z_eval = torch.tensor(z_list, dtype=torch.float32, device=model.device)
            for mod in mlp_modules:
                mod.current_theta = th_eval
                mod.current_z = z_eval

        with torch.no_grad():
            inputs_embeds = inject_seam_prefill_safe(model, seam_projection, inputs["input_ids"], v40_eval, token_idx=14)
            transformer_out = model.model(inputs_embeds=inputs_embeds, attention_mask=inputs["attention_mask"])
            last_hidden_token = transformer_out.last_hidden_state[:, -1, :]
            logits = model.lm_head(last_hidden_token)

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
            task_scores[task_id]["strict"] += int(is_st)
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
            "strict_acc": round(v["strict"] / c * 100, 2),
            "count": c,
        }

    # --- 8. Evaluation: Counterfactual V1 (54 Historical Milestone Pairs) ---
    print("\n========== EVALUATION 2: COUNTERFACTUAL V1 (54 HISTORICAL PAIRS = 108 INFERENCES) ==========")
    both_correct_cf_v1 = 0
    for pair in cf_pairs:
        if len(pair) >= 2:
            q1, q2 = pair[0], pair[1]
            f1 = [f"Question: {q1['input'].strip()}\nAnswer:"]
            f2 = [f"Question: {q2['input'].strip()}\nAnswer:"]
            i1 = tokenizer(f1, return_tensors="pt").to(model.device)
            i2 = tokenizer(f2, return_tensors="pt").to(model.device)
            v1 = torch.tensor([compute_fixed_fone_canchi_40d(q1.get("year", 2000))], dtype=torch.float32, device=model.device)
            v2 = torch.tensor([compute_fixed_fone_canchi_40d(q2.get("year", 2000))], dtype=torch.float32, device=model.device)

            if arm_type == "matlora":
                th1, z1 = compute_theta_z(q1.get("year", 2000))
                th2, z2 = compute_theta_z(q2.get("year", 2000))

            with torch.no_grad():
                if arm_type == "matlora":
                    for mod in mlp_modules:
                        mod.current_theta = torch.tensor([th1], device=model.device)
                        mod.current_z = torch.tensor([z1], device=model.device)
                emb1 = inject_seam_prefill_safe(model, seam_projection, i1["input_ids"], v1, token_idx=14)
                h1 = model.model(inputs_embeds=emb1, attention_mask=i1["attention_mask"]).last_hidden_state[:, -1, :]
                o1 = model.lm_head(h1)
                r1 = score_prediction_row(o1[0], q1["target"])[2]

                if arm_type == "matlora":
                    for mod in mlp_modules:
                        mod.current_theta = torch.tensor([th2], device=model.device)
                        mod.current_z = torch.tensor([z2], device=model.device)
                emb2 = inject_seam_prefill_safe(model, seam_projection, i2["input_ids"], v2, token_idx=14)
                h2 = model.model(inputs_embeds=emb2, attention_mask=i2["attention_mask"]).last_hidden_state[:, -1, :]
                o2 = model.lm_head(h2)
                r2 = score_prediction_row(o2[0], q2["target"])[2]

            if r1 and r2:
                both_correct_cf_v1 += 1

    cf_v1_rate = round(both_correct_cf_v1 / len(cf_pairs) * 100, 2) if cf_pairs else 0.0
    print(f"Counterfactual V1 (54P): BothCorrect = {both_correct_cf_v1}/{len(cf_pairs)} ({cf_v1_rate}%)")

    # --- 9. Evaluation: Counterfactual V2 (852 Rel Inverted Pairs = 1,704 Inferences) ---
    print("\n========== EVALUATION 3: COUNTERFACTUAL V2 (FULL 852 REL INVERSIONS = 1,704 INFERENCES) ==========")
    rel_cases = [c for c in eval_cases if c.get("task_id") == "Rel"]
    pattern_rel = re.compile(r"In\s+(\d{4}),\s+(.*?)\s+(?:was|is|were)\s+higher than\s+(\d{4})\?\s+Answer as Yes or No", re.IGNORECASE)

    both_correct_cf_v2 = 0
    total_cf_v2 = 0

    for case in rel_cases:
        inp = case.get("input", "").strip()
        m = pattern_rel.search(inp)
        if not m:
            continue
        total_cf_v2 += 1
        y1, content, y2 = int(m.group(1)), m.group(2), int(m.group(3))
        gold_orig = case.get("target", "").strip()
        gold_flipped = "No" if gold_orig.lower() == "yes" else "Yes"

        verb_match = re.search(r"\b(was|is|were)\b\s+higher than", inp, re.IGNORECASE)
        verb = verb_match.group(1) if verb_match else "was"
        prompt_flipped = f"In {y2}, {content} {verb} higher than {y1}? Answer as Yes or No"

        f_orig = [f"Question: {inp}\nAnswer:"]
        f_flip = [f"Question: {prompt_flipped}\nAnswer:"]

        i_orig = tokenizer(f_orig, return_tensors="pt").to(model.device)
        i_flip = tokenizer(f_flip, return_tensors="pt").to(model.device)

        v_orig = torch.tensor([compute_fixed_fone_canchi_40d(y1)], dtype=torch.float32, device=model.device)
        v_flip = torch.tensor([compute_fixed_fone_canchi_40d(y2)], dtype=torch.float32, device=model.device)

        with torch.no_grad():
            if arm_type == "matlora":
                th1, z1 = compute_theta_z(y1)
                for mod in mlp_modules:
                    mod.current_theta = torch.tensor([th1], device=model.device)
                    mod.current_z = torch.tensor([z1], device=model.device)
            emb_orig = inject_seam_prefill_safe(model, seam_projection, i_orig["input_ids"], v_orig, token_idx=14)
            h_orig = model.model(inputs_embeds=emb_orig, attention_mask=i_orig["attention_mask"]).last_hidden_state[:, -1, :]
            o_orig = model.lm_head(h_orig)
            r_orig = score_prediction_row(o_orig[0], gold_orig)[2]

            if arm_type == "matlora":
                th2, z2 = compute_theta_z(y2)
                for mod in mlp_modules:
                    mod.current_theta = torch.tensor([th2], device=model.device)
                    mod.current_z = torch.tensor([z2], device=model.device)
            emb_flip = inject_seam_prefill_safe(model, seam_projection, i_flip["input_ids"], v_flip, token_idx=14)
            h_flip = model.model(inputs_embeds=emb_flip, attention_mask=i_flip["attention_mask"]).last_hidden_state[:, -1, :]
            o_flip = model.lm_head(h_flip)
            r_flip = score_prediction_row(o_flip[0], gold_flipped)[2]

        if r_orig and r_flip:
            both_correct_cf_v2 += 1

    cf_v2_rate = round(both_correct_cf_v2 / total_cf_v2 * 100, 2) if total_cf_v2 > 0 else 0.0
    print(f"Counterfactual V2 (852 Rel Inversions): BothCorrect = {both_correct_cf_v2}/{total_cf_v2} ({cf_v2_rate}%)")

    # --- 10. Compile Results & Save Artifacts ---
    results_dict = {
        "canonical_acc": round(can_correct / total_eval * 100.0, 2),
        "debiased_mean_acc": round(deb_sum / total_eval * 100.0, 2),
        "strict_acc": round(strict_correct / total_eval * 100.0, 2),
        "total": total_eval,
        "task_summary": task_summary,
        "cf_54p_v1": {
            "both_correct": both_correct_cf_v1,
            "total": len(cf_pairs),
            "rate": cf_v1_rate,
        },
        "cf_852p_v2": {
            "both_correct": both_correct_cf_v2,
            "total": total_cf_v2,
            "rate": cf_v2_rate,
        },
        "hyperparameters": {
            "epochs": epochs,
            "micro_batch": micro_batch,
            "grad_accum": grad_accum,
            "effective_batch_size": micro_batch * grad_accum,
            "lr": lr,
            "lambda_intra": lambda_intra,
            "lambda_inter": lambda_inter,
            "margin_inter": margin_inter,
            "rank": rank,
            "alpha": alpha,
            "final_alpha": float(seam_projection.alpha.item()),
            "train_time_minutes": round(train_time / 60.0, 2),
        },
        "case_results": case_results,
    }

    summary_dict = {
        "arm_name": f"targeted_{arm_type}_5epochs",
        "epochs": epochs,
        "effective_batch_size": micro_batch * grad_accum,
        "canonical_acc": results_dict["canonical_acc"],
        "debiased_mean_acc": results_dict["debiased_mean_acc"],
        "strict_acc": results_dict["strict_acc"],
        "cf_v1_both_correct": both_correct_cf_v1,
        "cf_v1_rate": cf_v1_rate,
        "cf_v2_both_correct": both_correct_cf_v2,
        "cf_v2_rate": cf_v2_rate,
        "train_time_minutes": round(train_time / 60.0, 2),
        "final_alpha": float(seam_projection.alpha.item()),
    }

    # Save to Persistent Volume
    prefix = f"stage03_3b_targeted_{arm_type}"
    results_path = f"/vol/{prefix}_5epochs_results.json"
    summary_path = f"/vol/{prefix}_5epochs_summary.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    volume.commit()
    print(f"Committed 5-epoch results and summary to Modal Volume /vol/ ({prefix}_5epochs).")

    buf = io.BytesIO()
    torch.save(ckpt, buf)
    weights_bytes = buf.getvalue()

    return summary_dict, results_dict, weights_bytes


# =========================================================================
# LOCAL ENTRYPOINT
# =========================================================================

@app.local_entrypoint()
def main(arm: str = "static_mlp", force_retrain: bool = False):
    print(f"Executing Modal Local Entrypoint for 5-Epoch Run: arm={arm}, force_retrain={force_retrain}...")

    train_path = _REPO_ROOT / "data" / "tempun_full_clean" / "train_104k_clean.jsonl.gz"
    eval_path = _REPO_ROOT / "data" / "tempun_full_clean" / "eval_5292_clean.jsonl.gz"
    cf_path = _REPO_ROOT / "data" / "counterfactual" / "grounded_counterfactual_benchmark.jsonl"

    if not train_path.exists() or not eval_path.exists() or not cf_path.exists():
        print(f"Error: Missing local datasets! train={train_path.exists()}, eval={eval_path.exists()}, cf={cf_path.exists()}")
        sys.exit(1)

    print(f"Reading training bundle ({train_path.stat().st_size:,} bytes)...")
    train_bytes = train_path.read_bytes()
    eval_bytes = eval_path.read_bytes()
    cf_bytes = cf_path.read_bytes()

    if arm == "all":
        print("\n>>> Executing SEQUENTIAL Fail-Safe Modal Jobs for 5-Epoch Arms [static_mlp] and [matlora] <<<")
        print("Sequential execution guarantees 0 GPU concurrency errors and 0 volume commit race conditions.")

        print("\n=======================================================")
        print("STAGE 1/2: Running Arm 4 (Targeted Static MLP LoRA, 5 epochs)...")
        print("=======================================================")
        summary_mlp, res_mlp, w_mlp = train_and_evaluate_targeted_5epochs.remote(
            arm_type="static_mlp",
            train_bytes=train_bytes,
            eval_bytes=eval_bytes,
            cf_bytes=cf_bytes,
            epochs=5,
            lr=2e-4,
            micro_batch=32,
            grad_accum=4,
            rank=8,
            alpha=16.0,
            margin_inter=0.1,
            lambda_intra=0.5,
            lambda_inter=0.5,
            force_retrain=force_retrain,
        )
        print("Arm 4 Summary:", json.dumps(summary_mlp, indent=2))

        # Save Arm 4 receipts immediately
        out_dir = _REPO_ROOT / "artifacts" / "stage03_modal_5epochs"
        report_dir = _REPO_ROOT / "Reports" / "6-11Report"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "arm4_static_mlp_5epochs_summary.json").write_text(json.dumps(summary_mlp, indent=2), encoding="utf-8")
        (out_dir / "arm4_static_mlp_5epochs_results.json").write_text(json.dumps(res_mlp, indent=2), encoding="utf-8")
        (report_dir / "ARM_4_TARGETED_STATIC_MLP_5EPOCHS_SUMMARY.json").write_text(json.dumps(summary_mlp, indent=2), encoding="utf-8")
        (report_dir / "ARM_4_TARGETED_STATIC_MLP_5EPOCHS_RESULTS.json").write_text(json.dumps(res_mlp, indent=2), encoding="utf-8")
        print(f"[STAGE 1/2 COMPLETE] Saved Arm 4 receipts to {out_dir} and {report_dir}.")

        print("\n=======================================================")
        print("STAGE 2/2: Running Arm 5 (Targeted MaT-LoRA, 5 epochs)...")
        print("=======================================================")
        summary_mat, res_mat, w_mat = train_and_evaluate_targeted_5epochs.remote(
            arm_type="matlora",
            train_bytes=train_bytes,
            eval_bytes=eval_bytes,
            cf_bytes=cf_bytes,
            epochs=5,
            lr=2e-4,
            micro_batch=32,
            grad_accum=4,
            rank=8,
            alpha=16.0,
            margin_inter=0.1,
            lambda_intra=0.5,
            lambda_inter=0.5,
            force_retrain=force_retrain,
        )
        print("Arm 5 Summary:", json.dumps(summary_mat, indent=2))

        (out_dir / "arm5_matlora_5epochs_summary.json").write_text(json.dumps(summary_mat, indent=2), encoding="utf-8")
        (out_dir / "arm5_matlora_5epochs_results.json").write_text(json.dumps(res_mat, indent=2), encoding="utf-8")
        (report_dir / "ARM_5_TARGETED_MATLORA_5EPOCHS_SUMMARY.json").write_text(json.dumps(summary_mat, indent=2), encoding="utf-8")
        (report_dir / "ARM_5_TARGETED_MATLORA_5EPOCHS_RESULTS.json").write_text(json.dumps(res_mat, indent=2), encoding="utf-8")
        print(f"[STAGE 2/2 COMPLETE] Saved Arm 5 receipts to {out_dir} and {report_dir}.")

        # Update Evidence Index
        import hashlib
        def sha256_file(p: Path) -> str:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()

        idx_data = {
            "folder": "Reports/6-11Report",
            "updated_at": datetime.now().isoformat(),
            "status": "5EPOCHS_COMPLETED",
            "files": {
                item.name: {"size_bytes": item.stat().st_size, "sha256": sha256_file(item)}
                for item in sorted(report_dir.glob("*")) if item.is_file()
            }
        }
        (report_dir / "EVIDENCE-INDEX.json").write_text(json.dumps(idx_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Updated {report_dir / 'EVIDENCE-INDEX.json'}.")

        # Sync to Google Drive if present
        gdrive_dir = Path("G:/My Drive/NCKH/Receipts/3B_STAGE03_V2_SCIENTIFIC_CLOSURE_20261026/6-11Report")
        try:
            if gdrive_dir.parent.exists():
                gdrive_dir.mkdir(parents=True, exist_ok=True)
                for f in report_dir.glob("*.json"):
                    shutil.copy2(f, gdrive_dir / f.name)
                for f in report_dir.glob("*.md"):
                    shutil.copy2(f, gdrive_dir / f.name)
                print(f"Synchronized 5-epoch receipts to Google Drive: {gdrive_dir}")
        except Exception as e:
            print(f"Drive sync note: {e}")

        print("\nALL 5-EPOCH TARGETED CONTINUAL PRETRAINING ARMS COMPLETED SUCCESSFULLY.")

    else:
        print(f"\n>>> Running Modal Job for 5-Epoch Arm [{arm}] <<<")
        summary, results, w_bytes = train_and_evaluate_targeted_5epochs.remote(
            arm_type=arm,
            train_bytes=train_bytes,
            eval_bytes=eval_bytes,
            cf_bytes=cf_bytes,
            epochs=5,
            lr=2e-4,
            micro_batch=32,
            grad_accum=4,
            rank=8,
            alpha=16.0,
            margin_inter=0.1,
            lambda_intra=0.5,
            lambda_inter=0.5,
            force_retrain=force_retrain,
        )
        print("Completed Arm Summary:", json.dumps(summary, indent=2))

        out_dir = _REPO_ROOT / "artifacts" / "stage03_modal_5epochs"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"arm_{arm}_5epochs_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out_dir / f"arm_{arm}_5epochs_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved local 5-epoch receipts to {out_dir}")
