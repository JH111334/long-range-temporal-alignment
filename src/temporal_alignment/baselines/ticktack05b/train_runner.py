"""Execution and Training Runner for TickTack-inspired 0.5B Baseline.

Supports:
- Sequential 3-stage training: year:1947 -> year:1984 -> year:2022.
- Multi-task objective: NTP Loss + Temporal Contrastive Loss (Eq. 8, sigma=0.5, delta=1.0) + LoRA EWC (Eq. 9).
- Checkpoint extraction and Fisher accumulation across stages.
- Immediate stage-wise matched P12 evaluation on 180 cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .dataset import TempUNTrainDataset, collate_ticktack_batch
from .evaluator import evaluate_single_example
from .loss import LoRAEWCLoss, compute_ntp_loss, compute_temporal_contrastive_loss
from .model import TickTackQwen05BModel

STAGES = ("year:1947", "year:1984", "year:2022")


def seed_everything(seed: int) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_stage_fisher_matrix(
    model: TickTackQwen05BModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> Dict[str, torch.Tensor]:
    """Estimate diagonal Fisher Information Matrix on trainable LoRA parameters."""
    model.train()
    fisher: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data, device=device)

    total_samples = 0
    for b_idx, batch in enumerate(dataloader):
        if b_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        years = batch["years"]

        model.zero_grad()
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            years=years,
        )
        loss = compute_ntp_loss(out["logits"], labels)
        loss.backward()

        batch_sz = input_ids.shape[0]
        total_samples += batch_sz

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += (param.grad.data ** 2) * batch_sz

    if total_samples > 0:
        for name in fisher:
            fisher[name] /= float(total_samples)

    model.zero_grad()
    return fisher


def run_stage_training(
    model: TickTackQwen05BModel,
    tokenizer: Any,
    dataset: TempUNTrainDataset,
    stage_id: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    ewc_module: LoRAEWCLoss,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    num_epochs: int = 1,
    max_steps: Optional[int] = None,
    sigma_temporal: float = 0.5,
    delta_temporal: float = 1.0,
    lambda_ewc: float = 100.0,
) -> Dict[str, Any]:
    """Train model on a single temporal stage."""
    model.train()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_ticktack_batch,
    )

    step_count = 0
    total_loss_sum = 0.0
    ntp_loss_sum = 0.0
    temp_loss_sum = 0.0
    ewc_loss_sum = 0.0
    optimizer.zero_grad()

    epoch_idx = 0
    while True:
        epoch_idx += 1
        for batch in dataloader:
            step_count += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            years = batch["years"]
            year_tensor = batch["year_tensor"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                years=years,
            )

            # 1. Next-Token Prediction loss (FP32)
            loss_ntp = compute_ntp_loss(out["logits"], labels)

            # 2. Temporal contrastive loss (Eq. 8)
            loss_temp, pos_pairs = compute_temporal_contrastive_loss(
                hidden_states=out["pooled_hidden_state"],
                year_labels=year_tensor,
                delta=delta_temporal,
            )

            # 3. EWC penalty on trainable LoRA (Eq. 9)
            loss_ewc, _ = ewc_module(model)

            # Combined loss (Eq. 10)
            total_loss = loss_ntp + sigma_temporal * loss_temp + loss_ewc
            loss_for_backward = total_loss / grad_accum_steps

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                raise RuntimeError(f"Numerical failure: loss is {total_loss.item()} at step {step_count}")

            loss_for_backward.backward()

            total_loss_sum += total_loss.item()
            ntp_loss_sum += loss_ntp.item()
            temp_loss_sum += loss_temp.item()
            ewc_loss_sum += loss_ewc.item()

            if step_count % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad()

            if max_steps is not None and step_count >= max_steps:
                break

        if max_steps is not None and step_count >= max_steps:
            break
        if epoch_idx >= num_epochs:
            break

    avg_total = total_loss_sum / max(1, step_count)
    avg_ntp = ntp_loss_sum / max(1, step_count)
    avg_temp = temp_loss_sum / max(1, step_count)
    avg_ewc = ewc_loss_sum / max(1, step_count)

    return {
        "stage_id": stage_id,
        "steps_completed": step_count,
        "avg_total_loss": avg_total,
        "avg_ntp_loss": avg_ntp,
        "avg_temporal_loss": avg_temp,
        "avg_ewc_loss": avg_ewc,
    }


def evaluate_stage_on_180(
    model: TickTackQwen05BModel,
    tokenizer: Any,
    eval_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute matched 180-case evaluation under P12 free-vocabulary generation."""
    model.eval()
    results = []
    correct_total = 0
    correct_mcqa = 0
    correct_yn = 0
    invalid_total = 0

    mcqa_count = 0
    yn_count = 0

    for rec in eval_records:
        res = evaluate_single_example(model, tokenizer, rec)
        results.append(res)

        if res["is_yn"]:
            yn_count += 1
            if res["is_correct"]:
                correct_yn += 1
        else:
            mcqa_count += 1
            if res["is_correct"]:
                correct_mcqa += 1

        if not res["is_valid"]:
            invalid_total += 1
        if res["is_correct"]:
            correct_total += 1

    n_total = len(eval_records)
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
