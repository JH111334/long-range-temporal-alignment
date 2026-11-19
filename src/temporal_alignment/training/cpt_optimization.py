"""Continual Pre-Training (CPT) optimization utilities: AdamW for MLP LoRA, Warmup+Cosine/WSD LR, and Replay."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_cpt_optimizer_param_groups(
    model: nn.Module,
    seam_proj: Optional[nn.Module] = None,
    core_gen: Optional[nn.Module] = None,
    *,
    base_lr: float = 1.0e-4,
    loraplus_ratio: float = 1.0,
    adapter_head_lr: float = 2.0e-4,
    weight_decay: float = 0.01,
) -> List[Dict[str, Any]]:
    """Build AdamW parameter groups for MLP-only LoRA and temporal representation modules.

    Groups:
    - lora_A: lr = base_lr, weight_decay = weight_decay
    - lora_B: lr = base_lr * loraplus_ratio (1.0 for standard AdamW; >1.0 for LoRA+)
    - seam_proj / core_gen: lr = adapter_head_lr (weight_decay=0.0 on 1D norms/biases/alpha)
    """
    if base_lr <= 0.0:
        raise ValueError("base_lr must be positive")
    if loraplus_ratio < 1.0:
        raise ValueError("loraplus_ratio must be >= 1.0")
    if adapter_head_lr <= 0.0:
        raise ValueError("adapter_head_lr must be positive")

    lora_a_params: List[nn.Parameter] = []
    lora_b_params: List[nn.Parameter] = []
    other_decay_params: List[nn.Parameter] = []
    other_nodecay_params: List[nn.Parameter] = []

    seen_ids: Set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen_ids:
            continue
        seen_ids.add(pid)

        if "lora_A" in name:
            lora_a_params.append(param)
        elif "lora_B" in name:
            lora_b_params.append(param)
        elif param.ndim <= 1 or name.endswith(".bias"):
            other_nodecay_params.append(param)
        else:
            other_decay_params.append(param)

    head_decay_params: List[nn.Parameter] = []
    head_nodecay_params: List[nn.Parameter] = []

    for module in (seam_proj, core_gen):
        if module is None:
            continue
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            if param.ndim <= 1 or name.endswith(".bias") or "norm" in name or "alpha" in name:
                head_nodecay_params.append(param)
            else:
                head_decay_params.append(param)

    groups: List[Dict[str, Any]] = []
    if lora_a_params:
        groups.append({"name": "lora_A", "params": lora_a_params, "lr": base_lr, "weight_decay": weight_decay})
    if lora_b_params:
        groups.append(
            {
                "name": "lora_B",
                "params": lora_b_params,
                "lr": base_lr * loraplus_ratio,
                "weight_decay": weight_decay,
            }
        )
    if head_decay_params:
        groups.append(
            {
                "name": "temporal_head_decay",
                "params": head_decay_params,
                "lr": adapter_head_lr,
                "weight_decay": weight_decay,
            }
        )
    if head_nodecay_params:
        groups.append(
            {
                "name": "temporal_head_nodecay",
                "params": head_nodecay_params,
                "lr": adapter_head_lr,
                "weight_decay": 0.0,
            }
        )
    if other_decay_params:
        groups.append({"name": "other_decay", "params": other_decay_params, "lr": base_lr, "weight_decay": weight_decay})
    if other_nodecay_params:
        groups.append({"name": "other_nodecay", "params": other_nodecay_params, "lr": base_lr, "weight_decay": 0.0})

    if not groups:
        raise ValueError("No trainable parameters found when building CPT optimizer groups.")
    return groups


def create_cpt_lr_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    *,
    warmup_ratio: float = 0.04,
    schedule_type: str = "cosine",
    min_lr_ratio: float = 0.1,
    stable_ratio: float = 0.7,
) -> LambdaLR:
    """Create a CPT learning rate scheduler supporting Warmup+Cosine and WSD.

    - 'cosine': linear warmup over warmup_ratio, then cosine decay to min_lr_ratio.
    - 'wsd': linear warmup, stable plateau until stable_ratio, then cosine decay to min_lr_ratio.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0.0, 1.0)")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0.0, 1.0]")
    if schedule_type not in ("cosine", "wsd"):
        raise ValueError(f"Unsupported schedule_type: {schedule_type}")

    warmup_steps = int(round(total_steps * warmup_ratio))
    if warmup_ratio > 0.0 and warmup_steps == 0:
        warmup_steps = 1

    stable_end = int(round(total_steps * stable_ratio))
    stable_end = max(warmup_steps, min(total_steps, stable_end))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps and warmup_steps > 0:
            return float(current_step + 1) / float(warmup_steps)

        if schedule_type == "cosine":
            decay_steps = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, float(current_step - warmup_steps) / float(decay_steps)))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        # WSD: Warmup -> Stable -> Decay
        if current_step <= stable_end:
            return 1.0
        decay_steps = max(1, total_steps - stable_end)
        progress = min(1.0, max(0.0, float(current_step - stable_end) / float(decay_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def _record_prompt_hash(record: Mapping[str, Any]) -> str:
    text = str(record.get("input", "") or record.get("prompt", "") or record.get("text", "")).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def interleave_general_replay(
    nwp_records: Sequence[Mapping[str, Any]],
    replay_records: Sequence[Mapping[str, Any]],
    *,
    replay_ratio: float = 0.15,
    seed: int = 42,
    forbidden_eval_ids: Optional[Iterable[str]] = None,
    forbidden_prompt_hashes: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Deterministically mix 10%-20% general replay records into NWP training stream.

    Fail-closed against leakage: raises ValueError if any replay record overlaps
    with forbidden_eval_ids or forbidden_prompt_hashes.
    """
    if not 0.0 <= replay_ratio <= 0.5:
        raise ValueError(f"replay_ratio must be in [0.0, 0.5], got {replay_ratio}")
    if replay_ratio == 0.0 or not replay_records:
        return [dict(r, _source_stream="nwp") for r in nwp_records]

    forbidden_ids = set(forbidden_eval_ids or ())
    forbidden_hashes = set(forbidden_prompt_hashes or ())

    clean_replay: List[Dict[str, Any]] = []
    for rec in replay_records:
        rec_id = str(rec.get("id", rec.get("example_id", "")))
        if rec_id and rec_id in forbidden_ids:
            raise ValueError(f"Leakage detected: replay record id '{rec_id}' overlaps forbidden eval set.")
        p_hash = _record_prompt_hash(rec)
        if p_hash in forbidden_hashes:
            raise ValueError(f"Leakage detected: replay record prompt hash '{p_hash[:12]}' overlaps forbidden eval set.")
        item = dict(rec)
        item["_source_stream"] = "general_replay"
        clean_replay.append(item)

    n_nwp = len(nwp_records)
    target_replay_count = int(round(n_nwp * replay_ratio / max(1e-8, 1.0 - replay_ratio)))
    rng = random.Random(seed)

    sampled_replay: List[Dict[str, Any]] = []
    while len(sampled_replay) < target_replay_count:
        pool = list(clean_replay)
        rng.shuffle(pool)
        needed = target_replay_count - len(sampled_replay)
        sampled_replay.extend(pool[:needed])

    combined = [dict(r, _source_stream="nwp") for r in nwp_records] + sampled_replay
    rng.shuffle(combined)
    return combined
