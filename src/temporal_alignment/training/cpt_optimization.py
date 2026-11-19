"""Continual Pre-Training (CPT) optimization utilities: AdamW for MLP LoRA, Warmup+Cosine/WSD LR, and Replay."""

from __future__ import annotations

import hashlib
import math
import random
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """Create a CPT learning rate scheduler supporting Constant, Warmup+Cosine, and WSD.

    - 'constant': constant multiplier 1.0 across all steps.
    - 'cosine': linear warmup over warmup_ratio, then cosine decay to min_lr_ratio.
    - 'wsd': linear warmup, stable plateau until stable_ratio, then cosine decay to min_lr_ratio.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0.0, 1.0)")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0.0, 1.0]")
    if schedule_type not in ("constant", "cosine", "wsd"):
        raise ValueError(f"Unsupported schedule_type: {schedule_type}")

    warmup_steps = int(round(total_steps * warmup_ratio))
    if warmup_ratio > 0.0 and warmup_steps == 0:
        warmup_steps = 1

    stable_end = int(round(total_steps * stable_ratio))
    stable_end = max(warmup_steps, min(total_steps, stable_end))

    def lr_lambda(current_step: int) -> float:
        if schedule_type == "constant":
            return 1.0
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


_REPLAY_OPTION_RE = re.compile(r"(?:\boptions\s*:|\([a-d]\))", re.IGNORECASE)


def _record_prompt_hash(record: Mapping[str, Any]) -> str:
    text = str(record.get("input", "") or record.get("prompt", "") or record.get("text", "")).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalized_text_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def interleave_general_replay(
    nwp_records: Sequence[Mapping[str, Any]],
    replay_records: Sequence[Mapping[str, Any]],
    *,
    replay_ratio: float = 0.15,
    seed: int = 42,
    epoch: int = 0,
    allow_oversample: bool = False,
    forbidden_eval_ids: Optional[Iterable[str]] = None,
    forbidden_prompt_hashes: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Deterministically mix general replay records into the NWP training stream.

    Fail-closed against empty replay pool (when replay_ratio > 0), insufficient
    unique pool size (unless allow_oversample=True), MCQA/target-leakage contamination,
    and fake year metadata on replay records.
    """
    if not 0.0 <= replay_ratio <= 0.5:
        raise ValueError(f"replay_ratio must be in [0.0, 0.5], got {replay_ratio}")
    if replay_ratio == 0.0:
        out_nwp = [dict(r, _source_stream="nwp") for r in nwp_records]
        random.Random(seed + epoch * 1009).shuffle(out_nwp)
        return out_nwp
    if not replay_records:
        raise ValueError(f"Fail-closed: replay_ratio={replay_ratio} > 0 requires a non-empty replay corpus.")

    forbidden_ids = set(forbidden_eval_ids or ())
    forbidden_hashes = set(forbidden_prompt_hashes or ())

    clean_replay: List[Dict[str, Any]] = []
    for rec in replay_records:
        if rec.get("year") is not None:
            raise ValueError(f"Replay record contains unexpected year metadata: {rec.get('year')!r}")
        prompt_text = str(rec.get("input", "") or rec.get("prompt", "") or rec.get("text", ""))
        if _REPLAY_OPTION_RE.search(prompt_text):
            raise ValueError("Replay record contains forbidden MCQA option formatting.")
        rec_id = str(rec.get("id", rec.get("example_id", rec.get("source_id", ""))))
        if rec_id and rec_id in forbidden_ids:
            raise ValueError(f"Leakage detected: replay record id '{rec_id}' overlaps forbidden eval set.")
        p_hash = _record_prompt_hash(rec)
        p_norm_hash = _normalized_text_hash(prompt_text)
        if p_hash in forbidden_hashes or p_norm_hash in forbidden_hashes:
            raise ValueError(f"Leakage detected: replay record prompt hash '{p_hash[:12]}' overlaps forbidden eval set.")
        item = dict(rec)
        item["_source_stream"] = "general_replay"
        clean_replay.append(item)

    n_nwp = len(nwp_records)
    target_replay_count = int(round(n_nwp * replay_ratio / max(1e-8, 1.0 - replay_ratio)))
    if not allow_oversample and len(clean_replay) < target_replay_count:
        raise ValueError(
            f"Insufficient clean general replay records ({len(clean_replay)} < required {target_replay_count} "
            f"at replay_ratio={replay_ratio}) with allow_oversample=False."
        )

    rng = random.Random(seed + epoch * 1009)
    sampled_replay: List[Dict[str, Any]] = []
    while len(sampled_replay) < target_replay_count:
        pool = list(clean_replay)
        rng.shuffle(pool)
        needed = target_replay_count - len(sampled_replay)
        sampled_replay.extend(pool[:needed])

    combined = [dict(r, _source_stream="nwp") for r in nwp_records] + sampled_replay
    rng.shuffle(combined)
    return combined


def tokenize_supervised_pair(
    tokenizer: Any,
    prompt_text: str,
    target_text: str,
    *,
    max_length: int = 128,
) -> Dict[str, Any]:
    """Tokenize rendered prompt + target once with exact char-offset boundaries and no target truncation."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Fast tokenizer with offset_mapping capability is required for boundary-safe supervision.")

    clean_prompt = prompt_text.rstrip()
    clean_target = target_text.strip()
    if not clean_prompt or not clean_target:
        raise ValueError("Both prompt_text and target_text must be non-empty.")

    full_text = f"{clean_prompt} {clean_target}"
    prompt_char_len = len(clean_prompt)

    enc = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids_list: List[int] = list(enc["input_ids"])
    offsets: List[Tuple[int, int]] = [(int(st), int(en)) for st, en in enc["offset_mapping"]]
    if not offsets:
        raise ValueError("Tokenizer returned empty tokens.")

    max_covered_char = max((en for st, en in offsets if en > st), default=0)
    if max_covered_char < len(full_text):
        raise ValueError(
            f"Target truncation detected at max_length={max_length}: covered {max_covered_char}/{len(full_text)} chars."
        )

    prompt_mask_list: List[bool] = []
    labels_list: List[int] = []
    for tok_id, (st, en) in zip(input_ids_list, offsets):
        is_valid_tok = en > st
        is_prompt_tok = bool(is_valid_tok and en <= prompt_char_len)
        is_target_tok = bool(is_valid_tok and st >= prompt_char_len)
        prompt_mask_list.append(is_prompt_tok)
        labels_list.append(int(tok_id) if is_target_tok else -100)

    if sum(1 for lbl in labels_list if lbl != -100) == 0:
        raise ValueError("Zero supervised target tokens after exact Answer: boundary masking.")

    return {
        "full_text": full_text,
        "prompt_char_len": prompt_char_len,
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "prompt_mask": torch.tensor(prompt_mask_list, dtype=torch.bool),
        "offsets": offsets,
    }


def compute_masked_temporal_contrastive_loss(
    hidden_states: torch.Tensor,
    years: torch.Tensor,
    temporal_active: torch.Tensor,
    prompt_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    non_special_mask: Optional[torch.Tensor] = None,
    temperature: float = 0.07,
    margin: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Compute SupCon same-cycle + margin loss strictly over masked prompt tokens of temporal rows."""
    dev = hidden_states.device
    active_bool = temporal_active.to(device=dev, dtype=torch.bool)
    mask_valid = prompt_mask.to(device=dev, dtype=torch.bool) & attention_mask.to(device=dev, dtype=torch.bool)
    if non_special_mask is not None:
        mask_valid = mask_valid & non_special_mask.to(device=dev, dtype=torch.bool)

    row_has_prompt = active_bool & (mask_valid.sum(dim=1) > 0)
    n_valid_rows = int(row_has_prompt.sum().item())
    stats = {"valid_temporal_rows": n_valid_rows, "positive_pairs": 0, "negative_pairs": 0}

    zero_graph = hidden_states.sum() * 0.0
    if n_valid_rows <= 1:
        return zero_graph, zero_graph, stats

    h_sel = hidden_states[row_has_prompt].float()
    m_sel = mask_valid[row_has_prompt].to(dtype=torch.float32).unsqueeze(-1)
    y_sel = years.to(device=dev)[row_has_prompt]

    pooled = (h_sel * m_sel).sum(dim=1) / m_sel.sum(dim=1).clamp_min(1.0)
    u = F.normalize(pooled, p=2, dim=-1)
    sim_matrix = torch.matmul(u, u.T)

    cycles = (y_sel - 3) % 60
    labels_eq = torch.eq(cycles.unsqueeze(0), cycles.unsqueeze(1))
    mask_self = torch.eye(n_valid_rows, dtype=torch.bool, device=dev)
    pos_mask = labels_eq & (~mask_self)
    neg_mask = (~labels_eq) & (~mask_self)

    stats["positive_pairs"] = int(pos_mask.sum().item())
    stats["negative_pairs"] = int(neg_mask.sum().item())

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
        loss_intra = zero_graph

    if neg_mask.any():
        loss_inter = torch.clamp(sim_matrix - margin, min=0.0)[neg_mask].mean()
    else:
        loss_inter = zero_graph

    return loss_intra, loss_inter, stats
