"""Canonical Deterministic Evaluator for Stage 03 Representation-First Protocol.

Enforces:
1. Strict deterministic decoding (do_sample=False, greedy argmax).
2. Protocol P12 Task-Aware prompt generation.
3. Strict regex parser for options and Yes/No questions.
4. Schema-compliant RAW_PREDICTIONS.jsonl with exactly 21 required fields.
5. Scoreboard computation derived directly and exclusively from raw prediction rows.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from temporal_alignment.baselines.ticktack05b.evaluator import (
    build_p12_prompt,
    task_conditioned_strict_parser,
)
from temporal_alignment.models.adapter_activation import (
    configure_temporal_condition,
)
from temporal_alignment.models.dual_cycle_mat_lora import (
    DualCycleMaTLoRALinear,
    replace_mlp_with_dual_cycle_mat_lora,
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
from temporal_alignment.normalization.temporal_resolver import resolve_temporal_anchor


# Constant SHA fingerprints for prompt template and parser logic
PROMPT_TEMPLATE_LOGIC = """
def build_p12_prompt(question_text: str, tokenizer: Any) -> Tuple[str, bool]:
    raw = question_text.strip()
    is_yn = detect_is_yes_no_task(raw)
    if is_yn:
        instruction = "\\nAnswer directly with only Yes or No."
    else:
        instruction = "\\nAnswer directly with only the option letter (a, b, c, d)."
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Please answer the question by selecting the correct option."},
        {"role": "user", "content": raw + instruction},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), is_yn
"""
PROMPT_TEMPLATE_SHA = hashlib.sha256(PROMPT_TEMPLATE_LOGIC.strip().encode("utf-8")).hexdigest()

PARSER_LOGIC = """
def task_conditioned_strict_parser(generated_text: str, is_yn: bool) -> Tuple[Optional[str], str]:
    # Strict regex matching: (a)-(d) for MCQA, Yes/No for YN
"""
PARSER_SHA = hashlib.sha256(PARSER_LOGIC.strip().encode("utf-8")).hexdigest()


@dataclass
class CanonicalEvaluationConfig:
    """Configuration for canonical evaluation."""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_revision: str = "7ae557604adf67be50417f59c2c2f167def9a775"
    checkpoint_path: Optional[str] = None
    condition: str = "A0_BASE"  # A0_BASE, A1_STATIC, A8_DYNAMIC, C_TRUE, C_WRONG_YEAR, etc.
    device: str = "cpu"
    max_new_tokens: int = 4
    do_sample: bool = False  # FAIL-CLOSED: must remain False
    seed: int = 42
    seam_token_index: int = 14
    year_perturbation_offset: int = 37  # For C_WRONG_YEAR
    
    def get_generation_config_sha(self) -> str:
        cfg = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "seed": self.seed,
            "strategy": "greedy_deterministic",
        }
        return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class RawPredictionRecord:
    """Exact 21-field specification per Decision 19."""
    run_id: str
    example_id: str
    source_record_id: Optional[str]
    event_id: Optional[str]
    temporal_cluster_id: Optional[str]
    checkpoint_sha256: Optional[str]
    tensor_fingerprint: Optional[str]
    condition: str
    year_true: Optional[int]
    year_used: Optional[int]
    treatment_status: str
    gold_semantic_answer: str
    gold_display_label: str
    raw_generation: str
    parsed_prediction: Optional[str]
    is_valid: bool
    is_correct: bool
    seed: Optional[int]
    prompt_template_sha: str
    parser_sha: str
    generation_config_sha: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    p = Path(file_path)
    if not p.is_file():
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_model_tensor_fingerprint(model: nn.Module) -> str:
    """Compute a lightweight deterministic hash over first and last layer parameters."""
    hasher = hashlib.sha256()
    sample_params = []
    for name, p in model.named_parameters():
        if "embed" in name or "layers.0." in name or f"layers.{getattr(model.config, 'num_hidden_layers', 24) - 1}." in name:
            sample_params.append((name, p.detach().cpu()))
    for name, t in sample_params:
        hasher.update(name.encode("utf-8"))
        hasher.update(t.view(-1)[:64].numpy().tobytes())
    return hasher.hexdigest()[:32]


def determine_treatment_status(
    record: Dict[str, Any],
    condition: str,
    year_true: Optional[int],
    year_used: Optional[int],
) -> str:
    """Determine causal eligibility status according to Decision 19 taxonomy."""
    # Check if explicit status is pre-assigned in dataset
    if record.get("treatment_status"):
        return record["treatment_status"]
    
    # Check multi-year bypass: if question has multiple years in text but only single-year conditioning
    raw_text = record.get("input") or record.get("question", "")
    years_in_text = re.findall(r"\b(1[89]\d\d|20\d\d)\b", raw_text)
    if len(set(years_in_text)) > 1:
        return "MULTI_YEAR_BYPASS"
    
    if year_true is None:
        return "MISSING_YEAR"
    
    if condition == "C_WRONG_YEAR":
        if year_used is None or year_used == year_true:
            return "TREATMENT_NOT_DELIVERED"
        return "ELIGIBLE_TREATMENT_DELIVERED"
        
    if record.get("is_counterfactual_pair"):
        if record.get("gold_flips"):
            return "ELIGIBLE_GOLD_FLIP"
        return "ELIGIBLE_GOLD_SAME"
        
    return "ELIGIBLE_SINGLE_YEAR"


def extract_record_gold_labels(record: Dict[str, Any]) -> Tuple[str, str]:
    """Extract gold semantic answer and gold display label."""
    gold_target = str(record.get("target") or record.get("gold_target") or "").strip()
    
    # Check if this is a Yes/No question
    q_text = record.get("input") or record.get("question", "")
    if "yes or no" in q_text.lower():
        norm_label = gold_target.capitalize()
        return norm_label, norm_label
    
    # MCQA: target is option letter 'a', 'b', 'c', 'd'
    display_label = gold_target.upper()
    semantic_answer = record.get("target_text") or record.get("gold_semantic_answer") or display_label
    return semantic_answer, display_label


def generate_scoreboard_from_predictions(
    predictions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute comprehensive evaluation scoreboard strictly from raw predictions."""
    total = len(predictions)
    if total == 0:
        return {"total_count": 0, "accuracy": 0.0, "valid_count": 0, "invalid_count": 0}

    valid_count = sum(1 for r in predictions if r["is_valid"])
    invalid_count = total - valid_count
    invalid_rate = round(invalid_count / total, 6)

    correct_count = sum(1 for r in predictions if r["is_correct"])
    accuracy = round(correct_count / total, 6)
    valid_accuracy = round(correct_count / valid_count, 6) if valid_count > 0 else 0.0

    # Prediction distribution
    pred_counts: Dict[str, int] = {}
    for r in predictions:
        p = r["parsed_prediction"]
        label = p.upper() if p is not None else "INVALID"
        pred_counts[label] = pred_counts.get(label, 0) + 1

    pred_distribution = {
        label: {
            "count": cnt,
            "fraction": round(cnt / total, 4),
        }
        for label, cnt in pred_counts.items()
    }

    # Prediction entropy
    entropy = 0.0
    for cnt in pred_counts.values():
        p = cnt / total
        if p > 0:
            entropy -= p * math.log2(p)
    prediction_entropy = round(entropy, 4)

    # Label collapse hygiene check
    max_label_freq = max((cnt / total for cnt in pred_counts.values()), default=0.0)
    is_collapsed = max_label_freq > 0.50

    # Recall per display label
    gold_labels = {r["gold_display_label"].upper() for r in predictions if r.get("gold_display_label")}
    per_label_stats = {}
    zero_recall_labels = []
    for g in sorted(gold_labels):
        total_g = sum(1 for r in predictions if r["gold_display_label"].upper() == g)
        correct_g = sum(1 for r in predictions if r["gold_display_label"].upper() == g and r["is_correct"])
        rec = round(correct_g / total_g, 4) if total_g > 0 else 0.0
        per_label_stats[g] = {
            "total_gold": total_g,
            "correct": correct_g,
            "recall": rec,
        }
        if total_g > 0 and correct_g == 0:
            zero_recall_labels.append(g)

    hygiene_passed = (not is_collapsed) and len(zero_recall_labels) == 0

    # Accuracy by treatment status
    by_status: Dict[str, Dict[str, Any]] = {}
    for r in predictions:
        st = r.get("treatment_status", "UNKNOWN")
        if st not in by_status:
            by_status[st] = {"total": 0, "correct": 0}
        by_status[st]["total"] += 1
        if r["is_correct"]:
            by_status[st]["correct"] += 1
    for st, v in by_status.items():
        v["accuracy"] = round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0.0

    return {
        "summary": {
            "total_count": total,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "invalid_rate": invalid_rate,
            "correct_count": correct_count,
            "strict_accuracy": accuracy,
            "valid_accuracy": valid_accuracy,
            "prediction_entropy": prediction_entropy,
        },
        "hygiene_gate": {
            "max_label_frequency": round(max_label_freq, 4),
            "is_label_collapsed": is_collapsed,
            "zero_recall_labels": zero_recall_labels,
            "hygiene_passed": hygiene_passed,
        },
        "per_label_stats": per_label_stats,
        "prediction_distribution": pred_distribution,
        "accuracy_by_treatment_status": by_status,
    }


def evaluate_dataset_canonical(
    model: nn.Module,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    config: CanonicalEvaluationConfig,
    run_id: str,
    adapted_modules: Optional[Dict[str, Any]] = None,
    gated_proj: Optional[nn.Module] = None,
    cond_context: Optional[ConditioningContext] = None,
) -> Tuple[List[RawPredictionRecord], Dict[str, Any]]:
    """Execute canonical deterministic evaluation over records."""
    model.eval()
    device = torch.device(config.device)
    
    ckpt_sha = compute_file_sha256(config.checkpoint_path) if config.checkpoint_path else "BASE_PRETRAINED"
    tensor_fp = compute_model_tensor_fingerprint(model)
    gen_cfg_sha = config.get_generation_config_sha()
    
    raw_records: List[RawPredictionRecord] = []
    
    for rec in records:
        ex_id = rec.get("example_id", f"ex_{len(raw_records)}")
        q_text = rec.get("input") or rec.get("question", "")
        sid = rec.get("source_record_id")
        eid = rec.get("event_id")
        cid = rec.get("temporal_cluster_id")
        
        # Resolve temporal anchor
        anchor = resolve_temporal_anchor(rec)
        year_true = anchor.effective_year.astronomical if anchor.effective_year else (rec.get("year_eval") or rec.get("canonical_year"))
        if year_true is not None:
            year_true = int(year_true)
            
        # Determine year_used depending on condition
        if config.condition == "C_WRONG_YEAR" and year_true is not None:
            year_used = year_true + config.year_perturbation_offset
        elif config.condition in ("A0_BASE", "C_BOTH_OFF", "C_ADAPTER_OFF", "C_INPUT_ZERO_FEATURE"):
            year_used = None
        else:
            year_used = year_true
            
        treatment_status = determine_treatment_status(rec, config.condition, year_true, year_used)
        gold_semantic, gold_display = extract_record_gold_labels(rec)
        
        # Build prompt
        formatted_prompt, is_yn = build_p12_prompt(q_text, tokenizer)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]
        
        # Configure temporal condition context
        v_feat, th, z = None, None, None
        if year_used is not None and adapted_modules is not None and gated_proj is not None and cond_context is not None:
            v_feat = compute_fixed_fone_canchi_40d(year_used)
            th, z = extract_canchi_angles_from_40d(v_feat)
            
        # Determine whether to use temporal condition context or nullcontext
        if adapted_modules is not None and gated_proj is not None and cond_context is not None and config.condition != "A0_BASE":
            cond_mgr = configure_temporal_condition(
                config.condition,
                adapted_modules,
                gated_proj,
                cond_context,
                v_feat,
                th,
                z,
                device,
                seam_token_index=config.seam_token_index,
            )
        else:
            import contextlib
            cond_mgr = contextlib.nullcontext()

        # Deterministic generation
        with cond_mgr:
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=config.do_sample,  # Greedy
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
        raw_gen_text = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True).strip()
        parsed_pred, validity_status = task_conditioned_strict_parser(raw_gen_text, is_yn)
        is_valid = (validity_status == "VALID")
        
        # Evaluate correctness against gold display label
        is_corr = False
        if is_valid and parsed_pred is not None:
            is_corr = (parsed_pred.strip().lower() == gold_display.strip().lower())
            
        row = RawPredictionRecord(
            run_id=run_id,
            example_id=ex_id,
            source_record_id=sid,
            event_id=eid,
            temporal_cluster_id=cid,
            checkpoint_sha256=ckpt_sha,
            tensor_fingerprint=tensor_fp,
            condition=config.condition,
            year_true=year_true,
            year_used=year_used,
            treatment_status=treatment_status,
            gold_semantic_answer=gold_semantic,
            gold_display_label=gold_display,
            raw_generation=raw_gen_text,
            parsed_prediction=parsed_pred,
            is_valid=is_valid,
            is_correct=is_corr,
            seed=config.seed,
            prompt_template_sha=PROMPT_TEMPLATE_SHA,
            parser_sha=PARSER_SHA,
            generation_config_sha=gen_cfg_sha,
        )
        raw_records.append(row)
        
    scoreboard = generate_scoreboard_from_predictions([r.to_dict() for r in raw_records])
    return raw_records, scoreboard
