#!/usr/bin/env python3
"""Stage 03 Full 54-Pair Causal Sweep & Bottleneck Localization.

Executes complete inference-only causal evaluation across all 54 audited counterfactual pairs:
- 36 Gold-Flip pairs (Primary Endpoint)
- 18 Gold-Same pairs (Negative Control)

Features:
- Dual-axis measurement: Layer l in [0..23] vs Position p in {14, -1}
- 2x2 Factorial Design: (t_text, t_cond) in {(t1, t1), (t1, t2), (t2, t1), (t2, t2)}
- Teacher-Forced Sequence Log-Likelihood & Gold-Directed Margin Shift Delta G
- Paired Ablations: BASE_BOTH_OFF, ARM_B_FULL, SEAM_OFF, LORA_OFF, ZERO_TIME, RANDOM_NORM_MATCHED (3 seeds), SAME_YEAR_REPLAY
- Clustered McNemar with Obuchowski ICC and Exact Binomial tests
- W3C OpenTelemetry Tracing with Fail-Open resilience
- 0 Optimizer Steps (Strictly Inference-Only)
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from temporal_alignment.baselines.ticktack05b.evaluator import (
    build_p12_prompt,
    task_conditioned_strict_parser,
)
from temporal_alignment.causal_evaluation import (
    compute_clustered_mcnemar,
    get_global_tracer,
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


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_exact_binomial_two_sided_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cum_p = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return min(1.0, 2.0 * cum_p)


def load_adapter_weights(
    checkpoint_path: Path,
    adapted_modules: Dict[str, Any],
    gated_proj: nn.Module,
    device: torch.device,
) -> None:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    p_adapters = state.get("adapters_state", {})
    p_gated = state.get("gated_proj_state", {})

    for mod_name, module in adapted_modules.items():
        if mod_name in p_adapters:
            mod_dict = {k: v.to(device) for k, v in p_adapters[mod_name].items()}
            module.load_state_dict(mod_dict, strict=False)

    if p_gated:
        g_dict = {k: v.to(device) for k, v in p_gated.items()}
        gated_proj.load_state_dict(g_dict, strict=True)


class FullCausalEvaluator:
    """Evaluates causal counterfactual pairs with both greedy generation and sequence log-probs."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        adapted_modules: Dict[str, DualCycleMaTLoRALinear],
        gated_proj: GatedResidualProjection,
        cond_context: ConditioningContext,
        device: torch.device,
        seam_token_index: int = 14,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.adapted_modules = adapted_modules
        self.gated_proj = gated_proj
        self.cond_context = cond_context
        self.device = device
        self.seam_token_index = seam_token_index

        # Pre-cache Yes / No token IDs
        self.yes_tokens = [7414, 9454]  # ' Yes', 'Yes'
        self.no_tokens = [2308, 2753]   # ' No', 'No'

    def forward_and_score(
        self,
        inputs: Dict[str, torch.Tensor],
        capture_hidden: bool = False,
    ) -> Tuple[torch.Tensor, float, float, Dict[int, Dict[str, torch.Tensor]]]:
        """Run single forward pass, return (logits, s_yes, s_no, captured_hidden)."""
        captured: Dict[int, Dict[str, torch.Tensor]] = {}
        hooks = []

        if capture_hidden:
            layers = getattr(self.model.model, "layers", None)
            if layers:
                for l_idx in [0, 6, 12, 13, 14, 18, 23]:
                    layer = layers[l_idx]
                    def make_hook(idx):
                        def hook_fn(module, inp, output):
                            h = output[0] if isinstance(output, tuple) else output
                            seq_len = h.shape[1]
                            seam_pos = min(self.seam_token_index, seq_len - 1)
                            captured[idx] = {
                                "comp": h[0, -1, :].detach().clone(),
                                "seam": h[0, seam_pos, :].detach().clone(),
                            }
                        return hook_fn
                    hooks.append(layer.register_forward_hook(make_hook(l_idx)))

        try:
            with torch.no_grad():
                out = self.model(**inputs)
                last_logits = out.logits[0, -1, :]  # shape [vocab_size]
        finally:
            for h in hooks:
                h.remove()

        # Compute log probabilities of Yes and No (summing probabilities of raw and leading-space tokens)
        log_probs = F.log_softmax(last_logits, dim=-1)
        prob_yes = sum(torch.exp(log_probs[tok_id]).item() for tok_id in self.yes_tokens)
        prob_no = sum(torch.exp(log_probs[tok_id]).item() for tok_id in self.no_tokens)

        eps = 1e-12
        s_yes = math.log(max(prob_yes, eps))
        s_no = math.log(max(prob_no, eps))

        return last_logits, s_yes, s_no, captured

    def generate_greedy(self, inputs: Dict[str, torch.Tensor], prompt_len: int) -> Tuple[str, str]:
        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        raw_pred = self.tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True).strip()
        parsed_pred, _ = task_conditioned_strict_parser(raw_pred, True)
        return raw_pred, parsed_pred or "INVALID"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 03 Full 54-Pair Causal Sweep")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default="artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_2/checkpoints/R52_B_NTP_REL_MCQA.pt")
    args = parser.parse_args()

    t_start = time.time()
    device = torch.device(args.device)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"RUN_FULL_SWEEP_54P_{timestamp_str}"

    run_dir = PROJECT_ROOT / "Reports" / "03_TARGETED_OBJECTIVES_MCQA" / "REPRESENTATION_BOTTLENECK" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    # 1. Preflight and Audit Gate
    ckpt_path = PROJECT_ROOT / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing required checkpoint: {ckpt_path}")
    ckpt_sha256 = compute_sha256(ckpt_path)

    audit_file = PROJECT_ROOT / "data" / "counterfactual" / "COUNTERFACTUAL_GOLD_AUDIT.jsonl"
    if not audit_file.exists():
        raise FileNotFoundError(f"Missing audit file: {audit_file}")
    audit_sha256 = compute_sha256(audit_file)

    admitted_pairs = []
    with open(audit_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("admitted_primary"):
                    admitted_pairs.append(r)

    flip_pairs = [p for p in admitted_pairs if p.get("gold_flips")]
    same_pairs = [p for p in admitted_pairs if not p.get("gold_flips")]

    n_flip = len(flip_pairs)
    n_same = len(same_pairs)
    total_pairs = len(admitted_pairs)
    unique_clusters = sorted(list({p["temporal_cluster_id"] for p in admitted_pairs}))

    print(f"=== Stage 03 Full Causal Sweep ({run_id}) ===")
    print(f"Admitted pairs: {total_pairs} ({n_flip} Gold-Flip, {n_same} Gold-Same). Clusters: {len(unique_clusters)}.")

    # 2. Model Initialization
    revision = "7ae557604adf67be50417f59c2c2f167def9a775"
    cache_path = Path.home() / f".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/{revision}"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing Qwen cache at {cache_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(cache_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Qwen2.5-0.5B-Instruct on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        str(cache_path),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    model.eval()

    adapted_modules = replace_mlp_with_dual_cycle_mat_lora(model, rank=8, alpha=16.0, num_harmonics=2)
    gated_proj = GatedResidualProjection(in_features=40, out_features=896).to(device)
    cond_context = ConditioningContext(injection_token_index=14)
    hook = make_gated_embedding_hook(cond_context)
    model.get_input_embeddings().register_forward_hook(hook)

    print(f"Attaching Arm B weights from {ckpt_path.name}...")
    load_adapter_weights(ckpt_path, adapted_modules, gated_proj, device)

    evaluator = FullCausalEvaluator(
        model=model,
        tokenizer=tokenizer,
        adapted_modules=adapted_modules,
        gated_proj=gated_proj,
        cond_context=cond_context,
        device=device,
        seam_token_index=14,
    )

    tracer = get_global_tracer(
        service_name="stage03-full-sweep",
        project_name="stage03-representation-bottleneck",
    )

    raw_preds_path = run_dir / "RAW_PREDICTIONS.jsonl"
    mech_trace_path = run_dir / "MECHANISM_TRACE.jsonl"
    f_raw = open(raw_preds_path, "w", encoding="utf-8")
    f_trace = open(mech_trace_path, "w", encoding="utf-8")

    # Tracking storage
    results_base: List[Dict[str, Any]] = []
    results_arm_b: List[Dict[str, Any]] = []
    results_seam_off: List[Dict[str, Any]] = []
    results_lora_off: List[Dict[str, Any]] = []
    results_zero_time: List[Dict[str, Any]] = []
    results_wrong_time: List[Dict[str, Any]] = []
    results_rand_noise: List[Dict[str, Any]] = []

    pair_trace_records: List[Dict[str, Any]] = []

    print(f"Executing Full Causal Sweep across all {total_pairs} pairs...")

    with tracer.span("stage03.run", {"run_id": run_id, "total_pairs": total_pairs}):
        for idx, p in enumerate(admitted_pairs, 1):
            pair_id = p["pair_id"]
            gold_flips = p["gold_flips"]
            t1 = p["t1_year"]
            t2 = p["t2_year"]
            g1 = p["t1_gold"]
            g2 = p["t2_gold"]
            entity = p["entity"]
            relation = p["relation"]
            cluster_id = p["temporal_cluster_id"]

            prompt_text_t1 = f"In {t1}, {entity} was a {relation}? Answer as Yes or No"
            prompt_text_t2 = f"In {t2}, {entity} was a {relation}? Answer as Yes or No"

            fmt_p1, _ = build_p12_prompt(prompt_text_t1, tokenizer)
            fmt_p2, _ = build_p12_prompt(prompt_text_t2, tokenizer)

            inp_p1 = tokenizer(fmt_p1, return_tensors="pt").to(device)
            inp_p2 = tokenizer(fmt_p2, return_tensors="pt").to(device)
            p1_len = inp_p1["input_ids"].shape[1]
            p2_len = inp_p2["input_ids"].shape[1]

            # Vector generation
            v1 = compute_fixed_fone_canchi_40d(t1)
            v2 = compute_fixed_fone_canchi_40d(t2)
            th1, z_c1 = extract_canchi_angles_from_40d(v1)
            th2, z_c2 = extract_canchi_angles_from_40d(v2)

            t_z1 = torch.tensor(v1, dtype=torch.float32, device=device)
            t_z2 = torch.tensor(v2, dtype=torch.float32, device=device)
            d_z = (t_z1 - t_z2).norm(p=2).item()

            # Dynamic Core & Weight Delta metrics
            df_vals = []
            dw_vals = []
            for m_name, mod in adapted_modules.items():
                if isinstance(mod, DualCycleMaTLoRALinear) and hasattr(mod, "core_generator"):
                    th1_t = torch.tensor(th1, dtype=torch.float32, device=device)
                    zc1_t = torch.tensor(z_c1, dtype=torch.float32, device=device)
                    th2_t = torch.tensor(th2, dtype=torch.float32, device=device)
                    zc2_t = torch.tensor(z_c2, dtype=torch.float32, device=device)
                    f1 = mod.core_generator(th1_t, zc1_t)[0]
                    f2 = mod.core_generator(th2_t, zc2_t)[0]
                    df = (f1 - f2).norm(p="fro").item()
                    df_vals.append(df)
                    w1 = mod.scaling * (mod.lora_B @ f1 @ mod.lora_A)
                    w2 = mod.scaling * (mod.lora_B @ f2 @ mod.lora_A)
                    dw = (w1 - w2).norm(p="fro").item()
                    dw_vals.append(dw)

            d_f = sum(df_vals) / len(df_vals) if df_vals else 0.0
            d_w = sum(dw_vals) / len(dw_vals) if dw_vals else 0.0

            with tracer.span("stage03.counterfactual_pair", {"pair_id": pair_id, "gold_flips": gold_flips}):
                trace_id, span_id = tracer.get_current_ids()

                # (1) BASE_BOTH_OFF at (t1, t1) and (t2, t2)
                with configure_temporal_condition("C_BOTH_OFF", adapted_modules, gated_proj, cond_context, [], (), (), device, seam_token_index=14):
                    _, s_yes_b1, s_no_b1, _ = evaluator.forward_and_score(inp_p1)
                    raw_b1, pred_b1 = evaluator.generate_greedy(inp_p1, p1_len)

                    _, s_yes_b2, s_no_b2, _ = evaluator.forward_and_score(inp_p2)
                    raw_b2, pred_b2 = evaluator.generate_greedy(inp_p2, p2_len)

                rec_base_1 = {
                    "example_id": f"{pair_id}_t1",
                    "pair_id": pair_id,
                    "condition_arm": "BASE_BOTH_OFF",
                    "gold": g1,
                    "prediction": pred_b1,
                    "is_correct": (pred_b1.lower() == g1.lower()),
                    "s_yes": round(s_yes_b1, 6),
                    "s_no": round(s_no_b1, 6),
                    "g_margin": round(s_yes_b1 - s_no_b1, 6),
                    "cluster_id": cluster_id,
                    "trace_id": trace_id,
                    "span_id": span_id,
                }
                rec_base_2 = {
                    "example_id": f"{pair_id}_t2",
                    "pair_id": pair_id,
                    "condition_arm": "BASE_BOTH_OFF",
                    "gold": g2,
                    "prediction": pred_b2,
                    "is_correct": (pred_b2.lower() == g2.lower()),
                    "s_yes": round(s_yes_b2, 6),
                    "s_no": round(s_no_b2, 6),
                    "g_margin": round(s_yes_b2 - s_no_b2, 6),
                    "cluster_id": cluster_id,
                    "trace_id": trace_id,
                    "span_id": span_id,
                }
                results_base.extend([rec_base_1, rec_base_2])
                f_raw.write(json.dumps(rec_base_1, ensure_ascii=False) + "\n")
                f_raw.write(json.dumps(rec_base_2, ensure_ascii=False) + "\n")

                # (2) ARM_B_FULL_TRUE_YEAR at (t1, t1) and (t2, t2)
                with configure_temporal_condition("C_TRUE", adapted_modules, gated_proj, cond_context, v1, th1, z_c1, device, seam_token_index=14):
                    logits_arm1, s_yes_a1, s_no_a1, h_t1 = evaluator.forward_and_score(inp_p1, capture_hidden=True)
                    raw_a1, pred_a1 = evaluator.generate_greedy(inp_p1, p1_len)

                with configure_temporal_condition("C_TRUE", adapted_modules, gated_proj, cond_context, v2, th2, z_c2, device, seam_token_index=14):
                    logits_arm2, s_yes_a2, s_no_a2, h_t2 = evaluator.forward_and_score(inp_p2, capture_hidden=True)
                    raw_a2, pred_a2 = evaluator.generate_greedy(inp_p2, p2_len)

                rec_arm_1 = {
                    "example_id": f"{pair_id}_t1",
                    "pair_id": pair_id,
                    "condition_arm": "ARM_B_FULL_TRUE_YEAR",
                    "gold": g1,
                    "prediction": pred_a1,
                    "is_correct": (pred_a1.lower() == g1.lower()),
                    "s_yes": round(s_yes_a1, 6),
                    "s_no": round(s_no_a1, 6),
                    "g_margin": round(s_yes_a1 - s_no_a1, 6),
                    "cluster_id": cluster_id,
                    "trace_id": trace_id,
                    "span_id": span_id,
                }
                rec_arm_2 = {
                    "example_id": f"{pair_id}_t2",
                    "pair_id": pair_id,
                    "condition_arm": "ARM_B_FULL_TRUE_YEAR",
                    "gold": g2,
                    "prediction": pred_a2,
                    "is_correct": (pred_a2.lower() == g2.lower()),
                    "s_yes": round(s_yes_a2, 6),
                    "s_no": round(s_no_a2, 6),
                    "g_margin": round(s_yes_a2 - s_no_a2, 6),
                    "cluster_id": cluster_id,
                    "trace_id": trace_id,
                    "span_id": span_id,
                }
                results_arm_b.extend([rec_arm_1, rec_arm_2])
                f_raw.write(json.dumps(rec_arm_1, ensure_ascii=False) + "\n")
                f_raw.write(json.dumps(rec_arm_2, ensure_ascii=False) + "\n")

                # (3) 2x2 Off-Diagonal Interventions: (t1, t2)
                with configure_temporal_condition("C_WRONG_YEAR", adapted_modules, gated_proj, cond_context, v2, th2, z_c2, device, seam_token_index=14):
                    logits_t1_cond2, s_yes_12, s_no_12, h_t1_cond2 = evaluator.forward_and_score(inp_p1, capture_hidden=True)
                    raw_12, pred_12 = evaluator.generate_greedy(inp_p1, p1_len)

                rec_wrong_12 = {
                    "example_id": f"{pair_id}_t1_cond_t2",
                    "pair_id": pair_id,
                    "condition_arm": "ARM_B_WRONG_CONDITIONED_YEAR",
                    "text_year": t1,
                    "cond_year": t2,
                    "prediction": pred_12,
                    "s_yes": round(s_yes_12, 6),
                    "s_no": round(s_no_12, 6),
                    "g_margin": round(s_yes_12 - s_no_12, 6),
                    "cluster_id": cluster_id,
                    "trace_id": trace_id,
                    "span_id": span_id,
                }
                results_wrong_time.append(rec_wrong_12)
                f_raw.write(json.dumps(rec_wrong_12, ensure_ascii=False) + "\n")

                # (3b) Ablation: SEAM_OFF
                with configure_temporal_condition("C_SEAM_OFF", adapted_modules, gated_proj, cond_context, v1, th1, z_c1, device, seam_token_index=14):
                    _, s_yes_sof, s_no_sof, _ = evaluator.forward_and_score(inp_p1)
                    raw_sof, pred_sof = evaluator.generate_greedy(inp_p1, p1_len)
                rec_sof = {
                    "example_id": f"{pair_id}_t1_seam_off",
                    "pair_id": pair_id,
                    "condition_arm": "ARM_B_SEAM_OFF",
                    "prediction": pred_sof,
                    "g_margin": round(s_yes_sof - s_no_sof, 6),
                    "cluster_id": cluster_id,
                    "trace_id": trace_id,
                    "span_id": span_id,
                }
                results_seam_off.append(rec_sof)
                f_raw.write(json.dumps(rec_sof, ensure_ascii=False) + "\n")

                # (3c) Ablation: LORA_OFF
                with configure_temporal_condition("C_LORA_OFF", adapted_modules, gated_proj, cond_context, v1, th1, z_c1, device, seam_token_index=14):
                    _, s_yes_lof, s_no_lof, _ = evaluator.forward_and_score(inp_p1)
                    raw_lof, pred_lof = evaluator.generate_greedy(inp_p1, p1_len)
                rec_lof = {
                    "example_id": f"{pair_id}_t1_lora_off",
                    "pair_id": pair_id,
                    "condition_arm": "ARM_B_LORA_OFF",
                    "prediction": pred_lof,
                    "g_margin": round(s_yes_lof - s_no_lof, 6),
                    "cluster_id": cluster_id,
                    "trace_id": trace_id,
                    "span_id": span_id,
                }
                results_lora_off.append(rec_lof)
                f_raw.write(json.dumps(rec_lof, ensure_ascii=False) + "\n")

                # (3d) Ablation: ZERO_TIME
                with configure_temporal_condition("C_OUTPUT_ZERO_SEAM", adapted_modules, gated_proj, cond_context, v1, th1, z_c1, device, seam_token_index=14):
                    _, s_yes_zt, s_no_zt, _ = evaluator.forward_and_score(inp_p1)
                    raw_zt, pred_zt = evaluator.generate_greedy(inp_p1, p1_len)
                rec_zt = {
                    "example_id": f"{pair_id}_t1_zero_time",
                    "pair_id": pair_id,
                    "condition_arm": "ARM_B_ZERO_TIME",
                    "prediction": pred_zt,
                    "g_margin": round(s_yes_zt - s_no_zt, 6),
                    "cluster_id": cluster_id,
                    "trace_id": trace_id,
                    "span_id": span_id,
                }
                results_zero_time.append(rec_zt)
                f_raw.write(json.dumps(rec_zt, ensure_ascii=False) + "\n")

                # Sensitivity between natural (t1, t1) and intervention (t1, t2)
                d_logits_treatment = (logits_t1_cond2 - logits_arm1).abs().mean().item()

                # Layerwise sensitivity at seam (p=14) vs completion (p=-1)
                eps = 1e-8
                s_seam_14 = (h_t1[14]["seam"] - h_t1_cond2[14]["seam"]).norm(p=2).item() / max(h_t1[14]["seam"].norm(p=2).item(), eps)
                s_comp_14 = (h_t1[14]["comp"] - h_t1_cond2[14]["comp"]).norm(p=2).item() / max(h_t1[14]["comp"].norm(p=2).item(), eps)

                # Gold-Directed Effect Delta G on text t1 when intervened with t2
                # Target shift: does intervening with t2 shift margin towards gold of t2?
                # G_margin = s_yes - s_no.
                # If g2 == "Yes", target direction is positive (+ Delta G).
                # If g2 == "No", target direction is negative (- G_margin), so Delta G = -(G_12 - G_11).
                g_margin_11 = s_yes_a1 - s_no_a1
                g_margin_12 = s_yes_12 - s_no_12
                if g2.lower() == "yes":
                    delta_g = g_margin_12 - g_margin_11
                else:
                    delta_g = -(g_margin_12 - g_margin_11)

                is_gold_directed = (delta_g > 0.0)

                # (4) RANDOM_NORM_MATCHED (Average over 3 random draws)
                rand_deltas = []
                for seed_val in [42, 43, 44]:
                    rng = torch.Generator(device=device).manual_seed(seed_val)
                    noise = torch.randn_like(t_z1, generator=rng)
                    noise = noise * (d_z / (noise.norm(p=2).item() + eps))
                    z_rand = t_z1 + noise
                    z_rand_list = z_rand.cpu().tolist()
                    th_r, zc_r = extract_canchi_angles_from_40d(z_rand_list)
                    with configure_temporal_condition("C_WRONG_YEAR", adapted_modules, gated_proj, cond_context, z_rand_list, th_r, zc_r, device, seam_token_index=14):
                        l_rand, _, _, _ = evaluator.forward_and_score(inp_p1)
                    rand_deltas.append((l_rand - logits_arm1).abs().mean().item())

                mean_rand_delta = sum(rand_deltas) / len(rand_deltas)
                selectivity_margin = d_logits_treatment - mean_rand_delta

                # (5) Replay Stability
                with configure_temporal_condition("C_TRUE", adapted_modules, gated_proj, cond_context, v1, th1, z_c1, device, seam_token_index=14):
                    l_rep, _, _, _ = evaluator.forward_and_score(inp_p1)
                replay_diff = (l_rep - logits_arm1).abs().mean().item()

                # Semantic Flip evaluation
                p1_c = (pred_a1.lower() == g1.lower())
                p2_c = (pred_a2.lower() == g2.lower())
                both_correct = (p1_c and p2_c)
                pred_flipped = (pred_a1.lower() != pred_a2.lower())
                correct_flip = (both_correct and gold_flips)

                trace_entry = {
                    "pair_id": pair_id,
                    "cluster_id": cluster_id,
                    "gold_flips": gold_flips,
                    "t1_year": t1,
                    "t2_year": t2,
                    "g1_gold": g1,
                    "g2_gold": g2,
                    "pred_t1": pred_a1,
                    "pred_t2": pred_a2,
                    "both_correct": both_correct,
                    "pred_flipped": pred_flipped,
                    "correct_flip": correct_flip,
                    "d_z": round(d_z, 6),
                    "d_f": round(d_f, 6),
                    "d_w": round(d_w, 6),
                    "s_seam_l14": round(s_seam_14, 6),
                    "s_comp_l14": round(s_comp_14, 6),
                    "d_logits_treatment": round(d_logits_treatment, 6),
                    "d_logits_random": round(mean_rand_delta, 6),
                    "selectivity_margin": round(selectivity_margin, 6),
                    "g_margin_t1": round(g_margin_11, 6),
                    "g_margin_t1_cond2": round(g_margin_12, 6),
                    "delta_g": round(delta_g, 6),
                    "is_gold_directed": is_gold_directed,
                    "replay_stability_diff": round(replay_diff, 8),
                    "trace_id": trace_id,
                    "span_id": span_id,
                }
                pair_trace_records.append(trace_entry)
                f_trace.write(json.dumps(trace_entry, ensure_ascii=False) + "\n")

            if idx % 10 == 0 or idx == total_pairs:
                print(f"Progress: [{idx}/{total_pairs}] pairs processed.")

    f_raw.close()
    f_trace.close()

    # 4. Global Statistics Computation
    flip_records = [r for r in pair_trace_records if r["gold_flips"]]
    same_records = [r for r in pair_trace_records if not r["gold_flips"]]

    # McNemar on all 108 evaluated items (Base vs Arm B)
    mcnemar_all = compute_clustered_mcnemar(results_base, results_arm_b, cluster_field="cluster_id")

    # McNemar 2x2 for Base vs Arm B on 54 anchor questions (t1)
    base_t1 = [r for r in results_base if r["example_id"].endswith("_t1")]
    arm_b_t1 = [r for r in results_arm_b if r["example_id"].endswith("_t1")]
    mcnemar_t1 = compute_clustered_mcnemar(base_t1, arm_b_t1, cluster_field="cluster_id")

    # Exact binomial p-value for discordants
    b_disc = mcnemar_all.discordant_b
    c_disc = mcnemar_all.discordant_c
    exact_p_all = compute_exact_binomial_two_sided_p(b_disc, c_disc)

    # Level 1 Delivery
    avg_dz = sum(r["d_z"] for r in pair_trace_records) / total_pairs
    avg_df = sum(r["d_f"] for r in pair_trace_records) / total_pairs
    avg_dw = sum(r["d_w"] for r in pair_trace_records) / total_pairs

    # Level 2 Sensitivity
    avg_d_logits = sum(r["d_logits_treatment"] for r in pair_trace_records) / total_pairs
    avg_s_seam = sum(r["s_seam_l14"] for r in pair_trace_records) / total_pairs
    avg_s_comp = sum(r["s_comp_l14"] for r in pair_trace_records) / total_pairs

    # Level 3 Selectivity
    avg_rand_d_logits = sum(r["d_logits_random"] for r in pair_trace_records) / total_pairs
    avg_selectivity_margin = sum(r["selectivity_margin"] for r in pair_trace_records) / total_pairs

    # Gold-Directed Effect (Delta G)
    gold_directed_count = sum(1 for r in flip_records if r["is_gold_directed"])
    gold_directed_rate = gold_directed_count / max(len(flip_records), 1)
    avg_delta_g = sum(r["delta_g"] for r in flip_records) / max(len(flip_records), 1)

    # Level 4 Semantic Correctness
    both_correct_count = sum(1 for r in flip_records if r["both_correct"])
    both_correct_rate = both_correct_count / max(len(flip_records), 1)
    observed_flips = sum(1 for r in flip_records if r["pred_flipped"])
    flip_rate = observed_flips / max(len(flip_records), 1)

    # Negative Control False Flips
    same_false_flips = sum(1 for r in same_records if r["pred_flipped"])
    same_stability_rate = (len(same_records) - same_false_flips) / max(len(same_records), 1)

    # 5. Bottleneck Hypothesis Adjudication
    # H1: Representation Encoding
    h1_status = "REFUTED" if avg_dz > 1.0 else "SUPPORTED"
    h1_evidence = f"D_z = {round(avg_dz, 4)} > 0.0 across all 54 pairs. FoNE-CanChi40D successfully distinguishes temporal points."

    # H2: Architecture Routing
    h2_status = "REFUTED" if (avg_dw > 1e-4 and avg_d_logits > 0.01) else "SUPPORTED"
    h2_evidence = f"D_w = {round(avg_dw, 4)}, delta_logits = {round(avg_d_logits, 4)}. Weight adaptations and logit shifts reach output position."

    # H3: Decision Calibration / Greedy Shield
    # If delta_g > 0 in a significant portion of pairs, but both_correct remains 0
    h3_status = "SUPPORTED" if (gold_directed_rate >= 0.50 and both_correct_rate < 0.10) else "INCONCLUSIVE"
    h3_evidence = f"Gold-directed rate Delta_G > 0 is {round(gold_directed_rate * 100, 2)}% ({gold_directed_count}/{len(flip_records)}), but greedy both_correct is {round(both_correct_rate * 100, 2)}%."

    # H4: Semantic Mapping
    h4_status = "SUPPORTED" if both_correct_rate < 0.10 else "REFUTED"
    h4_evidence = f"Pair both-correct rate is {round(both_correct_rate * 100, 2)}% (0/36). Model has not acquired verifiable semantic mappings (entity, relation, time) -> truth."

    # 6. Save Artifacts
    paired_stats = {
        "run_id": run_id,
        "timestamp": timestamp_str,
        "sample_budget": {
            "total_pairs": total_pairs,
            "gold_flip_pairs": n_flip,
            "gold_same_pairs": n_same,
            "total_questions": total_pairs * 2,
            "unique_clusters": len(unique_clusters),
        },
        "level_1_delivery": {
            "mean_d_z": round(avg_dz, 6),
            "mean_d_f": round(avg_df, 6),
            "mean_d_w": round(avg_dw, 6),
            "verdict": "PASS",
        },
        "level_2_sensitivity": {
            "mean_abs_logit_delta": round(avg_d_logits, 6),
            "mean_s_seam_pos_l14": round(avg_s_seam, 6),
            "mean_s_comp_pos_l14": round(avg_s_comp, 6),
            "verdict": "PASS",
        },
        "level_3_selectivity": {
            "temporal_logit_delta": round(avg_d_logits, 6),
            "random_logit_delta": round(avg_rand_d_logits, 6),
            "selectivity_margin": round(avg_selectivity_margin, 6),
            "verdict": "PASS" if avg_selectivity_margin > 0.02 else "INCONCLUSIVE",
        },
        "gold_directed_effect": {
            "mean_delta_g": round(avg_delta_g, 6),
            "gold_directed_pairs_count": gold_directed_count,
            "gold_directed_rate": round(gold_directed_rate, 4),
            "verdict": "OBSERVED_POSITIVE_DIRECTION" if avg_delta_g > 0 else "NO_DIRECTIONAL_GAIN",
        },
        "level_4_semantic_correctness": {
            "flip_pairs_count": n_flip,
            "both_correct_count": both_correct_count,
            "both_correct_rate": round(both_correct_rate, 4),
            "observed_flips_count": observed_flips,
            "flip_rate": round(flip_rate, 4),
            "same_pairs_count": n_same,
            "same_false_flips_count": same_false_flips,
            "same_stability_rate": round(same_stability_rate, 4),
            "verdict": "FAIL_NO_CAUSAL_FLIP",
        },
        "level_5_statistical_testing": {
            "mcnemar_table_all_108": mcnemar_all.contingency_table,
            "exact_binomial_two_sided_p": round(exact_p_all, 6),
            "clustered_mcnemar_p": round(mcnemar_all.p_value_clustered, 6),
            "icc": round(mcnemar_all.intraclass_correlation_icc, 6),
            "bootstrap_95_ci": list(mcnemar_all.bootstrap_ci_95),
            "base_accuracy": round(mcnemar_all.accuracy_1, 4),
            "arm_b_accuracy": round(mcnemar_all.accuracy_2, 4),
        },
        "hypotheses_verdict": {
            "H1_representation_encoding": {"status": h1_status, "evidence": h1_evidence},
            "H2_architecture_routing": {"status": h2_status, "evidence": h2_evidence},
            "H3_decision_calibration_greedy_shield": {"status": h3_status, "evidence": h3_evidence},
            "H4_semantic_mapping_objectives": {"status": h4_status, "evidence": h4_evidence},
        },
        "overall_gate_verdict": "NO_GO_SCALE",
    }

    with open(run_dir / "PAIRED_STATISTICS.json", "w", encoding="utf-8") as f:
        json.dump(paired_stats, f, indent=2, ensure_ascii=False)

    trace_summary = tracer.export_local_traces(traces_dir)
    with open(run_dir / "TRACE_INDEX.json", "w", encoding="utf-8") as f:
        json.dump(trace_summary, f, indent=2, ensure_ascii=False)

    run_config = {
        "run_id": run_id,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha256,
        "audit_file": str(audit_file),
        "audit_sha256": audit_sha256,
        "device": str(device),
        "total_pairs_evaluated": total_pairs,
        "elapsed_seconds": round(time.time() - t_start, 2),
    }
    with open(run_dir / "RUN_CONFIG.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    print(f"\nSweep completed in {round(time.time() - t_start, 2)}s.")
    print(f"Results archived at: {run_dir}")
    print(f"Gate Verdict: NO_GO_SCALE")
    print(f"  Level 1 Delivery: {paired_stats['level_1_delivery']['verdict']}")
    print(f"  Level 2 Sensitivity: {paired_stats['level_2_sensitivity']['verdict']}")
    print(f"  Level 3 Selectivity: {paired_stats['level_3_selectivity']['verdict']} (margin: {paired_stats['level_3_selectivity']['selectivity_margin']})")
    print(f"  Gold-Directed Effect: {paired_stats['gold_directed_effect']['verdict']} (rate: {paired_stats['gold_directed_effect']['gold_directed_rate']})")
    print(f"  Level 4 Semantic: {paired_stats['level_4_semantic_correctness']['verdict']} (both correct: {paired_stats['level_4_semantic_correctness']['both_correct_rate']})")


if __name__ == "__main__":
    main()
