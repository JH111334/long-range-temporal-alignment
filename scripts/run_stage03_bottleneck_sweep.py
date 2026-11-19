#!/usr/bin/env python3
"""Stage 03 Representation Bottleneck Sweep & Causal Tracing Pipeline.

Executes comprehensive 5-level causality sweep:
Delivery -> Sensitivity -> Selectivity -> Semantic Correctness -> Robustness
Instruments Layer x Position axes and emits W3C OpenTelemetry traces with Fail-Open fallback.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure src in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from temporal_alignment.baselines.ticktack05b.evaluator import (
    build_p12_prompt,
    task_conditioned_strict_parser,
)
from temporal_alignment.causal_evaluation import (
    MechanismTraceResult,
    TemporalMechanismTracer,
    compute_clustered_bootstrap_ci,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 03 Bottleneck Sweep")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--max_pairs", type=int, default=15, help="Number of admitted pairs to run (default: 15, -1 for all)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default="artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_2/checkpoints/R52_B_NTP_REL_MCQA.pt")
    args = parser.parse_args()

    t_start = time.time()
    device = torch.device(args.device)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"RUN_BOTTLENECK_{timestamp_str}"

    out_dir = PROJECT_ROOT / "artifacts" / "stage03" / "representation_bottleneck" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    # Initialize OpenTelemetry Fail-Open Tracer
    tracer = get_global_tracer(
        service_name="stage03-causal-bottleneck",
        project_name="stage03-representation-bottleneck",
    )

    with tracer.span("stage03.run", {"run_id": run_id, "device": str(device)}):
        # 1. Preflight span
        with tracer.span("stage03.preflight", {"checkpoint": args.checkpoint}):
            ckpt_path = PROJECT_ROOT / args.checkpoint
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing required checkpoint: {ckpt_path}")
            ckpt_sha256 = compute_sha256(ckpt_path)

            audit_file = PROJECT_ROOT / "data" / "counterfactual" / "COUNTERFACTUAL_GOLD_AUDIT.jsonl"
            if not audit_file.exists():
                raise FileNotFoundError(f"Missing audit file: {audit_file}")

            # Read admitted pairs
            admitted_pairs = []
            with open(audit_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("admitted_primary"):
                            admitted_pairs.append(r)

            flip_pairs = [p for p in admitted_pairs if p.get("gold_flips")]
            same_pairs = [p for p in admitted_pairs if not p.get("gold_flips")]

            if args.max_pairs > 0:
                # Select balanced subset: 2/3 flip, 1/3 same
                n_flip = min(len(flip_pairs), int(args.max_pairs * (2/3)))
                n_same = min(len(same_pairs), args.max_pairs - n_flip)
                selected_pairs = flip_pairs[:n_flip] + same_pairs[:n_same]
            else:
                selected_pairs = admitted_pairs

            print(f"Preflight PASS. Selected {len(selected_pairs)} pairs ({sum(1 for p in selected_pairs if p['gold_flips'])} flip, {sum(1 for p in selected_pairs if not p['gold_flips'])} same).")

        # 2. Model initialization
        revision = "7ae557604adf67be50417f59c2c2f167def9a775"
        cache_path = Path.home() / f".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/{revision}"
        if not cache_path.exists():
            raise FileNotFoundError(f"Qwen snapshot missing at {cache_path}")

        tokenizer = AutoTokenizer.from_pretrained(str(cache_path), local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading Qwen2.5-0.5B-Instruct onto {device}...")
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

        print(f"Attaching Arm B checkpoint: {ckpt_path.name}")
        load_adapter_weights(ckpt_path, adapted_modules, gated_proj, device)

        # Mechanism Tracer
        mech_tracer = TemporalMechanismTracer(
            model=model,
            tokenizer=tokenizer,
            adapted_modules=adapted_modules,
            gated_proj=gated_proj,
            cond_context=cond_context,
            device=device,
            seam_token_index=14,
        )

        # 3. Sweep execution across pairs and conditions
        raw_predictions_path = out_dir / "RAW_PREDICTIONS.jsonl"
        mechanism_trace_path = out_dir / "MECHANISM_TRACE.jsonl"

        raw_pred_file = open(raw_predictions_path, "w", encoding="utf-8")
        mech_trace_file = open(mechanism_trace_path, "w", encoding="utf-8")

        all_trace_results: List[MechanismTraceResult] = []
        condition_results: Dict[str, List[Dict[str, Any]]] = {
            "B_FROZEN_FULL": [],
            "COUNTERFACTUAL_T2": [],
            "RANDOM_NORM_MATCHED": [],
            "A0_BASE": [],
        }

        print(f"Starting Bottleneck Tracing on {len(selected_pairs)} pairs...")

        for idx, p_info in enumerate(selected_pairs, 1):
            pair_id = p_info["pair_id"]
            gold_flips = p_info["gold_flips"]
            t1 = p_info["t1_year"]
            t2 = p_info["t2_year"]
            g1 = p_info["t1_gold"]
            g2 = p_info["t2_gold"]
            entity = p_info["entity"]
            relation = p_info["relation"]

            prompt_text = f"In {t1}, {entity} was a {relation}? Answer as Yes or No"

            with tracer.span("stage03.counterfactual_pair", {"pair_id": pair_id, "gold_flips": gold_flips}):
                trace_id, span_id = tracer.get_current_ids()

                # Detailed mechanism tracing between t1 and t2
                with tracer.span("temporal.mechanism_trace"):
                    trace_res = mech_tracer.trace_pair(
                        case_id=pair_id,
                        prompt_text=prompt_text,
                        t1=t1,
                        t2=t2,
                        gold_t1=g1,
                        gold_t2=g2,
                        candidate_labels=["Yes", "No"],
                    )
                    all_trace_results.append(trace_res)

                    trace_dict = trace_res.to_dict()
                    trace_dict["trace_id"] = trace_id
                    trace_dict["span_id"] = span_id
                    trace_dict["cluster_id"] = p_info["temporal_cluster_id"]
                    trace_dict["gold_flips"] = gold_flips
                    mech_trace_file.write(json.dumps(trace_dict, ensure_ascii=False) + "\n")

                # Condition Arm: B_FROZEN_FULL at (t1, t1)
                pred_t1_corr = (trace_res.prediction_t1.strip().lower() == g1.strip().lower())
                rec_b = {
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "pair_id": pair_id,
                    "example_id": f"{pair_id}_t1",
                    "condition_arm": "B_FROZEN_FULL",
                    "year_text": t1,
                    "year_cond": t1,
                    "gold": g1,
                    "prediction": trace_res.prediction_t1,
                    "is_correct": pred_t1_corr,
                    "cluster_id": p_info["temporal_cluster_id"],
                }
                condition_results["B_FROZEN_FULL"].append(rec_b)
                raw_pred_file.write(json.dumps(rec_b, ensure_ascii=False) + "\n")

                # Condition Arm: COUNTERFACTUAL_T2 at (t1, t2)
                pred_t2_matches_g2 = (trace_res.prediction_t2.strip().lower() == g2.strip().lower())
                rec_c = {
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "pair_id": pair_id,
                    "example_id": f"{pair_id}_t1_cond_t2",
                    "condition_arm": "COUNTERFACTUAL_T2",
                    "year_text": t1,
                    "year_cond": t2,
                    "gold": g2 if gold_flips else g1,
                    "prediction": trace_res.prediction_t2,
                    "is_correct": pred_t2_matches_g2,
                    "cluster_id": p_info["temporal_cluster_id"],
                    "is_flip": trace_res.is_flip,
                    "is_correct_flip": trace_res.is_correct_flip,
                }
                condition_results["COUNTERFACTUAL_T2"].append(rec_c)
                raw_pred_file.write(json.dumps(rec_c, ensure_ascii=False) + "\n")

                # Condition Arm: RANDOM_NORM_MATCHED
                rand_delta = trace_res.controls.get("matched_norm_random_delta", 0.0)
                rec_rand = {
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "pair_id": pair_id,
                    "condition_arm": "RANDOM_NORM_MATCHED",
                    "random_perturb_delta": rand_delta,
                    "temporal_delta": trace_res.mean_abs_logit_delta,
                    "selectivity_margin": round(trace_res.mean_abs_logit_delta - rand_delta, 6),
                    "cluster_id": p_info["temporal_cluster_id"],
                }
                condition_results["RANDOM_NORM_MATCHED"].append(rec_rand)
                raw_pred_file.write(json.dumps(rec_rand, ensure_ascii=False) + "\n")

                # Condition Arm: A0_BASE (No adapter forward)
                with tracer.span("model.base_forward"):
                    formatted_p, is_yn = build_p12_prompt(prompt_text, tokenizer)
                    b_inputs = tokenizer(formatted_p, return_tensors="pt").to(device)
                    p_len = b_inputs["input_ids"].shape[1]
                    with configure_temporal_condition("C_BOTH_OFF", adapted_modules, gated_proj, cond_context, [], (), (), device, seam_token_index=14):
                        with torch.no_grad():
                            b_gen = model.generate(**b_inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                    raw_b_pred = tokenizer.decode(b_gen[0][p_len:], skip_special_tokens=True).strip()
                    parsed_b, _ = task_conditioned_strict_parser(raw_b_pred, is_yn)
                    b_corr = (parsed_b.strip().lower() == g1.strip().lower())
                    rec_base = {
                        "run_id": run_id,
                        "trace_id": trace_id,
                        "span_id": span_id,
                        "pair_id": pair_id,
                        "example_id": f"{pair_id}_t1",
                        "condition_arm": "A0_BASE",
                        "year_text": t1,
                        "year_cond": None,
                        "gold": g1,
                        "prediction": parsed_b or "INVALID",
                        "is_correct": b_corr,
                        "cluster_id": p_info["temporal_cluster_id"],
                    }
                    condition_results["A0_BASE"].append(rec_base)
                    raw_pred_file.write(json.dumps(rec_base, ensure_ascii=False) + "\n")

            if idx % 5 == 0 or idx == len(selected_pairs):
                print(f"Progress: [{idx}/{len(selected_pairs)}] pairs processed.")

        raw_pred_file.close()
        mech_trace_file.close()

        # 4. Statistical Aggregation
        with tracer.span("stage03.aggregate"):
            total_n = len(selected_pairs)
            flip_n = sum(1 for p in selected_pairs if p["gold_flips"])
            same_n = sum(1 for p in selected_pairs if not p["gold_flips"])

            # Level 1: Delivery
            mean_dz = sum(r.d_z for r in all_trace_results) / total_n
            mean_df = sum(r.d_f_mean for r in all_trace_results) / total_n
            mean_dw = sum(r.d_w_mean for r in all_trace_results) / total_n

            # Level 2: Sensitivity
            mean_abs_logit_delta = sum(r.mean_abs_logit_delta for r in all_trace_results) / total_n
            
            # Layer x Position average sensitivities
            num_layers = len(all_trace_results[0].layerwise_sensitivity) if all_trace_results else 24
            avg_s_comp = [round(sum(r.layerwise_sensitivity[l] for r in all_trace_results) / total_n, 6) for l in range(num_layers)]
            avg_s_seam = [round(sum(r.sensitivity_seam_pos[l] for r in all_trace_results) / total_n, 6) for l in range(num_layers)]

            # Level 3: Selectivity
            mean_rand_delta = sum(r.controls.get("matched_norm_random_delta", 0.0) for r in all_trace_results) / total_n
            mean_selectivity_margin = mean_abs_logit_delta - mean_rand_delta
            mean_repeat_stability = sum(r.controls.get("same_year_repeated_delta", 0.0) for r in all_trace_results) / total_n

            # Level 4: Semantic Correctness
            flip_subset_traces = [r for r, p in zip(all_trace_results, selected_pairs) if p["gold_flips"]]
            same_subset_traces = [r for r, p in zip(all_trace_results, selected_pairs) if not p["gold_flips"]]

            flip_count_on_flip = sum(1 for r in flip_subset_traces if r.is_flip)
            corr_flip_count = sum(1 for r in flip_subset_traces if r.is_correct_flip)
            both_corr_count = sum(1 for r in flip_subset_traces if (r.prediction_t1.strip().lower() == r.gold_t1.strip().lower() and r.prediction_t2.strip().lower() == r.gold_t2.strip().lower()))

            flip_rate_on_flip = flip_count_on_flip / max(flip_n, 1)
            correct_flip_rate = corr_flip_count / max(flip_n, 1)
            both_correct_rate = both_corr_count / max(flip_n, 1)

            same_flip_count = sum(1 for r in same_subset_traces if r.is_flip)
            same_stability_rate = (same_n - same_flip_count) / max(same_n, 1)

            # Level 5: Robustness & Statistical McNemar
            b_corrects = [r["is_correct"] for r in condition_results["B_FROZEN_FULL"]]
            base_corrects = [r["is_correct"] for r in condition_results["A0_BASE"]]

            mcnemar_res = compute_clustered_mcnemar(
                condition_results["A0_BASE"],
                condition_results["B_FROZEN_FULL"],
                cluster_field="cluster_id",
            )

            scoreboard = {
                "run_id": run_id,
                "timestamp": timestamp_str,
                "sample_budget": {
                    "total_pairs": total_n,
                    "flip_pairs": flip_n,
                    "same_pairs": same_n,
                },
                "level_1_delivery": {
                    "mean_d_z": round(mean_dz, 6),
                    "mean_d_f": round(mean_df, 6),
                    "mean_d_w": round(mean_dw, 6),
                    "delivery_verdict": "PASS" if mean_dw > 1e-5 else "FAIL",
                },
                "level_2_sensitivity": {
                    "mean_abs_logit_delta": round(mean_abs_logit_delta, 6),
                    "layer14_sensitivity_at_seam_pos": avg_s_seam[14] if len(avg_s_seam) > 14 else 0.0,
                    "layer14_sensitivity_at_comp_pos": avg_s_comp[14] if len(avg_s_comp) > 14 else 0.0,
                    "max_sensitivity_layer_comp": int(max(range(len(avg_s_comp)), key=lambda l: avg_s_comp[l])),
                    "sensitivity_verdict": "PASS" if mean_abs_logit_delta > 0.01 else "FAIL",
                },
                "level_3_selectivity": {
                    "temporal_logit_delta": round(mean_abs_logit_delta, 6),
                    "matched_random_logit_delta": round(mean_rand_delta, 6),
                    "selectivity_margin": round(mean_selectivity_margin, 6),
                    "same_year_repeat_stability": round(mean_repeat_stability, 8),
                    "selectivity_verdict": "PASS" if mean_selectivity_margin > 0.02 else "FAIL_NON_SELECTIVE",
                },
                "level_4_semantic_correctness": {
                    "flip_pairs_count": flip_n,
                    "observed_flips_on_flip_gold": flip_count_on_flip,
                    "flip_rate_on_flip_gold": round(flip_rate_on_flip, 4),
                    "correct_flip_rate": round(correct_flip_rate, 4),
                    "pair_both_correct_rate": round(both_correct_rate, 4),
                    "same_pairs_count": same_n,
                    "same_stability_rate": round(same_stability_rate, 4),
                    "semantic_verdict": "PASS" if both_correct_rate >= 0.30 else "FAIL_NO_CAUSAL_FLIP",
                },
                "level_5_robustness_and_stats": {
                    "base_accuracy": round(sum(base_corrects) / total_n, 4),
                    "arm_b_accuracy": round(sum(b_corrects) / total_n, 4),
                    "mcnemar_table": mcnemar_res.contingency_table,
                    "mcnemar_p_value": round(mcnemar_res.p_value_standard, 6),
                    "clustered_p_value": round(mcnemar_res.p_value_clustered, 6),
                    "intraclass_correlation_icc": round(mcnemar_res.intraclass_correlation_icc, 6),
                    "bootstrap_95_ci": [round(mcnemar_res.bootstrap_ci_95[0], 4), round(mcnemar_res.bootstrap_ci_95[1], 4)],
                },
                "overall_causal_gate": "NO_GO_SCALE",
            }

            # If all 4 levels pass, evaluate promotion
            if (
                scoreboard["level_1_delivery"]["delivery_verdict"] == "PASS"
                and scoreboard["level_2_sensitivity"]["sensitivity_verdict"] == "PASS"
                and scoreboard["level_3_selectivity"]["selectivity_verdict"] == "PASS"
                and scoreboard["level_4_semantic_correctness"]["semantic_verdict"] == "PASS"
            ):
                scoreboard["overall_causal_gate"] = "CONDITIONAL_GO_SCALE"

            scoreboard_path = out_dir / "SCOREBOARD.json"
            with open(scoreboard_path, "w", encoding="utf-8") as f:
                json.dump(scoreboard, f, indent=2, ensure_ascii=False)

        # 5. Writeback and Trace Export
        with tracer.span("stage03.writeback"):
            trace_index = tracer.export_local_traces(traces_dir)

            run_receipt = {
                "run_id": run_id,
                "timestamp": timestamp_str,
                "elapsed_seconds": round(time.time() - t_start, 2),
                "device": str(device),
                "checkpoint": {
                    "path": str(ckpt_path),
                    "sha256": ckpt_sha256,
                },
                "scoreboard": scoreboard,
                "trace_summary": trace_index,
                "output_directory": str(out_dir),
            }

            with open(out_dir / "RUN_RECEIPT.json", "w", encoding="utf-8") as f:
                json.dump(run_receipt, f, indent=2, ensure_ascii=False)

            # Copy receipt to central runs dir
            central_runs_dir = PROJECT_ROOT / "Reports" / "03_TARGETED_OBJECTIVES_MCQA" / "REPRESENTATION_BOTTLENECK" / "runs"
            with open(central_runs_dir / f"{run_id}_RECEIPT.json", "w", encoding="utf-8") as f:
                json.dump(run_receipt, f, indent=2, ensure_ascii=False)

    print(f"\nBottleneck Sweep completed in {round(time.time() - t_start, 2)}s.")
    print(f"Results archived at: {out_dir}")
    print(f"Gate Verdict: {scoreboard['overall_causal_gate']}")
    print(f"  Level 1 Delivery: {scoreboard['level_1_delivery']['delivery_verdict']}")
    print(f"  Level 2 Sensitivity: {scoreboard['level_2_sensitivity']['sensitivity_verdict']}")
    print(f"  Level 3 Selectivity: {scoreboard['level_3_selectivity']['selectivity_verdict']} (margin: {scoreboard['level_3_selectivity']['selectivity_margin']})")
    print(f"  Level 4 Semantic: {scoreboard['level_4_semantic_correctness']['semantic_verdict']} (both correct: {scoreboard['level_4_semantic_correctness']['pair_both_correct_rate']})")


if __name__ == "__main__":
    main()
