"""CLI script for tracing causal temporal mechanism across counterfactual pairs.

Usage:
    py -3.12 scripts/trace_temporal_mechanism.py \
        --pairs data/counterfactual/grounded_gold_flip_pairs.jsonl \
        --checkpoint artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_2/checkpoints/R52_B_NTP_REL_MCQA.pt \
        --output_dir artifacts/evaluation/mechanism_traces \
        --max_pairs 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from temporal_alignment.causal_evaluation.mechanism_tracer import (
    MechanismTraceResult,
    TemporalMechanismTracer,
)
from temporal_alignment.models.dual_cycle_mat_lora import replace_mlp_with_dual_cycle_mat_lora
from temporal_alignment.models.gated_seam import (
    ConditioningContext,
    GatedResidualProjection,
    make_gated_embedding_hook,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace Temporal Mechanism")
    parser.add_argument("--pairs", type=str, default="data/counterfactual/grounded_gold_flip_pairs.jsonl")
    parser.add_argument("--checkpoint", type=str, default="artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_2/checkpoints/R52_B_NTP_REL_MCQA.pt")
    parser.add_argument("--output_dir", type=str, default="artifacts/evaluation/mechanism_traces")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_pairs", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t_start = time.time()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now().isoformat()}] Starting Temporal Mechanism Tracing...")
    print(f"Device: {device} | Checkpoint: {args.checkpoint}")

    # Load pairs data: each pair is represented by t1 and t2 in records
    pairs_file = Path(args.pairs)
    if not pairs_file.is_absolute():
        pairs_file = _REPO_ROOT / pairs_file

    records: List[Dict[str, Any]] = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Group into pairs by pair_id
    grouped_pairs: Dict[str, Dict[str, Any]] = {}
    for r in records:
        pid = r.get("pair_id")
        if not pid:
            continue
        if pid not in grouped_pairs:
            grouped_pairs[pid] = {
                "pair_id": pid,
                "entity": r.get("entity"),
                "relation": r.get("relation"),
                "prompt_template": r.get("input"),
            }
        role = r.get("role")
        if role == "anchor_t1":
            grouped_pairs[pid]["t1"] = r.get("year")
            grouped_pairs[pid]["gold_t1"] = r.get("target")
            grouped_pairs[pid]["prompt_t1"] = r.get("input")
        elif role == "anchor_t2":
            grouped_pairs[pid]["t2"] = r.get("year")
            grouped_pairs[pid]["gold_t2"] = r.get("target")
            grouped_pairs[pid]["prompt_t2"] = r.get("input")

    pair_list = list(grouped_pairs.values())
    if args.max_pairs is not None:
        pair_list = pair_list[:args.max_pairs]
    print(f"Loaded and grouped {len(pair_list)} counterfactual pairs.")

    # Load model and tokenizer
    revision = "7ae557604adf67be50417f59c2c2f167def9a775"
    cache_path = Path.home() / f".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/{revision}"
    
    tokenizer = AutoTokenizer.from_pretrained(str(cache_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = _REPO_ROOT / ckpt_path
    load_adapter_weights(ckpt_path, adapted_modules, gated_proj, device)

    tracer = TemporalMechanismTracer(
        model=model,
        tokenizer=tokenizer,
        adapted_modules=adapted_modules,
        gated_proj=gated_proj,
        cond_context=cond_context,
        device=device,
        seam_token_index=14,
    )

    results: List[MechanismTraceResult] = []
    for idx, p in enumerate(pair_list):
        pid = p["pair_id"]
        t1 = p["t1"]
        t2 = p["t2"]
        g1 = p["gold_t1"]
        g2 = p["gold_t2"]
        # Use template or prompt_t1
        prompt = p.get("prompt_t1") or p.get("prompt_template")
        print(f"[{idx+1}/{len(pair_list)}] Tracing {pid}: t1={t1} ({g1}) vs t2={t2} ({g2})...")
        res = tracer.trace_pair(
            case_id=pid,
            prompt_text=prompt,
            t1=t1,
            t2=t2,
            gold_t1=g1,
            gold_t2=g2,
        )
        results.append(res)

    # Save detailed traces
    trace_path = out_dir / "MECHANISM_TRACES.jsonl"
    with open(trace_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    # Compute summary
    n = len(results)
    avg_dz = sum(r.d_z for r in results) / n if n else 0.0
    avg_df = sum(r.d_f_mean for r in results) / n if n else 0.0
    avg_dw = sum(r.d_w_mean for r in results) / n if n else 0.0
    avg_logit_delta = sum(r.mean_abs_logit_delta for r in results) / n if n else 0.0
    avg_delta_m = sum(r.delta_gold_margin_t2 for r in results) / n if n else 0.0
    total_flips = sum(1 for r in results if r.is_flip)
    correct_flips = sum(1 for r in results if r.is_correct_flip)

    # Layerwise average sensitivity
    num_layers = len(results[0].layerwise_sensitivity) if results else 0
    layerwise_avg_s = [
        round(sum(r.layerwise_sensitivity[l] for r in results) / n, 6)
        for l in range(num_layers)
    ]
    layerwise_avg_cos = [
        round(sum(r.layerwise_cosine_sim[l] for r in results) / n, 6)
        for l in range(num_layers)
    ]

    summary = {
        "execution_metadata": {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(time.time() - t_start, 2),
            "device": str(device),
            "checkpoint": str(ckpt_path),
            "total_pairs_traced": n,
        },
        "aggregate_mechanism_metrics": {
            "mean_d_z": round(avg_dz, 6),
            "mean_d_f": round(avg_df, 6),
            "mean_d_w": round(avg_dw, 6),
            "mean_abs_logit_delta": round(avg_logit_delta, 6),
            "mean_gold_margin_shift_delta_m": round(avg_delta_m, 6),
            "prediction_flip_count": total_flips,
            "prediction_flip_rate": round(total_flips / n, 4) if n else 0.0,
            "correct_flip_count": correct_flips,
            "correct_flip_rate": round(correct_flips / n, 4) if n else 0.0,
        },
        "layerwise_profiles": {
            "layerwise_sensitivity_s_l": layerwise_avg_s,
            "layerwise_cosine_similarity": layerwise_avg_cos,
        },
        "mechanism_verdict": {
            "temporal_input_change_confirmed": avg_dz > 0.0,
            "dynamic_core_f_t_change_confirmed": avg_df > 0.0,
            "logit_sensitivity_confirmed": avg_logit_delta > 0.01,
            "gold_margin_improves": avg_delta_m > 0.0,
            "causal_flip_achieved": correct_flips > 0,
        },
    }

    sum_path = out_dir / "MECHANISM_SUMMARY.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n--- MECHANISM TRACING COMPLETE ---")
    print(f"Mean D_z: {avg_dz:.4f} | Mean D_F: {avg_df:.6f} | Mean D_W: {avg_dw:.6f}")
    print(f"Mean Logit Delta: {avg_logit_delta:.6f} | Mean Delta M: {avg_delta_m:.6f}")
    print(f"Prediction Flips: {total_flips}/{n} ({total_flips/n:.2%}) | Correct Flips: {correct_flips}/{n} ({correct_flips/n:.2%})")
    print(f"Outputs written to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
