"""CLI for Stage 03 Canonical Deterministic Evaluator.

Usage examples:
    # Run Base model A0 on TempUN 180:
    py -3.12 scripts/evaluate_stage03_canonical.py --dataset data/tempun/eval_mcqa.jsonl --condition A0_BASE --output_dir artifacts/evaluation/canonical/A0_BASE

    # Run Arm B Checkpoint C_TRUE on TempUN 180:
    py -3.12 scripts/evaluate_stage03_canonical.py --dataset data/tempun/eval_mcqa.jsonl --checkpoint artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_2/checkpoints/R52_B_NTP_REL_MCQA.pt --condition C_TRUE --output_dir artifacts/evaluation/canonical/R52_B_C_TRUE

    # Run Arm B Checkpoint C_WRONG_YEAR on TempUN 180:
    py -3.12 scripts/evaluate_stage03_canonical.py --dataset data/tempun/eval_mcqa.jsonl --checkpoint artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_2/checkpoints/R52_B_NTP_REL_MCQA.pt --condition C_WRONG_YEAR --output_dir artifacts/evaluation/canonical/R52_B_C_WRONG_YEAR
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

from temporal_alignment.causal_evaluation.canonical_evaluator import (
    CanonicalEvaluationConfig,
    compute_file_sha256,
    evaluate_dataset_canonical,
)
from temporal_alignment.models.dual_cycle_mat_lora import replace_mlp_with_dual_cycle_mat_lora
from temporal_alignment.models.gated_seam import (
    ConditioningContext,
    GatedResidualProjection,
    make_gated_embedding_hook,
)


def load_adapter_weights_into_model(
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
    parser = argparse.ArgumentParser(description="Canonical Deterministic Evaluator")
    parser.add_argument("--dataset", type=str, default="data/tempun/eval_mcqa.jsonl", help="Path to input jsonl")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to adapter checkpoint (.pt)")
    parser.add_argument("--condition", type=str, default="A0_BASE", help="Experimental condition name")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store evaluation outputs")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples evaluated")
    parser.add_argument("--run_id", type=str, default=None, help="Unique run identifier")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t_start = time.time()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"EVAL_{args.condition}_{timestamp_str}"
    
    print(f"[{datetime.now().isoformat()}] Starting Canonical Evaluation: {run_id}")
    print(f"Condition: {args.condition} | Device: {device} | Output: {out_dir}")
    
    # Load dataset
    data_path = Path(args.dataset)
    if not data_path.is_absolute():
        data_path = _REPO_ROOT / data_path
        
    records: List[Dict[str, Any]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    if args.max_samples is not None:
        records = records[:args.max_samples]
    print(f"Loaded {len(records)} evaluation records from {data_path.name}")
    
    # Load model and tokenizer
    revision = "7ae557604adf67be50417f59c2c2f167def9a775"
    cache_path = Path.home() / f".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/{revision}"
    
    print("Loading base model Qwen2.5-0.5B-Instruct...")
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
    
    adapted_modules = None
    gated_proj = None
    cond_context = None
    
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = _REPO_ROOT / ckpt_path
        print(f"Injecting dual-cycle MaT-LoRA adapters and loading checkpoint from {ckpt_path.name}...")
        adapted_modules = replace_mlp_with_dual_cycle_mat_lora(model, rank=8, alpha=16.0, num_harmonics=2)
        gated_proj = GatedResidualProjection(in_features=40, out_features=896).to(device)
        cond_context = ConditioningContext(injection_token_index=14)
        hook = make_gated_embedding_hook(cond_context)
        model.get_input_embeddings().register_forward_hook(hook)
        
        load_adapter_weights_into_model(ckpt_path, adapted_modules, gated_proj, device)
        
    config = CanonicalEvaluationConfig(
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        condition=args.condition,
        device=args.device,
        seed=args.seed,
    )
    
    print(f"Executing deterministic generation over {len(records)} examples...")
    raw_records, scoreboard = evaluate_dataset_canonical(
        model=model,
        tokenizer=tokenizer,
        records=records,
        config=config,
        run_id=run_id,
        adapted_modules=adapted_modules,
        gated_proj=gated_proj,
        cond_context=cond_context,
    )
    
    # Save RAW_PREDICTIONS.jsonl
    raw_path = out_dir / "RAW_PREDICTIONS.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in raw_records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            
    # Save SCOREBOARD.json
    sb_path = out_dir / "SCOREBOARD.json"
    with open(sb_path, "w", encoding="utf-8") as f:
        json.dump(scoreboard, f, indent=2, ensure_ascii=False)
        
    # Save EVALUATION_CONTRACT_RECEIPT.json
    receipt = {
        "run_id": run_id,
        "execution_timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(time.time() - t_start, 2),
        "dataset": {
            "path": str(data_path),
            "sha256": compute_file_sha256(data_path),
            "total_records": len(records),
        },
        "model": {
            "name": config.model_name,
            "revision": config.model_revision,
            "checkpoint_path": str(args.checkpoint) if args.checkpoint else None,
            "checkpoint_sha256": compute_file_sha256(args.checkpoint) if args.checkpoint else "BASE_PRETRAINED",
            "condition": args.condition,
            "generation_config_sha": config.get_generation_config_sha(),
        },
        "scoreboard_summary": scoreboard["summary"],
        "hygiene_gate": scoreboard["hygiene_gate"],
    }
    with open(out_dir / "EVALUATION_CONTRACT_RECEIPT.json", "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)
        
    print("\n--- EVALUATION COMPLETE ---")
    print(f"Total: {scoreboard['summary']['total_count']} | Strict Acc: {scoreboard['summary']['strict_accuracy']:.4f}")
    print(f"Invalid Rate: {scoreboard['summary']['invalid_rate']:.4f} | Prediction Entropy: {scoreboard['summary']['prediction_entropy']:.4f}")
    print(f"Hygiene Gate Passed: {scoreboard['hygiene_gate']['hygiene_passed']} (Max label freq: {scoreboard['hygiene_gate']['max_label_frequency']:.4f})")
    print(f"Outputs written to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
