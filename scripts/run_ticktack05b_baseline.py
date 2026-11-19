"""Command-line interface to train and evaluate the TickTack-inspired 0.5B baseline.

Usage:
    python scripts/run_ticktack05b_baseline.py --help
    python scripts/run_ticktack05b_baseline.py --canary --seed 1
    python scripts/run_ticktack05b_baseline.py --seeds 1 2 3 --output artifacts/ticktack05b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

_PKG_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from temporal_alignment.baselines.ticktack05b.dataset import (
    TempUNTrainDataset,
    collate_ticktack_batch,
)
from temporal_alignment.baselines.ticktack05b.evaluator import evaluate_single_example
from temporal_alignment.baselines.ticktack05b.loss import LoRAEWCLoss
from temporal_alignment.baselines.ticktack05b.model import TickTackQwen05BModel
from temporal_alignment.baselines.ticktack05b.train_runner import (
    STAGES,
    compute_stage_fisher_matrix,
    evaluate_stage_on_180,
    run_stage_training,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TickTack-inspired 0.5B baseline runner.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", type=str, default="7ae557604adf67be50417f59c2c2f167def9a775")
    parser.add_argument("--data-train", type=str, default="scratch/test_bundle/lrr_bundle/data/train_nwp.jsonl")
    parser.add_argument("--data-eval", type=str, default="data/tempun/eval_mcqa.jsonl")
    parser.add_argument("--output", type=str, default="artifacts/ticktack05b")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--canary", action="store_true", help="Run quick single-seed canary check")
    parser.add_argument("--max-steps-per-stage", type=int, default=None, help="Cap steps for quick testing")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--radius-mode", type=str, default="adapted_monotonic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing TickTack-inspired 0.5B baseline on device: {device}")
    print(f"Backbone: {args.model_name} (revision: {args.revision})")
    print(f"Radius mode: {args.radius_mode}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load eval records (180 cases)
    eval_records = []
    with open(args.data_eval, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_records.append(json.loads(line))
    print(f"Loaded {len(eval_records)} evaluation records from {args.data_eval}")

    active_seeds = [args.seeds[0]] if args.canary else args.seeds
    max_steps = 15 if args.canary else args.max_steps_per_stage

    summary_metrics: Dict[str, Any] = {
        "model": args.model_name,
        "revision": args.revision,
        "radius_mode": args.radius_mode,
        "seeds": active_seeds,
        "is_canary": args.canary,
        "results_per_seed": {},
    }

    for seed in active_seeds:
        print(f"\n==========================================")
        print(f"  STARTING TICKTACK-0.5B FOR SEED {seed}")
        print(f"==========================================")
        seed_everything(seed)

        # Fresh initialization for each seed
        model = TickTackQwen05BModel(
            base_model_name_or_path=args.model_name,
            revision=args.revision,
            radius_mode=args.radius_mode,
            torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
        ).to(device)

        # Trainable optimizer (AdamW on LoRA parameters only)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
        ewc_module = LoRAEWCLoss(lambda_ewc=100.0)

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        seed_stages_out: Dict[str, Any] = {}

        for stage_idx, stage_id in enumerate(STAGES):
            print(f"\n--- Seed {seed} | Stage {stage_idx}: {stage_id} ---")
            stage_suffix = f"after_{stage_idx:02d}"

            # Load stage dataset
            stage_ds = TempUNTrainDataset.from_file(
                args.data_train,
                tokenizer=tokenizer,
                stage_id=stage_id,
            )
            print(f"Loaded {len(stage_ds)} samples for {stage_id}")

            # Run training
            t_start = time.time()
            train_res = run_stage_training(
                model=model,
                tokenizer=tokenizer,
                dataset=stage_ds,
                stage_id=stage_id,
                device=device,
                optimizer=optimizer,
                ewc_module=ewc_module,
                batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum,
                max_steps=max_steps,
            )
            elapsed = time.time() - t_start
            train_res["elapsed_seconds"] = round(elapsed, 2)
            print(f"Completed {train_res['steps_completed']} steps in {elapsed:.1f}s | Avg Loss: {train_res['avg_total_loss']:.4f}")

            # Update Fisher matrix for EWC in subsequent stages
            if stage_idx < len(STAGES) - 1:
                print("Estimating diagonal Fisher Information on LoRA parameters...")
                stage_loader = torch.utils.data.DataLoader(
                    stage_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=collate_ticktack_batch,
                )
                fisher_dict = compute_stage_fisher_matrix(model, stage_loader, device=device, max_batches=15)
                ewc_module.update_task_fisher(model, fisher_dict)
                print(f"Fisher updated for {len(fisher_dict)} parameter tensors.")

            # Save stage checkpoint
            ckpt_dir = seed_dir / stage_suffix
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            lora_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}
            ckpt_path = ckpt_dir / "lora_weights.pt"
            torch.save(lora_state, ckpt_path)

            # Evaluate on full 180 TempUN cases
            print(f"Evaluating {stage_suffix} on 180 TempUN cases (P12 Task-Aware Free Gen)...")
            eval_res = evaluate_stage_on_180(model, tokenizer, eval_records)
            print(f"[{stage_suffix}] Strict Acc: {eval_res['strict_accuracy']}% | MCQA: {eval_res['mcqa_accuracy']}% | YN: {eval_res['yn_accuracy']}% | Invalid: {eval_res['invalid_rate']}%")

            # Save stage evaluation artifact
            eval_save_path = ckpt_dir / "eval_metrics.json"
            pred_save_path = ckpt_dir / "predictions.jsonl"

            # Write records to jsonl
            with open(pred_save_path, "w", encoding="utf-8") as f:
                for r in eval_res["records"]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            metrics_summary = {
                "stage_id": stage_id,
                "stage_suffix": stage_suffix,
                "train_metrics": train_res,
                "strict_accuracy": eval_res["strict_accuracy"],
                "mcqa_accuracy": eval_res["mcqa_accuracy"],
                "yn_accuracy": eval_res["yn_accuracy"],
                "invalid_rate": eval_res["invalid_rate"],
                "correct_count": eval_res["correct_count"],
                "invalid_count": eval_res["invalid_count"],
            }
            with open(eval_save_path, "w", encoding="utf-8") as f:
                json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

            seed_stages_out[stage_suffix] = metrics_summary

        summary_metrics["results_per_seed"][f"seed_{seed}"] = seed_stages_out

    # Save overall run summary
    with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nAll operations finished. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
