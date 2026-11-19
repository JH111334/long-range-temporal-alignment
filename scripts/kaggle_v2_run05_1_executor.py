"""Standalone Kaggle GPU Executor for V2 Run 05.1 (Forensic Phoenix Recovery).

Runs on Kaggle GPU (T4 / P100 / L4):
1. Prepares clean mock repo filesystem mapping Kaggle inputs.
2. Calls canonical train_run05_1_arms on CUDA.
3. Calls canonical run_full_evaluation on CUDA.
4. Emits all output artifacts into output_dir.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Setup paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import torch
from temporal_alignment.training_run05_1 import train_run05_1_arms
from evaluate_v2_run05_1_all_arms import run_full_evaluation


def main():
    parser = argparse.ArgumentParser(description="V2 Run 05.1 Kaggle GPU Executor")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing clean input dataset")
    parser.add_argument("--extra_data_dir", type=str, required=True, help="Directory containing train_nwp.jsonl and canary")
    parser.add_argument("--c1_checkpoint", type=str, required=True, help="Path to C1 compact checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save artifacts")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    extra_dir = Path(args.extra_data_dir)
    c1_path = Path(args.c1_checkpoint)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting V2 Run 05.1 Executor on {device_name} ===", flush=True)

    # 1. Setup mock repo layout in /kaggle/working/repo_root
    mock_root = Path("/kaggle/working/repo_root")
    if mock_root.exists():
        shutil.rmtree(mock_root, ignore_errors=True)
    mock_root.mkdir(parents=True, exist_ok=True)

    # data/tempun/
    d_tempun = mock_root / "data" / "tempun"
    d_tempun.mkdir(parents=True, exist_ok=True)
    shutil.copy(data_dir / "eval_mcqa.jsonl", d_tempun / "eval_mcqa.jsonl")
    shutil.copy(extra_dir / "train_nwp.jsonl", d_tempun / "train_nwp.jsonl")

    # artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_04/
    d_run04 = mock_root / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_04"
    d_run04.mkdir(parents=True, exist_ok=True)
    shutil.copy(data_dir / "MCQA4_TRAIN_CLEAN.jsonl", d_run04 / "MCQA4_TRAIN_CLEAN.jsonl")
    shutil.copy(data_dir / "MCQA4_DEV_CLEAN.jsonl", d_run04 / "MCQA4_DEV_CLEAN.jsonl")

    # artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_02/
    d_run02 = mock_root / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_02"
    d_run02.mkdir(parents=True, exist_ok=True)
    shutil.copy(data_dir / "train_relation_pairs.jsonl", d_run02 / "train_relation_pairs.jsonl")
    shutil.copy(data_dir / "dev_relation_pairs.jsonl", d_run02 / "dev_relation_pairs.jsonl")

    # artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_02/checkpoints/
    d_run02_ckpt = d_run02 / "checkpoints"
    d_run02_ckpt.mkdir(parents=True, exist_ok=True)
    shutil.copy(c1_path, d_run02_ckpt / "C1_COMPACT_ADAPTER_ONLY.pt")

    # artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_1/
    d_run051 = mock_root / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_1"
    d_run051.mkdir(parents=True, exist_ok=True)
    shutil.copy(extra_dir / "CANARY_TRUTH_AND_SPLIT.jsonl", d_run051 / "CANARY_TRUTH_AND_SPLIT.jsonl")

    print("Mock repo layout constructed successfully.", flush=True)

    # 2. Canonical Training on GPU
    print("\n--- Starting Canonical Training on CUDA ---", flush=True)
    train_res = train_run05_1_arms(repo_root=mock_root, steps=40, device_name=device_name)
    print("Training Completed! Divergence:", train_res.get("divergence_verification"), flush=True)

    # 3. Canonical Evaluation on GPU
    print("\n--- Starting Canonical Evaluation on CUDA ---", flush=True)
    eval_scoreboard = run_full_evaluation(device_name=device_name, repo_root=mock_root)
    print("Evaluation Completed successfully!", flush=True)

    # 4. Copy all artifacts to destination output_dir
    print(f"\nEmitting final artifacts to {out_dir}...", flush=True)
    shutil.copytree(d_run051, out_dir, dirs_exist_ok=True)

    print("=== V2 Run 05.1 Execution Finished Cleanly on Kaggle GPU! ===", flush=True)


if __name__ == "__main__":
    main()
