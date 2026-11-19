#!/usr/bin/env python3
"""
Backward-compatible shim for Ticktack Baseline generator and utilities.

All core modules have been refactored and organized under `baselines.ticktack`:
- Models & LoRA: `baselines.ticktack.models`
- Contrastive losses: `baselines.ticktack.losses`
- Date-token masking & injection: `baselines.ticktack.date_mask`
- Debiasing & evaluation: `baselines.ticktack.eval_debiasing`
- Kaggle Notebook builder: `baselines.ticktack.build_kaggle_notebook`

This forwarder preserves 100% backward compatibility for existing team scripts,
notebooks, and unit tests.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines.ticktack.build_kaggle_notebook import _find_clean_path
from baselines.ticktack import (
    CONTROL_CONFIGS,
    CLEAN_EVAL_PATH,
    CLEAN_TRAIN_PATH,
    METADATA_PATH,
    NOTEBOOK_PATH,
    OUTPUT_DIR,
    LoRALinear,
    apply_lora_to_model,
    compute_batch_te,
    compute_canchi_cycle,
    compute_intra_loss,
    compute_te_vector,
    compute_total_loss,
    create_kaggle_notebook,
    eval_choice_logit_single,
    evaluate_dataset,
    find_date_token_mask,
    generate_option_permutations,
    inject_te_embeddings,
)

__all__ = [
    "CONTROL_CONFIGS",
    "compute_canchi_cycle",
    "compute_batch_te",
    "compute_te_vector",
    "LoRALinear",
    "apply_lora_to_model",
    "compute_intra_loss",
    "compute_total_loss",
    "find_date_token_mask",
    "inject_te_embeddings",
    "generate_option_permutations",
    "eval_choice_logit_single",
    "evaluate_dataset",
    "_find_clean_path",
    "CLEAN_TRAIN_PATH",
    "CLEAN_EVAL_PATH",
    "OUTPUT_DIR",
    "NOTEBOOK_PATH",
    "METADATA_PATH",
    "create_kaggle_notebook",
]

if __name__ == "__main__":
    create_kaggle_notebook()
