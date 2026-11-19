"""
Ticktack Baseline Subpackage (Han et al., 2025).

Centralizes model architecture, Can Chi intra loss, date token masking,
choice logit debiasing evaluation, and Kaggle notebook generator.
"""

from baselines.ticktack.models import (
    CONTROL_CONFIGS,
    compute_canchi_cycle,
    compute_batch_te,
    compute_te_vector,
    LoRALinear,
    apply_lora_to_model,
)

from baselines.ticktack.losses import (
    compute_intra_loss,
    compute_total_loss,
)

from baselines.ticktack.date_mask import (
    find_date_token_mask,
    inject_te_embeddings,
)

from baselines.ticktack.eval_debiasing import (
    generate_option_permutations,
    eval_choice_logit_single,
    evaluate_dataset,
)

from baselines.ticktack.build_kaggle_notebook import (
    CLEAN_TRAIN_PATH,
    CLEAN_EVAL_PATH,
    OUTPUT_DIR,
    NOTEBOOK_PATH,
    METADATA_PATH,
    create_kaggle_notebook,
    _find_clean_path,
)

from baselines.ticktack.build_kaggle_notebook_3b_full import (
    create_kaggle_notebook_3b_full,
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
    "CLEAN_TRAIN_PATH",
    "CLEAN_EVAL_PATH",
    "OUTPUT_DIR",
    "NOTEBOOK_PATH",
    "METADATA_PATH",
    "create_kaggle_notebook",
    "create_kaggle_notebook_3b_full",
    "_find_clean_path",
]
