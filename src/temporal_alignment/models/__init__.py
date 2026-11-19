"""Models and Adapters for Temporal Alignment."""

from .fone_canchi_40d import (
    compute_canchi_8d,
    compute_fixed_fone_canchi_40d,
    compute_fone_32d,
    extract_canchi_angles_from_40d,
    verify_golden_manifest,
)
from .dual_cycle_mat_lora import (
    DualCycleCoreGenerator,
    DualCycleMaTLoRALinear,
    replace_mlp_with_dual_cycle_mat_lora,
    set_model_temporal_coordinates,
)
from .gated_seam import (
    ConditioningContext,
    GatedResidualProjection,
    make_gated_embedding_hook,
)
from .tempun_dataset import (
    STAGES,
    STAGE_YEARS,
    TempUNExample,
    load_tempun_eval_records,
    load_tempun_train_records,
    parse_strict_mcqa_answer,
    tokenize_masked_target_example,
)

__all__ = [
    "compute_fone_32d",
    "compute_canchi_8d",
    "compute_fixed_fone_canchi_40d",
    "extract_canchi_angles_from_40d",
    "verify_golden_manifest",
    "DualCycleCoreGenerator",
    "DualCycleMaTLoRALinear",
    "replace_mlp_with_dual_cycle_mat_lora",
    "set_model_temporal_coordinates",
    "ConditioningContext",
    "GatedResidualProjection",
    "make_gated_embedding_hook",
    "STAGES",
    "STAGE_YEARS",
    "TempUNExample",
    "load_tempun_train_records",
    "load_tempun_eval_records",
    "parse_strict_mcqa_answer",
    "tokenize_masked_target_example",
]
