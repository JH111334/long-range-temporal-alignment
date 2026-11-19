"""Stage 03 G1 Temporal-Injection Repair and Unconstrained Free-Generation Evaluation Package.

Exports contracts, locator, injector, adapters, parsers, and metrics.
"""

from .answer_parser import ParseResult, parse_answer
from .contracts import (
    AnswerMode,
    EvaluationConfig,
    GenerationPolicy,
    InvalidReason,
    ParserContract,
    PredictionRecord,
    TemporalContext,
    TemporalPolicy,
    YearMention,
)
from .generation_adapter import GeneratedContinuation, generate_unconstrained
from .metrics import aggregate_evaluation_metrics
from .model_adapter import (
    DualCycleCoreGenerator,
    GeneralizedDualCycleMaTLoRALinear,
    StaticLoRALinear,
    attach_adapters_to_model,
    load_checkpoint_weights,
    matlora_temporal_context,
)
from .prompt_adapter import extract_mcqa_option_texts, format_inference_prompt, generate_audited_permutations
from .temporal_injector import (
    GatedResidualProjection,
    compute_canchi_8d,
    compute_fixed_fone_canchi_40d,
    compute_fone_32d,
    compute_theta_z,
    inject_temporal_features,
)
from .temporal_locator import (
    StructuredLocatorError,
    build_temporal_mask,
    find_year_char_spans,
    resolve_temporal_mentions,
)

__all__ = [
    "TemporalPolicy",
    "AnswerMode",
    "ParserContract",
    "InvalidReason",
    "YearMention",
    "TemporalContext",
    "GenerationPolicy",
    "PredictionRecord",
    "EvaluationConfig",
    "format_inference_prompt",
    "extract_mcqa_option_texts",
    "generate_audited_permutations",
    "StructuredLocatorError",
    "find_year_char_spans",
    "resolve_temporal_mentions",
    "build_temporal_mask",
    "compute_fone_32d",
    "compute_canchi_8d",
    "compute_fixed_fone_canchi_40d",
    "compute_theta_z",
    "GatedResidualProjection",
    "inject_temporal_features",
    "StaticLoRALinear",
    "DualCycleCoreGenerator",
    "GeneralizedDualCycleMaTLoRALinear",
    "attach_adapters_to_model",
    "matlora_temporal_context",
    "load_checkpoint_weights",
    "GeneratedContinuation",
    "generate_unconstrained",
    "ParseResult",
    "parse_answer",
    "aggregate_evaluation_metrics",
]
