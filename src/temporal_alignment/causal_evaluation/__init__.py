"""Temporal alignment evaluation and diagnostic modules."""

from __future__ import annotations

from .canonical_evaluator import (
    CanonicalEvaluationConfig,
    evaluate_dataset_canonical,
    generate_scoreboard_from_predictions,
)
from .clustered_mcnemar import (
    ClusteredMcNemarResult,
    compute_clustered_mcnemar,
    compute_clustered_bootstrap_ci,
)
from .mechanism_tracer import (
    MechanismTraceResult,
    TemporalMechanismTracer,
)
from .otel_tracing import (
    Stage03Tracer,
    get_global_tracer,
)

__all__ = [
    "CanonicalEvaluationConfig",
    "evaluate_dataset_canonical",
    "generate_scoreboard_from_predictions",
    "ClusteredMcNemarResult",
    "compute_clustered_mcnemar",
    "compute_clustered_bootstrap_ci",
    "MechanismTraceResult",
    "TemporalMechanismTracer",
    "Stage03Tracer",
    "get_global_tracer",
]
