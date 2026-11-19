"""Scaffold adapters for governed external assets."""

from .contracts import DownloadPlan, ExternalAssetSpec, ExternalServiceAdapter
from .inference import (
    HuggingFaceInferenceAdapter,
    InferenceArtifact,
    InferencePlan,
    RemoteInferenceAdapter,
)

__all__ = [
    "DownloadPlan",
    "ExternalAssetSpec",
    "ExternalServiceAdapter",
    "HuggingFaceInferenceAdapter",
    "InferenceArtifact",
    "InferencePlan",
    "RemoteInferenceAdapter",
]
