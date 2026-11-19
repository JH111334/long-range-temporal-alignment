"""Hugging Face adapter scaffold; network execution is deliberately closed."""

from __future__ import annotations

from pathlib import Path

from ..errors import ContractError, EvidenceGateError
from .contracts import DownloadPlan, ExternalAssetSpec


class HuggingFaceAdapter:
    """Plan pinned Hub access while refusing unimplemented downloads."""

    name = "huggingface"

    def build_plan(self, spec: ExternalAssetSpec, cache_dir: Path) -> DownloadPlan:
        """Return a local-only plan for a pinned Hugging Face revision."""
        if spec.provider.casefold() != self.name:
            raise ContractError("asset provider must be huggingface")
        if spec.revision.casefold() in {"main", "master", "latest"}:
            raise ContractError("Hugging Face revision must be immutable")
        return DownloadPlan(spec=spec, cache_dir=cache_dir, adapter_name=self.name)

    def fetch(self, plan: DownloadPlan) -> Path:
        """Fail closed until an ADR and network tests approve implementation."""
        raise EvidenceGateError("HuggingFaceAdapter is scaffold-only; fetch is disabled")
