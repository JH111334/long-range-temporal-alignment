"""Kaggle adapter scaffold; versionless and live downloads are rejected."""

from __future__ import annotations

from pathlib import Path

from ..errors import ContractError, EvidenceGateError
from .contracts import DownloadPlan, ExternalAssetSpec


class KaggleAdapter:
    """Plan versioned Kaggle access while refusing network execution."""

    name = "kaggle"

    def build_plan(self, spec: ExternalAssetSpec, cache_dir: Path) -> DownloadPlan:
        """Return a local-only plan for a versioned Kaggle handle."""
        if spec.provider.casefold() != self.name:
            raise ContractError("asset provider must be kaggle")
        if "/versions/" not in spec.resource_id:
            raise ContractError("Kaggle resource_id must include /versions/<n>")
        return DownloadPlan(spec=spec, cache_dir=cache_dir, adapter_name=self.name)

    def fetch(self, plan: DownloadPlan) -> Path:
        """Fail closed until an ADR and network tests approve implementation."""
        raise EvidenceGateError("KaggleAdapter is scaffold-only; fetch is disabled")
