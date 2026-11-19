"""Fail-closed contracts for external dataset and model services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..contracts import _required
from ..errors import ContractError

ALLOWED_DATA_ROLES = frozenset({"train", "validation", "test", "reference-only"})


@dataclass(frozen=True, slots=True)
class ExternalAssetSpec:
    """Describe an approved, immutable external asset before any network call."""

    provider: str
    resource_id: str
    source_url: str
    accessed_on: str
    license_id: str
    terms_url: str
    revision: str
    sha256: str
    reviewer: str
    data_role: str

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            _required(str(getattr(self, field_name)), field_name)
        if self.data_role not in ALLOWED_DATA_ROLES:
            raise ContractError(f"unsupported data_role: {self.data_role}")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ContractError("sha256 must be 64 lowercase hexadecimal characters")
        if not self.source_url.startswith("https://") or not self.terms_url.startswith("https://"):
            raise ContractError("source_url and terms_url must use HTTPS")


@dataclass(frozen=True, slots=True)
class DownloadPlan:
    """Record a non-executing plan for one approved external asset."""

    spec: ExternalAssetSpec
    cache_dir: Path
    adapter_name: str
    local_only: bool = True


class ExternalServiceAdapter(Protocol):
    """Define planning and explicit fetch boundaries for a provider adapter."""

    def build_plan(self, spec: ExternalAssetSpec, cache_dir: Path) -> DownloadPlan:
        """Validate and return a plan without network or filesystem mutation."""

    def fetch(self, plan: DownloadPlan) -> Path:
        """Fetch an asset only after a separate implementation gate."""
