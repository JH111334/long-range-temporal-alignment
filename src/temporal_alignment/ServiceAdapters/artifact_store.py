"""Artifact-store boundary for future local or object-storage implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..errors import EvidenceGateError


class ArtifactStore(Protocol):
    """Define immutable artifact lookup without choosing a storage backend."""

    def resolve(self, artifact_id: str) -> Path:
        """Resolve an approved artifact identifier to a local path."""


class DisabledArtifactStore:
    """Fail-closed placeholder used until artifact storage is configured."""

    def resolve(self, artifact_id: str) -> Path:
        """Reject lookup so scaffold cannot masquerade as readiness."""
        raise EvidenceGateError(f"artifact store is not configured for {artifact_id!r}")
