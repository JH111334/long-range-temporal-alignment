"""Define collection and retention gates for temporal datasets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..contracts import DatasetManifest
from ..errors import EvidenceGateError


class DataLifecycleStage(StrEnum):
    """Name governed stages without implying that data was collected."""

    DISCOVERED = "discovered"
    APPROVED = "approved"
    COLLECTED = "collected"
    CURATED = "curated"
    FROZEN = "frozen"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class DataLifecyclePlan:
    """Describe prerequisites for collecting one dataset."""

    manifest: DatasetManifest
    group_keys: tuple[str, ...]
    retention_policy: str
    deletion_owner: str

    def assert_collection_ready(self) -> None:
        """Reject collection until governance and split ownership are explicit."""
        self.manifest.assert_ingestible()
        if not self.group_keys:
            raise EvidenceGateError("at least one group key is required before collection")
        if any(not key.strip() for key in self.group_keys):
            raise EvidenceGateError("group keys must not be empty")
        if not self.retention_policy.strip() or not self.deletion_owner.strip():
            raise EvidenceGateError("retention policy and deletion owner are required")
