"""Load and enforce dataset registry gates."""

from __future__ import annotations

from pathlib import Path

from .contracts import DatasetManifest
from .enums import AssetStatus, DataRole
from .errors import EvidenceGateError
from .io import load_json


def load_dataset_manifests(path: str | Path) -> dict[str, DatasetManifest]:
    payload = load_json(path)
    manifests: dict[str, DatasetManifest] = {}
    for item in payload.get("datasets", []):
        manifest = DatasetManifest(
            dataset_id=item["dataset_id"],
            official_url=item["official_url"],
            accessed_on=item["accessed_on"],
            license_or_terms=item["license_or_terms"],
            reviewer=item["reviewer"],
            revision=item["revision"],
            sha256=item["sha256"],
            role=DataRole(item["role"]),
            status=AssetStatus(item["status"]),
            notes=item.get("notes", ""),
        )
        if manifest.dataset_id in manifests:
            raise EvidenceGateError(f"duplicate dataset_id: {manifest.dataset_id}")
        manifests[manifest.dataset_id] = manifest
    return manifests


def require_approved_dataset(path: str | Path, dataset_id: str) -> DatasetManifest:
    manifests = load_dataset_manifests(path)
    try:
        manifest = manifests[dataset_id]
    except KeyError as exc:
        raise EvidenceGateError(f"unknown dataset: {dataset_id}") from exc
    manifest.assert_ingestible()
    return manifest
