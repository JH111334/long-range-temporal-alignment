"""Test the optional Kaggle adapter without making network calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from temporal_alignment.ServiceAdapters.kaggle_runtime import run_request


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request(root: Path, *, privacy: str = "public") -> Path:
    worker = root / "worker.py"
    worker.write_text("print('probe')\n", encoding="utf-8")
    source = root / "input.json"
    source.write_text("{}\n", encoding="utf-8")
    request = root / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": "kaggle-gpu-request-v1",
                "operation": "preflight",
                "project_root": str(root),
                "external_compute_approved": True,
                "privacy_class": privacy,
                "kernel_slug": "owner/probe",
                "worker": "worker.py",
                "worker_sha256": _hash(worker),
                "inputs": [
                    {
                        "path": "input.json",
                        "sha256": _hash(source),
                        "upload_approved": True,
                    }
                ],
                "expected_outputs": ["gpu_probe.json"],
                "run_root": "artifacts/probe",
                "report": "artifacts/probe/preflight.json",
            }
        ),
        encoding="utf-8",
    )
    return request


def test_preflight_builds_hashed_bundle_without_network(tmp_path: Path) -> None:
    report = run_request(_request(tmp_path))
    assert report["status"] == "PASS"
    assert report["network_executed"] is False
    assert report["quality_claim_allowed"] is False
    assert (tmp_path / "artifacts/probe/bundle/main.py").is_file()


def test_preflight_rejects_non_public_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="privacy_class=public"):
        run_request(_request(tmp_path, privacy="private"))
