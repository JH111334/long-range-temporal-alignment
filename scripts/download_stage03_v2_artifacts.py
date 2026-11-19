"""Download Stage 03 V2 Artifacts Resiliently from Kaggle.

Downloads:
- Job 2 V2: qh0205/lrr-stage03-3b-base-targeted-alignment -> artifacts/kaggle_outputs/job2_stage03_v2
- Job 3 V2: qh0205/lrr-stage03-3b-matlora-fone-full -> artifacts/kaggle_outputs/job3_matlora_v2
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from kaggle.api.kaggle_api_extended import KaggleApi
from scripts.download_kaggle_artifacts_resilient import download_kernel_outputs


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_and_download(kernel_slug: str, target_dir: Path) -> bool:
    api = KaggleApi()
    api.authenticate()

    print(f"\n[{datetime.now().isoformat()}] Checking kernel: {kernel_slug}")
    try:
        status_obj = api.kernels_status(kernel_slug)
        status = str(status_obj.status)
        failure = status_obj.failure_message
        print(f"Status: {status} | Failure message: {failure}")
    except Exception as e:
        print(f"Failed to query status: {e}")
        return False

    print(f"Downloading outputs for {kernel_slug} to {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    download_kernel_outputs(
        kernel=kernel_slug,
        target_dir=target_dir,
        file_pattern=r"\.(json|pt|log|txt)$",
        max_retries=5,
    )

    # Compute hashes of downloaded files
    print("\nDownloaded files and SHA-256 hashes:")
    manifest = {}
    for f in target_dir.glob("*"):
        if f.is_file():
            sha = compute_sha256(f)
            size = f.stat().st_size
            print(f"- {f.name} ({size} bytes): {sha}")
            manifest[f.name] = {"bytes": size, "sha256": sha}

    manifest_path = target_dir / "DOWNLOAD_MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return True


def main():
    print("=== Stage 03 V2 Artifacts Downloader ===")
    j2_dir = _REPO_ROOT / "artifacts" / "kaggle_outputs" / "job2_stage03_v2"
    j3_dir = _REPO_ROOT / "artifacts" / "kaggle_outputs" / "job3_matlora_v2"

    j2_done = check_and_download("qh0205/lrr-stage03-3b-base-targeted-alignment", j2_dir)
    j3_done = check_and_download("qh0205/lrr-stage03-3b-matlora-fone-full", j3_dir)

    print("\nSummary:")
    print(f"Job 2 V2: {'DOWNLOADED' if j2_done else 'NOT_READY'}")
    print(f"Job 3 V2: {'DOWNLOADED' if j3_done else 'NOT_READY'}")


if __name__ == "__main__":
    main()
