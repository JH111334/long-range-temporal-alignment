"""Monitor and Download Unified Evaluation Results for hoangkhang23.

Polls hoangkhang23/lrr-stage03-3b-unified-eval until COMPLETE,
then downloads:
- stage03_3b_base_targeted_results.json
- stage03_3b_matlora_fone_results.json
- stage03_3b_unified_comparison.json

Then triggers three-arm McNemar analysis.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
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


def get_hoangkhang_token() -> str:
    odt_path = _REPO_ROOT / "HKKAGGLE.odt"
    with zipfile.ZipFile(odt_path, "r") as z:
        xml_content = z.read("content.xml")
    root = ET.fromstring(xml_content)
    full_text = " ".join(elem.text for elem in root.iter() if elem.text).strip()
    return full_text.split("=")[-1].strip()


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_and_download(kernel_slug: str = "hoangkhang23/lrr-stage03-3b-unified-eval") -> str:
    token = get_hoangkhang_token()
    os.environ["KAGGLE_API_TOKEN"] = token

    api = KaggleApi()
    api.authenticate()

    dest_dir = _REPO_ROOT / "artifacts" / "kaggle_outputs" / "stage03_3b_unified_eval"
    dest_dir.mkdir(parents=True, exist_ok=True)

    status_obj = api.kernels_status(kernel_slug)
    status = str(status_obj.status)
    fail_msg = status_obj.failure_message
    print(f"[{datetime.now().isoformat()}] Kernel: {kernel_slug} | Status: {status} | Failure: {fail_msg}")

    if "COMPLETE" in status:
        print("Kernel is COMPLETE! Downloading results...")
        download_kernel_outputs(
            kernel=kernel_slug,
            target_dir=dest_dir,
            file_pattern=r"\.(json|log|txt)$",
            max_retries=5,
        )
        print("\nDownloaded files and hashes:")
        for f in dest_dir.glob("*.json"):
            sha = compute_sha256(f)
            print(f"- {f.name} ({f.stat().st_size:,} bytes): {sha}")

        # Also copy results to respective arm output directories for compatibility
        j2_dest = _REPO_ROOT / "artifacts" / "kaggle_outputs" / "job2_stage03_v2"
        j3_dest = _REPO_ROOT / "artifacts" / "kaggle_outputs" / "job3_matlora_v2"
        j2_dest.mkdir(parents=True, exist_ok=True)
        j3_dest.mkdir(parents=True, exist_ok=True)

        res2 = dest_dir / "stage03_3b_base_targeted_results.json"
        res3 = dest_dir / "stage03_3b_matlora_fone_results.json"
        if res2.exists():
            import shutil
            shutil.copy(res2, j2_dest / "stage03_3b_base_targeted_results.json")
            print("Copied Arm 2 results to job2_stage03_v2/")
        if res3.exists():
            import shutil
            shutil.copy(res3, j3_dest / "stage03_3b_matlora_fone_results.json")
            print("Copied Arm 3 results to job3_matlora_v2/")

        return "COMPLETE"

    elif "ERROR" in status:
        print("Kernel ended in ERROR. Inspecting logs...")
        logs = api.kernels_logs(kernel_slug)
        if logs:
            print("--- LOG TAIL ---")
            print(logs[-1000:])
        return "ERROR"

    return "RUNNING"


if __name__ == "__main__":
    check_and_download()
