"""Monitor and Download Arm 4 (Static MLP LoRA) Results for hoangkhang23.

Polls hoangkhang23/lrr-stage03-3b-static-mlp-lora until COMPLETE,
then downloads:
- stage03_3b_static_mlp_lora_weights.pt
- stage03_3b_static_mlp_lora_results.json
- training_summary.json

Then triggers four-arm comparison analysis.
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


def check_and_download(kernel_slug: str = "hoangkhang23/lrr-stage03-3b-static-mlp-lora") -> str:
    token = get_hoangkhang_token()
    os.environ["KAGGLE_API_TOKEN"] = token

    api = KaggleApi()
    api.authenticate()

    dest_dir = _REPO_ROOT / "artifacts" / "kaggle_outputs" / "job4_static_mlp_lora"
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
            file_pattern=r"\.(json|log|txt|pt)$",
            max_retries=5,
        )
        print("\nDownloaded files and hashes:")
        manifest = {}
        for f in dest_dir.iterdir():
            if f.is_file() and f.name != "DOWNLOAD_MANIFEST.json":
                sha = compute_sha256(f)
                manifest[f.name] = {"bytes": f.stat().st_size, "sha256": sha}
                print(f"- {f.name} ({f.stat().st_size:,} bytes): {sha}")

        with open(dest_dir / "DOWNLOAD_MANIFEST.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return "COMPLETE"

    elif "ERROR" in status:
        print("Kernel ended in ERROR. Inspecting logs...")
        try:
            from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
            owner, slug = kernel_slug.split("/")
            with api.build_kaggle_client() as k:
                req = ApiListKernelSessionOutputRequest()
                req.user_name = owner
                req.kernel_slug = slug
                resp = k.kernels.kernels_api_client.list_kernel_session_output(req)
                log_content = resp.log or ""
                log_file = dest_dir / f"{slug}.log"
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(log_content)
                print(f"Kernel log saved to {log_file} ({len(log_content)} chars)")
                # Print last 30 lines of log
                lines = log_content.splitlines()
                print("\n--- Last 30 lines of log ---")
                for l in lines[-30:]:
                    print(l)
        except Exception as e:
            print("Could not fetch kernel output:", e)
        return "ERROR"

    else:
        return "RUNNING"


def main():
    kernel_slug = "hoangkhang23/lrr-stage03-3b-static-mlp-lora"
    poll_interval = 60  # seconds

    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        check_and_download(kernel_slug)
        return

    print(f"Bắt đầu theo dõi Kernel Arm 4: {kernel_slug}...")
    while True:
        status = check_and_download(kernel_slug)
        if status in ("COMPLETE", "ERROR"):
            break
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
