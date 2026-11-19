"""Monitor Kaggle kernel execution for V2 Run 05.1 and download artifacts upon completion."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from download_kaggle_artifacts_resilient import download_kernel_outputs

KERNEL_SLUG = "qh0205/lrr-v2-run05-1-phoenix-recovery"
DEST_DIR = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_1"


def monitor_and_download():
    api = KaggleApi()
    api.authenticate()

    print(f"Monitoring Kaggle Kernel: {KERNEL_SLUG}...", flush=True)
    start_time = time.time()

    consecutive_errors = 0
    while True:
        try:
            status_res = api.kernels_status(KERNEL_SLUG)
            raw_status = str(status_res.get("status", ""))
            status = raw_status.upper()
            failure_msg = status_res.get("failureMessage")
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            print(f"Warning: api.kernels_status timed out (attempt {consecutive_errors}): {e}", flush=True)
            if consecutive_errors > 10:
                raise e
            time.sleep(10)
            continue

        elapsed = int(time.time() - start_time)
        print(f"[{elapsed}s] Kernel Status: {status} (failureMessage: {failure_msg})", flush=True)

        if status == "COMPLETE":
            print(f"\nKernel finished successfully in {elapsed}s! Downloading artifacts to {DEST_DIR}...", flush=True)
            DEST_DIR.mkdir(parents=True, exist_ok=True)
            download_kernel_outputs(KERNEL_SLUG, DEST_DIR)
            print("All artifacts downloaded successfully!", flush=True)
            break

        elif status in ["ERROR", "CANCELACK", "CANCELLED"]:
            print(f"\nKernel ended with failure status: {status} (message: {failure_msg})", flush=True)
            err_log_dir = _REPO_ROOT / "scratch" / ".tmp" / "run05_1_error_log"
            err_log_dir.mkdir(parents=True, exist_ok=True)
            try:
                download_kernel_outputs(KERNEL_SLUG, err_log_dir)
            except Exception as e:
                print(f"Could not download error logs: {e}", flush=True)
            raise RuntimeError(f"Kaggle kernel failed with status: {status}")

        time.sleep(15)


if __name__ == "__main__":
    monitor_and_download()
