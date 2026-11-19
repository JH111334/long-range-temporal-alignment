"""Resilient Kaggle monitor and downloader for V2 Run 05.1."""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEST_DIR = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_1"
def get_kaggle_auth() -> tuple[str, str]:
    """Retrieve Kaggle credentials securely from environment or ~/.kaggle/kaggle.json."""
    import os
    u = os.environ.get("KAGGLE_USERNAME")
    k = os.environ.get("KAGGLE_KEY")
    if u and k:
        return (u, k)
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        d = json.loads(cfg.read_text(encoding="utf-8"))
        return (d.get("username", ""), d.get("key", ""))
    return ("", "")


URL = f"https://www.kaggle.com/api/v1/kernels/output?userName=qh0205&kernelSlug=lrr-v2-run05-1-phoenix-recovery"


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    start_time = time.time()
    print(f"Monitoring Kaggle Kernel: {KERNEL_SLUG} on GPU...", flush=True)

    while True:
        elapsed = int(time.time() - start_time)
        try:
            status_obj = api.kernels_status(KERNEL_SLUG)
            status = str(status_obj.status)
        except Exception as e:
            print(f"[{elapsed}s] Warning getting status: {e}", flush=True)
            time.sleep(10)
            continue

        print(f"[{elapsed}s] Kernel Status: {status}", flush=True)

        if "COMPLETE" in status:
            print("\nKernel Execution Completed on Kaggle GPU! Downloading artifacts...", flush=True)
            r = requests.get(URL, auth=get_kaggle_auth(), timeout=60)
            data = r.json()
            files = data.get("files", [])
            print(f"Total files available to download: {len(files)}", flush=True)

            for f_info in files:
                f_name = f_info.get("fileName", "")
                f_url = f_info.get("url", "")
                if not f_name or not f_url:
                    continue

                # We want artifacts from /artifacts/run_05_1 or directly
                # If path starts with artifacts/run_05_1/, strip it to place into DEST_DIR
                target_rel = f_name
                if "artifacts/run_05_1/" in target_rel:
                    target_rel = target_rel.split("artifacts/run_05_1/")[-1]
                elif target_rel.startswith("v2_run05_1_app/"):
                    continue  # skip app files

                target_file = DEST_DIR / target_rel
                target_file.parent.mkdir(parents=True, exist_ok=True)

                if target_file.exists() and target_file.stat().st_size > 0:
                    print(f"Skipping already downloaded {target_rel} ({target_file.stat().st_size} bytes)", flush=True)
                    continue

                print(f"Downloading {target_rel}...", end=" ", flush=True)
                download_success = False
                for attempt in range(1, 6):
                    try:
                        resp = requests.get(f_url, timeout=60, stream=True)
                        resp.raise_for_status()
                        with open(target_file, "wb") as f_out:
                            for chunk in resp.iter_content(chunk_size=65536):
                                if chunk:
                                    f_out.write(chunk)
                        print(f"DONE ({target_file.stat().st_size} bytes)", flush=True)
                        download_success = True
                        break
                    except Exception as e:
                        print(f"\nAttempt {attempt}/5 failed for {target_rel}: {e}", flush=True)
                        time.sleep(3 * attempt)

                if not download_success:
                    print(f"FAILED to download {target_rel} after 5 attempts!", flush=True)

            print("\nAll artifacts downloaded successfully into DEST_DIR!", flush=True)
            break

        elif "ERROR" in status or "CANCEL" in status:
            print(f"\nKernel ended with status {status}! Fetching log...", flush=True)
            r = requests.get(URL, auth=get_kaggle_auth(), timeout=60)
            data = r.json()
            log_str = data.get("log", "")
            if log_str:
                try:
                    entries = json.loads(log_str)
                    print("\n--- KAGGLE EXECUTION LOG ---")
                    for e in entries:
                        print(e.get("data", ""), end="")
                except Exception:
                    print(log_str)
            raise RuntimeError(f"Kaggle kernel failed with status: {status}")

        time.sleep(15)


if __name__ == "__main__":
    main()
