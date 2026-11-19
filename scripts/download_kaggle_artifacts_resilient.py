"""Resilient downloader for Kaggle kernel session output files with retry and resume support."""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi


def download_kernel_outputs(
    kernel: str,
    target_dir: Path | str,
    file_pattern: Optional[str] = None,
    max_retries: int = 5,
    timeout: int = 30,
) -> None:
    api = KaggleApi()
    api.authenticate()
    owner_slug, kernel_slug, _ = api.parse_kernel_string(kernel)

    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    compiled_pattern = re.compile(file_pattern) if file_pattern else None

    # Setup resilient requests session
    session = requests.Session()
    retries = Retry(
        total=max_retries,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    page_token = None
    all_files = []

    print(f"Fetching file list for kernel {kernel}...")
    from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest

    with api.build_kaggle_client() as client:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = owner_slug
            req.kernel_slug = kernel_slug
            api._set_paging(req, page_size=100, page_token=page_token)
            response = client.kernels.kernels_api_client.list_kernel_session_output(req)
            if response.files:
                all_files.extend(response.files)
            page_token = response.next_page_token
            if not page_token:
                break

    print(f"Total files available on Kaggle: {len(all_files)}")

    for item in all_files:
        fn = item.file_name
        if compiled_pattern and not compiled_pattern.search(fn):
            continue

        dest_file = target_path / fn
        if dest_file.exists() and dest_file.stat().st_size > 0:
            print(f"Skipping already existing: {fn} ({dest_file.stat().st_size} bytes)")
            continue

        dest_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading: {fn} ...", end=" ", flush=True)

        success = False
        for attempt in range(1, max_retries + 1):
            try:
                resp = session.get(item.url, timeout=timeout, stream=True)
                resp.raise_for_status()
                with open(dest_file, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                print(f"DONE ({dest_file.stat().st_size} bytes)")
                success = True
                break
            except Exception as e:
                print(f"Attempt {attempt}/{max_retries} failed: {e}")
                time.sleep(2 * attempt)

        if not success:
            print(f"FAILED to download {fn} after {max_retries} attempts.")


if __name__ == "__main__":
    download_kernel_outputs(
        kernel="qh0205/lrr-fc105-m0-fc40-3seed",
        target_dir="artifacts/quarantine/M0_FC40/kaggle_3seed_json",
        file_pattern=r"\.(json|log|txt)$",
    )
