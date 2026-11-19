"""Upload Stage 03 Training Bundle Dataset for hoangkhang23.

Packages train_104k_clean.jsonl.gz, eval_5292_clean.jsonl.gz, and counterfactuals
into Kaggle dataset hoangkhang23/lrr-stage03-train-bundle.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def get_hoangkhang_token() -> str:
    odt_path = _REPO_ROOT / "HKKAGGLE.odt"
    with zipfile.ZipFile(odt_path, "r") as z:
        xml_content = z.read("content.xml")
    root = ET.fromstring(xml_content)
    full_text = " ".join(elem.text for elem in root.iter() if elem.text).strip()
    return full_text.split("=")[-1].strip()


def main():
    token = get_hoangkhang_token()
    os.environ["KAGGLE_API_TOKEN"] = token

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("Authenticated successfully with Kaggle.")

    target_dir = _REPO_ROOT / "artifacts" / "kaggle_datasets" / "hoangkhang23_stage03_train_bundle"
    target_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "title": "LRR Stage03 Train Bundle",
        "id": "hoangkhang23/lrr-stage03-train-bundle",
        "licenses": [
            {
                "name": "CC0-1.0"
            }
        ]
    }
    with open(target_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    src_train = _REPO_ROOT / "data" / "tempun_full_clean" / "train_104k_clean.jsonl.gz"
    src_eval = _REPO_ROOT / "data" / "tempun_full_clean" / "eval_5292_clean.jsonl.gz"
    src_cf = _REPO_ROOT / "data" / "counterfactual" / "COUNTERFACTUAL_GOLD_AUDIT.jsonl"

    shutil.copy2(src_train, target_dir / "train_104k_clean.jsonl.gz")
    shutil.copy2(src_eval, target_dir / "eval_5292_clean.jsonl.gz")
    shutil.copy2(src_cf, target_dir / "grounded_counterfactual_benchmark.jsonl")

    print(f"Prepared files in {target_dir}:")
    for item in target_dir.iterdir():
        print(f"  - {item.name} ({item.stat().st_size:,} bytes)")

    print("Uploading dataset hoangkhang23/lrr-stage03-train-bundle to Kaggle...")
    try:
        status = api.dataset_status("hoangkhang23/lrr-stage03-train-bundle")
        print(f"Dataset already exists with status: {status}. Creating new version...")
        res = api.dataset_create_version(
            folder=str(target_dir),
            version_notes="Upload clean 104k train and 5292 eval with counterfactuals",
            dir_mode="zip",
            quiet=False,
        )
        print("Dataset version created:", res)
    except Exception as e:
        print(f"Dataset not found or error ({e}), creating new dataset...")
        res = api.dataset_create_new(
            folder=str(target_dir),
            dir_mode="zip",
            quiet=False,
        )
        print("Dataset created:", res)


if __name__ == "__main__":
    main()
