"""Download Stage 03 Arm 4 Artifacts from Modal Volume and trigger Four-Arm McNemar Audit."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

_REPO_ROOT = Path(__file__).resolve().parents[1]

app = modal.App("download-stage03-artifacts")
volume = modal.Volume.from_name("stage03-artifacts", create_if_missing=True)


@app.function(volumes={"/vol": volume})
def fetch_volume_files() -> dict:
    vol_path = Path("/vol")
    files_data = {}
    for p in vol_path.iterdir():
        if p.is_file():
            files_data[p.name] = p.read_bytes()
    return files_data


def main():
    print("Checking Modal Volume 'stage03-artifacts'...")
    out_dir = _REPO_ROOT / "artifacts" / "kaggle_outputs" / "job4_static_mlp_lora"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = _REPO_ROOT / "Documents" / "Documenting" / "Results" / "STAGE03_3B_V2_20261026"
    results_dir.mkdir(parents=True, exist_ok=True)

    with app.run():
        files_data = fetch_volume_files.remote()

    print(f"Files found in Volume: {list(files_data.keys())}")
    if not files_data:
        print("Volume is currently empty or job has not committed files yet.")
        return False

    for name, data in files_data.items():
        dst1 = out_dir / name
        dst1.write_bytes(data)
        print(f"Saved to {dst1} ({len(data):,} bytes)")
        if name.endswith(".json"):
            dst2 = results_dir / name
            dst2.write_bytes(data)
            print(f"Copied to {dst2}")

    res_file = out_dir / "stage03_3b_static_mlp_lora_results.json"
    if res_file.exists():
        print("\nTriggering four-arm McNemar analysis...")
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "audit_stage03_3b_four_arm_comparison.py"),
            "--job4_file", str(res_file),
        ]
        subprocess.run(cmd, check=True)
        print("Four-arm McNemar analysis completed successfully.")
        return True
    return False


if __name__ == "__main__":
    main()
