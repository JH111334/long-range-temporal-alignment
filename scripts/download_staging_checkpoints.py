"""Download epoch checkpoints into scratch/checkpoint_staging.
"""
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGING_DIR = REPO_ROOT / "scratch" / "checkpoint_staging"
STAGING_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "stage03_3b_targeted_static_mlp_5epochs_ep1_weights.pt",
    "stage03_3b_targeted_static_mlp_5epochs_ep2_weights.pt",
    "stage03_3b_targeted_static_mlp_5epochs_ep3_weights.pt",
    "stage03_3b_targeted_static_mlp_5epochs_ep4_weights.pt",
    "stage03_3b_targeted_static_mlp_5epochs_ep5_weights.pt",
    "stage03_3b_targeted_matlora_5epochs_ep1_weights.pt",
    "stage03_3b_targeted_matlora_5epochs_ep2_weights.pt",
    "stage03_3b_targeted_matlora_5epochs_ep3_weights.pt",
    "stage03_3b_targeted_matlora_5epochs_ep4_weights.pt",
    "stage03_3b_targeted_matlora_5epochs_ep5_weights.pt",
]

for fn in FILES:
    dst = STAGING_DIR / fn
    if dst.exists():
        print(f"[ALREADY EXISTS] {fn} ({dst.stat().st_size} bytes)")
        continue
    print(f"[DOWNLOADING] {fn} -> {dst}...")
    cmd = ["modal", "volume", "get", "stage03-artifacts", fn, str(dst)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"  -> SUCCESS ({dst.stat().st_size} bytes)")
    else:
        print(f"  -> FAILED: {res.stderr}")

print("Download step complete.")
