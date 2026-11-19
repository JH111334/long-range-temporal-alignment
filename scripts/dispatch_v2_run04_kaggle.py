"""Dispatch V2 Run 04 Autonomous GPU Execution to Kaggle via Clean Input Dataset.

Architecture:
- Large files (C1 checkpoint 13 MB, datasets) mounted via Kaggle Dataset: qh0205/lrr-v2-run04-clean-input
- Light code payload (185 KB zip) embedded in kernel runner
- Kernel configured with GPU (T4/P100) and Internet enabled
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def build_and_push_kaggle():
    print("Preparing light Kaggle dispatch bundle for V2 Run 04...")

    bundle_dir = _REPO_ROOT / "artifacts" / "kaggle_bundles" / "v2_run04"
    app_dir = bundle_dir / "app_staging"
    if app_dir.exists():
        shutil.rmtree(app_dir, ignore_errors=True)
    app_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy Source Tree & Executor
    shutil.copytree(_REPO_ROOT / "src", app_dir / "src")
    shutil.copy(_REPO_ROOT / "scripts" / "kaggle_v2_run04_executor.py", app_dir / "executor.py")

    # 2. Create light payload zip
    zip_path = bundle_dir / "light_payload.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in app_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(app_dir))

    zip_size_kb = zip_path.stat().st_size / 1024
    print(f"Created light payload zip: {zip_size_kb:.2f} KB")

    # 3. Base64-encode light payload
    payload_bytes = zip_path.read_bytes()
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    b64_str = base64.b64encode(payload_bytes).decode("ascii")

    chunk_size = 76
    chunks = "\n".join(f'    "{b64_str[i:i + chunk_size]}"' for i in range(0, len(b64_str), chunk_size))

    # 4. Write Kaggle Driver Script
    dispatch_dir = bundle_dir / "dispatch_staging"
    if dispatch_dir.exists():
        shutil.rmtree(dispatch_dir, ignore_errors=True)
    dispatch_dir.mkdir(parents=True, exist_ok=True)

    driver_code = f'''from __future__ import annotations
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path("/kaggle/working")
INPUT_DATA_DIR = Path("/kaggle/input/lrr-v2-run04-clean-input")
PAYLOAD_SHA = "{payload_sha256}"
PAYLOAD_B64 = (
{chunks}
)

print(f"=== STARTING LRR V2 RUN 04 AUTONOMOUS EXECUTION ===")
print(f"Verifying input dataset at: {{INPUT_DATA_DIR}}")
if not INPUT_DATA_DIR.exists():
    # Check alternate input directory if dataset slug was changed
    input_dirs = list(Path("/kaggle/input").glob("*"))
    print(f"Available inputs in /kaggle/input: {{input_dirs}}")
    for d in input_dirs:
        if (d / "C1_COMPACT_ADAPTER_ONLY.pt").exists():
            INPUT_DATA_DIR = d
            print(f"Found input data in: {{INPUT_DATA_DIR}}")
            break

c1_ckpt = INPUT_DATA_DIR / "C1_COMPACT_ADAPTER_ONLY.pt"
if not c1_ckpt.exists():
    raise FileNotFoundError(f"Missing C1 checkpoint at: {{c1_ckpt}}")
print(f"Found C1 checkpoint: {{c1_ckpt}} ({{c1_ckpt.stat().st_size / (1024*1024):.2f}} MB)")

zip_p = ROOT / "payload.zip"
zip_p.write_bytes(base64.b64decode(PAYLOAD_B64))

actual_sha = hashlib.sha256(zip_p.read_bytes()).hexdigest()
if actual_sha != PAYLOAD_SHA:
    raise RuntimeError(f"Payload SHA256 mismatch: expected {{PAYLOAD_SHA}}, got {{actual_sha}}")

app_dir = ROOT / "v2_run04_app"
app_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_p) as zf:
    zf.extractall(app_dir)
zip_p.unlink()

print("Installing required Python dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "transformers", "accelerate"], check=True)

print("Starting V2 Run 04 GPU Executor on CUDA...")
out_artifacts = ROOT / "artifacts" / "run_04"
out_artifacts.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, str(app_dir / "executor.py"),
    "--data_dir", str(INPUT_DATA_DIR),
    "--c1_checkpoint", str(c1_ckpt),
    "--output_dir", str(out_artifacts),
]

sub_env = dict(os.environ)
sub_env["PYTHONPATH"] = str(app_dir / "src") + (":" + sub_env["PYTHONPATH"] if "PYTHONPATH" in sub_env else "")

proc = subprocess.run(cmd, env=sub_env, text=True)
if proc.returncode != 0:
    print(f"ERROR: Executor failed with exit code {{proc.returncode}}", file=sys.stderr)
    sys.exit(proc.returncode)

print("=== V2 RUN 04 KAGGLE GPU EXECUTION FINISHED SUCCESSFULLY ===")
'''

    driver_path = dispatch_dir / "kernel_run04.py"
    driver_path.write_text(driver_code, encoding="utf-8")

    # 5. Write kernel-metadata.json
    slug = "lrr-v2-run04-truth-repair"
    meta = {
        "id": f"qh0205/{slug}",
        "title": "LRR V2 Run04 Truth Repair",
        "code_file": "kernel_run04.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": ["qh0205/lrr-v2-run04-clean-input"],
        "competition_sources": [],
        "kernel_sources": [],
    }

    meta_path = dispatch_dir / "kernel-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 6. Push to Kaggle
    print(f"\nPushing kernel to Kaggle: qh0205/{slug}...")
    res = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(dispatch_dir)],
        capture_output=True,
        text=True,
    )
    print("STDOUT:\n" + res.stdout)
    if res.stderr:
        print("STDERR:\n" + res.stderr)

    if res.returncode != 0:
        raise RuntimeError(f"Kaggle push failed: {res.stderr}")

    print(f"Successfully dispatched kernel qh0205/{slug} with GPU acceleration!")


if __name__ == "__main__":
    build_and_push_kaggle()
