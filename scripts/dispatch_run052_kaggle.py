"""Dispatch V2 Run 05.2 Mechanism Contract Repair GPU Execution to Kaggle.

Pre-dispatch Gating:
- Verifies that Gatekeeper gates P0 through P5 are ALL in status PASS.
- If any prerequisite fails, dispatch halts immediately (fail-closed).

Architecture:
- Inputs mounted via Kaggle Dataset: qh0205/lrr-v2-run04-clean-input (C1 checkpoint, datasets)
- Light code payload (src/, executor, evaluator, preflight audit) embedded in kernel runner
- Kernel configured with GPU (T4/P100) and Internet enabled
- Zero local CPU usage for training or full inference
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
_RUN_DIR = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_2"


def verify_preflight_gates() -> None:
    """Enforce that gates P0 through P5 must all be PASS before dispatch."""
    gates_file = _RUN_DIR / "GATES.json"
    if not gates_file.exists():
        raise RuntimeError(f"Cannot dispatch: GATES.json missing at {gates_file}")

    with open(gates_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_gates = ["P0_SOURCE", "P1_DATA", "P2_MODEL", "P3_LOSS", "P4_SAMPLER", "P5_CANARY"]
    gates_dict = data.get("gates", {})

    for gid in required_gates:
        info = gates_dict.get(gid)
        if not info or info.get("status") != "PASS":
            raise RuntimeError(f"Fail-Closed Gatekeeper Check: Gate '{gid}' is NOT in PASS status! Current info: {info}")

    print("Pre-dispatch Gating Verified: Gates P0 through P5 are ALL PASS.", flush=True)


def build_and_push_kaggle():
    print("=== Dispatching V2 Run 05.2 to Kaggle ===", flush=True)
    verify_preflight_gates()

    bundle_dir = _REPO_ROOT / "artifacts" / "kaggle_bundles" / "v2_run05_2"
    app_dir = bundle_dir / "app_staging"
    if app_dir.exists():
        shutil.rmtree(app_dir, ignore_errors=True)
    app_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy Source Tree & Executor & Evaluator
    shutil.copytree(_REPO_ROOT / "src", app_dir / "src", dirs_exist_ok=True)
    shutil.copy(_REPO_ROOT / "scripts" / "kaggle_v2_run05_2_executor.py", app_dir / "executor.py")
    shutil.copy(_REPO_ROOT / "scripts" / "evaluate_v2_run05_2_all_arms.py", app_dir / "evaluate_v2_run05_2_all_arms.py")

    # 2. Copy Extra Data & Preflight Audit Files
    extra_data_dir = app_dir / "data"
    extra_data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(_REPO_ROOT / "data" / "tempun" / "train_nwp.jsonl", extra_data_dir / "train_nwp.jsonl")

    # Copy audit & contract artifacts from run_05_2
    for audit_f in _RUN_DIR.glob("*.json*"):
        if audit_f.is_file():
            shutil.copy(audit_f, extra_data_dir / audit_f.name)
    for proto_f in _RUN_DIR.glob("*.yaml"):
        if proto_f.is_file():
            shutil.copy(proto_f, extra_data_dir / proto_f.name)

    # 3. Create light payload zip
    zip_path = bundle_dir / "light_payload.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in app_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(app_dir))

    zip_size_kb = zip_path.stat().st_size / 1024
    print(f"Created light payload zip: {zip_size_kb:.2f} KB", flush=True)

    # 4. Base64-encode light payload
    payload_bytes = zip_path.read_bytes()
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    b64_str = base64.b64encode(payload_bytes).decode("ascii")

    chunk_size = 76
    chunks = "\n".join(f'    "{b64_str[i:i + chunk_size]}"' for i in range(0, len(b64_str), chunk_size))

    # 5. Write Kaggle Driver Script
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

print(f"=== STARTING LRR V2 RUN 05.2 MECHANISM REPAIR GPU EXECUTION ===", flush=True)
c1_candidates = list(Path("/kaggle/input").rglob("C1_COMPACT_ADAPTER_ONLY.pt"))
if not c1_candidates:
    all_files = [str(p) for p in Path("/kaggle/input").rglob("*")]
    print(f"All files in /kaggle/input: {{all_files}}", flush=True)
    raise FileNotFoundError("Missing C1 checkpoint in /kaggle/input")

c1_ckpt = c1_candidates[0]
INPUT_DATA_DIR = c1_ckpt.parent
print(f"Found C1 checkpoint at: {{c1_ckpt}} ({{c1_ckpt.stat().st_size / (1024*1024):.2f}} MB)", flush=True)
print(f"Using INPUT_DATA_DIR: {{INPUT_DATA_DIR}}", flush=True)

zip_p = ROOT / "payload.zip"
zip_p.write_bytes(base64.b64decode(PAYLOAD_B64))

actual_sha = hashlib.sha256(zip_p.read_bytes()).hexdigest()
if actual_sha != PAYLOAD_SHA:
    raise RuntimeError(f"Payload SHA256 mismatch: expected {{PAYLOAD_SHA}}, got {{actual_sha}}")

app_dir = ROOT / "v2_run05_2_app"
app_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_p) as zf:
    zf.extractall(app_dir)
zip_p.unlink()

print("Installing required Python dependencies...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "transformers", "accelerate"], check=True)

print("Starting V2 Run 05.2 GPU Executor on CUDA...", flush=True)
out_artifacts = ROOT / "artifacts" / "run_05_2"
out_artifacts.mkdir(parents=True, exist_ok=True)

extra_data_dir = app_dir / "data"

cmd = [
    sys.executable, str(app_dir / "executor.py"),
    "--data_dir", str(INPUT_DATA_DIR),
    "--extra_data_dir", str(extra_data_dir),
    "--c1_checkpoint", str(c1_ckpt),
    "--output_dir", str(out_artifacts),
]

sub_env = dict(os.environ)
sub_env["PYTHONPATH"] = str(app_dir / "src") + (":" + sub_env["PYTHONPATH"] if "PYTHONPATH" in sub_env else "")

proc = subprocess.run(cmd, env=sub_env, text=True)
if proc.returncode != 0:
    print(f"ERROR: Executor failed with exit code {{proc.returncode}}", file=sys.stderr, flush=True)
    sys.exit(proc.returncode)

print("=== V2 RUN 05.2 KAGGLE GPU EXECUTION FINISHED SUCCESSFULLY ===", flush=True)
'''

    driver_path = dispatch_dir / "kernel_run05_2.py"
    driver_path.write_text(driver_code, encoding="utf-8")

    # 6. Write kernel-metadata.json
    slug = "lrr-v2-run05-2-mechanism-repair"
    meta = {
        "id": f"qh0205/{slug}",
        "title": "LRR V2 Run05.2 Mechanism Repair",
        "code_file": "kernel_run05_2.py",
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

    # 7. Push to Kaggle using KaggleApi with retry
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    for attempt in range(1, 4):
        try:
            print(f"Pushing kernel to Kaggle via KaggleApi: qh0205/{slug} (attempt {attempt}/3)...", flush=True)
            push_res = api.kernels_push(str(dispatch_dir))
            print(f"Push Result: {push_res}", flush=True)
            print(f"Successfully dispatched kernel qh0205/{slug} to Kaggle with GPU acceleration!", flush=True)
            break
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}", flush=True)
            if attempt == 3:
                raise
            import time
            time.sleep(5)


if __name__ == "__main__":
    build_and_push_kaggle()
