"""Package executable TickTack-inspired 0.5B Baseline Kaggle bundle.

Features:
- Encapsulates native PyTorch TickTack model, dataset, loss, evaluator, and train runner.
- Packages data/train_nwp.jsonl (2,461 samples) and data/eval_mcqa.jsonl (180 samples).
- Supports both --canary (quick representative run) and full 3-seed training.
- Independent Kaggle slug: qh0205/lrr-ticktack05b-canary or qh0205/lrr-ticktack05b-3seed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

STAGES = ("year:1947", "year:1984", "year:2022")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_line(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def kernel_source(payload: bytes, seeds: List[int], is_canary: bool = False) -> str:
    payload_sha = digest(payload)
    encoded = base64.b64encode(payload).decode("ascii")
    chunks = "\n".join(f'    "{encoded[i:i + 76]}"' for i in range(0, len(encoded), 76))
    seeds_repr = json.dumps(seeds)
    canary_line = '    cmd.append("--canary")\n' if is_canary else ''

    return f'''from __future__ import annotations
import base64, hashlib, json, os, shutil, subprocess, sys, traceback, zipfile
from pathlib import Path

ROOT = Path("/kaggle/working")
PAYLOAD_SHA = "{payload_sha}"
PAYLOAD_B64 = (
{chunks}
)

path = ROOT / ".ticktack_payload.zip"
path.write_bytes(base64.b64decode(PAYLOAD_B64))
if hashlib.sha256(path.read_bytes()).hexdigest() != PAYLOAD_SHA:
    raise RuntimeError("BUNDLE_SHA256_MISMATCH")

with zipfile.ZipFile(path) as archive:
    for item in archive.infolist():
        name = Path(item.filename)
        if name.is_absolute() or ".." in name.parts:
            raise RuntimeError("UNSAFE_BUNDLE_PATH")
    archive.extractall(ROOT)
path.unlink()

bundle = ROOT / "ticktack_bundle"
output_dir = ROOT / "artifacts/ticktack05b"
output_dir.mkdir(parents=True, exist_ok=True)

try:
    os.environ["HF_HOME"] = "/kaggle/temp/ticktack_hf_cache"
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "--disable-pip-version-check", "transformers", "accelerate"], check=True)

    sys.path.insert(0, str(bundle / "SourceCode"))
    target_seeds = {seeds_repr}
    sub_env = dict(os.environ)
    sub_env["PYTHONPATH"] = str(bundle / "SourceCode") + (":" + sub_env["PYTHONPATH"] if "PYTHONPATH" in sub_env else "")

    cmd = [
        sys.executable, str(bundle / "SourceCode/scripts/run_ticktack05b_baseline.py"),
        "--data-train", str(bundle / "data/train_nwp.jsonl"),
        "--data-eval", str(bundle / "data/eval_mcqa.jsonl"),
        "--output", str(output_dir),
        "--seeds"
    ] + [str(s) for s in target_seeds]

{canary_line}    proc = subprocess.run(cmd, env=sub_env, capture_output=True, text=True)
    (output_dir / "execution.log").write_text(f"EXIT: {{proc.returncode}}\\nSTDOUT:\\n{{proc.stdout}}\\nSTDERR:\\n{{proc.stderr}}", encoding="utf-8")
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        sys.exit(proc.returncode)

except BaseException as error:
    traceback.print_exc()
    raise
finally:
    shutil.rmtree(bundle, ignore_errors=True)
'''


def build_ticktack_bundle(
    root: Path,
    output_zip: Path,
    seeds: List[int],
    is_canary: bool = False,
) -> Dict[str, Any]:
    root = root.resolve()
    train_path = root / "scratch/test_bundle/lrr_bundle/data/train_nwp.jsonl"
    eval_path = root / "data/tempun/eval_mcqa.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train data: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval data: {eval_path}")

    files: Dict[str, bytes] = {
        "ticktack_bundle/data/train_nwp.jsonl": train_path.read_bytes(),
        "ticktack_bundle/data/eval_mcqa.jsonl": eval_path.read_bytes(),
    }

    # Code files
    pkg_base = root / "src/temporal_alignment/baselines/ticktack05b"
    for py_file in pkg_base.glob("*.py"):
        rel = f"ticktack_bundle/SourceCode/temporal_alignment/baselines/ticktack05b/{py_file.name}"
        files[rel] = py_file.read_bytes()

    # Package inits and runner script
    files["ticktack_bundle/SourceCode/temporal_alignment/__init__.py"] = b'"""temporal_alignment root package."""\n'
    files["ticktack_bundle/SourceCode/temporal_alignment/baselines/__init__.py"] = b'"""Baselines package."""\n'

    runner_script = root / "scripts/run_ticktack05b_baseline.py"
    files["ticktack_bundle/SourceCode/scripts/run_ticktack05b_baseline.py"] = runner_script.read_bytes()

    # Create zip payload in memory
    import io
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fpath, data in sorted(files.items()):
            z.writestr(fpath, data)
    payload_bytes = bio.getvalue()

    kernel_py = kernel_source(payload_bytes, seeds=seeds, is_canary=is_canary)

    slug = "lrr-ticktack05b-canary" if is_canary else "lrr-ticktack05b-3seed"
    title = "LRR TickTack05B Canary" if is_canary else "LRR TickTack05B 3Seed"

    metadata = {
        "id": f"qh0205/{slug}",
        "title": title,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }

    output_dir = output_zip.parent / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "kernel.py").write_text(kernel_py, encoding="utf-8")
    (output_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "slug": f"qh0205/{slug}",
        "directory": str(output_dir),
        "payload_sha256": digest(payload_bytes),
        "payload_size_bytes": len(payload_bytes),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    active_seeds = [1] if args.canary else args.seeds
    info = build_ticktack_bundle(
        root=Path("."),
        output_zip=Path("scratch/ticktack_bundle.zip"),
        seeds=active_seeds,
        is_canary=args.canary,
    )
    print("Bundle created successfully:")
    print(json.dumps(info, indent=2))
