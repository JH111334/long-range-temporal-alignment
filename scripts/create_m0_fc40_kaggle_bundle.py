"""Package executable M0-FC40 Dual-Cycle MaT-LoRA 3-Seed Kaggle dispatch bundle.

Specification:
- Model Backbone: Qwen/Qwen2.5-0.5B-Instruct (commit 7ae557604adf67be50417f59c2c2f167def9a775)
- Adapter: Periodic Dual-Cycle MaT-LoRA (r=8, alpha=16.0, K=2, F_base=I_8) on 72 MLP modules
- Representation: FIXED_FONE_CANCHI_V1 (FoNE 32D + CanChi 8D = 40D concat)
- Conditioning Seam: Gated Residual Projection at token index 14 (alpha_0=0.0)
- Train: TempUN Small train_nwp.jsonl (2,461 samples) across year:1947, year:1984, year:2022
- Eval: TempUN Small eval_mcqa.jsonl (180 samples)
- Seeds: [1, 2, 3]
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Dict


STAGES = ("year:1947", "year:1984", "year:2022")
EXPECTED_COUNTS = {"year:1947": 733, "year:1984": 1338, "year:2022": 390}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_line(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def read_gzip_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def kernel_source(payload: bytes, seeds: List[int], run_id: str, is_canary: bool = False) -> str:
    payload_sha = digest(payload)
    encoded = base64.b64encode(payload).decode("ascii")
    chunks = "\n".join(f'    "{encoded[i:i + 76]}"' for i in range(0, len(encoded), 76))
    seeds_repr = json.dumps(seeds)
    canary_flag = '            "--canary",\n' if is_canary else ''
    log_dir_name = "M0_FC40_CANARY" if is_canary else "M0_FC40_3SEED"

    return f'''from __future__ import annotations
import base64, hashlib, json, os, runpy, shutil, subprocess, sys, traceback, zipfile
from pathlib import Path

ROOT = Path("/kaggle/working")
PAYLOAD_SHA = "{payload_sha}"
PAYLOAD_B64 = (
{chunks}
)

path = ROOT / ".m0_fc40_payload.zip"
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

bundle = ROOT / "lrr_bundle"
failure = ROOT / "artifacts/diagnostics/{run_id}/dispatch_failure.json"
stage = "dependency_install"

try:
    if sys.version_info < (3, 10):
        raise RuntimeError("PYTHON_VERSION_UNSUPPORTED")
    os.environ["HF_HOME"] = "/kaggle/temp/m0_fc40_hf_cache"
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "--disable-pip-version-check", "-r",
                    str(bundle / "requirements-kaggle.txt")], check=True)
    stage = "M0_FC40_EXECUTION"
    sys.path.insert(0, str(ROOT / "SourceCode"))
    target_seeds = {seeds_repr}
    sub_env = dict(os.environ)
    sub_env["PYTHONPATH"] = str(ROOT / "SourceCode") + (":" + sub_env["PYTHONPATH"] if "PYTHONPATH" in sub_env else "")

    for s in target_seeds:
        print(f"=== STARTING M0-FC40 EXECUTION FOR SEED {{s}} ===")
        proc = subprocess.run([
            sys.executable, str(ROOT / "SourceCode/run_m0_fc40.py"),
            "--bundle", str(bundle),
            "--output", str(ROOT / "artifacts"),
            "--seed", str(s),
{canary_flag}            "--resume"
        ], env=sub_env, capture_output=True, text=True)

        log_dir = ROOT / "artifacts/diagnostics/{log_dir_name}"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"seed_{{s}}_exec.log").write_text(
            f"EXIT_CODE: {{proc.returncode}}\\n\\nSTDOUT:\\n{{proc.stdout}}\\n\\nSTDERR:\\n{{proc.stderr}}\\n",
            encoding="utf-8"
        )
        print(f"--- SEED {{s}} RETURNED CODE {{proc.returncode}} ---")
        if proc.stdout:
            print("STDOUT:\\n" + proc.stdout)
        if proc.returncode != 0:
            if proc.stderr:
                print("STDERR:\\n" + proc.stderr, file=sys.stderr)
            raise RuntimeError(f"Seed {{s}} failed with exit code {{proc.returncode}}:\\n{{proc.stderr[-3000:]}}")
        print(f"=== COMPLETED M0-FC40 EXECUTION FOR SEED {{s}} ===")
except BaseException as error:
    failure.parent.mkdir(parents=True, exist_ok=True)
    failure.write_text(json.dumps({{
        "stage": stage,
        "exception_type": type(error).__name__,
        "message": str(error)[:3500],
        "payload_sha256": PAYLOAD_SHA
    }}, ensure_ascii=False, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
    traceback.print_exc()
    raise
finally:
    shutil.rmtree(ROOT / "lrr_bundle", ignore_errors=True)
    shutil.rmtree(ROOT / "SourceCode", ignore_errors=True)
'''


def build_m0_fc40_bundle(
    root: Path,
    output: Path,
    seeds: List[int] | None = None,
    slug: str | None = None,
    title: str | None = None,
    is_canary: bool = False,
) -> Dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    partner_root = (root.parent / "Long-Range-Representation").resolve()
    data_dir = partner_root / "Data/experiment_views/canchi_fone/TEMPUN_CANCHI_FONE_SMALL_V1"

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    active_seeds = [1] if is_canary else (seeds if seeds is not None else [1, 2, 3])

    # Verify data counts
    train_rows = read_gzip_jsonl(data_dir / "records/train_nwp.jsonl.gz")
    eval_rows = read_gzip_jsonl(data_dir / "records/eval_mcqa.jsonl.gz")

    if len(train_rows) != 2461:
        raise ValueError(f"Expected 2461 train rows, got {len(train_rows)}")
    if len(eval_rows) != 180:
        raise ValueError(f"Expected 180 eval rows, got {len(eval_rows)}")

    counts = {stage: sum(r["stage_id"] == stage for r in train_rows) for stage in STAGES}
    for stage, expected in EXPECTED_COUNTS.items():
        if counts[stage] != expected:
            raise ValueError(f"Stage {stage} count {counts[stage]} != expected {expected}")

    config = {
        "run_id": "M0_FC40_PILOT_V1_3SEED",
        "hypothesis_id": "TEMPORAL_LEARNING_M0_FC40_V1",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "adapter_type": "Periodic_Dual_Cycle_MaT_LoRA",
        "rank": 8,
        "alpha": 16.0,
        "num_harmonics": 2,
        "representation": "FIXED_FONE_CANCHI_V1",
        "feature_dim": 40,
        "conditioning_seam": {
            "type": "GATED_RESIDUAL_PREFILL",
            "injection_token_index": 14,
            "initial_alpha": 0.0,
            "projection": "Linear(40, 896) + LayerNorm(896, bias=False)",
        },
        "target_modules": ["gate_proj", "up_proj", "down_proj"],
        "num_target_modules": 72,
        "loss": "MASKED_TARGET",
        "stages": list(STAGES),
        "seeds": active_seeds,
    }

    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        revision = "LOCAL_DIRTY_HEAD"

    files: Dict[str, bytes] = {
        "lrr_bundle/config.json": json_line(config),
        "lrr_bundle/data/train_nwp.jsonl": b"".join(json_line(r) for r in train_rows),
        "lrr_bundle/data/eval_mcqa.jsonl": b"".join(json_line(r) for r in eval_rows),
        "lrr_bundle/requirements-kaggle.txt": b"transformers\npeft\naccelerate\nbitsandbytes==0.50.2\n",
    }

    # Code files to package
    src_models = root / "src/temporal_alignment/models"
    models_to_pack = [
        "__init__.py",
        "fone_canchi_40d.py",
        "dual_cycle_mat_lora.py",
        "gated_seam.py",
        "tempun_dataset.py",
        "run_m0_fc40.py",
    ]

    for fname in models_to_pack:
        fpath = src_models / fname
        if fpath.exists():
            content = fpath.read_bytes()
            files[f"SourceCode/temporal_alignment/models/{fname}"] = content
            if fname == "run_m0_fc40.py":
                files["SourceCode/run_m0_fc40.py"] = content

    # Add temporal_alignment __init__.py
    files["SourceCode/temporal_alignment/__init__.py"] = b'"""Temporal Alignment Framework."""\n'

    # Add temporal_alignment observability module
    src_obs = root / "src/temporal_alignment/observability"
    if src_obs.exists():
        for fname in ["__init__.py", "telemetry.py"]:
            fpath = src_obs / fname
            if fpath.exists():
                files[f"SourceCode/temporal_alignment/observability/{fname}"] = fpath.read_bytes()

    metadata = {
        "run_id": config["run_id"],
        "hypothesis_id": config["hypothesis_id"],
        "code_revision": revision,
        "training_counts": counts,
        "data_examples_train": len(train_rows),
        "data_examples_eval": len(eval_rows),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seeds": active_seeds,
        "is_canary": is_canary,
    }
    files["lrr_bundle/run_metadata.json"] = json_line(metadata)

    output.mkdir(parents=True, exist_ok=True)
    payload_path = output / "payload.zip"
    with zipfile.ZipFile(payload_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    kernel_dir = output / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    pilot_py = kernel_dir / "pilot.py"
    pilot_py.write_text(kernel_source(payload_path.read_bytes(), active_seeds, config["run_id"], is_canary=is_canary), encoding="utf-8", newline="\n")

    default_slug = "qh0205/lrr-fc105-m0-fc40-3seed"
    default_title = "LRR FC105 M0-FC40 3Seed"

    kernel_metadata = {
        "id": slug or default_slug,
        "title": title or default_title,
        "code_file": "pilot.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(kernel_metadata, indent=2) + "\n", encoding="utf-8")

    receipt = {
        "status": "PASS",
        "bundle_file": str(payload_path),
        "bundle_sha256": digest(payload_path.read_bytes()),
        "bundle_size_bytes": payload_path.stat().st_size,
        "staging_dir": str(kernel_dir),
        "kernel_slug": kernel_metadata["id"],
        "seeds": active_seeds,
        "training_examples": len(train_rows),
        "evaluation_examples": len(eval_rows),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    receipt_file = output / "M0_FC40_KAGGLE_BUNDLE_RECEIPT.json"
    receipt_file.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("artifacts/dispatch/M0_FC40"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--slug", type=str, default="qh0205/lrr-fc105-m0-fc40-3seed")
    parser.add_argument("--title", type=str, default="LRR FC105 M0-FC40 3Seed")
    parser.add_argument("--canary", action="store_true", help="Build bounded 15-step canary bundle")
    args = parser.parse_args()

    root = args.root.resolve()
    out = args.output if args.output.is_absolute() else root / args.output
    res = build_m0_fc40_bundle(
        root,
        out,
        seeds=args.seeds,
        slug=args.slug,
        title=("LRR FC105 M0-FC40 Canary" if args.canary else args.title),
        is_canary=args.canary,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
