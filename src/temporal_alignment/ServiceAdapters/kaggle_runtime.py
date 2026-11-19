"""Request-driven Kaggle GPU execution with provenance and privacy gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "kaggle-gpu-request-v1"
EXIT_INVALID = 2
SLUG = re.compile(r"^[a-z0-9_-]+/[a-z0-9_-]+$")
SENSITIVE_KEY = re.compile(r"(?:token|secret|password|credential|api_key)", re.IGNORECASE)
SENSITIVE_NAME = re.compile(
    r"(?:^|[._-])(?:kaggle|credential|secret|token|private[_-]?key)(?:[._-]|$)",
    re.IGNORECASE,
)


def contract_payload() -> dict[str, Any]:
    """Describe the optional adapter without importing the Kaggle SDK."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "OPTIONAL_DISABLED_BY_DEFAULT",
        "commands": ["contract", "run --request"],
        "operations": ["preflight", "execute"],
        "privacy_class": "public_only",
        "credential_source": "KAGGLE_CONFIG_DIR or ~/.kaggle; never request JSON",
        "default_network": False,
        "quality_claim_allowed": False,
    }


def run_request(request_path: Path) -> dict[str, Any]:
    """Validate, bundle and optionally execute one approved Kaggle request."""
    payload = _load_request(request_path)
    root = Path(str(payload.get("project_root", "."))).resolve()
    report_path = _inside(root, Path(str(payload["report"])))
    try:
        plan = _validate_plan(payload, root, request_path)
        bundle_manifest = _build_bundle(plan)
        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "operation": plan["operation"],
            "evidence_class": "contract_preflight_only",
            "quality_claim_allowed": False,
            "request_sha256": _sha256(request_path),
            "kernel_slug": plan["kernel_slug"],
            "bundle_manifest": bundle_manifest,
            "network_executed": False,
            "outputs": [],
        }
        if plan["operation"] == "execute":
            report.update(_execute(plan))
            report["evidence_class"] = "external_compute_execution_only"
        _write_json(report_path, report)
        return report
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "FAIL",
            "operation": str(payload.get("operation", "<missing>")),
            "quality_claim_allowed": False,
            "request_sha256": _sha256(request_path),
            "error": str(exc),
        }
        _write_json(report_path, failure)
        raise


def _load_request(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Request must be a JSON object.")
    _reject_secret_fields(value)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}.")
    if "report" not in value:
        raise ValueError("Request requires report.")
    return value


def _reject_secret_fields(value: object, prefix: str = "request") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if SENSITIVE_KEY.search(str(key)):
                raise ValueError(f"Secret-like field is forbidden: {prefix}.{key}.")
            _reject_secret_fields(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secret_fields(child, f"{prefix}[{index}]")


def _validate_plan(payload: dict[str, Any], root: Path, request_path: Path) -> dict[str, Any]:
    operation = str(payload.get("operation", "preflight"))
    if operation not in {"preflight", "execute"}:
        raise ValueError("operation must be preflight or execute.")
    if payload.get("external_compute_approved") is not True:
        raise ValueError("external_compute_approved must be true.")
    if payload.get("privacy_class") != "public":
        raise ValueError("Only privacy_class=public is accepted.")
    kernel_slug = str(payload.get("kernel_slug", ""))
    if not SLUG.fullmatch(kernel_slug):
        raise ValueError("kernel_slug must be owner/slug using lowercase safe characters.")
    worker = _inside(root, Path(str(payload.get("worker", ""))))
    if not worker.is_file() or worker.suffix.casefold() != ".py":
        raise ValueError("worker must be an existing Python file inside project_root.")
    if _sha256(worker) != str(payload.get("worker_sha256", "")):
        raise ValueError("worker_sha256 mismatch.")

    run_root = _inside(root, Path(str(payload.get("run_root", "artifacts/kaggle-run"))))
    report = _inside(root, Path(str(payload["report"])))
    inputs = _validate_inputs(payload.get("inputs", []), root)
    total_bytes = worker.stat().st_size + sum(item["path"].stat().st_size for item in inputs)
    limit = int(payload.get("max_upload_bytes", 100_000_000))
    if total_bytes > limit:
        raise ValueError(f"Bundle size {total_bytes} exceeds max_upload_bytes={limit}.")
    expected = payload.get("expected_outputs", [])
    if not isinstance(expected, list) or not expected:
        raise ValueError("expected_outputs must be a non-empty list.")
    expected_names = [_safe_name(str(item), "expected output") for item in expected]
    if bool(payload.get("enable_internet", False)) and payload.get("network_approved") is not True:
        raise ValueError("network_approved=true is required when enable_internet=true.")
    return {
        "operation": operation,
        "project_root": root,
        "request_path": request_path.resolve(),
        "run_root": run_root,
        "report": report,
        "worker": worker,
        "inputs": inputs,
        "kernel_slug": kernel_slug,
        "title": str(payload.get("title", kernel_slug.split("/", 1)[1])),
        "expected_outputs": expected_names,
        "enable_gpu": bool(payload.get("enable_gpu", True)),
        "enable_internet": bool(payload.get("enable_internet", False)),
        "poll_seconds": max(10, int(payload.get("poll_seconds", 30))),
        "timeout_seconds": max(60, int(payload.get("timeout_seconds", 7200))),
    }


def _validate_inputs(value: object, root: Path) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("inputs must be a list.")
    result: list[dict[str, Any]] = []
    destinations: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"inputs[{index}] must be an object.")
        if raw.get("upload_approved") is not True:
            raise ValueError(f"inputs[{index}].upload_approved must be true.")
        path = _inside(root, Path(str(raw.get("path", ""))))
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"inputs[{index}].path must be a regular file.")
        if SENSITIVE_NAME.search(path.name) or ".git" in path.parts:
            raise ValueError(f"Sensitive or repository-internal input is forbidden: {path.name}.")
        expected_hash = str(raw.get("sha256", ""))
        if _sha256(path) != expected_hash:
            raise ValueError(f"inputs[{index}].sha256 mismatch.")
        destination = _safe_name(str(raw.get("destination", path.name)), "input destination")
        if destination in destinations:
            raise ValueError(f"Duplicate input destination: {destination}.")
        destinations.add(destination)
        result.append({"path": path, "destination": destination, "sha256": expected_hash})
    return result


def _build_bundle(plan: dict[str, Any]) -> dict[str, Any]:
    bundle = plan["run_root"] / "bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "inputs").mkdir(parents=True)
    shutil.copy2(plan["worker"], bundle / "main.py")
    input_rows: list[dict[str, Any]] = []
    for item in plan["inputs"]:
        target = bundle / "inputs" / item["destination"]
        shutil.copy2(item["path"], target)
        input_rows.append(
            {
                "path": f"inputs/{item['destination']}",
                "sha256": item["sha256"],
                "bytes": target.stat().st_size,
            }
        )
    metadata = {
        "id": plan["kernel_slug"],
        "title": plan["title"],
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": str(plan["enable_gpu"]).lower(),
        "enable_internet": str(plan["enable_internet"]).lower(),
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    _write_json(bundle / "kernel-metadata.json", metadata)
    _write_json(bundle / "input-manifest.json", {"inputs": input_rows})
    return {
        "path": str(bundle),
        "main_sha256": _sha256(bundle / "main.py"),
        "metadata_sha256": _sha256(bundle / "kernel-metadata.json"),
        "inputs": input_rows,
    }


def _execute(plan: dict[str, Any]) -> dict[str, Any]:
    executable = shutil.which("kaggle")
    if executable is None:
        raise RuntimeError("Kaggle CLI is not installed or not on PATH.")
    credential = _credential_path()
    if not credential.is_file():
        raise RuntimeError("Kaggle credential file is missing from the configured directory.")
    bundle = plan["run_root"] / "bundle"
    _command([executable, "kernels", "push", "-p", str(bundle)])
    started = time.monotonic()
    final_status = ""
    while time.monotonic() - started <= plan["timeout_seconds"]:
        status_text = _command([executable, "kernels", "status", plan["kernel_slug"]])
        final_status = status_text.strip()
        normalized = final_status.casefold()
        if any(term in normalized for term in ("complete", "error", "cancel")):
            break
        time.sleep(plan["poll_seconds"])
    if "complete" not in final_status.casefold():
        raise RuntimeError(f"Kaggle kernel did not complete successfully: {final_status}")
    output_dir = plan["run_root"] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    _command(
        [executable, "kernels", "output", plan["kernel_slug"], "-p", str(output_dir), "--force"]
    )
    outputs: list[dict[str, Any]] = []
    for name in plan["expected_outputs"]:
        path = output_dir / name
        if not path.is_file():
            raise RuntimeError(f"Expected Kaggle output is missing: {name}.")
        outputs.append({"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size})
    return {
        "network_executed": True,
        "kaggle_cli": _command([executable, "--version"]).strip(),
        "final_status": final_status,
        "outputs": outputs,
    }


def _credential_path() -> Path:
    configured = os.environ.get("KAGGLE_CONFIG_DIR")
    base = Path(configured).expanduser() if configured else Path.home() / ".kaggle"
    return base / "kaggle.json"


def _command(arguments: Sequence[str]) -> str:
    result = subprocess.run(
        list(arguments),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Command failed with exit {result.returncode}: {message}")
    return result.stdout


def _safe_name(value: str, label: str) -> str:
    path = Path(value)
    if not value or path.name != value or value in {".", ".."}:
        raise ValueError(f"{label} must be a plain file name.")
    return value


def _inside(root: Path, value: Path) -> Path:
    path = value if value.is_absolute() else root / value
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes project_root: {value}.")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Expose a compact request-driven CLI."""
    parser = argparse.ArgumentParser(description="Optional governed Kaggle GPU adapter.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("contract")
    run = commands.add_parser("run")
    run.add_argument("--request", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect the contract or execute one request."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "contract":
            print(json.dumps(contract_payload(), ensure_ascii=False, indent=2))
            return 0
        report = run_request(args.request)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"KAGGLE_ADAPTER_ERROR: {exc}", file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
