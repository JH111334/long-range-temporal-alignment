"""Validate four live course repositories and append immutable V5 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from course_repo_catalog import PROJECTS, CourseProject

VALIDATION_ID = "course-streamlit-lineage-live-v1"


class ValidationLogError(RuntimeError):
    """Reject incomplete evidence or unsafe log targets."""


@dataclass(frozen=True, slots=True)
class PlannedLog:
    """Hold one new immutable checkpoint."""

    repo: Path
    target: Path
    content: str


def file_sha256(path: Path) -> str:
    """Hash exact file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_checked(repo: Path, command: list[str]) -> str:
    """Run one offline validation command and fail on non-zero exit."""
    result = subprocess.run(
        command,
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        details = (result.stdout + "\n" + result.stderr).strip()
        raise ValidationLogError(
            f"{repo.name}: command failed ({result.returncode}): {command}\n{details}"
        )
    return result.stdout.strip()


def latest_line(output: str) -> str:
    """Return the last non-empty command-output line."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else "exit 0; no stdout"


def atomic_create(path: Path, text: str) -> None:
    """Create one file atomically without overwriting an existing log."""
    if path.exists():
        raise ValidationLogError(f"immutable target already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise ValidationLogError(f"immutable target appeared during write: {path}")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def log_pattern(spec: CourseProject) -> re.Pattern[str]:
    """Return the checkpoint grammar for one repository."""
    return re.compile(
        rf"^{re.escape(spec.repo_name)}_Checkpoints_"
        r"(\d{2}-\d{2}-\d{2})_V([1-9]\d*)\.md$"
    )


def existing_validation_log(repo: Path) -> Path | None:
    """Find at most one checkpoint created by this validation."""
    matches = []
    for path in (repo / "Documents" / "General").glob("*.md"):
        if path.is_file() and f"validation_id: {VALIDATION_ID}" in path.read_text(encoding="utf-8"):
            matches.append(path)
    if len(matches) > 1:
        raise ValidationLogError(f"{repo.name}: duplicated validation_id")
    return matches[0] if matches else None


def next_target(repo: Path, spec: CourseProject, date_text: str) -> Path:
    """Allocate the next sequential checkpoint version for today."""
    versions: list[int] = []
    pattern = log_pattern(spec)
    for path in (repo / "Documents" / "General").glob("*.md"):
        match = pattern.fullmatch(path.name)
        if match and match.group(1) == date_text:
            versions.append(int(match.group(2)))
    version = max(versions, default=0) + 1
    return (
        repo / "Documents" / "General" / f"{spec.repo_name}_Checkpoints_{date_text}_V{version}.md"
    )


def project_package(spec: CourseProject) -> str:
    """Return the source package recorded in the project catalog."""
    return spec.package_name


def build_plan(root: Path, spec: CourseProject) -> PlannedLog | None:
    """Run live checks and prepare one evidence-bound checkpoint."""
    repo = (root / spec.repo_name).resolve()
    if not repo.is_dir():
        raise ValidationLogError(f"missing repository: {repo}")
    existing = existing_validation_log(repo)
    if existing is not None:
        print(f"NOOP {spec.repo_name}: {existing.name}")
        return None

    package = project_package(spec)
    artifacts = (
        "scripts/validate_project.py",
        f"src/{package}/ui/view_models.py",
        "streamlit_app/app.py",
        "tests/test_ui_lineage.py",
        "tests/test_streamlit_app.py",
    )
    missing = [relative for relative in artifacts if not (repo / relative).is_file()]
    if missing:
        raise ValidationLogError(f"{spec.repo_name}: missing artifacts: {missing}")

    validator_output = run_checked(
        repo, [sys.executable, "-X", "utf8", "-B", "scripts/validate_project.py"]
    )
    pytest_output = run_checked(
        repo,
        [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
    )
    diff_output = run_checked(repo, ["git", "diff", "--check"])
    if "15 passed" not in pytest_output:
        raise ValidationLogError(
            f"{spec.repo_name}: expected 15 passing tests, got {latest_line(pytest_output)}"
        )

    revision = latest_line(run_checked(repo, ["git", "rev-parse", "HEAD"]))
    worktree = run_checked(repo, ["git", "status", "--short"])
    worktree_count = len([line for line in worktree.splitlines() if line.strip()])
    now = datetime.now().astimezone()
    date_text = now.strftime("%d-%m-%y")
    target = next_target(repo, spec, date_text)
    version_match = log_pattern(spec).fullmatch(target.name)
    if version_match is None:
        raise ValidationLogError(f"invalid allocated target: {target}")
    version = f"V{version_match.group(2)}"
    artifact_rows = "\n".join(
        f"- {relative}: {file_sha256(repo / relative)}" for relative in artifacts
    )
    diff_result = "PASS; no whitespace errors" if not diff_output else latest_line(diff_output)
    content = f"""# Checkpoint — kiểm chứng Streamlit lineage trên repo thật

## Định danh

- validation_id: {VALIDATION_ID}
- Project: {spec.repo_name}
- Thời điểm: {now.isoformat(timespec="seconds")}
- Phiên bản: {version}
- Evidence class: offline_unit_test + Streamlit_AppTest + static_validator; không phải benchmark.
- Git revision: {revision}
- Worktree: dirty ({worktree_count} entries) do thay đổi local thuộc yêu cầu; chưa commit/push.

## Cấu hình hệ thống

- Windows 11; Python {sys.version.split()[0]}; NVIDIA GeForce RTX 2050 4096 MiB.
- Intel Core i5-12450H; RAM 16,866,684,928 bytes.
- Dataset/model/network/install/training/inference và peak VRAM theo model: not-run.

## Lệnh và cấu hình chạy

- Python validator: {latest_line(validator_output)}
- Pytest gồm unit tests và Streamlit AppTest: {latest_line(pytest_output)}
- git diff --check: {diff_result}
- Cache provider của pytest: disabled; external services: disabled.

## Kết quả

- PASS: dashboard tự chọn checkpoint mới nhất theo ngày/version.
- PASS: body Project/version, evidence class, SHA-256 và warnings được kiểm tra fail-closed.
- PASS: Streamlit thêm repo src vào import path, chỉ đọc và không có action button.
- Không có kết quả dataset/model quality, benchmark hoặc production readiness.

## Cảnh báo

- Test dùng fixture/scaffold offline; chưa chứng minh dữ liệu thật hoặc chất lượng truy xuất.
- Mọi inference ngoài máy vẫn phải qua registry, quyền sử dụng, immutable revision và privacy gate.

## Artifact và lineage

{artifact_rows}

## Việc tiếp theo

Chỉ mở data gate sau khi data dossier, license/terms, split leakage audit và baseline CPU được duyệt.
Không sửa checkpoint này; nếu có lần chạy mới, tạo version kế tiếp theo RULE.md.
"""
    return PlannedLog(repo, target, content)


def parse_args() -> argparse.Namespace:
    """Parse a bounded root and explicit mutation flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Validate all repositories before creating any immutable log."""
    args = parse_args()
    root = args.root.resolve()
    plans = tuple(plan for spec in PROJECTS if (plan := build_plan(root, spec)) is not None)
    for plan in plans:
        print(f"PLAN {plan.repo.name}: create {plan.target.name}")
    if not args.apply:
        print("DRY-RUN: validations ran; no logs written")
        return 0
    created: list[Path] = []
    try:
        for plan in plans:
            atomic_create(plan.target, plan.content)
            created.append(plan.target)
    except BaseException:
        for path in reversed(created):
            if path.is_file() and f"validation_id: {VALIDATION_ID}" in path.read_text(
                encoding="utf-8"
            ):
                path.unlink()
        raise
    print(f"APPLIED: {len(created)} immutable checkpoint(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
