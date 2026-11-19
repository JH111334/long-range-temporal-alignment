"""Plan or apply a reversible, offline Streamlit-lineage upgrade to four course repos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bootstrap_course_repositories import (
    streamlit_app,
    streamlit_app_test,
    ui_lineage_test,
    ui_module,
    validator_script,
)
from course_repo_catalog import PROJECTS, CourseProject

MIGRATION_ID = "course-streamlit-lineage-v1"
BACKUP_RELATIVE = Path(".migration-backups") / MIGRATION_ID


class MigrationError(RuntimeError):
    """Report a preflight, backup, write or validation failure."""


@dataclass(frozen=True, slots=True)
class FileChange:
    """Describe one exact before/after transition."""

    relative: str
    before: bytes | None
    after: bytes

    @property
    def action(self) -> str:
        """Return the user-facing action name."""
        return "create" if self.before is None else "update"


@dataclass(frozen=True, slots=True)
class RepositoryPlan:
    """Bundle one repository and its deterministic changes."""

    spec: CourseProject
    repo: Path
    desired: dict[str, str]
    changes: tuple[FileChange, ...]


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""
    return hashlib.sha256(payload).hexdigest()


def safe_target(repo: Path, relative: str) -> Path:
    """Resolve a repository-relative path and reject traversal."""
    target = (repo / relative).resolve()
    try:
        target.relative_to(repo.resolve())
    except ValueError as exc:
        raise MigrationError(f"path escapes repository: {relative}") from exc
    return target


def read_bytes(path: Path) -> bytes:
    """Read a file and preserve exact backup bytes."""
    try:
        return path.read_bytes()
    except OSError as exc:
        raise MigrationError(f"cannot read {path}: {exc}") from exc


def atomic_write(path: Path, payload: bytes) -> None:
    """Replace one file atomically after writing it beside the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def log_pattern(spec: CourseProject) -> re.Pattern[str]:
    """Return the repository-specific General log grammar."""
    return re.compile(
        rf"^{re.escape(spec.repo_name)}_(Checkpoints|Error)_(\d{{2}}-\d{{2}}-\d{{2}})_V([1-9]\d*)\.md$"
    )


def select_log(repo: Path, spec: CourseProject, kind: str) -> str:
    """Reuse this migration's log or allocate the next version for today."""
    general = repo / "Documents" / "General"
    pattern = log_pattern(spec)
    marker = f"migration_id: `{MIGRATION_ID}`"
    date_text = datetime.now().astimezone().strftime("%d-%m-%y")
    existing: list[Path] = []
    versions: list[int] = []
    for path in general.glob("*.md"):
        match = pattern.fullmatch(path.name)
        if match is None or match.group(1) != kind:
            continue
        text = read_bytes(path).decode("utf-8")
        if marker in text:
            existing.append(path)
        if match.group(2) == date_text:
            versions.append(int(match.group(3)))
    if len(existing) > 1:
        raise MigrationError(f"multiple {kind} logs contain {MIGRATION_ID} in {repo}")
    if existing:
        return existing[0].relative_to(repo).as_posix()
    version = max(versions, default=0) + 1
    name = f"{spec.repo_name}_{kind}_{date_text}_V{version}.md"
    return f"Documents/General/{name}"


def version_from_log(relative: str, spec: CourseProject) -> str:
    """Extract the canonical V-number from a selected log path."""
    match = log_pattern(spec).fullmatch(Path(relative).name)
    if match is None:
        raise MigrationError(f"selected log violates grammar: {relative}")
    return f"V{match.group(3)}"


def read_immutable_migration_log(repo: Path, relative: str) -> str | None:
    """Reuse an already-written migration log instead of changing its timestamp."""
    path = safe_target(repo, relative)
    if not path.is_file():
        return None
    text = read_bytes(path).decode("utf-8")
    marker = f"migration_id: `{MIGRATION_ID}`"
    if marker not in text:
        raise MigrationError(f"selected migration log lost its marker: {path}")
    return text


def render_error_log(spec: CourseProject, relative: str) -> str:
    """Record the measured analogue and confirmed stale course UI state."""
    version = version_from_log(relative, spec)
    match = log_pattern(spec).fullmatch(Path(relative).name)
    if match is None:
        raise MigrationError(f"selected error log violates grammar: {relative}")
    observed_on = datetime.strptime(match.group(2), "%d-%m-%y").date().isoformat()
    return f"""# Error — Streamlit import và lineage stale

## Định danh

- migration_id: `{MIGRATION_ID}`
- Project: `{spec.repo_name}`
- Thời điểm: `{observed_on}`; giờ chi tiết không thu thập, không suy đoán.
- Phiên bản: {version}
- Evidence class: `confirmed_static_course_audit + measured_temporal_AppTest`; không phải benchmark.

## Triệu chứng

- Expected: UI đọc checkpoint hợp lệ mới nhất và validator phát hiện lineage stale/sai contract.
- Actual: course app hard-code checkpoint V2 trong khi General đã có V3; validator cũ vẫn PASS.
- Temporal AppTest tương đồng đo thật `ModuleNotFoundError` vì `streamlit run` không tự thêm `src` vào `sys.path`.
- Streamlit 1.57 cảnh báo `use_container_width` đã quá hạn từ 2025-12-31.

## Cấu hình và đầu vào

- Scope course: source audit read-only trên `streamlit_app/app.py`, `ui/view_models.py`, validator và General.
- System profile: Windows 11; Python 3.12.10; RTX 2050 4096 MiB; i5-12450H; RAM 16,866,684,928 bytes.
- AppTest course trước migration: `not-run`; không nâng lỗi analogue thành kết quả chạy của bốn repo.
- Network/install/training/inference: `not-run`.

## Nguyên nhân

- `confirmed`: UI dùng literal V2 thay vì chọn latest theo grammar; view model chỉ có gate tĩnh.
- `confirmed by analogue`: app import package trước khi bootstrap `<repo>/src` vào `sys.path`.
- `confirmed API drift`: dùng tham số dataframe đã deprecated thay vì `width="stretch"`.

## Khắc phục hoặc rollback

- View model thuần Python chọn latest `(ngày, version)`, kiểm body version/evidence, tính SHA-256 và fail-closed.
- App chỉ đọc, bootstrap `ROOT/src`, dùng `width="stretch"`; backup nằm tại `{BACKUP_RELATIVE.as_posix()}`.

## Regression và trạng thái

- Unit tests lineage và Streamlit AppTest offline được thêm; migration helper chỉ static-validate.
- Sau apply phải chạy toàn bộ tests và validator; kết quả thật ghi checkpoint version mới, không sửa log này.
"""


def render_checkpoint_log(
    spec: CourseProject,
    relative: str,
    artifacts: dict[str, str],
) -> str:
    """Render an honest migration checkpoint with desired artifact hashes."""
    version = version_from_log(relative, spec)
    match = log_pattern(spec).fullmatch(Path(relative).name)
    if match is None:
        raise MigrationError(f"selected checkpoint violates grammar: {relative}")
    planned_on = datetime.strptime(match.group(2), "%d-%m-%y").date().isoformat()
    artifact_rows = "\n".join(
        f"- `{path}`: `{sha256_bytes(text.encode('utf-8'))}`"
        for path, text in sorted(artifacts.items())
    )
    return f"""# Checkpoint — dynamic read-only Streamlit lineage

## Định danh

- migration_id: `{MIGRATION_ID}`
- Project: `{spec.repo_name}`
- Ngày lập kế hoạch: `{planned_on}`
- Phiên bản: {version}
- Evidence class: `migration_checkpoint + static_check`; không phải benchmark.

## Cấu hình hệ thống

- Profile đã đo: NVIDIA GeForce RTX 2050 4096 MiB; Intel Core i5-12450H; RAM 16,866,684,928 bytes.
- Helper runtime: `{platform.platform()}`; Python `{platform.python_version()}`.
- Model-specific resource probe: `not-run`.

## Lệnh và cấu hình chạy

- `py -3.12 -B scripts/upgrade_course_streamlit_lineage.py --root <Documents> --apply`
- Network/install/Git mutation/data/model/training/inference: `disabled`.
- Unit tests, AppTest và repo validator: `not-run-by-migration`; chạy ngay sau apply.

## Kết quả

- Planned: latest checkpoint discovery đúng grammar, version/body parity, SHA-256, evidence class và warnings.
- Planned: Streamlit read-only có `ROOT/src` bootstrap và `width="stretch"`.
- Benchmark/model quality/production readiness: `none`.

## Cảnh báo

- Đây là static migration checkpoint, chưa phải kết quả AppTest của course repo.
- Nếu latest checkpoint sai grammar/UTF-8/version/evidence, UI hiển thị `invalid` và không fallback âm thầm.

## Artifact và lineage

{artifact_rows}

## Việc tiếp theo

Chạy unittest/pytest, Streamlit AppTest và validator; ghi exit code/test count thật vào checkpoint kế tiếp.
Rollback chỉ từ backup manifest và chỉ khi target chưa drift.
"""


def build_desired(repo: Path, spec: CourseProject) -> dict[str, str]:
    """Build deterministic source, tests, validator and immutable logs."""
    package = spec.package_name
    desired = {
        f"src/{package}/ui/view_models.py": ui_module(spec),
        "streamlit_app/app.py": streamlit_app(spec),
        "tests/test_ui_lineage.py": ui_lineage_test(spec),
        "tests/test_streamlit_app.py": streamlit_app_test(),
        "scripts/validate_project.py": validator_script(spec),
    }
    error_relative = select_log(repo, spec, "Error")
    desired[error_relative] = read_immutable_migration_log(
        repo, error_relative
    ) or render_error_log(spec, error_relative)
    checkpoint_relative = select_log(repo, spec, "Checkpoints")
    desired[checkpoint_relative] = read_immutable_migration_log(
        repo, checkpoint_relative
    ) or render_checkpoint_log(spec, checkpoint_relative, desired)
    return {path: text.rstrip() + "\n" for path, text in desired.items()}


def validate_desired(spec: CourseProject, desired: dict[str, str]) -> None:
    """Reject hard-coded lineage, invalid logs and Python syntax before writes."""
    app = desired["streamlit_app/app.py"]
    required_app = (
        "sys.path.insert",
        "load_dashboard",
        'width="stretch"',
        "dashboard.lineage.warnings",
    )
    for marker in required_app:
        if marker not in app:
            raise MigrationError(f"{spec.repo_name} app missing marker: {marker}")
    if re.search(r"Checkpoints_\d{2}-\d{2}-\d{2}_V[1-9]\d*", app):
        raise MigrationError(f"{spec.repo_name} app still hard-codes a checkpoint")
    view_path = f"src/{spec.package_name}/ui/view_models.py"
    for marker in (
        "load_latest_checkpoint",
        "sha256",
        "VERSION_PATTERN",
        "PROJECT_PATTERN",
        "warnings",
    ):
        if marker not in desired[view_path]:
            raise MigrationError(f"{spec.repo_name} view model missing marker: {marker}")
    for relative, text in desired.items():
        if relative.endswith(".py"):
            compile(text, relative, "exec")
        if "/General/" in relative:
            match = log_pattern(spec).fullmatch(Path(relative).name)
            if match is None:
                raise MigrationError(f"invalid generated General log: {relative}")
            headings = (
                (
                    "# Checkpoint",
                    "## Định danh",
                    "## Cấu hình hệ thống",
                    "## Lệnh và cấu hình chạy",
                    "## Kết quả",
                    "## Cảnh báo",
                    "## Artifact và lineage",
                    "## Việc tiếp theo",
                )
                if match.group(1) == "Checkpoints"
                else (
                    "# Error",
                    "## Định danh",
                    "## Triệu chứng",
                    "## Cấu hình và đầu vào",
                    "## Nguyên nhân",
                    "## Khắc phục hoặc rollback",
                    "## Regression và trạng thái",
                )
            )
            missing = [heading for heading in headings if heading not in text]
            if missing:
                raise MigrationError(f"{relative} missing headings: {missing}")


def preflight_repo(root: Path, spec: CourseProject) -> Path:
    """Verify the exact live-repository seams without changing them."""
    repo = (root / spec.repo_name).resolve()
    try:
        repo.relative_to(root.resolve())
    except ValueError as exc:
        raise MigrationError(f"repository escapes root: {repo}") from exc
    required = (
        "Documents/General/RULE.md",
        f"src/{spec.package_name}/ui/view_models.py",
        "streamlit_app/app.py",
        "scripts/validate_project.py",
    )
    missing = [relative for relative in required if not (repo / relative).is_file()]
    if missing:
        raise MigrationError(f"{spec.repo_name} missing preflight paths: {missing}")
    return repo


def build_plan(root: Path, spec: CourseProject) -> RepositoryPlan:
    """Compare desired UTF-8 files with one live repository."""
    repo = preflight_repo(root, spec)
    desired = build_desired(repo, spec)
    validate_desired(spec, desired)
    changes: list[FileChange] = []
    for relative, text in sorted(desired.items()):
        target = safe_target(repo, relative)
        after = text.encode("utf-8")
        before = read_bytes(target) if target.is_file() else None
        if before != after:
            changes.append(FileChange(relative, before, after))
    return RepositoryPlan(spec, repo, desired, tuple(changes))


def backup_plan(plan: RepositoryPlan) -> None:
    """Persist exact originals and an atomic rollback manifest before mutation."""
    backup_root = plan.repo / BACKUP_RELATIVE
    entries: list[dict[str, str | None]] = []
    for change in plan.changes:
        backup_relative: str | None = None
        if change.before is not None:
            backup = backup_root / "files" / Path(change.relative)
            if backup.exists() and read_bytes(backup) != change.before:
                raise MigrationError(f"backup drift: {backup}")
            if not backup.exists():
                atomic_write(backup, change.before)
            backup_relative = backup.relative_to(plan.repo).as_posix()
        entries.append(
            {
                "path": change.relative,
                "action": change.action,
                "before_sha256": (
                    sha256_bytes(change.before) if change.before is not None else None
                ),
                "after_sha256": sha256_bytes(change.after),
                "backup": backup_relative,
            }
        )
    manifest = {
        "migration_id": MIGRATION_ID,
        "repository": plan.spec.repo_name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "entries": entries,
        "rollback": "Restore backup only when current SHA-256 equals after_sha256; remove created file only under the same condition.",
    }
    atomic_write(
        backup_root / "manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def apply_plan(plan: RepositoryPlan) -> None:
    """Apply exact writes transactionally and restore originals on failure."""
    applied: list[FileChange] = []
    try:
        for change in plan.changes:
            atomic_write(safe_target(plan.repo, change.relative), change.after)
            applied.append(change)
    except BaseException as exc:
        for change in reversed(applied):
            target = safe_target(plan.repo, change.relative)
            if change.before is None:
                if target.is_file() and read_bytes(target) == change.after:
                    target.unlink()
            else:
                atomic_write(target, change.before)
        raise MigrationError(f"write failed and transaction was rolled back: {exc}") from exc
    for change in plan.changes:
        target = safe_target(plan.repo, change.relative)
        if not target.is_file() or read_bytes(target) != change.after:
            raise MigrationError(f"post-write verification failed: {target}")


def print_plan(plan: RepositoryPlan) -> None:
    """Print a compact dry-run summary without changing repositories."""
    if not plan.changes:
        print(f"NOOP {plan.spec.repo_name}")
        return
    print(f"PLAN {plan.spec.repo_name}: {len(plan.changes)} file(s)")
    for change in plan.changes:
        print(f"- {change.action}: {change.relative} -> {sha256_bytes(change.after)[:12]}")


def parse_args() -> argparse.Namespace:
    """Parse a bounded parent directory; dry-run is the default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Parent directory containing all four repos.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Back up and atomically apply the displayed plan.",
    )
    return parser.parse_args()


def main() -> int:
    """Plan all repositories, then optionally back up and apply them."""
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise MigrationError(f"root is not a directory: {root}")
    plans = tuple(build_plan(root, spec) for spec in PROJECTS)
    for plan in plans:
        print_plan(plan)
    if not args.apply:
        print("DRY-RUN: no files written")
        return 0
    for plan in plans:
        if plan.changes:
            backup_plan(plan)
    for plan in plans:
        if plan.changes:
            apply_plan(plan)
    refreshed = tuple(build_plan(root, spec) for spec in PROJECTS)
    if any(plan.changes for plan in refreshed):
        raise MigrationError("idempotency verification failed after apply")
    print("APPLIED: backup, atomic writes and idempotency verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
