"""Rebuild four course knowledge DOCX files from their canonical Markdown sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

REPOSITORIES = (
    "CS406-Visual-Asset-Retrieval",
    "DS111-Retrieval-Data-Analysis",
    "DS201-Temporal-Retrieval-Learning",
    "DS314-Evidence-Grounded-Retrieval",
)
SOURCE = Path("Documents/References/PROJECT-KNOWLEDGE.md")
OUTPUT = Path("Documents/References/PROJECT-KNOWLEDGE.docx")
BACKUP_ROOT = Path(".migration-backups/course-repositories-v4-20260826/docx-original")
MANIFEST = Path(".migration-backups/course-repositories-v4-20260826/docx-rebuild-manifest.json")


class RebuildError(RuntimeError):
    """Report a failed preflight, conversion, or round-trip validation."""


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    """Run one bounded local document command and capture its output."""
    return subprocess.run(command, check=True, capture_output=True, timeout=180)


def plain_text(path: Path) -> str:
    """Round-trip a DOCX through Pandoc for semantic marker checks."""
    return run(["pandoc", str(path), "--to=plain", "--wrap=none"]).stdout.decode(
        "utf-8", errors="replace"
    )


def validate_docx(path: Path, repo_name: str) -> dict[str, object]:
    """Validate OOXML structure and required updated content markers."""
    if not path.is_file() or path.stat().st_size < 10_000:
        raise RebuildError(f"DOCX thiếu hoặc quá nhỏ: {path}")
    if not zipfile.is_zipfile(path):
        raise RebuildError(f"DOCX không phải OOXML ZIP: {path}")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None or "word/document.xml" not in archive.namelist():
            raise RebuildError(f"DOCX có CRC/package lỗi: {path}")
    text = plain_text(path)
    required = (
        "Tóm tắt điều hành",
        "Đối chiếu chương trình",
        "Data-first",
        "ServiceAdapters",
        "anchor/parent",
        "Checkpoint REJECT|REPEAT",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RebuildError(f"DOCX {repo_name} thiếu marker: {missing}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "words": len(text.split()),
    }


def build_one(repo: Path, staging: Path) -> Path:
    """Render one Markdown source to a staging DOCX without touching the target."""
    source = repo / SOURCE
    output = repo / OUTPUT
    if not source.is_file() or not output.is_file():
        raise RebuildError(f"Thiếu source/output reference tại {repo}")
    staged = staging / repo.name / OUTPUT.name
    staged.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "pandoc",
            str(source),
            "--from=markdown+tex_math_dollars",
            "--to=docx",
            "--standalone",
            "--number-sections",
            f"--reference-doc={output}",
            "--metadata=lang:vi-VN",
            f"--output={staged}",
        ]
    )
    validate_docx(staged, repo.name)
    return staged


def replace_with_backup(repo: Path, staged: Path) -> dict[str, object]:
    """Back up the original once, then atomically replace the live DOCX."""
    output = repo / OUTPUT
    backup = repo / BACKUP_ROOT / OUTPUT
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(output, backup)
    original_sha = sha256(output)
    staged_sha = sha256(staged)
    if original_sha != staged_sha:
        sibling = output.with_name(f".{output.name}.rebuild.tmp")
        shutil.copy2(staged, sibling)
        os.replace(sibling, output)
    result = validate_docx(output, repo.name)
    result.update(
        {
            "before_sha256": original_sha,
            "changed": original_sha != staged_sha,
            "backup": str(backup),
        }
    )
    return result


def write_manifest(repo: Path, result: dict[str, object]) -> None:
    """Write a compact rebuild manifest inside the existing migration backup."""
    path = repo / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": SOURCE.as_posix(),
        "output": OUTPUT.as_posix(),
        "converter": "pandoc",
        "result": result,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse an explicit repository parent and safe mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--apply", action="store_true")
    modes.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Dry-run by default; build/replace only when --apply is explicit."""
    args = parse_args()
    root = args.root.resolve()
    repos = [root / name for name in REPOSITORIES]
    missing = [str(repo) for repo in repos if not repo.is_dir()]
    if missing:
        raise RebuildError("Thiếu repository: " + ", ".join(missing))
    if args.validate:
        results = [validate_docx(repo / OUTPUT, repo.name) for repo in repos]
        print(f"VALIDATION PASSED: {len(results)} course DOCX")
        return 0
    if not args.apply:
        for repo in repos:
            print(f"WOULD REBUILD {repo / OUTPUT}")
        print("DRY-RUN: không file nào được thay đổi")
        return 0

    scratch_parent = Path(__file__).resolve().parents[1] / ".tmp"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="course-docx-", dir=scratch_parent) as temporary:
        staging = Path(temporary)
        staged = {repo: build_one(repo, staging) for repo in repos}
        for repo, candidate in staged.items():
            result = replace_with_backup(repo, candidate)
            write_manifest(repo, result)
            print(f"REBUILT {repo.name}: {result['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
