#!/usr/bin/env python3
"""Bổ sung title metadata cho hai DOCX DSC2026 bằng thay thế nguyên tử."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from docx import Document

TITLES = {
    "Task2-LegalQA-Context.docx": "DSC2026 Task 2 — Context nền tảng",
    "Task2-Technical-Execution-Guide.docx": "DSC2026 Task 2 — Thực thi kỹ thuật ngắn gọn",
}


def update(docx_root: Path, backup_root: Path, *, apply: bool) -> int:
    """Kiểm tra target; khi apply, lưu backup rồi cập nhật metadata."""
    planned: list[tuple[Path, str]] = []
    for name, title in TITLES.items():
        target = (docx_root / name).resolve()
        if target.parent != docx_root.resolve() or not target.is_file():
            raise FileNotFoundError(f"Thiếu DOCX allowlisted: {target}")
        document = Document(target)
        if document.core_properties.title != title:
            planned.append((target, title))

    print(f"planned_changes={len(planned)}")
    if not apply:
        return 0

    backup_root.mkdir(parents=True, exist_ok=True)
    for target, title in planned:
        backup = backup_root / target.name
        if backup.exists():
            raise FileExistsError(f"Backup đã tồn tại: {backup}")
        shutil.copy2(target, backup)
        document = Document(target)
        document.core_properties.title = title
        temporary = target.with_suffix(".docx.tmp")
        document.save(temporary)
        os.replace(temporary, target)
        print(f"updated={target.name}")
    return 0


def main() -> int:
    """Đọc CLI dry-run mặc định và chạy cập nhật allowlisted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docx-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    return update(args.docx_root, args.backup_root, apply=args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
