from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

DATE = date.today().isoformat()
CHECKPOINT = re.compile(r".+_Checkpoints_\d{2}-\d{2}-\d{2}_V\d+\.md$")
ERROR = re.compile(r".+_Error_\d{2}-\d{2}-\d{2}_V\d+\.md$")


def yaml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(value, ensure_ascii=False)


def add_frontmatter(path: Path, fields: dict[str, Any]) -> None:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end < 0:
            raise ValueError("frontmatter hiện hữu không đóng")
        header = text[4:end].splitlines()
        keys = {line.split(":", 1)[0].strip() for line in header if ":" in line}
        for key, value in fields.items():
            if key not in keys:
                header.append(f"{key}: {yaml_value(value)}")
        body = text[end + 4 :].lstrip("\n")
        output = "---\n" + "\n".join(header) + "\n---\n\n" + body
    else:
        output = (
            "---\n"
            + "\n".join(f"{key}: {yaml_value(value)}" for key, value in fields.items())
            + "\n---\n\n"
            + text.lstrip()
        )
    path.write_text(output.rstrip() + "\n", encoding="utf-8", newline="\n")


def classify(root: Path, path: Path) -> tuple[str, dict[str, Any] | None]:
    docs = (root / "Documents").resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(docs):
        raise ValueError("file phải nằm trong Documents/")
    if resolved.suffix.lower() != ".md":
        raise ValueError("ingest chỉ nhận Markdown")
    relative = resolved.relative_to(docs)
    parts = [unicodedata.normalize("NFC", item).casefold() for item in relative.parts]
    if parts[0] == "home":
        raise ValueError("HOME chỉ là control plane; không ingest file tùy ý")
    if parts[0] == "decision-making":
        slug = re.sub(r"[^A-Za-z0-9]+", "-", resolved.stem).strip("-").upper() or "DOC"
        return "decision", {
            "loại": "quyết-định",
            "mã": f"PENDING-{slug}",
            "dự_án": root.name,
            "trạng_thái": "đề-xuất",
            "chuẩn_hiện_hành": False,
            "thay_thế": [],
            "bị_thay_thế_bởi": [],
            "tham_chiếu": [],
            "được_xác_thực_bởi": [],
            "cập_nhật_lần_cuối": DATE,
        }
    if parts[0] == "references":
        return "reference", {
            "loại": "tham-chiếu",
            "nguồn": f"local-file:{relative.as_posix()}",
            "độ_tin_cậy": "exploratory",
            "chủ_đề": resolved.stem,
            "dự_án": root.name,
            "trạng_thái": "đang-dùng",
            "chuẩn_hiện_hành": False,
            "cập_nhật_lần_cuối": DATE,
        }
    if parts[0] == "general":
        if resolved.name == "RULE.md":
            return "rule", None
        if CHECKPOINT.fullmatch(resolved.name) or ERROR.fullmatch(resolved.name):
            return "immutable-log", None
        raise ValueError("General chỉ nhận RULE.md hoặc checkpoint/error đúng grammar bất biến")
    raise ValueError("file không thuộc Decision-Making, References hoặc General")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest một node knowledge mới")
    parser.add_argument("path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    target = args.path.resolve()
    if not target.is_file():
        raise SystemExit(f"Không tìm thấy file: {target}")
    existing = target.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    if existing.startswith("---\n") and re.search(r"(?m)^loại\s*:", existing):
        print(f"ALREADY INGESTED: {target}")
        return 0
    kind, fields = classify(root, target)
    if fields is not None:
        if args.dry_run:
            print(f"DRY-RUN ingest {kind}: {target}")
            print(json.dumps(fields, ensure_ascii=False, indent=2))
        else:
            add_frontmatter(target, fields)
            print(f"INGESTED {kind}: {target}")
    else:
        print(f"ACCEPTED {kind}: {target}")
    validator = root / "scripts" / "validate_knowledge_vault.py"
    if not args.dry_run and validator.is_file():
        result = subprocess.run(
            [sys.executable, str(validator)],
            cwd=root,
            check=False,
        )
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
