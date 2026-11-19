from __future__ import annotations

import json
import re
import sys
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "Documents"
HOME = DOCS / "HOME"
REQUIRED_HOME = (
    "Trang chủ.md",
    "Bản đồ tri thức.md",
    "Quy tắc tri thức.md",
    "Chuẩn metadata.md",
    "Nhật ký thay đổi tri thức.md",
)
COMMON_FIELDS = (
    "loại",
    "dự_án",
    "trạng_thái",
    "chuẩn_hiện_hành",
    "cập_nhật_lần_cuối",
)
DECISION_FIELDS = (
    "mã",
    "thay_thế",
    "bị_thay_thế_bởi",
    "tham_chiếu",
    "được_xác_thực_bởi",
)
DECISION_STATUSES = {
    "đề-xuất",
    "đang-thử-nghiệm",
    "đang-dùng",
    "bị-thay-thế",
    "ngừng-sử-dụng",
}
RELIABILITY = {"canonical", "high", "medium", "exploratory"}
LINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")


def normalized(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\\", "/")).casefold()


def parse_value(raw: str) -> Any:
    value = raw.strip()
    if value == "true":
        return True
    if value == "false":
        return False
    if value.startswith("[") or value.startswith('"'):
        return json.loads(value)
    return value


def parse_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = next(
        (index for index in range(1, len(lines)) if lines[index].strip() == "---"),
        None,
    )
    if end is None:
        raise ValueError("frontmatter không đóng")
    result: dict[str, Any] = {}
    for line_number, line in enumerate(lines[1:end], start=2):
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"frontmatter dòng {line_number} không phải key: value")
        key, raw = line.split(":", 1)
        key = key.strip()
        if not key or key in result:
            raise ValueError(f"key rỗng hoặc lặp tại dòng {line_number}")
        result[key] = parse_value(raw)
    return result, "\n".join(lines[end + 1 :])


def strip_fences(text: str) -> str:
    output: list[str] = []
    in_fence = False
    backtick_fence = chr(96) * 3
    for line in text.splitlines():
        marker = line.lstrip()
        if marker.startswith(backtick_fence) or marker.startswith("~~~"):
            in_fence = not in_fence
            continue
        if not in_fence:
            output.append(line)
    return "\n".join(output)


def build_index(files: list[Path]) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    by_relative: dict[str, Path] = {}
    by_stem: dict[str, list[Path]] = {}
    for path in files:
        relative = path.relative_to(DOCS).with_suffix("").as_posix()
        by_relative[normalized(relative)] = path
        by_stem.setdefault(normalized(path.stem), []).append(path)
    return by_relative, by_stem


def resolve_link(
    raw: str,
    by_relative: dict[str, Path],
    by_stem: dict[str, list[Path]],
) -> tuple[Path | None, str | None]:
    target = raw.split("|", 1)[0].split("#", 1)[0].strip()
    if target.endswith(".md"):
        target = target[:-3]
    if target.startswith("Documents/"):
        target = target[len("Documents/") :]
    if "/" in target or "\\" in target:
        found = by_relative.get(normalized(target))
        return found, None if found else f"không tồn tại: {target}"
    matches = by_stem.get(normalized(target), [])
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, f"không tồn tại: {target}"
    return None, f"mơ hồ: {target} có {len(matches)} đích"


def link_values(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    result: list[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        result.extend(LINK_PATTERN.findall(item))
    return result


def main() -> int:
    errors: list[str] = []
    if not DOCS.is_dir():
        print(f"FAIL: thiếu {DOCS}")
        return 1
    for name in REQUIRED_HOME:
        if not (HOME / name).is_file():
            errors.append(f"thiếu HOME/{name}")
    if (HOME / "Welcome.md").exists():
        errors.append("Welcome.md mặc định chưa được retire")
    if (HOME / ".obsidian").exists():
        errors.append(".obsidian còn nằm dưới HOME; vault root bị sai")

    obsidian = DOCS / ".obsidian"
    for name in ("app.json", "appearance.json", "core-plugins.json", "graph.json"):
        path = obsidian / name
        if not path.is_file():
            errors.append(f"thiếu .obsidian/{name}")
            continue
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"JSON không hợp lệ {path}: {exc}")

    files = sorted(DOCS.rglob("*.md"))
    by_relative, by_stem = build_index(files)
    managed: dict[Path, dict[str, Any]] = {}
    bodies: dict[Path, str] = {}
    for path in files:
        try:
            fields, body = parse_frontmatter(path)
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: YAML subset không hợp lệ: {exc}")
            continue
        if "loại" in fields:
            managed[path] = fields
            bodies[path] = body

    for name in REQUIRED_HOME:
        path = HOME / name
        if path.is_file() and path not in managed:
            errors.append(f"HOME/{name} thiếu metadata contract")

    decision_ids: dict[str, Path] = {}
    decisions: dict[Path, dict[str, Any]] = {}
    reference_count = 0
    for path, fields in managed.items():
        relative = path.relative_to(ROOT)
        for key in COMMON_FIELDS:
            if key not in fields:
                errors.append(f"{relative}: thiếu field {key}")
        current = fields.get("chuẩn_hiện_hành")
        if not isinstance(current, bool):
            errors.append(f"{relative}: chuẩn_hiện_hành phải là boolean")
        try:
            date.fromisoformat(str(fields.get("cập_nhật_lần_cuối", "")))
        except ValueError:
            errors.append(f"{relative}: cập_nhật_lần_cuối phải là YYYY-MM-DD")

        kind = fields.get("loại")
        if kind == "quyết-định":
            decisions[path] = fields
            for key in DECISION_FIELDS:
                if key not in fields:
                    errors.append(f"{relative}: decision thiếu field {key}")
            decision_id = str(fields.get("mã", "")).strip()
            if not decision_id:
                errors.append(f"{relative}: mã decision rỗng")
            elif decision_id in decision_ids:
                errors.append(
                    f"mã decision lặp {decision_id}: {decision_ids[decision_id]} và {path}"
                )
            else:
                decision_ids[decision_id] = path
            status = fields.get("trạng_thái")
            if status not in DECISION_STATUSES:
                errors.append(f"{relative}: decision status ngoài contract: {status}")
            if status == "đang-dùng" and current is not True:
                errors.append(f"{relative}: đang-dùng phải chuẩn_hiện_hành=true")
            if status in {"bị-thay-thế", "ngừng-sử-dụng"} and current is not False:
                errors.append(f"{relative}: decision retired phải chuẩn_hiện_hành=false")
            if status == "bị-thay-thế" and not link_values(
                fields.get("bị_thay_thế_bởi", [])
            ):
                errors.append(f"{relative}: bị-thay-thế nhưng thiếu successor")
        elif kind == "tham-chiếu":
            reference_count += 1
            for key in ("nguồn", "độ_tin_cậy", "chủ_đề"):
                if key not in fields:
                    errors.append(f"{relative}: reference thiếu field {key}")
            if fields.get("độ_tin_cậy") not in RELIABILITY:
                errors.append(f"{relative}: độ_tin_cậy ngoài contract")
            if current is not False:
                errors.append(f"{relative}: Reference không được masquerade canonical")

    if not any(
        fields.get("trạng_thái") == "đang-dùng"
        and fields.get("chuẩn_hiện_hành") is True
        for fields in decisions.values()
    ):
        errors.append("không có active canonical decision được quản lý")
    if reference_count == 0:
        errors.append("không có reference trọng yếu được quản lý")

    scan_paths = set(managed)
    scan_paths.update(HOME / name for name in REQUIRED_HOME if (HOME / name).is_file())
    for path in sorted(scan_paths):
        text = strip_fences(path.read_text(encoding="utf-8"))
        for raw_link in LINK_PATTERN.findall(text):
            _, problem = resolve_link(raw_link, by_relative, by_stem)
            if problem:
                errors.append(f"{path.relative_to(ROOT)}: wikilink {problem}")

    for path, fields in decisions.items():
        for relation, reciprocal in (
            ("thay_thế", "bị_thay_thế_bởi"),
            ("bị_thay_thế_bởi", "thay_thế"),
        ):
            for raw_link in link_values(fields.get(relation, [])):
                target, problem = resolve_link(raw_link, by_relative, by_stem)
                if problem or target is None:
                    continue
                target_fields = decisions.get(target)
                if target_fields is None:
                    errors.append(
                        f"{path.relative_to(ROOT)}: {relation} trỏ tới node không phải decision"
                    )
                    continue
                reciprocal_targets: set[Path] = set()
                for reciprocal_link in link_values(target_fields.get(reciprocal, [])):
                    resolved, _ = resolve_link(reciprocal_link, by_relative, by_stem)
                    if resolved is not None:
                        reciprocal_targets.add(resolved)
                if path not in reciprocal_targets:
                    errors.append(
                        f"{path.relative_to(ROOT)} và {target.relative_to(ROOT)} thiếu supersession hai chiều"
                    )

    rule_path = HOME / "Quy tắc tri thức.md"
    if rule_path.is_file():
        rule_text = rule_path.read_text(encoding="utf-8")
        for marker in ("OBSERVATION", "VALIDATION", "COMPARE", "DECISION", "PROPAGATION"):
            if marker not in rule_text:
                errors.append(f"Quy tắc tri thức thiếu stage {marker}")
        for marker in ("REJECT", "REPEAT", "supersession"):
            if marker not in rule_text:
                errors.append(f"Quy tắc tri thức thiếu guard {marker}")

    if not (DOCS / "General" / "RULE.md").is_file():
        errors.append("thiếu Documents/General/RULE.md")

    if errors:
        print("Living Knowledge Vault: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        "Living Knowledge Vault: PASS "
        f"({len(files)} Markdown, {len(managed)} managed, "
        f"{len(decisions)} decisions, {reference_count} references)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
