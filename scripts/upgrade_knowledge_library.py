"""Nâng cấp cầu nối dự án trong thư viện DOCX Markdown theo cách có thể hoàn tác.

Script chỉ dùng thư viện chuẩn, mặc định chạy dry-run và không chạm CodingScratch.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

MAIN_SOURCE_DIR = Path(".build/content")
AIC_SOURCE_DIR = Path(".build/aic-docx-sources")
CATALOG_PATH = Path("tools/doc_catalog.py")
CODING_SCRATCH_DIR = Path("CodingScratch")
BACKUP_DIR = Path(".knowledge-library-backups")
EXPECTED_MAIN = 49
EXPECTED_AIC = 15
BRIDGE_START = "<!-- PROJECT_BRIDGE:START -->"
BRIDGE_END = "<!-- PROJECT_BRIDGE:END -->"
HARDWARE_LABEL = "NVIDIA GeForce RTX 2050, 4.096 MiB VRAM"
UTF8_BOM = b"\xef\xbb\xbf"
HEADING_PATTERN = re.compile(r"(?m)^#{1,6}[ \t]+.+$")
LINK_PATTERN = re.compile(r"!?\[[^\]\n]*\]\([^)]+\)|<https?://[^>\n]+>")

GLOBAL_TOKENS = (
    "AI Agent",
    "LLM",
    "CV",
    "NLP",
    "Retrieval",
    "CS406-Visual-Asset-Retrieval",
    "DS111-Retrieval-Data-Analysis",
    "DS201-Temporal-Retrieval-Learning",
    "DS314-Evidence-Grounded-Retrieval",
    "long-range-temporal-alignment",
    "Evidence-Grounded-Text-Agent",
    "AIC2026",
    "DSC-TinDipLaPo",
    "Multimodal-Asset-Retrieval",
    HARDWARE_LABEL,
    "ServiceAdapters",
)
PER_FILE_TOKENS = (
    "data-first",
    HARDWARE_LABEL,
    "ServiceAdapters",
    "Bài tập",
    "Mẹo",
    "Lỗi chí mạng",
)

FOCUS_BY_TAXONOMY = {
    "GeneralML": "DS111: baseline, thống kê, paired evaluation, leakage và error slices cho retrieval.",
    "GeneralDL": "DS201 và long-range: biểu diễn học được, temporal objective, ablation và inference nhẹ.",
    "GeneralNLP": "DS314 và Evidence: sparse/dense retrieval, reranking, citation, abstention và grounded answer.",
    "CV": "CS406: visual asset retrieval; AIC2026 và Multi-Asset là case study về index và provenance.",
    "NLP": "DS314: truy xuất văn bản tiếng Việt; Evidence và DSC là case study về bằng chứng và nhiễu.",
    "AI Agent": "Agent chỉ điều phối retriever/tool có quyền, ngân sách, stop condition và audit trail.",
    "AI Engineering": "Đóng gói baseline, adapter, artifact, metric chất lượng-latency-memory và rollback.",
    "Data Engineering": "DS111: data contract, manifest, lineage, group/temporal split và reproducible analysis.",
    "Docker": "Khóa môi trường CPU trước; GPU/service là profile tùy chọn sau preflight và benchmark.",
    "GitHub": "CI kiểm schema, leakage, metric và manifest; không đưa data/model có điều khoản vào Git.",
    "MathKN": "Nối công thức với embedding, ranking, uncertainty, temporal representation và tối ưu 4 GB VRAM.",
}

MAP_DETAIL = """
- Ma trận đồ án: `CS406-Visual-Asset-Retrieval` học CV retrieval; `DS111-Retrieval-Data-Analysis` học data audit và đánh giá; `DS201-Temporal-Retrieval-Learning` cùng `long-range-temporal-alignment` học biểu diễn thời gian dài; `DS314-Evidence-Grounded-Retrieval` cùng `Evidence-Grounded-Text-Agent` học retrieval có citation và abstention.
- Case study có kiểm soát: `AIC2026` cho video retrieval; `DSC-TinDipLaPo` cho dữ liệu nhiễu và bằng chứng; `Multimodal-Asset-Retrieval` cho asset, index và provenance. Case study không tự biến thành benchmark của đồ án mới.
- Chốt bài toán theo thứ tự: người dùng và quyết định cần hỗ trợ -> đơn vị dữ liệu/nhãn -> split chống leakage -> baseline nhỏ -> failure taxonomy -> một cải tiến có ablation -> demo chỉ đọc artifact đã kiểm định.
- Tính mới phù hợp sinh viên đến từ data contract, split khó, hard-negative, temporal/evidence constraint, calibration hoặc quy trình đánh giá; không đến từ việc chọn model lớn hơn.
""".strip()

SOURCE_DETAIL = """
- Hồ sơ nguồn phải tách paper gốc, official documentation, dataset/model card và internal decision. Mỗi data/model/service ghi URL chính thức, ngày truy cập, license/terms, revision/checksum, người rà soát và vai trò train/validation/test.
- Nguồn cho từng nhánh phải trả lời được input/output, đơn vị annotation, split, metric, compute, failure cases và điều kiện tái lập. Winning system chỉ là bằng chứng thiết kế trong đúng protocol, không phải công thức sao chép.
- Service bên ngoài được ưu tiên theo nhu cầu: Hugging Face cho model/dataset revision; Kaggle cho dataset có điều khoản; remote inference chỉ sau khi kiểm privacy, quota, latency, cost, cache và fallback.
""".strip()


class UpgradeError(RuntimeError):
    """Báo lỗi khi migration hoặc validation không bảo toàn hợp đồng."""


@dataclass(frozen=True)
class SourceChange:
    """Lưu một thay đổi nguồn có thể sao lưu và hoàn tác."""

    path: Path
    relative: Path
    corpus: str
    before: bytes
    after: bytes


def _sha256(data: bytes) -> str:
    """Tính SHA-256 cho dữ liệu byte."""
    return hashlib.sha256(data).hexdigest()


def _read_utf8(path: Path) -> tuple[str, bool]:
    """Đọc UTF-8 và giữ thông tin BOM để ghi lại nguyên dạng."""
    raw = path.read_bytes()
    return raw.decode("utf-8-sig"), raw.startswith(UTF8_BOM)


def _encode_utf8(text: str, with_bom: bool) -> bytes:
    """Mã hóa UTF-8 với chính sách BOM ban đầu."""
    payload = text.encode("utf-8")
    return UTF8_BOM + payload if with_bom else payload


def _source_paths(root: Path) -> tuple[list[Path], list[Path]]:
    """Lấy chính xác hai tập Markdown được phép nâng cấp."""
    main = sorted((root / MAIN_SOURCE_DIR).rglob("*.md"))
    aic = sorted((root / AIC_SOURCE_DIR).rglob("*.md"))
    return main, aic


def _taxonomy(relative: Path, corpus: str) -> str:
    """Suy ra taxonomy mà không đổi cây thư mục hiện hữu."""
    if corpus == "aic":
        return relative.parts[0].casefold()
    first = relative.parts[0] if len(relative.parts) > 1 else relative.stem
    if first == "ThuatNgu" and len(relative.parts) > 1:
        return relative.parts[1]
    return first


def _focus(relative: Path, corpus: str) -> str:
    """Chọn neo dự án phù hợp cho từng nhánh tri thức."""
    taxonomy = _taxonomy(relative, corpus)
    if corpus == "aic":
        return {
            "offline": "AIC2026 offline là case study data/artifact; ưu tiên sampling, manifest và model nhẹ.",
            "online": "AIC2026 online là case study candidate retrieval, fusion, temporal grouping và UI audit.",
            "database": "AIC2026 database là case study lineage, release pointer, exact search và rollback.",
        }.get(taxonomy, "AIC2026 chỉ được dùng như case study có protocol.")
    return FOCUS_BY_TAXONOMY.get(
        taxonomy,
        "Bản đồ chung nối bốn đồ án môn học với long-range, Evidence và các case study retrieval.",
    )


def _bridge_text(relative: Path, corpus: str) -> str:
    """Tạo cầu nối compact mà không thêm heading hoặc hyperlink."""
    detail = ""
    if corpus == "main" and relative.name == "00_Ban_do_hoc_tap.md":
        detail = "\n" + MAP_DETAIL
    elif corpus == "main" and relative.name == "99_Nguon_va_phuong_phap_kiem_chung.md":
        detail = "\n" + SOURCE_DETAIL
    return f"""{BRIDGE_START}
**Cầu nối dự án hiện hành**

- Trục thống nhất: AI Agent/LLM/CV/NLP/Retrieval, nhưng retriever, data contract và bằng chứng phải đứng trước generation hoặc agent.
- Neo taxonomy: {_focus(relative, corpus)}
- Tư duy data-first: xác định provenance, schema, label, duplicate, missingness, group/time dependency và split trước khi chọn model; khóa manifest trước mỗi so sánh.
- Phần cứng chuẩn: `{HARDWARE_LABEL}`. Batch, dtype, resolution/sequence length, latency và peak VRAM phải được đo và ghi đúng cấu hình; giữ CPU/exact/lexical baseline và không suy hiệu năng từ tên GPU.
- `ServiceAdapters`: core chỉ gọi contract fail-closed; adapter Hugging Face/Kaggle/remote inference/artifact store phải pin revision, kiểm license, secret, timeout, cache, quota, checksum và fallback local.
- Lộ trình: baseline tái lập -> data/split audit -> error slices -> một cải tiến -> ablation cùng protocol -> artifact có hash -> Streamlit chỉ đọc kết quả đã xác minh.
- Bài tập: viết input/output và data manifest nhỏ; chạy baseline CPU; tạo ba failure slices; chỉ sau đó mô tả adapter/model ứng viên và tiêu chí promote.
- Mẹo: cải thiện hard-negative, chunk/window, metadata filter, calibration, query normalization và caching trước khi tăng kích thước model.
- Lỗi chí mạng: leakage/near-duplicate, đổi data cùng lúc đổi model, fit preprocessing trước split, citation sai mapping, metric không cùng query set, gọi smoke test là benchmark, hoặc remote service không pin revision.{detail}
{BRIDGE_END}"""


def _upsert_bridge(text: str, bridge: str) -> str:
    """Chèn hoặc thay đúng một block, giữ newline và phần còn lại nguyên vẹn."""
    starts, ends = text.count(BRIDGE_START), text.count(BRIDGE_END)
    if starts != ends or starts > 1:
        raise UpgradeError("Marker PROJECT_BRIDGE bị thiếu cặp hoặc trùng.")
    newline = "\r\n" if "\r\n" in text else "\n"
    block = bridge.replace("\n", newline)
    if starts == 1:
        pattern = re.compile(re.escape(BRIDGE_START) + r".*?" + re.escape(BRIDGE_END), re.S)
        return pattern.sub(lambda _: block, text, count=1)
    front_matter = re.match(r"\A---\r?\n.*?\r?\n---\r?\n", text, re.S)
    offset = front_matter.end() if front_matter else 0
    return text[:offset] + newline + block + newline + text[offset:]


def _structure_signature(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Lấy signature heading và link phải được bảo toàn tuyệt đối."""
    return tuple(HEADING_PATTERN.findall(text)), tuple(LINK_PATTERN.findall(text))


def _catalog_sources(root: Path) -> set[Path]:
    """Đọc khóa Markdown của OUTPUT_PATHS bằng AST, không import code ngoài."""
    tree = ast.parse((root / CATALOG_PATH).read_text(encoding="utf-8-sig"))
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if not isinstance(target, ast.Name) or target.id != "OUTPUT_PATHS":
            continue
        if not isinstance(node.value, ast.Dict):
            break
        paths: set[Path] = set()
        for key in node.value.keys:
            if not isinstance(key, ast.Call) or not key.args:
                raise UpgradeError("Khóa OUTPUT_PATHS không phải Path literal.")
            value = key.args[0]
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                raise UpgradeError("Khóa OUTPUT_PATHS không phải chuỗi tĩnh.")
            paths.add(Path(value.value))
        return paths
    raise UpgradeError("Không tìm thấy OUTPUT_PATHS trong tools/doc_catalog.py.")


def _validate_content(
    root: Path,
    main: list[Path],
    aic: list[Path],
    content: dict[Path, str],
) -> None:
    """Kiểm count, catalog, marker và coverage mà không ghi file."""
    errors: list[str] = []
    if len(main) != EXPECTED_MAIN:
        errors.append(f"Nguồn chính: {len(main)}/{EXPECTED_MAIN}")
    if len(aic) != EXPECTED_AIC:
        errors.append(f"Nguồn AIC: {len(aic)}/{EXPECTED_AIC}")
    relative_main = {path.relative_to(root / MAIN_SOURCE_DIR) for path in main}
    if relative_main != _catalog_sources(root):
        errors.append("Tập nguồn chính không khớp khóa Markdown trong OUTPUT_PATHS")
    for path in main + aic:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            errors.append(f"Nguồn symlink hoặc vượt root: {path}")
            continue
        text = content[path]
        if text.count(BRIDGE_START) != 1 or text.count(BRIDGE_END) != 1:
            errors.append(f"Marker bridge không hợp lệ: {path.relative_to(root)}")
        missing = [token for token in PER_FILE_TOKENS if token not in text]
        if missing:
            errors.append(f"Thiếu coverage ở {path.relative_to(root)}: {', '.join(missing)}")
        if CODING_SCRATCH_DIR.name in path.relative_to(root).parts:
            errors.append(f"Nguồn vượt biên CodingScratch: {path.relative_to(root)}")
    corpus = "\n".join(content.values())
    missing_global = [token for token in GLOBAL_TOKENS if token not in corpus]
    if missing_global:
        errors.append("Thiếu coverage toàn thư viện: " + ", ".join(missing_global))
    if errors:
        raise UpgradeError("Validation thất bại:\n- " + "\n- ".join(errors))


def _plan(
    root: Path,
) -> tuple[list[Path], list[Path], dict[Path, str], list[SourceChange]]:
    """Dựng candidate trong bộ nhớ và chứng minh heading/link không đổi."""
    main, aic = _source_paths(root)
    content: dict[Path, str] = {}
    changes: list[SourceChange] = []
    for corpus, base, paths in (
        ("main", root / MAIN_SOURCE_DIR, main),
        ("aic", root / AIC_SOURCE_DIR, aic),
    ):
        for path in paths:
            before_text, with_bom = _read_utf8(path)
            relative = path.relative_to(base)
            after_text = _upsert_bridge(before_text, _bridge_text(relative, corpus))
            if _structure_signature(before_text) != _structure_signature(after_text):
                raise UpgradeError(f"Heading/link bị đổi ngoài ý muốn: {path.relative_to(root)}")
            before, after = path.read_bytes(), _encode_utf8(after_text, with_bom)
            content[path] = after_text
            if before != after:
                changes.append(SourceChange(path, relative, corpus, before, after))
    _validate_content(root, main, aic, content)
    return main, aic, content, changes


def _tree_digest(path: Path) -> str:
    """Tạo digest CodingScratch, bỏ cache sinh tự động và không sửa nội dung."""
    if not path.exists():
        return "ABSENT"
    digest = hashlib.sha256()
    files: list[Path] = []
    for directory, names, filenames in os.walk(path):
        names[:] = sorted(name for name in names if name not in {"__pycache__", ".pytest_cache"})
        files.extend(Path(directory) / name for name in sorted(filenames))
    for item in sorted(files):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    """Ghi file qua temporary sibling rồi thay thế atomically."""
    temporary = path.with_name(f".{path.name}.upgrade.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    """Ghi JSON manifest atomically với tiếng Việt đọc được."""
    data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_write(path, data)


def _prepare_backup(
    root: Path,
    changes: list[SourceChange],
    catalog_hash: str,
    coding_hash: str,
) -> tuple[Path, dict[str, object]]:
    """Sao lưu mọi file sẽ đổi và tạo manifest trạng thái prepared."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    backup = root / BACKUP_DIR / stamp
    entries: list[dict[str, str]] = []
    for change in changes:
        destination = backup / "files" / change.corpus / change.relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(change.path, destination)
        entries.append(
            {
                "path": change.path.relative_to(root).as_posix(),
                "backup": destination.relative_to(backup).as_posix(),
                "before_sha256": _sha256(change.before),
                "after_sha256": _sha256(change.after),
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "prepared",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "root": str(root),
        "main_source_count": EXPECTED_MAIN,
        "aic_source_count": EXPECTED_AIC,
        "catalog_sha256": catalog_hash,
        "coding_scratch_sha256_before": coding_hash,
        "preserved": ["catalog paths", "source paths", "headings", "links"],
        "entries": entries,
    }
    _write_manifest(backup / "manifest.json", manifest)
    return backup, manifest


def _restore(backup: Path, root: Path, changes: list[SourceChange]) -> None:
    """Khôi phục toàn bộ source từ backup nếu apply thất bại."""
    if any(change.path.is_relative_to(root / CODING_SCRATCH_DIR) for change in changes):
        raise UpgradeError("Rollback phát hiện thay đổi vượt biên CodingScratch.")
    for change in changes:
        source = backup / "files" / change.corpus / change.relative
        shutil.copy2(source, change.path)


def _apply(
    root: Path, main: list[Path], aic: list[Path], changes: list[SourceChange]
) -> Path | None:
    """Áp dụng candidate có backup, post-validation và rollback."""
    if not changes:
        return None
    catalog_hash = _sha256((root / CATALOG_PATH).read_bytes())
    coding_hash = _tree_digest(root / CODING_SCRATCH_DIR)
    backup, manifest = _prepare_backup(root, changes, catalog_hash, coding_hash)
    try:
        for change in changes:
            _atomic_write(change.path, change.after)
        live = {path: _read_utf8(path)[0] for path in main + aic}
        _validate_content(root, main, aic, live)
        if _sha256((root / CATALOG_PATH).read_bytes()) != catalog_hash:
            raise UpgradeError("doc_catalog.py thay đổi trong lúc apply.")
        coding_after = _tree_digest(root / CODING_SCRATCH_DIR)
        if coding_after != coding_hash:
            raise UpgradeError("CodingScratch thay đổi trong lúc apply.")
        manifest["status"] = "applied"
        manifest["applied_at_utc"] = datetime.now(UTC).isoformat()
        manifest["coding_scratch_sha256_after"] = coding_after
        _write_manifest(backup / "manifest.json", manifest)
        return backup / "manifest.json"
    except (OSError, UnicodeError, SyntaxError, UpgradeError) as exc:
        _restore(backup, root, changes)
        manifest["status"] = "rolled_back"
        manifest["error"] = str(exc)
        _write_manifest(backup / "manifest.json", manifest)
        raise


def _parser() -> argparse.ArgumentParser:
    """Tạo CLI với dry-run là chế độ mặc định."""
    parser = argparse.ArgumentParser(
        description="Refocus thư viện tri thức mà không đổi cấu trúc DOCX."
    )
    parser.add_argument("--root", type=Path, required=True, help="Root TuDuyToanHoc&ThuatToan.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", dest="mode", action="store_const", const="dry-run")
    modes.add_argument("--apply", dest="mode", action="store_const", const="apply")
    modes.add_argument("--validate", dest="mode", action="store_const", const="validate")
    parser.set_defaults(mode="dry-run")
    return parser


def main() -> int:
    """Chạy dry-run, apply hoặc validation theo yêu cầu."""
    arguments = _parser().parse_args()
    root = arguments.root.resolve()
    try:
        if not (root / MAIN_SOURCE_DIR).is_dir() or not (root / AIC_SOURCE_DIR).is_dir():
            raise UpgradeError("Root không chứa đủ hai thư mục nguồn tri thức.")
        if arguments.mode == "validate":
            coding_before = _tree_digest(root / CODING_SCRATCH_DIR)
            main_paths, aic_paths = _source_paths(root)
            current = {path: _read_utf8(path)[0] for path in main_paths + aic_paths}
            _validate_content(root, main_paths, aic_paths, current)
            if _tree_digest(root / CODING_SCRATCH_DIR) != coding_before:
                raise UpgradeError("CodingScratch thay đổi trong lúc validation.")
            print(
                f"VALIDATION PASSED: {len(main_paths)} main, {len(aic_paths)} AIC; CodingScratch unchanged."
            )
            return 0
        main_paths, aic_paths, _, changes = _plan(root)
        if arguments.mode == "dry-run":
            print(
                f"DRY RUN PASSED: {len(main_paths)} main, {len(aic_paths)} AIC, "
                f"{len(changes)} files would change; no files written."
            )
            return 0
        manifest = _apply(root, main_paths, aic_paths, changes)
        if manifest is None:
            print("APPLY PASSED: nội dung đã đúng; không tạo backup mới.")
        else:
            print(f"APPLY PASSED: {len(changes)} files; manifest={manifest}")
        return 0
    except (OSError, UnicodeError, SyntaxError, UpgradeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
