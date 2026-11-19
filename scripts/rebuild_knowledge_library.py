"""Rebuild and validate the 49 main plus 15 AIC knowledge-library DOCX files."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

MAIN_SOURCE = Path(".build/content")
AIC_SOURCE = Path(".build/aic-docx-sources")
CATALOG = Path("tools/doc_catalog.py")
REFERENCE = Path(".build/reference.docx")
BACKUP_PARENT = Path(".knowledge-library-backups")

AIC_OUTPUTS = {
    "online/system/Kien_truc_truy_xuat_tuong_tac_va_lien_ket.md": "AIC/ONLINE SYSTEM/Tư duy hệ thống và liên kết/Kiến trúc truy xuất tương tác và liên kết.docx",
    "online/models/Phan_tich_truy_van_va_candidate_retrieval.md": "AIC/ONLINE SYSTEM/Các models và công nghệ tương ứng và lý do chọn/Phân tích truy vấn và candidate retrieval.docx",
    "online/models/Fusion_reranking_VideoQA_va_agent_gioi_han.md": "AIC/ONLINE SYSTEM/Các models và công nghệ tương ứng và lý do chọn/Fusion, reranking, VideoQA và agent giới hạn.docx",
    "online/models/API_UI_quan_sat_va_submission_an_toan.md": "AIC/ONLINE SYSTEM/Các models và công nghệ tương ứng và lý do chọn/API, giao diện và submission an toàn.docx",
    "online/math/Nen_tang_xep_hang_fusion_temporal_va_danh_gia.md": "AIC/ONLINE SYSTEM/Tư duy thuật toán và toán học nền tảng/Nền tảng xếp hạng, fusion, temporal và đánh giá.docx",
    "database/system/Tu_duy_data_platform_va_lien_ket.md": "AIC/DATABASE SYSTEM/Tư duy hệ thống và liên kết/Tư duy data platform và liên kết.docx",
    "database/models/Storage_sharding_sao_luu_va_serving_release_1000_gio.md": "AIC/DATABASE SYSTEM/Các models và công nghệ tương ứng và lý do chọn/Storage, sharding, sao lưu và serving release.docx",
    "database/models/Stack_smoke_SQLite_filesystem_NumPy.md": "AIC/DATABASE SYSTEM/Các models và công nghệ tương ứng và lý do chọn/Stack smoke - SQLite, filesystem, NumPy.docx",
    "database/models/Stack_development_PostgreSQL_MinIO_Milvus.md": "AIC/DATABASE SYSTEM/Các models và công nghệ tương ứng và lý do chọn/Stack development - PostgreSQL, MinIO, Milvus.docx",
    "database/math/Nen_tang_du_lieu_chi_muc_nhat_quan_va_dung_luong.md": "AIC/DATABASE SYSTEM/Tư duy thuật toán và toán học nền tảng/Nền tảng dữ liệu, chỉ mục và dung lượng.docx",
    "offline/system/Vai_tro_bien_gioi_va_lien_ket.md": "AIC/OFFLINE SYSTEM/Tư duy hệ thống và liên kết/Vai trò, biên giới và liên kết.docx",
    "offline/models/Dong_goi_artifact_xay_index_va_ngan_sach_1000_gio.md": "AIC/OFFLINE SYSTEM/Các models và công nghệ tương ứng và lý do chọn/Artifact, index build và ngân sách 1.000 giờ.docx",
    "offline/models/Trich_xuat_multimodal_va_mo_hinh_de_xuat.md": "AIC/OFFLINE SYSTEM/Các models và công nghệ tương ứng và lý do chọn/Trích xuất multimodal và mô hình đề xuất.docx",
    "offline/models/Giai_ma_lay_mau_va_phan_doan_video.md": "AIC/OFFLINE SYSTEM/Các models và công nghệ tương ứng và lý do chọn/Giải mã, lấy mẫu và phân đoạn video.docx",
    "offline/math/Nen_tang_toan_hoc_va_thuat_toan_offline.md": "AIC/OFFLINE SYSTEM/Tư duy thuật toán và toán học nền tảng/Nền tảng toán học và thuật toán Offline.docx",
}


class LibraryBuildError(RuntimeError):
    """Report a mapping, conversion, package, or semantic validation failure."""


def sha256(path: Path) -> str:
    """Return one file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    """Run one bounded local document command."""
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, timeout=600)


def catalog_outputs(root: Path) -> dict[Path, Path]:
    """Read static OUTPUT_PATHS through AST without importing project code."""
    tree = ast.parse((root / CATALOG).read_text(encoding="utf-8-sig"))
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if not isinstance(target, ast.Name) or target.id != "OUTPUT_PATHS":
            continue
        if not isinstance(node.value, ast.Dict):
            break
        result: dict[Path, Path] = {}
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            if not isinstance(key, ast.Call) or not key.args or not isinstance(value, ast.Call):
                raise LibraryBuildError("OUTPUT_PATHS không dùng Path literal.")
            key_value = key.args[0]
            output_value = value.args[0]
            if not isinstance(key_value, ast.Constant) or not isinstance(
                output_value, ast.Constant
            ):
                raise LibraryBuildError("OUTPUT_PATHS không phải chuỗi tĩnh.")
            result[Path(str(key_value.value))] = Path(str(output_value.value))
        return result
    raise LibraryBuildError("Không tìm thấy OUTPUT_PATHS.")


def preflight(root: Path) -> tuple[dict[Path, Path], dict[Path, Path]]:
    """Validate exact 49/15 source-output mappings and existing deliverables."""
    main = catalog_outputs(root)
    aic = {Path(source): Path(output) for source, output in AIC_OUTPUTS.items()}
    if len(main) != 49 or len(aic) != 15:
        raise LibraryBuildError(f"Sai số lượng mapping: main={len(main)}, AIC={len(aic)}")
    for source, output in main.items():
        if not (root / MAIN_SOURCE / source).is_file() or not (root / output).is_file():
            raise LibraryBuildError(f"Thiếu main source/output: {source} -> {output}")
    for source, output in aic.items():
        if not (root / AIC_SOURCE / source).is_file() or not (root / output).is_file():
            raise LibraryBuildError(f"Thiếu AIC source/output: {source} -> {output}")
    return main, aic


def plain_text(path: Path, root: Path) -> str:
    """Round-trip DOCX to plain text with Pandoc."""
    result = run(["pandoc", str(path), "--to=plain", "--wrap=none"], cwd=root)
    return result.stdout.decode("utf-8", errors="replace")


def validate_one(path: Path, root: Path) -> dict[str, object]:
    """Validate OOXML, updated project bridge, and non-empty rendered text."""
    if not path.is_file() or path.stat().st_size < 10_000 or not zipfile.is_zipfile(path):
        raise LibraryBuildError(f"DOCX thiếu/quá nhỏ/không phải ZIP: {path}")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None or "word/document.xml" not in archive.namelist():
            raise LibraryBuildError(f"DOCX package lỗi: {path}")
    text = plain_text(path, root)
    required = ("Cầu nối dự án hiện hành", "ServiceAdapters", "data-first")
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise LibraryBuildError(f"{path.relative_to(root)} thiếu marker: {missing}")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "words": len(text.split()),
    }


def create_backup(root: Path, outputs: list[Path]) -> tuple[Path, dict[str, object]]:
    """Back up all 64 live DOCX files before any converter writes."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    backup = root / BACKUP_PARENT / stamp
    entries: list[dict[str, str]] = []
    for relative in outputs:
        source = root / relative
        target = backup / "docx-before" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entries.append(
            {
                "path": relative.as_posix(),
                "backup": target.relative_to(backup).as_posix(),
                "before_sha256": sha256(source),
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "prepared",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "entries": entries,
    }
    (backup / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return backup, manifest


def restore(root: Path, backup: Path, outputs: list[Path]) -> None:
    """Restore all DOCX outputs from the pre-build snapshot."""
    for relative in outputs:
        shutil.copy2(backup / "docx-before" / relative, root / relative)


def build_aic(root: Path, mappings: dict[Path, Path], staging: Path) -> dict[Path, Path]:
    """Build 15 AIC DOCX candidates into staging with existing styles as reference."""
    candidates: dict[Path, Path] = {}
    for source_relative, output_relative in mappings.items():
        source = root / AIC_SOURCE / source_relative
        current = root / output_relative
        candidate = staging / output_relative
        candidate.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "pandoc",
                str(source),
                "--from=markdown+tex_math_dollars",
                "--to=docx",
                "--standalone",
                "--number-sections",
                f"--reference-doc={current}",
                "--metadata=lang:vi-VN",
                f"--output={candidate}",
            ],
            cwd=root,
        )
        validate_one(candidate, root=staging)
        candidates[output_relative] = candidate
    return candidates


def atomic_replace(candidate: Path, target: Path) -> None:
    """Copy a staged file to a sibling and replace the target atomically."""
    sibling = target.with_name(f".{target.name}.rebuild.tmp")
    shutil.copy2(candidate, sibling)
    os.replace(sibling, target)


def parse_args() -> argparse.Namespace:
    """Parse explicit library root and a safe execution mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--apply", action="store_true")
    modes.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Dry-run by default; rebuild with backup and rollback only on --apply."""
    args = parse_args()
    root = args.root.resolve()
    main_map, aic_map = preflight(root)
    outputs = list(main_map.values()) + list(aic_map.values())
    if args.validate:
        results = [validate_one(root / relative, root) for relative in outputs]
        print(f"VALIDATION PASSED: {len(results)} DOCX (49 main + 15 AIC)")
        return 0
    if not args.apply:
        print("DRY-RUN PASSED: would rebuild 49 main + 15 AIC DOCX; no writes")
        return 0

    backup, manifest = create_backup(root, outputs)
    scratch_parent = Path(__file__).resolve().parents[1] / ".tmp"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    try:
        run(["py", "-3.12", "-B", "tools/build_docs.py"], cwd=root)
        with tempfile.TemporaryDirectory(prefix="aic-docx-", dir=scratch_parent) as temporary:
            candidates = build_aic(root, aic_map, Path(temporary))
            for relative, candidate in candidates.items():
                atomic_replace(candidate, root / relative)
        results = [validate_one(root / relative, root) for relative in outputs]
        run(["py", "-3.12", "-B", "tools/validate_docs.py"], cwd=root)
        manifest["status"] = "applied"
        manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
        manifest["results"] = results
        (backup / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"REBUILD PASSED: 64 DOCX; manifest={backup / 'manifest.json'}")
        return 0
    except (
        OSError,
        subprocess.SubprocessError,
        zipfile.BadZipFile,
        LibraryBuildError,
    ) as exc:
        restore(root, backup, outputs)
        manifest["status"] = "rolled_back"
        manifest["error"] = str(exc)
        (backup / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
