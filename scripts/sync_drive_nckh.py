#!/usr/bin/env python3
"""Dong bo du lieu va ma nguon sang thu muc Google Drive NCKH phuc vu AI/GPT va nghien cuu."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path, PurePosixPath

MANIFEST = Path("configs/Shared/sync_nckh_manifest.json")
ARCHIVES = ("Documents/Archive", "configs/Archive")
MIB = 1024**2
GIB = 1024**3
IGNORED_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    ".git",
    ".tmp",
    ".migration-backups",
    ".local-quarantine",
}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".bak", ".swp"}


def normalize(value: str) -> str:
    """Chuan hoa duong dan posix tuong doi."""
    value = value.replace("\\", "/").strip("/")
    parts = PurePosixPath(value).parts
    if not parts or any(part in (".", "..") for part in parts):
        raise ValueError(f"Duong dan khong hop le: {value!r}")
    return "/".join(parts)


def beneath(path: str, root: str) -> bool:
    """Kiem tra path nam trong hoac trung voi thu muc root."""
    return path == root or path.startswith(root + "/")


def archive(path: str) -> bool:
    """Kiem tra path co thuoc nhom luu tru bi loai tru."""
    return any(beneath(path.casefold(), root.casefold()) for root in ARCHIVES)


def generated(path: str) -> bool:
    """Loc cac file rac, file sinh tu dong va file he thong."""
    pure = PurePosixPath(path)
    return (
        any(part in IGNORED_DIRS for part in pure.parts)
        or pure.suffix.lower() in IGNORED_SUFFIXES
        or pure.name.lower() in {"desktop.ini", "thumbs.db", ".ds_store"}
    )


def load_manifest(path: Path) -> dict:
    """Doc va kiem tra tinh hop le cua manifest dong bo."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "include_files",
        "include_roots",
        "exclude_roots",
        "retired_drive_roots",
        "protected_drive_roots",
    ):
        data[key] = [normalize(item) for item in data.get(key, [])]
    if not all(item in data["exclude_roots"] for item in ARCHIVES):
        raise ValueError("Archive roots bat buoc phai bi loai tru")
    if any(archive(item) for key in ("include_files", "include_roots") for item in data[key]):
        raise ValueError("Duong dan Archive khong duoc phep nam trong allowlist")
    data["large_files"] = {
        normalize(key): value for key, value in data.get("large_files", {}).items()
    }
    return data


def allowed(path: str, size: int, data: dict) -> bool:
    """Kiem tra xem file co du dieu kien de dong bo khong."""
    path = normalize(path)
    if generated(path):
        return False
    if archive(path) or any(
        beneath(path.casefold(), root.casefold()) for root in data["exclude_roots"]
    ):
        return False
    if path not in data["include_files"] and not any(
        beneath(path, root) for root in data["include_roots"]
    ):
        return False
    if size > 100 * MIB and path not in data["large_files"]:
        return False
    return True


def drive_target() -> Path:
    """Xac dinh thu muc dich tren Google Drive."""
    candidates = [
        Path(r"G:\My Drive\NCKH"),
        Path(r"G:\Drive của tôi\NCKH"),
        Path(r"G:\My Drive"),
        Path(r"G:\Drive của tôi"),
    ]
    for candidate in candidates:
        if candidate.name == "NCKH" and candidate.is_dir():
            return candidate
        if candidate.name in ("My Drive", "Drive của tôi") and candidate.is_dir():
            nckh_dir = candidate / "NCKH"
            nckh_dir.mkdir(parents=True, exist_ok=True)
            return nckh_dir
    raise FileNotFoundError("Khong tim thay o Google Drive (G:) tren he thong")


def files(root: Path, *, skip_generated: bool = False):
    """Duyet cay thu muc bo qua cac thu muc sinh tu dong."""
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [
            name
            for name in dirs
            if (not skip_generated or name not in IGNORED_DIRS)
            and not (Path(base) / name).is_symlink()
        ]
        for name in names:
            path = Path(base) / name
            if not path.is_symlink():
                yield path


def source_files(root: Path, data: dict):
    """Chi thu thap cac file nam trong allowlist hop le cua workspace."""
    seen = set()
    for rel in data["include_files"]:
        path = root / rel
        if path.is_file() and not any(parent.is_symlink() for parent in (path, *path.parents)):
            seen.add(path)
            yield path
    for rel in data["include_roots"]:
        subtree = root / rel
        if subtree.is_dir() and not any(
            parent.is_symlink() for parent in (subtree, *subtree.parents)
        ):
            for path in files(subtree, skip_generated=True):
                if path not in seen:
                    seen.add(path)
                    yield path


def build_plan(source: Path, target: Path, data: dict) -> dict:
    """Lap ke hoach sao chep va cap nhat file tu workspace sang Drive."""
    active = {}
    dest_mapping = {}
    duplicate_mappings = []
    rejected_large = []
    for path in source_files(source, data):
        rel = path.relative_to(source).as_posix()
        size = path.stat().st_size
        if allowed(rel, size, data):
            if rel in dest_mapping:
                duplicate_mappings.append({"rel": rel, "first": str(dest_mapping[rel]), "second": str(path)})
            dest_mapping[rel] = path
            active[rel] = size
        elif size > 100 * MIB and (
            rel in data["include_files"]
            or any(beneath(rel, root) for root in data["include_roots"])
        ):
            rejected_large.append({"path": rel, "bytes": size})

    drive = {}
    if target.is_dir():
        for path in files(target, skip_generated=True):
            rel = path.relative_to(target).as_posix()
            if not generated(rel):
                drive[rel] = path.stat().st_size

    copy = sorted(active.keys() - drive.keys())
    update = sorted(
        rel
        for rel in active.keys() & drive.keys()
        if active[rel] != drive[rel]
        or abs((source / rel).stat().st_mtime - (target / rel).stat().st_mtime) > 1
    )
    stale = sorted(
        rel
        for rel in drive.keys() - active.keys()
        if not allowed(rel, drive[rel], data)
        and any(
            beneath(rel.casefold(), root.casefold())
            for root in data.get("retired_drive_roots", [])
        )
        and not any(
            beneath(rel.casefold(), root.casefold())
            for root in data.get("protected_drive_roots", [])
        )
    )

    return {
        "schema_version": "nckh-sync-plan-v1",
        "sync_mode": "ACTIVE_ALLOWLIST_ONLY",
        "canonical_source_root": str(source),
        "target_drive_root": str(target),
        "canonical_source_root_count": 1,
        "duplicate_destination_mapping_count": len(duplicate_mappings),
        "duplicate_mappings": duplicate_mappings,
        "active_file_count": len(active),
        "active_bytes": sum(active.values()),
        "drive_file_count": len(drive),
        "drive_bytes": sum(drive.values()),
        "copy": copy,
        "update": update,
        "delete_stale": [{"path": rel, "bytes": drive[rel]} for rel in stale],
        "stale_bytes": sum(drive[rel] for rel in stale),
        "rejected_large": rejected_large,
    }


def main() -> None:
    """Diem khoi chay dong bo hoa."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Duong dan thu muc nguon")
    parser.add_argument("--target", type=Path, help="Duong dan thu muc Google Drive")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--plan", type=Path, help="Ghi file plan JSON")
    parser.add_argument("--apply", action="store_true", help="Thuc thi sao chep vao Google Drive")
    args = parser.parse_args()

    source = (args.source or Path.cwd()).resolve()
    target = (args.target or drive_target()).resolve()

    if target == source or source in target.parents or target in source.parents:
        parser.error("Thu muc dich khong duoc trung hoac long nhau voi thu muc nguon")

    manifest_path = (args.manifest if args.manifest.is_absolute() else source / args.manifest).resolve()
    if not manifest_path.is_file():
        # Fallback to local default if source doesn't have it
        manifest_path = MANIFEST.resolve()

    data = load_manifest(manifest_path)
    plan = build_plan(source, target, data)

    if plan["duplicate_destination_mapping_count"] > 0:
        if args.apply:
            parser.error("Phat hien duplicate destination mappings, fail-closed: tu choi --apply")

    if args.plan:
        args.plan.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    summary = {key: value for key, value in plan.items() if not isinstance(value, list)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"Ke hoach: copy={len(plan['copy'])}, "
        f"update={len(plan['update'])}, "
        f"delete_stale={len(plan['delete_stale'])}, "
        f"duplicate_mappings={plan['duplicate_destination_mapping_count']}"
    )

    if args.apply:
        copied_count = 0
        for rel in plan["copy"] + plan["update"]:
            src_file = source / rel
            dest_file = target / rel
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest_file)
            copied_count += 1
        print(f"Da dong bo thanh cong {copied_count} files sang: {target}")

        deleted_count = 0
        for stale_item in plan.get("delete_stale", []):
            rel = stale_item["path"]
            stale_file = target / rel
            if stale_file.is_file():
                stale_file.unlink(missing_ok=True)
                deleted_count += 1
        if deleted_count > 0:
            print(f"Da don dep an toan {deleted_count} files stale khoi: {target}")


if __name__ == "__main__":
    main()
