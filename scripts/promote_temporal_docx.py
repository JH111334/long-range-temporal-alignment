"""Promote QA-passed temporal DOCX candidates with a reversible backup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / "Documents" / "References" / "DOCX"
BACKUP_ID = "temporal-docx-refresh-20260826-v1"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_package(path: Path) -> None:
    """Reject missing, tiny or malformed DOCX candidates."""

    if not path.is_file() or path.stat().st_size < 10_000:
        raise ValueError(f"invalid DOCX candidate: {path}")
    with zipfile.ZipFile(path) as archive:
        if "word/document.xml" not in archive.namelist():
            raise ValueError(f"missing document.xml: {path}")


def atomic_copy(source: Path, target: Path) -> None:
    """Copy one binary file atomically into an existing directory."""

    descriptor, temporary = tempfile.mkstemp(
        prefix=".docx-promote-", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args() -> argparse.Namespace:
    """Parse a candidate directory and explicit apply flag."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Validate, back up and optionally promote exactly ten matching DOCX files."""

    args = parse_args()
    candidate_dir = args.candidate.resolve()
    candidates = sorted(candidate_dir.glob("*.docx"))
    live = sorted(LIVE_DIR.glob("*.docx"))
    if len(candidates) != 10 or len(live) != 10:
        raise ValueError(f"expected 10 candidate/live DOCX files: {len(candidates)}/{len(live)}")
    if {path.name for path in candidates} != {path.name for path in live}:
        raise ValueError("candidate/live DOCX names differ")
    for path in candidates:
        validate_package(path)

    entries = []
    for candidate in candidates:
        current = LIVE_DIR / candidate.name
        entries.append(
            {
                "path": candidate.name,
                "original_sha256": sha256(current),
                "candidate_sha256": sha256(candidate),
            }
        )
        print(f"{candidate.name}: {entries[-1]['candidate_sha256']}")
    if not args.apply:
        print("dry-run; pass --apply to promote")
        return 0

    backup = ROOT / ".migration-backups" / BACKUP_ID
    if backup.exists():
        raise FileExistsError(f"backup already exists: {backup}")
    original = backup / "original"
    original.mkdir(parents=True)
    for path in live:
        shutil.copy2(path, original / path.name)
    (backup / "manifest.json").write_text(
        json.dumps(
            {
                "backup_id": BACKUP_ID,
                "candidate_dir": str(candidate_dir),
                "files": entries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for candidate in candidates:
        atomic_copy(candidate, LIVE_DIR / candidate.name)
    for candidate in candidates:
        if sha256(candidate) != sha256(LIVE_DIR / candidate.name):
            raise RuntimeError(f"promotion hash mismatch: {candidate.name}")
    print(f"apply=PASS; backup={backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
