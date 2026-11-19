"""Generate exact SHA256 manifest for all artifacts in V2 Run 05 Final."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN05_DIR = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_ab_ntp"


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest() -> None:
    manifest = {
        "schema_version": "1.0",
        "study_id": "V2_RUN05_FINAL_AB_NTP_MULTITASK",
        "artifacts_directory": str(_RUN05_DIR.relative_to(_REPO_ROOT)),
        "files": {},
    }

    for p in sorted(_RUN05_DIR.rglob("*")):
        if p.is_file() and p.name != "SHA256_MANIFEST.json":
            rel = str(p.relative_to(_RUN05_DIR)).replace("\\", "/")
            manifest["files"][rel] = {
                "size_bytes": p.stat().st_size,
                "sha256": compute_sha256(p),
            }

    out_path = _RUN05_DIR / "SHA256_MANIFEST.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Generated manifest with {len(manifest['files'])} files at {out_path}")


if __name__ == "__main__":
    generate_manifest()
