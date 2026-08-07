"""Validate repository structure, registries, fixtures, text encoding, and smoke run."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from temporal_alignment.io import file_sha256, load_examples  # noqa: E402
from temporal_alignment.manifests import load_dataset_manifests  # noqa: E402
from temporal_alignment.pipeline import run_smoke_pipeline  # noqa: E402


REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "AGENTS.md",
    "Workflows/flowinfo.md",
    "pyproject.toml",
    "configs/smoke.json",
    "data/synthetic/examples.jsonl",
    "registries/datasets.json",
    "schemas/temporal_example.schema.json",
    "src/temporal_alignment/contracts.py",
    "tests/test_pipeline.py",
    ".agents/skills/temporal-alignment-research/SKILL.md",
    "Documents/01_Dinh_nghia_bai_toan.docx",
    "Documents/10_Lo_trinh_NCKH_KLTN.docx",
    "Documents/weekly-brief-2026-08-07.html",
)


def main() -> int:
    failures: list[str] = []
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            failures.append(f"missing required file: {relative}")

    for path in ROOT.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            failures.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")

    mojibake_markers = tuple(
        "".join(chr(codepoint) for codepoint in codepoints)
        for codepoints in (
            (0x00C3,),
            (0x00C2,),
            (0x00C4,),
            (0x00E1, 0x00BB),
            (0x00E1, 0x00BA),
        )
    )
    for suffix in ("*.py", "*.md", "*.toml", "*.yaml", "*.yml", "*.json"):
        for path in ROOT.rglob(suffix):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeError as exc:
                failures.append(f"non-UTF-8 text {path.relative_to(ROOT)}: {exc}")
                continue
            if any(marker in text for marker in mojibake_markers):
                failures.append(f"possible mojibake in {path.relative_to(ROOT)}")

    fixture_path = ROOT / "data" / "synthetic" / "examples.jsonl"
    if fixture_path.is_file():
        try:
            examples = load_examples(fixture_path)
            if len(examples) != 7:
                failures.append("synthetic fixture must contain exactly 7 examples")
        except ValueError as exc:
            failures.append(str(exc))

    registry_path = ROOT / "registries" / "datasets.json"
    if registry_path.is_file() and fixture_path.is_file():
        try:
            manifests = load_dataset_manifests(registry_path)
            fixture = manifests["synthetic-contract-fixture-v1"]
            fixture.assert_ingestible()
            if fixture.sha256 != file_sha256(fixture_path):
                failures.append("synthetic fixture checksum does not match registry")
        except (KeyError, ValueError) as exc:
            failures.append(f"dataset registry gate failed: {exc}")

    docx_files = sorted((ROOT / "Documents").glob("*.docx"))
    if len(docx_files) != 10:
        failures.append(f"expected 10 DOCX files, found {len(docx_files)}")
    for path in docx_files:
        if path.stat().st_size < 10_000:
            failures.append(f"DOCX appears too small: {path.name}")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_smoke_pipeline(
                ROOT / "configs" / "smoke.json",
                Path(temp_dir) / "smoke.json",
            )
        if result.get("evidence_class") != "smoke_test":
            failures.append("smoke pipeline evidence class is not smoke_test")
    except (OSError, ValueError) as exc:
        failures.append(f"smoke pipeline failed: {exc}")

    if failures:
        print("VALIDATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"VALIDATION PASSED: {len(REQUIRED_FILES)} required files, 10 DOCX, smoke pipeline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
