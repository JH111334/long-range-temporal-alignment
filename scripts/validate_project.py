"""Validate repository structure, registries, fixtures, text encoding, and smoke run."""

from __future__ import annotations

import json
import re
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from temporal_alignment.io import file_sha256, load_examples  # noqa: E402
from temporal_alignment.manifests import load_dataset_manifests  # noqa: E402
from temporal_alignment.pipeline import run_smoke_pipeline  # noqa: E402
from temporal_alignment.ui import LineageError, load_latest_checkpoint  # noqa: E402

REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "AGENTS.md",
    "Workflows/flowinfo.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-research.txt",
    "configs/smoke.json",
    "configs/dataset_candidates.yaml",
    "configs/model_candidates.yaml",
    "data/synthetic/examples.jsonl",
    "registries/datasets.json",
    "registries/models.json",
    "schemas/temporal_example.schema.json",
    "schemas/temporal_qa_request.schema.json",
    "schemas/temporal_qa_response.schema.json",
    "src/temporal_alignment/contracts.py",
    "src/temporal_alignment/data_management/lifecycle.py",
    "src/temporal_alignment/normalization/vietnamese.py",
    "src/temporal_alignment/application/contracts.py",
    "src/temporal_alignment/workflow/planning.py",
    "tests/test_pipeline.py",
    "tests/test_planning_scaffold.py",
    "tests/test_service_adapters.py",
    "tests/test_ui_scaffold.py",
    ".agents/skills/temporal-alignment-research/SKILL.md",
    ".agents/skills/temporal-research-planning/SKILL.md",
    "docs/adr/0004-vietnam-first-historical-temporal-qa.md",
    "docs/adr/0005-lifecycle-and-application-contract.md",
    "docs/inconsistencies.md",
    "Documents/Decision-Making/10_Dinh_huong_Viet_Nam_2026.md",
    "Documents/Decision-Making/11_Vong_doi_du_lieu_mo_hinh_ung_dung.md",
    "Documents/Decision-Making/12_PROJECT-CONTEXT.md",
    "Documents/Decision-Making/13_KHA-THI-DATA-4GB.md",
    "Documents/Decision-Making/ERROR-EXPERIENCE.md",
    "Documents/References/Markdown/12_Khao_sat_sau_Ticktack_2025_2026.md",
    "Documents/References/Markdown/EXTERNAL-SERVICES-GUIDELINE.md",
    "Documents/References/DOCX/01_Dinh_nghia_bai_toan.docx",
    "Documents/References/DOCX/10_Lo_trinh_NCKH_KLTN.docx",
    "Documents/General/RULE.md",
    "Documents/References/Markdown/MCP-TOOLS.md",
    "Documents/References/Markdown/COURSE-PROJECTS-2026.md",
    "Documents/References/Markdown/09_Skills_tools_workflow.md",
    "Documents/Decision-Making/DEFERRED-MULTIMODAL.md",
    "Documents/Decision-Making/meeting_02.md",
    "Documents/References/HTML/progress-report-2026-08-23.html",
    "Documents/References/HTML/progress-report-2026-08-25.html",
    "Documents/References/HTML/progress-report-2026-08-26.html",
    "Documents/References/HTML/weekly-brief-2026-08-07.html",
    ".codex/config.toml",
    "requirements-ui.txt",
    "scripts/course_repo_catalog.py",
    "scripts/course_repo_templates.py",
    "scripts/bootstrap_course_repositories.py",
    "scripts/upgrade_course_repositories.py",
    "scripts/upgrade_noncourse_repositories.py",
    "scripts/upgrade_knowledge_library.py",
    "scripts/rebuild_course_docx.py",
    "scripts/rebuild_knowledge_library.py",
    "scripts/render_docx_qa.py",
    "scripts/write_final_project_logs.py",
    "scripts/write_course_lineage_validation_logs.py",
    "src/temporal_alignment/ServiceAdapters/contracts.py",
    "src/temporal_alignment/ServiceAdapters/inference.py",
    "src/temporal_alignment/ui/view_models.py",
    "streamlit_app/app.py",
)

GENERAL_LOG_PATTERN = re.compile(
    r"^long-range-temporal-alignment_(?P<kind>Checkpoints|Error)_"
    r"(?P<date>(?:0[1-9]|[12][0-9]|3[01])-(?:0[1-9]|1[0-2])-\d{2})_"
    r"V(?P<version>[1-9]\d*)\.md$"
)

DOCX_REQUIRED_MARKERS = {
    "01_Dinh_nghia_bai_toan.docx": "point/interval là kiểu cấu trúc độc lập",
    "02_Phan_tich_Ticktack.docx": "point/interval là trục cấu trúc riêng",
    "03_Du_lieu_va_benchmark.docx": "point|interval",
    "04_Bieu_dien_thoi_gian.docx": "point/interval được mã hóa",
    "05_Objective_EWC_LoRA.docx": "kết quả âm tính hợp lệ",
    "06_Kien_truc_va_contracts.docx": "temporal kind",
    "07_Models_frameworks_phan_cung.docx": "RTX 2050 4 GB",
    "08_Ke_hoach_thuc_nghiem.docx": "temporal_kind=point|interval",
    "09_Error_analysis_va_XAI.docx": "counterexample",
    "10_Lo_trinh_NCKH_KLTN.docx": "point/interval tách riêng",
}


def contains_mojibake(text: str) -> bool:
    """Detect common UTF-8-as-Windows-1252 sequences without rejecting Vietnamese Â."""
    markers = tuple(
        "".join(chr(codepoint) for codepoint in codepoints)
        for codepoints in (
            (0x00C2, 0x00A0),
            (0x00C3, 0x00A1),
            (0x00C3, 0x00A2),
            (0x00C3, 0x00A9),
            (0x00C3, 0x00B4),
            (0x00C4, 0x2018),
            (0x00E1, 0x00BB),
            (0x00E1, 0x00BA),
        )
    )
    return any(marker in text for marker in markers)


def read_docx_text(path: Path) -> str:
    """Extract visible DOCX text with only the standard library."""

    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    return " ".join(node.text or "" for node in root.iter() if node.tag.endswith("}t"))


def html_contract_violations(text: str) -> tuple[str, ...]:
    """Return missing structural markers for the supervisor-facing HTML report."""
    lowered = text.casefold()
    required_markers = (
        "<!doctype html",
        "<html",
        "<meta charset=",
        "<title>",
        "</html>",
    )
    return tuple(marker for marker in required_markers if marker not in lowered)


def general_log_violations(general_dir: Path) -> tuple[str, ...]:
    """Validate General whitelist, real dates, version continuity, and log fields."""
    failures: list[str] = []
    versions: dict[tuple[str, str], list[int]] = {}
    if not general_dir.is_dir():
        return ("Documents/General is missing",)

    for path in sorted(general_dir.iterdir()):
        if path.is_dir() and path.name == "Reports":
            continue
        if not path.is_file():
            failures.append(f"non-file entry in Documents/General: {path.name}")
            continue
        if path.name == "RULE.md" or path.suffix == ".json":
            continue
        match = GENERAL_LOG_PATTERN.fullmatch(path.name)
        if match is None:
            failures.append(f"non-log file in Documents/General: {path.name}")
            continue
        try:
            datetime.strptime(match.group("date"), "%d-%m-%y")
        except ValueError:
            failures.append(f"invalid calendar date in General log: {path.name}")
        key = (match.group("kind"), match.group("date"))
        versions.setdefault(key, []).append(int(match.group("version")))

        lowered = path.read_text(encoding="utf-8").casefold()
        common = ("project:", "thời điểm:", "phiên bản:")
        missing_common = [field for field in common if field not in lowered]
        if missing_common:
            failures.append(f"{path.name} missing fields: {', '.join(missing_common)}")
        if match.group("kind") == "Checkpoints":
            if "mức bằng chứng" not in lowered:
                failures.append(f"{path.name} missing evidence class")
            if not any(token in lowered for token in ("result", "kết quả", "trạng thái")):
                failures.append(f"{path.name} missing result/status")
        else:
            required_error = (
                "expected:",
                "actual:",
                "nguyên nhân:",
                "khắc phục",
                "regression",
            )
            missing_error = [field for field in required_error if field not in lowered]
            if missing_error:
                failures.append(f"{path.name} missing error fields: {', '.join(missing_error)}")

    for (kind, date), observed in sorted(versions.items()):
        expected = list(range(1, max(observed) + 1))
        if sorted(observed) != expected:
            failures.append(
                f"General versions must be continuous for {kind} {date}: "
                f"observed={sorted(observed)}, expected={expected}"
            )
    return tuple(failures)


def main() -> int:
    failures: list[str] = []
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            failures.append(f"missing required file: {relative}")

    failures.extend(general_log_violations(ROOT / "Documents" / "General"))
    try:
        load_latest_checkpoint(ROOT)
    except (OSError, LineageError) as exc:
        failures.append(f"dashboard checkpoint lineage failed: {exc}")

    for path in ROOT.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            failures.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")

    temporal_schema_path = ROOT / "schemas" / "temporal_example.schema.json"
    if temporal_schema_path.is_file():
        try:
            temporal_schema = json.loads(temporal_schema_path.read_text(encoding="utf-8"))
            properties = temporal_schema["properties"]
            required = set(temporal_schema["required"])
            if properties["granularity"].get("enum") != ["year", "month", "day"]:
                failures.append("granularity must contain only year, month and day")
            if properties["temporal_kind"].get("enum") != [
                "point",
                "interval",
                "duration",
                "relation",
            ]:
                failures.append("temporal_kind enum does not match the canonical contract")
            if "temporal_kind" not in required:
                failures.append("temporal_kind must be required in TemporalExample")
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            failures.append(f"temporal schema contract cannot be inspected: {exc}")

    for suffix in ("*.py", "*.md", "*.html", "*.toml", "*.yaml", "*.yml", "*.json"):
        for path in ROOT.rglob(suffix):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeError as exc:
                failures.append(f"non-UTF-8 text {path.relative_to(ROOT)}: {exc}")
                continue
            if contains_mojibake(text):
                failures.append(f"possible mojibake in {path.relative_to(ROOT)}")

    for report_name in (
        "progress-report-2026-08-23.html",
        "progress-report-2026-08-25.html",
        "progress-report-2026-08-26.html",
    ):
        html_report = ROOT / "Documents" / "References" / "HTML" / report_name
        if html_report.is_file():
            try:
                missing_markers = html_contract_violations(html_report.read_text(encoding="utf-8"))
                if missing_markers:
                    failures.append(f"{report_name} missing markers: " + ", ".join(missing_markers))
            except (OSError, UnicodeError) as exc:
                failures.append(f"cannot read {report_name}: {exc}")

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

    docx_files = sorted((ROOT / "Documents" / "References" / "DOCX").glob("*.docx"))
    source_markdown_files = sorted((ROOT / "docs" / "converted").glob("*.md"))
    if len(docx_files) != 10:
        failures.append(f"expected 10 DOCX files, found {len(docx_files)}")
    if len(source_markdown_files) != 10:
        failures.append(
            f"expected 10 DOCX source Markdown files, found {len(source_markdown_files)}"
        )
    if {path.stem for path in docx_files} != {path.stem for path in source_markdown_files}:
        failures.append("DOCX and source Markdown stems do not match")
    for path in docx_files:
        if path.stat().st_size < 10_000:
            failures.append(f"DOCX appears too small: {path.name}")
            continue
        marker = DOCX_REQUIRED_MARKERS.get(path.name)
        if marker is None:
            failures.append(f"DOCX freshness marker is not registered: {path.name}")
            continue
        try:
            docx_text = read_docx_text(path)
        except (OSError, KeyError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
            failures.append(f"DOCX package cannot be read: {path.name}: {exc}")
            continue
        if marker.casefold() not in docx_text.casefold():
            failures.append(f"DOCX is stale or missing canonical marker: {path.name}")

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
    print(
        f"VALIDATION PASSED: {len(REQUIRED_FILES)} required files, "
        "10 DOCX/source pairs, progress HTML, smoke pipeline"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
