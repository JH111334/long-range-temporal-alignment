#!/usr/bin/env python3
"""Tạo checkpoint/error ngày 26-08-26 từ các gate đã chạy; dry-run mặc định."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

DATE_TOKEN = "26-08-26"
PROJECTS_ROOT = Path(r"C:\Users\GF63\Documents")
COURSE_NAMES = {
    "CS406-Visual-Asset-Retrieval",
    "DS111-Retrieval-Data-Analysis",
    "DS201-Temporal-Retrieval-Learning",
    "DS314-Evidence-Grounded-Retrieval",
}
SYSTEM_PROFILE = (
    "Windows 11 Home Single Language 10.0.26200 build 26200; "
    "Intel Core i5-12450H 8C/12T; RAM 16.866.684.928 byte; "
    "NVIDIA GeForce RTX 2050 4.096 MiB, driver 596.36; Python 3.12.10"
)


@dataclass(frozen=True, slots=True)
class ProjectResult:
    """Giữ bằng chứng kiểm chứng thực tế cho một repository."""

    name: str
    commands: tuple[str, ...]
    results: tuple[str, ...]
    artifacts: tuple[str, ...]
    next_gate: str


PROJECT_RESULTS = (
    ProjectResult(
        "long-range-temporal-alignment",
        (
            "py -3.12 -X utf8 -B -m pytest -q -p no:cacheprovider --basetemp .tmp\\pytest-temporal-20260826-final",
            "py -3.12 -X utf8 -B scripts\\validate_project.py",
            "black --check --target-version py312 <changed Python, src, tests>",
            "qa_docx.py <10 temporal DOCX> --render-dir ... --preview-dir ...",
            "audit_project_documents.py <8 repository roots>",
        ),
        (
            "46/46 test PASS; smoke/fixture và audit regression only.",
            "Validator PASS: 68 required files, 10 DOCX/source pairs, progress HTML và smoke pipeline.",
            "Black PASS trên toàn bộ scripts/src/tests.",
            "DOCX structural/render/visual QA: 10/10 PASS, mỗi file 2 trang.",
            "Audit tài liệu: 8/8 repository có errors=0 và warnings=0.",
        ),
        (
            "Documents/References/HTML/progress-report-2026-08-26.html",
            "Documents/Decision-Making/03_Ma_tran_khoang_trong_va_cau_hoi.md",
            "Documents/Decision-Making/12_PROJECT-CONTEXT.md",
            ".agents/skills/temporal-research-planning/SKILL.md",
            "AGENTS.md",
            "src/temporal_alignment/contracts.py",
            "scripts/audit_project_documents.py",
            "scripts/validate_project.py",
            "tests/test_document_audit.py",
        ),
        "Chốt source dossier và annotation pilot trước mọi model/data download hoặc measured benchmark.",
    ),
    ProjectResult(
        "CS406-Visual-Asset-Retrieval",
        (
            "$env:PYTHONPATH='src'; py -3.12 -X utf8 -B -m unittest discover -s tests -v",
            "py -3.12 -X utf8 -B scripts\\validate_project.py",
            "qa_docx.py Documents\\References\\PROJECT-KNOWLEDGE.docx --render-dir ... --preview-dir ...",
        ),
        (
            "15/15 unit test PASS; cảnh báo Streamlit bare mode không làm test fail.",
            "Validator PASS: 43 file và 5 checkpoint tại thời điểm chạy.",
            "DOCX structural/render/visual QA: 1/1 PASS, 3 trang; không cần TOC cho brief 3 trang.",
        ),
        (
            "Documents/References/PROJECT-KNOWLEDGE.docx",
            "scripts/validate_project.py",
            ".agents/skills/course-project-delivery/SKILL.md",
            "AGENTS.md",
        ),
        "Chốt ảnh có quyền sử dụng, qrels query-by-image và leakage-safe split trước frozen vision probe.",
    ),
    ProjectResult(
        "DS111-Retrieval-Data-Analysis",
        (
            "$env:PYTHONPATH='src'; py -3.12 -X utf8 -B -m unittest discover -s tests -v",
            "py -3.12 -X utf8 -B scripts\\validate_project.py",
            "qa_docx.py Documents\\References\\PROJECT-KNOWLEDGE.docx --render-dir ... --preview-dir ...",
        ),
        (
            "15/15 unit test PASS; cảnh báo Streamlit bare mode không làm test fail.",
            "Validator PASS: 43 file và 5 checkpoint tại thời điểm chạy.",
            "DOCX structural/render/visual QA: 1/1 PASS, 3 trang; không cần TOC cho brief 3 trang.",
        ),
        (
            "Documents/References/PROJECT-KNOWLEDGE.docx",
            "scripts/validate_project.py",
            ".agents/skills/course-project-delivery/SKILL.md",
            "AGENTS.md",
        ),
        "Khóa data dictionary và versioned run table trước phân tích paired/slice/drift.",
    ),
    ProjectResult(
        "DS201-Temporal-Retrieval-Learning",
        (
            "$env:PYTHONPATH='src'; py -3.12 -X utf8 -B -m unittest discover -s tests -v",
            "py -3.12 -X utf8 -B scripts\\validate_project.py",
            "qa_docx.py Documents\\References\\PROJECT-KNOWLEDGE.docx --render-dir ... --preview-dir ...",
        ),
        (
            "15/15 unit test PASS; cảnh báo Streamlit bare mode không làm test fail.",
            "Validator PASS: 43 file và 5 checkpoint tại thời điểm chạy.",
            "DOCX structural/render/visual QA: 1/1 PASS, 3 trang; không cần TOC cho brief 3 trang.",
        ),
        (
            "Documents/References/PROJECT-KNOWLEDGE.docx",
            "scripts/validate_project.py",
            ".agents/skills/course-project-delivery/SKILL.md",
            "AGENTS.md",
        ),
        "Chốt Vietnamese temporal pilot, non-neural controls và evaluator trước small-DL resource probe.",
    ),
    ProjectResult(
        "DS314-Evidence-Grounded-Retrieval",
        (
            "$env:PYTHONPATH='src'; py -3.12 -X utf8 -B -m unittest discover -s tests -v",
            "py -3.12 -X utf8 -B scripts\\validate_project.py",
            "qa_docx.py Documents\\References\\PROJECT-KNOWLEDGE.docx --render-dir ... --preview-dir ...",
        ),
        (
            "15/15 unit test PASS; cảnh báo Streamlit bare mode không làm test fail.",
            "Validator PASS: 43 file và 5 checkpoint tại thời điểm chạy.",
            "DOCX structural/render/visual QA: 1/1 PASS, 3 trang; không cần TOC cho brief 3 trang.",
        ),
        (
            "Documents/References/PROJECT-KNOWLEDGE.docx",
            "scripts/validate_project.py",
            ".agents/skills/course-project-delivery/SKILL.md",
            "AGENTS.md",
        ),
        "Chốt corpus tiếng Việt, access scope, qrels và lexical baseline trước dense/rerank challenger.",
    ),
    ProjectResult(
        "Evidence-Grounded-Text-Agent",
        (
            "uv lock --check",
            "uv run --frozen --no-sync ruff format --check .",
            "uv run --frozen --no-sync ruff check .",
            "uv run --frozen --no-sync mypy",
            "uv run --frozen --no-sync pytest --cov=text_retrieval_agent --cov=legal_ir --cov=legal_qa --cov=ServiceAdapters --cov-report=term-missing -p no:cacheprovider",
            "uv run --frozen --no-sync python skills\\retrieval-delivery\\scripts\\validate_project.py",
        ),
        (
            "Lock PASS; Ruff format 83 file PASS; Ruff lint PASS; mypy 28 source file PASS.",
            "30/30 test PASS; coverage 92,65%, vượt gate 90%.",
            "Validator PASS: structure, allowlist và local-data policy.",
        ),
        (
            "Source/Shared/ServiceAdapters/inference.py",
            "skills/retrieval-delivery/scripts/validate_project.py",
            "AGENTS.md",
        ),
        "Chỉ mở neural/service candidate sau corpus/spec/split và access/citation evaluator được khóa.",
    ),
    ProjectResult(
        "DSC-TinDipLaPo",
        (
            "uv lock --check; uv run --frozen --no-sync ruff format --check .; uv run --frozen --no-sync ruff check .; uv run --frozen --no-sync mypy",
            "uv run --frozen --no-sync pytest --cov=text_retrieval_agent --cov-report=term-missing",
            "uv run --frozen --no-sync python skills\\retrieval-delivery\\scripts\\validate_project.py",
            "DSC2026: py -3.12 -X utf8 -B -m unittest discover -s tests -v; offline fixture CLI",
            "qa_docx.py <2 DSC2026 DOCX> --render-dir ... --preview-dir ...",
        ),
        (
            "Root: lock, Ruff format 99 file, lint, mypy 25 file và validator đều PASS.",
            "Root: 25/25 test PASS; coverage 93,95%, vượt gate 90%.",
            "DSC2026: 3/3 test PASS; fixture offline tạo 2 prediction gồm 1 citation và 1 abstention.",
            "Hai DOCX một trang: structural/render/visual QA 2/2 PASS, title metadata đầy đủ.",
        ),
        (
            "Documents/General/RULE.md",
            "skills/retrieval-delivery/SKILL.md",
            "DSC2026/Documents/docx/QA.md",
        ),
        "Không gọi fixture là điểm DSC; khóa organizer data, model allowlist và evaluation contract trước trial.",
    ),
    ProjectResult(
        "Multimodal-Asset-Retrieval",
        (
            "uv lock --check; uv run --frozen --no-sync ruff format --check .; uv run --frozen --no-sync ruff check .; uv run --frozen --no-sync mypy",
            "uv run --frozen --no-sync pytest --cov=multimodal_retrieval --cov-report=term-missing",
            "python skills\\retrieval-delivery\\scripts\\validate_project.py",
            "qa_docx.py <canonical DOCX> --render-dir ... --preview-dir ...",
        ),
        (
            "Lock, Ruff format 52 file, lint, mypy 18 file và validator đều PASS.",
            "11/11 test PASS; coverage 92,13%, vượt gate 90%.",
            "DOCX structural/render/visual QA: 1/1 PASS, 2 trang.",
        ),
        (
            "Documents/General/RULE.md",
            "skills/retrieval-delivery/SKILL.md",
            "AGENTS.md",
        ),
        "Giữ capability routing và provenance; chưa mở model/data/inference hoặc benchmark.",
    ),
)


ERRORS = {
    "long-range-temporal-alignment": """# Error — các run không hợp lệ trong closeout

## Phạm vi và hiện tượng

- Expected: pipeline, logging lineage, CLI dry-run và format gate đều hợp lệ.
- Actual: pipeline thất bại `1/45`; Black lần lượt báo 21 file scaffold và 8 script legacy; migration CLI exit 2; log V1 đầu thiếu trường canonical nên test dashboard và validator FAIL.

## Chẩn đoán và khắc phục

- `first_bad`: thay `TemporalInterval.start` từ `HistoricalYear` sang `HistoricalDate`; `last_known_good`: suite trước thay đổi.
- Nguyên nhân: pipeline chưa đọc `.start.year`; phạm vi Black ban đầu chưa gồm script legacy; CLI contract bị suy đoán; generator gộp metadata thay vì dùng trường canonical.
- Khắc phục: dùng `.start.year`; format toàn bộ Python; gọi script không cờ để dry-run; rollback đúng log mới và tái tạo theo schema.
- Regression: test, validator, dashboard lineage và Black đều phải PASS sau khi log V1 hợp lệ được tạo lại. Không có dataset/model/network/training/benchmark.
- Rủi ro còn lại: mọi thay đổi schema granularity/temporal kind phải có test contract và smoke pipeline.
""",
    "Evidence-Grounded-Text-Agent": """# Error — Ruff format và mypy adapter contract

## Phạm vi và hiện tượng

- Expected: Ruff format và mypy chấp nhận external inference contract.
- Actual: Ruff format FAIL ở 2 file; mypy FAIL 3 lỗi callable/protocol.
- Ruff format FAIL tại `Source/Shared/ServiceAdapters/inference.py` và validator local skill.
- Mypy FAIL 3 lỗi: callable test không thỏa `InferenceTransport` vì protocol để tên tham số có thể truyền keyword.

## Chẩn đoán và khắc phục

- `first_bad`: external inference scaffold trước closeout; `last_known_good`: baseline trước adapter mới.
- Nguyên nhân: protocol ngầm yêu cầu tên tham số tương thích dù transport được inject theo vị trí.
- Khắc phục tại lớp contract: đánh dấu hai tham số `InferenceTransport.__call__` positional-only; Ruff format đúng hai file.
- Regression: Ruff format/lint PASS, mypy 28 file PASS, 30/30 test PASS, coverage 92,65%, validator PASS.
- Không có corpus/model/network/inference/training/benchmark; rủi ro còn lại là transport thật vẫn phải qua privacy/provenance/schema tests.
""",
    "DSC-TinDipLaPo": """# Error — DOCX QA lỗi trên tài liệu một trang

## Phạm vi và hiện tượng

- Expected: helper render/inspect hai DOCX một trang.
- Actual: `IndexError` khi tính median của danh sách các trang đứng trước trang cuối nhưng danh sách rỗng.
- `first_bad`: nhánh underfilled-last-page trong global `author-validate-docx`; `last_known_good`: structural/round-trip QA không render.

## Chẩn đoán và khắc phục

- Nguyên nhân: guard `result.pages > 1` đặt sau phép truy cập danh sách.
- Khắc phục: chỉ tính `preceding_lengths` và cảnh báo trang cuối khi PDF có hơn một trang; bổ sung title metadata bằng backup và thay thế nguyên tử.
- Regression: headless render/structural/visual QA 2/2 PASS, mỗi file 1 trang, không còn warning metadata.
- Không chạy remote model, organizer data, training hoặc benchmark; fixture CLI vẫn chỉ là smoke evidence.
""",
}

COURSE_LOG_ERROR = """# Error — checkpoint chưa khớp schema course validator

## Triệu chứng

- Expected: checkpoint mới qua validator và Streamlit lineage gate.
- Actual: thiếu bốn heading canonical, `Evidence class` và mục `Cảnh báo`; validator FAIL trước mọi model/data run.

## Cấu hình và đầu vào

- Đầu vào là log V1 vừa sinh; không có corpus, model, network, training hoặc benchmark.
- `first_bad`: generator dùng schema chung; `last_known_good`: checkpoint ngày 25-08-26.

## Nguyên nhân

- Nguyên nhân: chưa đọc contract riêng của course validator trước khi render log.

## Khắc phục hoặc rollback

- Khắc phục: rollback đúng log chưa hợp lệ; thêm heading, `Evidence class`, lineage và warning section; tạo lại V1 bằng fail-if-exists.
- Rollback không đụng log cũ hoặc artifact dự án.

## Regression và trạng thái

- Regression: validator course phải PASS sau khi tái tạo; test 15/15 trước đó vẫn là bằng chứng riêng.
- Trạng thái: `RESOLVED`; rủi ro còn lại là mọi họ repository phải preflight validator contract riêng.
"""

SCHEMA_REPAIR_ERROR = """# Error — generator chưa tôn trọng schema theo từng họ repository

## Phạm vi và hiện tượng

- Expected: 7 external validator PASS sau khi ghi log.
- Actual: 4 course validator và Evidence validator FAIL; DSC và Multimodal PASS.

## Chẩn đoán và khắc phục

- `first_bad`: batch log V1 đầu; `last_known_good`: validator trước khi thêm log.
- Nguyên nhân: generator dùng một schema chung; course yêu cầu heading/lineage/warning, Evidence yêu cầu `Loại` và `Evidence class`.
- Khắc phục: rollback đúng 6 log FAIL, giữ 5 log PASS; render theo contract từng họ và thêm preflight marker.
- Regression: chạy lại 7 external validator và current dashboard/validator; không có data/model/network/training/benchmark.
- Rủi ro còn lại: validator contract phải được đọc trước mọi batch migration mới.
"""


def sha256(path: Path) -> str:
    """Tính SHA-256 cho artifact hiện hữu."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(root: Path) -> tuple[str, str, str]:
    """Thu revision, branch và worktree summary bằng lệnh read-only."""

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "--no-optional-locks", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip()

    revision = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current") or "DETACHED"
    lines = run("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    untracked = sum(line.startswith("??") for line in lines)
    return (
        revision,
        branch,
        f"dirty={len(lines)}; untracked={untracked}" if lines else "clean",
    )


def artifact_lines(result: ProjectResult, root: Path) -> str:
    """Render artifact hashes và dừng nếu artifact thiếu."""
    lines = []
    for relative in result.artifacts:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu artifact: {path}")
        lines.append(f"- `{relative}`: `{sha256(path)}`")
    return "\n".join(lines)


def render_checkpoint(result: ProjectResult, captured_at: str, *, version: int = 1) -> str:
    """Render checkpoint tĩnh, không nâng thành model benchmark."""
    root = PROJECTS_ROOT / result.name
    revision, branch, worktree = git_state(root)
    commands = "\n".join(f"- `{item}`" for item in result.commands)
    outcomes = "\n".join(f"- {item}" for item in result.results)
    return f"""# Checkpoint — closeout tài liệu, scaffold và quality gates

## Định danh

- Project: {result.name}
- Ngày: {DATE_TOKEN}
- Thời điểm: `{captured_at}` (`Asia/Bangkok`)
- Phiên bản: V{version}
- Loại: checkpoint
- Owner: local repository owner
- Git: `{revision}`; branch `{branch}`; worktree `{worktree}`
- Evidence class: `static_check|smoke_test|rendered_document_qa`
- Mức bằng chứng: `static_check|smoke_test|rendered_document_qa`; GPU không dùng.
- Kết quả: `PASS` cho các gate đã liệt kê.
- `anchor_checkpoint|parent_checkpoint|hypothesis_id|evaluation_contract_hash`: `NOT_APPLICABLE` vì không chạy measured comparison/benchmark.

## Cấu hình hệ thống

- System: {SYSTEM_PROFILE}
- GPU không dùng; model/data/service đều tắt.

## Lệnh và cấu hình chạy

{commands}

## Kết quả

{outcomes}

## Artifact và lineage

{artifact_lines(result, root)}

## Cảnh báo

- Dataset/model download, remote inference, training, measured quality, latency/VRAM benchmark và production claim: `NOT_RUN`.

## Việc tiếp theo

- {result.next_gate}
- Replay bằng các lệnh trên; rollback thay đổi tài liệu dùng `.migration-backups`, không sửa log cũ.
"""


def render_error(name: str, captured_at: str, *, version: int = 1, body: str | None = None) -> str:
    """Gắn metadata tối thiểu vào error thực tế đã được khắc phục."""
    root = PROJECTS_ROOT / name
    revision, branch, worktree = git_state(root)
    metadata = f"""
## Định danh

- Project: {name}
- Ngày: {DATE_TOKEN}
- Thời điểm: `{captured_at}` (`Asia/Bangkok`)
- Phiên bản: V{version}
- Loại: error
- Evidence class: `measured_operational_error`
- Git: `{revision}`; branch `{branch}`; worktree `{worktree}`
- System: {SYSTEM_PROFILE}
- Trạng thái: `RESOLVED`; log đã loại secret.
"""
    payload = (body if body is not None else ERRORS[name]).rstrip()
    first_heading_end = payload.find("\n")
    return payload[:first_heading_end] + "\n" + metadata + payload[first_heading_end:] + "\n"


def build_records(
    selected_projects: set[str] | None = None,
    *,
    include_current_repair: bool = False,
    include_vietnam_checkpoints: bool = False,
    include_audit_checkpoint: bool = False,
) -> list[tuple[Path, str]]:
    """Preflight exact V1 targets và tạo nội dung UTF-8."""
    captured_at = datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds")
    records: list[tuple[Path, str]] = []
    selected = (
        selected_projects
        if selected_projects is not None
        else {result.name for result in PROJECT_RESULTS}
    )
    unknown = selected - {result.name for result in PROJECT_RESULTS}
    if unknown:
        raise ValueError(f"Project ngoài allowlist: {sorted(unknown)}")
    for result in PROJECT_RESULTS:
        if result.name not in selected:
            continue
        general = (PROJECTS_ROOT / result.name / "Documents" / "General").resolve()
        if not (general / "RULE.md").is_file():
            raise FileNotFoundError(f"Thiếu RULE.md: {general}")
        checkpoint = general / f"{result.name}_Checkpoints_{DATE_TOKEN}_V1.md"
        records.append((checkpoint, render_checkpoint(result, captured_at)))
        if result.name in ERRORS or result.name in COURSE_NAMES:
            error = general / f"{result.name}_Error_{DATE_TOKEN}_V1.md"
            body = COURSE_LOG_ERROR if result.name in COURSE_NAMES else None
            records.append((error, render_error(result.name, captured_at, version=1, body=body)))
    if include_current_repair:
        name = "long-range-temporal-alignment"
        general = PROJECTS_ROOT / name / "Documents" / "General"
        target = general / f"{name}_Error_{DATE_TOKEN}_V2.md"
        records.append(
            (
                target,
                render_error(
                    name,
                    captured_at,
                    version=2,
                    body=SCHEMA_REPAIR_ERROR,
                ),
            )
        )
    if include_vietnam_checkpoints:
        integration_names = COURSE_NAMES | {"long-range-temporal-alignment"}
        for result in PROJECT_RESULTS:
            if result.name not in integration_names:
                continue
            general = PROJECTS_ROOT / result.name / "Documents" / "General"
            target = general / f"{result.name}_Checkpoints_{DATE_TOKEN}_V2.md"
            records.append((target, render_checkpoint(result, captured_at, version=2)))
    if include_audit_checkpoint:
        result = next(
            item for item in PROJECT_RESULTS if item.name == "long-range-temporal-alignment"
        )
        general = PROJECTS_ROOT / result.name / "Documents" / "General"
        target = general / f"{result.name}_Checkpoints_{DATE_TOKEN}_V3.md"
        records.append((target, render_checkpoint(result, captured_at, version=3)))
    for target, content in records:
        if target.exists():
            raise FileExistsError(f"Không ghi đè log: {target}")
        content.encode("utf-8", errors="strict")
        lowered = content.casefold()
        version_match = target.stem.rsplit("_V", 1)[-1]
        common = (
            "project:",
            "thời điểm:",
            f"phiên bản: v{version_match}",
            "loại:",
            "evidence class:",
        )
        missing = [marker for marker in common if marker not in lowered]
        if "_checkpoints_" in target.name.casefold():
            missing.extend(
                marker for marker in ("mức bằng chứng:", "kết quả:") if marker not in lowered
            )
        else:
            missing.extend(
                marker
                for marker in (
                    "expected:",
                    "actual:",
                    "nguyên nhân:",
                    "khắc phục",
                    "regression",
                )
                if marker not in lowered
            )
        if missing:
            raise ValueError(f"Log thiếu marker {missing}: {target}")
        project_name = target.name.split("_Checkpoints_", 1)[0].split("_Error_", 1)[0]
        if project_name in COURSE_NAMES:
            headings = (
                (
                    "# Checkpoint",
                    "## Định danh",
                    "## Cấu hình hệ thống",
                    "## Lệnh và cấu hình chạy",
                    "## Kết quả",
                    "## Artifact và lineage",
                    "## Việc tiếp theo",
                    "## Cảnh báo",
                )
                if "_Checkpoints_" in target.name
                else (
                    "# Error",
                    "## Định danh",
                    "## Triệu chứng",
                    "## Cấu hình và đầu vào",
                    "## Nguyên nhân",
                    "## Khắc phục hoặc rollback",
                    "## Regression và trạng thái",
                )
            )
            absent = [heading for heading in headings if heading not in content]
            if absent:
                raise ValueError(f"Course log thiếu heading {absent}: {target}")
    return records


def apply_records(records: list[tuple[Path, str]]) -> None:
    """Stage rồi hard-link fail-if-exists; rollback target mới khi lỗi."""
    staged: list[tuple[Path, Path]] = []
    created: list[Path] = []
    try:
        for target, content in records:
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content.rstrip() + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            staged.append((temporary, target))
        for temporary, target in staged:
            os.link(temporary, target)
            created.append(target)
            temporary.unlink()
    except Exception:
        for temporary, _ in staged:
            if temporary.exists():
                temporary.unlink()
        for target in reversed(created):
            target.unlink(missing_ok=True)
        raise


def main() -> int:
    """In dry-run hoặc ghi toàn bộ target sau preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--projects",
        nargs="*",
        help="Exact project names cần tạo V1; mặc định là toàn bộ allowlist.",
    )
    parser.add_argument("--include-current-repair", action="store_true")
    parser.add_argument("--include-vietnam-checkpoints", action="store_true")
    parser.add_argument("--include-audit-checkpoint", action="store_true")
    args = parser.parse_args()
    records = build_records(
        set(args.projects) if args.projects is not None else None,
        include_current_repair=args.include_current_repair,
        include_vietnam_checkpoints=args.include_vietnam_checkpoints,
        include_audit_checkpoint=args.include_audit_checkpoint,
    )
    for target, content in records:
        print(
            f"{'APPLY' if args.apply else 'DRY-RUN'} {target} sha256={hashlib.sha256(content.encode()).hexdigest()}"
        )
    if args.apply:
        apply_records(records)
        print(f"created={len(records)}; overwritten=0")
    else:
        print(f"planned={len(records)}; pass --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
