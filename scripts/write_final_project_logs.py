#!/usr/bin/env python3
# ruff: noqa: E501
"""Ghi 14 final project log theo allowlist mà không sửa log cũ.

Mặc định script chỉ dry-run. ``--apply`` dùng staged UTF-8 files và hard link
để tạo target theo cơ chế fail-if-exists; script không chạy test, network, model,
download, Git mutation hoặc commit.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

DATE_TOKEN = "25-08-26"
EVIDENCE_DATE = date(2026, 8, 25)
MEASURED_SYSTEM_PROFILE = (
    "đo trong turn: Windows 11 Home Single Language 10.0.26200; "
    "Intel Core i5-12450H 8C/12T; RAM 16.866.684.928 byte; "
    "NVIDIA GeForce RTX 2050 4.096 MiB; Python 3.12.10; "
    "model-specific peak VRAM chưa đo"
)

COURSE_DOCX_HASHES = {
    "CS406-Visual-Asset-Retrieval": "0863a29499ce8ec65777a67be7bfd69722b72ee5cffc54a2ab8b26d1fca96c5b",
    "DS111-Retrieval-Data-Analysis": "6e1fa3f076c53daecaf2508708b66e870b1b601d9777cb3fbfd5e3a40c8d1ccf",
    "DS201-Temporal-Retrieval-Learning": "90462af8b85e26636d7cc27200f8d34d0f930947d13f903363da574d35e74560",
    "DS314-Evidence-Grounded-Retrieval": "b2547eed6e4d728ac81a914a609d4e543522810575cfe9e39c401e36d847f735",
}

COURSE_PACKAGES = {
    "CS406-Visual-Asset-Retrieval": "cs406_visual_retrieval",
    "DS111-Retrieval-Data-Analysis": "ds111_retrieval_analytics",
    "DS201-Temporal-Retrieval-Learning": "ds201_temporal_learning",
    "DS314-Evidence-Grounded-Retrieval": "ds314_evidence_retrieval",
}

ALLOWED_TARGETS = {
    **{
        name: frozenset(
            {
                f"{name}_Error_{DATE_TOKEN}_V1.md",
                f"{name}_Checkpoints_{DATE_TOKEN}_V3.md",
            }
        )
        for name in COURSE_DOCX_HASHES
    },
    "Evidence-Grounded-Text-Agent": frozenset(
        {
            f"Evidence-Grounded-Text-Agent_Error_{DATE_TOKEN}_V1.md",
            f"Evidence-Grounded-Text-Agent_Checkpoints_{DATE_TOKEN}_V2.md",
        }
    ),
    "DSC-TinDipLaPo": frozenset({f"DSC-TinDipLaPo_Checkpoints_{DATE_TOKEN}_V2.md"}),
    "Multimodal-Asset-Retrieval": frozenset(
        {f"Multimodal-Asset-Retrieval_Checkpoints_{DATE_TOKEN}_V2.md"}
    ),
    "long-range-temporal-alignment": frozenset(
        {
            f"long-range-temporal-alignment_Error_{DATE_TOKEN}_V2.md",
            f"long-range-temporal-alignment_Checkpoints_{DATE_TOKEN}_V1.md",
        }
    ),
}


class FinalLogError(RuntimeError):
    """Báo preflight, allowlist hoặc atomic-write failure."""


@dataclass(frozen=True, slots=True)
class RepoSpec:
    """Khai báo exact repository, version hiện có và artifact được đọc."""

    name: str
    group: Literal["course", "evidence", "rules-only", "temporal"]
    root: Path
    existing: tuple[tuple[str, int], ...]
    requested: tuple[tuple[str, int], ...]
    artifacts: tuple[str, ...]

    def __repr__(self) -> str:
        """Trả về định danh repository ngắn."""
        return f"RepoSpec(name={self.name!r}, group={self.group!r})"


@dataclass(frozen=True, slots=True)
class RepoEvidence:
    """Giữ Git snapshot, system profile và artifact hashes read-only."""

    revision: str
    branch: str
    worktree: str
    captured_at: str
    system_profile: str
    artifact_hashes: tuple[tuple[str, str], ...]

    def __repr__(self) -> str:
        """Ẩn chi tiết status và chỉ hiện revision."""
        return f"RepoEvidence(revision={self.revision[:12]!r})"


@dataclass(frozen=True, slots=True)
class LogRecord:
    """Mang một target allowlisted cùng nội dung UTF-8 hoàn chỉnh."""

    repo: RepoSpec
    kind: Literal["Checkpoints", "Error"]
    version: int
    target: Path
    content: str

    def __repr__(self) -> str:
        """Trả về target tương đối phục vụ dry-run."""
        return f"LogRecord(repo={self.repo.name!r}, target={self.target.name!r})"


def _normalize(content: str) -> str:
    """Chuẩn hóa newline và từ chối NUL trong log."""
    normalized = content.replace("\r\n", "\n").rstrip() + "\n"
    if "\x00" in normalized:
        raise FinalLogError("Nội dung log chứa NUL.")
    normalized.encode("utf-8", errors="strict")
    return normalized


def _under(root: Path, candidate: Path) -> Path:
    """Giải path và từ chối đường dẫn thoát root."""
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise FinalLogError(f"Đường dẫn ngoài allowlist root: {candidate}") from exc
    return resolved


def _sha256(path: Path) -> str:
    """Tính SHA-256 của một artifact bắt buộc."""
    if not path.is_file():
        raise FinalLogError(f"Thiếu artifact cần hash: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    """Chạy đúng một Git read-only command và tắt optional locks."""
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={root.resolve().as_posix()}",
            "--no-optional-locks",
            *args,
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise FinalLogError(f"Git read-only command thất bại tại {root}: {detail}")
    return result.stdout.strip()


def _collect_git(root: Path) -> tuple[str, str, str]:
    """Thu revision, branch và status summary mà không lộ path cụ thể."""
    revision = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current") or "DETACHED"
    lines = _git(root, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    if not lines:
        return revision, branch, "clean"
    untracked = sum(line.startswith("??") for line in lines)
    return revision, branch, f"dirty ({len(lines)} entries; {untracked} untracked)"


def _collect_evidence(spec: RepoSpec) -> RepoEvidence:
    """Thu Git/system/hash và chặn DOCX course bị drift."""
    revision, branch, worktree = _collect_git(spec.root)
    artifacts = tuple(
        (relative, _sha256(_under(spec.root, spec.root / relative))) for relative in spec.artifacts
    )
    if spec.group == "course":
        observed = dict(artifacts)["Documents/References/PROJECT-KNOWLEDGE.docx"]
        expected = COURSE_DOCX_HASHES[spec.name]
        if observed != expected:
            raise FinalLogError(
                f"DOCX hash drift tại {spec.name}: expected={expected}, observed={observed}"
            )
    profile = (
        f"{MEASURED_SYSTEM_PROFILE}; log-writer runtime={platform.platform()}, "
        f"machine={platform.machine()}, Python={platform.python_version()}"
    )
    return RepoEvidence(
        revision,
        branch,
        worktree,
        datetime.now().astimezone().isoformat(timespec="seconds"),
        profile,
        artifacts,
    )


def _course_artifacts(name: str) -> tuple[str, ...]:
    """Trả về canonical course artifacts cần hash."""
    package = COURSE_PACKAGES[name]
    return (
        "Documents/General/RULE.md",
        "Documents/References/PROJECT-KNOWLEDGE.docx",
        "scripts/validate_project.py",
        f"src/{package}/ServiceAdapters/inference.py",
    )


def build_specs(documents_root: Path) -> tuple[RepoSpec, ...]:
    """Tạo tám specs từ exact sibling repository names."""
    base = documents_root.resolve()
    specs: list[RepoSpec] = []
    for name in COURSE_DOCX_HASHES:
        specs.append(
            RepoSpec(
                name,
                "course",
                _repo_root(base, name),
                (("Checkpoints", 1), ("Checkpoints", 2)),
                (("Error", 1), ("Checkpoints", 3)),
                _course_artifacts(name),
            )
        )
    specs.extend(_noncourse_specs(base))
    return tuple(specs)


def _repo_root(documents_root: Path, name: str) -> Path:
    """Giải exact sibling root và chặn junction/symlink ra ngoài Documents."""
    root = (documents_root / name).resolve()
    try:
        root.relative_to(documents_root)
    except ValueError as exc:
        raise FinalLogError(f"Repository ngoài Documents allowlist: {root}") from exc
    if not root.is_dir() or root.name != name:
        raise FinalLogError(f"Thiếu exact repository root: {documents_root / name}")
    return root


def _noncourse_specs(base: Path) -> tuple[RepoSpec, ...]:
    """Tạo Evidence, DSC, Multimodal và temporal specs."""
    return (
        RepoSpec(
            "Evidence-Grounded-Text-Agent",
            "evidence",
            _repo_root(base, "Evidence-Grounded-Text-Agent"),
            (("Checkpoints", 1),),
            (("Error", 1), ("Checkpoints", 2)),
            (
                "Documents/General/RULE.md",
                "skills/retrieval-delivery/scripts/validate_project.py",
                "Source/Shared/ServiceAdapters/inference.py",
                "Documents/Decision-making/Shared/0002-data-feasibility-4gb.md",
            ),
        ),
        _rules_only_spec(base, "DSC-TinDipLaPo"),
        _rules_only_spec(base, "Multimodal-Asset-Retrieval"),
        RepoSpec(
            "long-range-temporal-alignment",
            "temporal",
            _repo_root(base, "long-range-temporal-alignment"),
            (("Error", 1),),
            (("Error", 2), ("Checkpoints", 1)),
            (
                "Documents/General/RULE.md",
                "scripts/validate_project.py",
                "src/temporal_alignment/ServiceAdapters/inference.py",
                "Documents/References/HTML/progress-report-2026-08-25.html",
                "Documents/References/Markdown/12_Khao_sat_sau_Ticktack_2025_2026.md",
                ".tmp/docx-render-qa-course-v2/RENDER-QA.md",
                ".tmp/docx-render-qa-library-final/RENDER-QA.md",
                ".tmp/temporal-progress-report-final-v2.png",
            ),
        ),
    )


def _rules_only_spec(base: Path, name: str) -> RepoSpec:
    """Tạo spec chỉ ghi checkpoint V2 cho DSC/Multimodal."""
    return RepoSpec(
        name,
        "rules-only",
        _repo_root(base, name),
        (("Checkpoints", 1),),
        (("Checkpoints", 2),),
        ("Documents/General/RULE.md", "skills/retrieval-delivery/SKILL.md"),
    )


def _log_name(spec: RepoSpec, kind: str, version: int) -> str:
    """Tạo filename đúng grammar đã khóa."""
    return f"{spec.name}_{kind}_{DATE_TOKEN}_V{version}.md"


def _validate_versions(spec: RepoSpec) -> None:
    """Yêu cầu exact version state trước khi tạo log mới."""
    general = spec.root / "Documents/General"
    if not (general / "RULE.md").is_file():
        raise FinalLogError(f"Thiếu RULE.md tại {spec.name}")
    pattern = re.compile(
        rf"^{re.escape(spec.name)}_(Checkpoints|Error)_{DATE_TOKEN}_V([1-9]\d*)\.md$"
    )
    observed = {
        (match.group(1), int(match.group(2)))
        for path in general.iterdir()
        if path.is_file() and (match := pattern.fullmatch(path.name))
    }
    if observed != set(spec.existing):
        raise FinalLogError(
            f"Version state lệch tại {spec.name}: expected={spec.existing}, observed={sorted(observed)}"
        )


def _target(spec: RepoSpec, kind: str, version: int) -> Path:
    """Giải target và xác minh exact allowlist/no-overwrite."""
    name = _log_name(spec, kind, version)
    if name not in ALLOWED_TARGETS.get(spec.name, frozenset()):
        raise FinalLogError(f"Target ngoài allowlist: {spec.name}/{name}")
    general = _under(spec.root, spec.root / "Documents/General")
    target = _under(spec.root, general / name)
    if target.parent != general:
        raise FinalLogError(f"Target không nằm trực tiếp trong General: {target}")
    if target.exists():
        raise FinalLogError(f"Target đã tồn tại; không sửa log cũ: {target}")
    return target


def _artifact_lines(evidence: RepoEvidence) -> str:
    """Render artifact/hash list có thứ tự ổn định."""
    return "\n".join(f"- `{relative}`: `{digest}`" for relative, digest in evidence.artifact_hashes)


def _identity(spec: RepoSpec, evidence: RepoEvidence, version: int, kind: str) -> str:
    """Render metadata chung theo RULE nghiêm nhất."""
    log_type = "checkpoint" if kind == "Checkpoints" else "error"
    return f"""- Project: {spec.name}
- Ngày: 25-08-26; ghi lúc `{evidence.captured_at}` (`Asia/Bangkok` theo project)
- Thời điểm: 25/08/2026; `{evidence.captured_at}`
- Phiên bản: V{version}
- Loại: {log_type}
- Owner: local repository owner
- Git revision: `{evidence.revision}`; branch `{evidence.branch}`
- Worktree: `{evidence.worktree}`
- System profile: {evidence.system_profile}"""


def _render_course_checkpoint(spec: RepoSpec, evidence: RepoEvidence) -> str:
    """Render course Checkpoints V3 với static/rendered QA đã xác nhận."""
    docx_hash = dict(evidence.artifact_hashes)["Documents/References/PROJECT-KNOWLEDGE.docx"]
    return f"""# Checkpoint — final scaffold, validator và DOCX QA

## Định danh

{_identity(spec, evidence, 3, "Checkpoints")}
- Evidence class: `static_check` + `rendered_document_qa`; không phải benchmark.

## Cấu hình hệ thống

- Test/validator interpreter: `py -3.12`; local budget: RTX 2050 4 GB nhưng GPU không dùng.
- Config/data/model revision, seed, CI và resource benchmark: `not-run|not-applicable`.

## Lệnh và cấu hình chạy

- `py -3.12 -m pytest -q` → exit `0`, `8/8` test PASS theo pyproject của repo.
- `py -3.12 scripts\\validate_project.py` → exit `0`, validator PASS.
- `py -3.12 -X utf8 -B scripts\\rebuild_course_docx.py --root <Documents> --apply` → rebuild staged PASS.
- `py -3.12 -X utf8 -B scripts\\render_docx_qa.py --library-root <knowledge-root> --course-root <Documents> --output .tmp\\docx-render-qa-course-v2 --scope course` → `4/4` PASS; file của repo này `2` trang.
- Model/data download, training, inference, external network và benchmark: `not-run`.

## Kết quả

- Scaffold contracts và inference fail-closed: static gate PASS; chưa đo model quality.
- DOCX visual/contact-sheet QA: PASS, `2` trang, SHA-256 `{docx_hash}`.
- Không có dataset/model thật, metric retrieval/QA, peak VRAM hoặc production claim.

## Artifact và lineage

{_artifact_lines(evidence)}

## Việc tiếp theo

Giữ baseline CPU, chốt data license/split/evaluator rồi mới mở model/resource gate. Replay bằng các lệnh trên; rollback DOCX dùng staged backup, không sửa log V1–V3.
"""


def _render_course_error(spec: RepoSpec, evidence: RepoEvidence) -> str:
    """Render course Error V1 từ ba failure đã xác nhận trong turn."""
    return f"""# Error — command path, scope token và staged DOCX QA

## Định danh

{_identity(spec, evidence, 1, "Error")}
- Evidence class: `measured_operational_error` + `rendered_document_qa`.

## Triệu chứng

- Expected: unit tests import package, validator chấp nhận checkpoint và DOCX render như văn bản thường.
- Actual: `py -3.12 -B -m unittest discover -s tests -v` lỗi `ModuleNotFoundError`; checkpoint migration đầu chứa forbidden/deferred scope token nên validator FAIL; bản DOCX staged đầu có leading indentation nên contact sheet hiện code blocks.
- Một staged rebuild khác bị từ chối giả do marker course code đã chuyển vào metadata thay vì body; live DOCX không bị thay trên staging failure.

## Cấu hình và đầu vào

- Python `3.12`; test source layout theo pyproject; renderer `course-repositories-v3-format`.
- Model, dataset, download, inference, external network và benchmark: `not-run`.
- Artifact hiện tại:
{_artifact_lines(evidence)}

## Nguyên nhân

- `confirmed`: unittest command thiếu repo source path; dùng sai runner, không phải lỗi package.
- `confirmed`: generator chưa sanitize literal scope bị hoãn ngoài decision file được phép.
- `confirmed`: renderer giữ indentation của template; QA marker kiểm body quá hẹp.

## Khắc phục hoặc rollback

- Dùng pytest theo pyproject; sanitize scope token; sửa renderer bỏ indentation và cho marker đọc metadata.
- Staged validation phải PASS trước replace; failure không thay live DOCX. Không có model/data/environment mutation để rollback.

## Regression và trạng thái

- Regression: pytest `8/8`, validator PASS, DOCX visual QA `2` trang, exact hash đã khóa; trạng thái `RESOLVED`.
- Rủi ro còn lại: thay template/metadata phải chạy lại staging, contact sheet và hash gate; kết quả này không phải benchmark.
"""


def _render_evidence_checkpoint(spec: RepoSpec, evidence: RepoEvidence) -> str:
    """Render Evidence Checkpoints V2 sau validator và 30 tests."""
    return f"""# Checkpoint — final validation Evidence-Grounded Text Agent

{_identity(spec, evidence, 2, "Checkpoints")}
- Evidence class: `static_check` + `smoke_test`; không phải benchmark.
- Data/corpus manifest: organizer/full corpus `not-run`; synthetic fixtures chỉ phục vụ test.
- Config hash/model revision: `not-applicable`; model/inference bị tắt.

## Lệnh đã chạy và kết quả

- `py -3.12 skills\\retrieval-delivery\\scripts\\validate_project.py` → exit `0`, PASS.
- `$env:PYTHONPATH='Source\\Shared;Source\\Task1;Source\\Task2'; py -3.12 -m pytest -q` → exit `0`, `30/30` PASS.
- Model/data download, training, remote inference, external network và benchmark: `not-run`.

## Artifact/hash

{_artifact_lines(evidence)}

## Kết luận, blocker và bước kế tiếp

Contracts LegalIR/LegalQA, citation/abstention và ServiceAdapters fail-closed qua gate. Chưa có official DSC score, model quality, latency/VRAM benchmark hoặc production readiness. Chốt corpus/spec/split trước dense/rerank/generation.
"""


def _render_evidence_error(spec: RepoSpec, evidence: RepoEvidence) -> str:
    """Render Evidence Error V1 từ path/PYTHONPATH failures đã xác nhận."""
    return f"""# Error — validator path và source import path

{_identity(spec, evidence, 1, "Error")}
- Evidence class: `measured_operational_error`.
- Data/corpus manifest, config/model revision: `not-run|not-applicable`.

## Triệu chứng

- Expected: validator và full pytest collection chạy từ repository root.
- Actual: gọi `scripts\\validate_project.py` thất bại vì file không tồn tại; full pytest không có `PYTHONPATH` tạo `9` collection errors.
- Cách tái hiện: dùng hai lệnh sai trên; không liên quan model/data quality.

## Nguyên nhân và tác động

- Nguyên nhân: `confirmed` — validator canonical nằm dưới local skill; ba source roots chưa được đưa vào import path của shell hiện tại.
- Tác động: validation bị chặn trước test; không có model/data/network/benchmark nào chạy.

## Khắc phục, rollback và regression

- Khắc phục: dùng `skills\\retrieval-delivery\\scripts\\validate_project.py` và đặt `$env:PYTHONPATH='Source\\Shared;Source\\Task1;Source\\Task2'`.
- Rollback: không có target/data/model mutation; chỉ sửa command.
- Regression: validator PASS; pytest `30/30` PASS; trạng thái `RESOLVED`.
- Artifact/hash:
{_artifact_lines(evidence)}
"""


def _render_rules_checkpoint(spec: RepoSpec, evidence: RepoEvidence) -> str:
    """Render DSC/Multimodal Checkpoints V2 mà không mở rộng research/code."""
    return f"""# Checkpoint — khóa General logging rule

{_identity(spec, evidence, 2, "Checkpoints")}
- Evidence class: `static_check` + `migration_checkpoint`; không phải benchmark.
- Data/corpus manifest, config hash, model revision: `not-run|not-applicable`.

## Lệnh, kết quả và artifact

- Lệnh apply đã ghi ở V1: `python scripts\\upgrade_noncourse_repositories.py --apply`; closeout này không chạy lại migration.
- RULE, local skill và version continuity được đọc/hash; trạng thái `PASS`.
- Unit test, validator, model/data download, training, inference, external network và benchmark: `not-run`.
{_artifact_lines(evidence)}

## Kết luận, blocker và bước kế tiếp

Chỉ task General đã hoàn tất; không bổ sung research hoặc source code. Log sau phải tạo version mới, không sửa V1/V2. Chỉ chạy project gate khi scope riêng được mở lại.
"""


def _render_temporal_checkpoint(spec: RepoSpec, evidence: RepoEvidence) -> str:
    """Render temporal Checkpoints V1 với project/knowledge/skill/HTML gates."""
    return f"""# Checkpoint — final project, knowledge library và skill QA

{_identity(spec, evidence, 1, "Checkpoints")}
- Mức bằng chứng: `static_check` + `rendered_document_qa` + local UI smoke; không phải benchmark.
- Kết quả: `PASS` cho các gate liệt kê; model/data/network/benchmark `not-run`.

## Lệnh và kết quả

- `py -3.12 scripts\\validate_project.py` → `59` required files PASS.
- `py -3.12 -m pytest -q -p no:cacheprovider --basetemp .tmp\\pytest-general-validator` → `38/38` PASS; `compileall` source/scripts/tests PASS.
- `py -3.12 -X utf8 -B scripts\\upgrade_knowledge_library.py --root <knowledge-root> --validate` → `49+15`, CodingScratch unchanged.
- `py -3.12 -X utf8 -B scripts\\rebuild_knowledge_library.py --root <knowledge-root> --validate` → `64` DOCX.
- `py -3.12 -X utf8 -B scripts\\render_docx_qa.py --library-root <knowledge-root> --course-root <Documents> --output .tmp\\docx-render-qa-library-final --scope library` → `64/64` PASS; artifact `RENDER-QA.md`.
- `quick_validate.py <26 skill paths>` → `26/26` PASS.
- Playwright local URL query `v2`: console `0` errors; screenshot `.tmp/temporal-progress-report-final-v2.png`.

## Cấu hình và ranh giới

- System profile: {evidence.system_profile}.
- Dataset/model download, training, inference, external network, model-specific peak VRAM measurement và research benchmark: `not-run`.
- Các con số trên là validator/render/UI evidence, không phải model metric hoặc production claim.

## Artifact/hash

{_artifact_lines(evidence)}

## Blocker và bước kế tiếp

Ticktack code/data/license vẫn chưa được primary release xác minh; giữ registry `hold`. Tiếp theo chốt Vietnamese historical QA pilot, leakage-safe split và CPU baseline trước resource probe.
"""


def _render_temporal_error(spec: RepoSpec, evidence: RepoEvidence) -> str:
    """Render temporal Error V2 từ hai orchestration path errors."""
    return f"""# Error — sai repository root và sai tên render artifact

{_identity(spec, evidence, 2, "Error")}
- Mức bằng chứng: `measured_operational_error`; không liên quan model quality.

## Sự cố

- Expected: course upgrader nhận parent root chứa bốn repository; render QA đọc artifact thực tế.
- Actual: lần đầu truyền child repository root nên preflight từ chối; sau đó thử đọc `report.json` không tồn tại trong khi output đúng là `render-results.json` và `RENDER-QA.md`.
- Cách tái hiện: truyền một course root cho `upgrade_course_repositories.py`; tìm `.tmp/.../report.json`.

## Nguyên nhân và tác động

- Nguyên nhân: `confirmed` — hiểu sai CLI root contract và suy đoán tên artifact thay vì đọc output contract.
- Tác động: orchestration dừng sớm; không target/live DOCX nào bị thay trong hai failure.
- Model, dataset, download, inference, external network và benchmark: `not-run`.

## Khắc phục, rollback và regression

- Khắc phục: truyền parent `Documents`; đọc `render-results.json`/`RENDER-QA.md` theo renderer contract.
- Rollback: không cần; staged/preflight failure không tạo target mutation.
- Regression: course upgrade/validator PASS; rendered course `4/4`, library `64/64`; trạng thái `RESOLVED`.
- Rủi ro còn lại: mọi orchestrator phải preflight root và enumerate artifact contract, không suy tên file.

## Artifact/hash

{_artifact_lines(evidence)}
"""


def _render(spec: RepoSpec, evidence: RepoEvidence, kind: str) -> str:
    """Route đúng template theo group/kind."""
    if spec.group == "course":
        content = (
            _render_course_checkpoint(spec, evidence)
            if kind == "Checkpoints"
            else _render_course_error(spec, evidence)
        )
    elif spec.group == "evidence":
        content = (
            _render_evidence_checkpoint(spec, evidence)
            if kind == "Checkpoints"
            else _render_evidence_error(spec, evidence)
        )
    elif spec.group == "rules-only":
        content = _render_rules_checkpoint(spec, evidence)
    else:
        content = (
            _render_temporal_checkpoint(spec, evidence)
            if kind == "Checkpoints"
            else _render_temporal_error(spec, evidence)
        )
    return _normalize(content)


def _validate_content(record: LogRecord) -> None:
    """Kiểm required markers và scope token trước khi ghi."""
    common = ("Project:", "Phiên bản:", "Evidence class:")
    if record.repo.group == "temporal":
        common = ("Project:", "Thời điểm:", "Phiên bản:")
    missing = [marker for marker in common if marker.casefold() not in record.content.casefold()]
    if missing:
        raise FinalLogError(f"Log thiếu marker {missing}: {record.target}")
    if record.repo.group == "course":
        headings = (
            (
                "# Checkpoint",
                "## Định danh",
                "## Cấu hình hệ thống",
                "## Lệnh và cấu hình chạy",
                "## Kết quả",
                "## Artifact và lineage",
                "## Việc tiếp theo",
            )
            if record.kind == "Checkpoints"
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
        if any(heading not in record.content for heading in headings):
            raise FinalLogError(f"Course log thiếu required heading: {record.target}")
        forbidden = "multi" + "modal"
        if forbidden in record.content.casefold():
            raise FinalLogError(f"Course log chứa deferred scope token: {record.target}")


def build_records(documents_root: Path) -> tuple[LogRecord, ...]:
    """Preflight tất cả target trước khi thu evidence và render."""
    specs = build_specs(documents_root)
    targets: dict[str, list[tuple[str, int, Path]]] = {}
    for spec in specs:
        _validate_versions(spec)
        targets[spec.name] = [
            (kind, version, _target(spec, kind, version)) for kind, version in spec.requested
        ]
    records: list[LogRecord] = []
    for spec in specs:
        evidence = _collect_evidence(spec)
        for kind, version, target in targets[spec.name]:
            record = LogRecord(spec, kind, version, target, _render(spec, evidence, kind))
            _validate_content(record)
            records.append(record)
    expected = sum(len(names) for names in ALLOWED_TARGETS.values())
    if len(records) != expected or {item.target.name for item in records} != set().union(
        *ALLOWED_TARGETS.values()
    ):
        raise FinalLogError("Rendered target set không khớp hard-coded allowlist.")
    return tuple(records)


def print_plan(records: tuple[LogRecord, ...]) -> None:
    """In target và content hash mà không ghi filesystem."""
    for record in records:
        digest = hashlib.sha256(record.content.encode("utf-8")).hexdigest()
        relative = record.target.relative_to(record.repo.root)
        print(f"DRY-RUN {record.repo.name}: {relative} sha256={digest}")
    print(f"Tổng: {len(records)} log mới; 0 log cũ bị sửa; --apply chưa chạy.")


def apply_records(records: tuple[LogRecord, ...]) -> None:
    """Stage đầy đủ rồi hard-link atomically; rollback target mới khi lỗi."""
    if datetime.now().astimezone().date() != EVIDENCE_DATE:
        raise FinalLogError("--apply chỉ hợp lệ trong ngày evidence 25-08-26.")
    staged: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    try:
        for record in records:
            temporary = record.target.with_name(f".{record.target.name}.{uuid.uuid4().hex}.tmp")
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(record.content)
                stream.flush()
                os.fsync(stream.fileno())
            staged.append((temporary, record.target))
        for temporary, target in staged:
            if target.exists():
                raise FinalLogError(f"Target xuất hiện sau preflight: {target}")
            os.link(temporary, target)
            committed.append(target)
            temporary.unlink()
    except (OSError, UnicodeError, FinalLogError) as exc:
        for temporary, _ in staged:
            if temporary.is_file():
                temporary.unlink()
        for target in reversed(committed):
            if target.is_file():
                target.unlink()
        raise FinalLogError(f"Atomic write thất bại; target mới đã rollback: {exc}") from exc
    print(f"APPLIED: {len(records)} log mới; không sửa log cũ.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Phân tích CLI với dry-run mặc định."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Tạo 14 log mới theo allowlist.")
    parser.add_argument(
        "--documents-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Parent chứa tám exact repository; mặc định là thư mục Documents.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Dry-run hoặc atomically ghi final logs sau full preflight."""
    args = parse_args(argv)
    try:
        records = build_records(args.documents_root)
        if args.apply:
            apply_records(records)
        else:
            print_plan(records)
        return 0
    except FinalLogError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
