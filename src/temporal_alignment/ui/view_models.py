"""Build deterministic UI data without importing Streamlit."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

PROJECT_NAME = "long-range-temporal-alignment"
CHECKPOINT_PATTERN = re.compile(
    rf"^{PROJECT_NAME}_Checkpoints_"
    r"(?P<date>\d{2}-\d{2}-\d{2})_V(?P<version>[1-9]\d*)\.md$"
)


class LineageError(RuntimeError):
    """Reject missing or malformed dashboard lineage artifacts."""


@dataclass(frozen=True, slots=True)
class GateView:
    """Represent one project gate in the dashboard."""

    name: str
    status: str
    evidence_class: str
    next_condition: str


@dataclass(frozen=True, slots=True)
class ArtifactLineage:
    """Describe the exact checkpoint used by the read-only UI."""

    path: str
    version: str
    sha256: str
    evidence_class: str
    warnings: tuple[str, ...]

    def rows(self) -> list[dict[str, str]]:
        """Return compact key/value rows for table-oriented UIs."""
        return [
            {"field": "path", "value": self.path},
            {"field": "version", "value": self.version},
            {"field": "sha256", "value": self.sha256},
            {"field": "evidence_class", "value": self.evidence_class},
            {"field": "warnings", "value": "; ".join(self.warnings)},
        ]


@dataclass(frozen=True, slots=True)
class DashboardViewModel:
    """Contain read-only status data for the Streamlit layer."""

    title: str
    scope: str
    warning: str
    gates: tuple[GateView, ...]
    lineage: ArtifactLineage

    def rows(self) -> list[dict[str, str]]:
        """Return serializable rows for table-oriented UIs."""
        return [asdict(gate) for gate in self.gates]


def _project_root(project_root: Path | None) -> Path:
    """Resolve the repository root without accepting a missing directory."""
    root = project_root.resolve() if project_root else Path(__file__).resolve().parents[3]
    if not root.is_dir():
        raise LineageError(f"Project root không tồn tại: {root}")
    return root


def _evidence_class(text: str) -> str:
    """Extract an explicit evidence label from a checkpoint."""
    for label in ("Mức bằng chứng:", "Evidence class:"):
        match = re.search(rf"(?im)^\s*-?\s*{re.escape(label)}\s*(.+)$", text)
        if match:
            return match.group(1).strip()
    raise LineageError("Checkpoint thiếu Mức bằng chứng/Evidence class.")


def _metadata_field(text: str, labels: tuple[str, ...], field_name: str) -> str:
    """Extract one required Markdown metadata field."""
    for label in labels:
        match = re.search(rf"(?im)^\s*-?\s*{re.escape(label)}\s*(.+)$", text)
        if match:
            return match.group(1).strip().strip("`")
    raise LineageError(f"Checkpoint thiếu trường {field_name}.")


def _lineage_warnings(text: str) -> tuple[str, ...]:
    """Return conservative warnings derived from explicit checkpoint wording."""
    lowered = text.casefold()
    warnings: list[str] = []
    if "không phải benchmark" in lowered or "not a benchmark" in lowered:
        warnings.append("Checkpoint tự khai báo không phải benchmark.")
    if "not-run" in lowered or "chưa chạy" in lowered or "chưa đo" in lowered:
        warnings.append("Checkpoint còn phép đo hoặc bước thực thi chưa chạy.")
    if "worktree" in lowered and "dirty" in lowered:
        warnings.append("Worktree tại thời điểm checkpoint chưa sạch.")
    if not warnings:
        warnings.append("Chỉ diễn giải theo mức bằng chứng ghi trong checkpoint.")
    return tuple(warnings)


def load_latest_checkpoint(project_root: Path | None = None) -> ArtifactLineage:
    """Load the newest immutable checkpoint and compute its exact hash."""
    root = _project_root(project_root)
    general = root / "Documents" / "General"
    candidates: list[tuple[datetime, int, Path]] = []
    if general.is_dir():
        for path in general.iterdir():
            match = CHECKPOINT_PATTERN.fullmatch(path.name)
            if match is None or not path.is_file():
                continue
            try:
                log_date = datetime.strptime(match.group("date"), "%d-%m-%y")
            except ValueError as exc:
                raise LineageError(f"Checkpoint có ngày không hợp lệ: {path.name}") from exc
            candidates.append((log_date, int(match.group("version")), path))
    if not candidates:
        raise LineageError("Không tìm thấy checkpoint hợp lệ cho dashboard.")

    _, version, latest = max(candidates, key=lambda item: (item[0], item[1]))
    payload = latest.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LineageError(f"Checkpoint không phải UTF-8: {latest.name}") from exc
    body_project = _metadata_field(text, ("Project:",), "Project")
    body_version = _metadata_field(text, ("Phiên bản:", "Version:"), "Phiên bản")
    expected_version = f"V{version}"
    if body_project != PROJECT_NAME:
        raise LineageError(
            f"Project trong checkpoint không khớp: {body_project!r} != {PROJECT_NAME!r}."
        )
    if body_version != expected_version:
        raise LineageError(
            f"Phiên bản trong checkpoint không khớp tên file: "
            f"{body_version!r} != {expected_version!r}."
        )
    return ArtifactLineage(
        path=latest.relative_to(root).as_posix(),
        version=expected_version,
        sha256=hashlib.sha256(payload).hexdigest(),
        evidence_class=_evidence_class(text),
        warnings=_lineage_warnings(text),
    )


def build_dashboard_view(project_root: Path | None = None) -> DashboardViewModel:
    """Return read-only gate and lineage data; never load a model or dataset."""
    return DashboardViewModel(
        title="Long-Range Temporal Alignment",
        scope="Temporal retrieval/QA tiếng Việt có bằng chứng và khả năng từ chối",
        warning="Scaffold: chưa có benchmark, dataset thật hoặc model production.",
        gates=(
            GateView("Core CPU", "smoke-only", "smoke_test", "Duy trì unit test"),
            GateView("Dữ liệu thật", "hold", "candidate", "Hoàn tất manifest và license"),
            GateView("Model", "hold", "candidate", "Resource probe và revision bất biến"),
            GateView("Ứng dụng", "scaffold", "scaffold", "Chỉ đọc artifact đã duyệt"),
        ),
        lineage=load_latest_checkpoint(project_root),
    )
