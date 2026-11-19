"""Generate four deterministic course-repository scaffolds without downloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent

from course_repo_catalog import PROJECTS, CourseProject
from course_repo_templates import (
    initial_checkpoint_relative,
    render_agents,
    render_architecture,
    render_checkpoint,
    render_common_failures,
    render_curriculum_alignment,
    render_data_model_eval,
    render_deferred,
    render_documents_readme,
    render_error_experience,
    render_feasibility,
    render_general_rule,
    render_html,
    render_knowledge,
    render_problem_contract,
    render_project_context,
    render_research,
    render_root_readme,
    render_services,
    render_skill,
    render_skill_gates,
    render_tools_decision,
    render_workflow,
)

MIT_LICENSE = """MIT License

Copyright (c) 2026 JH111334

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

MCP_CONFIG = """# Project-scoped overrides; global MCP configuration is unchanged.
[mcp_servers.filesystem]
enabled = true

[mcp_servers.memory_keeper]
enabled = true

[mcp_servers.playwright]
enabled = true

[mcp_servers.docker]
enabled = false

[mcp_servers.github]
enabled = false

[mcp_servers.powerpoint]
enabled = false

[mcp_servers.socialcrawl]
enabled = false

[mcp_servers.winston_ai]
enabled = false

[mcp_servers.youtube_knowledge]
enabled = false
"""

GITIGNORE = """# Python
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.mypy_cache/
.venv/
venv/

# Secrets and local configuration
.env
.env.*
!.env.example
*.pem
*.key

# Data, models, caches and generated artifacts
data/raw/
data/derived/
models/
checkpoints/
artifacts/
cache/
.cache/
*.bin
*.safetensors
*.pt
*.pth
*.onnx

# UI and OS
.streamlit/secrets.toml
.DS_Store
Thumbs.db

# Document QA scratch
.docx-qa/
*.pdf
"""

GITATTRIBUTES = """* text=auto
*.py text eol=lf
*.md text eol=lf
*.html text eol=lf
*.toml text eol=lf
*.json text eol=lf
*.yaml text eol=lf
*.yml text eol=lf
*.txt text eol=lf
*.docx binary
"""

ENV_EXAMPLE = """# Không điền secret thật vào file này hoặc Git.
HF_TOKEN=
HF_HOME=
KAGGLE_API_TOKEN=
KAGGLEHUB_CACHE=
PROJECT_ARTIFACT_ROOT=
"""


def pyproject(spec: CourseProject) -> str:
    """Render planned package metadata and optional dependency groups."""
    research = "\n".join(f'    "{item}",' for item in spec.research_dependencies)
    return f"""[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "{spec.repo_name.casefold()}"
version = "0.1.0"
description = "Scaffold contracts and baselines for {spec.short_title}"
readme = "README.md"
requires-python = ">=3.11"
license = {{ text = "MIT" }}
authors = [{{ name = "JH111334" }}]
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8,<10", "ruff>=0.9,<1", "mypy>=1.14,<2"]
ui = ["streamlit>=1.44,<2"]
services = ["huggingface_hub>=0.30,<2", "kagglehub>=1,<2", "httpx>=0.28,<1"]
research = [
{research}
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.11"
strict = true
"""


def requirements() -> str:
    """Render a single planned requirements entrypoint."""
    return """# Kế hoạch dependency; KHÔNG cài chỉ để kiểm tra scaffold.
# Rà license, Python/CUDA compatibility và tạo lockfile trước lần cài đầu tiên.
-e .[dev,ui,services,research]
"""


def config_json(spec: CourseProject) -> str:
    """Render an explicit scaffold-stage configuration."""
    payload = {
        "schema_version": "1.0",
        "project": spec.repo_name,
        "evidence_level": "scaffold",
        "network_enabled": False,
        "training_enabled": False,
        "stages": list(spec.stages),
        "notes": "No dataset/model download and no benchmark claim.",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def registry_json(spec: CourseProject, kind: str) -> str:
    """Render held data/model candidates without pretending ingest readiness."""
    candidates = spec.data_candidates if kind == "datasets" else spec.model_candidates
    entries = [
        {
            "name": item.name,
            "official_url": item.url,
            "accessed_on": "2026-08-24",
            "role": item.role,
            "gate_note": item.gate,
            "license_or_terms": None,
            "revision": None,
            "sha256": None,
            "reviewer": None,
            "data_role": "candidate" if kind == "datasets" else "model-candidate",
            "status": "hold",
        }
        for item in candidates
    ]
    payload = {
        "schema_version": "1.0",
        "asset_kind": kind,
        "ingest_rule": "Only approved entries with terms, immutable revision, checksum, reviewer and role are ingestible.",
        "entries": entries,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def input_schema(spec: CourseProject) -> str:
    """Render a minimal schema mirroring the Python input contract."""
    fields: dict[str, dict[str, object]]
    required: list[str]
    if spec.course_code == "CS406":
        fields = {
            "query_id": {"type": "string", "minLength": 1},
            "image_uri": {"type": "string", "minLength": 1},
            "corpus_version": {"type": "string", "minLength": 1},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
        }
        required = ["query_id", "image_uri", "corpus_version", "top_k"]
    elif spec.course_code == "DS111":
        fields = {
            "run_id": {"type": "string", "minLength": 1},
            "run_uri": {"type": "string", "minLength": 1},
            "qrels_uri": {"type": "string", "minLength": 1},
            "schema_version": {"type": "string", "minLength": 1},
        }
        required = list(fields)
    elif spec.course_code == "DS201":
        year = {
            "type": "object",
            "properties": {
                "era": {"enum": ["BCE", "CE"]},
                "year": {"type": "integer", "minimum": 1},
            },
            "required": ["era", "year"],
            "additionalProperties": False,
        }
        fields = {
            "query_id": {"type": "string", "minLength": 1},
            "text": {"type": "string", "minLength": 1},
            "start": year,
            "end": year,
            "corpus_version": {"type": "string", "minLength": 1},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
        }
        required = list(fields)
    else:
        fields = {
            "query_id": {"type": "string", "minLength": 1},
            "text": {"type": "string", "minLength": 1},
            "corpus_version": {"type": "string", "minLength": 1},
            "access_scope": {"type": "string", "minLength": 1},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
        }
        required = list(fields)
    payload = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": spec.input_type,
        "type": "object",
        "properties": fields,
        "required": required,
        "additionalProperties": False,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def output_schema(spec: CourseProject) -> str:
    """Render a conservative output envelope schema."""
    payload = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"{spec.output_type}Envelope",
        "type": "object",
        "properties": {
            "request_id": {"type": "string", "minLength": 1},
            "evidence_level": {"enum": ["scaffold", "smoke_test", "measured"]},
            "results": {"type": "array", "items": {"type": "object"}},
            "warnings": {"type": "array", "items": {"type": "string"}},
            "config_hash": {"type": "string"},
            "data_manifest_hash": {"type": "string"},
            "code_revision": {"type": "string"},
        },
        "required": [
            "request_id",
            "evidence_level",
            "results",
            "warnings",
            "config_hash",
            "data_manifest_hash",
            "code_revision",
        ],
        "additionalProperties": False,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def contracts_module(spec: CourseProject) -> str:
    """Render project-specific request/result dataclasses."""
    temporal_import = (
        "from .normalization.historical_time import HistoricalYear\n"
        if spec.course_code == "DS201"
        else ""
    )
    body = spec.contracts_code
    if temporal_import:
        body = body.replace("from .normalization.historical_time import HistoricalYear\n\n\n", "")
    return f'''"""Input and output contracts for {spec.course_code}."""

from __future__ import annotations

from dataclasses import dataclass

{temporal_import}from .errors import ProjectContractError


def _require(value: str, name: str) -> None:
    """Reject blank identifiers and text fields."""
    if not isinstance(value, str) or not value.strip():
        raise ProjectContractError(f"{{name}} must be a non-empty string")


{body.rstrip()}
'''


def package_init(spec: CourseProject) -> str:
    """Render public contract exports."""
    return f'''"""{spec.course_code} research scaffold; no benchmark claim."""

from .contracts import {spec.input_type}, {spec.output_type}

__all__ = ["{spec.input_type}", "{spec.output_type}"]
__version__ = "0.1.0"
'''


def errors_module() -> str:
    """Render shared fail-closed exceptions."""
    return '''"""Project-specific failures used by scaffold contracts."""


class ProjectError(RuntimeError):
    """Base error for a failed project contract or gate."""


class ProjectContractError(ProjectError, ValueError):
    """Raised when input, output, schema, or configuration is invalid."""


class EvidenceGateError(ProjectError):
    """Raised when data/model/service evidence is incomplete."""


class AdapterDisabledError(ProjectError):
    """Raised because scaffold adapters cannot perform network operations."""
'''


def pipeline_module(spec: CourseProject) -> str:
    """Render deterministic stage planning, not pipeline execution."""
    stages = "\n".join(f'    "{stage}",' for stage in spec.stages)
    return f'''"""Deterministic scaffold stage plan; no data or model execution."""

from __future__ import annotations

from dataclasses import dataclass


STAGES: tuple[str, ...] = (
{stages}
)


@dataclass(frozen=True, slots=True)
class PipelinePlan:
    """Describe stages and evidence level without executing them."""

    project: str
    stages: tuple[str, ...]
    evidence_level: str = "scaffold"
    network_enabled: bool = False
    training_enabled: bool = False


def build_pipeline_plan() -> PipelinePlan:
    """Return the approved scaffold plan."""
    return PipelinePlan(project="{spec.repo_name}", stages=STAGES)
'''


def data_manifest_module() -> str:
    """Render governed data manifest contracts."""
    return '''"""Governed external data manifest and ingest gate."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import EvidenceGateError, ProjectContractError


@dataclass(frozen=True, slots=True)
class DataManifest:
    """Describe one versioned data asset with rights and split role."""

    asset_id: str
    official_url: str
    accessed_on: str
    license_or_terms: str
    revision: str
    sha256: str
    reviewer: str
    data_role: str
    status: str = "hold"

    def __post_init__(self) -> None:
        values = (self.asset_id, self.official_url, self.accessed_on, self.license_or_terms,
                  self.revision, self.sha256, self.reviewer, self.data_role)
        if any(not value.strip() for value in values):
            raise ProjectContractError("manifest fields must not be blank")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ProjectContractError("sha256 must be 64 lowercase hexadecimal characters")

    def assert_ingestible(self) -> None:
        """Allow ingest only after an explicit approval state."""
        if self.status != "approved":
            raise EvidenceGateError(f"asset {self.asset_id!r} is not approved")
'''


def preprocessing_module(spec: CourseProject) -> str:
    """Render preprocessing interface."""
    return f'''"""Preprocessing boundary for {spec.input_type}."""

from __future__ import annotations

from typing import Protocol

from ..contracts import {spec.input_type}


class Preprocessor(Protocol):
    """Transform a validated request without fitting on held-out data."""

    @property
    def version(self) -> str:
        """Return an immutable preprocessing version."""

    def transform(self, request: {spec.input_type}) -> {spec.input_type}:
        """Return a normalized request or raise a contract error."""
'''


def retrieval_module(spec: CourseProject) -> str:
    """Render retrieval interface."""
    return f'''"""Retrieval interface independent of a concrete backend."""

from __future__ import annotations

from typing import Protocol

from ..contracts import {spec.input_type}, {spec.output_type}


class Retriever(Protocol):
    """Return ranked, provenance-bearing results."""

    @property
    def version(self) -> str:
        """Return an immutable retriever version."""

    def search(self, request: {spec.input_type}) -> tuple[{spec.output_type}, ...]:
        """Search an already approved local corpus."""
'''


def evaluation_module() -> str:
    """Render evidence-aware evaluation contracts."""
    return '''"""Evaluation result contracts with uncertainty and invalid rates."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ProjectContractError


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Store one metric with sample size and optional confidence interval."""

    name: str
    value: float
    sample_size: int
    ci_low: float | None = None
    ci_high: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or self.sample_size < 1:
            raise ProjectContractError("metric name and positive sample_size are required")
        if (self.ci_low is None) != (self.ci_high is None):
            raise ProjectContractError("confidence interval endpoints must appear together")
        if self.ci_low is not None and not self.ci_low <= self.value <= self.ci_high:
            raise ProjectContractError("metric value must lie inside its confidence interval")


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Attach provenance and evidence class to evaluated metrics."""

    run_id: str
    metrics: tuple[MetricSummary, ...]
    invalid_output_rate: float
    seeds: tuple[int, ...]
    config_hash: str
    data_manifest_hash: str
    code_revision: str
    evidence_level: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.invalid_output_rate <= 1.0:
            raise ProjectContractError("invalid_output_rate must be in [0, 1]")
        if not self.metrics:
            raise ProjectContractError("at least one metric is required")
'''


def application_module(spec: CourseProject) -> str:
    """Render a narrow application service."""
    return f'''"""Application service that depends on injected local retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import {spec.input_type}, {spec.output_type}
from ..retrieval.contracts import Retriever


@dataclass(slots=True)
class ResearchApplication:
    """Expose search without owning data downloads, training, or UI state."""

    retriever: Retriever

    def search(self, request: {spec.input_type}) -> tuple[{spec.output_type}, ...]:
        """Delegate to an injected retriever and preserve its failure contract."""
        return self.retriever.search(request)
'''


def ui_module(spec: CourseProject) -> str:
    """Render a pure-Python, fail-closed checkpoint lineage view model."""
    return f'''"""Read the latest governed checkpoint for a read-only UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import re

REPO_NAME = "{spec.repo_name}"
CHECKPOINT_PATTERN = re.compile(
    rf"^{{re.escape(REPO_NAME)}}_Checkpoints_(\\d{{{{2}}}}-\\d{{{{2}}}}-\\d{{{{2}}}})_V([1-9]\\d*)\\.md$"
)
EVIDENCE_PATTERN = re.compile(
    r"^\\s*-\\s*evidence class:\\s*(.+?)\\s*$", re.IGNORECASE | re.MULTILINE
)
VERSION_PATTERN = re.compile(
    r"^\\s*-\\s*phiên bản:\\s*(V[1-9]\\d*)\\s*$", re.IGNORECASE | re.MULTILINE
)
PROJECT_PATTERN = re.compile(
    r"^\\s*-\\s*project:\\s*(.+?)\\s*$", re.IGNORECASE | re.MULTILINE
)
WARNING_SECTION_PATTERN = re.compile(
    r"^## Cảnh báo\\s*$\\n(?P<body>.*?)(?=^## |\\Z)", re.IGNORECASE | re.MULTILINE | re.DOTALL
)


@dataclass(frozen=True, slots=True)
class GateView:
    """Represent one implementation gate."""

    name: str
    status: str
    evidence_level: str
    next_condition: str


@dataclass(frozen=True, slots=True)
class CheckpointLineage:
    """Expose immutable checkpoint identity without returning raw log content."""

    status: str
    relative_path: str
    checkpoint_date: str
    version: str
    evidence_class: str
    sha256: str
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DashboardView:
    """Bundle static gates with dynamic, read-only lineage evidence."""

    gates: tuple[GateView, ...]
    lineage: CheckpointLineage


def _unavailable(message: str, *, status: str = "unavailable") -> CheckpointLineage:
    """Return a fail-closed lineage record with an actionable warning."""
    return CheckpointLineage(status, "", "", "", "unavailable", "", (message,))


def _candidate_key(path: Path) -> tuple[datetime, int]:
    """Return the chronological key already validated by the filename grammar."""
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"invalid checkpoint filename: {{path.name}}")
    date_text, version_text = match.groups()
    return datetime.strptime(date_text, "%d-%m-%y"), int(version_text)


def load_latest_checkpoint(repo_root: Path) -> CheckpointLineage:
    """Read the latest valid checkpoint or return a fail-closed state."""
    root = repo_root.resolve()
    general = root / "Documents" / "General"
    if not general.is_dir():
        return _unavailable("Không tìm thấy Documents/General.")
    candidates: list[Path] = []
    malformed: list[str] = []
    prefix = f"{{REPO_NAME}}_Checkpoints_"
    try:
        entries = tuple(general.iterdir())
    except OSError as exc:
        return _unavailable(f"Không đọc được Documents/General: {{exc}}")
    for path in entries:
        if not path.is_file() or not path.name.startswith(prefix):
            continue
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is None:
            malformed.append(path.name)
            continue
        try:
            datetime.strptime(match.group(1), "%d-%m-%y")
        except ValueError:
            malformed.append(path.name)
            continue
        candidates.append(path)
    if malformed:
        names = ", ".join(sorted(malformed))
        return _unavailable(f"Checkpoint sai grammar; từ chối chọn lineage: {{names}}", status="invalid")
    if not candidates:
        return _unavailable("Không có checkpoint đúng grammar.")
    latest = max(candidates, key=_candidate_key)
    match = CHECKPOINT_PATTERN.fullmatch(latest.name)
    if match is None:
        return _unavailable("Checkpoint mới nhất không còn khớp grammar.", status="invalid")
    date_text, version_text = match.groups()
    try:
        payload = latest.read_bytes()
        text = payload.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        return _unavailable(f"Không đọc được checkpoint mới nhất: {{exc}}", status="invalid")
    digest = sha256(payload).hexdigest()
    relative = latest.relative_to(root).as_posix()
    expected_version = f"V{{version_text}}"
    version_match = VERSION_PATTERN.search(text)
    if version_match is None or version_match.group(1).upper() != expected_version:
        actual = version_match.group(1) if version_match is not None else "missing"
        return CheckpointLineage(
            "invalid", relative, date_text, expected_version, "unavailable", digest,
            (f"Phiên bản trong checkpoint không khớp filename: {{actual}} != {{expected_version}}.",),
        )
    project_match = PROJECT_PATTERN.search(text)
    actual_project = (
        project_match.group(1).replace(chr(96), "").strip()
        if project_match is not None
        else "missing"
    )
    if actual_project != REPO_NAME:
        return CheckpointLineage(
            "invalid", relative, date_text, expected_version, "unavailable", digest,
            (f"Project trong checkpoint không khớp repository: {{actual_project}} != {{REPO_NAME}}.",),
        )
    evidence_match = EVIDENCE_PATTERN.search(text)
    if evidence_match is None:
        return CheckpointLineage(
            "invalid", relative, date_text, expected_version, "unavailable", digest,
            ("Checkpoint thiếu trường Evidence class; không suy diễn trạng thái.",),
        )
    evidence = evidence_match.group(1).replace("`", "").strip()
    warnings: list[str] = []
    warning_match = WARNING_SECTION_PATTERN.search(text)
    if warning_match is not None:
        warnings.extend(
            line.strip().removeprefix("-").strip()
            for line in warning_match.group("body").splitlines()
            if line.strip().startswith("-") and line.strip().removeprefix("-").strip()
        )
    folded = text.casefold()
    if "không phải benchmark" in folded or "chưa có benchmark" in folded:
        warnings.append("Checkpoint không phải benchmark; không suy diễn chất lượng mô hình.")
    if "not-run" in folded:
        warnings.append("Checkpoint còn hạng mục not-run; cần đọc log trước quyết định tiếp theo.")
    return CheckpointLineage(
        "available", relative, date_text, expected_version, evidence, digest, tuple(dict.fromkeys(warnings))
    )


def load_dashboard(repo_root: Path | None = None) -> DashboardView:
    """Load dynamic lineage and append it to stable planning gates."""
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[3]
    lineage = load_latest_checkpoint(root)
    gates = (
        GateView("Problem", "planned", "planning", "Xác nhận người dùng/evaluator"),
        GateView("Core", "scaffold", "scaffold", "Unit + validator"),
        GateView("Data/model", "hold", "candidate", "Manifest + license + revision/checksum"),
        GateView("Baseline", "not-started", "none", "{spec.baselines[0]}"),
        GateView("Ứng dụng", "scaffold", "scaffold", "Artifact read-only"),
        GateView("Lineage", lineage.status, lineage.evidence_class, lineage.relative_path or "Sửa checkpoint"),
    )
    return DashboardView(gates, lineage)


def dashboard_rows(view: DashboardView) -> list[dict[str, str]]:
    """Return table rows without filesystem writes or service calls."""
    return [asdict(gate) for gate in view.gates]


def lineage_rows(lineage: CheckpointLineage) -> list[dict[str, str]]:
    """Return version, hash and evidence fields for Streamlit rendering."""
    return [
        {{"Trường": "Checkpoint", "Giá trị": lineage.relative_path or "unavailable"}},
        {{"Trường": "Ngày", "Giá trị": lineage.checkpoint_date or "unavailable"}},
        {{"Trường": "Version", "Giá trị": lineage.version or "unavailable"}},
        {{"Trường": "Evidence class", "Giá trị": lineage.evidence_class}},
        {{"Trường": "SHA-256", "Giá trị": lineage.sha256 or "unavailable"}},
    ]
'''


def service_base_module() -> str:
    """Render shared external-service planning contracts."""
    return '''"""Fail-closed contracts for all external resource providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..errors import ProjectContractError


@dataclass(frozen=True, slots=True)
class ExternalResourceSpec:
    """Require provenance, rights, revision and checksum before a call."""

    provider: str
    resource_id: str
    official_url: str
    accessed_on: str
    license_or_terms: str
    revision: str
    sha256: str
    reviewer: str
    data_role: str
    status: str = "approved"

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ProjectContractError(f"{name} must be a non-empty string")
        if self.status != "approved":
            raise ProjectContractError("external resource status must be approved")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ProjectContractError("sha256 must be 64 lowercase hexadecimal characters")
        if not self.official_url.startswith("https://"):
            raise ProjectContractError("official_url must use HTTPS")


@dataclass(frozen=True, slots=True)
class DownloadPlan:
    """Describe a call plan without executing network or filesystem writes."""

    resource: ExternalResourceSpec
    cache_dir: Path
    adapter_name: str
    local_only: bool = True


class ProviderAdapter(Protocol):
    """Define planning and explicit fetch boundaries."""

    def build_plan(self, resource: ExternalResourceSpec, cache_dir: Path) -> DownloadPlan:
        """Return a validated plan without executing it."""

    def fetch(self, plan: DownloadPlan) -> Path:
        """Fetch only after a separate ADR enables implementation."""
'''


def provider_module(provider: str, class_name: str, rule: str) -> str:
    """Render a provider adapter that validates then refuses fetch."""
    return f'''"""{class_name} scaffold; network execution remains disabled."""

from __future__ import annotations

from pathlib import Path

from ..errors import AdapterDisabledError, ProjectContractError
from .base import DownloadPlan, ExternalResourceSpec


class {class_name}:
    """Plan governed access to {provider}; never fetch in scaffold."""

    name = "{provider}"

    def build_plan(self, resource: ExternalResourceSpec, cache_dir: Path) -> DownloadPlan:
        """Validate provider-specific immutability requirements."""
        if resource.provider.casefold() != self.name:
            raise ProjectContractError("resource provider must be {provider}")
        {rule}
        return DownloadPlan(resource=resource, cache_dir=cache_dir, adapter_name=self.name)

    def fetch(self, plan: DownloadPlan) -> Path:
        """Fail closed until network behavior has ADR, mocks and rollback."""
        raise AdapterDisabledError("{class_name}.fetch is disabled in scaffold")
'''


def artifact_store_module() -> str:
    """Render immutable artifact-store boundary."""
    return '''"""Artifact-store boundary without a configured backend."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..errors import AdapterDisabledError


class ArtifactStore(Protocol):
    """Resolve immutable artifact identifiers."""

    def resolve(self, artifact_id: str) -> Path:
        """Return a local immutable artifact path."""


class DisabledArtifactStore:
    """Reject access until storage and integrity contracts are configured."""

    def resolve(self, artifact_id: str) -> Path:
        """Fail closed and name the missing artifact."""
        raise AdapterDisabledError(f"artifact store is disabled for {artifact_id!r}")
'''


def inference_task(spec: CourseProject) -> str:
    """Return the narrow external-inference task allowed by each project."""
    return {
        "CS406": "image-feature-extraction",
        "DS111": "artifact-schema-probe",
        "DS201": "feature-extraction",
        "DS314": "feature-extraction",
    }[spec.course_code]


def inference_module(spec: CourseProject) -> str:
    """Render a schema-bound inference plan whose execution stays disabled."""
    return f'''"""Validate external inference plans while execution stays disabled."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import AdapterDisabledError, ProjectContractError

EXPECTED_TASK = "{inference_task(spec)}"
MUTABLE_REVISIONS = {{"", "main", "master", "latest", "default", "auto", "live"}}
PRIVACY_CLASSES = {{"public", "internal", "restricted"}}


def _require_immutable(value: str, name: str) -> None:
    """Reject empty or floating revisions."""
    if not isinstance(value, str) or value.strip().casefold() in MUTABLE_REVISIONS:
        raise ProjectContractError(f"{{name}} must be an immutable revision")
    if len(value.strip()) < 7:
        raise ProjectContractError(f"{{name}} is too short to identify an immutable revision")


@dataclass(frozen=True, slots=True)
class InferenceSpec:
    """Describe a pinned, schema-bound external inference request."""

    provider: str
    provider_revision: str
    model_id: str
    model_revision: str
    task: str
    timeout_seconds: float
    privacy_class: str
    request_schema: str
    output_schema: str
    execution_enabled: bool = False

    def __post_init__(self) -> None:
        for name in ("provider", "model_id", "task", "request_schema", "output_schema"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ProjectContractError(f"{{name}} must be a non-empty string")
        _require_immutable(self.provider_revision, "provider_revision")
        _require_immutable(self.model_revision, "model_revision")
        if self.task != EXPECTED_TASK:
            raise ProjectContractError(f"task must be {{EXPECTED_TASK!r}}")
        if not 0 < self.timeout_seconds <= 120:
            raise ProjectContractError("timeout_seconds must be in (0, 120]")
        if self.privacy_class not in PRIVACY_CLASSES:
            raise ProjectContractError("unsupported privacy_class")
        if self.execution_enabled:
            raise ProjectContractError("execution must remain disabled in scaffold")


@dataclass(frozen=True, slots=True)
class InferencePlan:
    """Return a validated plan without network execution."""

    spec: InferenceSpec
    adapter_name: str = "governed-inference"


class InferenceAdapter:
    """Build public-data plans and fail closed on execution."""

    def build_plan(self, spec: InferenceSpec) -> InferencePlan:
        """Validate privacy before creating an offline plan."""
        if spec.privacy_class != "public":
            raise ProjectContractError("external inference only accepts public data")
        return InferencePlan(spec=spec)

    def execute(self, plan: InferencePlan) -> object:
        """Reject calls until an ADR enables a tested provider implementation."""
        raise AdapterDisabledError("InferenceAdapter.execute is disabled in scaffold")
'''


def inference_test(spec: CourseProject) -> str:
    """Render minimum offline validation for the inference contract."""
    return f'''"""Validate immutable, private-safe and fail-closed inference plans."""

import unittest

from {spec.package_name}.ServiceAdapters.inference import InferenceAdapter, InferenceSpec
from {spec.package_name}.errors import AdapterDisabledError, ProjectContractError


def valid_spec(**changes: object) -> InferenceSpec:
    """Build one public synthetic inference specification."""
    values: dict[str, object] = {{
        "provider": "synthetic-provider",
        "provider_revision": "provider-2026-08-25",
        "model_id": "fixture/model",
        "model_revision": "a" * 40,
        "task": "{inference_task(spec)}",
        "timeout_seconds": 30.0,
        "privacy_class": "public",
        "request_schema": "schemas/input.schema.json@1.0",
        "output_schema": "schemas/output.schema.json@1.0",
    }}
    values.update(changes)
    return InferenceSpec(**values)  # type: ignore[arg-type]


class InferenceAdapterTests(unittest.TestCase):
    """Keep network execution unavailable until a later ADR."""

    def test_valid_plan_stays_disabled(self) -> None:
        adapter = InferenceAdapter()
        plan = adapter.build_plan(valid_spec())
        self.assertFalse(plan.spec.execution_enabled)
        with self.assertRaises(AdapterDisabledError):
            adapter.execute(plan)

    def test_mutable_revisions_are_rejected(self) -> None:
        with self.assertRaises(ProjectContractError):
            valid_spec(model_revision="latest")
        with self.assertRaises(ProjectContractError):
            valid_spec(provider_revision="default")

    def test_non_public_data_is_rejected(self) -> None:
        with self.assertRaises(ProjectContractError):
            InferenceAdapter().build_plan(valid_spec(privacy_class="restricted"))

    def test_task_and_timeout_are_bounded(self) -> None:
        with self.assertRaises(ProjectContractError):
            valid_spec(task="wrong-task")
        with self.assertRaises(ProjectContractError):
            valid_spec(timeout_seconds=0)


if __name__ == "__main__":
    unittest.main()
'''


def extra_source_files(spec: CourseProject) -> dict[str, str]:
    """Return domain-specific scaffold modules."""
    package = f"src/{spec.package_name}"
    if spec.course_code == "CS406":
        return {
            f"{package}/features/__init__.py": '"""Visual feature contracts."""\n',
            f"{package}/features/contracts.py": '''"""Versioned visual feature boundary."""

from dataclasses import dataclass

from ..errors import ProjectContractError


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Store one finite, versioned visual descriptor."""

    asset_id: str
    values: tuple[float, ...]
    feature_version: str

    def __post_init__(self) -> None:
        if not self.asset_id.strip() or not self.feature_version.strip() or not self.values:
            raise ProjectContractError("asset_id, feature_version and values are required")
        if any(value != value or value in {float("inf"), float("-inf")} for value in self.values):
            raise ProjectContractError("feature values must be finite")
''',
        }
    if spec.course_code == "DS111":
        return {
            f"{package}/analysis/__init__.py": '"""Statistical analysis contracts."""\n',
            f"{package}/analysis/contracts.py": '''"""Paired comparison planning contracts."""

from dataclasses import dataclass

from ..errors import ProjectContractError


@dataclass(frozen=True, slots=True)
class PairedComparisonPlan:
    """Describe a paired run comparison without computing it."""

    baseline_run_id: str
    candidate_run_id: str
    metric: str
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if self.baseline_run_id == self.candidate_run_id:
            raise ProjectContractError("baseline and candidate runs must differ")
        if not 0.0 < self.confidence_level < 1.0:
            raise ProjectContractError("confidence_level must be in (0, 1)")
''',
            f"{package}/quality/__init__.py": '"""Data-quality audit contracts."""\n',
            f"{package}/quality/audit.py": '''"""Quality finding records; no dataframe dependency."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QualityFinding:
    """Describe one reproducible data-quality finding."""

    check_id: str
    severity: str
    row_count: int
    evidence_uri: str
''',
            f"{package}/visualization/__init__.py": '"""Visualization-neutral report models."""\n',
            f"{package}/visualization/view_models.py": '''"""Chart specification without a plotting backend."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChartSpec:
    """Describe a chart and its denominator/provenance."""

    chart_id: str
    title: str
    metric: str
    denominator: int
    source_artifact_id: str
''',
        }
    if spec.course_code == "DS201":
        return {
            f"{package}/normalization/__init__.py": '"""Historical time normalization."""\n',
            f"{package}/normalization/historical_time.py": '''"""Explicit BCE/CE year representation without ambiguous zero."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import ProjectContractError


class Era(str, Enum):
    """Supported historical eras."""

    BCE = "BCE"
    CE = "CE"


@dataclass(frozen=True, slots=True)
class HistoricalYear:
    """Represent a historical year with no civil year zero."""

    era: Era
    year: int

    def __post_init__(self) -> None:
        if self.year < 1:
            raise ProjectContractError("HistoricalYear.year must be positive")

    @property
    def astronomical(self) -> int:
        """Map 1 BCE to 0 and 1 CE to 1 for arithmetic only."""
        return self.year if self.era is Era.CE else 1 - self.year
''',
            f"{package}/representations/__init__.py": '"""Temporal representation contracts."""\n',
            f"{package}/representations/contracts.py": '''"""Label published, assumed and proposed temporal methods."""

from dataclasses import dataclass

from ..errors import ProjectContractError


@dataclass(frozen=True, slots=True)
class RepresentationSpec:
    """Describe one temporal representation and its evidence class."""

    name: str
    method_class: str
    dimension: int
    version: str

    def __post_init__(self) -> None:
        allowed = {"published-equation", "operational-assumption", "proposed-method"}
        if self.method_class not in allowed or self.dimension < 1:
            raise ProjectContractError("invalid representation specification")
''',
            f"{package}/training/__init__.py": '"""Training gates and plans."""\n',
            f"{package}/training/contracts.py": '''"""Training plan without a Torch dependency."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    """Record immutable training inputs and resource budget."""

    model_revision: str
    data_manifest_hash: str
    split_hash: str
    seed: int
    max_vram_mb: int
    enabled: bool = False
''',
            f"{package}/model_management/__init__.py": '"""Model registry contracts."""\n',
            f"{package}/model_management/registry.py": '''"""Immutable model artifact identity."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """Bind a model artifact to code, data and configuration."""

    artifact_id: str
    model_revision: str
    config_hash: str
    data_manifest_hash: str
    code_revision: str
    evidence_level: str
''',
        }
    return {
        f"{package}/corpus/__init__.py": '"""Corpus contracts."""\n',
        f"{package}/corpus/manifest.py": '''"""Authorized chunk identity for a governed corpus."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CorpusChunk:
    """Preserve source, version and access scope for one chunk."""

    document_id: str
    chunk_id: str
    text: str
    source_uri: str
    corpus_version: str
    access_scope: str
''',
        f"{package}/fusion/__init__.py": '"""Rank fusion contracts."""\n',
        f"{package}/fusion/contracts.py": '''"""Versioned fusion plan."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FusionPlan:
    """Describe bounded retrievers and a deterministic fusion method."""

    retriever_versions: tuple[str, ...]
    method: str = "rrf"
    rrf_k: int = 60
''',
        f"{package}/agent_policy/__init__.py": '"""Bounded post-retrieval policy."""\n',
        f"{package}/agent_policy/policy.py": '''"""Bounded policy scaffold; no LLM call."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentPolicy:
    """Limit candidate and tool steps after retrieval has passed its gate."""

    max_steps: int = 3
    max_evidence_chunks: int = 5
    require_citation: bool = True
    allow_generation: bool = False
''',
    }


def streamlit_app(spec: CourseProject) -> str:
    """Render a read-only Streamlit entrypoint."""
    return f'''"""Read-only Streamlit scaffold for {spec.course_code}."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from {spec.package_name}.ui.view_models import (  # noqa: E402
    dashboard_rows,
    lineage_rows,
    load_dashboard,
)


def main() -> None:
    """Render project gates; never download data, train, or predict."""
    st.set_page_config(page_title="{spec.short_title}", layout="wide")
    st.title("{spec.title}")
    st.warning("Scaffold: chưa có dataset/model thật hoặc kết quả benchmark.")
    dashboard = load_dashboard(ROOT)
    st.dataframe(dashboard_rows(dashboard), width="stretch", hide_index=True)
    st.subheader("Lineage checkpoint mới nhất")
    st.dataframe(lineage_rows(dashboard.lineage), width="stretch", hide_index=True)
    if dashboard.lineage.status != "available":
        st.error("Lineage không hợp lệ; UI từ chối suy diễn trạng thái project.")
    for warning in dashboard.lineage.warnings:
        st.warning(warning)
    st.info("UI chỉ đọc checkpoint và artifact metadata; không download, train hoặc inference.")


if __name__ == "__main__":
    main()
'''


def ui_lineage_test(spec: CourseProject) -> str:
    """Render offline tests for latest-checkpoint discovery and fail-closed parsing."""
    return f'''"""Validate dynamic checkpoint lineage without network or Streamlit."""

from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

from {spec.package_name}.ui.view_models import load_dashboard, load_latest_checkpoint


def checkpoint_text(version: str, *, evidence: bool = True) -> str:
    """Return a minimal governed checkpoint fixture."""
    evidence_line = "- Evidence class: `static_check`; không phải benchmark." if evidence else ""
    return f"""# Checkpoint — fixture

## Định danh

- Project: {spec.repo_name}
- Phiên bản: {{version}}
{{evidence_line}}

## Cảnh báo

- Chưa có benchmark.
"""


class CheckpointLineageTests(unittest.TestCase):
    """Keep checkpoint selection deterministic, immutable and fail-closed."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.general = self.root / "Documents" / "General"
        self.general.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_checkpoint(self, name: str, content: str) -> Path:
        """Write one UTF-8 fixture under the temporary General directory."""
        path = self.general / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_selects_latest_date_then_version_and_hashes_bytes(self) -> None:
        self.write_checkpoint(
            "{spec.repo_name}_Checkpoints_24-08-26_V9.md", checkpoint_text("V9")
        )
        self.write_checkpoint(
            "{spec.repo_name}_Checkpoints_25-08-26_V1.md", checkpoint_text("V1")
        )
        latest = self.write_checkpoint(
            "{spec.repo_name}_Checkpoints_25-08-26_V2.md", checkpoint_text("V2")
        )
        before = sorted(path.name for path in self.general.iterdir())
        lineage = load_latest_checkpoint(self.root)
        self.assertEqual(lineage.status, "available")
        self.assertEqual(lineage.version, "V2")
        self.assertEqual(lineage.sha256, sha256(latest.read_bytes()).hexdigest())
        self.assertIn("static_check", lineage.evidence_class)
        self.assertTrue(lineage.warnings)
        dashboard = load_dashboard(self.root)
        self.assertEqual(dashboard.lineage, lineage)
        self.assertEqual(before, sorted(path.name for path in self.general.iterdir()))

    def test_malformed_checkpoint_like_name_fails_closed(self) -> None:
        self.write_checkpoint(
            "{spec.repo_name}_Checkpoints_25-08-26_V1.md", checkpoint_text("V1")
        )
        self.write_checkpoint(
            "{spec.repo_name}_Checkpoints_2026-08-25_V2.md", checkpoint_text("V2")
        )
        lineage = load_latest_checkpoint(self.root)
        self.assertEqual(lineage.status, "invalid")
        self.assertEqual(lineage.sha256, "")

    def test_body_version_mismatch_fails_closed(self) -> None:
        self.write_checkpoint(
            "{spec.repo_name}_Checkpoints_25-08-26_V2.md", checkpoint_text("V1")
        )
        lineage = load_latest_checkpoint(self.root)
        self.assertEqual(lineage.status, "invalid")
        self.assertIn("không khớp", lineage.warnings[0])

    def test_missing_evidence_fails_closed_but_keeps_hash(self) -> None:
        path = self.write_checkpoint(
            "{spec.repo_name}_Checkpoints_25-08-26_V1.md",
            checkpoint_text("V1", evidence=False),
        )
        lineage = load_latest_checkpoint(self.root)
        self.assertEqual(lineage.status, "invalid")
        self.assertEqual(lineage.sha256, sha256(path.read_bytes()).hexdigest())

    def test_project_mismatch_fails_closed(self) -> None:
        text = checkpoint_text("V1").replace("{spec.repo_name}", "wrong-project")
        self.write_checkpoint(
            "{spec.repo_name}_Checkpoints_25-08-26_V1.md", text
        )
        lineage = load_latest_checkpoint(self.root)
        self.assertEqual(lineage.status, "invalid")
        self.assertIn("Project", lineage.warnings[0])

    def test_missing_general_or_checkpoint_is_unavailable(self) -> None:
        self.assertEqual(load_latest_checkpoint(self.root).status, "unavailable")
        self.assertEqual(load_latest_checkpoint(self.root / "missing").status, "unavailable")


if __name__ == "__main__":
    unittest.main()
'''


def streamlit_app_test() -> str:
    """Render an offline AppTest that guards imports and read-only widgets."""
    return '''"""Smoke-test the read-only Streamlit app without network access."""

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "streamlit_app" / "app.py"


class StreamlitAppTests(unittest.TestCase):
    """Require a renderable app with lineage tables and no action buttons."""

    def test_app_renders_offline_without_import_errors(self) -> None:
        app = AppTest.from_file(str(APP), default_timeout=10).run()
        self.assertEqual(list(app.exception), [])
        self.assertGreaterEqual(len(app.dataframe), 2)
        self.assertTrue(any("UI chỉ đọc" in item.value for item in app.info))
        self.assertEqual(len(app.button), 0)


if __name__ == "__main__":
    unittest.main()
'''


def contract_test(spec: CourseProject) -> str:
    """Render project-specific contract examples."""
    if spec.course_code == "CS406":
        construction = 'ImageQuery("q1", "fixture/query.jpg", "corpus-v1", 5)'
        assertion = "self.assertEqual(request.top_k, 5)"
        invalid = 'ImageQuery("q1", "fixture/query.jpg", "corpus-v1", 0)'
    elif spec.course_code == "DS111":
        construction = 'RetrievalRunInput("run-1", "fixture/run.tsv", "fixture/qrels.tsv", "1.0")'
        assertion = 'self.assertEqual(request.schema_version, "1.0")'
        invalid = 'AnalysisReport("run-1", "unknown", (("nDCG@10", 0.5),))'
    elif spec.course_code == "DS201":
        construction = 'TemporalQuery("q1", "Sự kiện xảy ra khi nào?", HistoricalYear(Era.BCE, 1), HistoricalYear(Era.CE, 1), "corpus-v1", 5)'
        assertion = "self.assertEqual(request.start.astronomical, 0)"
        invalid = "HistoricalYear(Era.CE, 0)"
    else:
        construction = 'RetrievalQuery("q1", "thủ tục công khai", "corpus-v1", "public", 5)'
        assertion = 'self.assertEqual(request.access_scope, "public")'
        invalid = 'RetrievalQuery("q1", "text", "corpus-v1", "public", 0)'
    temporal_import = (
        f"from {spec.package_name}.normalization.historical_time import Era, HistoricalYear\n"
        if spec.course_code == "DS201"
        else ""
    )
    return f'''"""Validate project-specific request/output contracts."""

import unittest

from {spec.package_name}.contracts import {spec.input_type}, {spec.output_type}
from {spec.package_name}.errors import ProjectContractError
{temporal_import}

class ContractTests(unittest.TestCase):
    """Check valid construction and fail-closed invalid inputs."""

    def test_valid_input(self) -> None:
        request = {construction}
        {assertion}

    def test_invalid_input_or_output(self) -> None:
        with self.assertRaises(ProjectContractError):
            {invalid}


if __name__ == "__main__":
    unittest.main()
'''


def scaffold_test(spec: CourseProject) -> str:
    """Render pipeline and service-adapter tests."""
    return f'''"""Validate stage order and disabled network adapters."""

from pathlib import Path
import unittest

from {spec.package_name}.ServiceAdapters.base import ExternalResourceSpec
from {spec.package_name}.ServiceAdapters.huggingface import HuggingFaceAdapter
from {spec.package_name}.errors import AdapterDisabledError, ProjectContractError
from {spec.package_name}.pipeline import STAGES, build_pipeline_plan


def approved_resource(revision: str = "a" * 40) -> ExternalResourceSpec:
    """Return a synthetic manifest used only for contract tests."""
    return ExternalResourceSpec(
        provider="huggingface",
        resource_id="org/repo",
        official_url="https://example.org/resource",
        accessed_on="2026-08-24",
        license_or_terms="fixture-only",
        revision=revision,
        sha256="0" * 64,
        reviewer="test-reviewer",
        data_role="reference-only",
    )


class ScaffoldTests(unittest.TestCase):
    """Ensure scaffold cannot be mistaken for an executing system."""

    def test_stage_plan_is_offline_scaffold(self) -> None:
        plan = build_pipeline_plan()
        self.assertEqual(plan.stages, STAGES)
        self.assertFalse(plan.network_enabled)
        self.assertFalse(plan.training_enabled)
        self.assertEqual(plan.evidence_level, "scaffold")

    def test_huggingface_rejects_mutable_revision_and_fetch(self) -> None:
        adapter = HuggingFaceAdapter()
        with self.assertRaises(ProjectContractError):
            adapter.build_plan(approved_resource("main"), Path("cache"))
        plan = adapter.build_plan(approved_resource(), Path("cache"))
        with self.assertRaises(AdapterDisabledError):
            adapter.fetch(plan)


if __name__ == "__main__":
    unittest.main()
'''


def validator_script(spec: CourseProject) -> str:
    """Render the zero-dependency v3 structural validator."""
    return f'''"""Validate v3 course repository structure and offline contracts."""

from __future__ import annotations

import json
import re
import tomllib
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_NAME = "{spec.repo_name}"
FORBIDDEN_SCOPE_TERM = "multi" + "modal"
DEFERRED_SCOPE_FILE = (
    ROOT / "Documents" / "Decision-Making" / ("04-DEFERRED-" + FORBIDDEN_SCOPE_TERM.upper() + ".md")
)
IGNORED_PARTS = {{".git", ".migration-backups", ".pytest_cache", ".venv", "__pycache__"}}
LOG_PATTERN = re.compile(
    rf"^{{re.escape(REPO_NAME)}}_(Checkpoints|Error)_(\\d{{{{2}}}}-\\d{{{{2}}}}-\\d{{{{2}}}})_V([1-9]\\d*)\\.md$"
)
BODY_VERSION_PATTERN = re.compile(
    r"^\\s*-\\s*phiên bản:\\s*(V[1-9]\\d*)\\s*$", re.IGNORECASE | re.MULTILINE
)
BODY_PROJECT_PATTERN = re.compile(
    r"^\\s*-\\s*project:\\s*(.+?)\\s*$", re.IGNORECASE | re.MULTILINE
)
EVIDENCE_CLASS_PATTERN = re.compile(
    r"^\\s*-\\s*evidence class:\\s*(.+?)\\s*$", re.IGNORECASE | re.MULTILINE
)

REQUIRED = (
    "README.md", "LICENSE", "AGENTS.md", "requirements.txt", "pyproject.toml",
    ".env.example", ".codex/config.toml", "Workflows/flowinfo.md",
    "configs/scaffold.json", "registries/datasets.json", "registries/models.json",
    "schemas/input.schema.json", "schemas/output.schema.json",
    "Documents/README.md", "Documents/Decision-Making/00-PROBLEM-CONTRACT.md",
    "Documents/Decision-Making/01-ARCHITECTURE-AND-GATES.md",
    "Documents/Decision-Making/02-DATA-MODEL-EVALUATION.md",
    "Documents/Decision-Making/ERROR-EXPERIENCE.md",
    "Documents/Decision-Making/03-TOOLS-AND-SERVICES.md",
    "Documents/Decision-Making/04-DEFERRED-" + FORBIDDEN_SCOPE_TERM.upper() + ".md",
    "Documents/Decision-Making/05-PROJECT-CONTEXT.md",
    "Documents/Decision-Making/06-FEASIBILITY-DATA-4GB.md",
    "Documents/References/RESEARCH-AND-DATASETS.md",
    "Documents/References/VIETNAM-CURRICULUM-ALIGNMENT.md",
    "Documents/References/EXTERNAL-SERVICES-GUIDELINE.md",
    "Documents/References/PROJECT-KNOWLEDGE.md",
    "Documents/References/PROJECT-PLAN.html",
    "Documents/References/COMMON-FAILURES-AND-PREVENTION.md",
    "Documents/General/RULE.md",
    "src/{spec.package_name}/contracts.py",
    "src/{spec.package_name}/ServiceAdapters/base.py",
    "src/{spec.package_name}/ServiceAdapters/huggingface.py",
    "src/{spec.package_name}/ServiceAdapters/kaggle.py",
    "src/{spec.package_name}/ServiceAdapters/inference.py",
    "src/{spec.package_name}/ui/view_models.py",
    "streamlit_app/app.py", "tests/test_contracts.py", "tests/test_scaffold.py",
    "tests/test_inference_adapter.py", "tests/test_ui_lineage.py", "tests/test_streamlit_app.py",
    ".agents/skills/course-project-delivery/SKILL.md",
    ".agents/skills/course-project-delivery/agents/openai.yaml",
)
CHECKPOINT_HEADINGS = (
    "# Checkpoint", "## Định danh", "## Cấu hình hệ thống",
    "## Lệnh và cấu hình chạy", "## Kết quả", "## Artifact và lineage", "## Việc tiếp theo",
)
ERROR_HEADINGS = (
    "# Error", "## Định danh", "## Triệu chứng", "## Cấu hình và đầu vào",
    "## Nguyên nhân", "## Khắc phục hoặc rollback", "## Regression và trạng thái",
)


def possible_mojibake(text: str) -> bool:
    """Detect common UTF-8 decoded as a legacy Windows encoding."""
    markers = tuple(
        "".join(chr(codepoint) for codepoint in codepoints)
        for codepoints in (
            (0x00C3, 0x0192),
            (0x00C3, 0x00A1),
            (0x00C4, 0x2018),
            (0x00E1, 0x00BB),
            (0x00E2, 0x20AC, 0x201C),
            (0x00E2, 0x2020),
        )
    )
    return any(marker in text for marker in markers)


def iter_project_files() -> list[Path]:
    """Return files outside caches and Git internals."""
    return [
        path for path in ROOT.rglob("*")
        if path.is_file() and not any(part in IGNORED_PARTS for part in path.parts)
    ]


def validate_general(failures: list[str]) -> int:
    """Validate General membership, log dates, versions and headings."""
    general = ROOT / "Documents" / "General"
    if not general.is_dir():
        failures.append("missing Documents/General directory")
        return 0
    groups: dict[tuple[str, str], list[int]] = {{}}
    checkpoint_entries: list[tuple[datetime, int, Path, str]] = []
    checkpoints = 0
    for path in sorted(general.iterdir()):
        if path.is_file() and path.name == "RULE.md":
            continue
        if not path.is_file():
            failures.append(f"unexpected General entry: {{path.name}}")
            continue
        match = LOG_PATTERN.fullmatch(path.name)
        if match is None:
            failures.append(f"unexpected General file: {{path.name}}")
            continue
        kind, date_text, version_text = match.groups()
        try:
            parsed_date = datetime.strptime(date_text, "%d-%m-%y")
        except ValueError:
            failures.append(f"invalid log date: {{path.name}}")
            continue
        groups.setdefault((kind, date_text), []).append(int(version_text))
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        headings = CHECKPOINT_HEADINGS if kind == "Checkpoints" else ERROR_HEADINGS
        for heading in headings:
            if heading not in text:
                failures.append(f"{{path.name}} missing heading: {{heading}}")
        if kind == "Checkpoints":
            checkpoint_entries.append((parsed_date, int(version_text), path, text))
            checkpoints += 1
    for key, versions in groups.items():
        ordered = sorted(versions)
        if ordered != list(range(1, max(ordered) + 1)):
            failures.append(f"non-contiguous log versions for {{key}}: {{ordered}}")
    if checkpoints < 1:
        failures.append("at least one checkpoint log is required")
    else:
        _, version, path, text = max(checkpoint_entries, key=lambda item: (item[0], item[1]))
        body_version = BODY_VERSION_PATTERN.search(text)
        expected = f"V{{version}}"
        if body_version is None or body_version.group(1).upper() != expected:
            failures.append(f"latest checkpoint body version does not match {{path.name}}")
        if EVIDENCE_CLASS_PATTERN.search(text) is None:
            failures.append(f"latest checkpoint missing Evidence class: {{path.name}}")
        body_project = BODY_PROJECT_PATTERN.search(text)
        actual_project = (
            body_project.group(1).replace(chr(96), "").strip()
            if body_project is not None
            else "missing"
        )
        if actual_project != REPO_NAME:
            failures.append(f"latest checkpoint Project does not match repository: {{path.name}}")
        if "## Cảnh báo" not in text:
            failures.append(f"latest checkpoint missing warning section: {{path.name}}")
    return checkpoints


def validate_documents(failures: list[str]) -> None:
    """Require decision-critical v3 markers and the new UI lineage path."""
    markers = {{
        "Documents/General/RULE.md": ("Quy tắc tên file", "Error log", "Không ghi đè log"),
        "Documents/Decision-Making/05-PROJECT-CONTEXT.md": (
            "Pitch 30 giây", "Người dùng và quyết định",
        ),
        "Documents/Decision-Making/06-FEASIBILITY-DATA-4GB.md": (
            "Data dossier", "RTX 2050 4 GB",
        ),
        "Documents/References/RESEARCH-AND-DATASETS.md": ("Dataset", "Stop rule"),
        "Documents/References/EXTERNAL-SERVICES-GUIDELINE.md": (
            "Hugging Face Inference", "privacy", "fail-closed",
        ),
        "Documents/References/PROJECT-KNOWLEDGE.md": (
            "Data-first", "Lỗi chí mạng", "ServiceAdapters",
        ),
        "Documents/References/COMMON-FAILURES-AND-PREVENTION.md": (
            "knowledge base", "Documents/General/RULE.md",
        ),
        "streamlit_app/app.py": ("load_dashboard", "lineage_rows", 'width="stretch"'),
        "src/{spec.package_name}/ui/view_models.py": (
            "load_latest_checkpoint", "sha256", "evidence_class", "warnings",
        ),
    }}
    for relative, required_markers in markers.items():
        path = ROOT / relative
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            failures.append(f"cannot read {{relative}} as UTF-8: {{exc}}")
            continue
        for marker in required_markers:
            if marker not in text:
                failures.append(f"{{relative}} missing marker: {{marker}}")
        if relative == "streamlit_app/app.py" and re.search(
            r"Checkpoints_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]_V[1-9][0-9]*", text
        ):
            failures.append("Streamlit must not hard-code a checkpoint version")


def validate_registries(failures: list[str]) -> None:
    """Block approved assets whose provenance fields are incomplete."""
    for name in ("datasets.json", "models.json"):
        path = ROOT / "registries" / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        for entry in payload.get("entries", []):
            if entry.get("status") != "approved":
                continue
            fields = ("official_url", "accessed_on", "license_or_terms", "revision", "sha256", "reviewer")
            for field in fields:
                if not entry.get(field):
                    failures.append(f"approved {{name}} entry missing {{field}}")


def main() -> int:
    """Return non-zero when any repository acceptance condition fails."""
    failures: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            failures.append(f"missing required file: {{relative}}")

    documents = ROOT / "Documents"
    if documents.is_dir():
        root_files = sorted(path.name for path in documents.iterdir() if path.is_file())
        if root_files != ["README.md"]:
            failures.append(f"Documents root must only contain README.md, found {{root_files}}")

    checkpoints = validate_general(failures)
    validate_documents(failures)
    files = iter_project_files()
    for path in files:
        if path.suffix.casefold() == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                failures.append(f"invalid JSON {{path.relative_to(ROOT)}}: {{exc}}")
        elif path.suffix.casefold() == ".toml":
            try:
                tomllib.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
                failures.append(f"invalid TOML {{path.relative_to(ROOT)}}: {{exc}}")

    text_suffixes = {{".py", ".md", ".html", ".toml", ".json", ".yaml", ".yml", ".txt"}}
    for path in files:
        if path.suffix.casefold() not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError as exc:
            failures.append(f"non-UTF-8 text {{path.relative_to(ROOT)}}: {{exc}}")
            continue
        if possible_mojibake(text):
            failures.append(f"possible mojibake in {{path.relative_to(ROOT)}}")
        if path.resolve() != DEFERRED_SCOPE_FILE.resolve() and FORBIDDEN_SCOPE_TERM in text.casefold():
            failures.append(f"deferred scope leaked into {{path.relative_to(ROOT)}}")
        if path.suffix == ".py":
            try:
                compile(text, str(path), "exec")
            except SyntaxError as exc:
                failures.append(f"Python syntax error {{path.relative_to(ROOT)}}: {{exc}}")

    html_path = ROOT / "Documents" / "References" / "PROJECT-PLAN.html"
    if html_path.is_file():
        html = html_path.read_text(encoding="utf-8").casefold()
        for marker in ("<!doctype html", "<html", "<meta charset=", "<title>", "</html>"):
            if marker not in html:
                failures.append(f"HTML missing marker: {{marker}}")

    docx_path = ROOT / "Documents" / "References" / "PROJECT-KNOWLEDGE.docx"
    if docx_path.is_file() and (docx_path.stat().st_size < 10_000 or not zipfile.is_zipfile(docx_path)):
        failures.append("PROJECT-KNOWLEDGE.docx is missing a valid DOCX package")

    validate_registries(failures)

    if failures:
        print("VALIDATION FAILED")
        for failure in failures:
            print(f"- {{failure}}")
        return 1
    print(f"VALIDATION PASSED: {{len(REQUIRED)}} required files, {{checkpoints}} checkpoint(s); offline scaffold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def github_workflow() -> str:
    """Render a no-install scaffold validation workflow."""
    return """name: quality

on:
  push:
  pull_request:

jobs:
  scaffold:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Unit tests
        env:
          PYTHONPATH: src
        run: python -m unittest discover -s tests -v
      - name: Project validator
        run: python scripts/validate_project.py
"""


def common_package_files(spec: CourseProject) -> dict[str, str]:
    """Return all source, test and UI scaffold files."""
    package = f"src/{spec.package_name}"
    files = {
        f"{package}/__init__.py": package_init(spec),
        f"{package}/errors.py": errors_module(),
        f"{package}/contracts.py": contracts_module(spec),
        f"{package}/pipeline.py": pipeline_module(spec),
        f"{package}/data/__init__.py": '"""Data lifecycle contracts."""\n',
        f"{package}/data/manifest.py": data_manifest_module(),
        f"{package}/preprocessing/__init__.py": '"""Preprocessing contracts."""\n',
        f"{package}/preprocessing/contracts.py": preprocessing_module(spec),
        f"{package}/retrieval/__init__.py": '"""Retrieval contracts."""\n',
        f"{package}/retrieval/contracts.py": retrieval_module(spec),
        f"{package}/evaluation/__init__.py": '"""Evaluation contracts."""\n',
        f"{package}/evaluation/metrics.py": evaluation_module(),
        f"{package}/application/__init__.py": '"""Application service boundary."""\n',
        f"{package}/application/service.py": application_module(spec),
        f"{package}/ui/__init__.py": '"""Presentation-neutral UI models."""\n',
        f"{package}/ui/view_models.py": ui_module(spec),
        f"{package}/ServiceAdapters/__init__.py": '"""Governed external-service adapter scaffolds."""\n',
        f"{package}/ServiceAdapters/base.py": service_base_module(),
        f"{package}/ServiceAdapters/huggingface.py": provider_module(
            "huggingface",
            "HuggingFaceAdapter",
            'if resource.revision.casefold() in {"main", "master", "latest"}:\n            raise ProjectContractError("Hugging Face revision must be immutable")',
        ),
        f"{package}/ServiceAdapters/kaggle.py": provider_module(
            "kaggle",
            "KaggleAdapter",
            'if "/versions/" not in resource.resource_id:\n            raise ProjectContractError("Kaggle resource_id must include /versions/<n>")',
        ),
        f"{package}/ServiceAdapters/{spec.domain_adapter_module}.py": provider_module(
            spec.domain_adapter_provider,
            spec.domain_adapter_class,
            'if resource.revision.casefold() in {"latest", "live"}:\n            raise ProjectContractError("domain source revision must be immutable")',
        ),
        f"{package}/ServiceAdapters/artifact_store.py": artifact_store_module(),
        f"{package}/ServiceAdapters/inference.py": inference_module(spec),
        "streamlit_app/app.py": streamlit_app(spec),
        "tests/test_contracts.py": contract_test(spec),
        "tests/test_scaffold.py": scaffold_test(spec),
        "tests/test_inference_adapter.py": inference_test(spec),
        "tests/test_ui_lineage.py": ui_lineage_test(spec),
        "tests/test_streamlit_app.py": streamlit_app_test(),
    }
    files.update(extra_source_files(spec))
    return files


def validate_generated_files(spec: CourseProject, files: dict[str, str]) -> None:
    """Reject a stale or internally inconsistent scaffold before any write."""
    checkpoint = initial_checkpoint_relative(spec)
    general = {path for path in files if path.startswith("Documents/General/")}
    expected_general = {"Documents/General/RULE.md", checkpoint}
    if general != expected_general:
        raise ValueError(f"invalid General file map for {spec.repo_name}: {sorted(general)}")
    required_v3 = {
        "Documents/Decision-Making/03-TOOLS-AND-SERVICES.md",
        "Documents/Decision-Making/04-DEFERRED-MULTIMODAL.md",
        "Documents/Decision-Making/05-PROJECT-CONTEXT.md",
        "Documents/Decision-Making/06-FEASIBILITY-DATA-4GB.md",
        "Documents/References/COMMON-FAILURES-AND-PREVENTION.md",
        f"src/{spec.package_name}/ServiceAdapters/inference.py",
        f"src/{spec.package_name}/ui/view_models.py",
        "tests/test_inference_adapter.py",
        "tests/test_ui_lineage.py",
        "tests/test_streamlit_app.py",
    }
    missing = sorted(required_v3.difference(files))
    if missing:
        raise ValueError(f"missing v3 scaffold files for {spec.repo_name}: {missing}")
    app = files["streamlit_app/app.py"]
    if "load_dashboard" not in app or "Checkpoints_" in app:
        raise ValueError("Streamlit must discover lineage dynamically without a checkpoint literal")
    for relative, source in files.items():
        if relative.endswith(".py"):
            compile(source, relative, "exec")


def repository_files(spec: CourseProject) -> dict[str, str]:
    """Build the complete text-file map for one repository."""
    files = {
        "README.md": render_root_readme(spec),
        "LICENSE": MIT_LICENSE,
        "AGENTS.md": render_agents(spec),
        ".gitignore": GITIGNORE,
        ".gitattributes": GITATTRIBUTES,
        ".env.example": ENV_EXAMPLE,
        ".codex/config.toml": MCP_CONFIG,
        "pyproject.toml": pyproject(spec),
        "requirements.txt": requirements(),
        "Workflows/flowinfo.md": render_workflow(spec),
        "configs/scaffold.json": config_json(spec),
        "registries/datasets.json": registry_json(spec, "datasets"),
        "registries/models.json": registry_json(spec, "models"),
        "registries/experiments.json": json.dumps(
            {"schema_version": "1.0", "entries": []}, ensure_ascii=False, indent=2
        )
        + "\n",
        "schemas/input.schema.json": input_schema(spec),
        "schemas/output.schema.json": output_schema(spec),
        "Documents/README.md": render_documents_readme(),
        "Documents/Decision-Making/00-PROBLEM-CONTRACT.md": render_problem_contract(spec),
        "Documents/Decision-Making/01-ARCHITECTURE-AND-GATES.md": render_architecture(spec),
        "Documents/Decision-Making/02-DATA-MODEL-EVALUATION.md": render_data_model_eval(spec),
        "Documents/Decision-Making/ERROR-EXPERIENCE.md": render_error_experience(spec),
        "Documents/Decision-Making/03-TOOLS-AND-SERVICES.md": render_tools_decision(spec),
        "Documents/Decision-Making/04-DEFERRED-MULTIMODAL.md": render_deferred(),
        "Documents/Decision-Making/05-PROJECT-CONTEXT.md": render_project_context(spec),
        "Documents/Decision-Making/06-FEASIBILITY-DATA-4GB.md": render_feasibility(spec),
        "Documents/References/RESEARCH-AND-DATASETS.md": render_research(spec),
        "Documents/References/VIETNAM-CURRICULUM-ALIGNMENT.md": render_curriculum_alignment(spec),
        "Documents/References/EXTERNAL-SERVICES-GUIDELINE.md": render_services(spec),
        "Documents/References/PROJECT-KNOWLEDGE.md": render_knowledge(spec),
        "Documents/References/PROJECT-PLAN.html": render_html(spec),
        "Documents/References/COMMON-FAILURES-AND-PREVENTION.md": render_common_failures(spec),
        "Documents/General/RULE.md": render_general_rule(spec),
        initial_checkpoint_relative(spec): render_checkpoint(spec),
        ".agents/skills/course-project-delivery/SKILL.md": render_skill(spec),
        ".agents/skills/course-project-delivery/agents/openai.yaml": dedent("""\
            interface:
              display_name: "Course Project Delivery"
              short_description: "Baseline-first research and delivery gates"
              default_prompt: "Use $course-project-delivery to plan or implement one governed project stage."
            """),
        ".agents/skills/course-project-delivery/references/gates.md": render_skill_gates(),
        "scripts/validate_project.py": validator_script(spec),
        ".github/workflows/quality.yml": github_workflow(),
    }
    files.update(common_package_files(spec))
    validate_generated_files(spec, files)
    return files


def write_file(path: Path, content: str) -> None:
    """Write one new UTF-8 file and refuse accidental overwrite."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8", newline="\n")


def build_repository(output_root: Path, spec: CourseProject) -> Path:
    """Create one repository scaffold in a new directory."""
    target = output_root / spec.repo_name
    if target.exists():
        raise FileExistsError(f"target already exists: {target}")
    for relative, content in repository_files(spec).items():
        write_file(target / relative, content)
    return target


def parse_args() -> argparse.Namespace:
    """Parse a single bounded output location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".course-repo-staging"),
        help="New parent directory; generation refuses existing repository targets.",
    )
    return parser.parse_args()


def main() -> int:
    """Generate all four repositories without network or package installation."""
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    for spec in PROJECTS:
        target = build_repository(output_root, spec)
        print(f"CREATED {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
