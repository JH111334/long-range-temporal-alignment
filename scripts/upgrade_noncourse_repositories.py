#!/usr/bin/env python3
# ruff: noqa: E501, UP017
"""Nâng cấp có kiểm soát ba repository không thuộc nhóm môn học.

Script mặc định chỉ lập kế hoạch. Chỉ ``--apply`` mới ghi dữ liệu. Không cài
dependency, tải dữ liệu/model, gọi mạng, chạy Git hoặc tạo commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

SCRIPT_ID = "upgrade-noncourse-repositories"
SCRIPT_VERSION = 1
BACKUP_DIRECTORY = ".migration-backups/upgrade_noncourse_repositories"
BLOCK_START = f"<!-- BEGIN {SCRIPT_ID}:v{SCRIPT_VERSION} -->"
BLOCK_END = f"<!-- END {SCRIPT_ID}:v{SCRIPT_VERSION} -->"
LOG_PATTERN = re.compile(r"^.+_(?:Checkpoints|Error)_\d{2}-\d{2}-\d{2}_V[1-9]\d*\.md$")


class MigrationError(RuntimeError):
    """Báo lỗi preflight hoặc migration cần dừng fail-closed."""


@dataclass(frozen=True, slots=True)
class Action:
    """Mô tả một thay đổi file có thể sao lưu và hoàn tác."""

    kind: Literal["write", "move"]
    target: str
    reason: str
    content: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        """Từ chối action thiếu payload hoặc đường dẫn nguồn."""
        if self.kind == "write" and self.content is None:
            raise ValueError("Write action phải có content.")
        if self.kind == "move" and not self.source:
            raise ValueError("Move action phải có source.")

    def __repr__(self) -> str:
        """Trả về mô tả ngắn phục vụ chẩn đoán."""
        return f"Action(kind={self.kind!r}, target={self.target!r})"


@dataclass(slots=True)
class RepositoryPlan:
    """Giữ kế hoạch thay đổi cho đúng một repository."""

    name: str
    root: Path
    actions: list[Action] = field(default_factory=list)

    def add(self, action: Action) -> None:
        """Thêm action nếu chưa có target trùng nhau."""
        touched = {item.target for item in self.actions}
        if action.target in touched:
            raise MigrationError(f"Target bị lập kế hoạch hai lần: {action.target}")
        self.actions.append(action)

    def __repr__(self) -> str:
        """Trả về trạng thái kế hoạch ngắn."""
        return f"RepositoryPlan(name={self.name!r}, actions={len(self.actions)})"


@dataclass(frozen=True, slots=True)
class BackupContext:
    """Gắn plan với thư mục và manifest sao lưu."""

    plan: RepositoryPlan
    root: Path
    manifest_path: Path

    def __repr__(self) -> str:
        """Trả về vị trí manifest phục vụ rollback."""
        return f"BackupContext(repo={self.plan.name!r}, root={str(self.root)!r})"


def _normalize_text(content: str) -> str:
    """Chuẩn hóa line ending và newline cuối file."""
    return content.replace("\r\n", "\n").rstrip() + "\n"


def _sha256(path: Path) -> str | None:
    """Tính SHA-256 cho file hiện có, hoặc trả về None."""
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, relative_path: str) -> Path:
    """Giải đường dẫn tương đối và chặn thoát khỏi repository."""
    candidate = (root / Path(relative_path)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise MigrationError(f"Đường dẫn vượt repository: {relative_path}") from exc
    return candidate


def _read_text(path: Path) -> str:
    """Đọc UTF-8 và báo lỗi rõ nếu file không hợp lệ."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise MigrationError(f"File không phải UTF-8: {path}") from exc


def _managed_update(existing: str, body: str) -> str:
    """Thêm hoặc thay đúng một managed block trong file văn bản."""
    normalized = existing.replace("\r\n", "\n")
    start_count = normalized.count(BLOCK_START)
    end_count = normalized.count(BLOCK_END)
    if (start_count, end_count) not in {(0, 0), (1, 1)}:
        raise MigrationError("Managed block thiếu marker hoặc bị lặp.")
    block = f"{BLOCK_START}\n{body.strip()}\n{BLOCK_END}"
    if start_count == 0:
        return _normalize_text(f"{normalized.rstrip()}\n\n{block}")
    prefix, remainder = normalized.split(BLOCK_START, maxsplit=1)
    _, suffix = remainder.split(BLOCK_END, maxsplit=1)
    return _normalize_text(f"{prefix.rstrip()}\n\n{block}{suffix}")


def _plan_write(plan: RepositoryPlan, relative_path: str, content: str, reason: str) -> None:
    """Lập action ghi khi nội dung đích chưa đúng."""
    target = _resolve(plan.root, relative_path)
    desired = _normalize_text(content)
    if target.is_dir():
        raise MigrationError(f"Target write đang là thư mục: {target}")
    if target.is_file() and _read_text(target).replace("\r\n", "\n") == desired:
        return
    plan.add(Action("write", relative_path, reason, content=desired))


def _plan_managed_block(plan: RepositoryPlan, relative_path: str, body: str, reason: str) -> None:
    """Lập action cập nhật managed block mà không xóa nội dung người dùng."""
    target = _resolve(plan.root, relative_path)
    if not target.is_file():
        raise MigrationError(f"Thiếu file cần cập nhật managed block: {target}")
    _plan_write(plan, relative_path, _managed_update(_read_text(target), body), reason)


def _plan_move(
    plan: RepositoryPlan, source_relative: str, target_relative: str, reason: str
) -> None:
    """Lập action move, chặn ghi đè khác nội dung."""
    source = _resolve(plan.root, source_relative)
    target = _resolve(plan.root, target_relative)
    if not source.exists():
        return
    if not source.is_file():
        raise MigrationError(f"Chỉ migrate file, không migrate thư mục: {source}")
    if target.is_dir():
        raise MigrationError(f"Đích migrate đang là thư mục: {target}")
    if target.is_file() and source.read_bytes() != target.read_bytes():
        raise MigrationError(f"Xung đột nội dung khi migrate: {source} -> {target}")
    plan.add(Action("move", target_relative, reason, source=source_relative))


def render_general_rule(project_name: str) -> str:
    """Tạo quy tắc bắt buộc cho Documents/General."""
    return f"""# Quy tắc Documents/General — {project_name}

## Phạm vi được phép

`Documents/General` chỉ chứa `RULE.md` và hai loại log bất biến:

- `{project_name}_Checkpoints_DD-MM-YY_Vn.md`: ghi một lần chạy, cấu hình,
  artifact, kết quả và mức bằng chứng.
- `{project_name}_Error_DD-MM-YY_Vn.md`: ghi lỗi hoặc kết quả không đạt, bằng
  chứng, nguyên nhân đã xác nhận/giả thuyết, rollback và regression gate.

Không đặt nghiên cứu, ADR, hướng dẫn MCP/service, kiến trúc, meeting note, báo cáo
hoặc tài liệu tổng quan trong `General`; chuyển chúng sang `Decision-making` hoặc
`References` theo vai trò. Dùng `DD-MM-YY` vì dấu `/` là ký tự phân tách đường dẫn
trên Windows. `Vn` bắt đầu từ `V1` và tăng trong cùng ngày, không ghi đè log cũ.

## Trường bắt buộc

Mỗi log phải có: `Project`, `Ngày`, `Phiên bản`, `Loại`, `Evidence class`, owner,
Git revision, system profile, data/corpus manifest, config hash, lệnh đã chạy,
kết quả hoặc triệu chứng, artifact/hash, kết luận, blocker và bước kế tiếp.
Checkpoint phải phân biệt `scaffold`, `static_check`, `smoke_test`, `measured` và
`benchmark`; Error phải ghi `confirmed` hoặc `hypothesis`, rollback và regression
test. Không ghi token, PII, dữ liệu riêng, model weight hoặc transcript debug dài.

## Quy trình

1. Chốt ngày và tìm phiên bản lớn nhất đúng loại trong ngày.
2. Tạo file kế tiếp; không sửa file log đã khóa, trừ sửa lỗi chính tả có ghi chú.
3. Ghi cấu hình thực đo; RTX 2050 4 GB/16 GB chỉ là profile lịch sử cần đo lại.
4. Gắn config/data/code revision và đường dẫn artifact trước khi kết luận.
5. Nghiên cứu hoặc tinh chỉnh hệ thống phải cập nhật tài liệu canonical tương ứng;
   log chỉ ghi thao tác và bằng chứng của lần chạy.
"""


def render_skill_rule(project_name: str) -> str:
    """Tạo managed block về General cho local skill."""
    return f"""## Quy tắc log bắt buộc

Trước khi ghi `Documents/General`, đọc `Documents/General/RULE.md`. Project
`{project_name}` chỉ cho phép `RULE.md`,
`{project_name}_Checkpoints_DD-MM-YY_Vn.md` và
`{project_name}_Error_DD-MM-YY_Vn.md`; `Vn` tăng theo loại trong cùng ngày và
không ghi đè log cũ. Nghiên cứu/ADR/hướng dẫn/MCP đi vào `Decision-making` hoặc
`References`. Log phải có system profile, data/config/code revision, evidence
class, artifact/hash và rollback hoặc bước kế tiếp; không gọi scaffold/smoke là
benchmark.
"""


def render_checkpoint(project_name: str, version: int, actions: list[Action]) -> str:
    """Tạo checkpoint cho chính lần migration được áp dụng."""
    now = datetime.now().astimezone()
    action_lines = "\n".join(
        f"  - `{item.kind}` `{item.target}`: {item.reason}" for item in actions
    )
    return f"""# Checkpoint nâng cấp quy ước repository

- Project: {project_name}
- Ngày: {now:%Y-%m-%d %H:%M:%S %z}
- Phiên bản: V{version}
- Loại: checkpoint
- Evidence class: migration_applied
- Owner: local repository owner
- Git revision: NOT_CAPTURED — script không gọi Git
- System profile: RTX 2050 4 GB VRAM, RAM 16 GB là profile lịch sử; cần đo lại khi benchmark
- Data/corpus manifest: không đọc hoặc thay đổi dữ liệu
- Config hash: không áp dụng
- Lệnh dự kiến: `python scripts/upgrade_noncourse_repositories.py --apply`

## Thay đổi

{action_lines}

## Ranh giới kết quả

Đây là bằng chứng migration tài liệu/scaffold, không phải smoke test, benchmark,
kết quả mô hình hoặc mức sẵn sàng production. Script không cài package, tải
data/model, gọi service, commit hoặc push.

## Kết luận và bước kế tiếp

Chạy validator/test được liệt kê trong repository đích. Chỉ tạo benchmark log mới
khi có data/config/code revision, metric protocol và resource artifact thực đo.
"""


def render_data_feasibility() -> str:
    """Tạo ADR compact về dữ liệu và giới hạn 4 GB."""
    return """# ADR 0002 — Data feasibility và gate RTX 2050 4 GB

Trạng thái: `accepted-for-planning`; chưa phải kết quả benchmark.

## Vấn đề và đầu ra

- Task 1 LegalIR: query pháp lý → tối đa 100 document ID có thứ hạng, provenance
  và khả năng abstain khi corpus/rules không đủ.
- Task 2 LegalQA: query + các chunk được phép → câu trả lời trích xuất hoặc
  abstention; citation phải thuộc active corpus version.
- Warm-up và synthetic fixture chỉ kiểm schema/smoke. Chưa có full corpus và scoring
  specification thì không được suy diễn metric chính thức của DSC.

## Gate dữ liệu

1. Registry phải có URL chính thức, ngày truy cập, terms/license, checksum/revision,
   owner, data role và access scope.
2. Chuẩn hóa điều/khoản, tiêu đề, số hiệu, cơ quan, ngày hiệu lực và văn bản thay thế
   nhưng giữ raw text cùng provenance.
3. Deduplicate trước split; split theo văn bản/nguồn/lĩnh vực/thời gian. Một phiên
   bản luật hoặc near-duplicate không được đi qua train và held-out.
4. Tạo private-like holdout và slice theo lĩnh vực, query dài/ngắn, điều khoản bị
   sửa đổi, không có đáp án và nhiều văn bản cạnh tranh.
5. Hard negatives lấy từ cùng lĩnh vực hoặc văn bản gần tên nhưng khác điều/khoản;
   không mine từ test rồi quay lại training.

## Thang triển khai

| Tier | Baseline/candidate | Quyết định |
|---|---|---|
| 0, CPU | BM25, TF-IDF, RRF; Task 2 extractive evidence | Triển khai và đóng băng evaluator trước |
| 1 | Frozen compact encoder/ViLegalBERT; exact search | Benchmark batch 1 sau resource probe |
| 2 | Vietnamese embedding khoảng 0,6B, rerank top 20–100 | Chạy tuần tự, cache embedding, đo peak VRAM |
| 3 | Remote inference đã pin model/provider/revision | Chỉ sau privacy, quota, timeout và schema gate |
| Loại P1 | Fine-tune 7B–72B, winner ensemble, full GraphRAG | Không phù hợp baseline local 4 GB |

Không suy VRAM từ số tham số. Mỗi candidate phải ghi precision, input length,
batch, peak VRAM/RAM, latency p50/p95, OOM/fallback và artifact hash.

## Metric và stop condition

Task 1 báo per-query recall/rank metric theo spec khi có, macro-F2 chỉ khi organizer
xác nhận. Task 2 báo support/completeness, citation integrity, abstention và invalid
rate tách khỏi retrieval. Dừng candidate nếu không hơn paired baseline, lỗi citation,
OOM sau cấu hình tối thiểu, hoặc chi phí remote vượt budget đã duyệt.
"""


def render_project_context() -> str:
    """Tạo context bền vững để tiếp tục Evidence project ở phiên mới."""
    return """# Project context — Evidence-Grounded-Text-Agent

Ngày khóa: 25/08/2026. Trạng thái hiện tại là planning/scaffold; chưa có benchmark
hoặc model/dataset thật được duyệt.

## Mục tiêu đã khóa

- Task 1: retrieval-first LegalIR tiếng Việt, trả document ID có provenance và
  abstention; BM25/TF-IDF/RRF trước dense/rerank.
- Task 2: chỉ tạo câu trả lời extractive/evidence-grounded từ active corpus;
  citation phải đúng điều/khoản, phiên bản và phạm vi truy cập.
- Đóng góp phù hợp sinh viên: data contract, hierarchy-aware chunking, hard
  negatives, group/private-like holdout, threshold/abstention và error analysis.
- Không tái lập winner 32B–72B, không generation/agent trước retrieval gate và
  không gửi hồ sơ/PII ra dịch vụ ngoài.

## Nguồn sự thật

- Quyết định retrieval-first: `0001-retrieval-before-generation.md`.
- Khả thi/data/4 GB: `0002-data-feasibility-4gb.md`.
- Nguồn/paper: `Documents/References/Shared/papers-and-baselines.md`.
- Service: `Documents/References/Shared/external-inference.md` và
  `Source/Shared/ServiceAdapters/`.
- Lỗi đã chắt lọc: `ERROR-EXPERIENCE.md`; lỗi chạy thật chỉ vào `General` theo
  `RULE.md`.

## Gate tiếp theo

Khóa organizer specification/corpus rights → dựng corpus manifest và legal
hierarchy → group/private-like split + qrels → BM25/RRF artifact → error slices →
frozen compact encoder/rerank top-k nếu paired gain và resource probe đạt.

RTX 2050 4 GB là trần hoạch định; mọi probe phải đo input length, batch,
precision, peak RAM/VRAM, p50/p95 và fallback. Streamlit chỉ đọc artifact có hash.
"""


def render_papers_and_baselines() -> str:
    """Tạo evidence map từ nguồn primary/official."""
    return """# Papers và baseline có thể kiểm chứng

Các kết quả dưới đây thuộc protocol của nguồn, không phải DSC score hoặc tuyên bố
leaderboard hiện tại.

| Nguồn primary/official | Claim giới hạn | Quyết định project |
|---|---|---|
| [VLSP 2025 DRiLL overview](https://aclanthology.org/2025.vlsp-1.16.pdf) | 3.000+ vấn đề, 59.636 văn bản; BM25 private macro-F2 0,3365, baseline hai tầng 0,5512; hệ top dùng hybrid/LTR/HyDE và LLM 32B–72B | Adopt split private-like, hierarchy, hard negatives; reject tái lập winner trên 4 GB |
| [ViDRILL](https://aclanthology.org/2025.vlsp-1.17.pdf) | BM25 + dense + reranker, dual-level chunking và hard negatives | Benchmark cấu trúc hai tầng với model nhỏ |
| [Simple two-stage DRiLL](https://aclanthology.org/2025.vlsp-1.20/) | Sparse+dense, cross-encoder, title/document chunk và hard-negative mining | Hướng cải thiện khả thi đầu tiên |
| [ViRE](https://aclanthology.org/2026.findings-eacl.110/) | Benchmark retrieval tiếng Việt đa miền; lexical hybrid ổn định và model scale không dự báo tốt chất lượng | Adopt domain/data-regime analysis; benchmark Vietnamese encoder sau probe |
| [ViLegalLM](https://aclanthology.org/2026.findings-acl.1801/) | Công bố ViLegalBERT 135M và ViLegalQwen 1,5B/1,7B cùng code/corpus/model | Benchmark ViLegalBERT; model lớn chỉ remote hoặc probe có giới hạn |
| [BEIR](https://openreview.net/pdf?id=wCu6T5xFjeJ) | BM25 là zero-shot baseline bền; dense/rerank có thể mạnh nhưng đắt và thất bại OOD | Giữ BM25, domain slice và paired ablation |
| [RAG-RewardBench](https://aclanthology.org/2025.findings-acl.877/) | Đánh giá multi-hop, citation chi tiết, abstention và bằng chứng xung đột | Dùng taxonomy cho Task 2, không suy chất lượng trả lời chỉ từ retrieval |

## Sai lầm cần chặn

- Tuning chunk/threshold trên test; random split làm rò cùng văn bản hoặc phiên bản.
- Đánh đồng warm-up/synthetic smoke với benchmark hoặc dùng public split để chọn
  model đến khi overfit; VLSP DRiLL cho thấy nhiều đội giảm trên private.
- Chọn model theo kích thước hoặc score khác candidate pool/protocol.
- Để reranker/generator che lỗi candidate generation; phải lưu recall trước rerank.
- Trả lời pháp lý không citation, citation sai phiên bản/hiệu lực hoặc không abstain.
"""


def render_external_inference_guide() -> str:
    """Tạo guideline external inference fail-closed."""
    return """# External inference — lý do, hợp đồng và lộ trình

Remote inference chỉ dùng để thử candidate không vừa 4 GB hoặc tái dùng model đã
được duyệt; không phải đường tắt bỏ qua data/evaluation gate. Core retrieval phải
chạy được khi service tắt.

## Hợp đồng bắt buộc

1. Pin `model_id`, provider và immutable revision; cấm `main`, `latest` hoặc silent
   fallback. Preflight license/gated state/task/files trước download/inference.
2. Request ghi `request_id`, schema version, task, input hash, privacy class và
   timeout hữu hạn. Không gửi PII, hồ sơ riêng hoặc organizer data nếu chưa duyệt.
3. Response phải khớp request ID, model/provider/revision và response schema; sai
   provenance phải fail-closed, không chuyển lỗi thành câu trả lời/vector rỗng.
4. Retry chỉ lỗi transient 429/502/503, có backoff/cap; không retry validation,
   privacy, authentication hoặc schema error. Ghi latency, usage và provider.
5. Cache ngoài Git theo input/model/revision/config hash; token chỉ qua secret store.

## Hugging Face và Kaggle

- [`HfApi.model_info`](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api)
  dùng để lấy SHA, card/license, gated state, pipeline và provider metadata.
- [`snapshot_download`](https://huggingface.co/docs/huggingface_hub/en/guides/download)
  phải dùng exact revision và `allow_patterns`; incomplete cache là lỗi, không dùng
  artifact một phần.
- [`InferenceClient`](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client)
  luôn đặt timeout; model tồn tại trên Hub không chứng minh provider đang phục vụ.
- [Endpoint autoscaling](https://huggingface.co/docs/inference-endpoints/autoscaling)
  có cold start/502 khi scale-to-zero; adapter trả `REMOTE_NOT_READY` sau retry cap.
- [KaggleHub](https://github.com/Kaggle/kagglehub) phải pin `/versions/N`, managed
  cache và manifest; có mặt trên Kaggle không đồng nghĩa được phép tái phân phối.

## Lộ trình code

`InferenceConfig` khóa provenance/privacy/timeout → `InferenceRequest` khóa schema
→ adapter kiểm gate → transport tùy chọn → xác minh `InferenceResponse` → artifact
manifest. Scaffold mặc định `enabled=False` và không có transport, vì vậy không thể
gọi mạng. Unit test phải phủ disabled, mutable revision, privacy và provenance lệch.
"""


def render_service_adapter() -> str:
    """Tạo scaffold inference thuần standard library."""
    return '''"""Định nghĩa external inference adapter mặc định fail-closed."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{40,64}$")


class InferenceAdapterError(RuntimeError):
    """Làm lớp gốc cho lỗi external inference."""


class InferenceDisabledError(InferenceAdapterError):
    """Báo remote inference chưa được bật hoặc chưa có transport."""


class InferencePrivacyError(InferenceAdapterError):
    """Báo request vượt privacy policy."""


class InferenceSchemaError(InferenceAdapterError):
    """Báo request/response sai schema."""


class InferenceProvenanceError(InferenceAdapterError):
    """Báo response không khớp model/provider/revision đã pin."""


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Khóa model, provider, revision, timeout và privacy gate."""

    model_id: str
    provider: str
    revision: str
    enabled: bool = False
    timeout_seconds: float = 30.0
    allow_sensitive_data: bool = False
    request_schema: str = "egta.inference.request.v1"
    response_schema: str = "egta.inference.response.v1"

    def __post_init__(self) -> None:
        """Từ chối provenance mơ hồ và timeout không an toàn."""
        if not self.model_id.strip() or not self.provider.strip():
            raise ValueError("model_id và provider không được rỗng.")
        if not IMMUTABLE_REVISION.fullmatch(self.revision):
            raise ValueError("revision phải là commit/checksum hex 40-64 ký tự.")
        if not 0 < self.timeout_seconds <= 120:
            raise ValueError("timeout_seconds phải trong khoảng (0, 120].")


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """Mang payload có schema và privacy classification rõ ràng."""

    request_id: str
    task: str
    payload: Mapping[str, object]
    schema: str = "egta.inference.request.v1"
    contains_sensitive_data: bool = False

    def __post_init__(self) -> None:
        """Từ chối ID, task hoặc payload rỗng."""
        if not self.request_id.strip() or not self.task.strip() or not self.payload:
            raise ValueError("request_id, task và payload phải có giá trị.")


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    """Mang output cùng provenance bắt buộc của service."""

    request_id: str
    model_id: str
    provider: str
    revision: str
    schema: str
    output: Mapping[str, object]


class InferenceTransport(Protocol):
    """Mô tả transport được inject; scaffold không cung cấp network client."""

    def __call__(
        self, config: InferenceConfig, request: InferenceRequest
    ) -> Mapping[str, object]:
        """Gửi request và trả mapping chưa tin cậy từ provider."""
        ...


class ExternalInferenceAdapter:
    """Kiểm contract trước/sau transport và dừng khi chưa được duyệt."""

    def __init__(
        self, config: InferenceConfig, transport: InferenceTransport | None = None
    ) -> None:
        """Lưu config và transport mà không thực hiện I/O."""
        self._config = config
        self._transport = transport

    def __repr__(self) -> str:
        """Ẩn payload/token và chỉ hiện provenance an toàn."""
        return (
            "ExternalInferenceAdapter("
            f"model_id={self._config.model_id!r}, provider={self._config.provider!r}, "
            f"revision={self._config.revision!r}, enabled={self._config.enabled!r})"
        )

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Xác minh gate, gọi transport tùy chọn và kiểm response."""
        if not self._config.enabled:
            raise InferenceDisabledError("Remote inference mặc định bị tắt.")
        if self._transport is None:
            raise InferenceDisabledError("Chưa cấu hình transport đã được duyệt.")
        if request.schema != self._config.request_schema:
            raise InferenceSchemaError("Request schema không khớp config.")
        if request.contains_sensitive_data and not self._config.allow_sensitive_data:
            raise InferencePrivacyError("Privacy policy từ chối dữ liệu nhạy cảm.")
        raw = self._transport(self._config, request)
        return self._parse_response(request, raw)

    def _parse_response(
        self, request: InferenceRequest, raw: Mapping[str, object]
    ) -> InferenceResponse:
        """Từ chối response thiếu schema, output hoặc provenance chính xác."""
        required = ("request_id", "model_id", "provider", "revision", "schema", "output")
        if any(key not in raw for key in required) or not isinstance(
            raw["output"], Mapping
        ):
            raise InferenceSchemaError("Response thiếu trường bắt buộc hoặc output sai kiểu.")
        expected = (
            request.request_id,
            self._config.model_id,
            self._config.provider,
            self._config.revision,
            self._config.response_schema,
        )
        observed = tuple(str(raw[key]) for key in required[:5])
        if observed != expected:
            raise InferenceProvenanceError(
                "Response không khớp request/model/provider/revision/schema."
            )
        return InferenceResponse(
            request_id=observed[0],
            model_id=observed[1],
            provider=observed[2],
            revision=observed[3],
            schema=observed[4],
            output=cast(Mapping[str, object], raw["output"]),
        )
'''


def render_service_init() -> str:
    """Tạo public API của ServiceAdapters."""
    return '''"""Xuất public contract cho external inference adapter."""

from .inference import (
    ExternalInferenceAdapter,
    InferenceAdapterError,
    InferenceConfig,
    InferenceDisabledError,
    InferencePrivacyError,
    InferenceProvenanceError,
    InferenceRequest,
    InferenceResponse,
    InferenceSchemaError,
    InferenceTransport,
)

__all__ = [
    "ExternalInferenceAdapter",
    "InferenceAdapterError",
    "InferenceConfig",
    "InferenceDisabledError",
    "InferencePrivacyError",
    "InferenceProvenanceError",
    "InferenceRequest",
    "InferenceResponse",
    "InferenceSchemaError",
    "InferenceTransport",
]
'''


def render_service_tests() -> str:
    """Tạo unit test tối thiểu cho scaffold inference."""
    return '''"""Kiểm tra external inference scaffold fail-closed."""

from collections.abc import Mapping

import pytest
from ServiceAdapters import (
    ExternalInferenceAdapter,
    InferenceConfig,
    InferenceDisabledError,
    InferencePrivacyError,
    InferenceProvenanceError,
    InferenceRequest,
)

REVISION = "a" * 40


def request(*, sensitive: bool = False) -> InferenceRequest:
    """Tạo request synthetic không chứa dữ liệu organizer."""
    return InferenceRequest(
        request_id="req-001",
        task="feature-extraction",
        payload={"inputs": ["văn bản synthetic"]},
        contains_sensitive_data=sensitive,
    )


def response(config: InferenceConfig, item: InferenceRequest) -> Mapping[str, object]:
    """Tạo response synthetic khớp provenance."""
    return {
        "request_id": item.request_id,
        "model_id": config.model_id,
        "provider": config.provider,
        "revision": config.revision,
        "schema": config.response_schema,
        "output": {"embedding": [0.0, 1.0]},
    }


def test_remote_inference_is_disabled_by_default() -> None:
    config = InferenceConfig("org/model", "huggingface", REVISION)
    with pytest.raises(InferenceDisabledError):
        ExternalInferenceAdapter(config).infer(request())


def test_mutable_revision_is_rejected() -> None:
    with pytest.raises(ValueError):
        InferenceConfig("org/model", "huggingface", "main")


def test_sensitive_request_is_rejected_before_transport() -> None:
    config = InferenceConfig("org/model", "huggingface", REVISION, enabled=True)
    with pytest.raises(InferencePrivacyError):
        ExternalInferenceAdapter(config, response).infer(request(sensitive=True))


def test_response_provenance_mismatch_is_rejected() -> None:
    config = InferenceConfig("org/model", "huggingface", REVISION, enabled=True)

    def wrong_provider(_: InferenceConfig, item: InferenceRequest) -> Mapping[str, object]:
        payload = dict(response(config, item))
        payload["provider"] = "unexpected-provider"
        return payload

    with pytest.raises(InferenceProvenanceError):
        ExternalInferenceAdapter(config, wrong_provider).infer(request())


def test_matching_synthetic_response_is_returned() -> None:
    config = InferenceConfig("org/model", "huggingface", REVISION, enabled=True)
    result = ExternalInferenceAdapter(config, response).infer(request())
    assert result.output == {"embedding": [0.0, 1.0]}
'''


def render_evidence_agents_block() -> str:
    """Tạo rule bổ sung cho AGENTS của Evidence repository."""
    return """## Data, external inference và logging gate

- Đọc `Documents/Decision-making/Shared/0002-data-feasibility-4gb.md` trước khi
  đổi data/split/model; baseline CPU và exact search đi trước neural candidate.
- External inference chỉ qua `Source/Shared/ServiceAdapters`, pin
  model/provider/revision, timeout/privacy/schema và mặc định fail-closed.
- Không gửi PII, organizer/private corpus hoặc secret ra service chưa được duyệt.
- Không dùng emoji, sticker, banner ký hiệu hoặc thông báo ăn mừng trong mã, test,
  log hay CLI; chỉ giữ ký hiệu có nghĩa kỹ thuật hoặc dữ liệu protocol đã ghi rõ.
- Đọc `Documents/General/RULE.md` trước khi tạo checkpoint/error log; file khác
  phải vào `Decision-making` hoặc `References`.
- Validator canonical:
  `uv run python skills/retrieval-delivery/scripts/validate_project.py`.
"""


def render_evidence_readme_block() -> str:
    """Tạo README block mô tả scaffold và tài liệu mới."""
    return """## Data feasibility và external inference scaffold

Hướng triển khai trên RTX 2050 4 GB bắt đầu bằng BM25/TF-IDF/RRF và extractive
evidence; frozen model/reranker chỉ được benchmark sau split/leakage/resource gate.
`Source/Shared/ServiceAdapters` hiện chỉ là contract scaffold thuần Python: remote
inference tắt mặc định, không có network transport và không chứng minh model quality.

- [Project context](Documents/Decision-making/Shared/0003-project-context.md)
- [Data feasibility và gate 4 GB](Documents/Decision-making/Shared/0002-data-feasibility-4gb.md)
- [Papers và baseline](Documents/References/Shared/papers-and-baselines.md)
- [External inference](Documents/References/Shared/external-inference.md)
- [Quy tắc checkpoint/error log](Documents/General/RULE.md)

Unit test scaffold: `uv run pytest tests/Shared/test_service_adapters.py`.
Validator: `uv run python skills/retrieval-delivery/scripts/validate_project.py`.
"""


def _update_pyproject(existing: str) -> str:
    """Đăng ký ServiceAdapters package bằng exact textual gate."""
    normalized = existing.replace("\r\n", "\n")
    package_line = '  "Source/Shared/ServiceAdapters",'
    if package_line in normalized:
        return _normalize_text(normalized)
    anchor = '  "Source/Shared/text_retrieval_agent",'
    if normalized.count(anchor) != 1:
        raise MigrationError("Không tìm thấy package anchor duy nhất trong pyproject.toml.")
    return _normalize_text(normalized.replace(anchor, f"{anchor}\n{package_line}"))


def _update_validator(existing: str) -> str:
    """Thêm required paths và rule kiểm Documents/General vào validator."""
    text = existing.replace("\r\n", "\n")
    import_marker = "import re\n"
    if import_marker not in text:
        anchor = "import json\n"
        if text.count(anchor) != 1:
            raise MigrationError("Không tìm thấy import anchor trong validator.")
        text = text.replace(anchor, anchor + import_marker)
    text = _inject_validator_required_paths(text)
    text = _inject_validator_function(text)
    return _inject_validator_call(text)


def _inject_validator_required_paths(text: str) -> str:
    """Chèn các file mới vào REQUIRED_PATHS một lần."""
    marker = "    # BEGIN upgrade-noncourse-repositories required paths"
    if marker in text:
        return text
    anchor = '    "tests/Task2/test_legal_qa.py",\n)'
    if text.count(anchor) != 1:
        raise MigrationError("Không tìm thấy REQUIRED_PATHS anchor trong validator.")
    block = """    # BEGIN upgrade-noncourse-repositories required paths
    "Documents/General/RULE.md",
    "Documents/Decision-making/Shared/0002-data-feasibility-4gb.md",
    "Documents/Decision-making/Shared/0003-project-context.md",
    "Documents/References/Shared/papers-and-baselines.md",
    "Documents/References/Shared/external-inference.md",
    "Source/Shared/ServiceAdapters/__init__.py",
    "Source/Shared/ServiceAdapters/inference.py",
    "tests/Shared/test_service_adapters.py",
    # END upgrade-noncourse-repositories required paths
)"""
    return text.replace(anchor, anchor.removesuffix(")") + block)


def _inject_validator_function(text: str) -> str:
    """Chèn validator tên/nội dung log General."""
    marker = "def _validate_general_documents(root: Path) -> list[str]:"
    if marker in text:
        return text
    anchor = "\ndef main() -> int:\n"
    if text.count(anchor) != 1:
        raise MigrationError("Không tìm thấy main anchor trong validator.")
    block = r'''

GENERAL_LOG_PATTERN = re.compile(
    r"^Evidence-Grounded-Text-Agent_(?:Checkpoints|Error)_\d{2}-\d{2}-\d{2}_V[1-9]\d*\.md$"
)


def _validate_general_documents(root: Path) -> list[str]:
    """Chỉ cho phép RULE và checkpoint/error log có metadata tối thiểu."""
    errors: list[str] = []
    general = root / "Documents/General"
    required_fields = (
        "- Project:",
        "- Ngày:",
        "- Phiên bản:",
        "- Loại:",
        "- Evidence class:",
    )
    for path in sorted(general.iterdir()):
        if path.name == "RULE.md":
            continue
        if not path.is_file() or not GENERAL_LOG_PATTERN.fullmatch(path.name):
            errors.append(f"Invalid Documents/General entry: {path.name}")
            continue
        content = path.read_text(encoding="utf-8")
        missing = [field for field in required_fields if field not in content]
        if missing:
            errors.append(
                f"General log missing fields in {path.name}: {', '.join(missing)}"
            )
        expected_type = "- Loại: error" if "_Error_" in path.name else "- Loại: checkpoint"
        if expected_type not in content:
            errors.append(f"General log type mismatch: {path.name}")
    return errors
'''
    return text.replace(anchor, _normalize_text(block).rstrip() + anchor)


def _inject_validator_call(text: str) -> str:
    """Gọi General validator trước gate lỗi cuối."""
    call = "    errors.extend(_validate_general_documents(root))\n"
    if call in text:
        return _normalize_text(text)
    anchor = "    errors.extend(_validate_local_data(root, tracked))\n"
    if text.count(anchor) != 1:
        raise MigrationError("Không tìm thấy validation-call anchor.")
    return _normalize_text(text.replace(anchor, anchor + call))


def _legacy_target(project_name: str, file_name: str) -> str:
    """Chọn vị trí References cho file không phải log trong General."""
    if file_name.casefold() == "mcp-tools.md":
        if project_name == "Evidence-Grounded-Text-Agent":
            return "Documents/References/Shared/MCP-TOOLS.md"
        return "Documents/References/MCP-TOOLS.md"
    return f"Documents/References/Legacy-General/{file_name}"


def _plan_general_cleanup(plan: RepositoryPlan) -> None:
    """Di chuyển mọi file General sai loại sang References."""
    general = plan.root / "Documents/General"
    if not general.exists():
        return
    if not general.is_dir():
        raise MigrationError(f"Documents/General không phải thư mục: {general}")
    for path in sorted(general.iterdir(), key=lambda item: item.name.casefold()):
        if path.name == "RULE.md" or LOG_PATTERN.fullmatch(path.name):
            continue
        relative = path.relative_to(plan.root).as_posix()
        _plan_move(
            plan,
            relative,
            _legacy_target(plan.name, path.name),
            "Đưa tài liệu không phải runtime/error log ra khỏi General",
        )


def _next_log_version(plan: RepositoryPlan, kind: str, date_token: str) -> int:
    """Tìm version kế tiếp theo project, loại và ngày."""
    general = plan.root / "Documents/General"
    pattern = re.compile(
        rf"^{re.escape(plan.name)}_{re.escape(kind)}_{re.escape(date_token)}_V([1-9]\d*)\.md$"
    )
    versions = []
    if general.is_dir():
        for path in general.iterdir():
            match = pattern.fullmatch(path.name)
            if match:
                versions.append(int(match.group(1)))
    return max(versions, default=0) + 1


def _add_checkpoint(plan: RepositoryPlan) -> None:
    """Thêm checkpoint chỉ khi có thay đổi thực sự khác checkpoint."""
    if not plan.actions:
        return
    date_token = datetime.now().astimezone().strftime("%d-%m-%y")
    version = _next_log_version(plan, "Checkpoints", date_token)
    relative = f"Documents/General/{plan.name}_Checkpoints_{date_token}_V{version}.md"
    content = render_checkpoint(plan.name, version, list(plan.actions))
    _plan_write(plan, relative, content, "Ghi checkpoint cho migration đã áp dụng")


def _build_evidence_plan(root: Path) -> RepositoryPlan:
    """Lập đầy đủ task General, research và inference cho Evidence repository."""
    plan = RepositoryPlan("Evidence-Grounded-Text-Agent", root)
    _plan_general_cleanup(plan)
    _plan_write(
        plan,
        "Documents/General/RULE.md",
        render_general_rule(plan.name),
        "Khóa quy tắc General",
    )
    _plan_write(
        plan,
        "Documents/Decision-making/Shared/0002-data-feasibility-4gb.md",
        render_data_feasibility(),
        "Chốt data feasibility và gate 4 GB",
    )
    _plan_write(
        plan,
        "Documents/Decision-making/Shared/0003-project-context.md",
        render_project_context(),
        "Lưu context bền vững của project",
    )
    _plan_write(
        plan,
        "Documents/References/Shared/papers-and-baselines.md",
        render_papers_and_baselines(),
        "Bổ sung evidence map primary/official",
    )
    _plan_write(
        plan,
        "Documents/References/Shared/external-inference.md",
        render_external_inference_guide(),
        "Bổ sung guideline external inference",
    )
    _plan_write(
        plan,
        "Source/Shared/ServiceAdapters/__init__.py",
        render_service_init(),
        "Tạo public adapter API",
    )
    _plan_write(
        plan,
        "Source/Shared/ServiceAdapters/inference.py",
        render_service_adapter(),
        "Tạo inference scaffold fail-closed",
    )
    _plan_write(
        plan,
        "tests/Shared/test_service_adapters.py",
        render_service_tests(),
        "Tạo unit test adapter tối thiểu",
    )
    _plan_managed_block(plan, "AGENTS.md", render_evidence_agents_block(), "Đồng bộ project gates")
    _plan_managed_block(
        plan, "README.md", render_evidence_readme_block(), "Đồng bộ tài liệu điều hướng"
    )
    _plan_managed_block(
        plan,
        "skills/retrieval-delivery/SKILL.md",
        render_skill_rule(plan.name),
        "Khóa local-skill logging rule",
    )
    pyproject = _read_text(root / "pyproject.toml")
    _plan_write(
        plan,
        "pyproject.toml",
        _update_pyproject(pyproject),
        "Đăng ký ServiceAdapters package",
    )
    validator_path = "skills/retrieval-delivery/scripts/validate_project.py"
    validator = _read_text(root / validator_path)
    _plan_write(
        plan,
        validator_path,
        _update_validator(validator),
        "Thêm required paths và General validator",
    )
    _add_checkpoint(plan)
    return plan


def _build_rules_only_plan(name: str, root: Path) -> RepositoryPlan:
    """Lập task General-only, không mở rộng research hoặc source code."""
    plan = RepositoryPlan(name, root)
    _plan_general_cleanup(plan)
    _plan_write(
        plan,
        "Documents/General/RULE.md",
        render_general_rule(name),
        "Khóa quy tắc General",
    )
    _plan_managed_block(
        plan,
        "skills/retrieval-delivery/SKILL.md",
        render_skill_rule(name),
        "Khóa local-skill logging rule",
    )
    _add_checkpoint(plan)
    return plan


def build_plans(documents_root: Path) -> list[RepositoryPlan]:
    """Xác minh repository đích và lập ba kế hoạch tách biệt."""
    roots = {
        name: (documents_root / name).resolve()
        for name in (
            "Evidence-Grounded-Text-Agent",
            "DSC-TinDipLaPo",
            "Multimodal-Asset-Retrieval",
        )
    }
    missing = [name for name, root in roots.items() if not root.is_dir()]
    if missing:
        raise MigrationError("Thiếu repository: " + ", ".join(missing))
    return [
        _build_evidence_plan(roots["Evidence-Grounded-Text-Agent"]),
        _build_rules_only_plan("DSC-TinDipLaPo", roots["DSC-TinDipLaPo"]),
        _build_rules_only_plan("Multimodal-Asset-Retrieval", roots["Multimodal-Asset-Retrieval"]),
    ]


def _action_paths(action: Action) -> list[str]:
    """Liệt kê mọi đường dẫn cần snapshot cho một action."""
    paths = [action.target]
    if action.source:
        paths.append(action.source)
    return list(dict.fromkeys(paths))


def _atomic_write(path: Path, content: str, run_id: str) -> None:
    """Ghi UTF-8 qua file tạm cùng thư mục rồi replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{run_id}.tmp")
    temporary.write_text(_normalize_text(content), encoding="utf-8", newline="\n")
    temporary.replace(path)


def _manifest_payload(plan: RepositoryPlan, run_id: str) -> dict[str, object]:
    """Tạo manifest trước thay đổi với hash và trạng thái file."""
    entries: list[dict[str, object]] = []
    for action in plan.actions:
        paths = []
        for relative in _action_paths(action):
            path = _resolve(plan.root, relative)
            paths.append({"path": relative, "existed": path.is_file(), "sha256": _sha256(path)})
        entries.append(
            {
                "kind": action.kind,
                "source": action.source,
                "target": action.target,
                "reason": action.reason,
                "paths_before": paths,
            }
        )
    return {
        "script": SCRIPT_ID,
        "version": SCRIPT_VERSION,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(plan.root),
        "status": "prepared",
        "actions": entries,
        "rollback": "Khôi phục files/<relative-path>; xóa đúng target có existed=false.",
    }


def _prepare_backup(plan: RepositoryPlan, run_id: str) -> BackupContext:
    """Sao lưu mọi file bị chạm và ghi manifest trước migration."""
    backup_root = _resolve(plan.root, f"{BACKUP_DIRECTORY}/{run_id}")
    files_root = backup_root / "files"
    manifest = _manifest_payload(plan, run_id)
    backup_root.mkdir(parents=True, exist_ok=False)
    copied: set[str] = set()
    for action in plan.actions:
        for relative in _action_paths(action):
            if relative in copied:
                continue
            source = _resolve(plan.root, relative)
            if source.is_file():
                destination = files_root / Path(relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            copied.add(relative)
    manifest_path = backup_root / "manifest.json"
    _atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2), run_id)
    return BackupContext(plan, backup_root, manifest_path)


def _update_manifest(context: BackupContext, status: str, error: str | None = None) -> None:
    """Cập nhật trạng thái manifest mà không làm mất bằng chứng trước đó."""
    payload = json.loads(_read_text(context.manifest_path))
    payload["status"] = status
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    if error:
        payload["error"] = error
    _atomic_write(
        context.manifest_path,
        json.dumps(payload, ensure_ascii=False, indent=2),
        str(payload["run_id"]),
    )


def _apply_action(plan: RepositoryPlan, action: Action, run_id: str) -> None:
    """Áp dụng một action đã preflight và backup."""
    target = _resolve(plan.root, action.target)
    if action.kind == "write":
        assert action.content is not None
        _atomic_write(target, action.content, run_id)
        return
    assert action.source is not None
    source = _resolve(plan.root, action.source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if source.read_bytes() != target.read_bytes():
            raise MigrationError(f"Target đổi sau preflight: {target}")
        source.unlink()
        return
    source.replace(target)


def _rollback(context: BackupContext, error: str) -> None:
    """Khôi phục trạng thái file trước migration theo manifest."""
    payload = json.loads(_read_text(context.manifest_path))
    snapshots: dict[str, bool] = {}
    for action in payload["actions"]:
        for item in action["paths_before"]:
            snapshots[str(item["path"])] = bool(item["existed"])
    for relative, existed in reversed(list(snapshots.items())):
        target = _resolve(context.plan.root, relative)
        backup = context.root / "files" / Path(relative)
        if existed:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, target)
        elif target.is_file():
            target.unlink()
    _update_manifest(context, "rolled_back", error)


def apply_plans(plans: list[RepositoryPlan]) -> None:
    """Backup tất cả, áp dụng tuần tự và rollback mọi repo khi có lỗi."""
    active = [plan for plan in plans if plan.actions]
    if not active:
        print("Không có thay đổi cần áp dụng.")
        return
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    contexts: list[BackupContext] = []
    try:
        for plan in active:
            contexts.append(_prepare_backup(plan, run_id))
        for context in contexts:
            for action in context.plan.actions:
                _apply_action(context.plan, action, run_id)
            _update_manifest(context, "applied")
    except (OSError, ValueError, MigrationError) as exc:
        rollback_errors: list[str] = []
        for context in reversed(contexts):
            try:
                _rollback(context, str(exc))
            except (OSError, ValueError, MigrationError) as rollback_exc:
                rollback_errors.append(f"{context.plan.name}: {rollback_exc}")
        detail = f"Migration thất bại và đã yêu cầu rollback: {exc}"
        if rollback_errors:
            detail += "; rollback lỗi: " + " | ".join(rollback_errors)
        raise MigrationError(detail) from exc
    for context in contexts:
        print(f"APPLIED {context.plan.name}: {len(context.plan.actions)} action")
        print(f"  Manifest: {context.manifest_path}")


def print_plan(plans: list[RepositoryPlan]) -> None:
    """In kế hoạch deterministic mà không ghi file."""
    total = 0
    for plan in plans:
        print(f"[{plan.name}] {plan.root}")
        if not plan.actions:
            print("  Không có thay đổi.")
            continue
        for action in plan.actions:
            source = f"{action.source} -> " if action.source else ""
            print(f"  {action.kind.upper():5} {source}{action.target}")
            print(f"        {action.reason}")
        total += len(plan.actions)
    print(f"Tổng action: {total}. Chế độ hiện tại: dry-run.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Phân tích CLI với dry-run là mặc định an toàn."""
    parser = argparse.ArgumentParser(
        description="Nâng cấp Evidence đầy đủ và General-only cho DSC/Multimodal."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Ghi thay đổi sau backup; mặc định chỉ dry-run.",
    )
    parser.add_argument(
        "--documents-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Thư mục cha chứa ba repository; mặc định là Documents cạnh workspace.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Lập kế hoạch hoặc áp dụng migration đã backup."""
    args = parse_args(argv)
    try:
        plans = build_plans(args.documents_root.resolve())
        if args.apply:
            apply_plans(plans)
        else:
            print_plan(plans)
        return 0
    except MigrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
