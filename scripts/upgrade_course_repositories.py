"""Nâng cấp có backup cho bốn repository môn học theo hợp đồng data-first.

Mặc định script chỉ in kế hoạch. Chỉ ``--apply`` mới ghi file; ``--validate-only``
kiểm tra trạng thái đích mà không thay đổi repository. Script không tải dữ liệu,
mô hình, package và không gọi mạng.

Rollback thủ công dùng ``.migration-backups/course-repositories-v3-format/manifest.json``:
khôi phục file có bản gốc từ ``original/<relative-path>`` và xóa file mới có
``original_sha256 = null``. Script không tự rollback để tránh ghi đè thay đổi mới.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from textwrap import dedent as _stdlib_dedent

from course_repo_catalog import PROJECTS, CourseProject

MIGRATION_ID = "course-repositories-v3-format"
BACKUP_RELATIVE = Path(".migration-backups") / MIGRATION_ID
LEGACY_PATHS = (
    "Documents/General/COURSE.md",
    "Documents/General/MCP-TOOLS.md",
    "Documents/General/DEFERRED-MULTIMODAL.md",
    "Documents/General/STATUS.md",
    "Documents/General/DOCX-QA.md",
)
CHECKPOINT_HEADINGS = (
    "# Checkpoint",
    "## Định danh",
    "## Cấu hình hệ thống",
    "## Lệnh và cấu hình chạy",
    "## Kết quả",
    "## Artifact và lineage",
    "## Việc tiếp theo",
)
ERROR_HEADINGS = (
    "# Error",
    "## Định danh",
    "## Triệu chứng",
    "## Cấu hình và đầu vào",
    "## Nguyên nhân",
    "## Khắc phục hoặc rollback",
    "## Regression và trạng thái",
)


def dedent(text: str) -> str:
    """Dedent templates even when interpolated multiline blocks start at column zero."""
    normalized = _stdlib_dedent(text)
    first_nonempty = next((line for line in normalized.splitlines() if line.strip()), "")
    if not first_nonempty.startswith("        "):
        return normalized
    return "\n".join(
        line[8:] if line.startswith("        ") else line for line in normalized.splitlines()
    )


class MigrationError(RuntimeError):
    """Báo lỗi preflight, backup hoặc validation có thể hành động được."""


@dataclass(frozen=True, slots=True)
class UpgradeProfile:
    """Giữ nội dung nâng cấp riêng của một project."""

    pitch: str
    feasibility_focus: tuple[str, ...]
    minimum_data: tuple[str, ...]
    data_failures: tuple[str, ...]
    low_resource: tuple[str, ...]
    service_roles: tuple[str, ...]
    research_questions: tuple[str, ...]
    critical_tips: tuple[str, ...]
    inference_task: str


@dataclass(frozen=True, slots=True)
class FileChange:
    """Mô tả một phép ghi hoặc xóa trong repository đích."""

    relative: str
    operation: str
    content: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class RepositoryPlan:
    """Chứa trạng thái đích đầy đủ cho một repository."""

    spec: CourseProject
    profile: UpgradeProfile
    repo: Path
    checkpoint_relative: str
    desired: dict[str, str]
    deletes: tuple[str, ...]


PROFILES = {
    "CS406": UpgradeProfile(
        pitch=(
            "Giúp người quản lý kho ảnh tìm bản trùng hoặc ảnh tương tự có nguồn rõ ràng, "
            "trước khi đầu tư vào mô hình thị giác nặng."
        ),
        feasibility_focus=(
            "Chọn một bộ sưu tập Việt Nam có quyền tái sử dụng và attribution theo từng asset.",
            "Định nghĩa query taxonomy: trùng biến đổi, cùng đối tượng, cùng địa điểm và negative khó.",
            "Đo chi phí tạo qrels, kiểm tra người dùng pilot và giới hạn corpus trước khi chọn encoder.",
        ),
        minimum_data=(
            "Pilot đề xuất 500–2.000 ảnh sau license audit; đây là ngân sách thiết kế, không phải dataset đã có.",
            "Mỗi asset có URI nguồn, license, attribution, checksum, kích thước, EXIF và duplicate-group.",
            "Qrels có hướng dẫn gán nhãn, double-check mẫu bất đồng và split theo nguồn/địa điểm/duplicate-group.",
        ),
        data_failures=(
            "Ảnh hỏng, EXIF orientation sai, decompression bomb hoặc preprocessing khác giữa query/corpus.",
            "Near-duplicate lọt qua split làm frozen embedding có gain giả.",
            "API truy cập được nhưng quyền tái sử dụng hoặc attribution không đủ.",
        ),
        low_resource=(
            "CPU: pHash → histogram màu → HOG + exact search; khóa evaluator trước candidate.",
            "RTX 2050 4 GB: frozen lightweight embedding, batch 1–4 và resolution cố định sau VRAM probe.",
            "Ưu tiên crop/rotation/color/compression stress test, normalization và rerank top-k trước fine-tuning.",
        ),
        service_roles=(
            "Wikimedia: metadata, license và snapshot manifest; không crawl mặc định.",
            "Hugging Face Hub/Inference: chỉ probe encoder đã pin; ảnh public mới được gửi ra ngoài.",
            "Kaggle: chỉ dùng version bất biến khi là nguồn được phép; không thay nguồn benchmark chính thức.",
        ),
        research_questions=(
            "Gain còn lại bao nhiêu sau khi loại near-duplicate và đổi nguồn?",
            "Sai số đến từ representation hay decode/normalization/query taxonomy?",
            "Provenance và cảnh báo có giảm thời gian rà soát thủ công trong pilot không?",
        ),
        critical_tips=(
            "Không dùng mAP cao trên query trùng làm bằng chứng semantic retrieval.",
            "Không fit whitening/threshold trên test và không đổi resolution giữa baseline/candidate.",
            "Luôn lưu thumbnail kiểm tra lỗi nhưng không đưa ảnh bị hạn chế vào Git.",
        ),
        inference_task="image-feature-extraction",
    ),
    "DS111": UpgradeProfile(
        pitch=(
            "Biến run retrieval thành quyết định có căn cứ: dữ liệu có sạch, gain có ổn định và kết luận có "
            "đổi khi phân tích đúng group hay không."
        ),
        feasibility_focus=(
            "Chốt artifact contract dùng chung với CS406, DS201 và DS314 trước khi viết dashboard.",
            "Bảo đảm có ít nhất hai run paired trên cùng query set; không so mean của hai tập khác nhau.",
            "Chọn độ rộng CI/effect size có ý nghĩa quyết định thay vì chạy kiểm định theo thói quen.",
        ),
        minimum_data=(
            "Run table version hóa: query_id, item_id, rank, score, latency, run/config revision.",
            "Qrels và query metadata có source/entity/event/time group, slice và exclusion reason.",
            "Data dictionary, unique-key policy, missing policy và manifest hash cho từng run.",
        ),
        data_failures=(
            "Bootstrap theo row thay vì query/group làm CI quá hẹp.",
            "Hai run không paired hoặc denominator của slice thay đổi nhưng vẫn so delta.",
            "Dashboard làm mất invalid rows, missingness hoặc query không có qrels.",
        ),
        low_resource=(
            "GPU không nằm trên đường găng; CPU + Pandas cho fixture và bảng nhỏ.",
            "DuckDB/Parquet projection, filter pushdown và chunked scan khi dữ liệu vượt RAM.",
            "Tối ưu schema, partition, sample audit và vectorized aggregation trước scale-out.",
        ),
        service_roles=(
            "Artifact store là nguồn chính; chỉ ingest run đã có manifest/hash và quyền truy cập.",
            "Hugging Face Inference không cần cho baseline; adapter chỉ là probe ngoài critical path.",
            "Kaggle/open-data chỉ cung cấp input version hóa, không được thay data contract nội bộ.",
        ),
        research_questions=(
            "Kết luận có đổi dấu khi resample theo source/group không?",
            "Slice nào tạo gain và slice nào bị mất coverage?",
            "Drift dữ liệu hay drift evaluator giải thích thay đổi metric tốt hơn?",
        ),
        critical_tips=(
            "Luôn báo sample size, denominator, invalid rate và effect size cạnh metric.",
            "Không impute rank/score thiếu trước khi xác định nguyên nhân schema.",
            "Không gọi synthetic fixture là bằng chứng độ tin cậy của run thật.",
        ),
        inference_task="artifact-schema-probe",
    ),
    "DS201": UpgradeProfile(
        pitch=(
            "Đo xem biểu diễn thời gian giúp truy xuất đúng bằng chứng lịch sử hay chỉ ghi nhớ pattern, "
            "trên protocol tiếng Việt có BCE/CE và source/time-held-out rõ ràng."
        ),
        feasibility_focus=(
            "Ưu tiên xác minh release và quyền dùng corpus; paper citation không đồng nghĩa dataset sẵn có.",
            "Chốt nhiệm vụ retrieval, qrels và temporal interval trước frozen encoder hoặc LoRA.",
            "Tách published-equation, operational-assumption và proposed-method trong mọi run.",
        ),
        minimum_data=(
            "Pilot vài trăm đến khoảng một nghìn chunk được phép dùng; quy mô cuối phụ thuộc chi phí qrels.",
            "Chunk có source URI, event/entity group, HistoricalYear/interval, uncertainty, license và checksum.",
            "Query/qrels gồm exact year, range, before/after, BCE/CE boundary và hard negative cùng nội dung khác thời.",
        ),
        data_failures=(
            "Dùng year 0 mơ hồ, đổi calendar convention hoặc parser giữa train/test.",
            "Source/event template leak làm temporal representation có gain giả.",
            "Tokenizer fragmentation hoặc invalid output bị quy nhầm thành lỗi representation.",
        ),
        low_resource=(
            "CPU: lexical → Gregorian scalar → Fourier → published Ticktack equation.",
            "RTX 2050 4 GB: frozen probe trước; pin sequence length, batch 1, precision và peak VRAM.",
            "Chỉ mở sub-1B LoRA/quantization sau held-out gain; 3B+ hoặc multi-seed lớn chuyển lab/cloud.",
        ),
        service_roles=(
            "Official sources: snapshot tài liệu và metadata; không suy release từ paper.",
            "Hugging Face Hub: pin tokenizer/model commit; Inference chỉ exploratory nếu runtime không pin được.",
            "Kaggle chỉ dùng version cụ thể có terms; không dùng mirror để hợp thức hóa provenance thiếu.",
        ),
        research_questions=(
            "Representation có giảm alias mà vẫn giữ monotonicity và IR metric không?",
            "Gain còn lại trên source/time-held-out và BCE/CE boundary slice bao nhiêu?",
            "Parser/tokenizer hay temporal objective đóng góp nhiều hơn?",
        ),
        critical_tips=(
            "Không gọi fixture là TempLS reproduction khi code/data gốc chưa xác minh.",
            "Luôn có no-time, scalar và Fourier control trước proposed hybrid.",
            "Không dùng endpoint floating để tạo claim benchmark tái lập.",
        ),
        inference_task="feature-extraction",
    ),
    "DS314": UpgradeProfile(
        pitch=(
            "Trả đúng đoạn tài liệu tiếng Việt có citation và biết từ chối, thay vì dùng LLM che một retriever yếu "
            "hoặc vượt phạm vi truy cập."
        ),
        feasibility_focus=(
            "Chọn một corpus công khai cụ thể; data.gov.vn cấp cổng khám phá, không phải dataset đã chốt.",
            "Thiết kế query intent, qrels, chunk policy và active-corpus version cùng lúc.",
            "Citation integrity và unauthorized-hit rate là hard gate trước generation/agent.",
        ),
        minimum_data=(
            "Snapshot tài liệu một domain hẹp như giáo dục hoặc dịch vụ công, có license/terms và checksum.",
            "Chunk giữ document/chunk ID, heading, source URI, version, access scope và parent relation.",
            "Query/qrels có unanswerable, source-held-out, template duplicate và hard negative cùng thuật ngữ.",
        ),
        data_failures=(
            "Chunk cắt mất điều kiện/ngoại lệ hoặc citation trỏ sang corpus version khác.",
            "Template/source leak làm dense/rerank gain biến mất khi hold-out.",
            "Gửi tài liệu internal/restricted tới dịch vụ ngoài hoặc filter access sau retrieval.",
        ),
        low_resource=(
            "CPU: BM25 → word/character TF-IDF → RRF → deterministic evidence/abstention.",
            "RTX 2050 4 GB: precompute frozen small embeddings theo batch đo được.",
            "Chỉ rerank top-k hẹp; generation/agent mở sau retrieval, citation và latency gate.",
        ),
        service_roles=(
            "Official-document adapter tạo snapshot có terms/version; không crawl toàn cổng.",
            "Hugging Face Hub/Inference chỉ cho public text và model revision pin được.",
            "Kaggle chỉ là nguồn version hóa phụ; access filtering/citation validation luôn chạy local.",
        ),
        research_questions=(
            "Hybrid retrieval có vượt BM25/RRF trên source-held-out ở cùng budget không?",
            "Chunk policy nào cân bằng recall và citation completeness?",
            "Abstention tăng an toàn bao nhiêu với mức coverage chấp nhận được?",
        ),
        critical_tips=(
            "Không index trước khi gắn access scope và corpus version.",
            "Không đánh giá answer fluency thay retrieval/citation metrics.",
            "Không gửi nội dung không-public ra endpoint dù token đã được bảo mật.",
        ),
        inference_task="feature-extraction",
    ),
}


def bullets(items: tuple[str, ...]) -> str:
    """Render một danh sách Markdown ngắn."""
    return "\n".join(f"- {item}" for item in items)


def numbered(items: tuple[str, ...]) -> str:
    """Render một danh sách Markdown có thứ tự."""
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


def source_rows(spec: CourseProject) -> str:
    """Render các nguồn đã có với vai trò quyết định, không nâng claim."""
    return "\n".join(
        f"| [{source.label}]({source.url}) | {source.use} | primary/official pointer |"
        for source in spec.sources
    )


def candidate_rows(spec: CourseProject) -> str:
    """Render dataset candidates ở trạng thái hold."""
    return "\n".join(
        f"| [{item.name}]({item.url}) | {item.role} | {item.gate} | `hold` |"
        for item in spec.data_candidates
    )


def render_rule(spec: CourseProject) -> str:
    """Tạo quy tắc duy nhất cho thư mục General."""
    return dedent(f"""\
        # Quy tắc `Documents/General`

        ## Phạm vi

        `General` chỉ chứa `RULE.md` và log bất biến. Tri thức, quyết định, hướng dẫn, lịch môn và báo cáo
        không được đặt ở đây. Không sửa log cũ để làm đẹp kết quả; tạo version mới.

        ## Quy tắc tên file

        - `{spec.repo_name}_Checkpoints_DD-MM-YY_V<N>.md`
        - `{spec.repo_name}_Error_DD-MM-YY_V<N>.md`

        Windows không cho phép `/` trong tên file nên ngày `DD/MM/YY` được biểu diễn bằng `DD-MM-YY`.
        Với cùng ngày và cùng loại, version bắt đầu `V1` và tăng liên tục `V2`, `V3`.

        ## Checkpoint log

        Ghi mục tiêu, lệnh thật, exit code, cấu hình hệ thống, config/data/model/code revision, evidence class,
        artifact/hash, kết quả và next gate. Lệnh chưa chạy phải ghi `not-run`; scaffold/smoke không phải benchmark.

        ## Error log

        Chỉ ghi lỗi thật hoặc kết quả âm tính. Phân biệt `confirmed` và `hypothesis`; lưu expected/actual,
        evidence, root cause, fix/rollback và regression status. Risk chung thuộc `References`.

        ## Version và bất biến

        Không ghi đè log. Mọi metric phải nêu denominator/split/seed/CI khi phù hợp và gắn config hash,
        data-manifest hash, code revision, resource measurement cùng evidence class.

        ## Bảo mật và mức bằng chứng

        Không log token, secret, dữ liệu hạn chế hoặc nội dung request nhạy cảm. Chỉ dùng `measured` khi run
        tái lập được; `paper_reported`, `proposed`, `scaffold` và `smoke_test` phải ghi đúng tên.
        """)


def render_project_context(spec: CourseProject, profile: UpgradeProfile) -> str:
    """Tạo bối cảnh môn học và bài toán để trình bày nhanh."""
    schedule = "\n".join(f"- {line}" for line in spec.schedule)
    return dedent(f"""\
        # Project Context — {spec.course_code}

        ## Bối cảnh

        - Môn: `{spec.course_code}` — {spec.course_name}
        - Project: **{spec.title}**
        - Mức hiện tại: `planning/scaffold`; chưa có benchmark hoặc dataset/model được duyệt.

        {schedule}

        ## Pitch 30 giây

        {profile.pitch}

        ## Người dùng và quyết định

        - Người dùng: {spec.primary_user}
        - Quyết định: {spec.supported_decision}
        - Bài toán: {spec.problem}
        - Giá trị Việt Nam: {spec.vietnam_value}
        - Chuyển giao: {spec.global_transfer}

        ## Câu hỏi phải chốt với giảng viên và teammate

        {numbered(profile.research_questions)}

        ## Phạm vi giao nhận

        Baseline CPU, data contract, qrels/evaluator, leakage audit, error analysis và demo đọc artifact là
        đường găng. Mọi candidate nặng phải vượt data/resource gate và có stop condition trước khi triển khai.
        """)


def render_feasibility(spec: CourseProject, profile: UpgradeProfile) -> str:
    """Tạo hồ sơ khả thi, dữ liệu và ngân sách RTX 2050 4 GB."""
    return dedent(f"""\
        # Khả thi, dữ liệu và ngân sách 4 GB — {spec.course_code}

        ## Kết luận hiện tại

        Bài toán **khả thi có điều kiện**. Điểm nghẽn là data/qrels và protocol, không phải thiếu mô hình lớn.
        Tất cả registry entries còn `hold`; không tải asset cho đến khi đủ provenance gate.

        ## Feasibility gate

        {numbered(profile.feasibility_focus)}

        Go khi có user decision rõ, corpus hợp pháp, qrels tối thiểu, split chống leakage, baseline/evaluator chạy
        CPU và lịch nhóm đủ cho error analysis. No-go hoặc thu hẹp khi thiếu quyền dữ liệu, không tạo được qrels,
        gain chỉ do leakage hoặc chi phí vượt RTX 2050 4 GB/CPU budget.

        ## Minimum viable data

        {bullets(profile.minimum_data)}

        ## Data dossier bắt buộc

        | Nhóm | Trường phải có |
        |---|---|
        | Nguồn | official URL, accessed date, owner, terms/license, revision/snapshot, checksum |
        | Nội dung | schema, size, language/domain, missingness, duplicate, bias, sensitive fields |
        | Nhãn | annotation guide, qrels format, disagreement, invalid/unknown policy |
        | Split | source/entity/event/time group, near-duplicate audit, train-only fit |
        | Chi phí | download/storage, labeling hours, CPU/GPU time, cache/retention |
        | Quyết định | role, reviewer, status, stop condition, fallback dataset |

        ## Lỗi dữ liệu chí mạng

        {bullets(profile.data_failures)}

        ## Lộ trình phần cứng

        {numbered(profile.low_resource)}

        Mỗi probe ghi model/input size, batch/sequence/corpus size, precision, p50/p95 latency, peak RAM/VRAM,
        disk và wall time. Không suy khả năng chạy chỉ từ số tham số hoặc quảng cáo model card.
        """)


def render_research(spec: CourseProject, profile: UpgradeProfile) -> str:
    """Tạo bản khảo sát nghiên cứu và data theo decision evidence."""
    return dedent(f"""\
        # Khảo sát nghiên cứu và dữ liệu — {spec.course_code}

        Cập nhật thiết kế: 25/08/2026. Số liệu paper luôn là `paper_reported`; URL paper không chứng minh data,
        code, checkpoint hoặc license đang sẵn dùng. Winning project chỉ được gọi là winning sau khi xác minh
        task, năm, bảng xếp hạng và protocol chính thức.

        ## Khoảng trống thực dụng

        {spec.problem} Giá trị cần chứng minh bằng held-out evaluation và pilot, không suy từ độ mới kiến trúc.

        ## Câu hỏi khảo sát

        {numbered(profile.research_questions)}

        ## Nguồn đã xác minh làm evidence pointer

        | Nguồn | Dùng để quyết định | Loại |
        |---|---|---|
        {source_rows(spec)}

        ## Dataset candidates

        | Candidate | Vai trò | Gate còn thiếu | Trạng thái |
        |---|---|---|---|
        {candidate_rows(spec)}

        ## Ma trận đọc paper/project thực tế

        Với mỗi paper hoặc hệ thống: ghi task/user, data release/license, split/leakage, baseline, candidate,
        evaluator, hardware, latency, ablation, failure cases, code/checkpoint revision và phần chuyển giao được.
        So like-for-like; không lấy leaderboard khác domain làm lý do chọn model.

        ## Stop rule

        Dừng mở rộng research khi đã đủ một problem contract, một dataset primary + fallback, một evaluator,
        ba baseline hợp lý, một candidate có giả thuyết lỗi cụ thể và một protocol tái lập trong ngân sách.
        """)


def render_services(spec: CourseProject, profile: UpgradeProfile) -> str:
    """Tạo hướng dẫn dịch vụ ngoài và inference fail-closed."""
    package = spec.package_name
    return dedent(f"""\
        # Hướng dẫn ServiceAdapters — {spec.course_code}

        ## Tại sao cần adapter

        Core không gọi mạng. `src/{package}/ServiceAdapters/` gom provenance, privacy, timeout, retry,
        revision, checksum và failure contract để pipeline vẫn test được offline. Dịch vụ chỉ giảm chi phí tải/probe;
        không thay data understanding, evaluator hoặc evidence.

        ## Vai trò tối ưu cho project

        {bullets(profile.service_roles)}

        ## Luồng từ ý tưởng đến code

        1. Viết use case và failure cần giải quyết; nếu baseline chưa chỉ ra lỗi thì chưa gọi service.
        2. Điền registry: official URL, accessed date, terms, immutable revision/checksum, reviewer và role.
        3. Chọn privacy class `public|internal|restricted`; external inference chỉ nhận `public`.
        4. Tạo request/output schema, task, model/provider revision và timeout trong `inference.py`.
        5. `build_plan()` chỉ validate; `execute()` tiếp tục fail-closed cho đến khi có ADR, mock, quota và rollback.
        6. Khi được mở gate: cache ngoài Git, verify checksum/schema, tạo manifest rồi chạy pipeline offline.
        7. Log latency/resource/cost/error bằng checkpoint; kết quả âm tính vào error log.

        ## Hugging Face Hub

        Dùng full commit cho `snapshot_download`/`hf_hub_download`, giới hạn file bằng `allow_patterns`, lưu resolved
        commit/file list/SHA-256 và dùng `local_files_only=True` cho run đóng băng. Nguồn:
        [Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download).

        ## Hugging Face Inference

        Phải pin model revision và provider/runtime revision. Nếu endpoint không cho pin runtime, chỉ gắn
        `exploratory`, không dùng cho benchmark. Gửi schema tối thiểu, timeout hữu hạn, không retry lỗi 4xx,
        redact token và không gửi dữ liệu `internal/restricted`. So chi phí/latency với local CPU/GPU fallback.

        ## Kaggle và nguồn domain

        Kaggle dùng `owner/dataset/versions/n`, không dùng latest; kiểm tra terms, file list, archive path, size,
        checksum và schema. Domain adapter tôn trọng robots/rate-limit, snapshot/version và quyền từng asset.

        ## RTX 2050 4 GB

        Ưu tiên tải artifact một lần rồi inference offline, batch nhỏ và candidate top-k hẹp. Remote inference chỉ
        hợp lý khi dữ liệu public, local probe không vừa ngân sách và endpoint cho provenance/revision đủ mạnh.

        ## Tips và lỗi thường gặp

        {bullets(profile.critical_tips)}

        Luôn có mock/failure-injection cho timeout, 429, schema drift, checksum mismatch, cache miss và endpoint
        unavailable. UI chỉ đọc artifact; không đặt nút download/train/inference không kiểm soát trong Streamlit.
        """)


def render_knowledge(spec: CourseProject, profile: UpgradeProfile) -> str:
    """Tạo tài liệu học nhanh nhưng đủ chiều sâu để triển khai."""
    stages = " → ".join(f"`{stage}`" for stage in spec.stages)
    return dedent(f"""\
        ---
        title: "{spec.course_code} — {spec.title}"
        author: "JH111334"
        date: "25/08/2026"
        lang: vi-VN
        evidence: planning/scaffold
        ---

        # Tóm tắt điều hành

        {profile.pitch} Đóng góp thực dụng là contract + data protocol + error analysis trong ngân sách sinh viên;
        chưa có benchmark và không mặc định mô hình lớn tốt hơn.

        # Cách tư duy bài toán

        1. Bắt đầu từ người dùng và quyết định, không bắt đầu từ model.
        2. Chuyển quyết định thành input/output/error/abstention contract.
        3. Hiểu data generation, coverage, label noise, duplicate và split trước feature engineering.
        4. Khóa evaluator và baseline; chỉ thêm candidate để sửa một failure slice đã đo.
        5. Ghi artifact lineage và stop khi gain không qua held-out/resource gate.

        # Đầu vào, đầu ra và lỗi

        ## Đầu vào

        {bullets(spec.input_contract)}

        ## Đầu ra

        {bullets(spec.output_contract)}

        ## Ngoài phạm vi

        {bullets(spec.exclusions)}

        # Pipeline và ý nghĩa từng bước

        {stages}

        Mỗi stage nhận artifact có schema/version/hash, fail-closed và xuất artifact kế tiếp. Split có trước mọi
        fit/tuning; evaluator không đổi giữa baseline/candidate; UI không sở hữu download, training hoặc inference.

        # Data-first: kiến thức phải nắm

        {bullets(profile.minimum_data)}

        Trước coding, hãy vẽ một dòng cho mỗi record: nó sinh từ đâu, ai sở hữu, có thể sai thế nào, thuộc group nào,
        được dùng ở split nào và metric nào bị ảnh hưởng. Đây thường là nơi tạo gain bền vững hơn tăng model size.

        # Baseline và cải thiện thông minh

        ## Baseline

        {numbered(spec.baselines)}

        ## Candidate sau gate

        {numbered(spec.candidates)}

        ## Ngân sách

        {numbered(profile.low_resource)}

        # Evaluation và bằng chứng

        Metrics: {", ".join(spec.metrics)}. Báo per-query/per-slice, sample size, invalid rate, seed/95% CI khi
        stochastic, p50/p95 latency, peak RAM/VRAM, config hash, data-manifest hash và code revision.

        # ServiceAdapters

        {bullets(profile.service_roles)}

        `inference.py` mô hình hóa provider/model revision, task, timeout, privacy class và schemas nhưng execution
        mặc định fail-closed. Chỉ ADR riêng mới được mở network path; Streamlit vẫn chỉ đọc artifact.

        # Lỗi chí mạng và phòng tránh

        {bullets(profile.data_failures + profile.critical_tips)}

        # Lộ trình 12 tuần

        1. Tuần 1–2: user interview, problem contract, source/license/data dossier.
        2. Tuần 3–4: schema, fixtures, qrels, group/time split và leakage tests.
        3. Tuần 5–6: baseline CPU, evaluator parity và run artifact.
        4. Tuần 7–8: failure slices; chỉ mở candidate sửa lỗi đã xác định.
        5. Tuần 9–10: ablation, seeds/CI, latency/resource và failure injection.
        6. Tuần 11–12: Streamlit read-only, DOCX/HTML QA, limitations và reproducibility replay.

        # Dàn ý thuyết phục teammate

        Nêu user pain → quyết định → data có thể lấy hợp pháp → chi phí qrels → baseline rẻ → failure slice → candidate
        nhỏ nhất sửa được lỗi → cách đo → stop condition. Không bán kiến trúc; bán protocol và khả năng học từ data.
        """)


def render_project_html(spec: CourseProject, profile: UpgradeProfile) -> str:
    """Tạo báo cáo HTML tự chứa, tối giản và đồng bộ với context/data gate."""
    sources = "\n".join(
        f'<li><a href="{html.escape(source.url, quote=True)}">{html.escape(source.label)}</a>: '
        f"{html.escape(source.use)}</li>"
        for source in spec.sources
    )
    minimum_data = "\n".join(f"<li>{html.escape(item)}</li>" for item in profile.minimum_data)
    low_resource = "\n".join(f"<li>{html.escape(item)}</li>" for item in profile.low_resource)
    service_roles = "\n".join(f"<li>{html.escape(item)}</li>" for item in profile.service_roles)
    return dedent(f"""\
        <!doctype html>
        <html lang="vi">
        <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{html.escape(spec.course_code)} — Kế hoạch project 25/08/2026</title>
        <style>
        :root {{ font-family: Arial, sans-serif; color: #111; background: #fff; }}
        body {{ max-width: 960px; margin: 28px auto; padding: 0 24px; line-height: 1.5; }}
        h1, h2 {{ line-height: 1.25; }}
        .box {{ border: 1px solid #777; padding: 12px; margin: 14px 0; }}
        code {{ overflow-wrap: anywhere; }}
        @media print {{ body {{ margin: 0; max-width: none; }} a {{ color: #111; }} }}
        </style>
        </head>
        <body>
        <h1>{html.escape(spec.course_code)} — {html.escape(spec.title)}</h1>
        <div class="box"><strong>Cập nhật:</strong> 25/08/2026<br>
        <strong>Mức bằng chứng:</strong> planning/scaffold; chưa có benchmark hoặc dataset/model thật.<br>
        <strong>Phần cứng local:</strong> RTX 2050, ngân sách 4 GB VRAM; phải đo lại từng probe.</div>
        <h2>Pitch và quyết định</h2>
        <p>{html.escape(profile.pitch)}</p>
        <p><strong>Người dùng:</strong> {html.escape(spec.primary_user)}</p>
        <p><strong>Quyết định hỗ trợ:</strong> {html.escape(spec.supported_decision)}</p>
        <h2>Khả thi và dữ liệu tối thiểu</h2>
        <p>Điểm nghẽn là corpus/qrels/protocol. Chỉ đi tiếp khi quyền dữ liệu, split chống leakage,
        evaluator và baseline CPU đều đạt.</p><ul>{minimum_data}</ul>
        <h2>Lộ trình phần cứng</h2><ol>{low_resource}</ol>
        <h2>ServiceAdapters</h2><ul>{service_roles}</ul>
        <p>Core chạy offline; Hub/Kaggle/inference phải pin revision, privacy, timeout, schema và
        mặc định fail-closed. Streamlit chỉ đọc artifact có lineage.</p>
        <h2>Gate giao nhận</h2>
        <ol><li>Problem/user/data dossier.</li><li>Schema, qrels, group/source/entity/event/time split.</li>
        <li>CPU baseline và evaluator parity.</li><li>Error slices, candidate nhỏ nhất và ablation.</li>
        <li>Resource artifact, limitations, DOCX/HTML QA và replay.</li></ol>
        <h2>Nguồn primary/official</h2><ul>{sources}</ul>
        </body>
        </html>
        """)


def render_failures(spec: CourseProject, profile: UpgradeProfile) -> str:
    """Tách risk/tips khỏi error log vận hành."""
    return dedent(f"""\
        # Lỗi thường gặp và cách phòng tránh — {spec.course_code}

        Đây là knowledge base, không phải log sự cố đã xác minh. Lỗi thật hoặc kết quả âm tính phải ghi theo
        `Documents/General/RULE.md`.

        ## Khảo sát và tính khả thi

        - Chọn model trước user/evaluator → quay lại problem contract và feasibility gate.
        - Suy data availability từ paper → kiểm tra release, terms, revision, checksum và reviewer.
        - Tuyên bố nhu cầu Việt Nam không có pilot → gắn `hypothesis` và phỏng vấn trước claim.

        ## Data và evaluation

        {bullets(profile.data_failures)}

        - Random split khi có source/entity/event/time overlap → group split và near-duplicate audit trước fit.
        - Chỉ báo best seed/mean → thêm invalid rate, denominator, slices, seeds/CI và effect size.

        ## Model, service và ứng dụng

        {bullets(profile.critical_tips)}

        - Tải latest hoặc log token → immutable revision, secret redaction, cache ngoài Git và rollback.
        - UI tự chạy pipeline → giữ Streamlit read-only, hiển thị lineage/warning/evidence class.
        """)


def render_tools_decision(spec: CourseProject) -> str:
    """Tạo quyết định tĩnh về tool/MCP ngoài General."""
    return dedent(f"""\
        # Tools, MCP và services — {spec.course_code}

        ## Quyết định

        Project dùng cấu hình `.codex/config.toml`; không thay global MCP. `filesystem`, `memory_keeper` và
        `playwright` phục vụ file/session/UI smoke; tool ngoài critical path giữ disabled và được mở lại qua ADR.

        ## Chuỗi công cụ

        - Research: nguồn primary/official, claim gần citation.
        - Code: Python chuẩn, unit tests, validator; không cần package để kiểm tra scaffold.
        - DOCX: Markdown → Pandoc → LibreOffice PDF → PyMuPDF inspection.
        - Streamlit: chỉ đọc artifact; AppTest/Playwright sau khi dependency được duyệt.
        - Service: Hub/Kaggle/domain/inference chỉ qua `ServiceAdapters` và registry gate.

        Cấu hình/runtime đã chạy thuộc checkpoint log; lỗi tool thật thuộc error log. File này chỉ giữ rationale.
        """)


def render_deferred() -> str:
    """Tạo quyết định hoãn một nhánh ngoài scope trong đúng file cho phép."""
    return dedent("""\
        # Hướng multimodal tạm hoãn

        Nhánh này không thuộc scope học kỳ và không đi vào source, requirements, metric, báo cáo chính hoặc UI.
        Chỉ mở lại bằng ADR sau khi baseline chính có corpus hợp pháp, qrels/evaluator cố định, held-out result,
        resource budget và failure slice chứng minh modality mới là cần thiết.
        """)


def render_documents_readme() -> str:
    """Tạo bản đồ tài liệu sau migration."""
    return dedent("""\
        # Documents

        Chỉ `README.md` nằm ở cấp gốc.

        - `Decision-Making/`: problem, architecture/gates, feasibility/data/4 GB, project context, `ERROR-EXPERIENCE.md`, tool/service và quyết định tạm hoãn.
        - `References/`: research/dataset, knowledge, service guide, DOCX/HTML và tips phòng tránh.
        - `General/`: chỉ `RULE.md`, checkpoint logs và error logs đúng quy tắc tên.

        `paper_reported`/`candidate` không phải kết quả của nhóm; `scaffold`/`smoke_test` không phải benchmark;
        chỉ dùng `measured` khi đủ split, config/data/code hash, seed/CI và resource artifact.
        """)


def render_agents(spec: CourseProject) -> str:
    """Tạo quy ước repository đồng bộ với log và inference gate."""
    return dedent(f"""\
        # Quy ước project {spec.course_code}

        - Bắt đầu bằng baseline CPU tái lập; không gọi scaffold/smoke/synthetic là kết quả nghiên cứu.
        - Core Python không gọi mạng; Hub/Kaggle/domain/inference chỉ qua `ServiceAdapters` và manifest gate.
        - Không tải data/model nếu thiếu official URL, accessed date, terms, immutable revision/checksum, reviewer và role.
        - Split theo group/source/entity/event/time, audit near-duplicate và fit preprocessing trên train.
        - Kết quả gắn config/data/code hash, seed/CI, invalid rate, resource và evidence class.
        - RTX 2050 4 GB là trần local; đo peak trước candidate, không suy khả năng chạy từ model size.
        - Hàm nhỏ, type hints, docstring ngắn, fail-closed; đổi schema/split/objective/adapter phải có test.
        - Không dùng emoji, sticker, banner ký hiệu hoặc thông báo ăn mừng trong mã, test, log hay CLI; chỉ giữ ký hiệu có nghĩa kỹ thuật hoặc dữ liệu protocol đã ghi rõ.
        - `Documents/General` chỉ có `RULE.md` và log đúng tên; không sửa log cũ hoặc ghi risk như sự cố thật.
        - Streamlit chỉ đọc artifact và hiện lineage/warning; không download/train/inference từ UI.
        - Viết tiếng Việt có dấu; giữ data/model/cache/artifact lớn và secret ngoài Git.
        - Trước commit chạy `$env:PYTHONPATH='src'; py -3.12 -B -m unittest discover -s tests -v` và
          `py -3.12 -B scripts/validate_project.py`.
        """)


def render_skill(spec: CourseProject) -> str:
    """Tạo local skill phản ánh workflow mới."""
    return dedent(f"""\
        ---
        name: course-project-delivery
        description: Hoạch định và triển khai {spec.course_code} theo data-first, baseline CPU, log bất biến, service/inference gate và bằng chứng tái lập.
        ---

        # Course Project Delivery

        ## Workflow bắt buộc

        1. Đọc `AGENTS.md`, `references/gates.md`, `Documents/General/RULE.md` và problem contract.
        2. Xác định user decision, stage, schemas, evidence class, resource budget và stop condition.
        3. Hoàn thành data dossier, qrels/split/leakage và evaluator trước model/service nặng.
        4. Dùng fixture + CPU baseline; RTX 2050 4 GB chỉ mở sau resource probe.
        5. Chặn download/inference nếu registry, privacy, revision, timeout hoặc schema gate thiếu.
        6. Tách `paper_reported`, `proposed`, `scaffold`, `smoke_test` và `measured`.
        7. `ServiceAdapters/inference.py` giữ execution fail-closed đến khi ADR + mocks + rollback đạt.
        8. UI chỉ đọc artifact, hiện lineage/warning; không train/download/inference từ giao diện.
        9. Ghi lệnh/config/result vào checkpoint; chỉ ghi lỗi thật/kết quả âm vào error log version mới.
        10. Chạy unit tests và validator; dừng khi acceptance gate đạt, không mở rộng scope.

        ## Tổ chức tài liệu

        Quyết định và bài học lỗi đã chắt lọc vào `Decision-Making` (`ERROR-EXPERIENCE.md`); tri thức/nguồn/hướng dẫn
        vào `References`; `General` chỉ chứa `RULE.md` và log đúng grammar. Nội dung project dùng tiếng Việt có dấu,
        compact và nguồn gần mệnh đề.
        """)


def render_skill_gates() -> str:
    """Tạo gate reference dùng chung cho local skill."""
    return dedent("""\
        # Gates

        | Gate | Điều kiện tối thiểu |
        |---|---|
        | Problem | User, decision, input/output/error, metric, budget, stop condition |
        | Feasibility | Pilot, team/time, qrels cost, minimum data, fallback và go/no-go |
        | Data | Official URL, date, terms, revision/checksum, reviewer, role và data dossier |
        | Split | Group/source/entity/event/time + near-duplicate audit + train-only fit |
        | Baseline | CPU, fixture, evaluator parity, artifact hash và failure slices |
        | Resource | RTX 2050 4 GB probe: input/batch/precision/latency/RAM/VRAM/disk |
        | Service | Privacy, task, schemas, immutable model/provider revisions, timeout, mock, rollback |
        | Measured | Frozen test, seed/CI/slices/invalid rate, config/data/code/resource |
        | Application | Read-only artifact, provenance, warning/abstention và failure tests |
        | Log | RULE-compliant checkpoint/error filename, fields, evidence class và no-secret |
        """)


def render_inference_module(spec: CourseProject, profile: UpgradeProfile) -> str:
    """Tạo adapter inference thuần Python, không import SDK/network."""
    return dedent(f'''\
        """Validate external inference plans while execution stays disabled."""

        from __future__ import annotations

        from dataclasses import dataclass

        from ..errors import AdapterDisabledError, ProjectContractError

        EXPECTED_TASK = "{profile.inference_task}"
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
        ''')


def render_inference_test(spec: CourseProject, profile: UpgradeProfile) -> str:
    """Tạo unit tests tối thiểu cho inference contract."""
    package = spec.package_name
    return dedent(f'''\
        """Validate immutable, private-safe and fail-closed inference plans."""

        import unittest

        from {package}.ServiceAdapters.inference import InferenceAdapter, InferenceSpec
        from {package}.errors import AdapterDisabledError, ProjectContractError


        def valid_spec(**changes: object) -> InferenceSpec:
            """Build one public synthetic inference specification."""
            values: dict[str, object] = {{
                "provider": "synthetic-provider",
                "provider_revision": "provider-2026-08-25",
                "model_id": "fixture/model",
                "model_revision": "a" * 40,
                "task": "{profile.inference_task}",
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
        ''')


VALIDATOR_TEMPLATE = r'''"""Validate upgraded course repository structure and offline contracts."""

from __future__ import annotations

import json
import re
import tomllib
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_NAME = "__REPO_NAME__"
PACKAGE = "__PACKAGE__"
FORBIDDEN_SCOPE_TERM = "multi" + "modal"
DEFERRED_SCOPE_FILE = ROOT / "Documents" / "Decision-Making" / (
    "04-DEFERRED-" + FORBIDDEN_SCOPE_TERM.upper() + ".md"
)
IGNORED_PARTS = {".git", ".migration-backups", ".venv", "__pycache__"}
LOG_PATTERN = re.compile(
    rf"^{re.escape(REPO_NAME)}_(Checkpoints|Error)_(\d{{2}}-\d{{2}}-\d{{2}})_V([1-9]\d*)\.md$"
)
REQUIRED = (
    "README.md", "LICENSE", "AGENTS.md", "requirements.txt", "pyproject.toml",
    ".env.example", ".codex/config.toml", "Workflows/flowinfo.md",
    "configs/scaffold.json", "registries/datasets.json", "registries/models.json",
    "schemas/input.schema.json", "schemas/output.schema.json", "Documents/README.md",
    "Documents/Decision-Making/00-PROBLEM-CONTRACT.md",
    "Documents/Decision-Making/01-ARCHITECTURE-AND-GATES.md",
    "Documents/Decision-Making/02-DATA-MODEL-EVALUATION.md",
    "Documents/Decision-Making/ERROR-EXPERIENCE.md",
    "Documents/Decision-Making/03-TOOLS-AND-SERVICES.md",
    "Documents/Decision-Making/04-DEFERRED-" + FORBIDDEN_SCOPE_TERM.upper() + ".md",
    "Documents/Decision-Making/05-PROJECT-CONTEXT.md",
    "Documents/Decision-Making/06-FEASIBILITY-DATA-4GB.md",
    "Documents/References/RESEARCH-AND-DATASETS.md",
    "Documents/References/EXTERNAL-SERVICES-GUIDELINE.md",
    "Documents/References/PROJECT-KNOWLEDGE.md",
    "Documents/References/PROJECT-KNOWLEDGE.docx",
    "Documents/References/PROJECT-PLAN.html",
    "Documents/References/COMMON-FAILURES-AND-PREVENTION.md",
    "Documents/General/RULE.md", f"src/{PACKAGE}/contracts.py",
    f"src/{PACKAGE}/ServiceAdapters/base.py",
    f"src/{PACKAGE}/ServiceAdapters/huggingface.py",
    f"src/{PACKAGE}/ServiceAdapters/kaggle.py",
    f"src/{PACKAGE}/ServiceAdapters/inference.py",
    "streamlit_app/app.py", "tests/test_contracts.py", "tests/test_scaffold.py",
    "tests/test_inference_adapter.py", ".agents/skills/course-project-delivery/SKILL.md",
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
            (0x00C3, 0x0192), (0x00C3, 0x00A1), (0x00C4, 0x2018),
            (0x00E1, 0x00BB), (0x00E2, 0x20AC, 0x201C), (0x00E2, 0x2020),
        )
    )
    return any(marker in text for marker in markers)


def iter_project_files() -> list[Path]:
    """Return files outside generated caches, backups and Git internals."""
    return [
        path for path in ROOT.rglob("*")
        if path.is_file() and not any(part in IGNORED_PARTS for part in path.parts)
    ]


def validate_general(failures: list[str]) -> int:
    """Validate log grammar, dates, versions and required headings."""
    general = ROOT / "Documents" / "General"
    if not general.is_dir():
        failures.append("missing Documents/General directory")
        return 0
    groups: dict[tuple[str, str], list[int]] = {}
    checkpoints = 0
    for path in sorted(general.iterdir()):
        if not path.is_file() or path.name == "RULE.md":
            continue
        match = LOG_PATTERN.fullmatch(path.name)
        if match is None:
            failures.append(f"unexpected General file: {path.name}")
            continue
        kind, date_text, version_text = match.groups()
        try:
            datetime.strptime(date_text, "%d-%m-%y")
        except ValueError:
            failures.append(f"invalid log date: {path.name}")
        groups.setdefault((kind, date_text), []).append(int(version_text))
        text = path.read_text(encoding="utf-8")
        headings = CHECKPOINT_HEADINGS if kind == "Checkpoints" else ERROR_HEADINGS
        for heading in headings:
            if heading not in text:
                failures.append(f"{path.name} missing heading: {heading}")
        checkpoints += kind == "Checkpoints"
    for key, versions in groups.items():
        ordered = sorted(versions)
        if ordered != list(range(1, max(ordered) + 1)):
            failures.append(f"non-contiguous log versions for {key}: {ordered}")
    if checkpoints < 1:
        failures.append("at least one checkpoint log is required")
    return checkpoints


def validate_documents(failures: list[str]) -> None:
    """Require decision-critical sections in upgraded documents."""
    markers = {
        "Documents/General/RULE.md": ("Quy tắc tên file", "Error log", "Không ghi đè log"),
        "Documents/Decision-Making/05-PROJECT-CONTEXT.md": ("Pitch 30 giây", "Người dùng và quyết định"),
        "Documents/Decision-Making/06-FEASIBILITY-DATA-4GB.md": ("Data dossier", "RTX 2050 4 GB"),
        "Documents/References/RESEARCH-AND-DATASETS.md": ("Dataset candidates", "Stop rule"),
        "Documents/References/EXTERNAL-SERVICES-GUIDELINE.md": (
            "Hugging Face Inference", "privacy", "fail-closed",
        ),
        "Documents/References/PROJECT-KNOWLEDGE.md": ("Data-first", "Lỗi chí mạng", "ServiceAdapters"),
    }
    for relative, required_markers in markers.items():
        path = ROOT / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for marker in required_markers:
            if marker not in text:
                failures.append(f"{relative} missing marker: {marker}")


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
                    failures.append(f"approved {name} entry missing {field}")


def main() -> int:
    """Return non-zero when any upgraded acceptance condition fails."""
    failures: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            failures.append(f"missing required file: {relative}")
    root_files = sorted(path.name for path in (ROOT / "Documents").iterdir() if path.is_file())
    if root_files != ["README.md"]:
        failures.append(f"Documents root must only contain README.md, found {root_files}")
    checkpoints = validate_general(failures)
    validate_documents(failures)
    for path in iter_project_files():
        if path.suffix.casefold() not in {".py", ".md", ".html", ".toml", ".json", ".yaml", ".yml", ".txt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError as exc:
            failures.append(f"non-UTF-8 text {path.relative_to(ROOT)}: {exc}")
            continue
        if possible_mojibake(text):
            failures.append(f"possible mojibake in {path.relative_to(ROOT)}")
        if path != DEFERRED_SCOPE_FILE and FORBIDDEN_SCOPE_TERM in text.casefold():
            failures.append(f"deferred scope leaked into {path.relative_to(ROOT)}")
        if path.suffix == ".py":
            try:
                compile(text, str(path), "exec")
            except SyntaxError as exc:
                failures.append(f"Python syntax error {path.relative_to(ROOT)}: {exc}")
    for path in ROOT.rglob("*.json"):
        if any(part in IGNORED_PARTS for part in path.parts):
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            failures.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")
    for path in ROOT.rglob("*.toml"):
        if any(part in IGNORED_PARTS for part in path.parts):
            continue
        try:
            tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
            failures.append(f"invalid TOML {path.relative_to(ROOT)}: {exc}")
    validate_registries(failures)
    html_path = ROOT / "Documents" / "References" / "PROJECT-PLAN.html"
    if html_path.is_file():
        html = html_path.read_text(encoding="utf-8").casefold()
        for marker in ("<!doctype html", "<html", "<meta charset=", "<title>", "</html>"):
            if marker not in html:
                failures.append(f"HTML missing marker: {marker}")
    docx_path = ROOT / "Documents" / "References" / "PROJECT-KNOWLEDGE.docx"
    if docx_path.is_file() and (docx_path.stat().st_size < 10_000 or not zipfile.is_zipfile(docx_path)):
        failures.append("PROJECT-KNOWLEDGE.docx is not a valid package")
    if failures:
        print("VALIDATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"VALIDATION PASSED: {len(REQUIRED)} required files; {checkpoints} checkpoint logs; offline scaffold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def render_validator(spec: CourseProject) -> str:
    """Tạo validator không dependency cho trạng thái sau migration."""
    return (
        VALIDATOR_TEMPLATE.replace("__REPO_NAME__", spec.repo_name)
        .replace("__PACKAGE__", spec.package_name)
        .strip()
        + "\n"
    )


def sha256_bytes(payload: bytes) -> str:
    """Tính SHA-256 cho backup manifest và so sánh idempotent."""
    return hashlib.sha256(payload).hexdigest()


def read_text(path: Path) -> str:
    """Đọc UTF-8 và báo đường dẫn khi encoding sai."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise MigrationError(f"không đọc được UTF-8: {path}: {exc}") from exc


def safe_target(repo: Path, relative: str) -> Path:
    """Resolve đường dẫn và chặn thoát khỏi repository."""
    target = (repo / relative).resolve()
    try:
        target.relative_to(repo.resolve())
    except ValueError as exc:
        raise MigrationError(f"đường dẫn thoát repository: {relative}") from exc
    return target


def git_value(repo: Path, *args: str) -> str:
    """Đọc metadata Git nhỏ; trả unknown nếu Git không sẵn sàng."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "unknown"


def checkpoint_pattern(spec: CourseProject) -> re.Pattern[str]:
    """Tạo regex log dành riêng cho một repository."""
    return re.compile(
        rf"^{re.escape(spec.repo_name)}_(Checkpoints|Error)_(\d{{2}}-\d{{2}}-\d{{2}})_V([1-9]\d*)\.md$"
    )


def select_checkpoint(repo: Path, spec: CourseProject) -> str:
    """Tái dùng checkpoint migration hoặc cấp version kế tiếp theo ngày."""
    general = repo / "Documents" / "General"
    marker = f"migration_id: `{MIGRATION_ID}`"
    existing: list[Path] = []
    versions: list[int] = []
    date_text = datetime.now().astimezone().strftime("%d-%m-%y")
    pattern = checkpoint_pattern(spec)
    if general.is_dir():
        for path in general.glob("*.md"):
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            if match.group(1) == "Checkpoints" and marker in read_text(path):
                existing.append(path)
            if match.group(1) == "Checkpoints" and match.group(2) == date_text:
                versions.append(int(match.group(3)))
    if len(existing) > 1:
        raise MigrationError(f"nhiều checkpoint cùng migration trong {repo}")
    if existing:
        return existing[0].relative_to(repo).as_posix()
    version = max(versions, default=0) + 1
    name = f"{spec.repo_name}_Checkpoints_{date_text}_V{version}.md"
    return f"Documents/General/{name}"


def legacy_hashes(repo: Path) -> str:
    """Render hash của tài liệu legacy thay vì chép nội dung vào log."""
    rows: list[str] = []
    for relative in LEGACY_PATHS:
        path = repo / relative
        digest = sha256_bytes(path.read_bytes()) if path.is_file() else "missing"
        rows.append(f"- `{relative}`: `{digest}`")
    return "\n".join(rows)


def legacy_quote(repo: Path, relative: str) -> str:
    """Giữ snapshot legacy trong checkpoint dưới dạng block quote compact."""
    path = repo / relative
    if not path.is_file():
        return "> Không có file legacy tại thời điểm migration."
    return "\n".join(f"> {line}" if line else ">" for line in read_text(path).strip().splitlines())


def render_checkpoint(repo: Path, spec: CourseProject) -> str:
    """Tạo checkpoint migration, không giả lập unit/benchmark result."""
    now = datetime.now().astimezone()
    branch = git_value(repo, "branch", "--show-current")
    revision = git_value(repo, "rev-parse", "--short=12", "HEAD")
    status = git_value(repo, "status", "--short")
    worktree = (
        "clean"
        if status == "unknown" or not status
        else f"dirty ({len(status.splitlines())} entries)"
    )
    return dedent(f"""\
        # Checkpoint — migration tài liệu và service scaffold

        ## Định danh

        - migration_id: `{MIGRATION_ID}`
        - ngày/giờ lập kế hoạch: `{now.isoformat(timespec="seconds")}`
        - timezone: `{now.tzname() or "unknown"}`
        - project: `{spec.repo_name}`
        - evidence class: `migration_checkpoint`; không phải benchmark.

        ## Cấu hình hệ thống

        - OS: `{platform.platform()}`
        - Python: `{platform.python_version()}`
        - machine: `{platform.machine() or "unknown"}`
        - ngân sách local: `RTX 2050 4 GB VRAM` do người dùng cung cấp; migration không đo GPU.

        ## Lệnh và cấu hình chạy

        - Entry point: `py -3.12 -B scripts/upgrade_course_repositories.py --root <parent> --apply`
        - Network/package install/data/model download: `disabled`
        - Unit tests dự án: `not-run-by-migration`; chạy riêng sau apply.
        - Validator đích: script thực hiện structural validation trước exit code 0.

        ## Kết quả

        Migration tái tổ chức `General`, bổ sung tài liệu data-first/4 GB/service, cập nhật skill/validator/UI path
        và tạo inference adapter fail-closed. Kết quả đo mô hình vẫn là `none`.

        ### Snapshot trạng thái legacy

        {legacy_quote(repo, "Documents/General/STATUS.md")}

        ### Snapshot DOCX QA legacy

        {legacy_quote(repo, "Documents/General/DOCX-QA.md")}

        ## Artifact và lineage

        - branch trước migration: `{branch}`
        - code revision trước migration: `{revision}`
        - worktree trước migration: `{worktree}`
        - backup: `{BACKUP_RELATIVE.as_posix()}`
        - hashes legacy trước migration:

        {legacy_hashes(repo)}

        ## Việc tiếp theo

        Chạy unit tests, validator dự án, tái sinh DOCX/HTML từ Markdown và thực hiện round-trip/render QA; ghi
        kết quả thật vào checkpoint version mới.
        Chỉ mở data/model/inference gate sau khi registry, privacy, resource và evaluator đầy đủ.
        """)


def update_root_readme(text: str) -> str:
    """Thay các mô tả tài liệu legacy mà không ghi đè phần README khác."""
    updated = text.replace(
        "- `Documents/Decision-Making/`: problem contract, kiến trúc/gates, dữ liệu–mô hình–đánh giá và `ERROR-EXPERIENCE.md`.",
        "- `Documents/Decision-Making/`: problem, kiến trúc/gates, dữ liệu–đánh giá và quyết định tool/service.",
    )
    updated = updated.replace(
        "- `Documents/General/`: lịch môn, trạng thái, MCP và log tổng quan.",
        "- `Documents/General/`: chỉ `RULE.md`, checkpoint logs và error logs đúng grammar.",
    )
    if "checkpoint logs và error logs" not in updated:
        raise MigrationError("README.md không có seam tài liệu đã biết")
    return updated


def update_streamlit(text: str, checkpoint_relative: str) -> str:
    """Chuyển lineage khỏi STATUS legacy sang checkpoint migration."""
    new_reference = checkpoint_relative
    updated = text.replace("Documents/General/STATUS.md", new_reference)
    updated = re.sub(
        r"Documents/General/[A-Za-z0-9-]+_Checkpoints_\d{2}-\d{2}-\d{2}_V[1-9]\d*\.md",
        new_reference,
        updated,
    )
    if new_reference not in updated:
        raise MigrationError("streamlit_app/app.py không có lineage seam đã biết")
    return updated


def update_gitignore(text: str) -> str:
    """Giữ backup migration ngoài Git mà không thay rule hiện có."""
    marker = ".migration-backups/"
    if marker in text.splitlines():
        return text
    return text.rstrip() + f"\n{marker}\n"


def sanitize_checkpoint(text: str) -> str:
    """Không để tên nhánh hoãn xuất hiện ngoài decision file chuyên biệt."""
    return re.sub(r"multimodal", "deferred-scope", text, flags=re.IGNORECASE)


def update_error_experience(text: str) -> str:
    """Giữ knowledge base lỗi và phân biệt nó với error log vận hành."""
    marker = "## Ranh giới với error log vận hành"
    if marker in text:
        return text
    return (
        text.rstrip()
        + "\n\n"
        + marker
        + "\n\nFile này giữ lỗi thường gặp, kinh nghiệm đã chắt lọc và cách phòng tránh; "
        "không phải bằng chứng một sự cố đã xảy ra trong project. Lỗi thật hoặc kết quả âm tính "
        "phải tạo `Documents/General/<Project>_Error_DD-MM-YY_Vn.md` theo `RULE.md`; "
        "không tạo error log rỗng.\n"
    )


def build_desired(
    repo: Path, spec: CourseProject, profile: UpgradeProfile, checkpoint: str
) -> dict[str, str]:
    """Tạo map nội dung đích cho một repository."""
    current_checkpoint = repo / checkpoint
    checkpoint_text = sanitize_checkpoint(
        read_text(current_checkpoint)
        if current_checkpoint.is_file()
        else render_checkpoint(repo, spec)
    )
    root_readme = update_root_readme(read_text(repo / "README.md"))
    streamlit = update_streamlit(read_text(repo / "streamlit_app" / "app.py"), checkpoint)
    gitignore = update_gitignore(read_text(repo / ".gitignore"))
    package = spec.package_name
    return {
        ".gitignore": gitignore,
        "README.md": root_readme,
        "AGENTS.md": render_agents(spec),
        "Documents/README.md": render_documents_readme(),
        "Documents/Decision-Making/03-TOOLS-AND-SERVICES.md": render_tools_decision(spec),
        "Documents/Decision-Making/04-DEFERRED-MULTIMODAL.md": render_deferred(),
        "Documents/Decision-Making/ERROR-EXPERIENCE.md": update_error_experience(
            read_text(repo / "Documents/Decision-Making/ERROR-EXPERIENCE.md")
        ),
        "Documents/Decision-Making/05-PROJECT-CONTEXT.md": render_project_context(spec, profile),
        "Documents/Decision-Making/06-FEASIBILITY-DATA-4GB.md": render_feasibility(spec, profile),
        "Documents/References/RESEARCH-AND-DATASETS.md": render_research(spec, profile),
        "Documents/References/PROJECT-KNOWLEDGE.md": render_knowledge(spec, profile),
        "Documents/References/PROJECT-PLAN.html": render_project_html(spec, profile),
        "Documents/References/EXTERNAL-SERVICES-GUIDELINE.md": render_services(spec, profile),
        "Documents/References/COMMON-FAILURES-AND-PREVENTION.md": render_failures(spec, profile),
        "Documents/General/RULE.md": render_rule(spec),
        checkpoint: checkpoint_text,
        ".agents/skills/course-project-delivery/SKILL.md": render_skill(spec),
        ".agents/skills/course-project-delivery/references/gates.md": render_skill_gates(),
        f"src/{package}/ServiceAdapters/inference.py": render_inference_module(spec, profile),
        "tests/test_inference_adapter.py": render_inference_test(spec, profile),
        "scripts/validate_project.py": render_validator(spec),
        "streamlit_app/app.py": streamlit,
    }


def preflight_repo(root: Path, spec: CourseProject) -> Path:
    """Xác minh đúng repository và các seam bắt buộc trước khi lập plan."""
    repo = safe_target(root, spec.repo_name)
    required = (
        "AGENTS.md",
        "README.md",
        ".gitignore",
        "Documents/General",
        "streamlit_app/app.py",
        ".agents/skills/course-project-delivery/SKILL.md",
        "scripts/validate_project.py",
    )
    missing = [relative for relative in required if not (repo / relative).exists()]
    if missing:
        raise MigrationError(f"{spec.repo_name} thiếu preflight paths: {missing}")
    legacy_names = {Path(relative).name for relative in LEGACY_PATHS if "/General/" in relative}
    allowed = legacy_names | {"RULE.md"}
    unexpected = [
        path.name
        for path in (repo / "Documents" / "General").iterdir()
        if path.is_file()
        and path.name not in allowed
        and checkpoint_pattern(spec).fullmatch(path.name) is None
    ]
    if unexpected:
        raise MigrationError(
            f"{spec.repo_name} có General files chưa phân loại: {sorted(unexpected)}"
        )
    return repo


def build_repository_plan(root: Path, spec: CourseProject) -> RepositoryPlan:
    """Lập trạng thái đích mà không ghi file."""
    profile = PROFILES[spec.course_code]
    repo = preflight_repo(root, spec)
    checkpoint = select_checkpoint(repo, spec)
    desired = build_desired(repo, spec, profile, checkpoint)
    return RepositoryPlan(spec, profile, repo, checkpoint, desired, LEGACY_PATHS)


def pending_changes(plan: RepositoryPlan) -> list[FileChange]:
    """So trạng thái hiện tại với desired state."""
    changes: list[FileChange] = []
    for relative, content in plan.desired.items():
        path = safe_target(plan.repo, relative)
        if not path.is_file() or read_text(path) != content:
            changes.append(FileChange(relative, "write", content, "managed desired state"))
    for relative in plan.deletes:
        if safe_target(plan.repo, relative).exists():
            changes.append(FileChange(relative, "delete", None, "legacy path retired"))
    return changes


def backup_change(plan: RepositoryPlan, change: FileChange) -> None:
    """Backup file hiện có một lần và từ chối overwrite khi source đã drift."""
    source = safe_target(plan.repo, change.relative)
    if not source.is_file():
        return
    backup = safe_target(plan.repo, (BACKUP_RELATIVE / "original" / change.relative).as_posix())
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        if backup.read_bytes() != source.read_bytes():
            desired = change.content.encode("utf-8") if change.content is not None else None
            if desired is None or source.read_bytes() != desired:
                raise MigrationError(f"backup đã tồn tại nhưng source drifted: {source}")
        return
    shutil.copy2(source, backup)


def atomic_write(path: Path, content: str) -> None:
    """Ghi UTF-8 bằng replace nguyên tử trong cùng thư mục."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content.rstrip() + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def manifest_payload(
    plan: RepositoryPlan, changes: list[FileChange], status: str
) -> dict[str, object]:
    """Tạo manifest rollback có hash source/desired."""
    entries: list[dict[str, object]] = []
    for change in changes:
        backup = safe_target(plan.repo, (BACKUP_RELATIVE / "original" / change.relative).as_posix())
        original = backup.read_bytes() if backup.is_file() else None
        desired = change.content.encode("utf-8") if change.content is not None else None
        entries.append(
            {
                "path": change.relative,
                "operation": change.operation,
                "original_sha256": (sha256_bytes(original) if original is not None else None),
                "desired_sha256": (sha256_bytes(desired) if desired is not None else None),
            }
        )
    return {
        "migration_id": MIGRATION_ID,
        "project": plan.spec.repo_name,
        "status": status,
        "changes": entries,
    }


def write_manifest(plan: RepositoryPlan, changes: list[FileChange], status: str) -> None:
    """Ghi trạng thái backup/apply để partial failure quan sát được."""
    path = safe_target(plan.repo, (BACKUP_RELATIVE / "manifest.json").as_posix())
    content = (
        json.dumps(manifest_payload(plan, changes, status), ensure_ascii=False, indent=2) + "\n"
    )
    atomic_write(path, content)


def apply_changes(plan: RepositoryPlan, changes: list[FileChange]) -> None:
    """Áp dụng writes trước, deletes sau; tất cả source đã được backup."""
    writes = sorted(
        (change for change in changes if change.operation == "write"),
        key=lambda item: item.relative,
    )
    deletes = sorted(
        (change for change in changes if change.operation == "delete"),
        key=lambda item: item.relative,
    )
    for change in writes:
        if change.content is None:
            raise MigrationError(f"write thiếu content: {change.relative}")
        atomic_write(safe_target(plan.repo, change.relative), change.content)
    for change in deletes:
        target = safe_target(plan.repo, change.relative)
        if target.is_file():
            target.unlink()


def validate_log_files(plan: RepositoryPlan, failures: list[str]) -> None:
    """Kiểm tra General chỉ còn RULE và logs đúng grammar."""
    general = plan.repo / "Documents" / "General"
    groups: dict[tuple[str, str], list[int]] = {}
    checkpoints = 0
    for path in sorted(general.iterdir()):
        if not path.is_file() or path.name == "RULE.md":
            continue
        match = checkpoint_pattern(plan.spec).fullmatch(path.name)
        if match is None:
            failures.append(f"unexpected General file: {path.name}")
            continue
        kind, date_text, version_text = match.groups()
        try:
            datetime.strptime(date_text, "%d-%m-%y")
        except ValueError:
            failures.append(f"invalid date: {path.name}")
        groups.setdefault((kind, date_text), []).append(int(version_text))
        headings = CHECKPOINT_HEADINGS if kind == "Checkpoints" else ERROR_HEADINGS
        text = read_text(path)
        failures.extend(
            f"{path.name} missing {heading}" for heading in headings if heading not in text
        )
        checkpoints += kind == "Checkpoints"
    for key, values in groups.items():
        if sorted(values) != list(range(1, max(values) + 1)):
            failures.append(f"non-contiguous versions {key}: {sorted(values)}")
    if checkpoints < 1:
        failures.append("missing checkpoint log")


def validate_repository(plan: RepositoryPlan) -> list[str]:
    """So repository với desired state và compile các Python file được quản lý."""
    failures: list[str] = []
    for relative, expected in plan.desired.items():
        path = safe_target(plan.repo, relative)
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = read_text(path)
        if actual != expected.rstrip() + "\n" and actual != expected:
            failures.append(f"content drift: {relative}")
        if path.suffix == ".py":
            try:
                compile(actual, str(path), "exec")
            except SyntaxError as exc:
                failures.append(f"syntax: {relative}: {exc}")
    failures.extend(
        f"legacy remains: {relative}"
        for relative in plan.deletes
        if (plan.repo / relative).exists()
    )
    validate_log_files(plan, failures)
    return failures


def print_plan(plan: RepositoryPlan, changes: list[FileChange]) -> None:
    """In kế hoạch dry-run có đường dẫn backup rõ ràng."""
    print(f"PROJECT {plan.spec.repo_name}")
    if not changes:
        print("  NO-CHANGES")
        return
    for change in changes:
        print(f"  {change.operation.upper():6} {change.relative} — {change.reason}")
    print(f"  BACKUP {BACKUP_RELATIVE.as_posix()}/original/")


def parse_args() -> argparse.Namespace:
    """Parse root và đúng một mode; dry-run là mặc định."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Thư mục cha chứa đúng bốn course repositories.",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--dry-run",
        dest="mode",
        action="store_const",
        const="dry-run",
        help="Chỉ in plan.",
    )
    modes.add_argument(
        "--apply",
        dest="mode",
        action="store_const",
        const="apply",
        help="Backup rồi áp dụng.",
    )
    modes.add_argument(
        "--validate-only",
        dest="mode",
        action="store_const",
        const="validate-only",
        help="Chỉ kiểm tra trạng thái sau migration; không ghi file.",
    )
    parser.set_defaults(mode="dry-run")
    return parser.parse_args()


def main() -> int:
    """Lập plan cho cả bốn repo, rồi dry-run, apply hoặc validate."""
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise MigrationError(f"--root không phải thư mục: {root}")
    plans = [build_repository_plan(root, spec) for spec in PROJECTS]
    changes_by_repo = [(plan, pending_changes(plan)) for plan in plans]
    if args.mode == "dry-run":
        for plan, changes in changes_by_repo:
            print_plan(plan, changes)
        print("DRY-RUN: không file nào được thay đổi")
        return 0
    if args.mode == "validate-only":
        failures = [(plan, validate_repository(plan)) for plan in plans]
        has_failures = False
        for plan, project_failures in failures:
            if project_failures:
                has_failures = True
                print(f"INVALID {plan.spec.repo_name}")
                for failure in project_failures:
                    print(f"- {failure}")
            else:
                print(f"VALID {plan.spec.repo_name}")
        return 1 if has_failures else 0
    for plan, changes in changes_by_repo:
        for change in changes:
            backup_change(plan, change)
    for plan, changes in changes_by_repo:
        if changes:
            write_manifest(plan, changes, "backed_up")
    for plan, changes in changes_by_repo:
        if not changes:
            print(f"NO-CHANGES {plan.spec.repo_name}")
            continue
        apply_changes(plan, changes)
        failures = validate_repository(plan)
        if failures:
            raise MigrationError(
                f"post-apply validation failed for {plan.spec.repo_name}: {failures}"
            )
        write_manifest(plan, changes, "applied")
        print(f"APPLIED {plan.spec.repo_name}: {len(changes)} changes")
    print(f"BACKUP ROOT PER REPOSITORY: {BACKUP_RELATIVE.as_posix()}")
    print(
        "ROLLBACK: restore original/<relative-path>; remove manifest entries with original_sha256=null."
    )
    print("Không chạy unit tests, tải package, data/model hoặc network call.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MigrationError as exc:
        print(f"MIGRATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
