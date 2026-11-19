"""Refine seven sibling repositories without downloads or model execution.

The script is dry-run by default.  Apply mode writes only known documentation
and local-instruction files, creates per-repository backups, and never edits an
existing checkpoint or error log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from bootstrap_course_repositories import validator_script
from course_repo_catalog import PROJECTS, CourseProject
from course_repo_templates import (
    render_agents,
    render_common_failures,
    render_curriculum_alignment,
    render_error_experience,
    render_general_rule,
    render_knowledge,
    render_services,
    render_skill,
    render_skill_gates,
    render_workflow,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS_ROOT = ROOT.parent
MIGRATION_ID = "code-hygiene-policy-v1"
COURSE_BY_REPO = {item.repo_name: item for item in PROJECTS}
REPOSITORIES = (
    *COURSE_BY_REPO,
    "Evidence-Grounded-Text-Agent",
    "DSC-TinDipLaPo",
    "Multimodal-Asset-Retrieval",
)
LOG_PATTERN = re.compile(r".+_(?:Checkpoints|Error)_\d{2}-\d{2}-\d{2}_V[1-9]\d*\.md$")
MOJIBAKE = tuple(
    "".join(chr(codepoint) for codepoint in sequence)
    for sequence in (
        (0x00C3,),
        (0x00C6,),
        (0x00E1, 0x00BA),
        (0x00E1, 0x00BB),
        (0x00EF, 0x00BF, 0x00BD),
        (0xFFFD,),
    )
)
BLOCK_BEGIN = "<!-- BEGIN research-refinement:v1 -->"
BLOCK_END = "<!-- END research-refinement:v1 -->"


@dataclass(frozen=True, slots=True)
class Change:
    """Describe one planned text-file write."""

    repo: Path
    relative: Path
    content: str
    reason: str


def sha256_bytes(payload: bytes) -> str:
    """Return a stable SHA-256 digest."""

    return hashlib.sha256(payload).hexdigest()


def read_text(path: Path) -> str:
    """Read one UTF-8 file, returning an empty string for a new file."""

    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def normalize_text(text: str) -> str:
    """Normalize line endings and guarantee one final newline."""

    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n"


def space_markdown_lists(text: str) -> str:
    """Separate adjacent list items so active guidance stays scannable."""

    return re.sub(r"(?m)(?<!\n)\n(?=(?:\d+\. |- ))", "\n\n", text)


def managed_block(body: str) -> str:
    """Wrap a durable idempotent Markdown block."""

    return f"{BLOCK_BEGIN}\n{body.strip()}\n{BLOCK_END}"


def upsert_block(text: str, body: str) -> str:
    """Insert or replace the managed refinement block."""

    block = managed_block(body)
    pattern = re.compile(
        re.escape(BLOCK_BEGIN) + r".*?" + re.escape(BLOCK_END),
        flags=re.DOTALL,
    )
    if pattern.search(text):
        return normalize_text(pattern.sub(block, text))
    return normalize_text(text + "\n" + block)


def insert_html_block(text: str, body: str) -> str:
    """Insert an idempotent HTML section before body close."""

    block = managed_block(body)
    pattern = re.compile(
        re.escape(BLOCK_BEGIN) + r".*?" + re.escape(BLOCK_END),
        flags=re.DOTALL,
    )
    if pattern.search(text):
        return normalize_text(pattern.sub(block, text))
    if "</body>" not in text:
        raise ValueError("HTML target has no closing body tag")
    return normalize_text(text.replace("</body>", block + "\n</body>"))


def project_slices(name: str) -> str:
    """Return the initial slice registry for a repository."""

    if name == "Multimodal-Asset-Retrieval":
        return (
            "modality presence/missing, query modality, fusion conflict, "
            "component score, provenance và source"
        )
    return (
        "intent, lexical-semantic gap, length, domain, judged coverage, "
        "hard-negative, citation và abstention"
    )


def render_noncourse_rule(name: str) -> str:
    """Render the complete General rule for a non-course repository."""

    slices = project_slices(name)
    return f"""# Quy tắc Documents/General — {name}

## Phạm vi và tên file

General chỉ chứa RULE.md và hai họ log bất biến:

- {name}_Checkpoints_DD-MM-YY_Vn.md
- {name}_Error_DD-MM-YY_Vn.md

DD-MM-YY thay DD/MM/YY vì Windows dùng dấu gạch chéo làm ký tự đường dẫn. Vn bắt
đầu từ V1, tăng theo ngày và loại; không ghi đè hoặc sửa log cũ. Tri thức, ADR,
hướng dẫn, meeting note và báo cáo thuộc Decision-Making, References hoặc Docs.

## Checkpoint tối thiểu

Ghi project, ngày/version, owner, Git/worktree, system profile, lệnh/exit code,
data-corpus/config/model/code revision, seed, evidence class, artifact/hash, kết
quả, giới hạn, replay/rollback và next gate. Thiếu trường nào ghi không thu thập,
không suy đoán.

Với run so sánh, khóa anchor_checkpoint, parent_checkpoint, hypothesis_id,
evaluation_role=development|error_analysis|sealed_test, gold_status và
evaluation_contract_hash. Contract bao gồm data manifest, split, gold/qrels,
evaluator, metric, parser và slice registry. Khác contract hoặc đổi nhiều biến
không tách được phải ghi comparison=confounded.

Báo paired delta theo case/query/group so với anchor và parent: overall, slice,
denominator, seed/CI và wins|ties|losses. Artifact chi tiết giữ output/rank/score
cũ-mới, slice ID, gold/proxy, invalid/abstention và error category; log chỉ dẫn
URI/hash.

## Học từ hành vi dữ liệu

Khóa case budget và quy tắc chọn trước run: regression lớn nhất, ca được cứu, lỗi
dai dẳng, low-margin/high-variability, counterexample và mẫu random có seed trên
lát cắt trọng yếu. Slice khởi điểm: {slices}. Mỗi slice phải có definition,
revision, count và metadata; slice post-hoc là exploratory cho tới run xác nhận.

Tách observation khỏi cause. Kết luận hypothesis bằng
supported|refuted|inconclusive, kèm dự đoán có thể bác bỏ, minimal next experiment
và stop/rollback gate. Gold partial phải ghi judged coverage và unjudged policy.
Gold absent chỉ cho phép quality, stability, behavioral test hoặc human audit;
pseudo-label, synthetic label và LLM judge là proxy, không phải gold.

## Error

Error chỉ dành cho run không hợp lệ hoặc quality gate định trước chuyển FAIL.
Ghi checkpoint liên quan, first_bad, last_known_good, invariant, stage, ca/lát
cắt, expected/actual, reproducer, log đã lọc secret, bằng chứng ủng hộ/phản bác
nguyên nhân, containment/rollback, regression và residual risk.

Run hợp lệ nhưng metric kém hơn là Checkpoint decision=REJECT|REPEAT. Giả thuyết
bị bác bỏ không tự tạo Error.

## Ngôn ngữ và bảo mật

Viết tiếng Việt có dấu, viết hoa đầu câu và dùng câu ngắn. Không chép debug
narrative; chỉ dẫn artifact đã lọc và có hash. Không ghi token, secret, PII,
organizer/private payload, model weights hoặc transcript dài.
"""


def retrieval_agents(name: str) -> str:
    """Render Vietnamese root instructions for the two legal retrieval repos."""

    identity = (
        "Đây là workspace private của nhóm UIT DSC 2026 cho Task 1 LegalIR và Task 2 LegalQA."
        if name == "DSC-TinDipLaPo"
        else "Đây là hệ truy xuất văn bản tiếng Việt–Anh cho tập tài liệu được quản trị."
    )
    protected_assets = (
        "organizer data, run, submission, corpus, index, model, cache, trace"
        if name == "DSC-TinDipLaPo"
        else "private corpus, index, model, cache, trace"
    )
    data_gate = (
        "- Với organizer data, khóa quyền truy cập, split, schema, checksum và "
        "submission contract trước khi đổi data/model."
        if name == "DSC-TinDipLaPo"
        else "- Đọc `Documents/Decision-making/Shared/0002-data-feasibility-4gb.md` "
        "trước khi đổi data/split/model; baseline CPU và exact search đi trước neural candidate."
    )
    return f"""# Quy ước dự án {name}

## Định danh

{identity} Task 1
LegalIR trả document ID có provenance; Task 2 LegalQA chỉ trả lời từ evidence đã
truy xuất và có thể abstain. Runtime hiện tại là baseline deterministic gồm BM25,
TF-IDF, RRF, access-scope filter, citation validation và extractive fallback.

Không tuyên bố dense retrieval, reranking, chất lượng sinh, production hoặc điểm
DSC chính thức trước khi được triển khai và đánh giá theo quy định BTC.
Không bịa query, document ID, citation, access right, permission, luật, score hoặc
answer không được evidence hỗ trợ.

## Workflow bắt buộc

Đọc Workflows/flowinfo.md, skills/retrieval-delivery/SKILL.md và skill đúng task
trước thay đổi kiến trúc, data, indexing, retrieval, agent, evaluation, API hoặc
UI.

document manifest → deterministic chunks → independent retrievers → fusion →
bounded policy → evidence citations hoặc abstention

## Quy tắc kỹ thuật

- Giữ query, document_id, chunk_id, source URI, access scope, corpus version, rank và score.
- Lọc access scope trước candidate generation và trước answer assembly.
- Retrieval phải chạy khi LLM/service không khả dụng.
- Citation chỉ trỏ tới chunk đã truy xuất trong active corpus version.
- Giới hạn số bước agent và số evidence bằng cấu hình có version.
- Chỉ dùng model trong configs/Shared/model_allowlist.json; quy định BTC ưu tiên.
- Tách Task 1/Task 2 về data, code, config, test, experiment, submission và docs.
- Giữ {protected_assets} và secret ngoài Git.
- Đánh giá retrieval, abstention, citation integrity, latency và failure behavior.
- Không dùng emoji, sticker, banner ký hiệu hoặc thông báo ăn mừng trong mã, test, log hay CLI; chỉ giữ ký hiệu có nghĩa kỹ thuật hoặc dữ liệu protocol đã ghi rõ.

## Data và external inference gate

{data_gate}
- External inference chỉ qua `Source/Shared/ServiceAdapters`; pin model/provider/revision, timeout, privacy và schema; mặc định fail-closed.
- Không gửi PII, organizer/private corpus hoặc secret ra service chưa được duyệt.

## Học từ kết quả và tài liệu

Đọc Documents/General/RULE.md trước comparison. Khóa anchor/parent, hypothesis,
evaluation role, gold status, evaluation contract, paired case/slice delta và
counterexample. Run hợp lệ nhưng kém hơn là Checkpoint REJECT|REPEAT; Error chỉ
dành cho invalid run hoặc quality gate FAIL.

Tài liệu do project sở hữu viết tiếng Việt có dấu, viết hoa đầu câu, gọn và có
nguồn gần mệnh đề. Không chép debug narrative hoặc dùng scaffold/smoke làm kết quả.

## Hoàn thành

Fixture ingestion/search/answer/evaluation chạy từ môi trường sạch; unauthorized
document không vào candidate/citation; test, Ruff, mypy, workflow validator và
lockfile check đạt. Validator canonical là
`uv run python skills/retrieval-delivery/scripts/validate_project.py`; tài liệu
tách implemented khỏi roadmap.
"""


def multimodal_agents() -> str:
    """Render Vietnamese root instructions for the asset repository."""

    return """# Quy ước dự án Multimodal-Asset-Retrieval

## Định danh

Đây là capstone truy xuất asset theo manifest bằng văn bản, ảnh mẫu và metadata,
giữ component score, evidence, provenance và trạng thái modality. Runtime hiện
tại là deterministic Simple baseline. Không gọi color descriptor là semantic
vision hoặc tuyên bố production nếu chưa có qrels, corpus version, latency và replay.

## Workflow bắt buộc

Đọc Workflows/flowinfo.md và skills/retrieval-delivery/SKILL.md. Triển khai lát
cắt nhỏ nhất: manifest → contract validation → candidate generation → fusion →
evidence-bearing results → evaluation.

## Quy tắc kỹ thuật

- Giữ asset_id, source URI, modality score, rank, evidence và provenance.
- Tách candidate generation khỏi fusion; Simple dùng exact search làm baseline.
- Stage routing và integrity mode là contract tường minh; không dùng magic checksum sentinel.
- Route công việc theo capability trước khi áp dụng in-process allowlist.
- Coordinator preflight input, đăng ký canonical pending state rồi mới enqueue.
- Ưu tiên một request/config document và optional env file thay long CLI flag set.
- Worker phải fail trước startup khi identity, queue, resource hoặc device mâu thuẫn.
- Model đắt chỉ chạy trên candidate bounded qua adapter có version và fallback.
- Modality lỗi phải trả trạng thái nhìn thấy được, không âm thầm bỏ mẫu.
- Giữ raw asset, embedding, index, checkpoint, cache và secret ngoài Git.
- Không claim CLIP, ANN, API hoặc UI trước implementation và validation tương ứng.
- Không dùng emoji, sticker, banner ký hiệu hoặc thông báo ăn mừng trong mã, test, log hay CLI; chỉ giữ ký hiệu có nghĩa kỹ thuật hoặc dữ liệu protocol đã ghi rõ.

## Học từ kết quả và tài liệu

Đọc Documents/General/RULE.md. Khóa anchor/parent, hypothesis, evaluation role,
gold status, evaluation contract, paired case/slice delta và counterexample.
Run hợp lệ nhưng kém hơn là Checkpoint REJECT|REPEAT; Error chỉ dành cho invalid
run hoặc quality gate FAIL.

Tài liệu viết tiếng Việt có dấu, viết hoa đầu câu và gọn; tiếng Anh chỉ giữ cho
API, identifier, mode và thuật ngữ cần thiết. Không chép debug narrative.

## Hoàn thành

Contract/failure rõ; fixture search/evaluation chạy sạch; runtime, operations
contract và Development roadmap được tách riêng. UI/model/diagram không phải
implementation evidence. Test, Ruff, mypy, workflow validator và lockfile check đạt.
"""


def retrieval_skill(name: str) -> str:
    """Render the root retrieval-delivery skill in Vietnamese."""

    package = "text_retrieval_agent"
    validator = "uv run python skills/retrieval-delivery/scripts/validate_project.py"
    return f"""---
name: retrieval-delivery
description: Triển khai thay đổi retrieval có phạm vi, evidence, replay và failure gate trong {name}.
---

# Giao nhận retrieval

## Gate đầu vào

1. Đọc AGENTS.md, Workflows/flowinfo.md và Documents/General/RULE.md.

2. Kiểm tra Git; bảo toàn thay đổi ngoài phạm vi.

3. Khóa input/output, corpus/query, access scope, evidence, fallback và stop gate.

4. Chạy deterministic retrieval/fixture trước LLM, external index hoặc hạ tầng.

5. Chỉ dùng model thuộc allowlist và revision đã duyệt.

## Học từ run

Trước comparison, khóa anchor/parent, hypothesis, evaluation role, gold status,
evaluation contract, slice registry và case budget. Báo paired delta,
wins|ties|losses, counterexample; tách observation khỏi cause và kết luận
supported|refuted|inconclusive. Run hợp lệ nhưng kém hơn là Checkpoint
REJECT|REPEAT; Error chỉ dành cho invalid run hoặc quality gate FAIL.

## Gate hoàn thành

    uv lock --check
    uv run ruff format --check .
    uv run ruff check .
    uv run mypy
    uv run pytest --cov={package} --cov-report=term-missing
    {validator}

Output sinh không có citation đã truy xuất và được phép là không hợp lệ. Tài liệu
viết tiếng Việt có dấu, viết hoa đầu câu, gọn và không chép debug narrative.
"""


def multimodal_retrieval_skill() -> str:
    """Render the asset-specific delivery skill without legal-RAG assumptions."""

    return """---
name: retrieval-delivery
description: Triển khai multimodal asset retrieval theo contract, provenance, capability, replay và failure gate.
---

# Giao nhận multimodal retrieval

## Gate đầu vào

1. Đọc `AGENTS.md`, `Workflows/flowinfo.md` và `Documents/General/RULE.md`.

2. Kiểm tra Git; bảo toàn thay đổi ngoài phạm vi.

3. Khóa input/output, corpus/query, evidence, validation, fallback và stop gate.

4. Dùng fixture và exact Simple baseline trước hạ tầng hoặc model dependency.

## Operations gate

Dùng mode Simple/Development tường minh. Preflight input và đăng ký canonical
pending state trước enqueue. Route theo capability queue, sau đó mới áp dụng
allowlist. Từ chối worker env mâu thuẫn trước startup. Không dùng checksum
sentinel hoặc long task-specific CLI; ưu tiên request/config document và optional
env file.

## Học từ run

Khóa anchor/parent, hypothesis, evaluation role, gold status, evaluation contract,
slice registry và case budget. Báo paired delta, wins/ties/losses, counterexample;
tách observation khỏi cause. Run hợp lệ nhưng kém hơn là Checkpoint
REJECT|REPEAT; Error chỉ dành cho invalid run hoặc quality gate FAIL.

## Gate hoàn thành

    uv lock --check
    uv run ruff format --check .
    uv run ruff check .
    uv run mypy
    uv run pytest --cov=multimodal_retrieval --cov-report=term-missing
    python skills/retrieval-delivery/scripts/validate_project.py

UI, model adapter hoặc architecture diagram không phải implementation evidence.
UI/report phải hiện provenance, capability status và evidence class. Tài liệu viết
tiếng Việt có dấu, viết hoa đầu câu, gọn và không chép debug narrative.
"""


def legal_ir_skill() -> str:
    """Render the Task 1 local skill in Vietnamese."""

    return """---
name: dsc-legal-ir-delivery
description: Triển khai Task 1 LegalIR với candidate bounded và bằng chứng retrieval tái lập.
---

# Giao nhận Task 1

1. Đọc AGENTS.md, Workflows/flowinfo.md và ADR Task 1.
2. Khóa corpus manifest, split, qrels, access scope và output schema BTC.
3. Chạy BM25 + TF-IDF/RRF trước neural adapter; từ chối model ngoài allowlist.
4. Giữ document/article ID, component score, rank và corpus version.
5. Retrieve rộng trước rerank bounded; so cùng query set và evaluation contract.
6. Báo metric, latency, memory, failure, paired case/slice delta và counterexample.
7. Macro-F2 là metric local cho tới khi BTC xác nhận.
8. Giữ model, index, organizer data, run và submission ngoài Git.

Chạy gate retrieval-delivery gốc và warm-up validator khi file tồn tại. Log theo
Documents/General/RULE.md; tài liệu viết tiếng Việt có dấu, gọn.
"""


def legal_qa_skill() -> str:
    """Render the Task 2 local skill in Vietnamese."""

    return """---
name: dsc-legal-qa-delivery
description: Triển khai Task 2 LegalQA với evidence, citation, fallback và abstention.
---

# Giao nhận Task 2

1. Đọc AGENTS.md, Workflows/flowinfo.md và ADR Task 2.
2. Đóng băng Task 1 retrieval release và active corpus version.
3. Kiểm tra extractive fallback trước generator; từ chối model ngoài allowlist.
4. Chỉ đưa chunk bounded, được phép vào answer stage.
5. Citation và quoted span phải resolve tới retrieved evidence.
6. Không bịa luật, điều, nguồn hoặc document ID.
7. Thiếu evidence, timeout, model lỗi hoặc malformed output thì extract hoặc abstain.
8. Đánh giá answer tách retrieval; báo failure, paired case/slice delta và counterexample.

Chạy gate retrieval-delivery gốc và warm-up validator khi file tồn tại. Log theo
Documents/General/RULE.md; tài liệu viết tiếng Việt có dấu, gọn.
"""


def skill_learning_block(name: str) -> str:
    """Render a fallback block for any unrecognized local skill."""

    return f"""## Quy tắc tài liệu và học từ kết quả

Nội dung do {name} sở hữu viết tiếng Việt có dấu, viết hoa đầu câu, gọn và không
chép debug narrative. Khi kết quả được nhập vào lịch sử project, đọc
Documents/General/RULE.md; khóa anchor/parent, evaluation contract, gold status,
case/slice delta và counterexample. Tách observation khỏi cause. Run hợp lệ nhưng
kém hơn là Checkpoint REJECT|REPEAT; Error chỉ dành cho invalid run hoặc quality
gate FAIL."""


def prior_art_task1() -> str:
    """Render the Vietnamese Task 1 evidence map."""

    return """# Bản đồ bằng chứng Task 1

UIT DSC định danh Task 1 là Legal Information Retrieval trên Codabench. Trang
công khai chưa cung cấp đầy đủ metric, schema nộp và quy định; thông tin local do
người dùng cung cấp phải được BTC ưu tiên.

| Nguồn | Phát hiện | Cách dùng |
|---|---|---|
| [ALQAC 2024](https://sites.google.com/view/alqac-2024/home) | Legal retrieval tiếng Việt tương tự dùng relevance cấp điều và F2 | Chỉ diagnostic local, không phải luật UIT |
| [NOWJ1 2023](https://arxiv.org/abs/2309.09070) | Lexical + neural, cấu trúc điều/khoản và learning-to-rank | Giữ cấu trúc; hoãn learned fusion |
| [NOWJ 2024](https://doi.org/10.1109/KSE63888.2024.11063594) | Winning system kết hợp ranking signal | Benchmark fusion/rerank theo stage |
| [ViDRILL](https://aclanthology.org/2025.vlsp-1.17/) | BM25 + dense + cross-encoder; chunk/hard negative quan trọng | Multi-stage bounded sau baseline |
| [ViRE](https://aclanthology.org/2026.findings-eacl.110/) | Embedding tiếng Việt + lexical mạnh trên nhiều domain | Candidate hybrid, cần held-out |
| [ALQAC code](https://github.com/baohl00/alqac24) | Implementation retrieval/QA công khai | Tham khảo adapter, không chép post-process |
| [ViRE code](https://github.com/longstnguyen/ViRE) | Evaluation code cho IR tiếng Việt | Tham khảo protocol |

Quyết định: lexical deterministic trước. Neural pair đầu là
`AITeamVN/Vietnamese_Embedding_v2` + `AITeamVN/Vietnamese_Reranker`; BGE-M3,
E5, Qwen3 và VietLegal-Harrier chỉ là challenger. Tất cả vẫn bị chặn bởi qrels,
hard-negative analysis, resource probe và cùng evaluation contract; chưa model nào
được bật hoặc có điểm DSC.
"""


def prior_art_task2() -> str:
    """Render the Vietnamese Task 2 evidence map."""

    return """# Bản đồ bằng chứng Task 2

UIT DSC định danh Task 2 là Legal Question Answering trên Codabench. Trang công
khai chưa cung cấp đầy đủ metric, schema nộp và quy định; thông tin local do
người dùng cung cấp phải được BTC ưu tiên.

| Nguồn | Phát hiện | Cách dùng |
|---|---|---|
| [ALQAC 2024](https://sites.google.com/view/alqac-2024/home) | QA tương tự có extractive, yes/no và multiple choice | Phân tích type sau official schema |
| [ALQAC summary](https://openreview.net/forum?id=uDg8v77IBe) | Retrieval và answer evaluation tách riêng | Giữ evidence hai tầng |
| [NeCo 2023](https://arxiv.org/abs/2309.05500) | Extraction theo question type | Giữ extractive path |
| [NOWJ1 2023](https://arxiv.org/abs/2309.09070) | Tách extraction khỏi classification | Không dùng một prompt cho mọi type |
| [ViLQA](https://github.com/ntphuc149/ViLQA) | So span extraction và generation | Benchmark khi corpus hợp lệ |
| [ALQAC code](https://github.com/baohl00/alqac24) | Retrieval + LLM công khai | Chỉ tham khảo implementation |
| [Vietnamese Law QA](https://github.com/ngothanhnam0910/Vietnamese-Law-Question-Answering-system) | Ứng dụng hybrid có external API | Tham khảo kiến trúc, service vẫn gated |

Quyết định: evidence-bound extraction là baseline. `Qwen/Qwen2.5-1.5B-Instruct`
là general comparator đầu; `AITeamVN/Vi-Qwen2-1.5B-RAG` là retrieval-tuned
challenger. Cả hai chỉ mở sau retrieval release, citation gate, resource/privacy
gate và held-out evaluation; chưa model nào được bật hoặc chứng minh tốt hơn.
"""


def multimodal_steps() -> str:
    """Render a compact Vietnamese Simple-to-Development plan."""

    return """# Lộ trình từ Simple đến Development

Cập nhật: 26/08/2026. Đây là kế hoạch capstone, không phải protocol chính thức AIC 2026.

## 1. Sự thật hiện tại

- Đã có manifest, lexical/color exact retrieval, RRF, evaluation và contract integrity/routing.
- PostgreSQL, MinIO, Redis/RQ runtime; CLIP/E5; Qdrant/release; API/UI/Agent chưa integrated.
- Contract, sơ đồ, port reachable hoặc embedding export không tự là runtime hay retrieval-quality evidence.

## 2. Khóa Simple

Version manifest; reject ID bất ổn/trùng; giữ millisecond nguyên, modality status,
provenance, source revision, multipart/shard và component score/rank. Keyframe ID
gồm video_id + frame_id + keyframe_ordinal. Exact fixture replay là acceptance baseline.

## 3. Đóng operations gap

request → typed validation → object preflight → canonical pending state →
capability queue → worker lifecycle → artifact/mapping/visible failure.

PostgreSQL sở hữu state/idempotency; MinIO lưu input/output bất biến; Redis/RQ
route metadata/keyframe/modality/embedding. Coordinator không đăng ký pending
được thì không enqueue. Missing object fail trước queue; duplicate ID có error ổn
định; cleanup bắt buộc queue cụ thể và dry-run. Simple dùng integrity none ở
queue boundary, không checksum sentinel.

## 4. Semantic embedding và Development integrity

Adapter encoder phải pin model, preprocessing, dimension, dtype, normalization và
device. Quét đủ shard/part, xuất matrix + mapping 1:1 + failure rows; không truncate.
So exact semantic retrieval với deterministic baseline trên qrels cố định. Artifact
được promote dùng SHA-256, reject missing/mismatch/corrupt/count drift và quarantine.

## 5. Index và serving release

Build Qdrant từ portable matrix + mapping; point ID tách backend UUID. So exact
top-k trên cùng query/filter. Ghi index build, source batch, config, metric và
backend ref. Validate release rồi đổi active pointer; test stale/missing/schema,
activation failure và last-known-good rollback.

## 6. Online retrieval và UI

Candidate API bounded, metadata filter, fusion evidence, temporal grouping, dedup
và rerank có timeout/fallback. Trả release ID, asset/video ID, timestamp, evidence
và modality status. UI có query/filter/evidence/context/compare/export cùng loading,
empty, partial failure, stale và retry. Submit/export giữ preview + dry-run cho tới
khi protocol/quyền được xác nhận.

## 7. Automation cuối cùng

Agent chỉ gọi tool allowlisted có typed schema, top-k/timeout/retry budget, fallback
và audit trace; không bịa ID/timestamp hoặc quét archive bằng prompt. External
submission luôn cần deterministic validation và human-confirmed dry-run.

## 8. Học từ kết quả

Mọi comparison khóa anchor/parent, evaluation contract, gold status và paired
case/slice delta. Lát cắt gồm modality missing, query modality, fusion conflict,
component score và provenance. Giữ regression, rescued, persistent failure,
counterexample và seeded random sample; observation không phải cause.

## 9. Package giao nhận

Tách implementation, measured result, roadmap và historical AIC evidence. Giao
architecture, data contract, evaluation protocol, lệnh tái lập, limitation và
DOCX đã render-QA. Public repo chỉ có fixture được phép. Không push/publish trước
khi rà remote, visibility và dữ liệu kèm theo.
"""


def context_block(name: str) -> str:
    """Render durable per-project memory for the new rules."""

    return """## Ngữ cảnh refine 26/08/2026

- Tài liệu canonical dùng tiếng Việt có dấu, viết hoa đầu câu, gọn và không chứa
  debug narrative; tài liệu lịch sử và General logs vẫn bất biến.
- Trước comparison phải đọc Documents/General/RULE.md, khóa anchor/parent,
  hypothesis, evaluation role, gold status, evaluation contract, slice registry
  và case budget.
- Học từ paired case/slice delta, regression, rescued, persistent failure,
  counterexample và random sample có seed; tách observation khỏi cause.
- Baseline nhỏ, data knowledge, qrels/evaluator và resource gate đi trước model.
- Trạng thái vẫn là planning/scaffold hoặc historical evidence; chưa phát sinh
  benchmark mới từ lần refine này."""


def course_knowledge_block(spec: CourseProject) -> str:
    """Render the compact course alignment section for the DOCX source."""

    return f"""# Căn chỉnh chương trình và học từ run

Project giữ đúng lõi {spec.course_code} — {spec.course_name}. Dữ kiện UIT và đối
chiếu HCMUS/HCMUT nằm tại VIETNAM-CURRICULUM-ALIGNMENT.md; đề cương giảng viên
cung cấp khi vào học có quyền ưu tiên.

Trước comparison, khóa anchor/parent, hypothesis, evaluation role, gold status,
evaluation contract, slice registry và case budget. Báo paired case/slice delta,
counterexample; tách observation khỏi cause. Run hợp lệ nhưng kém hơn là
Checkpoint REJECT|REPEAT, không phải Error."""


def course_html_block(spec: CourseProject) -> str:
    """Render a small HTML section for live course plans."""

    return f"""<section>
<h2>Căn chỉnh chương trình và học từ kết quả</h2>
<p>Project giữ đúng lõi {spec.course_code} — {spec.course_name}. Xem
<a href="VIETNAM-CURRICULUM-ALIGNMENT.md">đối chiếu UIT/HCMUS/HCMUT</a>; đề cương
giảng viên cung cấp khi vào học có quyền ưu tiên.</p>
<p>Mỗi comparison khóa anchor/parent và evaluation contract, báo paired
case/slice delta, counterexample và hypothesis
<code>supported|refuted|inconclusive</code>.</p>
</section>"""


def translated_nested_skills() -> dict[str, str]:
    """Return full Vietnamese replacements for three nested DSC2026 skills."""

    return {
        "dsc2026-legalqa-delivery": """---
name: dsc2026-legalqa-delivery
description: Thay đổi DSC2026 Task 2 LegalQA với evidence, abstention và retrieval handoff có gate.
---

# Giao nhận DSC2026 LegalQA

Đọc Documents/decision-making/0001-evidence-bound-task2-baseline.md và data/model
register trước khi sửa code. Giữ question/document/chunk ID, source, access scope,
corpus version, rank và score. Lọc scope trước evidence selection và answer/citation.

Extractive phải chạy bằng standard library; model adapter là tùy chọn và fallback
về extractive hoặc abstain. Chỉ dùng model trong policy/allowlist. Tách fixture
smoke khỏi organizer/held-out result; không suy legal, benchmark hoặc submission claim.

Model trial phải đọc experiment gate, khóa retrieval release và evaluation
contract, báo paired case/slice delta cùng counterexample. Chạy unittest và fixture
CLI sau sửa. Tài liệu viết tiếng Việt có dấu, gọn.
""",
        "dsc2026-hf-gpu-inference": """---
name: dsc2026-hf-gpu-inference
description: Chạy thử Hugging Face GPU cho DSC2026 Task 2 với secret, data và fallback gate.
---

# Hugging Face GPU inference

Đọc Documents/huggingface-gpu-inference.md trước remote call. Xác minh model
allowlist, endpoint retention và quyền gửi evidence; mặc định không gửi organizer
data. HF_TOKEN chỉ ở environment/secret store, không vào file, history artifact,
test output hoặc docs.

Dùng context bounded, timeout hữu hạn; bắt đầu synthetic fixture và một request
tuần tự. Remote mode chỉ mở bằng flag explicit; ghi endpoint type, model revision,
GPU, context, latency, cost/quota và failure. Giữ extractive fallback/abstention.
Reachability không phải model quality hoặc DSC readiness.

Khi nhập kết quả vào project history, khóa evaluation contract, paired cases/slices
và counterexample theo root Documents/General/RULE.md.
""",
        "dsc2026-experiment-memory": """---
name: dsc2026-experiment-memory
description: Ghi hoặc lấy context Task 2 mà không trộn fixture với claim chính thức.
---

# Memory thí nghiệm DSC2026

Truy hồi hẹp theo project, topic và task/release tags; docs local và chỉ dẫn BTC
mới nhất ưu tiên. Chỉ lưu input/retrieval release, model/revision, config/evaluation
contract hash, metric/failure evidence, paired case/slice delta, decision và next action.

Gắn official, measured, prior-art, decision hoặc roadmap. Fixture measured chỉ
đúng cho fixture, không phải official score. Không lưu raw organizer QA, token,
secret endpoint URL hoặc private corpus text. Dùng record schema khi tạo note mới.
""",
    }


def plan_course(repo: Path, spec: CourseProject) -> list[Change]:
    """Plan bounded updates for one course repository."""

    changes = [
        Change(repo, Path("AGENTS.md"), render_agents(spec), "refine local rules"),
        Change(
            repo,
            Path("Documents/General/RULE.md"),
            render_general_rule(spec),
            "refine immutable logging contract",
        ),
        Change(
            repo,
            Path(".agents/skills/course-project-delivery/SKILL.md"),
            render_skill(spec),
            "refine local delivery skill",
        ),
        Change(
            repo,
            Path(".agents/skills/course-project-delivery/references/gates.md"),
            render_skill_gates(),
            "refine skill gates",
        ),
        Change(
            repo,
            Path("Documents/References/VIETNAM-CURRICULUM-ALIGNMENT.md"),
            render_curriculum_alignment(spec),
            "add curriculum evidence",
        ),
        Change(
            repo,
            Path("Documents/Decision-Making/ERROR-EXPERIENCE.md"),
            render_error_experience(spec),
            "separate lessons from invalid-run logs",
        ),
        Change(
            repo,
            Path("Documents/References/COMMON-FAILURES-AND-PREVENTION.md"),
            render_common_failures(spec),
            "separate valid negative results from errors",
        ),
        Change(
            repo,
            Path("Documents/References/EXTERNAL-SERVICES-GUIDELINE.md"),
            render_services(spec),
            "refine service and inference gates",
        ),
        Change(
            repo,
            Path("Workflows/flowinfo.md"),
            render_workflow(spec),
            "align checkpoint workflow",
        ),
        Change(
            repo,
            Path("scripts/validate_project.py"),
            validator_script(spec),
            "require curriculum and active document gates",
        ),
    ]
    context = Path("Documents/Decision-Making/05-PROJECT-CONTEXT.md")
    changes.append(
        Change(
            repo,
            context,
            upsert_block(read_text(repo / context), context_block(repo.name)),
            "update durable context",
        )
    )
    knowledge = Path("Documents/References/PROJECT-KNOWLEDGE.md")
    changes.append(
        Change(
            repo,
            knowledge,
            upsert_block(render_knowledge(spec), course_knowledge_block(spec)),
            "rebuild canonical DOCX source",
        )
    )
    html_path = Path("Documents/References/PROJECT-PLAN.html")
    changes.append(
        Change(
            repo,
            html_path,
            insert_html_block(read_text(repo / html_path), course_html_block(spec)),
            "align HTML plan",
        )
    )
    readme = Path("Documents/README.md")
    changes.append(
        Change(
            repo,
            readme,
            upsert_block(
                read_text(repo / readme),
                "## Đối chiếu chương trình\n\nXem Documents/References/VIETNAM-CURRICULUM-ALIGNMENT.md. Dữ kiện chính thức và suy luận project được tách riêng.",
            ),
            "link curriculum evidence",
        )
    )
    return changes


def plan_legal_repo(repo: Path) -> list[Change]:
    """Plan updates for Evidence-Grounded or DSC root repositories."""

    changes = [
        Change(
            repo,
            Path("AGENTS.md"),
            retrieval_agents(repo.name),
            "translate and refine root rules",
        ),
        Change(
            repo,
            Path("Documents/General/RULE.md"),
            render_noncourse_rule(repo.name),
            "refine immutable logging contract",
        ),
        Change(
            repo,
            Path("skills/retrieval-delivery/SKILL.md"),
            retrieval_skill(repo.name),
            "translate root skill",
        ),
        Change(
            repo,
            Path("skills/dsc-legal-ir-delivery/SKILL.md"),
            legal_ir_skill(),
            "translate Task 1 skill",
        ),
        Change(
            repo,
            Path("skills/dsc-legal-qa-delivery/SKILL.md"),
            legal_qa_skill(),
            "translate Task 2 skill",
        ),
        Change(
            repo,
            Path("Documents/References/Task1/competition-and-prior-art.md"),
            prior_art_task1(),
            "translate Task 1 evidence map",
        ),
        Change(
            repo,
            Path("Documents/References/Task2/competition-and-prior-art.md"),
            prior_art_task2(),
            "translate Task 2 evidence map",
        ),
    ]
    if repo.name == "Evidence-Grounded-Text-Agent":
        context = Path("Documents/Decision-making/Shared/0003-project-context.md")
    else:
        context = Path("Documents/Decision-making/Shared/0002-project-context.md")
    context_text = read_text(repo / context)
    if not context_text.strip():
        context_text = f"# Ngữ cảnh dự án — {repo.name}\n"
    changes.append(
        Change(
            repo,
            context,
            upsert_block(context_text, context_block(repo.name)),
            "update durable context",
        )
    )
    standard = Path("Documents/Docs/Shared/document-standard.md")
    changes.append(
        Change(
            repo,
            standard,
            upsert_block(
                read_text(repo / standard),
                "## Chuẩn viết refine\n\nViết tiếng Việt có dấu, viết hoa đầu câu, dùng câu ngắn và nguồn gần mệnh đề. Không chép debug narrative. Tài liệu lịch sử và General logs bất biến; tạo version mới theo RULE.md.",
            ),
            "refine documentation standard",
        )
    )
    return changes


def plan_multimodal(repo: Path) -> list[Change]:
    """Plan updates for the multimodal asset repository."""

    changes = [
        Change(
            repo,
            Path("AGENTS.md"),
            multimodal_agents(),
            "translate and refine root rules",
        ),
        Change(
            repo,
            Path("Documents/General/RULE.md"),
            render_noncourse_rule(repo.name),
            "refine immutable logging contract",
        ),
        Change(
            repo,
            Path("skills/retrieval-delivery/SKILL.md"),
            multimodal_retrieval_skill(),
            "restore asset-specific skill",
        ),
        Change(
            repo,
            Path("Documents/References/steps.md"),
            multimodal_steps(),
            "translate and compact active roadmap",
        ),
    ]
    context = Path("Documents/Decision-Making/PROJECT-CONTEXT.md")
    context_text = read_text(repo / context)
    if not context_text.strip():
        context_text = f"# Ngữ cảnh dự án — {repo.name}\n"
    changes.append(
        Change(
            repo,
            context,
            upsert_block(context_text, context_block(repo.name)),
            "add durable context",
        )
    )
    for relative in (
        Path("Documents/References/aic-simple-to-development-architecture.md"),
        Path("Documents/References/simple-to-development-architecture.md"),
    ):
        text = read_text(repo / relative)
        old = (
            "Tên smoke chỉ còn là alias tương thích trong loader và luôn resolve "
            "thành simple. Tài liệu và lệnh mới phải dùng Simple/simple."
        )
        if old in text:
            text = text.replace(
                old,
                "Tên smoke chỉ còn là alias tương thích trong loader và resolve thành simple.\n\nTài liệu và lệnh mới phải dùng Simple/simple.",
            )
        text = space_markdown_lists(text)
        changes.append(Change(repo, relative, text, "compact active architecture prose"))
    return changes


def add_unrecognized_skill_blocks(changes: list[Change], repo: Path) -> None:
    """Ensure every local skill sees the new durable rules."""

    planned = {change.relative.as_posix().casefold() for change in changes}
    for skill in repo.rglob("SKILL.md"):
        if any(part in {".git", ".migration-backups", ".venv"} for part in skill.parts):
            continue
        relative = skill.relative_to(repo)
        if relative.as_posix().casefold() in planned:
            continue
        content = upsert_block(read_text(skill), skill_learning_block(repo.name))
        changes.append(Change(repo, relative, content, "refine remaining local skill"))


def build_plan(documents_root: Path) -> list[Change]:
    """Build and validate the complete migration plan."""

    root = documents_root.resolve()
    changes: list[Change] = []
    for name in REPOSITORIES:
        repo = (root / name).resolve()
        if repo.parent != root or not repo.is_dir():
            raise ValueError(f"missing or unsafe repository: {repo}")
        if name in COURSE_BY_REPO:
            repo_changes = plan_course(repo, COURSE_BY_REPO[name])
        elif name == "Multimodal-Asset-Retrieval":
            repo_changes = plan_multimodal(repo)
        else:
            repo_changes = plan_legal_repo(repo)
        if name == "DSC-TinDipLaPo":
            nested = translated_nested_skills()
            for skill_name, content in nested.items():
                relative = Path("DSC2026/skills") / skill_name / "SKILL.md"
                repo_changes.append(
                    Change(repo, relative, content, "translate nested DSC2026 skill")
                )
        add_unrecognized_skill_blocks(repo_changes, repo)
        changes.extend(repo_changes)
    validate_plan(changes)
    return changes


def validate_plan(changes: list[Change]) -> None:
    """Reject unsafe targets, duplicate writes and malformed content."""

    seen: set[tuple[Path, Path]] = set()
    for change in changes:
        key = (change.repo, change.relative)
        if key in seen:
            raise ValueError(f"duplicate target: {change.repo.name}/{change.relative}")
        seen.add(key)
        target = (change.repo / change.relative).resolve()
        if change.repo.resolve() not in target.parents:
            raise ValueError(f"target escapes repository: {target}")
        if LOG_PATTERN.fullmatch(change.relative.name):
            raise ValueError(f"immutable log targeted: {change.relative}")
        normalized = normalize_text(change.content)
        if any(marker in normalized for marker in MOJIBAKE):
            raise ValueError(f"possible mojibake in planned file: {change.relative}")
        if change.relative.suffix in {".md", ".html"} and len(normalized.strip()) < 20:
            raise ValueError(f"planned text is unexpectedly short: {change.relative}")
        if change.relative.suffix == ".html":
            lowered = normalized.casefold()
            if "<!doctype html" not in lowered or "</html>" not in lowered:
                raise ValueError(f"invalid HTML shell: {change.relative}")


def changed_only(changes: list[Change]) -> list[Change]:
    """Return writes whose normalized content differs from disk."""

    return [
        Change(item.repo, item.relative, normalize_text(item.content), item.reason)
        for item in changes
        if normalize_text(read_text(item.repo / item.relative)) != normalize_text(item.content)
    ]


def backup_repo(repo: Path, changes: list[Change]) -> Path:
    """Create a non-overwriting backup and manifest for one repository."""

    backup = repo / ".migration-backups" / MIGRATION_ID
    if backup.exists():
        raise FileExistsError(f"backup already exists: {backup}")
    original_root = backup / "original"
    original_root.mkdir(parents=True)
    entries: list[dict[str, str | None]] = []
    for change in changes:
        source = repo / change.relative
        original_hash: str | None = None
        if source.is_file():
            payload = source.read_bytes()
            original_hash = sha256_bytes(payload)
            destination = original_root / change.relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        entries.append(
            {
                "path": change.relative.as_posix(),
                "reason": change.reason,
                "original_sha256": original_hash,
                "new_sha256": sha256_bytes(normalize_text(change.content).encode("utf-8")),
            }
        )
    manifest = {
        "migration_id": MIGRATION_ID,
        "repository": repo.name,
        "files": entries,
    }
    (backup / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return backup


def atomic_write(path: Path, content: str) -> None:
    """Atomically replace one UTF-8 text file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=".refine-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(normalize_text(content))
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def apply_changes(changes: list[Change]) -> None:
    """Backup then apply changes repository by repository."""

    by_repo: dict[Path, list[Change]] = {}
    for change in changes:
        by_repo.setdefault(change.repo, []).append(change)
    for repo in by_repo:
        backup = repo / ".migration-backups" / MIGRATION_ID
        if backup.exists():
            raise FileExistsError(f"backup already exists: {backup}")
    for repo, repo_changes in by_repo.items():
        backup_repo(repo, repo_changes)
    for repo, repo_changes in by_repo.items():
        for change in repo_changes:
            atomic_write(repo / change.relative, change.content)


def parse_args() -> argparse.Namespace:
    """Parse dry-run/apply arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents-root", type=Path, default=DEFAULT_DOCUMENTS_ROOT)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Print the plan or apply it with backups."""

    args = parse_args()
    changes = changed_only(build_plan(args.documents_root))
    for change in changes:
        print(f"{change.repo.name}: {change.relative.as_posix()} — {change.reason}")
    print(f"planned_changes={len(changes)}")
    if not args.apply:
        print("dry-run; pass --apply to write")
        return 0
    apply_changes(changes)
    print("apply=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
