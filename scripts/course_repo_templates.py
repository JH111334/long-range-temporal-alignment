"""Vietnamese documentation templates for course repository scaffolds."""

from __future__ import annotations

from html import escape

from course_repo_catalog import Candidate, CourseProject

INITIAL_CHECKPOINT_DATE = "25-08-26"
INITIAL_CHECKPOINT_ISO = "2026-08-25"


def initial_checkpoint_relative(spec: CourseProject) -> str:
    """Return the deterministic v3 bootstrap checkpoint path."""
    name = f"{spec.repo_name}_Checkpoints_{INITIAL_CHECKPOINT_DATE}_V1.md"
    return f"Documents/General/{name}"


def bullets(items: tuple[str, ...]) -> str:
    """Render compact Markdown bullets."""
    return "\n".join(f"- {item}" for item in items)


def numbered(items: tuple[str, ...]) -> str:
    """Render a numbered Markdown list."""
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


def candidate_rows(candidates: tuple[Candidate, ...]) -> str:
    """Render candidate registry rows."""
    if not candidates:
        return "| Chưa chọn | — | Không cần ở baseline | Chỉ thêm khi problem contract yêu cầu |"
    return "\n".join(
        f"| [{item.name}]({item.url}) | {item.role} | {item.gate} | `hold` |" for item in candidates
    )


def render_root_readme(spec: CourseProject) -> str:
    """Render the repository entrypoint."""
    return f"""# {spec.repo_name}

Scaffold học tập–nghiên cứu cho môn **{spec.course_code} — {spec.course_name}**.

## Bài toán đã chọn

**{spec.title}.** {spec.problem}

Giá trị Việt Nam: {spec.vietnam_value}

Khả năng chuyển giao: {spec.global_transfer}

## Trạng thái bằng chứng

- Code hiện là `scaffold`: contracts, stage plan, adapter fail-closed và fixture tests.
- Chưa tải dataset/model, chưa training và chưa có benchmark `measured`.
- Registry candidate ở trạng thái `hold`; MIT của code không cấp lại quyền cho data/model.
- Nhánh mở rộng ngoài phạm vi hiện tại chỉ được lưu trong một tài liệu hoãn riêng.

## Đầu vào

{bullets(spec.input_contract)}

## Đầu ra

{bullets(spec.output_contract)}

## Luồng

`{" → ".join(spec.stages)}`

## Cấu trúc

- `Documents/Decision-Making/`: problem, kiến trúc/gates, data/4 GB, context, tool/service và quyết định tạm hoãn.
- `Documents/References/`: khảo sát, tri thức, hướng dẫn dịch vụ, lỗi thường gặp, nguồn Markdown để sinh DOCX và HTML.
- `Documents/General/`: chỉ `RULE.md`, checkpoint logs và error logs đúng grammar.
- `src/{spec.package_name}/`: core Python thuần CPU và các package theo stage.
- `src/{spec.package_name}/ServiceAdapters/`: sườn Hugging Face, Kaggle, nguồn domain và artifact store; không gọi mạng.
- `streamlit_app/`: UI chỉ đọc latest checkpoint đúng grammar và hiện version/SHA-256/evidence/warnings.
- `registries/`, `schemas/`, `configs/`: provenance và input/output contract.

## Chạy kiểm tra scaffold

```powershell
$env:PYTHONPATH = "src"
py -3.12 -m unittest discover -s tests -v
py -3.12 scripts/validate_project.py
```

`requirements.txt` là kế hoạch dependency. Chỉ cài sau khi rà phiên bản, license và ngân sách:

```powershell
py -3.12 -m pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

## Giấy phép

Code và tài liệu do repository sở hữu dùng MIT. Dataset, model, paper và dịch vụ giữ điều khoản của nguồn tương ứng.
"""


def render_documents_readme() -> str:
    """Render the Documents index."""
    return """# Documents

Chỉ `README.md` nằm ở cấp gốc.

- `Decision-Making/`: problem, architecture/gates, feasibility/data/4 GB, context, tool/service và quyết định tạm hoãn.
- `References/`: research/dataset, tri thức, service guide, lỗi thường gặp, `PROJECT-KNOWLEDGE.md` làm nguồn sinh DOCX và `PROJECT-PLAN.html`.
- `General/`: chỉ `RULE.md`, checkpoint logs và error logs đúng quy tắc tên; không chứa tri thức hay quyết định.

Mức bằng chứng: `candidate`/`paper_reported` không phải kết quả của nhóm; `scaffold` và `smoke_test` không phải benchmark; chỉ dùng `measured` khi đủ data/config/code hash, split, seed, CI và resource artifact.
"""


def render_problem_contract(spec: CourseProject) -> str:
    """Render a falsifiable problem contract."""
    return f"""# Problem Contract — {spec.course_code}

## Người dùng và quyết định

- Người dùng chính: {spec.primary_user}
- Quyết định được hỗ trợ: {spec.supported_decision}
- Vấn đề: {spec.problem}
- Giá trị Việt Nam: {spec.vietnam_value}
- Chuyển giao quốc tế: {spec.global_transfer}

## Đầu vào bắt buộc

{bullets(spec.input_contract)}

## Đầu ra bắt buộc

{bullets(spec.output_contract)}

## Ngoài phạm vi

{bullets(spec.exclusions)}

## Tiêu chí bác bỏ hoặc đổi hướng

{bullets(spec.falsification)}

## Acceptance gate đầu tiên

Fixture nhỏ chạy CPU, contract invalid phải fail-closed, split/leakage audit có test, baseline đơn giản có qrels/evaluator tái lập và report không chứa claim vượt evidence class. Chỉ sau đó mới mở data/model/service gate.
"""


def render_architecture(spec: CourseProject) -> str:
    """Render stage and hardware planning."""
    rows = "\n".join(
        f"| {index} | `{stage}` | Input stage trước | Artifact/validation của `{stage}` | Fail-closed, log reason |"
        for index, stage in enumerate(spec.stages, start=1)
    )
    return f"""# Kiến trúc, gates và phần cứng

## Stage plan

| # | Stage | Input | Output | Failure contract |
|---:|---|---|---|---|
{rows}

Luồng: `{" → ".join(spec.stages)}`.

Mỗi stage khi triển khai phải có owner, schema version, input/output hash, telemetry tối thiểu, retry/timeout giới hạn và rollback. Pipeline scaffold hiện chỉ trả kế hoạch; không tự tải, train hoặc ghi artifact.

## Hardware tiers

{numbered(spec.hardware)}

Không suy ra khả năng chạy từ số tham số. Trước khi mở tier phải đo peak RAM/VRAM, disk, wall time và batch/sequence/corpus size thực.

## Gates

1. **Problem**: use case và tiêu chí bác bỏ được duyệt.
2. **Data**: URL chính thức, ngày truy cập, terms/license, immutable revision/checksum, reviewer và data role.
3. **Split/leakage**: group/source/time policy và near-duplicate audit cố định.
4. **Baseline**: CPU baseline, evaluator parity và fixture regression đạt.
5. **Model/resource**: model card/allowlist, revision, precision, seed và resource probe đạt.
6. **Application**: artifact read-only, provenance/warning/abstention hiển thị; rollback có test.
7. **Measured**: nhiều seed khi có stochastic training, CI/slice/invalid rate và config/data/code hash đầy đủ.
"""


def render_data_model_eval(spec: CourseProject) -> str:
    """Render data, method, split, evaluation, and service choices."""
    baseline_rows = "\n".join(
        f"| B{index} | {item} | baseline |" for index, item in enumerate(spec.baselines)
    )
    candidate_rows_text = "\n".join(
        f"| C{index} | {item} | candidate; chỉ mở sau gate |"
        for index, item in enumerate(spec.candidates, start=1)
    )
    method_rows = "\n".join(filter(None, (baseline_rows, candidate_rows_text)))
    return f"""# Dữ liệu, phương pháp và đánh giá

## Dataset candidates

| Candidate | Vai trò | Gate còn thiếu | Trạng thái |
|---|---|---|---|
{candidate_rows(spec.data_candidates)}

## Model candidates

| Candidate | Vai trò | Gate còn thiếu | Trạng thái |
|---|---|---|---|
{candidate_rows(spec.model_candidates)}

Không candidate nào được tải chỉ vì có tên trong bảng. Registry JSON là nguồn trạng thái máy đọc; tài liệu này giải thích quyết định.

## Thứ tự phương pháp

| ID | Phương pháp | Mức |
|---|---|---|
{method_rows}

## Split và leakage

{spec.split_policy}

Split phải được tạo trước bước học từ dữ liệu; preprocessing/threshold/tuning chỉ fit trên train. Báo riêng overlap theo source, entity/event/template, checksum/perceptual hash hoặc text similarity phù hợp domain.

## Metrics

{bullets(spec.metrics)}

Báo per-query/per-slice cùng aggregate; nêu sample size, invalid rate, seed và 95% CI. Không tối ưu trên test, không đổi evaluator giữa baseline/candidate và không gọi smoke fixture là benchmark.
"""


def render_error_experience(spec: CourseProject) -> str:
    """Render compact lifecycle error lessons."""
    project_risk = spec.falsification[0]
    return f"""# Error Experience

File này giữ kinh nghiệm đã chắt lọc từ các project trước và quyết định phòng tránh; không phải log sự cố của repository hiện tại. Lỗi thật tạo file `Documents/General/<Project>_Error_DD-MM-YY_V<N>.md`; run hợp lệ nhưng kém hơn tạo Checkpoint `REJECT|REPEAT` theo `RULE.md`.

| Giai đoạn | Sai lầm thường gặp | Gate ngăn tái phát |
|---|---|---|
| Khảo sát | Chọn bài toán theo mô hình nổi bật thay vì người dùng/evaluator | Problem contract + tối đa một stress test phụ |
| Khả thi | Đánh giá nhu cầu Việt Nam không có khảo sát | Gắn `hypothesis`; phỏng vấn/pilot trước claim ứng dụng |
| Data discovery | Dùng trang tổng hợp làm nguồn cuối | Khám phá rộng, xác minh bằng paper/card/docs chính thức |
| Thu thập | Tải “latest”, thiếu license/checksum/reviewer | Registry gate trước mọi service call |
| Quản lý data | Trộn raw/derived, mất lineage hoặc đưa data riêng vào Git | Manifest/hash/data role + ignore check + access log |
| Split | Random split làm near-duplicate/source leak | {spec.split_policy} |
| Preprocess | Fit toàn corpus, đổi normalization giữa run | Fit train-only, version và regression fixtures |
| Training | Chạy model nặng trước baseline/resource probe | Tier gate; peak RAM/VRAM; early stop và rollback |
| Evaluation | Chỉ báo mean/best seed; parser error bị tính thành model error | Invalid rate, seed/CI/slice và evaluator parity |
| Predict | Không kiểm tra schema/access/evidence trước trả kết quả | Input/output contract và fail-closed response |
| Visualization | Dashboard thiếu denominator, lineage hoặc cảnh báo | UI chỉ đọc artifact, hiện version/hash/evidence class |
| Streamlit | Nút UI tự tải/train hoặc giữ secret trong state/log | Tách service layer, disable action mặc định, redaction test |
| Model management | Checkpoint không gắn config/data/code revision | Registry + immutable artifact + rollback pointer |
| Tài liệu | Scaffold/smoke bị viết thành kết quả đo | Claim ledger và source gần mệnh đề |

Rủi ro đặc thù đầu tiên: {project_risk}

## Ranh giới bằng chứng

Risk chung không chứng minh lỗi đã xảy ra. Không tạo error log rỗng; khi có lỗi phải phân biệt `confirmed`/`hypothesis`, ghi expected/actual, evidence, root cause, fix/rollback và regression status.
"""


def render_common_failures(spec: CourseProject) -> str:
    """Render the compact reference of critical failure patterns."""
    return f"""# Lỗi thường gặp và cách phòng tránh — {spec.course_code}

Đây là knowledge base, không phải log vận hành. Lỗi thật tạo Error; run hợp lệ nhưng kém hơn tạo Checkpoint `REJECT|REPEAT` theo `Documents/General/RULE.md`.

## Năm lỗi chí mạng

| Lỗi | Dấu hiệu | Cách chặn |
|---|---|---|
| Scope đi trước user/data | Bắt đầu bằng model, chưa có quyết định hoặc qrels | Khóa problem contract, minimum viable data và evaluator trước code nặng |
| Leakage/near-duplicate | Gain mất khi đổi source/group/time | Split theo group/source/entity/event/time, dedup trước split và audit overlap |
| Data lineage mơ hồ | Không tái dựng được record, label hoặc run | Manifest, license/terms, revision/checksum, data role và train-only fit |
| Đánh giá chọn lọc | Chỉ báo best seed/mean, đổi evaluator giữa run | Frozen test, per-query/slice, invalid rate, seed/CI và resource artifact |
| Service/UI vượt quyền | Latest revision, secret trong log, UI tự tải/train | Immutable revision, privacy/timeout/schema, fail-closed adapter và UI read-only |

## Rủi ro đặc thù

{bullets(spec.falsification)}

Chỉ mở candidate để sửa một failure slice đã đo; dừng hoặc thu hẹp khi stop condition xảy ra.
"""


def render_research(spec: CourseProject) -> str:
    """Render verified prior-art and candidate notes."""
    source_rows = "\n".join(
        f"| [{source.label}]({source.url}) | {source.use} | primary/official |"
        for source in spec.sources
    )
    return f"""# Khảo sát nghiên cứu và dữ liệu — {spec.course_code}

Cập nhật thiết kế: {INITIAL_CHECKPOINT_ISO}. Nguồn khám phá có thể dùng HyperAI/Papers with Code archive, nhưng mọi quyết định dưới đây phải liên kết về paper, repository hoặc tài liệu chính thức. Số liệu paper là `paper_reported`; URL paper không chứng minh data, code, checkpoint hay license đang sẵn dùng.

## Khoảng trống thực dụng

{spec.problem}

- Việt Nam-first: {spec.vietnam_value}
- Chuyển giao: {spec.global_transfer}
- Giả thuyết cần đo, không phải fact ứng dụng: người dùng pilot nhận được lợi ích rõ hơn so với tìm kiếm/kiểm tra thủ công ở cùng ngân sách.

## Nguồn đã chọn

| Nguồn | Dùng để quyết định | Loại |
|---|---|---|
{source_rows}

## Dataset registry view

| Candidate | Vai trò | Gate | Trạng thái |
|---|---|---|---|
{candidate_rows(spec.data_candidates)}

## Model registry view

| Candidate | Vai trò | Gate | Trạng thái |
|---|---|---|---|
{candidate_rows(spec.model_candidates)}

## Quy tắc khảo sát tiếp

1. Ghi query, ngày, nguồn khám phá và primary URL.
2. Kiểm tra task, split, qrels, metric, license/terms, release revision và known limitations.
3. So baseline/evaluator/hardware like-for-like; số liệu paper luôn gắn `paper_reported`.
4. Chỉ đưa candidate vào `approved` khi reviewer ký manifest và checksum/revision bất biến.
5. Kết quả âm tính trên protocol đúng vẫn là kết quả hợp lệ; không mở rộng scope để “cứu” giả thuyết.

## Data audit trước code

Với mỗi record phải trả lời: sinh từ đâu, ai sở hữu, schema/đơn vị nào, có thể thiếu/sai/trùng ra sao, thuộc group và split nào, nhãn được tạo thế nào, metric nào bị ảnh hưởng. Chốt minimum viable data, chi phí qrels và fallback trước khi tải corpus lớn.

## Stop rule

Dừng hoặc thu hẹp nếu quyền dữ liệu không rõ, không tạo được qrels/evaluator đáng tin, gain chỉ đến từ leakage, hoặc resource probe vượt ngân sách. Không dùng model lớn để che data contract yếu.
"""


def render_services(spec: CourseProject) -> str:
    """Render external service use guidance."""
    return f"""# Hướng dẫn gọi dịch vụ ngoài

Code nằm tại `src/{spec.package_name}/ServiceAdapters/`. Mọi `fetch()` và `execute()` đều fail-closed. Không cài package hoặc gọi mạng chỉ để thử scaffold.

## Gate chung

Trước mọi download phải có: official URL, accessed date, terms/license, immutable revision, expected checksum, reviewer, purpose và data role. Token dùng biến môi trường/secret store; không ghi vào `.env`, notebook, Git, exception hoặc Streamlit state.

Luồng: metadata review → approval → pinned plan → cache ngoài Git → checksum → content/schema validation → manifest → pipeline read-only.

## Hugging Face

Chỉ dùng `hf_hub_download`/`snapshot_download` sau gate; `revision` phải là full commit, giới hạn `allow_patterns`, ghi resolved commit/file list/SHA-256. Benchmark đóng băng dùng cache và `local_files_only=True`. Nguồn: [official download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download).

```python
# Ví dụ thiết kế, không chạy trong scaffold.
snapshot_download(repo_id=repo_id, revision=full_commit, cache_dir=cache_dir)
```

## Hugging Face Inference

Chỉ cân nhắc remote inference khi local CPU/RTX 2050 4 GB probe không đáp ứng và dữ liệu có `privacy_class=public`. Request phải khóa `task`, model revision, provider/runtime revision, input/output schema và timeout hữu hạn; endpoint không pin được revision chỉ dùng cho `exploratory`, không dùng làm benchmark.

Luồng triển khai: xác nhận failure slice → so local fallback → duyệt privacy/cost/quota → tạo plan bất biến → mock timeout/429/schema drift → failure injection → ADR mở execution → lưu response metadata/hash. Không retry lỗi 4xx, không gửi dữ liệu `internal/restricted`, không log token hoặc payload nhạy cảm.

## Kaggle

Dùng handle `owner/dataset/versions/n`, không dùng latest cho run đo; auth qua `KAGGLE_API_TOKEN`. Kiểm tra terms, archive path, file list, size, checksum và schema sau tải. Nguồn: [KaggleHub chính thức](https://github.com/Kaggle/kagglehub).

```python
# Ví dụ thiết kế, không chạy trong scaffold.
kagglehub.dataset_download("owner/dataset/versions/1", output_dir=cache_dir)
```

## Nguồn domain

Adapter `{spec.domain_adapter_class}` dùng provider `{spec.domain_adapter_provider}`. Nó chỉ lập kế hoạch cho URL đã duyệt; không crawl hàng loạt, không vượt robots/rate limit và không suy ra quyền sử dụng từ khả năng truy cập API.

## Artifact store và ứng dụng

Artifact store phải immutable theo ID/hash. Streamlit chỉ đọc artifact đã duyệt; không download/train/inference/upload. Khi triển khai thật cần timeout, retry có giới hạn, idempotency, quota, redaction, mock tests, failure injection và rollback ADR.
"""


def render_project_context(spec: CourseProject) -> str:
    """Render course logistics and the decision-facing project pitch."""
    return f"""# Project Context — {spec.course_code}

## Bối cảnh

- Môn: `{spec.course_code}` — {spec.course_name}
- Project: **{spec.title}**
- Mức bằng chứng: `planning/scaffold`; chưa có benchmark hoặc dataset/model được duyệt.

{bullets(spec.schedule)}

## Pitch 30 giây

Xây một baseline tái lập và data/evaluation protocol cho **{spec.title}** trong ngân sách sinh viên. Đóng góp ưu tiên là hiểu dữ liệu, chống leakage, trả kết quả có provenance và biết dừng; candidate chỉ được mở để sửa một failure slice đã đo.

## Người dùng và quyết định

- Người dùng: {spec.primary_user}
- Quyết định hỗ trợ: {spec.supported_decision}
- Bài toán: {spec.problem}
- Giá trị Việt Nam: {spec.vietnam_value}
- Chuyển giao: {spec.global_transfer}

## Phạm vi giao nhận

Baseline CPU, data contract, qrels/evaluator, leakage audit, error analysis và demo đọc artifact là đường găng. Thay đổi lịch hoặc scope phải cập nhật tài liệu quyết định và tạo checkpoint mới; không sửa log cũ.

Đối chiếu chương trình và ranh giới môn học nằm tại `Documents/References/VIETNAM-CURRICULUM-ALIGNMENT.md`. Quy tắc học từ run nằm tại `Documents/General/RULE.md`; không suy nguyên nhân từ metric tổng hoặc một ví dụ đẹp.
"""


def render_feasibility(spec: CourseProject) -> str:
    """Render data-first feasibility and the four-gigabyte resource budget."""
    return f"""# Khả thi, dữ liệu và ngân sách 4 GB — {spec.course_code}

## Kết luận hiện tại

Bài toán **khả thi có điều kiện**. Điểm nghẽn phải kiểm chứng là quyền dữ liệu, qrels/evaluator, leakage và thời gian nhóm; mô hình lớn không thay thế các gate này. Registry còn `hold`, vì vậy chưa được suy ra asset đang sẵn dùng.

## Minimum viable data

{bullets(spec.input_contract)}

## Data dossier bắt buộc

| Nhóm | Trường phải có |
|---|---|
| Nguồn | official URL, accessed date, owner, terms/license, revision/snapshot, checksum |
| Nội dung | schema, size, language/domain, missingness, duplicate, bias, sensitive fields |
| Nhãn | annotation guide, qrels format, disagreement, invalid/unknown policy |
| Split | source/entity/event/time group, near-duplicate audit, train-only fit |
| Chi phí | download/storage, labeling hours, CPU/GPU time, cache/retention |
| Quyết định | data role, reviewer, status, fallback và stop condition |

## RTX 2050 4 GB và lộ trình phần cứng

Profile máy đã đo ngày 25/08/2026: `NVIDIA GeForce RTX 2050` với `4096 MiB VRAM`, `Intel Core i5-12450H` và `16,866,684,928 bytes RAM`. Đây là bằng chứng cấu hình host, không chứng minh model cụ thể chạy vừa; ưu tiên fixture/baseline CPU và vẫn đo peak cho từng probe GPU.

{numbered(spec.hardware)}

Mỗi probe ghi input/corpus size, batch/sequence, precision, p50/p95 latency, peak RAM/VRAM, disk và wall time. Không suy khả năng chạy từ số tham số hoặc model card.

## Go/no-go và stop conditions

Go khi corpus hợp pháp, qrels tối thiểu, split chống leakage, evaluator parity và baseline CPU đều đạt. Thu hẹp hoặc dừng khi:

{bullets(spec.falsification)}
"""


def render_curriculum_alignment(spec: CourseProject) -> str:
    """Render a compact, source-bound VNU-HCM curriculum alignment."""
    return f"""# Đối Chiếu Chương Trình Đào Tạo Việt Nam — {spec.course_code}

Ngày rà soát: 26/08/2026. Tài liệu phân biệt dữ kiện chính thức và suy luận thiết kế; không thay đề cương do giảng viên cung cấp khi vào học.

## Dữ kiện chính thức

{bullets(spec.curriculum_facts)}

## Suy luận thiết kế

{bullets(spec.curriculum_inferences)}

## Ranh giới không mở rộng

{bullets(spec.curriculum_exclusions)}

Các project chỉ chia sẻ data contract, provenance, logging, evaluation và resource gate. Không trộn model, requirements hoặc domain logic giữa môn.

## Giới hạn bằng chứng

{spec.curriculum_evidence_limit}

## Nguồn

{bullets(tuple(f"[{source.label}]({source.url}) — {source.use}" for source in spec.curriculum_sources))}
"""


def render_general_rule(spec: CourseProject) -> str:
    """Render the only mutable rule document permitted in General."""
    slices = {
        "CS406": "source, resolution, aspect ratio, lighting, occlusion, object size, near-duplicate và query type",
        "DS111": "source, time, group, missingness, duplicate, outlier, class/target distribution",
        "DS201": "granularity, distance, entity/event, source, sequence length, hard-negative và ambiguity",
        "DS314": "intent, lexical–semantic gap, length, domain, judged coverage, hard-negative, citation và abstention",
    }[spec.course_code]
    return f"""# Quy tắc `Documents/General`

## Phạm vi

`General` chỉ chứa `RULE.md` và log bất biến. Tri thức, quyết định, hướng dẫn, lịch môn và báo cáo không đặt ở đây. Không sửa log cũ để làm đẹp kết quả; tạo version mới.

## Quy tắc tên file

- `{spec.repo_name}_Checkpoints_DD-MM-YY_V<N>.md`
- `{spec.repo_name}_Error_DD-MM-YY_V<N>.md`

Windows không cho phép `/` trong tên file nên ngày `DD/MM/YY` được biểu diễn bằng `DD-MM-YY`. Với cùng ngày và cùng loại, version bắt đầu `V1` và tăng liên tục.

## Checkpoint log

Ghi mục tiêu, lệnh thật, exit code, cấu hình hệ thống, config/data/model/code revision, evidence class, artifact/hash, kết quả và next gate. Lệnh chưa chạy ghi `not-run`; scaffold/smoke không phải benchmark.

Với run so sánh, khóa `anchor_checkpoint`, `parent_checkpoint`, `hypothesis_id`, `evaluation_role`, `gold_status` và `evaluation_contract_hash`. Contract gồm data manifest, split, gold/qrels, evaluator, metric, parser và slice registry. Khác contract hoặc đổi nhiều biến không tách được phải ghi `comparison=confounded`.

Báo paired delta theo case/query/group so với anchor và parent: overall, slice, denominator, seed/CI và `wins|ties|losses`. Artifact chi tiết giữ old/new output/rank/score, slice ID, gold/proxy, invalid/abstention và error category; log chỉ lưu URI/hash.

## Error log

Chỉ ghi run không hợp lệ hoặc quality gate định trước chuyển `FAIL`. Ghi checkpoint liên quan, first-bad/last-known-good, invariant, expected/actual, ca/lát cắt, reproducer, evidence ủng hộ/phản bác nguyên nhân, fix/rollback và regression status. Run hợp lệ nhưng metric kém hơn là Checkpoint `decision=REJECT|REPEAT`, không phải Error. Risk chung thuộc `References`.

## Học từ hành vi dữ liệu

Khóa trước case budget và quy tắc chọn: regression lớn nhất, ca được cứu, lỗi dai dẳng, low-margin/high-variability, counterexample và mẫu random có seed trên lát cắt trọng yếu. Slice phải có định nghĩa, revision, count và metadata; slice post-hoc là exploratory cho tới khi được xác nhận ở run sau.

Slice khởi điểm của project: {slices}. Đây là registry ứng viên; chỉ giữ slice có định nghĩa và coverage đủ.

Tách `observation` khỏi `cause`. Kết luận hypothesis bằng `supported|refuted|inconclusive`, kèm dự đoán có thể bác bỏ, minimal next experiment và stop/rollback gate. Nếu `gold_status=partial`, ghi judged coverage/unjudged policy; nếu `absent`, chỉ báo quality, stability, behavioral test hoặc human audit, không tuyên bố effectiveness.

## Version và bất biến

Không ghi đè log. Mọi metric phải nêu denominator/split/seed/CI khi phù hợp và gắn config hash, data-manifest hash, code revision, resource measurement cùng evidence class.

## Bảo mật

Không log token, secret, dữ liệu hạn chế hoặc payload nhạy cảm.

## Ngôn ngữ

Viết tiếng Việt có dấu, viết hoa đầu câu và dùng câu ngắn. Không chép debug narrative; chỉ liên kết artifact đã lọc và có hash.
"""


def render_checkpoint(spec: CourseProject) -> str:
    """Render an honest initial checkpoint without simulated execution."""
    return f"""# Checkpoint — khởi tạo scaffold v3

## Định danh

- ngày: `{INITIAL_CHECKPOINT_ISO}`
- project: `{spec.repo_name}`
- Phiên bản: V1
- evidence class: `scaffold`; không phải benchmark.

## Cấu hình hệ thống

- OS/Python/Git revision của lần bootstrap: `not-captured-by-deterministic-generator`
- GPU đã đo 25/08/2026: `NVIDIA GeForce RTX 2050`, `4096 MiB VRAM`
- CPU: `Intel Core i5-12450H`
- RAM: `16,866,684,928 bytes`
- model-specific resource probe: `not-run`; từng candidate vẫn phải đo peak RAM/VRAM/latency.

## Lệnh và cấu hình chạy

- Entry point: `py -3.12 scripts/bootstrap_course_repositories.py --output-root <new-directory>`
- Network/package install/data/model download: `disabled`
- Unit tests và validator trong repository đích: `not-run`; phải chạy sau khi sinh.

## Kết quả

Scaffold v3 khai báo contracts, tài liệu data-first, ServiceAdapters fail-closed và Streamlit read-only. Dataset/model thật, training và benchmark: `none`.

## Cảnh báo

- Chưa có benchmark hoặc dataset/model được duyệt.
- Unit tests, validator, DOCX QA và model-specific resource probe còn `not-run`.

## Artifact và lineage

- config/data/model/code hash: `not-computed`
- checkpoint: `{initial_checkpoint_relative(spec)}`
- tài liệu nguồn DOCX: `Documents/References/PROJECT-KNOWLEDGE.md`

## Việc tiếp theo

Chạy unit tests và validator, sinh/QA DOCX, chốt data dossier/evaluator và ghi kết quả thật vào checkpoint version mới. Chỉ mở data/model/inference gate sau khi registry, privacy, resource và rollback đầy đủ.
"""


def render_tools_decision(spec: CourseProject) -> str:
    """Render project-scoped tool and service decisions."""
    return f"""# Công cụ, MCP và service — {spec.course_code}

`.codex/config.toml` bật `filesystem`, `memory_keeper`, `playwright`; tắt `docker`, `github`, `powerpoint`, `socialcrawl`, `winston_ai`, `youtube_knowledge` ở cấp project. Không sửa cấu hình toàn cục.

- Research: web + primary/official sources; công cụ khám phá không thay xác minh nguồn gốc.
- Code: filesystem/terminal, unittest/pytest/Ruff; core không gọi mạng.
- Streamlit: `st.testing.v1.AppTest` cho app contract, Playwright cho browser smoke.
- DOCX: Markdown → Pandoc → LibreOffice PDF → PyMuPDF inspection.
- Data/model/inference: chỉ qua `ServiceAdapters`, immutable revision, privacy/schema/timeout gate và mặc định fail-closed.
- GitHub: `git`/`gh`, kiểm tra private visibility và remote SHA khi được ủy quyền.

Cấu hình/runtime đã chạy thuộc checkpoint log; lỗi tool thật thuộc error log. Hoàn tác MCP ở project-scope rồi khởi động phiên Codex mới. Nguồn: [Codex config](https://developers.openai.com/codex/config-reference), [Streamlit AppTest](https://docs.streamlit.io/develop/concepts/app-testing).
"""


def render_deferred() -> str:
    """Render the sole deferred scope note permitted to mention the branch."""
    return """# Hướng multimodal tạm hoãn

Nhánh này không thuộc scope học kỳ hiện tại và không được đưa vào requirements, source pipeline, metric, báo cáo chính hoặc UI. File chỉ ghi quyết định hoãn để các phiên sau không vô tình mở rộng phạm vi.

Chỉ xem xét multimodal sau khi baseline chính có dataset hợp pháp, evaluator cố định, kết quả held-out, resource budget và ADR riêng. Khi đó phải chứng minh modality mới giải quyết lỗi cụ thể, không chỉ làm kiến trúc phức tạp hơn.
"""


def render_knowledge(spec: CourseProject) -> str:
    """Render the maintainable source for the DOCX knowledge document."""
    return f"""---
title: "{spec.course_code} — {spec.title}"
author: "JH111334"
date: "26/08/2026"
lang: vi-VN
evidence: planning/scaffold
---

# Tóm tắt điều hành

Project chọn bài toán **{spec.title}**. Mục tiêu học kỳ là tạo baseline tái lập, protocol dữ liệu/đánh giá và error analysis; không chạy theo kiến trúc nặng khi chưa có qrels và leakage audit. {spec.problem}

Giá trị tại Việt Nam là một giả thuyết ứng dụng cần pilot: {spec.vietnam_value} Khả năng chuyển giao quốc tế đến từ hợp đồng và evaluation protocol: {spec.global_transfer}

# Đối chiếu chương trình

Project giữ đúng lõi học phần `{spec.course_code} — {spec.course_name}`. Dữ kiện UIT và đối chiếu HCMUS/HCMUT được tách khỏi suy luận thiết kế tại `VIETNAM-CURRICULUM-ALIGNMENT.md`; đề cương do giảng viên cung cấp khi vào học có quyền ưu tiên.

# Đầu vào, đầu ra và phạm vi

## Đầu vào

{bullets(spec.input_contract)}

## Đầu ra

{bullets(spec.output_contract)}

## Không làm trong baseline

{bullets(spec.exclusions)}

# Quy trình kỹ thuật

`{" → ".join(spec.stages)}`

Mỗi stage nhận artifact có version/hash, kiểm tra schema và lỗi đóng, rồi tạo artifact kế tiếp. Data/service/model adapters không được nằm trong core; core chạy CPU và chỉ phụ thuộc chuẩn Python. Streamlit là lớp trình bày read-only, không sở hữu training hoặc download.

# Data-first: dữ liệu và mô hình

## Dataset candidates

| Candidate | Vai trò | Gate | Trạng thái |
|---|---|---|---|
{candidate_rows(spec.data_candidates)}

## Model candidates

| Candidate | Vai trò | Gate | Trạng thái |
|---|---|---|---|
{candidate_rows(spec.model_candidates)}

Không asset nào được xem là có sẵn cho đến khi registry đủ official URL, accessed date, license/terms, immutable revision/checksum, reviewer và data role.

# Phương pháp và đánh giá

Baseline theo thứ tự:

{numbered(spec.baselines)}

Candidate sau gate:

{numbered(spec.candidates)}

Split: {spec.split_policy}

Metrics:

{bullets(spec.metrics)}

Mọi kết quả stochastic cần nhiều seed và 95% CI; luôn báo invalid rate, slice và latency/resource. Scaffold/smoke không phải benchmark.

Mỗi comparison khóa anchor/parent, hypothesis, evaluation role, gold status và evaluation contract hash. Báo paired delta theo case/query/group và lát cắt; tách observation khỏi cause, giữ counterexample và minimal next experiment.

# Phần cứng và ServiceAdapters

{numbered(spec.hardware)}

Hugging Face và Kaggle chỉ đi qua `ServiceAdapters`, dùng revision/version bất biến, cache ngoài Git, checksum và secret redaction. Inference plan phải khóa task, model/provider revision, privacy class, input/output schema và timeout; execution mặc định fail-closed. Nguồn domain cũng phải có terms/license và rate-limit contract. Không cài dependency trong giai đoạn sinh scaffold.

# Lỗi chí mạng, rủi ro và stop conditions

{bullets(spec.falsification)}

Nếu một stop condition xảy ra, giữ kết quả âm tính, rollback candidate và không mở rộng scope để hợp thức hóa chi phí đã bỏ ra.

# Lộ trình 12 tuần

1. Tuần 1–2: problem interview, source/license audit, schemas và fixtures.
2. Tuần 3–4: preprocessing, group/source/time split và leakage tests.
3. Tuần 5–6: baseline CPU + evaluator parity + run artifacts.
4. Tuần 7–8: error analysis/slices; chỉ mở frozen candidate nếu baseline gate đạt.
5. Tuần 9–10: ablation, seed/CI, resource/latency và failure injection.
6. Tuần 11: Streamlit read-only, DOCX/HTML, reproducibility replay.
7. Tuần 12: freeze registry, limitations, demo và bàn giao.

# Nguồn chính

{bullets(tuple(f"[{source.label}]({source.url}) — {source.use}" for source in spec.sources))}
"""


def render_html(spec: CourseProject) -> str:
    """Render a self-contained, grayscale project plan."""
    stage_rows = "".join(
        f"<tr><td>{index}</td><td><code>{escape(stage)}</code></td><td>Schema, validation, artifact hash</td></tr>"
        for index, stage in enumerate(spec.stages, start=1)
    )
    source_items = "".join(
        f'<li><a href="{escape(source.url)}">{escape(source.label)}</a>: {escape(source.use)}</li>'
        for source in spec.sources
    )
    metric_items = "".join(f"<li>{escape(item)}</li>" for item in spec.metrics)
    risk_items = "".join(f"<li>{escape(item)}</li>" for item in spec.falsification)
    return f"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(spec.course_code)} — Kế hoạch project</title>
<style>
:root {{ font-family: Arial, sans-serif; color: #111; background: #fff; }}
body {{ max-width: 980px; margin: 28px auto; padding: 0 24px; line-height: 1.5; }}
h1, h2 {{ line-height: 1.2; }}
.meta, .notice {{ border: 1px solid #777; padding: 12px; margin: 14px 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
th, td {{ border: 1px solid #777; padding: 7px; text-align: left; vertical-align: top; }}
code {{ overflow-wrap: anywhere; }}
@media print {{ body {{ margin: 0; max-width: none; }} a {{ color: #111; }} }}
</style>
</head>
<body>
<h1>{escape(spec.course_code)} — {escape(spec.title)}</h1>
<div class="meta"><strong>Môn:</strong> {escape(spec.course_name)}<br><strong>Cập nhật:</strong> 25/08/2026<br><strong>Mức bằng chứng:</strong> scaffold/planning; chưa có benchmark.<br><strong>Profile máy đã đo:</strong> RTX 2050 4096 MiB, i5-12450H, RAM 16,866,684,928 bytes; từng probe phải đo peak riêng.</div>
<h2>Vấn đề và tính hữu ích</h2>
<p>{escape(spec.problem)}</p>
<p><strong>Việt Nam:</strong> {escape(spec.vietnam_value)}</p>
<p><strong>Chuyển giao:</strong> {escape(spec.global_transfer)}</p>
<p><strong>Ranh giới môn học:</strong> giữ lõi {escape(spec.course_code)}; xem <code>VIETNAM-CURRICULUM-ALIGNMENT.md</code>. Dữ kiện chương trình và suy luận project được tách riêng.</p>
<h2>Luồng hệ thống</h2>
<table><thead><tr><th>#</th><th>Stage</th><th>Output tối thiểu</th></tr></thead><tbody>{stage_rows}</tbody></table>
<h2>Đánh giá</h2><ul>{metric_items}</ul>
<p>Mỗi comparison khóa anchor/parent và evaluation contract; báo paired case/slice delta, counterexample và hypothesis <code>supported|refuted|inconclusive</code>.</p>
<h2>Rủi ro/tiêu chí dừng</h2><ul>{risk_items}</ul>
<div class="notice"><strong>Data/model/service gate:</strong> không tải asset nếu thiếu official URL, ngày truy cập, terms/license, immutable revision/checksum, reviewer và data role. Inference cần privacy/task/schema/timeout; Streamlit chỉ đọc artifact đã duyệt.</div>
<h2>Lộ trình</h2>
<ol><li>Problem/data/license/evaluator.</li><li>Fixture, split và leakage.</li><li>CPU baseline.</li><li>Error analysis và candidate gate.</li><li>Ablation, resource, UI read-only và giao nhận.</li></ol>
<h2>Nguồn chính</h2><ul>{source_items}</ul>
</body>
</html>
"""


def render_workflow(spec: CourseProject) -> str:
    """Render workflow instructions."""
    return f"""# Workflow

1. Đọc `AGENTS.md`, local skill và problem contract.
2. Kiểm kê Git/data/model/service state; bảo toàn thay đổi ngoài phạm vi.
3. Chọn đúng một stage trong `{" → ".join(spec.stages)}`.
4. Viết/đổi contract và test trước implementation.
5. Không mở service/model gate nếu registry hoặc resource probe chưa đạt.
6. Chạy unit tests và `scripts/validate_project.py`.
7. Cập nhật registry/evidence class và tạo checkpoint version mới; chỉ tạo Error khi run không hợp lệ hoặc quality gate định trước `FAIL`.

Definition of done không cho phép dùng scaffold, fixture, static check hoặc smoke test để tuyên bố benchmark/production readiness.
"""


def render_agents(spec: CourseProject) -> str:
    """Render repository-local operating rules."""
    return f"""# Quy ước project {spec.course_code}

- Bắt đầu bằng baseline tái lập CPU; không gọi scaffold/smoke/synthetic là kết quả nghiên cứu.
- Core Python không gọi mạng. Hub/Kaggle/domain/inference chỉ qua `ServiceAdapters` và manifest/privacy/schema gate.
- Không tải data/model thật nếu thiếu official URL, accessed date, license/terms, immutable revision/checksum, reviewer và data role.
- Split theo group/source/entity/event/time phù hợp; audit near-duplicate; không random split khi có leakage risk.
- Fit tokenizer, scaler, vocabulary và preprocessing học từ dữ liệu chỉ trên train; tái dùng artifact bất biến cho validation/test.
- Kết quả phải có config hash, data manifest hash, code revision, seed/CI, invalid rate, resource và evidence class.
- RTX 2050 4 GB là trần local; đo peak trước candidate, không suy khả năng chạy từ model size.
- Source code có type hints, docstring ngắn, hàm nhỏ và fail-closed. Đổi encoder/objective/split/schema phải có test.
- Không dùng emoji, sticker, banner ký hiệu hoặc thông báo ăn mừng trong mã, test, log hay CLI; chỉ giữ ký hiệu có nghĩa kỹ thuật hoặc dữ liệu protocol đã ghi rõ.
- Đổi provider hoặc ServiceAdapter phải có contract test, timeout/failure test và không làm core phụ thuộc mạng.
- `Documents/General` chỉ có `RULE.md` và log đúng tên; không sửa log cũ. Run hợp lệ nhưng kém hơn là Checkpoint `REJECT|REPEAT`; Error chỉ dành cho run không hợp lệ hoặc quality gate `FAIL`.
- Trước comparison, khóa anchor, parent, hypothesis, evaluation role, gold status và evaluation contract hash; báo paired case/slice delta, counterexample và minimal next experiment.
- Streamlit chỉ đọc latest checkpoint đúng grammar, hiện version/SHA-256/evidence/warnings; không download/train/inference.
- Nội dung project viết tiếng Việt có dấu, viết hoa đầu câu và format gọn; tiếng Anh chỉ cho thuật ngữ, API, identifier hoặc nguồn phù hợp. Không chép debug narrative vào tài liệu canonical.
- Tích hợp bối cảnh Việt Nam vào problem–data–method–evaluation–application bằng evidence; không tách thành nhánh trang trí hoặc bonus chưa qua gate.
- Nhánh ngoài phạm vi chỉ được ghi trong file hoãn riêng, không đi vào source/requirements/report chính.
- Giữ data, model, cache, artifact lớn và secret ngoài Git.
- Trước commit chạy `$env:PYTHONPATH='src'; py -3.12 -m unittest discover -s tests -v` và `py -3.12 scripts/validate_project.py`.
"""


def render_skill(spec: CourseProject) -> str:
    """Render a reusable local skill."""
    return f"""---
name: course-project-delivery
description: Hoạch định và triển khai {spec.course_code} theo data-first, baseline CPU, log bất biến, service/inference gate và bằng chứng tái lập. Dùng khi sửa docs, schema, pipeline, evaluation, Streamlit hoặc ServiceAdapters trong repository này.
---

# Course Project Delivery

## Workflow bắt buộc

1. Đọc `AGENTS.md`, `references/gates.md`, `Documents/General/RULE.md` và problem contract.

2. Xác định user decision, stage, schemas, evidence class, resource budget và stop condition.

   Giá trị tại Việt Nam phải nối với data source, method, evaluation slice và use case trong cùng lập luận; không tạo nhánh ứng dụng riêng thiếu evidence. Chỉ mở bonus khi data/technique gate cho phép.

3. Hoàn thành data dossier, qrels/split/leakage và evaluator trước model/service nặng.

4. Dùng fixture + baseline CPU; RTX 2050 4 GB chỉ mở sau resource probe.

5. Chặn download/inference nếu registry, privacy, revision, timeout hoặc schema gate thiếu.

6. Tách `paper_reported`, `proposed`, `scaffold`, `smoke_test` và `measured`.

7. `ServiceAdapters/inference.py` giữ execution fail-closed đến khi ADR + mocks + rollback đạt.

8. UI chỉ đọc latest checkpoint đúng grammar, hiện version/SHA-256/evidence/warnings; không train/download/inference.

9. Trước comparison, khóa anchor/parent, hypothesis, evaluation role, gold status, evaluation contract, case budget và slice registry. Báo paired delta/counterexample; tách observation khỏi cause.

10. Ghi run hợp lệ nhưng kém hơn vào Checkpoint `REJECT|REPEAT`; Error chỉ dành cho run không hợp lệ hoặc quality gate `FAIL`.

11. Chạy unit tests và validator; dừng khi acceptance gate đạt, không mở rộng scaffold.

## Ngôn ngữ và giao nhận

Viết nội dung do project sở hữu bằng tiếng Việt có dấu, viết hoa đầu câu, ngắn gọn và chuyên nghiệp; không chép debug narrative. Quyết định vào `Documents/Decision-Making`, nguồn/DOCX/HTML vào `Documents/References`; `Documents/General` chỉ chứa `RULE.md` và log đúng grammar. Chỉ file hoãn riêng được mô tả nhánh ngoài scope.
"""


def render_skill_gates() -> str:
    """Render skill gates reference."""
    return """# Gates

| Gate | Điều kiện tối thiểu |
|---|---|
| Problem | User, decision, input, output, metric, budget, stop condition |
| Feasibility | Pilot, qrels cost, minimum data, fallback và go/no-go |
| Data | Official URL, date, terms/license, revision/checksum, reviewer, role |
| Split | Group/source/time policy + near-duplicate audit |
| Baseline | CPU, fixture, evaluator parity, artifact hash |
| Model | Card/allowlist, immutable revision, resource probe, rollback |
| Resource | RTX 2050 4 GB probe: input/batch/precision/latency/RAM/VRAM/disk |
| Service | Privacy, task, schemas, immutable model/provider revisions, timeout, mock, rollback |
| Measured | Anchor/parent, evaluation contract, paired case/slice delta, seed/CI/invalid/resource |
| Application | Read-only artifact, provenance, warning/abstention, failure tests |
| Log | RULE-compliant filename, gold status, case budget/counterexample, evidence class và no-secret |
"""
