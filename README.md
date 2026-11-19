# Long-Range Temporal Alignment

Scaffold nghiên cứu cho **hỏi đáp lịch sử Việt Nam có trục thời gian, dẫn chứng và khả năng từ chối**. Ticktack (arXiv:2503.04150) là baseline theo năm; hướng chính mở độ hạt sang tháng/ngày, còn point/interval là kiểu cấu trúc thời gian tách biệt.

## Khung Mục Tiêu Nghiên Cứu Ba Tầng (Goal Ladder)

Theo chuẩn hóa quản trị ngày 05/11/2026 tại [ADR 23: Goal Ladder and Acceptance Criteria](Documents/Decision-Making/23_GOAL_LADDER_AND_ACCEPTANCE_20261105.md) và [Bản đồ ánh xạ mục tiêu (Crosswalk)](Documents/Global/GOAL_BRANCH_STAGE_CROSSWALK.md):
- **GOAL 1 (`G1`) — YEAR-only Closed-Answer Temporal QA (Phạm vi hiện hành)**:
  - Cải thiện đồng thời năng lực nhận diện mốc năm lịch sử (BCE/CE) và suy luận trật tự trước-sau.
  - Định dạng: Trắc nghiệm 4 lựa chọn đối kháng (`A/B/C/D`) và nhị phân (`Yes/No`).
  - Tiêu chuẩn nghiệm thu cốt lõi: Độ chính xác nội dung câu trả lời đóng (`Canonical Accuracy`, `Debiased Mean`, `Strict Accuracy 4/4 và 2/2`, `Counterfactuals Both-Correct`, `Retention`). Các thay đổi logit shift hay khoảng cách biểu diễn chỉ là phép chẩn đoán cơ chế bổ trợ (Diagnostic Only).
  - Trạng thái: `ACTIVE_CURRENT_SCOPE` (5 epochs Stage 03 đạt mức `METRIC_ARITHMETIC_ONLY`, đang giữ lệnh `QUARANTINED_PENDING_PROVENANCE` chờ Replay GPU đối chứng).
- **GOAL 2 (`G2`) — YEAR + MONTH Closed-Answer Temporal QA (Mở rộng tiếp theo)**:
  - Mở rộng độ hạt thời gian sang hai cấp độ kết hợp: Năm và Tháng.
  - Ràng buộc: Mặt nạ phân giải độ hạt, nhất quán xuyên cấp độ (tháng $\subset$ năm), triệt tiêu hoàn toàn ảo giác tháng.
  - Trạng thái: `NEXT_SCOPED_GOAL / NOT_YET_MEASURED`.
- **GOAL 3 (`G3`) — General Temporal QA (Mục tiêu tối thượng)**:
  - Hợp nhất năng lực nhận diện, truy xuất bằng chứng và suy luận trật tự để giải quyết toàn diện hỏi đáp lịch sử Việt Nam (Closed QA + Open-Ended QA có dẫn chứng + Khả năng từ chối trả lời khi thiếu căn cứ).
  - Trạng thái: `ULTIMATE_RESEARCH_GOAL / TARGET_NOT_ACHIEVED`.

## Trạng thái

- Đây là scaffold và smoke baseline, chưa phải kết quả benchmark hay tuyên bố SOTA.
- Dữ liệu trong `data/synthetic/` là dữ liệu tổng hợp dùng kiểm thử hợp đồng.
- TempLS chưa có bản phát hành chính thức được xác minh; không dùng số liệu mô phỏng để thay thế benchmark này.
- Mọi dataset thật phải qua manifest nguồn, ngày truy cập, điều khoản, checksum, người rà soát và vai trò split.
- VLSP 2025, nguồn lịch sử Việt Nam và HistoryBankQA hiện đều là `hold/candidate`; repository chưa ingest chúng.
- Các package con mới là scaffold contract, chưa có backend retrieval, model training hoặc ứng dụng production.

## Câu hỏi nghiên cứu

1. Biểu diễn precision-aware đơn điệu–đa chu kỳ cho `year|month|day` có giảm lỗi thời gian và lỗi chọn evidence so với scalar, Fourier và Ticktack-year trên cùng evaluator không?
2. Lexical retrieval kết hợp interval filter/rerank có cải thiện evidence retrieval trên split source/entity/event/time-held-out trước khi thêm model học được không?
3. Bonus tokenizer và chuẩn hóa biểu thức thời gian Việt có sửa failure slice đã đo mà vẫn giữ provenance, uncertainty và abstention không?

## Luồng hệ thống

`Khảo sát nguồn → Evidence gate → Chuẩn hóa → Quality/dedup → Split → Retrieval → Representation → Reasoning → Evaluation → Registry → Answer/abstention`

Các hợp đồng nền nằm trong `src/temporal_alignment/contracts.py`; scaffold vòng đời nằm trong các package `data_management/`, `normalization/`, `retrieval/`, `model_management/`, `workflow/` và `application/`. Pipeline smoke không tải model hoặc dataset từ mạng.

## Compute theo capability

| Profile | Mục tiêu | Ràng buộc |
|---|---|---|
| `local_validation` | Contract, debug, unit/integration, sample preprocessing, small retrieval/index | Bounded data, batch, workers và duration |
| `external_gpu` | Full training, large embedding/index hoặc GPU-heavy probe | Approved public/licensed input và immutable request |
| `full_experiment` | Multi-seed confirmation hoặc paper-scale reproduction | Frozen evaluator, budget, provenance và rollback |

Cả ba profile dùng cùng source package, schema, artifact và evaluator; khác ở scale,
duration và compute. Máy hiện tại là một `local_validation` host đã đo, không phải giới
hạn architecture. Qwen3-0.6B và Qwen2.5-0.5B-Instruct vẫn chỉ là candidate chưa khóa
revision hoặc resource profile.

`SourceAPI/` chuẩn hóa external temporal evidence trước preprocessing và giữ cả raw
temporal representation lẫn canonical interval nếu có. Chưa có provider thật được bật.

## Chạy nhanh

```powershell
$env:PYTHONPATH = "src"
py -3.12 -m temporal_alignment --config configs/smoke.json --output artifacts/smoke_run.json
py -3.12 -m unittest discover -s tests -v
py -3.12 scripts/validate_project.py
```

Khi chưa cài package, chạy từ root repository với:

```powershell
py -3.12 -m pip install -r requirements.txt
```

## Bằng chứng và tài liệu

- `Documents/Decision-Making/`: quyết định, problem contract và log kinh nghiệm lỗi (bao gồm ADR 23 về Goal Ladder).
- `Documents/Global/`: bản đồ ánh xạ mục tiêu G1/G2/G3, registry nhánh, giao thức nhánh.
- `Documents/References/`: Markdown tham chiếu, DOCX giao nộp và HTML báo cáo.
- `Documents/General/`: chỉ quy tắc và checkpoint/error log có cấu hình chạy; tri thức/quyết định nằm ở hai nhóm trên.
- `registries/`: trạng thái nguồn dữ liệu, model và thí nghiệm.
- `docs/adr/`: quyết định kiến trúc và ranh giới tái lập.
- `docs/converted/`: nguồn Markdown duy trì để tái dựng/đối chiếu 10 DOCX.
- `requirements-research.txt`: dependency model tùy chọn; chỉ cài sau data/model/resource gate.
- `AGENTS.md`, `Workflows/flowinfo.md`: quy tắc cho các phiên làm việc sau.

Bản báo cáo tiến độ mới nhất cho giảng viên: `Documents/References/HTML/progress-report-2026-08-26.html`.

## Giấy phép

Mã nguồn được phát hành theo MIT. Dataset, model và paper giữ giấy phép/điều khoản của nguồn tương ứng; MIT của repository không cấp lại quyền cho các tài sản đó.
