---
name: temporal-research-planning
description: Hoạch định nghiên cứu long-range temporal representation từ lựa chọn vấn đề đến dữ liệu, phương pháp, phần cứng, quản trị mô hình và ứng dụng. Dùng khi cần chọn hướng Việt Nam-first khả thi, đặc tả đầu vào/đầu ra, lập ADR hoặc đồng bộ tài liệu với scaffold; không dùng để gọi smoke test là kết quả nghiên cứu.
---

# Temporal Research Planning

## Kết quả cần đạt

Biến một hướng nghiên cứu temporal rộng thành một quyết định có thể kiểm chứng: một bài toán chính gắn liền với `goal_id` cụ thể trong Goal Ladder (G1 → G2 → G3 theo ADR 23), người dùng và giá trị sử dụng rõ, dữ liệu có quyền sử dụng, baseline tái lập được, giả thuyết đơn biến có tiêu chí bác bỏ, ngân sách phần cứng và đầu ra có provenance.

## Ràng buộc vận hành hiện hành

- **Lệnh dừng huấn luyện mới (`HOLD_NO_GO_TRAINING`)**: Giữ nguyên trạng thái không gửi thêm GPU job huấn luyện mới (`NO_GO_NEW_GPU_RUN`) cho tới khi hoàn tất nghiệm thu độc lập Goal 1 qua Replay GPU trên mẫu đại diện đã cam kết.
- **Thử nghiệm đơn biến (Single-Variable Hypothesis)**: Mọi kế hoạch thử nghiệm so sánh phải kiểm soát nghiêm ngặt duy nhất một biến can thiệp giữa Baseline và Candidate, giữ nguyên tuyệt đối split, evaluator, prompts và seeds.

## Quy trình

1. Kiểm kê `AGENTS.md`, `references/research-gates.md`, `Documents/Decision-Making/23_GOAL_LADDER_AND_ACCEPTANCE_20261105.md`, `Documents/Global/GOAL_BRANCH_STAGE_CROSSWALK.md`, ADR, registry, báo cáo, mã và test trước khi đề xuất.

2. Xác định rõ `goal_id` phụng sự (`G1`, `G2` hoặc `G3`). Lập ma trận tối đa ba bài toán ứng viên. Chấm theo giá trị cho Việt Nam, khả năng chuyển giao quốc tế, dữ liệu và giấy phép, khả năng đánh giá, năng lực nhóm, chi phí phần cứng và rủi ro triển khai.

   Bối cảnh Việt Nam phải nối trực tiếp với evidence, data source, method choice, evaluation slice và use case trong cùng lập luận. Không tạo một nhánh “ứng dụng Việt Nam” rời; chỉ giữ bonus khi data/technique gate cho phép.

3. Chọn đúng một bài toán chính và tối đa một stress test phụ. Ghi rõ điều bị loại và lý do; không mở rộng sang một hệ thống tổng quát khi chưa qua gate.

4. Viết problem contract: người dùng, quyết định được hỗ trợ, input, output, abstention/error, metric, lát cắt, phạm vi thời gian, ngôn ngữ, latency và ngân sách.

5. Lập data lifecycle: khảo sát nguồn, evidence gate, thu thập có phép, canonicalization bằng `HistoricalYear`/`HistoricalDate`/`TemporalInterval`, tách granularity khỏi temporal kind, kiểm soát chất lượng, group/time/source split, version/hash, truy cập, lưu giữ và xóa.

6. Tách phương pháp thành `published-equation`, `operational-assumption` và `proposed-method`. So scalar, Fourier và baseline mạnh trước encoder/objective phức tạp.

7. Vẽ pipeline logic từ nguồn đến artifact/ứng dụng; mỗi stage phải có input schema, output schema, validation, lỗi fail-closed, telemetry và chủ sở hữu.

8. Lập risk register cho khảo sát, annotation, leakage, contamination, tokenizer, alias, objective scaling, evaluator/parser, EWC/LoRA, OOM, drift, rollback và misuse.

9. Chia phần cứng theo gate: Tier 0 CPU, Tier 1 tokenizer/frozen probe, Tier 2 PEFT nhỏ, Tier 3 lab/cloud. Không suy ra khả năng chạy từ số tham số; phải đo peak RAM/VRAM và latency.

10. Đồng bộ quyết định vào ADR, reference Markdown, registry, scaffold/tests và báo cáo tiến độ. Chạy test và `scripts/validate_project.py` tại checkpoint.

11. Trước model lớn, ưu tiên data audit, retrieval, lỗi theo lát cắt và frozen probe trên RTX 2050 4 GB; chỉ mở PEFT khi có gain held-out và resource artifact.

Đọc [references/decision-contract.md](references/decision-contract.md) khi lập ma trận lựa chọn, problem contract hoặc gate chuyển giai đoạn.

## Ranh giới bằng chứng

- `scaffold`: sườn import được hoặc fail-closed; không chứng minh chất lượng.

- `smoke_test`: kiểm tra hợp đồng và luồng nhỏ; không phải benchmark.

- `paper_reported`: số liệu do paper báo cáo, chưa được nhóm tái lập.

- `measured`: có data/model revision, split, seed, config/data/code hash và resource artifact.

- Mọi nhận định về nhu cầu Việt Nam phải có nguồn hoặc ghi rõ là giả thuyết cần khảo sát người dùng.

- Không tải dataset/model thật nếu registry thiếu URL chính thức, ngày truy cập, license/terms, revision/checksum, reviewer và data role.

## Điều kiện dừng

Dừng hoặc đổi hướng khi không có dữ liệu hợp pháp, evaluator không phân biệt năng lực với lỗi parser, gain biến mất trên held-out split, hoặc chi phí vượt tier được duyệt. Kết quả âm tính có kiểm soát vẫn hợp lệ; không thêm module, framework hoặc dashboard chỉ để làm scaffold trông đầy đủ.

## Nhật ký vận hành

Đọc `Documents/General/RULE.md` trước khi ghi log. `General` chỉ có `RULE.md`, `long-range-temporal-alignment_Checkpoints_DD-MM-YY_Vn.md` và `long-range-temporal-alignment_Error_DD-MM-YY_Vn.md`; không đổi grammar hoặc sửa log cũ.

Mỗi kế hoạch `measured|benchmark` phải khóa `anchor_checkpoint`, `parent_checkpoint`, `goal_id`, hypothesis, evaluation role, gold status và evaluation contract hash. Chốt paired unit, lát cắt, case budget, selection rule, counterexample, expected delta và stop gate trước run. Nếu contract thay đổi, mở experiment family mới và ghi `confounded`; không so gain trực tiếp.

Sau run, tách quan sát khỏi nguyên nhân và kết luận `supported|refuted|inconclusive`. Run hợp lệ nhưng kém hơn là Checkpoint với `decision=REJECT|REPEAT`; Error chỉ dành cho run invalid hoặc quality gate định trước `FAIL`. Khảo sát, quyết định và diễn giải dài vẫn thuộc `Decision-Making`/`References`.

## Quy ước giao nhận

Tài liệu do dự án sở hữu dùng tiếng Việt có dấu, viết hoa đầu câu, format tối giản và nguồn đặt gần mệnh đề; không chép debug narrative. Quyết định vào `Documents/Decision-Making`, nguồn/hướng dẫn vào `Documents/References`, chỉ nhật ký đúng quy ước vào `Documents/General`. Mọi scaffold phải ghi rõ `scaffold`; nhánh đa phương thức chỉ được ghi trong file hoãn riêng cho đến khi có ADR mở scope.
