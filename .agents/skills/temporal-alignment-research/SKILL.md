---
name: temporal-alignment-research
description: Thiết kế, tái lập và kiểm chứng nghiên cứu temporal alignment hoặc long-range time representation trong repository này. Dùng khi khảo sát paper/dataset, thêm encoder hay objective, xây split/evaluator, chạy model probe hoặc LoRA, phân tích lỗi, chuẩn bị benchmark và báo cáo tiến độ. Bắt buộc phân biệt công thức đã công bố, giả định vận hành, phương pháp đề xuất, smoke test và kết quả đo.
---

# Temporal Alignment Research

## Mục tiêu

Biến một ý tưởng temporal alignment thành thí nghiệm có nguồn, hợp đồng, baseline và tiêu chí bác bỏ rõ ràng. Ưu tiên Tier 0 có thể tái lập trước khi dùng model hoặc GPU.

## Kiểm tra bắt buộc theo Goal Ladder (ADR 23)

Mọi đề xuất nghiên cứu, kế hoạch chạy thử nghiệm hoặc báo cáo kết quả bắt buộc phải khai báo rõ `goal_id` theo [[Decision-Making/23_GOAL_LADDER_AND_ACCEPTANCE_20261105|ADR 23]]:
- `G1` (YEAR-only closed-answer temporal QA): Nhận diện và suy luận trắc nghiệm đóng mốc năm (`A/B/C/D`, `Yes/No`).
- `G2` (YEAR + MONTH closed-answer temporal QA): Mở rộng kết hợp năm và tháng, nhất quán xuyên độ hạt.
- `G3` (General temporal QA): Hỏi đáp đóng + mở có dẫn chứng lịch sử Việt Nam và từ chối hợp lệ.

**Phân định cơ chế và mục tiêu nghiệm thu**:
- Mọi điều hòa biểu diễn nội tại (FoNE, Ticktack, MaT-LoRA, khoảng cách vector $D_z, D_F, D_W$, biên độ dịch chuyển $\Delta \text{logits}$, tỷ lệ va chạm tần số FoNE) chỉ được phân loại là **MECHANISM / DIAGNOSTIC ONLY**, tuyệt đối không coi là `GOAL_GAIN`.
- Đối với `G1`, `GOAL_GAIN` bắt buộc phải được chứng minh bằng độ chính xác trả lời đóng (`Canonical Accuracy`, `Debiased Mean`, `Strict Accuracy 4/4 và 2/2`, `Counterfactuals Both-Correct`, `Retention Rate`).
- Các kết quả 5 epochs hiện tại của Stage 03 giữ nguyên phân loại `METRIC_ARITHMETIC_ONLY` và trạng thái `QUARANTINED_PENDING_PROVENANCE` cho đến khi có GPU replay xác nhận độc lập.

## Workflow bắt buộc

1. Đọc `references/research-gates.md`, `Documents/Decision-Making/23_GOAL_LADDER_AND_ACCEPTANCE_20261105.md` và các ADR liên quan trong `docs/adr/`.

2. Ghi paper, dataset và model vào registry; gắn nguồn chính thức, ngày truy cập, license/terms và trạng thái.

3. Không ingest dataset thật nếu evidence gate chưa đạt. Không thay TempLS bằng dữ liệu mô phỏng rồi gọi là reproduction.

4. Chuẩn hóa qua `HistoricalYear`, `HistoricalDate` và `TemporalInterval`; tách `granularity=year|month|day` khỏi `temporal_kind=point|interval|duration|relation`.

5. Chia group/time/source trước preprocessing học từ dữ liệu; kiểm tra near-duplicate giữa split.

6. Chạy baseline theo thứ tự: Gregorian scalar → Fourier → published Ticktack → operational assumption → proposed hybrid.

7. Kiểm tra trước model: alias 60 năm, tính đơn điệu, batch-size sensitivity, invalid-output rate và evaluator parity.

8. Chỉ mở Tier 1/2 sau resource probe. Mọi model phải cố định revision; LoRA phải nêu rank, alpha, dropout, target modules, precision, seed và peak RAM/VRAM.

9. Với EWC, chỉ rõ tham số trainable, checkpoint tham chiếu, cách ước lượng Fisher và coefficient. Dừng nếu penalty đặt lên tham số frozen hoặc không có reference tương ứng.

10. Xuất artifact có config hash, data manifest hash, code revision, mức bằng chứng, metric theo lát cắt và warnings.

11. Với máy RTX 2050 4 GB, ưu tiên sửa dữ liệu, split, evaluator và retrieval; model/PEFT chỉ là candidate sau khi probe peak RAM/VRAM và chứng minh gain held-out.

## Học từ kết quả

- Trước run `measured|benchmark`, đọc `Documents/General/RULE.md` và khóa `anchor_checkpoint`, `parent_checkpoint`, `hypothesis_id`, `goal_id`, evaluation role, gold status cùng evaluation contract hash.

- Chỉ so sánh khi data/split/gold/evaluator/metric/parser/slice registry tương thích. Xuất paired delta theo mẫu và lát cắt, case budget, selection rule, counterexample cùng artifact/hash; khác contract phải ghi `confounded`.

- Không có gold thì chỉ dùng data-quality signal, stability, behavioral test hoặc human audit. Ghi pseudo-label, synthetic label và LLM judge là `proxy`; không suy ra effectiveness.

- Tách `observation` khỏi `cause`. Kết luận hypothesis bằng `supported|refuted|inconclusive`, rồi nêu dự đoán có thể bác bỏ, minimal next experiment và stop/rollback gate.

- Run hợp lệ nhưng kém hơn vẫn là Checkpoint với `decision=REJECT|REPEAT`. Chỉ tạo Error khi run invalid hoặc quality gate định trước `FAIL`; diễn giải dài thuộc `Decision-Making`/`References`.

## Ranh giới tuyên bố

- `published-equation`: đúng công thức paper đã in, kể cả hành vi lỗi có thể kiểm thử.

- `operational-assumption`: chi tiết cần bổ sung để chạy nhưng paper không nêu.

- `proposed-method`: giả thuyết của nhóm; không gọi SOTA khi chưa có benchmark đối chứng.

- `smoke_test`: xác nhận pipeline/hợp đồng, không đo chất lượng mô hình.

- `measured`: chỉ dùng khi có dataset/model revision, split, seed, resource và artifact thực.

## Gate hoàn thành

Một thay đổi chỉ hoàn thành khi test encoder/objective/split liên quan đạt, `scripts/validate_project.py` đạt, registry và ADR được cập nhật, và báo cáo không trộn bằng chứng với quyết định nội bộ.

## Ngôn ngữ và tài liệu

Viết nội dung do dự án sở hữu bằng tiếng Việt có dấu, viết hoa đầu câu, gọn và chuyên nghiệp; không chép debug narrative. Chỉ giữ tiếng Anh cho thuật ngữ, API, tên lớp/hàm và trích dẫn phù hợp. Tài liệu mới phải vào đúng `Documents/Decision-Making` hoặc `Documents/References`; `Documents/General` chỉ nhận `RULE.md` và hai họ checkpoint/error log định nghĩa trong file đó. Cấp gốc `Documents` chỉ giữ file tổng quan. Mọi UI/HTML phải hiển thị provenance, evidence class và trạng thái gate, không dùng trang trí để che thiếu hụt artifact.
