---
loại: "quyết-định"
mã: "DECISION-STAGE03-ZERO-SHOT-CROSS-TASK-TRANSFER-20261116"
dự_án: "long-range-temporal-alignment"
trạng_thái: "đang-dùng"
chuẩn_hiện_hành: true
thay_thế: []
bị_thay_thế_bởi: []
tham_chiếu: ["23_GOAL_LADDER_AND_ACCEPTANCE_20261105", "24_EVAL_CONTRACT_G1_FREEGEN_V2"]
được_xác_thực_bởi: []
cập_nhật_lần_cuối: "2026-11-19"
---

# Quyết Định Phạm Vi Nghiên Cứu Stage 03: Chuyển Giao Tác Vụ Zero-Shot Từ Tiền Huấn Luyện Số Học NWP Sang Trắc Nghiệm MCQA

## 1. Tóm Lược Quyết Định Học Thuật (Supervisor Resolution)

Ngày 16/11/2026, giảng viên hướng dẫn khoa học đã chính thức phê chuẩn định hướng phương pháp luận cho Giai đoạn Stage 03 Goal 1 (`G1 = YEAR-only closed-answer temporal QA`):

1. **Khóa chặt phạm vi huấn luyện (Training Scope):** Duy trì 100% dữ liệu tiền huấn luyện tăng dần số học tự nhiên (`data/tempun_full_clean/train_104k_clean.jsonl.gz`, 104.130 mẫu Next-Word Prediction). Tuyệt đối **không thu thập, gán nhãn hay giám sát thêm dữ liệu trắc nghiệm MCQA** (No MCQA Supervision / SFT).
2. **Khóa chặt bài toán khoa học (Research Problem):** Định vị toàn bộ nghiên cứu là bài toán **Chuyển Giao Tri Thức Thời Gian Đa Tác Vụ Không Cần Giám Sát (Zero-Shot Cross-Task & Cross-Format Temporal Transfer)**:
   - Mô hình chỉ học biểu diễn thời gian qua dự đoán token số học tự nhiên trong câu trần thuật.
   - Năng lực định vị và suy luận thời gian được kiểm thử trên tập phát triển trắc nghiệm 4 lựa chọn (`eval_5292_clean.jsonl.gz`, TempUN MCQA + Rel) thông qua sinh tự do không ràng buộc (Unconstrained Free Generation) kết hợp 4 phép xoay vòng phương án triệt tiêu thiên kiến vị trí (4-fold cyclic debiasing).
3. **Nguyên tắc đối chuẩn bài báo gốc (Baseline Reporting Protocol):** Báo cáo kết quả đối chiếu với công bố của Ticktack (arXiv:2503.04150v3) trên tinh thần minh bạch phương pháp: bài báo gốc có sử dụng tập huấn luyện trắc nghiệm nội bộ kín (in-domain MCQA + QA data), trong khi đề tài của chúng ta chỉ sử dụng dữ liệu mở và chứng minh năng lực biểu diễn tự thân khi chuyển giao zero-shot.

---

## 2. Ý Nghĩa Khoa Học và Tính Mới Của Đề Tài (Scientific Novelty)

Việc không sử dụng dữ liệu trắc nghiệm để huấn luyện biến một giới hạn dữ liệu thành một **đóng góp khoa học mang tính độc lập và vượt trội về mặt phương pháp luận**:

- **Tránh lối tắt ghi nhớ định dạng (Bypassing Format Shortcut Learning):** Khi một mô hình được huấn luyện bằng câu hỏi trắc nghiệm rồi đánh giá bằng trắc nghiệm, điểm số cao có thể chỉ là ảo ảnh từ việc học thuộc mẫu câu `(a) ... (b) ... (c) ... (d)`. 
- **Minh chứng tính tổng quát của Biểu diễn Hình học Thời gian:** Mục tiêu cốt lõi của đề tài là chứng minh rằng ma trận điều biến điều hòa MaT-LoRA $( \theta, z )$ và tầng tiêm Seam Linear thực sự học được **tô-pô hình học tuần hoàn của trục thời gian**. Nếu biểu diễn này đúng đắn, nó phải chuyển giao được năng lực định hướng chú ý sang tác vụ trắc nghiệm mà không cần dạy trước định dạng trắc nghiệm.

---

## 3. Tác Động Lên Cổng Chuẩn Bị Huấn Luyện 5 Epochs (Unblocking Gate)

Quyết định này giải tỏa điểm nghẽn chiến lược quan trọng nhất ghi nhận tại `Reports/6-11Report/G1_5EPOCH_READINESS_GATE.json`:
- **Trạng thái trước:** `NO_GO_5EP_PENDING_SUPERVISOR_SCOPE` do chưa rõ có cần SFT trắc nghiệm hay không.
- **Trạng thái hiện tại:** **`READY_FOR_FACTORIAL_DESIGN`**. Toàn bộ thiết kế thừa số cho đợt huấn luyện 5 epochs tiếp theo sẽ tập trung thuần túy vào việc cô lập cơ chế biểu diễn:
  - So sánh đối chứng ghép cặp: **Arm 4 (Static LoRA + Seam Linear)** đối đầu **Arm 5 (MaT-LoRA Dynamic Core + Seam Linear)** trên cùng tập `train_104k_clean.jsonl.gz`, cùng hàm mất mát NWP, cùng số lượng tham số tương đương và seed cố định.

---

## 4. Ràng Buộc Kiến Trúc Bất Biến & Tập Cải Tiến Khóa Chặt (Cập nhật 19/11/2026)

1. **Khóa ranh giới mô-đun (MLP Linear Only — NO Attention):** Toàn bộ nhánh `v2-targeted-continual-alignment` cho Goal `G1` (YEAR-only, Zero-Shot NWP $\to$ MCQA/Rel) **chỉ can thiệp ở các tầng MLP Linear (`gate_proj`, `up_proj`, `down_proj`, 108 modules trên Qwen2.5-3B)**. Tuyệt đối **không can thiệp vào khối Attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`)** và **không dùng DoRA/DLoRA** (`attach_adapters_to_model` ép lỗi fail-closed nếu nhận tham số Attention).
2. **Sửa đổi Phạm vi Continual Alignment 19/11/2026 (`NWP + Non-Target General Replay`):**
   - Trong bối cảnh lịch sử V2 thuần 100% NWP, mô hình thiếu giữ nhịp đọc-hiểu tổng quát; trong lần chạy V3 thuần NWP (chỉ 4 epochs với `asinh`/`powers10`/`MULTI_YEAR_SPAN`), điểm MCQA tụt về `19.53%–20.31%` kèm `14.19%–14.83%` lỗi định dạng thật ngay cả sau khi sửa parser.
   - Bổ sung thiết lập **NWP (`104,130`) + Non-Target General Replay (`15%`, trích từ SQuAD v1.1 train `CC BY-SA 4.0` lọc sạch mốc thời gian, không dùng MCQA/Rel hay đáp án DEV)** với **0% giám sát bài toán đích** (*Zero Target-Task Supervision*). Lưu ý: thiết lập 100% NWP thuần trước đây được ghi nhận riêng là mốc lịch sử, không còn áp dụng nguyên văn cho 2 lượt chạy sửa lỗi `R1` (`G1_SINGLE_YEAR_REPLAY_FIXED`) và `R2` (`G1_SINGLE_YEAR_REPLAY_COSINE`); việc mở rộng phạm vi chính thức cho luận văn cần ghi nhận sự đồng thuận của GVHD.
   - Quay về neo biểu diễn hiệu quả V2 (`dyadic` 32D + `linear` Can-Chi 8D + `z = (year-1947)/75`) và tiêm đơn mốc tại thân câu hỏi (`YEAR_SPAN` trước `Options:`, sửa triệt để 597 câu `WB` bằng quy tắc vai trò ngữ nghĩa `range_start`/`range_end`/`subject_year`).
