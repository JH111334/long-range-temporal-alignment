---
document_class: "OFFICIAL_DERIVED"
authority: "EVIDENCE_GROUNDED_INDEX"
retention: "PERMANENT"
status: "VALIDATED"
goal_scope: "GOAL1_YEAR_CLOSED_ANSWER"
evaluation_role: "SYSTEM_REPORT_DIRECTORY"
---

# Thư mục Báo cáo Tiến độ 16+/11 — Stage 03 G1

Thư mục này chứa đầy đủ chứng từ số liệu, phân tích toán học và báo cáo học thuật cho giai đoạn kiểm định lại hệ thống đánh giá căn chỉnh thời gian Stage 03 G1 (Mục tiêu 1: Trắc nghiệm đóng mốc Năm lịch sử), thực hiện trên phần cứng GPU NVIDIA A10G (24GB VRAM) vào ngày 2026-11-08.

> **Lưu ý hệ quy chiếu:** Các thư mục `01_CORE_CONFIRMATIONS/` và `02_SCIENTIFIC_INSIGHTS/` bên dưới là kết quả **Eval Replay V2 (128 mẫu cố định, ngày 08/11/2026)** trên checkpoint 5-epoch cũ — không phải kết quả huấn luyện 5-epoch mới. Phần đối chiếu cùng hệ quy chiếu giữa **5-Epoch Training cũ** và **5-Epoch Training V3** nằm tại `ERROR_DIAGNOSIS/`.

## Cấu trúc Thư mục

### 1. `01_CORE_CONFIRMATIONS/` (Các Xác nhận Kiểm chứng Then chốt)
Chứa các chứng từ tham số kỹ thuật và báo cáo chứng minh tính đúng đắn của hệ thống:
- [`01_CHECKPOINT_INTEGRITY_RECEIPT.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/16+/11Report/01_CORE_CONFIRMATIONS/01_CHECKPOINT_INTEGRITY_RECEIPT.json): Bằng chứng nạp 100% 216 LoRA + 4 Seam + 5 Core tensors, không NaN/Inf, $\alpha = -0.006561$.
- [`02_LINEAR_GATE_PARITY_RECEIPT.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/16+/11Report/01_CORE_CONFIRMATIONS/02_LINEAR_GATE_PARITY_RECEIPT.json): Chứng minh toán học và định lượng về tính đúng đắn của tầng cổng tuyến tính (`gate_mode="linear"`), khắc phục lỗi thổi phồng $76\times$ và đảo chiều của sigmoid.
- [`03_FREEGEN_FEASIBILITY_RECEIPT.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/16+/11Report/01_CORE_CONFIRMATIONS/03_FREEGEN_FEASIBILITY_RECEIPT.json): Bằng chứng thực nghiệm về tỷ lệ đầu ra không hợp lệ chỉ $1.72\%$ khi sinh tự do không áp đặt.
- [`CONFIRMATIONS_SUMMARY.md`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/16+/11Report/01_CORE_CONFIRMATIONS/CONFIRMATIONS_SUMMARY.md): Báo cáo tổng hợp ba xác nhận kiểm chứng kỹ thuật.

### 2. `02_SCIENTIFIC_INSIGHTS/` (Các Phát hiện Khoa học Then chốt)
Chứa các chứng từ dữ liệu và báo cáo phân tích hiện tượng khoa học:
- [`01_BASE_SLOT_BIAS_RECEIPT.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/16+/11Report/02_SCIENTIFIC_INSIGHTS/01_BASE_SLOT_BIAS_RECEIPT.json): Dữ liệu thực nghiệm về hiện tượng thiên lệch vị trí tiền huấn luyện (chọn ô `a`) trên Base Qwen2.5-3B.
- [`02_ARM5_ENTITY_TRACKING_RECEIPT.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/16+/11Report/02_SCIENTIFIC_INSIGHTS/02_ARM5_ENTITY_TRACKING_RECEIPT.json): Bằng chứng bám vết thực thể lịch sử (`0.08`) qua 4/4 vòng xoay của Arm 5 (MaT-LoRA + Seam).
- [`03_ROTATION_DEBIASING_ROLE_RECEIPT.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/16+/11Report/02_SCIENTIFIC_INSIGHTS/03_ROTATION_DEBIASING_ROLE_RECEIPT.json): Chứng minh phương pháp luận về vai trò bắt buộc của kỹ thuật khử thiên lệch bằng xoay vòng 4 vị trí.
- [`04_JEV_DECISION_RECEIPT.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/16+/11Report/02_SCIENTIFIC_INSIGHTS/04_JEV_DECISION_RECEIPT.json): Dữ liệu telemetry từ hệ thống Jev System One (`jev-1.14.0`, primitives `noul` và `choice`).
- [`INSIGHTS_SUMMARY.md`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/16+/11Report/02_SCIENTIFIC_INSIGHTS/INSIGHTS_SUMMARY.md): Báo cáo tổng hợp ba phát hiện khoa học và đóng góp của Jev System One.

### 3. `ERROR_DIAGNOSIS/` (Snapshot Code Cũ & Biên nhận 3 Lỗi Lịch sử)
Lưu giữ snapshot mã nguồn/cấu hình cũ (tĩnh) cùng các JSON biên nhận đối chứng cho 3 lỗi trước khi nâng cấp V3:
- [`legacy_bugs_code_snapshot.py`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/16+/11Report/ERROR_DIAGNOSIS/legacy_bugs_code_snapshot.py): Snapshot nguyên văn các đoạn mã cũ gây lỗi `token_idx=14`/thiếu năm thứ hai, lỗi `torch.sigmoid(self.alpha)`, lỗi trục tuyến tính `z`/`radius` và thiếu LR scheduler.
- [`BEFORE_AFTER_CONFIG_SNAPSHOT.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/16+/11Report/ERROR_DIAGNOSIS/BEFORE_AFTER_CONFIG_SNAPSHOT.json): Bản chụp so sánh cấu hình trước–sau giữa legacy V1/V2 và V3 khóa chặt.
- [`ERROR_DIAGNOSIS_RECEIPT.json`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/16+/11Report/ERROR_DIAGNOSIS/ERROR_DIAGNOSIS_RECEIPT.json): Biên nhận JSON ghi nhận kết quả đo kiểm thực tế 3 lỗi trên dữ liệu `train_104k_clean` và `eval_5292_clean`.
