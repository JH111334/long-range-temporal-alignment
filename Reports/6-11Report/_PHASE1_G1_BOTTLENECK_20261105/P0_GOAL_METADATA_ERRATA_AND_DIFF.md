# Báo Cáo Đính Chính Siêu Dữ Liệu và Bản Vá Ngữ Cảnh P0 (P0 Metadata Errata & Context Diff)

**Ngày lập:** 05/11/2026  
**Dự án:** `long-range-temporal-alignment`  
**Chủ thể thẩm quyền khoa học:** Người dùng (Chủ nhiệm đề tài) + ChatGPT (Cố vấn phương pháp luận)  
**Đơn vị thực thi:** Antigravity (Executor / Auditor / Documentation Maintainer)  
**Nhánh công tác:** `v2-targeted-continual-alignment` (chẩn đoán G1) / `GLOBAL_MAINTENANCE` (đồng bộ crosswalk)  
**Mục tiêu trọng tâm:** `GOAL 1 (G1) — YEAR-only closed-answer temporal QA`

---

## 1. Đính Chính P0a — Đường Dẫn Dữ Liệu Thực Tế (Dataset Pointer Erratum)

### 1.1. Hiện trạng trước sửa đổi
Trong tài liệu quản trị `Documents/Global/GOAL_BRANCH_STAGE_CROSSWALK.json` và `.md` thiết lập ngày 05/11/2026, liên kết dữ liệu của Goal 1 được ghi nhận là:
- `data/training/train_104k_clean.jsonl.gz`
- `data/evaluation/eval_5292_clean.jsonl.gz`

### 1.2. Kết quả kiểm toán vật lý độc lập (Physical Disk Audit)
Kiểm tra hệ thống tệp cục bộ xác nhận:
- Thư mục `data/training/` và `data/evaluation/` **hoàn toàn không tồn tại** trên ổ đĩa cục bộ (`exists=False`).
- Thư mục dữ liệu chuẩn tắc duy nhất được triển khai trong thực tế huấn luyện và đánh giá trên Modal là `data/tempun_full_clean/`:
  1. `data/tempun_full_clean/train_104k_clean.jsonl.gz`:
     - Kích thước: `1.774.192` bytes
     - SHA-256: `111d2d0908533a17ac10c98533802053631b708a23846524daa224bd1465697d`
     - Số lượng mẫu: $104.130$ dòng
  2. `data/tempun_full_clean/eval_5292_clean.jsonl.gz`:
     - Kích thước: `278.971` bytes
     - SHA-256: `a624d8d258f9676f00e504d16c35dc401cc1cce96613ca56af970daa32a12fc3`
     - Số lượng mẫu: $5.292$ dòng (phân loại: Development Benchmark)

Căn cứ mã nguồn thực thi tại `scripts/run_stage03_targeted_arms4_5_5epochs_modal.py` (dòng 348-349):
```python
train_path = _REPO_ROOT / "data" / "tempun_full_clean" / "train_104k_clean.jsonl.gz"
eval_path = _REPO_ROOT / "data" / "tempun_full_clean" / "eval_5292_clean.jsonl.gz"
```

### 1.3. Hành động khắc phục đã thực hiện
- Đã cập nhật chính xác đường dẫn tệp trong `Documents/Global/GOAL_BRANCH_STAGE_CROSSWALK.json` và `Documents/Global/GOAL_BRANCH_STAGE_CROSSWALK.md` trỏ về `data/tempun_full_clean/train_104k_clean.jsonl.gz` và `data/tempun_full_clean/eval_5292_clean.jsonl.gz`.
- Kiểm tra toàn bộ độc giả (readers) của bảng ánh xạ crosswalk; không gây đứt gãy phụ thuộc mã nguồn hay công cụ kiểm tra tự động.

---

## 2. Đính Chính P0b — Khóa Lược Đồ Bản Kê Bất Biến (Manifest Schema Key Erratum)

### 2.1. Quan sát lỗi hình thức
Tệp bản kê niêm phong `Reports/6-11Report/GOAL_ALIGNMENT_20261105/UNMODIFIED_SCIENTIFIC_ARTIFACTS.json` chứa một khóa chuỗi rỗng `""` ở dòng đầu tiên thay vì `"$schema"` do lỗi thoát chuỗi trong quá trình kết xuất JSON tự động.

### 2.2. Nguyên tắc bảo toàn chứng từ niêm phong (Historical Immutability)
Tuân thủ nghiêm ngặt nguyên tắc pháp y dữ liệu: **Tuyệt đối không sửa đè trực tiếp (in-place mutation) lên tệp đã niêm phong**.
- Tệp gốc `UNMODIFIED_SCIENTIFIC_ARTIFACTS.json` được giữ nguyên vẹn 100% với mã băm xác nhận:
  - SHA-256: `f3c5a5bdbbf92604bc8696fb945cd66d76d889f38481976887a83341c7b72f26`
  - Số lượng tài sản niêm phong: `51` tệp.

### 2.3. Bản kê tách rời kế thừa V2 (Detached V2 Manifest)
Ban hành tệp bản kê đính chính tách rời:
- Đường dẫn: `Reports/6-11Report/GOAL_ALIGNMENT_20261105/UNMODIFIED_SCIENTIFIC_ARTIFACTS_V2.json` (và bản sao đồng bộ tại `Reports/6-11Report/_PHASE1_G1_BOTTLENECK_20261105/UNMODIFIED_SCIENTIFIC_ARTIFACTS_V2.json`).
- Khóa `"$schema"` đã được phục hồi chuẩn mực theo RFC JSON Schema Draft 2020-12.
- Bảo toàn trọn vẹn danh mục 51 tài sản khoa học với đường dẫn, kích thước byte và mã băm SHA-256 khớp 100% với tệp gốc.

---

## 3. Rà Soát P0c — Ranh Giới Nghiệm Thu ADR 23 (ADR23 Scope Boundary Clarification)

Để đảm bảo tính trung thực khoa học tuyệt đối và tránh nhầm lẫn giữa **kế hoạch dự kiến** và **bằng chứng thực nghiệm đã đạt được**, dự án xác lập các phân định ranh giới sau:

1. **Quy mô $N=2.000$, 25 quốc gia chưa từng thấy, dữ liệu 2023–2025**:
   - Đây là **đề xuất thiết kế kiểm định (Proposed Evaluation Design)** phục vụ các vòng nghiên cứu tiếp theo, **chưa phải tập dữ liệu đã sẵn có hoặc đã qua kiểm chuẩn**.
   - Tập `eval_5292` hiện hữu là **Development Benchmark** (đã qua nhiều lần dò tìm siêu tham số), không được phép tự tuyên bố là tập kiểm tra độc lập nguyên vẹn (Untouched Test).

2. **Hạn mức kinh phí \$0.20 USD**:
   - Đây là **hạn mức ngân sách đề xuất (Proposed Budget Target)**, chưa phải chi phí thực tế đã được kiểm chứng cho toàn bộ quy trình Replay GPU.
   - Phân tích chi phí thời gian thực (Modal A10G startup, model weight download ~6.5 GB, CUDA init) cho thấy chi phí có thể dao động từ \$0.05 (nếu có cache volume) đến \$0.22 (nếu tải lại từ HuggingFace).

3. **Phân định bản chất hai nguồn dữ liệu**:
   - Dữ liệu chuỗi số thống kê toàn cầu TempUN và bộ câu hỏi lịch sử Việt Nam là **hai cấu trúc năng lực độc lập (distinct constructs)**. Tuyệt đối không gộp chung vào một chỉ số bình quân đơn lẻ không phân tầng.

Nội dung chi tiết của các kiến nghị hiệu chỉnh được biên soạn tại [PROPOSED_GATE_REFINEMENTS.md](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Documents/Decision-Making/PROPOSED_GATE_REFINEMENTS.md).
