# Phán Quyết Khoa Học: V2 Run 04 — Sửa Sai Chân Lý Dữ Liệu và Căn Chỉnh Nhân Quả

**Ngày thực hiện:** 2026-10-15  
**Trạng thái thẩm định:** `CAUSALITY_INCONCLUSIVE`  
**Tập dữ liệu:** Khế ước chân lý toán học đơn nhất 100% (400 Train Clean, 120 Dev Clean, 50/50 Higher/Lower, 25/25/25/25 a/b/c/d).

## 1. Kết Quả Tổng Hợp 4 Bảng Điểm

| Mô Hình | S1: Clean MCQA Acc (120) | S1: Higher Acc (60) | S1: Lower Acc (60) | S2: Relation Acc (60) | S3: Oracle Diag Acc (60) | S4: Retention Slice Acc (60) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **W0_BASE** | 40.0% | 43.33% | 36.67% | 48.33% | 38.33% | 31.67% |
| **RUN02_C1** | 21.67% | 23.33% | 20.0% | 75.0% | 23.33% | 38.33% |
| **R4_A_CLEAN_MCQA_ONLY** | 59.17% | 61.67% | 56.67% | 48.33% | 70.0% | 33.33% |
| **R4_B_CLEAN_JOINT_TRUE (Primary)** | **65.83%** | 73.33% | 58.33% | **86.67%** | **65.0%** | **28.33%** |
| **R4_C_CLEAN_JOINT_SHUFFLED (Control)** | 64.17% | 73.33% | 55.0% | 70.0% | 60.0% | 33.33% |
| **R4_D_CLEAN_JOINT_REPLAY** | 65.83% | 73.33% | 58.33% | 86.67% | 65.0% | 28.33% |

## 2. Kiểm Định Thống Kê Nhân Quả
- Chênh lệch giữa True Conditioning (R4_B) và Shuffled Conditioning (R4_C): `+1.66 pp` (McNemar p = `0.625`).
- Chẩn đoán Oracle Evidence (S3): Khi được cung cấp trực tiếp 2 số liệu trong câu hỏi, mô hình đạt `65.0%`, chứng minh năng lực so sánh số học của mô hình hoàn toàn bình thường khi có căn cứ sự thật.
