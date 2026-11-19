# BÁO CÁO PHÂN BỔ NGUYÊN NHÂN M0-FC40 (FACTORIAL ATTRIBUTION REPORT)

## 1. Kết Quả Thực Nghiệm Ghép Cặp (Paired Protocol Evaluation)

| Mô hình / Checkpoint | Giao thức | V2 Strict Acc | V2 Invalid Rate | MCQA Acc (n=16) | Yes/No Acc (n=8) | Legacy Strict Acc | Legacy Inv Rate |
|---|---|---|---|---|---|---|---|
| `BASE_MODEL_FROZEN` | `P01` | 20.83% | 45.83% | 12.50% | 37.50% | 20.83% | 45.83% |
| `BASE_MODEL_FROZEN` | `P11` | 8.33% | 75.00% | 12.50% | 0.00% | 8.33% | 75.00% |
| `BASE_MODEL_FROZEN` | `P12` | 41.67% | 0.00% | 31.25% | 62.50% | 41.67% | 0.00% |
| `CANARY_SEED1_AFTER02` | `P01` | 33.33% | 0.00% | 31.25% | 37.50% | 33.33% | 0.00% |
| `CANARY_SEED1_AFTER02` | `P11` | 20.83% | 50.00% | 31.25% | 0.00% | 20.83% | 50.00% |
| `CANARY_SEED1_AFTER02` | `P12` | 37.50% | 0.00% | 25.00% | 62.50% | 37.50% | 0.00% |
| `FULL_SEED1_AFTER02` | `P01` | 12.50% | 66.67% | 0.00% | 37.50% | 12.50% | 66.67% |
| `FULL_SEED1_AFTER02` | `P11` | 37.50% | 0.00% | 25.00% | 62.50% | 37.50% | 0.00% |
| `FULL_SEED1_AFTER02` | `P12` | 41.67% | 0.00% | 31.25% | 62.50% | 41.67% | 0.00% |
| `FULL_SEED2_AFTER02` | `P01` | 8.33% | 70.83% | 0.00% | 25.00% | 8.33% | 70.83% |
| `FULL_SEED2_AFTER02` | `P11` | 16.67% | 66.67% | 0.00% | 50.00% | 16.67% | 66.67% |
| `FULL_SEED2_AFTER02` | `P12` | 37.50% | 0.00% | 31.25% | 50.00% | 37.50% | 0.00% |
| `FULL_SEED3_AFTER02` | `P01` | 8.33% | 75.00% | 0.00% | 25.00% | 8.33% | 75.00% |
| `FULL_SEED3_AFTER02` | `P11` | 16.67% | 66.67% | 0.00% | 50.00% | 16.67% | 66.67% |
| `FULL_SEED3_AFTER02` | `P12` | 33.33% | 0.00% | 18.75% | 62.50% | 33.33% | 0.00% |

## 2. Phân Tích Cơ Chế Sinh Của Mô Hình

- **P01 (Plain Text Instruction)**: Mô hình không có ChatML bao bọc thường xuyên sinh `No. (a)...` hoặc `No...`, trong đó chữ cái đầu là `No`. Bộ parser của partner trích xuất được `(a)` trên MCQA nên tình cờ cho độ chính xác cao giả tạo (pseudo-accuracy).
- **P11 (ChatML Generic Instruction)**: Mô hình huấn luyện full run bị suy sụp phân phối (distributional collapse) sang token `Yes`, sinh `Yes` trên toàn bộ tập MCQA. Parser cũ chấp nhận `Yes` là hợp lệ khiến tỷ lệ Invalid bị báo cáo là 0.00%, trong khi độ chính xác MCQA thực tế là 0%.
- **P12 (ChatML Task-Aware Instruction)**: Khi được phân biệt rõ ràng chỉ dẫn giữa MCQA (`a, b, c, d`) và Yes/No (`Yes or No`), ta quan sát được phản ứng chính xác của từng checkpoint.

## 3. Kiểm Toán Parity và Checkpoint Integrity

- **`CANARY_SEED1_AFTER02`**: Module count = 72, Gate Alpha = -0.0036, Proj Norm = 1.9139, All Finite = `True`.
- **`FULL_SEED1_AFTER02`**: Module count = 72, Gate Alpha = 0.0006, Proj Norm = 2.5397, All Finite = `True`.
- **`FULL_SEED2_AFTER02`**: Module count = 72, Gate Alpha = -0.0004, Proj Norm = 2.4672, All Finite = `True`.
- **`FULL_SEED3_AFTER02`**: Module count = 72, Gate Alpha = -0.0008, Proj Norm = 2.5882, All Finite = `True`.