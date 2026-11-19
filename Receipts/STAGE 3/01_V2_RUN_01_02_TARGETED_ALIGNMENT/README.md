# Thư Mục Nghiệm Thu Thực Nghiệm V2: Căn Chỉnh Thời Gian Nhân Quả (Run 01)

## 1. Mục Tiêu và Phạm Vi
Thực nghiệm kiểm chứng giả thuyết nhân quả về sự suy thoái và năng lực thích ứng thời gian của mô hình M0-FC40 trên benchmark TempUN Small (180 mẫu).
Toàn bộ thực nghiệm tuân thủ nghiêm ngặt các ranh giới:
- Mô hình nền `Qwen/Qwen2.5-0.5B-Instruct` đóng băng 100%.
- Giao diện vector đặc trưng `FIXED_FONE_CANCHI_V1` 40D bất biến.
- Đánh giá sinh tự do toàn từ vựng theo khuôn mẫu ChatML P12 (`RAW_GREEDY_V1`).

## 2. Kết Quả Đo Lường Ghép Cặp (4-Condition Factorial Evaluation)

| Điều kiện | Mô tả điều kiện | Strict Accuracy | Invalid Rate | MCQA Acc (n=156) | Yes/No Acc (n=24) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **C0** | M0 Baseline (Single Year $e(t)$) | **28.89%** | 0.0% | 22.44% | 70.83% |
| **C1** | Targeted Temporal Relation | **28.89%** | 0.0% | 22.44% | 70.83% |
| **C2** | Negative Control (Shuffled Year) | **28.89%** | 0.0% | 22.44% | 70.83% |
| **C3** | Format Control (Zero Vector $0_{40D}$) | **27.78%** | 0.0% | 21.15% | 70.83% |

## 3. Kiểm Định Thống Kê và Bằng Chứng Nhân Quả
- **Độ lợi so với M0 Baseline ($C_1 - C_0$)**: 0.0% ($p = 1.0$).
- **Độ lợi so với Đối Chứng Âm ($C_1 - C_2$)**: 0.0% ($p = 0.61708$).
- **Độ lợi so với Kiểm Soát Định Dạng ($C_1 - C_3$)**: 1.11% ($p = 0.77283$).
- **Độ nhạy Logit**: Mean max logit delta đạt 0.9763, xác nhận vector 40D truyền kích hoạt vào mô hình nhưng bị chế ngự bởi prior của câu hỏi.

## 4. Danh Mục Hồ Sơ Nghiệm Thu
- `CLAIM_EVIDENCE_AUDIT.json`: Kiểm toán định lượng các giả thuyết.
- `ROOT_CAUSE_HYPOTHESES.json`: Báo cáo chẩn đoán Stage 3 chi tiết.
- `RESEARCH_PRIOR_ART_MATRIX.json`: Khảo sát ma trận 8 công trình nghiên cứu.
- `SOURCE_AND_DATA_CONTRACT.json`: Hợp đồng nguồn dữ liệu và mã checksum.
- `INTERVENTION_SELECTION.md`: Luận chứng lựa chọn can thiệp quan hệ thời gian.
- `PRE_REGISTERED_EXPERIMENT.json`: Hồ sơ đăng ký trước thực nghiệm.
- `NEGATIVE_CONTROL_VALIDATION.json`: Nghiệm thu đối chứng âm.
- `TIME_SENSITIVITY_REPORT.json`: Báo cáo độ nhạy logit theo thời gian.
- `RESULTS_AND_PAIRED_STATS.json`: Kết quả chi tiết và kiểm định McNemar.
- `ERROR_CASES.jsonl`: Phân loại toàn bộ các ca lỗi theo taxonomy.
- `RUN_RECEIPT.yaml`: Terminal receipt chuẩn hóa.
- `SHA256_MANIFEST.json`: Bảng băm toàn vẹn tệp.
