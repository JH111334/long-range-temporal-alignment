# Phán Quyết Khoa Học Sau Kiểm Toán Pháp Y và Tái Đánh Giá Chuẩn Tắc M0-FC40

## 1. Kết Quả Đo Lường Chuẩn Tắc (180 Câu TempUN Small)

Sau khi khắc phục hoàn toàn lỗi nạp trọng số (`CONFIRMED_LOADER_BYPASS`), mô hình $M_0\text{-FC40}$ đã được nạp đúng $100\%$ các ma trận LoRA $A, B$, bộ sinh lõi điều hòa (Core Generator) và cổng chiếu Seam $\alpha$ trên toàn bộ 72 module MLP. Kết quả đo đạc chính thức dưới giao thức $P_{12}$ (`RAW_GREEDY_V1`):

| Mô hình | Strict ACC Mean | Seed 1 | Seed 2 | Seed 3 | MCQA Acc Mean | YN Acc Mean | Invalid Rate | $\Delta$ vs Base |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Qwen2.5-0.5B** | 32.22% | — | — | — | 26.28% | 70.83% | 0.00% | — |
| **M0-FC40 (Sau Sửa Nạp Trọng Số)** | **29.81% \pm 2.8%** | 27.22% | 32.78% | 29.44% | 24.15% | 66.66% | 0.00% | **-2.41 pp** |
| **TickTack-inspired 0.5B** | **35.19% \pm 1.4%** | 33.89% | 35.0% | 36.67% | ~29.70% | ~70.83% | 0.00% | **+2.97 pp** |

## 2. Kết Luận Pháp Y và Thống Kê

1. **Bản chất của con số 58/180 trước đây:** Đã được cô lập hoàn toàn do lỗi không khớp khóa từ điển (`adapters_state` vs `adapted_modules`). Việc trùng khớp tuyệt đối 58/180 trên cả 3 seed trước đó là do mô hình thực thi chính là mô hình nền.
2. **Hiệu năng thực tế của M0-FC40 khi đã nạp trọng số:** Mô hình đạt **29.81% \pm 2.8%** Strict Accuracy. Phân tích từng ca tại [08_casewise_comparison.csv](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/audits/M0_FC40_CAUSAL_INFERENCE_RECOVERY_20261015/08_casewise_comparison.csv) cho thấy mô hình sinh ra phân phối logits khác biệt rõ rệt (mean max logit delta ~ 16.2), và lật đáp án trên nhiều ca kiểm thử.
3. **Ý nghĩa thống kê so sánh:** Khoảng tin cậy Bootstrap 95% và kiểm định McNemar cho thấy sự chênh lệch hiệu năng giữa các mô hình ở quy mô 0.5B trên 180 ca là `STATISTICALLY_INCONCLUSIVE` ($p > 0.05$ trên các seed ghép cặp). Không thể tuyên bố một mô hình vượt trội tuyệt đối mang ý nghĩa thống kê khi khoảng tin cậy vẫn bao trùm giá trị 0.
