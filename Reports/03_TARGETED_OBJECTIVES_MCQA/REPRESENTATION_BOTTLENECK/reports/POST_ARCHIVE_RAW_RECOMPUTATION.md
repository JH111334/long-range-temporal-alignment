# Báo Cáo Tái Tính Toán Pháp Y Dữ Liệu Thô và Định Vị Sự Sụp Đổ Nhãn (Phase A Report)

**Dự án**: `long-range-temporal-alignment`  
**Nhánh thực hiện**: `v2-targeted-continual-alignment`  
**Thời điểm khóa sổ**: `2026-10-20`  
**Mã đợt chạy khảo sát**: `RUN_FULL_SWEEP_54P_20261019_161756`  
**Checkpoint kiểm định**: [`R52_B_NTP_REL_MCQA.pt`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/artifacts/experiments/V2_CAUSAL_TEMPORAL_ALIGNMENT/run_05_2/checkpoints/R52_B_NTP_REL_MCQA.pt) (SHA-256: `1af285df69ddaf94d21dcb858e8bd2d5e532874b484131d7fa3a31a4eac41988`)  
**Tập dữ liệu chuẩn tắc**: [`COUNTERFACTUAL_GOLD_AUDIT.jsonl`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/data/counterfactual/COUNTERFACTUAL_GOLD_AUDIT.jsonl) (SHA-256: `49eea7fe6fad69d4f3695e7cd8fbeac2c383e65b76d0802801666f9982cdeba5`)

---

## 1. Tái Tính Toán Thống Kê Dữ Liệu Thô 432 Bản Ghi

Từ tệp dữ liệu thô [`RAW_PREDICTIONS.jsonl`](file:///c:/Users/GF63/Documents/long-range-temporal-alignment/Reports/03_TARGETED_OBJECTIVES_MCQA/REPRESENTATION_BOTTLENECK/runs/RUN_FULL_SWEEP_54P_20261019_161756/RAW_PREDICTIONS.jsonl), toàn bộ 432 bản ghi được phân rã chuẩn xác:
- Cánh tay đối chứng nền (`BASE_BOTH_OFF`): 108 câu hỏi.
- Cánh tay can thiệp chính (`ARM_B_FULL_TRUE_YEAR`): 108 câu hỏi.
- Bốn cánh tay phụ can thiệp cục bộ (54 câu mỗi nhánh): WRONG_YEAR, SEAM_OFF, LORA_OFF, ZERO_TIME.

### Bảng Tiếp Liên 2x2 Ghép Cặp (Matched Paired Contingency Table, N=108)

```
                                      Arm B Checkpoint
                                 Đúng (Correct)     Sai (Incorrect)
Base Model    Đúng (Correct)         a = 2               b = 59
              Sai (Incorrect)        c = 44              d = 3
```

- Độ chính xác Base Model: 61 / 108 = 56.48%
- Độ chính xác Arm B Checkpoint: 46 / 108 = 42.59%
- Độ chênh lệch có dấu (Signed Delta): -13.89 pp (Arm B kém hơn Base).
- Kiểm định McNemar chính xác hai phía (Exact Binomial Test): p = 0.167444 > 0.05
- Khoảng tin cậy Bootstrap cụm 95% có dấu (Clustered Bootstrap CI on signed delta): Delta = -11.98 pp +- 8.37 pp, 95% CI = [-25.91 pp, 0.00 pp]

---

## 2. Phát Hiện Pháp Y Cốt Lõi: Sự Sụp Đổ Nhãn 'All-No' của Arm B

- Base Model: 103 nhãn 'Yes' (95.37%), 5 nhãn 'No' (4.63%). Entropy dự đoán: 0.2705 bits.
- Arm B Checkpoint: 108 nhãn 'No' (100.00%), 0 nhãn 'Yes' (0.00%). Entropy dự đoán: 0.0000 bits (sụp đổ nhãn hoàn toàn).
- Recall lớp 'Yes' (N=62 câu): Base = 95.16%, Arm B = 0.00% (0/62).
- Recall lớp 'No' (N=46 câu): Base = 4.35%, Arm B = 100.00% (46/46).
- Balanced Accuracy: Base = 49.75%, Arm B = 50.00%.

### Bóc Tách Bản Chất '18/18 Gold-Same Stability'
> Bản chất của 18/18 stability là hệ quả thuần túy của hiện tượng sụp đổ nhãn (All-No collapse). Do mô hình luôn luôn sinh 'No' trên 100% câu hỏi, hiển nhiên trên cả hai câu hỏi của mọi cặp Gold-Same, dự đoán đều là 'No' và do đó trùng khớp nhau (false stability). Mô hình hoàn toàn không thể hiện bất kỳ sự phân biệt ngữ nghĩa thời gian nào.

---

## 3. Kiểm Toán Thiên Kiến Hướng Dữ Liệu (Dataset Order Bias)

Kiểm toán toàn bộ 54 cặp tại `data/counterfactual/COUNTERFACTUAL_GOLD_AUDIT.jsonl` phát hiện:
1. 100% (36 / 36) cặp Gold-Flip có thứ tự chân lý cố định: t1: 'Yes' -> t2: 'No'. Hoàn toàn không có bất kỳ cặp nào có chiều ngược lại (t1: 'No' -> t2: 'Yes').
2. Khi tập dữ liệu có 100% chiều Yes -> No được đánh giá trên mô hình bị sụp đổ nhãn sang 'No', câu hỏi t1 luôn sai (0/36) và câu hỏi t2 luôn đúng (36/36), dẫn tới BothCorrect = 0/36 (0.0%).

---

## 4. Kiểm Toán Dấu Vết Cơ Chế & Biên Độ Delta G

Trong `MECHANISM_TRACE.jsonl`, độ đo Delta G: Delta G = G(x1, t1, z1) - G(x1, t1, z2) chỉ đo can thiệp trên x1.
- Tỷ lệ Delta G > 0: 13 / 36 = 36.11%, trung bình Delta G = -0.058017.
- Kiểm định nhị thức so với mức ngẫu nhiên 50%: p = 0.13250 > 0.05. Không thể kết luận Delta G khác biệt có ý nghĩa thống kê so với phân bố ngẫu nhiên.

---

## 5. Kết Luận Phase A và Điều Kiện Chuyển Tiếp Sang Phase B

1. Định vị điểm nghẽn: Thất bại bắt nguồn từ sự sụp đổ nhãn nhị phân sang 'No' của checkpoint Arm B.
2. Khuyết tật phép đo cũ: Phép đo cũ chỉ đo can thiệp trên x1, thiếu ô đối ngẫu (x2, z1) trên văn bản x2.
3. Chuyển tiếp Phase B: Kích hoạt ngay phép thử 2x2 text-year x conditioned-year trên toàn bộ 54 cặp để bóc tách triệt để đóng góp của văn bản câu hỏi x(t) và vector biểu diễn z(t).
